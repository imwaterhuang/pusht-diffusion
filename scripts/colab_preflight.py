"""L4 正式训练前的真实流程、恢复、小批拟合和吞吐检查。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import time

# CUDA 精确恢复检查需要确定性矩阵乘法；必须在创建 CUDA context 前设置。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader, default_collate
from torchvision.models import ResNet18_Weights

from pusht_diffusion.checkpoints import load_checkpoint
from pusht_diffusion.config import default_config
from pusht_diffusion.data import PushTDataset
from pusht_diffusion.learner.models import DiffusionPolicy
from pusht_diffusion.learner.training import diffusion_loss, train, train_step
from pusht_diffusion.reproducibility import DeterministicBatchSampler, seed_everything
from pusht_diffusion.utils import atomic_json, canonical_hash, code_fingerprint
from verify_training_resume import assert_equal


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()


def make_state(config):
    """只组装现有 train_step 的输入，梯度、EMA 等算法仍由用户代码执行。"""
    seed_everything(config["training"]["seed"])
    model = DiffusionPolicy(cond_dim=644).cuda().train()
    ema = deepcopy(model).requires_grad_(False).eval()
    c = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["learning_rate"],
                                  betas=tuple(c["betas"]), weight_decay=c["weight_decay"])
    lr = get_scheduler(c["lr_schedule"], optimizer=optimizer,
                       num_warmup_steps=c["warmup_steps"], num_training_steps=c["steps"])
    noise = DDPMScheduler(num_train_timesteps=config["diffusion"]["train_timesteps"],
                          beta_schedule=config["diffusion"]["beta_schedule"],
                          prediction_type="epsilon", clip_sample=False)
    return model, {"optimizer": optimizer, "lr_scheduler": lr, "noise_scheduler": noise,
                   "ema_model": ema, "step": 0}


def verify_recovery(config, dataset, source, out):
    def run(name, resume, stop):
        train(config, {"dataset_root": dataset, "output_dir": str(out / name)},
              resume=str(resume), device="cuda", max_updates=stop)
        clear_memory()
        return out / name / "last.pt"

    # 第一次恢复直接读取 Drive 镜像，顺便验证持久化文件可以使用。
    continuous = run("continuous", source, 102)
    split = run("split", source, 101)
    split = run("split", split, 102)
    a, b = load_checkpoint(continuous), load_checkpoint(split)
    fields = ["step", "raw", "ema", "optimizer", "lr", "sampler", "rng"]
    for field in fields:
        assert_equal(a[field], b[field], field)
    return {"passed": True, "start_step": 100, "end_step": 102,
            "exact_equal_fields": fields, "loaded_drive_mirror": str(source)}


def profile_training(config, dataset):
    model, state = make_state(config)
    sampler = DeterministicBatchSampler(len(dataset), config["training"]["batch_size"], 0)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=8,
                        generator=torch.Generator().manual_seed(0))
    batches = iter(loader)
    records = []
    # 两步预热后记录 20 步：数据等待、主机到 GPU 传输、完整参数更新分别计时。
    for index in range(22):
        started = time.perf_counter()
        batch = next(batches)
        loaded = time.perf_counter()
        batch = {key: value.cuda() if torch.is_tensor(value) else value for key, value in batch.items()}
        torch.cuda.synchronize()
        transferred = time.perf_counter()
        metrics = train_step(model, batch, state)
        torch.cuda.synchronize()
        finished = time.perf_counter()
        sampler.mark_consumed()
        if index >= 2:
            records.append({"data_wait_s": loaded - started, "transfer_s": transferred - loaded,
                            "update_s": finished - transferred, "loss": metrics["loss"]})
    result = {"batch_size": config["training"]["batch_size"], "num_workers": 8, "warmup_steps": 2,
              "measured_steps": len(records), "mean_seconds": {
                  key: sum(row[key] for row in records) / len(records)
                  for key in ["data_wait_s", "transfer_s", "update_s"]}, "measurements": records}
    result["estimated_40000_update_hours_excluding_saves_eval"] = (
        sum(result["mean_seconds"].values()) * 40000 / 3600)
    return result


def fit_fixed_batch(config, dataset):
    model, state = make_state(config)
    weights = ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=False)
    pretrained_verified = torch.equal(model.visual_encoder.backbone[0].weight.detach().cpu(), weights["conv1.weight"])
    if not pretrained_verified:
        raise AssertionError("Pretrained conv1 initialization mismatch")
    del weights
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(731))[:16].tolist()
    batch = default_collate([dataset[index] for index in indices])
    batch = {key: value.cuda() if torch.is_tensor(value) else value for key, value in batch.items()}

    # 独立 generator 产生固定检查噪声，不消耗训练用的全局随机序列。
    generator = torch.Generator().manual_seed(123456)
    probes = [(torch.randint(0, 100, (16,), generator=generator).cuda(),
               torch.randn(batch["actions"].shape, generator=generator).cuda()) for _ in range(4)]

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        for k, noise in probes:
            noisy = state["noise_scheduler"].add_noise(batch["actions"], noise, k)
            prediction = model(noisy, batch["images"], batch["positions"], k)
            losses.append(float(diffusion_loss(prediction, noise, batch["valid_mask"])))
        model.train()
        return sum(losses) / len(losses)

    checks = [{"step": 0, "fixed_probe_loss": evaluate()}]
    for step in range(1, 401):
        metrics = train_step(model, batch, state)
        if step % 100 == 0:
            checks.append({"step": step, "fixed_probe_loss": evaluate(), "training_loss": metrics["loss"]})
            print(f"小批拟合 {step}/400 | 固定检查 loss {checks[-1]['fixed_probe_loss']:.5f}", flush=True)
    ratio = checks[-1]["fixed_probe_loss"] / checks[0]["fixed_probe_loss"]
    # 0.3 是本次诊断预先选定的下降比例，不是策略泛化或成功率门槛。
    passed = all(math.isfinite(row["fixed_probe_loss"]) for row in checks) and ratio <= 0.3
    return {"passed": passed, "sample_indices": indices, "batch_size": 16, "updates": 400,
            "fixed_probe_count": len(probes), "checks": checks, "final_initial_ratio": ratio,
            "diagnostic_ratio_threshold": 0.3, "pretrained_conv1_verified": pretrained_verified,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "train_timesteps": state["noise_scheduler"].config.num_train_timesteps,
            "lr_schedule": "Unmodified formal 500-step warmup and 40000-step cosine schedule"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mirror", required=True)
    args = parser.parse_args()
    out, mirror = Path(args.output).resolve(), Path(args.mirror).resolve()
    out.mkdir(parents=True, exist_ok=True)
    mirror.mkdir(parents=True, exist_ok=True)
    config = default_config()
    result = {"kind": "real_l4_preflight", "num_workers": 8, "passed": False, "code_fingerprint": code_fingerprint(),
              "config_fingerprint": canonical_hash(config),
              "limitations": ["Not closed-loop policy evidence", "Diagnostic weights are discarded; formal run starts fresh"]}
    started = time.perf_counter()

    def record():
        result["elapsed_seconds"] = time.perf_counter() - started
        atomic_json(out / "preflight.json", result)
        atomic_json(mirror / "preflight.json", result)

    try:
        if not torch.cuda.is_available() or "L4" not in torch.cuda.get_device_name(0):
            raise RuntimeError("This preflight requires an attached NVIDIA L4 GPU")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        result["device"] = torch.cuda.get_device_name(0)
        result["versions"] = {name: version(name) for name in ["torch", "torchvision", "diffusers", "numpy"]}
        result["cuda_version"] = torch.version.cuda
        result["hardware"] = {"gpu": result["device"], "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory}
        result["deterministic_algorithms"] = True
        dataset = PushTDataset(args.dataset)
        reference = json.loads((Path(__file__).resolve().parents[1] / "reports/data-audit.json").read_text())
        if dataset.audit["dataset_fingerprint"] != reference["dataset_fingerprint"]:
            raise ValueError("Dataset fingerprint differs from the verified reference dataset")
        result["data_fingerprint"] = dataset.audit["dataset_fingerprint"]
        record()

        print("开始 L4 100 步真实流程检查（含 Drive 镜像）", flush=True)
        train(config, {"dataset_root": args.dataset, "output_dir": str(out / "smoke100"),
                       "checkpoint_mirror": str(mirror / "smoke100")}, device="cuda", max_updates=100)
        saved = load_checkpoint(mirror / "smoke100/last.pt", expected_config=config,
                                expected_data=result["data_fingerprint"],
                                expected_normalization=dataset.normalization.to_dict())
        if saved["step"] != 100:
            raise AssertionError("Smoke checkpoint must contain exactly 100 completed updates")
        result["smoke"] = {"passed": True, "updates": 100, "drive_checksum_valid": True}
        del saved
        clear_memory()
        record()

        print("开始 L4 精确恢复检查", flush=True)
        result["recovery"] = verify_recovery(config, args.dataset, mirror / "smoke100/last.pt", out / "resume")
        clear_memory()
        record()
        print("测量 batch64 数据等待、传输和参数更新耗时", flush=True)
        result["profile"] = profile_training(config, dataset)
        clear_memory()
        record()
        print("开始固定 16 样本拟合检查", flush=True)
        result["small_batch_fit"] = fit_fixed_batch(config, dataset)
        clear_memory()
        result["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
        record()
        if not result["small_batch_fit"]["passed"]:
            raise RuntimeError("Fixed-batch diagnostic failed its predeclared loss-reduction threshold")
        result["passed"] = True
        record()
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        record()
        raise


if __name__ == "__main__":
    main()
