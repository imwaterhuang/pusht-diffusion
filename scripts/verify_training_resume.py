"""真实恢复检查：比较连续两步与一步后重启再一步，不替代小批拟合/策略评测。"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch

from pusht_diffusion.checkpoints import load_checkpoint
from pusht_diffusion.learner.training import train
from pusht_diffusion.utils import atomic_json, code_fingerprint


def assert_equal(left, right, path="state"):
    """逐张量精确比较；不能只看 loss 或 completed step。"""
    if isinstance(left, torch.Tensor):
        if not torch.equal(left, right):
            raise AssertionError(f"Tensor mismatch: {path}")
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            assert_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            assert_equal(a, b, f"{path}[{index}]")
    else:
        assert left == right, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="已有且附带校验文件的真实检查点")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    started = time.perf_counter()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = load_checkpoint(args.checkpoint)
    config, first_step = source["config"], source["step"]
    del source
    gc.collect()
    assert first_step + 2 <= config["training"]["steps"], "Need two remaining steps"

    def run(name, resume, stop):
        train(config, {"dataset_root": args.dataset, "output_dir": str(out / name)},
              resume=str(resume), device=args.device, max_updates=stop)
        gc.collect()
        return out / name / "last.pt"

    # 两条路径使用完全相同的模型、真实数据和原 train_step。
    continuous = run("continuous", args.checkpoint, first_step + 2)
    split = run("split", args.checkpoint, first_step + 1)
    split = run("split", split, first_step + 2)
    a, b = load_checkpoint(continuous), load_checkpoint(split)
    fields = ["step", "raw", "ema", "optimizer", "lr", "sampler", "rng"]
    for field in fields:
        assert_equal(a[field], b[field], field)
    result = {"kind": "real_training_resume_equivalence", "passed": True,
              "device": args.device, "start_step": first_step, "end_step": first_step + 2,
              "real_updates_executed": 4, "exact_equal_fields": fields,
              "code_fingerprint": code_fingerprint(),
              "elapsed_seconds": time.perf_counter() - started,
              "limitations": ["Same-device recovery only; no cross-device equivalence claim",
                              "Not a small-batch overfit or closed-loop policy evaluation"]}
    atomic_json(out / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
