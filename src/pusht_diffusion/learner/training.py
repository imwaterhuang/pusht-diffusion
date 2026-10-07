"""User-owned diffusion training with checkpoint and progress plumbing."""
from __future__ import annotations
from collections import deque
from typing import Any
from torch import Tensor, nn
import torch
from diffusers import DDPMScheduler

from torch.utils.data import DataLoader
from ..data import PushTDataset
from ..reproducibility import DeterministicBatchSampler, seed_everything

from .models import DiffusionPolicy

from diffusers.optimization import get_scheduler

from copy import deepcopy

from pathlib import Path

from ..checkpoints import load_checkpoint, make_checkpoint, save_checkpoint
from ..reproducibility import capture_training_rng, restore_training_rng
from ..utils import code_fingerprint

def diffusion_loss(predicted_noise: Tensor, target_noise: Tensor, valid_mask: Tensor) -> Tensor:
    """Masked mean squared error over valid action coordinates. No padding leakage."""
    error = (predicted_noise - target_noise) ** 2
    Loss = (error * valid_mask.unsqueeze(-1)).sum() / (valid_mask.sum() * error.shape[-1])
    return Loss

def train_step(model: nn.Module, batch: dict[str, Tensor], state: dict[str, Any]) -> dict[str, float]:
    """User draws k/noise, adds noise, predicts, computes loss/backward, clips and updates.

    Advance learning rate, exponential moving average and consumed sampler
    cursor only on successful parameter updates. No diagnostic trainer supplied.
    """
    device = next(model.parameters()).device

    images = batch['images'].to(device) # [B, 2, 3, 96, 96]
    positions = batch['positions'].to(device) # [B, 2, 2]
    actions = batch['actions'].to(device) # [B, 16, 2]
    valid_mask = batch['valid_mask'].to(device) # [B, 16]

    optimizer = state['optimizer']
    noise_scheduler = state['noise_scheduler']

    optimizer.zero_grad()

    k = torch.randint(0, 100, (images.shape[0],), device=device) # [B]
    noise = torch.randn_like(actions) # [B, 16, 2]
    noisy_actions = noise_scheduler.add_noise(actions, noise, k) # [B, 16, 2]

    predicted_noise = model(noisy_actions, images, positions, k) # [B, 16, 2]
    loss = diffusion_loss(predicted_noise, noise, valid_mask)

    if not torch.isfinite(loss):
        raise ValueError(f"Non-finite loss {loss.item()} at step {state.get('step', 'unknown')}")
    
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)

    optimizer.step()
    state['lr_scheduler'].step()
    state['step'] += 1
    update_ema(model, state['ema_model'], state['step'])

    return {"loss": loss.item()}

@torch.no_grad()
def update_ema(model, averaged_model, completed_updates):
    decay = min(0.999, (1 + completed_updates) / (10 + completed_updates))

    for ema_param, param in zip(
        averaged_model.parameters(), model.parameters()
    ):
        ema_param.mul_(decay).add_(param, alpha=1 - decay)

    for ema_buffer, buffer in zip(
        averaged_model.buffers(), model.buffers()
    ):
        ema_buffer.copy_(buffer)

def train(
    config: dict, paths: dict[str, str], *, resume: str | None = None,
    device: str | None = None, max_updates: int | None = None,
) -> None:
    """Run user-written updates; max_updates is an absolute completed-step limit."""

    if max_updates is not None and max_updates < 1:
        raise ValueError("max_updates must be positive")
    stop_step = min(config["training"]["steps"], max_updates) if max_updates is not None else config["training"]["steps"]

    seed = config["training"]["seed"]
    seed_everything(seed)

    # 数据集
    dataset = PushTDataset(paths["dataset_root"], image_cache=paths.get("image_cache"))

    # 恢复前先验证实验、数据和归一化，避免把不同实验的状态混在一起。
    restored = None
    if resume is not None:
        restored = load_checkpoint(
            resume,
            expected_config=config,
            expected_data=dataset.audit["dataset_fingerprint"],
            expected_normalization=dataset.normalization.to_dict(),
        )
        if restored["step"] > stop_step:
            raise ValueError("Checkpoint step exceeds requested total stop step")

    # sampler
    sampler = DeterministicBatchSampler(
        dataset_size=len(dataset), 
        batch_size=config["training"]["batch_size"],
        seed=seed,
    )

    # dataloader
    loader_generater = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=8,  # 8 个工作进程并行读取、解码训练数据。
        generator=loader_generater,
    )
    
    if device is not None:
        device = torch.device(device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = DiffusionPolicy(cond_dim=516 + 128).to(device)
    model.train()

    ema_model = deepcopy(model)
    ema_model.requires_grad_(False)
    ema_model.eval()

    #创建优化器和加噪调度器
    training_config = config["training"]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        betas=tuple(training_config["betas"]),
    )

    lr_scheduler = get_scheduler(
        name=training_config["lr_schedule"],
        optimizer=optimizer,
        num_warmup_steps=training_config["warmup_steps"],
        num_training_steps=training_config["steps"],
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["diffusion"]["train_timesteps"],
        beta_schedule=config["diffusion"]["beta_schedule"],
        prediction_type="epsilon",
        clip_sample=False,
    )

    state = {
        "optimizer": optimizer,
        "noise_scheduler": noise_scheduler,
        "lr_scheduler": lr_scheduler,
        "ema_model": ema_model,
        "step": 0,
    }
    
    checkpoint_path = Path(paths["output_dir"]) / "last.pt"
    source_fingerprint = code_fingerprint()
    
    if restored is not None:
        model.load_state_dict(restored["raw"], strict=True)
        ema_model.load_state_dict(restored["ema"], strict=True)
        optimizer.load_state_dict(restored["optimizer"])
        lr_scheduler.load_state_dict(restored["lr"])
        sampler.load_state_dict(restored["sampler"])
        state["step"] = restored["step"]
        consumed = sampler.epoch * sampler.batches_per_epoch + sampler.batch_in_epoch
        if consumed != state["step"]:
            raise ValueError("Checkpoint sampler position disagrees with completed updates")
        restore_training_rng(restored["rng"], {"loader": loader_generater})
        print(f"✓ 已恢复：step {state['step']} → 目标 {stop_step}", flush=True)

    if state["step"] == stop_step:
        return

    # 当前数据读取不含随机增强。新建 iterator 会消耗 loader generator，
    # 因此在其创建后再次还原 RNG，保持后续扩散噪声和保存状态与连续训练一致。
    data_iter = iter(loader)
    if restored is not None:
        restore_training_rng(restored["rng"], {"loader": loader_generater})
    recent_losses = deque(maxlen=100)
    progress_width = 0

    # max_updates 是累计停止步数；恢复时只补做剩余更新。
    for step in range(state["step"], stop_step):
        batch = next(data_iter)
        metrics = train_step(model, batch, state)
        sampler.mark_consumed()

        completed_steps = state["step"]
        recent_losses.append(metrics["loss"])

        should_save = (completed_steps % training_config["save_every"] == 0) or (completed_steps == stop_step)

        if should_save:
            payload = make_checkpoint(
                ema=ema_model.state_dict(),
                data_fingerprint=dataset.audit["dataset_fingerprint"],
                optimizer=optimizer.state_dict(),
                lr=lr_scheduler.state_dict(),
                step=completed_steps,
                config=config,
                code_fingerprint=source_fingerprint,
                normalization=dataset.normalization.to_dict(),
                raw=model.state_dict(),
                sampler=sampler.state_dict(),
                rng=capture_training_rng({"loader":loader_generater}),
            )

            result = save_checkpoint(checkpoint_path, payload, mirror=paths.get("checkpoint_mirror"))

            # 每个计划评测节点保留独立权重，方便之后闭环评测与选择。
            if completed_steps % training_config["evaluate_every"] == 0:
                save_checkpoint(
                    checkpoint_path.with_name(f"step_{completed_steps:06d}.pt"),
                    payload,
                    mirror=paths.get("checkpoint_mirror"),
                )

            # 先清除当前进度行，再单独保留简短的保存记录。
            print(
                f"\r{' ' * progress_width}\r"
                f"✓ 已保存：step {completed_steps} → {Path(result['path']).name}",
                flush=True,
            )
            progress_width = 0

        mean_loss = sum(recent_losses) / len(recent_losses)
        progress = (
            f"训练 {completed_steps}/{stop_step}"
            f" | 当前 loss {metrics['loss']:.4f}"
            f" | 近{len(recent_losses)}步均值 {mean_loss:.4f}"
        )
        progress_width = max(progress_width, len(progress))
        print(f"\r{progress:<{progress_width}}", end="", flush=True)

    print()  # 结束时换行，避免后续终端输出接在进度后面。
