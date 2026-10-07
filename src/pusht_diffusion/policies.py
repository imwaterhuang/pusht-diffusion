"""Policy adapters shared by evaluation and the interactive page."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from .config import validate_config
from .learner.models import (
    DiffusionPolicy as LearnerDiffusionPolicy,
    UnetDiffusionModel,
    VisualEncoder,
    diffusion_step_embedding,
)
from .normalization import ACTNormalization, MinMaxNormalization, observation_tensors
from .runtime import inference_context, isolated_random_state, scene_generator
from .utils import file_hash


class Predictor(Protocol):
    name: str
    provenance: dict

    def predict_action_chunk(self, history: list[dict], *, scene_seed: int, replan_index: int) -> np.ndarray: ...


class _CheckpointUNet(LearnerDiffusionPolicy):
    """Inference interface for the user's existing U-Net, with unchanged state keys."""

    def __init__(self, config: dict) -> None:
        # 检查点会覆盖所有权重；这里复用原有模块，不再下载 ImageNet 权重。
        torch.nn.Module.__init__(self)
        self.unet = UnetDiffusionModel(
            cond_dim=config["model"]["condition_dim"] + config["model"]["unet"]["timestep_dim"]
        )
        self.visual_encoder = VisualEncoder(projection_dim=256, initialize_pretrained=False)

    def encode_observation(self, images: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # 拼接顺序与 learner 的 forward 完全相同，每帧的视觉特征紧接位置。
        return torch.cat([self.visual_encoder(images), positions], dim=-1).flatten(1)

    def denoise(self, noisy_actions: torch.Tensor, k: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        condition = torch.cat([condition, diffusion_step_embedding(k, dim=128)], dim=-1)
        return self.unet(noisy_actions, condition)


class DiffusionPolicy:
    def __init__(
        self,
        model: torch.nn.Module,
        config: dict,
        normalization: MinMaxNormalization,
        *,
        device: str = "cpu",
        provenance: dict,
    ) -> None:
        self.model = model
        self.config = validate_config(config)
        self.normalization = normalization
        self.device = torch.device(device)
        self.name = config["model"]["name"]
        self.provenance = provenance

    def predict_action_chunk(self, history: list[dict], *, scene_seed: int, replan_index: int) -> np.ndarray:
        images, positions = observation_tensors(history, self.normalization, self.device)
        generator = scene_generator(scene_seed, replan_index, self.config["sampling"]["noise_seed"])
        return self.sample(images, positions, generator=generator)

    def sample(
        self,
        images: torch.Tensor,
        positions: torch.Tensor,
        *,
        generator: torch.Generator,
        diagnostic_ddpm: bool = False,
    ) -> np.ndarray:
        """Encode once, denoise a 16-action chunk, then return physical targets."""
        # 创建采样调度器，设置从高噪声到低噪声的 k 序列。
        scheduler = make_scheduler(self.config, diagnostic_ddpm=diagnostic_ddpm)
        step_count = 100 if diagnostic_ddpm else self.config["sampling"]["steps"]
        scheduler.set_timesteps(step_count, device=self.device)

        step_options = {"generator": generator}
        if not diagnostic_ddpm:
            step_options["eta"] = 0.0

        with inference_context(self.model):
            # 同一次规划只编码一次观测，从随机动作噪声开始去噪。
            condition = self.model.encode_observation(images, positions)  # [1, 516]
            noisy_actions = torch.randn((1, 16, 2), generator=generator, dtype=torch.float32).to(
                self.device
            )  # [1, 16, 2]

            for k in scheduler.timesteps:
                batch_k = k.expand(1).to(self.device, dtype=torch.long)  # [1]
                predicted_noise = self.model.denoise(noisy_actions, batch_k, condition)  # [1, 16, 2]
                noisy_actions = scheduler.step(predicted_noise, k, noisy_actions, **step_options).prev_sample

            # 最后转回环境中的物理坐标，供控制器执行动作前缀。
            actions = self.normalization.denormalize_action(noisy_actions)  # [1, 16, 2]
            actions = actions.squeeze(0).cpu().numpy()  # [16, 2]

        if not np.isfinite(actions).all():
            raise ValueError("Diffusion policy produced nonfinite actions")
        return np.clip(actions, 0, 512).astype(np.float32)


def make_scheduler(config: dict, *, diagnostic_ddpm: bool = False):
    from diffusers import DDIMScheduler, DDPMScheduler

    diffusion = config["diffusion"]
    options = {
        "num_train_timesteps": diffusion["train_timesteps"],
        "beta_schedule": diffusion["beta_schedule"],
        "prediction_type": diffusion["prediction_type"],
        "clip_sample": False,
        "thresholding": False,
        "timestep_spacing": diffusion["timestep_spacing"],
        "steps_offset": diffusion["steps_offset"],
        "rescale_betas_zero_snr": False,
    }

    if diagnostic_ddpm:
        return DDPMScheduler(**options)
    return DDIMScheduler(**options, set_alpha_to_one=True)


def load_diffusion_policy(path: str | Path, *, config: dict, audit: dict, device: str = "cpu") -> DiffusionPolicy:
    from .checkpoints import load_checkpoint

    config = validate_config(config)
    if config["model"]["name"] != "unet":
        raise NotImplementedError("The current learner implements only U-Net; DiT inference is not implemented")

    checkpoint = load_checkpoint(
        path,
        expected_config=config,
        expected_data=audit["dataset_fingerprint"],
        expected_normalization=audit["normalization"],
    )
    with isolated_random_state():
        model = _CheckpointUNet(config)
        model.load_state_dict(checkpoint["ema"], strict=True)
        model.to(device).eval()

    provenance = {
        key: checkpoint[key]
        for key in (
            "step",
            "model_type",
            "config_fingerprint",
            "data_fingerprint",
            "code_fingerprint",
        )
    }
    provenance.update(
        {
            "checkpoint": str(Path(path).resolve()),
            "checkpoint_sha256": file_hash(path),
            "weights": "ema",
            "sampling": config["sampling"],
            "kind": "trained_policy",
        }
    )
    return DiffusionPolicy(
        model,
        config,
        MinMaxNormalization(**checkpoint["normalization"]),
        device=device,
        provenance=provenance,
    )


class ACTPolicy:
    name = "act"

    def __init__(self, model, normalization, provenance, device="cpu") -> None:
        self.model = model
        self.normalization = normalization
        self.provenance = provenance
        self.device = torch.device(device)

    def predict_action_chunk(self, history: list[dict], *, scene_seed: int, replan_index: int) -> np.ndarray:
        images, positions = observation_tensors(history, self.normalization, self.device)
        with inference_context(self.model):
            prediction = self.model(images[:, -1], positions[:, -1])
            actions = self.normalization.denormalize_action(prediction)
            actions = actions.squeeze(0).cpu().numpy()

        if not np.isfinite(actions).all():
            raise ValueError("ACT policy produced nonfinite actions")
        return np.clip(actions, 0, 512).astype(np.float32)


def load_act_policy(path: str | Path, *, audit: dict, device: str = "cpu") -> ACTPolicy:
    from .baselines.act import ActionPolicy

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    model_config = checkpoint["config"]["model"]

    if metadata["architecture"] != "act_v1" or model_config["name"] != "act":
        raise ValueError("Not an ACT checkpoint")
    if metadata["dataset_fingerprint"] != audit["dataset_fingerprint"]:
        raise ValueError("ACT data fingerprint mismatch")

    normalization = metadata["normalization"]
    expected = audit["legacy_act_statistics"]
    for source, prefix in (("agent_position", "position"), ("action", "action")):
        for statistic in ("mean", "std"):
            actual_values = normalization[f"{prefix}_{statistic}"]
            expected_values = expected[source][statistic]
            if not np.allclose(actual_values, expected_values, rtol=1e-6, atol=1e-6):
                raise ValueError("ACT normalization mismatch")

    if normalization["std_floor"] != expected["std_floor"]:
        raise ValueError("ACT normalization floor mismatch")

    model_options = {key: value for key, value in model_config.items() if key != "name"}
    model_options["pretrained"] = False
    with isolated_random_state():
        model = ActionPolicy(**model_options)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(device).eval()

    provenance = {
        "kind": "trained_policy",
        "architecture": "act_v1",
        "checkpoint": str(Path(path).resolve()),
        "checkpoint_sha256": file_hash(path),
        "step": checkpoint["step"],
        "data_fingerprint": metadata["dataset_fingerprint"],
        "normalization": normalization,
        "config": checkpoint["config"],
        "latent": "z=0",
        "history_frames_consumed": 1,
    }
    return ACTPolicy(model, ACTNormalization(**normalization), provenance, device)
