from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import torch
from torch import Tensor
from PIL import Image

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


@dataclass(frozen=True)
class MinMaxNormalization:
    position_min: list[float]
    position_max: list[float]
    action_min: list[float]
    action_max: list[float]
    kind: str = 'minmax_v1'

    def __post_init__(self):
        if self.kind != 'minmax_v1':
            raise ValueError('Expected minmax_v1 normalization')
        for prefix in ('position', 'action'):
            lo, hi = np.asarray(getattr(self, prefix + '_min')), np.asarray(getattr(self, prefix + '_max'))
            if lo.shape != (2,) or hi.shape != (2,) or not np.isfinite([lo, hi]).all() or (hi <= lo).any():
                raise ValueError(f'Invalid {prefix} min/max')

    def to_dict(self):
        return asdict(self)

    def _bounds(self, value: Tensor, prefix: str):
        return (value.new_tensor(getattr(self, prefix + '_min')), value.new_tensor(getattr(self, prefix + '_max')))

    def normalize_position(self, value: Tensor) -> Tensor:
        lo, hi = self._bounds(value, 'position')
        return 2 * (value - lo) / (hi - lo) - 1

    def normalize_action(self, value: Tensor) -> Tensor:
        lo, hi = self._bounds(value, 'action')
        return 2 * (value - lo) / (hi - lo) - 1

    def denormalize_action(self, value: Tensor) -> Tensor:
        lo, hi = self._bounds(value, 'action')
        return (value + 1) * (hi - lo) / 2 + lo


@dataclass(frozen=True)
class ACTNormalization:
    position_mean: list[float]
    position_std: list[float]
    action_mean: list[float]
    action_std: list[float]
    std_floor: float = 1e-6

    def __post_init__(self):
        for name in ('position_mean', 'position_std', 'action_mean', 'action_std'):
            v = np.asarray(getattr(self, name))
            if v.shape != (2,) or not np.isfinite(v).all() or (name.endswith('std') and (v <= 0).any()):
                raise ValueError('Invalid ACT normalization')

    def normalize_position(self, value: Tensor) -> Tensor:
        return (value - value.new_tensor(self.position_mean)) / value.new_tensor(self.position_std).clamp_min(
            self.std_floor
        )

    def denormalize_action(self, value: Tensor) -> Tensor:
        return value * value.new_tensor(self.action_std).clamp_min(self.std_floor) + value.new_tensor(self.action_mean)


def image_tensor(image: np.ndarray | Tensor) -> Tensor:
    # 图像转成 [3, 96, 96] 的浮点数，再按 ImageNet 统计量归一化。
    if isinstance(image, Tensor):
        value = image.float()
        if value.shape != (3, 96, 96):
            raise ValueError('Expected CHW image 3x96x96')
        if image.dtype == torch.uint8:
            value = value / 255
    else:
        value = (
            torch.from_numpy(
                np.asarray(
                    Image.fromarray(np.asarray(image, dtype=np.uint8)).resize((96, 96), Image.Resampling.BILINEAR)
                ).copy()
            )
            .permute(2, 0, 1)
            .float()
            / 255
        )
    return (value - IMAGENET_MEAN) / IMAGENET_STD


def observation_tensors(history: list[dict], normalization, device='cpu') -> tuple[Tensor, Tensor]:
    if len(history) != 2:
        raise ValueError('Exactly two adjacent observations required')

    # 增加 batch 维度，和模型的 images、positions 输入保持一致。
    images = torch.stack([image_tensor(o['pixels']) for o in history])[None]  # [1, 2, 3, 96, 96]
    positions = torch.from_numpy(np.stack([np.asarray(o['agent_pos'], dtype=np.float32) for o in history]))[
        None
    ]  # [1, 2, 2]

    return images.to(device), normalization.normalize_position(positions).to(device)
