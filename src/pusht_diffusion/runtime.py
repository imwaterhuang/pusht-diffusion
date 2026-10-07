"""Mode and random-state boundaries for evaluation, with no training operations."""

from __future__ import annotations
from contextlib import contextmanager
import hashlib
import torch
from .reproducibility import capture_random_states, restore_random_states


@contextmanager
def isolated_random_state():
    state = capture_random_states()
    mps = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    try:
        yield
    finally:
        restore_random_states(state)
        if mps is not None:
            torch.mps.set_rng_state(mps)


@contextmanager
def inference_context(model: torch.nn.Module):
    modes = [(m, m.training) for m in model.modules()]
    try:
        model.eval()
        with isolated_random_state(), torch.inference_mode():
            yield
    finally:
        # Calling train(root_mode) would destroy intentionally frozen BN/dropout modes.
        for module, mode in modes:
            module.training = mode


def noise_seed(scene_seed: int, replan_index: int, base_seed: int = 0) -> int:
    if min(scene_seed, replan_index, base_seed) < 0:
        raise ValueError('Seeds and replan index must be nonnegative')
    raw = f'pusht-noise-v1:{base_seed}:{scene_seed}:{replan_index}'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') % (2**63 - 1)


def scene_generator(scene_seed: int, replan_index: int, base_seed: int = 0) -> torch.Generator:
    # CPU noise is identical for the paired models, regardless of their device.
    return torch.Generator(device='cpu').manual_seed(noise_seed(scene_seed, replan_index, base_seed))
