"""The zero-output stub checks only the engineering adapter. It is not a model."""

import numpy as np
import torch
from pusht_diffusion.config import default_config
from pusht_diffusion.policies import DiffusionPolicy, make_scheduler
from pusht_diffusion.normalization import MinMaxNormalization
from pusht_diffusion.runtime import scene_generator


class ZeroOutputInterfaceStub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.child = torch.nn.Identity()
        self.encodes = 0
        self.noise_steps = []

    def encode_observation(self, images, positions):
        self.encodes += 1
        assert images.shape == (1, 2, 3, 96, 96)
        return images.new_zeros((1, 516))

    def denoise(self, x, k, condition):
        self.noise_steps.append(int(k.item()))
        return torch.zeros_like(x)


def test_actual_ddim_adapter_uses_one_encoding_isolated_noise_and_final_clip():
    config = default_config()
    stub = ZeroOutputInterfaceStub()
    stub.train()
    stub.child.eval()
    norm = MinMaxNormalization([0, 0], [512, 512], [0, 0], [512, 512])
    p = DiffusionPolicy(stub, config, norm, provenance={'kind': 'mock'})
    obs = {'pixels': np.zeros((96, 96, 3), dtype=np.uint8), 'agent_pos': np.array([0.0, 0.0])}
    before = torch.get_rng_state().clone()
    a = p.predict_action_chunk([obs, obs], scene_seed=73, replan_index=2)
    assert torch.equal(before, torch.get_rng_state())
    assert stub.encodes == 1 and len(stub.noise_steps) == 20 and stub.noise_steps == list(range(95, -1, -5))
    assert stub.training and not stub.child.training
    np.testing.assert_array_equal(a, p.predict_action_chunk([obs, obs], scene_seed=73, replan_index=2))
    assert not np.array_equal(a, p.predict_action_chunk([obs, obs], scene_seed=73, replan_index=3))
    assert a.shape == (16, 2) and a.min() >= 0 and a.max() <= 512
    scheduler = make_scheduler(config)
    assert not scheduler.config.clip_sample and not scheduler.config.thresholding
    scheduler.set_timesteps(20)
    x = torch.randn((1, 16, 2), generator=scene_generator(73, 2))
    for k in scheduler.timesteps:
        x = scheduler.step(torch.zeros_like(x), k, x, eta=0.0).prev_sample
    assert x.abs().max() > 1  # proves normalized intermediate/sample is not clipped
    np.testing.assert_array_equal(a, np.clip(norm.denormalize_action(x)[0].numpy(), 0, 512))


def test_ddpm_diagnostic_factory_and_path():
    c = default_config()
    scheduler = make_scheduler(c, diagnostic_ddpm=True)
    assert scheduler.config.num_train_timesteps == 100 and not scheduler.config.clip_sample
    stub = ZeroOutputInterfaceStub()
    norm = MinMaxNormalization([0, 0], [512, 512], [0, 0], [512, 512])
    p = DiffusionPolicy(stub, c, norm, provenance={'kind': 'mock'})
    out = p.sample(
        torch.zeros(1, 2, 3, 96, 96), torch.zeros(1, 2, 2), generator=scene_generator(4, 0), diagnostic_ddpm=True
    )
    assert len(stub.noise_steps) == 100 and np.isfinite(out).all()
