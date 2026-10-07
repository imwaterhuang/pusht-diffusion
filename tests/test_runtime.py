from copy import deepcopy
import os
import random
import numpy as np
import pytest
import torch
from pusht_diffusion.config import default_config
from pusht_diffusion.checkpoints import make_checkpoint, save_checkpoint, load_checkpoint
from pusht_diffusion.reproducibility import DeterministicBatchSampler, capture_training_rng, restore_training_rng
from pusht_diffusion.runtime import inference_context, scene_generator
from pusht_diffusion.ownership import check_ownership
from pusht_diffusion.learner.models import LearnerDiffusionModel
from pusht_diffusion.learner.training import train


def payload():
    sampler = DeterministicBatchSampler(17, 4, 0)
    next(iter(sampler))
    sampler.mark_consumed()
    return make_checkpoint(
        config=default_config(),
        data_fingerprint='a' * 64,
        normalization={
            'position_min': [0, 0],
            'position_max': [512, 512],
            'action_min': [0, 0],
            'action_max': [512, 512],
            'kind': 'minmax_v1',
        },
        step=1,
        raw={'engineering_state': torch.tensor([1.0])},
        ema={'engineering_state': torch.tensor([2.0])},
        optimizer={'state': {}, 'param_groups': []},
        lr={'last_epoch': 1},
        sampler=sampler.state_dict(),
        rng=capture_training_rng({'noise': torch.Generator().manual_seed(7)}),
        code_fingerprint='b' * 64,
    )


def test_checkpoint_roundtrip_mirror_and_rejection(tmp_path):
    p = payload()
    path = tmp_path / 'last.pt'
    save_checkpoint(path, p, mirror=tmp_path / 'drive')
    q = load_checkpoint(
        path, expected_config=default_config(), expected_data='a' * 64, expected_normalization=p['normalization']
    )
    assert q['sampler'] == p['sampler']
    torch.testing.assert_close(q['raw']['engineering_state'], p['raw']['engineering_state'])
    assert path.read_bytes() == (tmp_path / 'drive/last.pt').read_bytes()
    for expected in (
        {'expected_config': default_config('dit')},
        {'expected_data': 'c' * 64},
        {'expected_normalization': {}},
    ):
        with pytest.raises(ValueError):
            load_checkpoint(path, **expected)
    invalid = deepcopy(p)
    invalid['architecture']['horizon'] = 8
    with pytest.raises(ValueError):
        save_checkpoint(tmp_path / 'invalid.pt', invalid)
    path.write_bytes(path.read_bytes() + b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        load_checkpoint(path)


def test_sampler_consumed_cursor_ignores_prefetch():
    a = DeterministicBatchSampler(17, 4, 3)
    it = iter(a)
    first = next(it)
    second = next(it)
    third = next(it)
    a.mark_consumed()
    b = DeterministicBatchSampler(17, 4, 3)
    b.load_state_dict(a.state_dict())
    bt = iter(b)
    assert next(bt) == second and next(bt) == third
    assert sorted(first + second + third + next(it) + next(it)) == list(range(17))


def test_named_rng_and_mixed_modes_restore():
    g = torch.Generator().manual_seed(7)
    state = capture_training_rng({'noise': g})
    expected = torch.randn(4, generator=g)
    restore_training_rng(state, {'noise': g})
    torch.testing.assert_close(torch.randn(4, generator=g), expected)
    model = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Dropout())
    model.train()
    model[1].eval()
    before = torch.get_rng_state().clone()
    numpy = np.random.get_state()
    python = random.getstate()
    with pytest.raises(RuntimeError):
        with inference_context(model):
            assert not any(x.training for x in model.modules())
            torch.randn(5)
            np.random.randn()
            random.random()
            raise RuntimeError('interrupt')
    assert model.training and model[0].training and not model[1].training
    assert torch.equal(before, torch.get_rng_state()) and random.getstate() == python
    np.testing.assert_array_equal(numpy[1], np.random.get_state()[1])


@pytest.mark.skipif(
    os.environ.get('PUSHT_DELIVERY_AUDIT') != '1',
    reason='Delivery-only ownership audit; learner implementations are allowed after handoff',
)
def test_user_owned_boundaries():
    checked = check_ownership()['functions']
    assert any(x.endswith(':train_step') for x in checked) and any(x.endswith(':denoise') for x in checked)
    with pytest.raises(NotImplementedError):
        LearnerDiffusionModel(default_config())
    with pytest.raises(NotImplementedError):
        train(default_config(), {})
