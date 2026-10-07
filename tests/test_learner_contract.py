"""Run these after the learner completes modules. No stand-in model or trainer.

Unimplemented constructors/methods are reported as explicit skips, never replaced.
These checks deliberately do not implement loss, backward, or parameter updates.
"""

import pytest
import torch
from pusht_diffusion.config import default_config
from pusht_diffusion.learner.models import LearnerDiffusionModel
from pusht_diffusion.runtime import isolated_random_state, inference_context


@pytest.mark.parametrize('name', ['unet', 'dit'])
def test_real_learner_interface_and_weight_reload(name, tmp_path):
    config = default_config(name)
    with isolated_random_state():
        try:
            model = LearnerDiffusionModel(config, initialize_pretrained=False)
        except NotImplementedError as exc:
            pytest.skip(f'Learner has not implemented {name}: {exc}')
    assert sum(p.numel() for p in model.parameters()) > 0, 'Real learner model must contain parameters'
    g = torch.Generator().manual_seed(48)
    images = torch.randn((2, 2, 3, 96, 96), generator=g)
    positions = torch.rand((2, 2, 2), generator=g) * 2 - 1
    noisy = torch.randn((2, 16, 2), generator=g)
    k = torch.tensor([1, 90])
    try:
        with inference_context(model):
            condition = model.encode_observation(images, positions)
            assert condition.shape == (2, 516) and torch.isfinite(condition).all()
            changed = model.encode_observation(images, positions + 0.1)
            assert not torch.equal(condition, changed), 'Position condition is disconnected'
            prediction = model.denoise(noisy, k, condition)
            assert prediction.shape == (2, 16, 2) and torch.isfinite(prediction).all()
        path = tmp_path / f'{name}-state.pt'
        torch.save(model.state_dict(), path)
        with isolated_random_state():
            reloaded = LearnerDiffusionModel(config, initialize_pretrained=False)
        reloaded.load_state_dict(torch.load(path, weights_only=True), strict=True)
        with inference_context(reloaded):
            condition2 = reloaded.encode_observation(images, positions)
            prediction2 = reloaded.denoise(noisy, k, condition2)
        torch.testing.assert_close(condition, condition2, rtol=0, atol=0)
        torch.testing.assert_close(prediction, prediction2, rtol=0, atol=0)
    except NotImplementedError as exc:
        pytest.skip(f'Learner forward is unfinished: {exc}')


def test_user_loss_contract_when_implemented():
    from pusht_diffusion.learner.training import diffusion_loss

    predicted = torch.tensor([[[1.0, 0.0], [float('nan'), float('nan')]]])
    target = torch.zeros_like(predicted)
    mask = torch.tensor([[True, False]])
    try:
        loss = diffusion_loss(predicted, target, mask)
    except NotImplementedError as exc:
        pytest.skip(f'Learner loss is unfinished: {exc}')
    assert loss.shape == () and torch.isfinite(loss) and loss.item() == pytest.approx(0.5)
