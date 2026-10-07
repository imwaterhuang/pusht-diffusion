"""Read-only inference compatibility check against an actual learner checkpoint.

Run from the repository root with PYTHONPATH=src. This performs no training and
does not claim that finite actions establish closed-loop policy quality.
"""

from __future__ import annotations

import argparse
import json
from unittest.mock import patch

import numpy as np
import torch

from pusht_diffusion.checkpoints import load_checkpoint
from pusht_diffusion.config import default_config
from pusht_diffusion.data import PushTDataset
from pusht_diffusion.policies import load_diffusion_policy
from pusht_diffusion.runtime import scene_generator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)

    dataset = PushTDataset(args.data_root)
    config = default_config("unet")
    # Inference must load checkpoint weights without downloading pretrained ones.
    with patch("torch.hub.download_url_to_file", side_effect=AssertionError("Unexpected download")):
        policy = load_diffusion_policy(args.checkpoint, config=config, audit=dataset.audit)
    checkpoint = load_checkpoint(args.checkpoint, expected_config=config)
    assert set(policy.model.state_dict()) == set(checkpoint["ema"])
    assert all(torch.equal(t, checkpoint["ema"][k]) for k, t in policy.model.state_dict().items())

    sample = dataset[1]
    images, positions = sample["images"][None], sample["positions"][None]
    x = torch.randn(1, 16, 2, generator=torch.Generator().manual_seed(7))
    k = torch.tensor([50])
    with torch.inference_mode():
        # Inherited forward is the user's implementation, not a copied reference.
        direct = policy.model(x, images, positions, k)
        condition = policy.model.encode_observation(images, positions)
        cached = policy.model.denoise(x, k, condition)
    torch.testing.assert_close(cached, direct, rtol=0, atol=0)

    a = policy.sample(images, positions, generator=scene_generator(0, 0, 0))
    b = policy.sample(images, positions, generator=scene_generator(0, 0, 0))
    assert a.shape == (16, 2) and a.dtype == np.float32
    assert np.isfinite(a).all() and ((a >= 0) & (a <= 512)).all()
    np.testing.assert_array_equal(a, b)

    try:
        load_diffusion_policy(args.checkpoint, config=default_config("dit"), audit=dataset.audit)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("Unimplemented DiT must be rejected")

    print(json.dumps({
        "kind": "real_checkpoint_inference_check",
        "checkpoint_step": checkpoint["step"],
        "ema_exact_load": True,
        "cached_denoise_equals_learner_forward": True,
        "real_dataset_sample": 1,
        "ddim_steps": config["sampling"]["steps"],
        "action_shape": list(a.shape),
        "finite_actions": True,
        "same_seed_same_actions": True,
        "dit_rejected": True,
        "limits": "No closed-loop evaluation or policy-quality claim",
    }, indent=2))


if __name__ == "__main__":
    main()
