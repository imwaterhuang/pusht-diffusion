"""Read-only reference comparison and real ACT smoke evidence; never trains.

The old model file is loaded only for this optional migration check. Production
package code has no imports/path dependencies on the old workspace.
"""

from __future__ import annotations
import argparse
from collections import deque
import importlib.util
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import json
import time
import numpy as np
import torch
from pusht_diffusion.data import PushTDataset, audit_data
from pusht_diffusion.normalization import observation_tensors
from pusht_diffusion.policies import load_act_policy
from pusht_diffusion.runtime import isolated_random_state, inference_context
from pusht_diffusion.scenes import load_scenes
from pusht_diffusion.evaluation import evaluate, recompute, replay_video, render_report
from pusht_diffusion.environment import make_env, SceneState
from pusht_diffusion.ownership import check_ownership
from pusht_diffusion.utils import atomic_json, file_hash, code_fingerprint


def main():
    p = argparse.ArgumentParser()
    for name in ('reference', 'dataset', 'act', 'audit', 'scenes', 'output'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    audit = audit_data(a.dataset)
    assert audit == json.loads(Path(a.audit).read_text())
    d = PushTDataset(a.dataset, audit=audit)
    boundary_indices = sorted(
        {
            0,
            len(d) - 1,
            *[i for i, w in enumerate(d.windows) if w[1] == 0],
            *[i for i, w in enumerate(d.windows) if w[2] == w[4] - 2],
        }
    )
    # Exercise scalar/window contracts for every episode, decode only first/last and sample history.
    for idx in boundary_indices:
        e, local, t, start, end = d.windows[idx]
        assert start <= t < end - 1 and local == t - start
    samples = [d[i] for i in (0, 1, len(d) - 1)]
    for sample in samples:
        assert sample['images'].shape == (2, 3, 96, 96)
        assert sample['positions'].shape == (2, 2)
        assert sample['actions'].shape == (16, 2)
        assert sample['valid_mask'].dtype == torch.bool
    count_before = d.images.decode_count
    d[0]
    assert d.images.decode_count == count_before
    source = Path(a.reference) / 'src/mini_wam/models/act.py'
    spec = importlib.util.spec_from_file_location('reference_act_migration_check', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = torch.load(a.act, map_location='cpu', weights_only=True)
    options = {k: v for k, v in payload['config']['model'].items() if k != 'name'}
    options['pretrained'] = False
    with isolated_random_state():
        original = module.ActionPolicy(**options).eval()
        original.load_state_dict(payload['model'], strict=True)
    adapter = load_act_policy(a.act, audit=audit)
    scene = load_scenes(a.scenes)['scenes'][0]
    env = make_env(12)
    parity_samples = []
    try:
        obs, _ = env.reset(
            seed=scene['environment_seed'], options={'reset_to_state': SceneState(**scene['state']).as_array()}
        )
        history = deque([obs, obs], maxlen=2)
        for sample_index in range(8):
            images, positions = observation_tensors(list(history), adapter.normalization)
            # The independently loaded original always consumes only the LATEST frame.
            with inference_context(original):
                baseline = original(images[:, -1], positions[:, -1])
                baseline = np.clip(adapter.normalization.denormalize_action(baseline)[0].numpy(), 0, 512)
            actual = adapter.predict_action_chunk(
                list(history), scene_seed=scene['environment_seed'], replan_index=sample_index
            )
            error = float(np.max(np.abs(baseline - actual)))
            assert error == 0
            parity_samples.append(
                {
                    'environment_step': sample_index,
                    'max_absolute_action_error': error,
                    'exact_equal': bool(np.array_equal(actual, baseline)),
                    'history_images_differ': bool(not np.array_equal(history[0]['pixels'], history[1]['pixels'])),
                    'history_positions_differ': bool(
                        not np.array_equal(history[0]['agent_pos'], history[1]['agent_pos'])
                    ),
                }
            )
            if sample_index < 7:
                obs, _, _, _, _ = env.step(actual[0])
                history.append(obs)
    finally:
        env.close()
    assert sum(x['history_images_differ'] for x in parity_samples) > 0
    error = max(x['max_absolute_action_error'] for x in parity_samples)
    result = evaluate(adapter, load_scenes(a.scenes), out / 'act-smoke.json', max_steps=12, evidence_kind='real_smoke')
    assert recompute(result) == result['summary']
    video = replay_video(out / 'act-smoke.json', scene['scene_id'], out / 'act-smoke.mp4')
    report = render_report([out / 'act-smoke.json'], out / 'smoke-report.json')
    packages = {}
    for name in (
        'torch',
        'torchvision',
        'numpy',
        'pyarrow',
        'Pillow',
        'gym-pusht',
        'gymnasium',
        'fastapi',
        'uvicorn',
        'imageio',
        'av',
        'diffusers',
        'regex',
        'importlib-metadata',
        'zipp',
        'pytest',
        'httpx',
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    proof = {
        'schema': 'pusht_engineering_evidence_v1',
        'training_executed': False,
        'code_fingerprint': code_fingerprint(),
        'data': {
            'fingerprint': audit['dataset_fingerprint'],
            'episodes': audit['episode_count'],
            'frames': audit['frame_count'],
            'windows': len(d),
            'boundary_windows_checked': len(boundary_indices),
            'images_decoded_for_sample_checks': d.images.decode_count,
            'bounded_repeated_access_no_extra_decode': True,
        },
        'act_parity': {
            'checkpoint_sha256': file_hash(a.act),
            'reference_source': str(source),
            'reference_source_sha256': file_hash(source),
            'vendored_source_sha256': file_hash('src/pusht_diffusion/baselines/act.py'),
            'observation_source': 'real simulator reset plus seven successive environment steps',
            'sample_count': len(parity_samples),
            'adjacent_different_image_pairs': sum(x['history_images_differ'] for x in parity_samples),
            'samples': parity_samples,
            'max_absolute_action_error': error,
            'exact_equal': all(x['exact_equal'] for x in parity_samples),
        },
        'act_smoke': {
            'kind': 'real_smoke',
            'path': str((out / 'act-smoke.json').resolve()),
            'steps': result['episodes'][0]['metrics']['steps'],
            'summary': result['summary'],
        },
        'video': video,
        'ownership': check_ownership(),
        'packages': packages,
        'elapsed_seconds': time.perf_counter() - started,
        'limitations': [
            'No diffusion model exists yet.',
            'No training or end-to-end optimizer resume has been tested.',
            '12-step ACT smoke is not formal effectiveness evidence.',
            'Browser verification is handled separately by parent.',
        ],
    }
    atomic_json(out / 'verification.json', proof, frozen=True)
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
