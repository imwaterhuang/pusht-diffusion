"""Frozen development scenes; final scenes require real, locked selected models."""

from __future__ import annotations
import json
from pathlib import Path
from importlib.metadata import version
import numpy as np
from .environment import SceneState, random_scene, validate_scene
from .utils import canonical_hash, file_hash, atomic_json

STATE_KEYS = {'agent_x', 'agent_y', 'block_x', 'block_y', 'block_angle'}


def state_key(state: dict) -> tuple:
    return tuple(SceneState(**{k: state[k] for k in STATE_KEYS}).as_array().tolist())


def collect_exclusions(paths: list[str | Path]) -> dict:
    seeds = set()
    states = set()
    sources = []

    def visit(value):
        if isinstance(value, dict):
            # Frozen pools carry their prior exclusion ledger. Preserve that history
            # transitively when final scenes exclude the development/debug files.
            if {'seeds', 'states', 'sources'} <= value.keys():
                seeds.update(int(x) for x in value['seeds'])
                states.update(tuple(float(x) for x in state) for state in value['states'])
            if STATE_KEYS <= value.keys():
                try:
                    states.add(state_key(value))
                except (ValueError, TypeError):
                    pass
            for key in ('environment_seed', 'scene_seed', 'seed'):
                if type(value.get(key)) is int and value[key] >= 0:
                    seeds.add(value[key])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for path in paths:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(p)
        payload = (
            [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
            if p.suffix == '.jsonl'
            else json.loads(p.read_text())
        )
        visit(payload)
        sources.append({'path': str(p.resolve()), 'sha256': file_hash(p)})
    return {'seeds': sorted(seeds), 'states': [list(x) for x in sorted(states)], 'sources': sources}


def generate_scenes(
    path: str | Path, *, split: str, count: int, generation_seed: int, exclusions: dict, selection: dict | None = None
) -> dict:
    if split not in ('development', 'debug', 'final'):
        raise ValueError('Unknown split')
    if count < 1 or (split == 'development' and count != 50) or (split == 'final' and count != 100):
        raise ValueError('Frozen pool must contain 50 development / 100 final scenes')
    if split == 'final' and selection is None:
        raise ValueError('Final scenes require a validated selection lock')
    used_seeds = set(exclusions['seeds'])
    used_states = {tuple(x) for x in exclusions['states']}
    rng = np.random.default_rng(generation_seed)
    records = []
    for _ in range(count * 300):
        if len(records) == count:
            break
        seed = int(rng.integers(0, 2**31 - 1))
        scene = random_scene(seed)
        key = tuple(scene.as_array().tolist())
        if seed in used_seeds or key in used_states:
            continue
        try:
            validate_scene(scene)
        except ValueError:
            continue
        used_seeds.add(seed)
        used_states.add(key)
        records.append(
            {'scene_id': f'diffusion-{split}-{len(records):03d}', 'environment_seed': seed, 'state': scene.to_dict()}
        )
    if len(records) != count:
        raise RuntimeError('Could not create unique legal scenes')
    p = {
        'schema': 'pusht_scenes_v1',
        'split': split,
        'generation_seed': generation_seed,
        'environment_version': version('gym-pusht'),
        'count': count,
        'scenes_hash': canonical_hash(records),
        'scenes': records,
        'exclusions': exclusions,
        'selection': selection,
    }
    atomic_json(path, p, frozen=True)
    return p


def load_scenes(path: str | Path) -> dict:
    p = json.loads(Path(path).read_text())
    if (
        p.get('schema') != 'pusht_scenes_v1'
        or p['count'] != len(p['scenes'])
        or p['scenes_hash'] != canonical_hash(p['scenes'])
    ):
        raise ValueError('Scene schema/count/hash mismatch')
    if p.get('environment_version') != version('gym-pusht'):
        raise ValueError('Scene environment-version mismatch')
    split = p.get('split')
    if split not in ('development', 'debug', 'final'):
        raise ValueError('Invalid scene split')
    if p['count'] < 1 or (split == 'development' and p['count'] != 50) or (split == 'final' and p['count'] != 100):
        raise ValueError('Invalid scene count for split')
    ids = set()
    seeds = set()
    states = set()
    for record in p['scenes']:
        if (
            type(record.get('environment_seed')) is not int
            or record['environment_seed'] < 0
            or not isinstance(record.get('scene_id'), str)
        ):
            raise ValueError('Invalid scene seed/id')
        validate_scene(SceneState(**record['state']))
        key = state_key(record['state'])
        if record['scene_id'] in ids or record['environment_seed'] in seeds or key in states:
            raise ValueError('Duplicate frozen scenes')
        ids.add(record['scene_id'])
        seeds.add(record['environment_seed'])
        states.add(key)
    if p['split'] == 'final' and not p.get('selection'):
        raise ValueError('Final pool missing selected model lock')
    return p


def lock_selection(path: str | Path, *, unet: str, dit: str, act: str, audit: dict, development: str) -> dict:
    from .policies import load_diffusion_policy, load_act_policy
    from .config import default_config

    if load_scenes(development)['split'] != 'development':
        raise ValueError('Expected frozen development pool')
    models = {}
    for name, checkpoint in [('unet', unet), ('dit', dit)]:
        policy = load_diffusion_policy(checkpoint, config=default_config(name), audit=audit)
        if policy.provenance['step'] <= 0:
            raise ValueError('Selected checkpoint must be trained')
        models[name] = policy.provenance
    models['act'] = load_act_policy(act, audit=audit).provenance
    p = {
        'schema': 'pusht_selection_v1',
        'models': models,
        'sampling': default_config()['sampling'],
        'development_hash': file_hash(development),
        'development_path': str(Path(development).resolve()),
    }
    p['lock_hash'] = canonical_hash(p)
    atomic_json(path, p, frozen=True)
    return p


def validate_selection(path: str | Path, audit: dict) -> dict:
    from .policies import load_diffusion_policy, load_act_policy
    from .config import default_config

    p = json.loads(Path(path).read_text())
    body = {k: v for k, v in p.items() if k != 'lock_hash'}
    if p.get('schema') != 'pusht_selection_v1' or p.get('lock_hash') != canonical_hash(body):
        raise ValueError('Invalid selection lock')
    if file_hash(p['development_path']) != p['development_hash'] or p['sampling'] != default_config()['sampling']:
        raise ValueError('Development/sampling changed after lock')
    for name, record in p['models'].items():
        if file_hash(record['checkpoint']) != record['checkpoint_sha256']:
            raise ValueError('Selected checkpoint changed')
        if name in ('unet', 'dit'):
            load_diffusion_policy(record['checkpoint'], config=default_config(name), audit=audit)
        elif name == 'act':
            load_act_policy(record['checkpoint'], audit=audit)
    if set(p['models']) != {'unet', 'dit', 'act'}:
        raise ValueError('Both trained models and ACT required')
    return p
