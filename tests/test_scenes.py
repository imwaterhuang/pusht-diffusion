import json
from importlib.metadata import version
import pytest
from pusht_diffusion.scenes import load_scenes, collect_exclusions
from pusht_diffusion.environment import legal_scene
from pusht_diffusion.utils import atomic_json, canonical_hash


def test_scene_guards_environment_split_count_and_legality(tmp_path):
    seed, scene = legal_scene(0)
    records = [{'scene_id': 'debug-0', 'environment_seed': seed, 'state': scene.to_dict()}]
    base = {
        'schema': 'pusht_scenes_v1',
        'split': 'debug',
        'count': 1,
        'environment_version': version('gym-pusht'),
        'scenes': records,
        'scenes_hash': canonical_hash(records),
    }
    path = tmp_path / 'scenes.json'
    atomic_json(path, base)
    assert load_scenes(path)['count'] == 1
    for change in ({'environment_version': '0.0.0'}, {'split': 'unknown'}, {'split': 'development'}, {'count': 2}):
        atomic_json(path, {**base, **change})
        with pytest.raises(ValueError):
            load_scenes(path)
    base['scenes'][0]['state']['agent_x'] = -99
    base['scenes_hash'] = canonical_hash(base['scenes'])
    atomic_json(path, base)
    with pytest.raises(ValueError):
        load_scenes(path)


def test_jsonl_debug_exclusions(tmp_path):
    path = tmp_path / 'interactive-debug.jsonl'
    path.write_text(
        json.dumps(
            {
                'environment_seed': 55,
                'state': {'agent_x': 100, 'agent_y': 100, 'block_x': 200, 'block_y': 200, 'block_angle': 0},
            }
        )
        + '\n'
    )
    excluded = collect_exclusions([path])
    assert excluded['seeds'] == [55] and len(excluded['states']) == 1


def test_exclusion_history_is_inherited_transitively(tmp_path):
    path = tmp_path / 'development.json'
    atomic_json(
        path,
        {
            'scenes': [],
            'exclusions': {'seeds': [100, 200], 'states': [[100.0, 100.0, 200.0, 200.0, 0.0]], 'sources': []},
        },
    )
    inherited = collect_exclusions([path])
    assert inherited['seeds'] == [100, 200] and inherited['states'] == [[100.0, 100.0, 200.0, 200.0, 0.0]]
