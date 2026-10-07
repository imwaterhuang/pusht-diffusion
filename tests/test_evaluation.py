import numpy as np
import pytest
from pusht_diffusion.evaluation import evaluate, recompute, paired_metrics, checkpoint_score, episode_metrics
from pusht_diffusion.scenes import generate_scenes
from pusht_diffusion.utils import canonical_hash


class MockEnvironment:
    def __init__(self, max_steps):
        self.step_index = 0

    def obs(self):
        return {
            'pixels': np.full((96, 96, 3), self.step_index, dtype=np.uint8),
            'agent_pos': np.array([self.step_index, self.step_index]),
        }

    def reset(self, **kwargs):
        return self.obs(), {'coverage': 0.0}

    def step(self, action):
        self.step_index += 1
        return self.obs(), 0.0, False, False, {'coverage': [0.5, 0.87, 0.9, 0.8, 0.95, 0.96][self.step_index - 1]}

    def close(self):
        pass


class MockPolicy:
    name = 'mock'
    provenance = {'kind': 'mock'}

    def __init__(self):
        self.history = []

    def predict_action_chunk(self, history, **kwargs):
        self.history.append([int(o['agent_pos'][0]) for o in history])
        return np.ones((16, 2), dtype=np.float32)


def scenes():
    records = [
        {
            'scene_id': 'mock-0',
            'environment_seed': 3,
            'state': {'agent_x': 100, 'agent_y': 100, 'block_x': 250, 'block_y': 250, 'block_angle': 0},
        }
    ]
    return {'scenes': records, 'scenes_hash': canonical_hash(records), 'split': 'debug'}


def test_continue_after_87_strict_thresholds_and_adjacent_history(tmp_path):
    p = MockPolicy()
    r = evaluate(p, scenes(), tmp_path / 'mock.json', evidence_kind='mock', env_factory=MockEnvironment)
    assert p.history == [[0, 0], [3, 4]]
    m = r['episodes'][0]['metrics']
    assert m['steps'] == 6 and m['first_crossing_87'] == 3 and m['first_crossing_95'] == 6
    assert r['episodes'][0]['terminal_reason'] == 'threshold_95'
    r['summary'] = {'false_cached_value': 42}
    assert recompute(r)['success_count_87'] == 1
    assert paired_metrics(r, r)['87']['both_success']['count'] == 1
    with pytest.raises(ValueError):
        evaluate(MockPolicy(), scenes(), tmp_path / 'formal.json', evidence_kind='formal', env_factory=MockEnvironment)
    assert checkpoint_score(recompute(r), 5) > checkpoint_score(recompute(r), 6)


def test_final_creation_requires_selection(tmp_path):
    with pytest.raises(ValueError, match='selection'):
        generate_scenes(
            tmp_path / 'final.json', split='final', count=100, generation_seed=7, exclusions={'seeds': [], 'states': []}
        )
    assert not (tmp_path / 'final.json').exists()


def test_three_policy_report_has_all_pairs_and_labels(tmp_path):
    from copy import deepcopy
    from pusht_diffusion.evaluation import render_report
    from pusht_diffusion.utils import atomic_json

    base = evaluate(MockPolicy(), scenes(), tmp_path / 'base.json', evidence_kind='mock', env_factory=MockEnvironment)
    paths = []
    for name in ('act', 'unet', 'dit'):
        r = deepcopy(base)
        r['policy'] = name
        p = tmp_path / f'{name}.json'
        atomic_json(p, r)
        paths.append(p)
    report = render_report(paths, tmp_path / 'report.json')
    assert len(report['paired']) == 3
    text = (tmp_path / 'report.md').read_text()
    assert 'mock' in text and 'real_smoke' in text and 'formal' in text
    assert 'act / unet' in text and 'act / dit' in text and 'unet / dit' in text
