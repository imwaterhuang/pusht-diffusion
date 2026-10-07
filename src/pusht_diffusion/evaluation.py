"""Policy-independent evaluation; all metrics recompute from raw per-step traces."""

from __future__ import annotations
from collections import deque
import math
import time
from pathlib import Path
from importlib.metadata import version
import json
import numpy as np
from .environment import SceneState, make_env
from .utils import atomic_json, canonical_hash, file_hash, code_fingerprint
from .runtime import isolated_random_state, noise_seed

SCHEMA = 'pusht_results_v1'


def wilson(success: int, n: int) -> list[float]:
    if n < 1:
        return [0.0, 1.0]
    z = 1.959963984540054
    p = success / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0, center - margin), min(1, center + margin)]


def episode_metrics(episode: dict) -> dict:
    trace = episode['trace']
    coverages = [float(episode['initial_coverage'])] + [float(row['coverage']) for row in trace]
    if not np.isfinite(coverages).all() or min(coverages) < 0 or max(coverages) > 1.000001:
        raise ValueError('Invalid coverage trace')
    if [row['step'] for row in trace] != list(range(1, len(trace) + 1)):
        raise ValueError('Non-contiguous step trace')
    first = {
        str(int(threshold * 100)): next((i for i, c in enumerate(coverages) if c > threshold), None)
        for threshold in (0.87, 0.95)
    }
    return {
        'success_87': first['87'] is not None,
        'success_95': first['95'] is not None,
        'first_crossing_87': first['87'],
        'first_crossing_95': first['95'],
        'final_coverage': coverages[-1],
        'max_coverage': max(coverages),
        'steps': len(trace),
        'reward': sum(float(row['reward']) for row in trace),
    }


def recompute(result: dict) -> dict:
    if result.get('schema') != SCHEMA or not result.get('episodes'):
        raise ValueError('Empty/incompatible result')
    metrics = [episode_metrics(ep) for ep in result['episodes']]
    n = len(metrics)
    latencies = [p['latency_ms'] for ep in result['episodes'] for p in ep['plans']]
    out = {
        'episode_count': n,
        'mean_final_coverage': float(np.mean([x['final_coverage'] for x in metrics])),
        'mean_max_coverage': float(np.mean([x['max_coverage'] for x in metrics])),
        'mean_steps': float(np.mean([x['steps'] for x in metrics])),
        'inference_p50_ms': float(np.percentile(latencies, 50)) if latencies else None,
        'inference_p95_ms': float(np.percentile(latencies, 95)) if latencies else None,
    }
    for threshold in (87, 95):
        success = sum(x[f'success_{threshold}'] for x in metrics)
        out[f'success_count_{threshold}'] = success
        out[f'success_rate_{threshold}'] = success / n
        out[f'wilson_95_{threshold}'] = wilson(success, n)
        crossings = [x[f'first_crossing_{threshold}'] for x in metrics if x[f'first_crossing_{threshold}'] is not None]
        out[f'mean_first_crossing_{threshold}'] = float(np.mean(crossings)) if crossings else None
    return out


def checkpoint_score(summary: dict, step: int) -> tuple:
    return (summary['success_count_87'], summary['mean_max_coverage'], -step)


def paired_metrics(left: dict, right: dict) -> dict:
    if (
        left['scenes_hash'] != right['scenes_hash']
        or left['protocol'] != right['protocol']
        or left['evidence_kind'] != right['evidence_kind']
    ):
        raise ValueError('Pair requires identical scenes, protocol and evidence kind')
    a = {e['scene_id']: episode_metrics(e) for e in left['episodes']}
    b = {e['scene_id']: episode_metrics(e) for e in right['episodes']}
    if a.keys() != b.keys():
        raise ValueError('Scene set differs')
    result = {}
    for threshold in (87, 95):
        groups = {'both_success': [], 'left_only': [], 'right_only': [], 'both_failed': []}
        for scene in a:
            x, y = a[scene][f'success_{threshold}'], b[scene][f'success_{threshold}']
            group = 'both_success' if x and y else 'left_only' if x else 'right_only' if y else 'both_failed'
            groups[group].append(scene)
        result[str(threshold)] = {k: {'count': len(v), 'scenes': v} for k, v in groups.items()}
    return result


def evaluate(
    policy,
    scenes: dict,
    output: str | Path,
    *,
    max_steps: int = 300,
    evidence_kind: str = 'formal',
    env_factory=make_env,
) -> dict:
    if not 1 <= max_steps <= 300:
        raise ValueError('Evaluation must use 1..300 steps')
    if evidence_kind not in ('formal', 'real_smoke', 'mock'):
        raise ValueError('Unknown evidence kind')
    if evidence_kind == 'formal':
        if (
            max_steps != 300
            or policy.provenance.get('kind') != 'trained_policy'
            or scenes['split'] not in ('development', 'final')
        ):
            raise ValueError('Formal results require a trained policy, frozen pool and full 300-step protocol')
    if scenes['split'] == 'final':
        expected = scenes['selection']['models'].get(policy.name)
        if (
            evidence_kind != 'formal'
            or not expected
            or expected['checkpoint_sha256'] != policy.provenance.get('checkpoint_sha256')
        ):
            raise ValueError('Final pool permits locked selected checkpoints only')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite result evidence')

    # 每个场景独立重置，保存原始规划和逐步执行记录。
    episodes = []
    with isolated_random_state():
        for record in scenes['scenes']:
            env = env_factory(max_steps)
            history = deque(maxlen=2)
            trace = []
            plans = []
            try:
                obs, info = env.reset(
                    seed=record['environment_seed'],
                    options={'reset_to_state': SceneState(**record['state']).as_array()},
                )
                history.extend([obs, obs])
                initial_coverage = float(info.get('coverage', 0))
                initial_position = np.asarray(obs['agent_pos']).tolist()
                reason = 'threshold_95' if initial_coverage > 0.95 else 'time_limit'
                ended = initial_coverage > 0.95

                while len(trace) < max_steps and not ended:
                    # 观测历史 -> 预测 16 步动作 -> 检查物理坐标范围。
                    tick = time.perf_counter()
                    actions = np.asarray(
                        policy.predict_action_chunk(
                            list(history), scene_seed=record['environment_seed'], replan_index=len(plans)
                        ),
                        dtype=np.float32,
                    )
                    latency = (time.perf_counter() - tick) * 1000
                    if (
                        actions.shape != (16, 2)
                        or not np.isfinite(actions).all()
                        or (actions < 0).any()
                        or (actions > 512).any()
                    ):
                        raise ValueError('Policy must return legal finite physical actions [16,2]')
                    plans.append(
                        {
                            'replan_index': len(plans),
                            'at_step': len(trace),
                            'noise_seed': (
                                noise_seed(
                                    record['environment_seed'],
                                    len(plans),
                                    policy.provenance.get('sampling', {}).get('noise_seed', 0),
                                )
                                if policy.name in ('unet', 'dit')
                                else None
                            ),
                            'latency_ms': latency,
                            'actions': actions.tolist(),
                        }
                    )

                    # 每次只执行前 4 步，每一步都更新相邻观测历史。
                    for offset, action in enumerate(actions[:4]):
                        before = np.asarray(history[-1]['agent_pos']).tolist()
                        obs, reward, terminated, truncated, info = env.step(action)
                        history.append(obs)
                        coverage = float(info.get('coverage', 0))
                        trace.append(
                            {
                                'step': len(trace) + 1,
                                'plan_index': len(plans) - 1,
                                'action_offset': offset,
                                'action': action.tolist(),
                                'agent_position_before': before,
                                'agent_position': np.asarray(obs['agent_pos']).tolist(),
                                'coverage': coverage,
                                'reward': float(reward),
                                'terminated': bool(terminated),
                                'truncated': bool(truncated),
                            }
                        )
                        if coverage > 0.95:
                            reason = 'threshold_95'
                            ended = True
                        elif terminated:
                            reason = 'environment_terminated'
                            ended = True
                        elif truncated or len(trace) >= max_steps:
                            reason = 'time_limit'
                            ended = True
                        if ended:
                            break

                episode = {
                    **record,
                    'initial_coverage': initial_coverage,
                    'initial_position': initial_position,
                    'trace': trace,
                    'plans': plans,
                    'terminal_reason': reason,
                }
                episode['metrics'] = episode_metrics(episode)
                episodes.append(episode)
            finally:
                env.close()

    # 从原始轨迹重算汇总指标，再保存评测证据。
    result = {
        'schema': SCHEMA,
        'evidence_kind': evidence_kind,
        'policy': policy.name,
        'provenance': policy.provenance,
        'scenes_hash': scenes['scenes_hash'],
        'scene_split': scenes['split'],
        'protocol': {
            'max_steps': max_steps,
            'execute_steps': 4,
            'thresholds': [0.87, 0.95],
            'comparison': '>',
            'stop': 'coverage > .95 or environment done or max_steps',
            'history_update': 'every_environment_step',
        },
        'environment': {'gym_pusht': version('gym-pusht'), 'gymnasium': version('gymnasium')},
        'evaluator_code_fingerprint': code_fingerprint(),
        'episodes': episodes,
    }
    result['summary'] = recompute(result)
    atomic_json(output, result, frozen=True)
    return result


def render_report(paths: list[str | Path], output: str | Path) -> dict:
    from itertools import combinations

    results = [json.loads(Path(p).read_text()) for p in paths]
    data = {
        'results': [
            {
                'source': str(Path(p).resolve()),
                'sha256': file_hash(p),
                'policy': r['policy'],
                'evidence_kind': r['evidence_kind'],
                'summary': recompute(r),
            }
            for p, r in zip(paths, results, strict=True)
        ],
        'paired': [],
    }
    for i, j in combinations(range(len(results)), 2):
        left, right = results[i], results[j]
        item = {
            'left': left['policy'],
            'right': right['policy'],
            'left_source': str(paths[i]),
            'right_source': str(paths[j]),
        }
        try:
            item['metrics'] = paired_metrics(left, right)
        except ValueError as exc:
            item['unavailable_reason'] = str(exc)
        data['paired'].append(item)
    atomic_json(output, data)
    lines = [
        '# Push-T 评测报告',
        '',
        '证据类别：formal = 正式评测；real_smoke = 真实短闭环工程检查；mock = 模拟接口检查。',
        '',
        '| 策略 | 证据类别 | 场景数 | >87% 成功 | >95% 成功 | 平均最大覆盖率 | 规划中位耗时（毫秒） |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    provenance_lines = []
    for item in data['results']:
        m = item['summary']
        latency = m['inference_p50_ms']
        lines.append(
            f"| {item['policy']} | {item['evidence_kind']} | {m['episode_count']} | {m['success_count_87']} | {m['success_count_95']} | {m['mean_max_coverage']:.4f} | {latency if latency is not None else '无规划'} |"
        )
        provenance_lines.extend(
            [
                '',
                f"来源：`{item['source']}`；文件校验：`{item['sha256']}`。",
                f"87% 成功率的 Wilson 95% 置信区间：{m['wilson_95_87']}；95% 阈值对应区间：{m['wilson_95_95']}。",
            ]
        )
    lines.extend(provenance_lines)
    lines.extend(['', '## 配对结果', ''])
    for pair in data['paired']:
        lines.extend([f"### {pair['left']} / {pair['right']}", ''])
        if 'unavailable_reason' in pair:
            lines.append('不可比较：' + pair['unavailable_reason'])
            continue
        for threshold, groups in pair['metrics'].items():
            lines.append(
                f"覆盖率 > {threshold}%：双方成功 {groups['both_success']['count']}；仅前者 {groups['left_only']['count']}；仅后者 {groups['right_only']['count']}；双方失败 {groups['both_failed']['count']}。"
            )
    lines.extend(
        [
            '',
            '以上区间只反映本批场景的不确定性，不代表跨训练种子的稳定性。真实短闭环和模拟检查不得解释为策略正式效果。',
            '',
        ]
    )
    Path(output).with_suffix('.md').write_text('\n'.join(lines))
    return data


def replay_video(result_path: str | Path, scene_id: str, output: str | Path) -> dict:
    import imageio.v3 as iio

    result = json.loads(Path(result_path).read_text())
    if result.get('evidence_kind') == 'mock':
        raise ValueError('Mock traces cannot become real evidence videos')
    if result['environment']['gym_pusht'] != version('gym-pusht'):
        raise ValueError('Replay environment version mismatch')
    episode = next(e for e in result['episodes'] if e['scene_id'] == scene_id)
    env = make_env(result['protocol']['max_steps'])
    frames = []
    errors = []
    try:
        env.reset(
            seed=episode['environment_seed'], options={'reset_to_state': SceneState(**episode['state']).as_array()}
        )
        frames.append(np.asarray(env.render()))
        for row in episode['trace']:
            _, _, _, _, info = env.step(np.asarray(row['action'], dtype=np.float32))
            errors.append(abs(float(info['coverage']) - row['coverage']))
            frames.append(np.asarray(env.render()))
    finally:
        env.close()
    if max(errors, default=0) > 1e-7:
        raise ValueError('Replay diverged from recorded coverage')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output, np.stack(frames), plugin='pyav', fps=10, codec='libx264', out_pixel_format='yuv420p')
    proof = {
        'source_results': str(Path(result_path).resolve()),
        'source_sha256': file_hash(result_path),
        'scene_id': scene_id,
        'evidence_kind': result['evidence_kind'],
        'video_sha256': file_hash(output),
        'max_coverage_error': max(errors, default=0),
        'frames': len(frames),
        'method': 'deterministic replay of saved physical actions',
    }
    atomic_json(output.with_suffix('.provenance.json'), proof)
    return proof
