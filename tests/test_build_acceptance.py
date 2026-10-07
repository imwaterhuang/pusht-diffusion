"""Synthetic-only report checks; temporary movies are not real policy evidence."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from pusht_diffusion.evaluation import episode_metrics, recompute
from pusht_diffusion.utils import atomic_json, canonical_hash, file_hash

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location("build_acceptance", Path(__file__).parents[1] / "scripts/build_acceptance.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def fixture_bundle(tmp_path, monkeypatch):
    records = [{"scene_id": f"final-{i:03d}", "environment_seed": i, "state": {
        "agent_x": 100, "agent_y": 100 + i, "block_x": 250, "block_y": 250, "block_angle": 0,
    }} for i in range(50)]
    episodes = []
    for i, record in enumerate(records):
        coverage = .96 if i % 2 == 0 else .5
        episode = {**record, "initial_coverage": 0.0, "plans": [{"latency_ms": 10.0}],
                   "trace": [{"step": 1, "coverage": coverage, "reward": coverage,
                              "terminated": True, "truncated": False, "action": [coverage, 0]}],
                   "terminal_reason": "environment_terminated"}
        episode["metrics"] = episode_metrics(episode)
        episodes.append(episode)
    candidates = [{"step": step, "checkpoint_sha256": f"mock-{step}",
                   "summary": {"success_count_87": step // 5000, "mean_max_coverage": .8}}
                  for step in range(5000, 40001, 5000)]
    exclusions = {"seeds": [], "states": [], "sources": []}
    selection = {"schema": "pusht_selection_v2", "models": {"unet": {"checkpoint_sha256": "mock-40000"}},
                 "exclusions_hash": canonical_hash(exclusions), "final_generation_seed": 7,
                 "candidates": candidates, "chosen_step": 40000, "chosen_sha256": "mock-40000",
                 "score_order": ["success_count_87", "mean_max_coverage", "earlier_step"],
                 "video_example_rule": {"success_example": "first scene_id in frozen order with success_95",
                                        "failure_example": "first scene_id in frozen order without success_87"}}
    selection["lock_hash"] = canonical_hash(selection)
    pool = {"schema": "pusht_scenes_v2", "count": 50, "split": "final", "protocol_amendment": builder.AMENDMENT,
            "scenes": records, "scenes_hash": canonical_hash(records), "selection": selection,
            "exclusions": exclusions, "generation_seed": 7}
    scenes_path, selection_path, results_path = [tmp_path / f"{n}.json" for n in ("scenes", "selection", "result")]
    atomic_json(scenes_path, pool)
    atomic_json(selection_path, selection)
    manifest = {"scenes_file_sha256": file_hash(scenes_path), "checkpoint_sha256": "mock-40000",
                "runtime": {}, "evaluator_code_fingerprint": "mock-code"}
    atomic_json(tmp_path / "manifest.json", manifest)
    result = {"schema": "pusht_results_v1", "evidence_kind": "formal", "policy": "unet", "scene_split": "final",
              "scenes_hash": pool["scenes_hash"], "protocol": builder.PROTOCOL, "episodes": episodes,
              "evaluator_code_fingerprint": "mock-code",
              "runner": {"complete": True, "expected_scene_count": 50, "protocol_amendment": builder.AMENDMENT,
                         "manifest_sha256": canonical_hash(manifest), "runtime": {}},
              "provenance": {"checkpoint_sha256": "mock-40000", "step": 40000, "weights": "ema",
                             "kind": "trained_policy", "sampling": builder.default_config()["sampling"],
                             "config_fingerprint": canonical_hash(builder.default_config())}}
    result["summary"] = recompute(result)
    atomic_json(results_path, result)
    monkeypatch.setattr(builder, "frozen_scenes", lambda _: pool)
    return results_path, scenes_path, selection_path, result


def test_validates_full_trace_and_rejects_corrupt_summary_or_scene(tmp_path, monkeypatch):
    results, scenes, selection, original = fixture_bundle(tmp_path, monkeypatch)
    _, _, lock, rows = builder.validate_inputs(results, scenes, selection)
    assert builder.choose_videos(rows, lock) == {"success": "final-000", "failure": "final-001"}
    bad = deepcopy(original)
    bad["summary"]["success_count_87"] += 1
    atomic_json(results, bad)
    with pytest.raises(ValueError, match="summary"):
        builder.validate_inputs(results, scenes, selection)
    bad = deepcopy(original)
    bad["episodes"][0]["scene_id"] = "wrong-scene"
    atomic_json(results, bad)
    with pytest.raises(ValueError, match="scene order"):
        builder.validate_inputs(results, scenes, selection)
    all_failed = [{**row, "success_95": False, "success_87": False} for row in rows]
    assert builder.choose_videos(all_failed, lock)["success"] is None


def test_streamed_montage_has_every_scene_and_verified_replay(tmp_path, monkeypatch):
    pytest.importorskip("av")
    results, _, _, result = fixture_bundle(tmp_path, monkeypatch)
    calls = []

    class FakeEnv:
        def reset(self, seed, options):
            calls.append(seed)
            self.coverage = 0.0
            return {}, {"coverage": 0.0}

        def step(self, action):
            self.coverage = float(action[0])
            return {}, 0.0, True, False, {"coverage": self.coverage}

        def render(self):
            return np.full((64, 64, 3), round(self.coverage * 255), dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr(builder, "make_env", lambda _: FakeEnv())
    proof = builder.montage(result, results, tmp_path)
    assert calls == list(range(50))
    assert proof["scene_count"] == 50 and proof["frames"] == 10
    assert proof["max_coverage_error"] < 1e-7
    assert builder.montage(result, results, tmp_path) == proof
    assert len(calls) == 50  # Existing valid evidence is reused, not rerun.
    gif_proof = builder.make_gif(tmp_path / "all50.mp4", tmp_path / "preview.gif")
    assert gif_proof["fps"] == 5 and gif_proof["frames"] == 5
    from PIL import Image
    with Image.open(tmp_path / "preview.gif") as image:
        assert image.width <= 320 and image.height <= 320
