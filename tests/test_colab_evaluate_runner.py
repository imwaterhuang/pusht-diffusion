"""Mock-only persistence checks: these fixtures are not trained-policy evidence."""
import argparse
from copy import deepcopy
import importlib.util
from importlib.metadata import version
from pathlib import Path

import pytest

from pusht_diffusion.evaluation import recompute
from pusht_diffusion.utils import atomic_json, canonical_hash


spec = importlib.util.spec_from_file_location(
    "colab_evaluate", Path(__file__).parents[1] / "scripts" / "colab_evaluate.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_resume_and_drive_recovery_without_repeating_completed_scenes(tmp_path, monkeypatch):
    checkpoint, audit, scene_path = [tmp_path / name for name in ("fake.pt", "audit.json", "scenes.json")]
    checkpoint.write_bytes(b"mock-only")
    atomic_json(audit, {})
    records = [{"scene_id": f"mock-only-{i}", "environment_seed": i, "state": {}} for i in range(2)]
    pool = {"count": 2, "split": "development", "scenes": records, "scenes_hash": canonical_hash(records)}
    atomic_json(scene_path, pool)
    monkeypatch.setattr(runner, "frozen_scenes", lambda _: pool)
    monkeypatch.setattr(runner, "load_diffusion_policy", lambda *a, **kw: object())
    calls = []

    def fake_evaluate(policy, scenes, output, **kwargs):
        record = scenes["scenes"][0]
        calls.append(record["scene_id"])
        assert kwargs == {"max_steps": 300, "evidence_kind": "formal"}
        result = {
            "schema": "pusht_results_v1", "evidence_kind": "formal", "policy": "unet",
            "protocol": runner.PROTOCOL, "scenes_hash": pool["scenes_hash"], "scene_split": "development",
            "provenance": {"checkpoint_sha256": runner.file_hash(checkpoint)},
            "evaluator_code_fingerprint": runner.code_fingerprint(),
            "environment": {"gym_pusht": version("gym-pusht"), "gymnasium": version("gymnasium")},
            "episodes": [{**record, "initial_coverage": 0.0, "plans": [{"latency_ms": 1.0}],
                          "trace": [{"step": 1, "coverage": 0.96, "reward": 1.0}]}],
        }
        result["summary"] = recompute(result)
        return result

    monkeypatch.setattr(runner, "evaluate", fake_evaluate)
    args = argparse.Namespace(checkpoint=checkpoint, audit=audit, scenes=scene_path,
                              output_dir=tmp_path / "local", mirror_dir=tmp_path / "drive",
                              device="cpu", cpu_threads=1)
    first = runner.run(args)
    assert len(calls) == 2 and first["summary"]["episode_count"] == 2
    assert first["runner"]["complete"]
    assert runner.run(args) == first and len(calls) == 2
    # A new Colab machine has no local files, but recovers both scenes from Drive.
    args.output_dir = tmp_path / "new-machine"
    assert runner.run(args) == first and len(calls) == 2
    envelope_path = args.output_dir / "episodes" / "0000.json"
    envelope = runner.read(envelope_path)
    envelope["result"]["episodes"][0]["trace"][0]["coverage"] = 0.1
    atomic_json(envelope_path, envelope)
    with pytest.raises(ValueError, match="checksum"):
        runner.run(args)


def test_amended_final_requires_v2_protocol_and_hashed_lock(tmp_path, monkeypatch):
    path = tmp_path / "final50.json"
    records = [{"scene_id": f"final-{i}", "environment_seed": i, "state": {
        "agent_x": 100, "agent_y": 100 + i, "block_x": 250, "block_y": 250, "block_angle": 0,
    }} for i in range(50)]
    lock = {"models": {"unet": {"checkpoint_sha256": "example"}}}
    lock["lock_hash"] = canonical_hash(lock)
    pool = {"schema": "pusht_scenes_v2", "split": "final", "count": 50,
            "scenes": records, "scenes_hash": canonical_hash(records),
            "environment_version": version("gym-pusht"), "protocol_amendment": runner.AMENDMENT,
            "selection": lock, "exclusions": {"sources": [{"path": "development"}], "seeds": [], "states": []},
            "protocol": {"max_steps": 300, "execute_steps": 4, "thresholds": [0.87, 0.95],
                         "comparison": ">", "sampling": runner.default_config()["sampling"]}}
    monkeypatch.setattr(runner, "validate_scene", lambda _: None)
    atomic_json(path, pool)
    assert runner.frozen_scenes(path) == pool
    for key, value in (("schema", "pusht_scenes_v1"), ("protocol", {}), ("selection", {})):
        invalid = deepcopy(pool)
        invalid[key] = value
        atomic_json(path, invalid)
        with pytest.raises(ValueError):
            runner.frozen_scenes(path)
