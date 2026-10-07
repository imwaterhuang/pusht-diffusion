"""Engineering tests with synthetic result envelopes; no policy quality claims."""
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import select_and_freeze as selection
from pusht_diffusion.config import default_config
from pusht_diffusion.evaluation import recompute
from pusht_diffusion.utils import atomic_json, canonical_hash, file_hash


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    development = repo / "scenes/development.json"
    pool = selection.frozen_scenes(development)
    audit = json.loads((repo / "reports/data-audit.json").read_text())
    audit_path = tmp_path / "audit.json"
    atomic_json(audit_path, audit)
    root = tmp_path / "evaluations"
    provenances = {}
    for step in selection.STEPS:
        checkpoint = tmp_path / f"step_{step}.pt"
        checkpoint.write_bytes(f"synthetic weights {step}".encode())
        digest = file_hash(checkpoint)
        atomic_json(checkpoint.with_suffix(".pt.sha256.json"), {"sha256": digest})
        provenance = {"step": step, "weights": "ema", "kind": "trained_policy", "model_type": "unet",
                      "sampling": default_config()["sampling"], "config_fingerprint": canonical_hash(default_config()),
                      "data_fingerprint": audit["dataset_fingerprint"], "checkpoint": str(checkpoint),
                      "checkpoint_sha256": digest}
        provenances[str(checkpoint)] = provenance
        # 10000 和 15000 并列最优，必须按规则选较早的 10000。
        coverage = 0.9 if step in (10000, 15000) else 0.5
        episodes = [{**record, "initial_coverage": coverage, "trace": [], "plans": []}
                    for record in pool["scenes"]]
        runtime = {"device": "synthetic-cpu-test"}
        manifest = {"checkpoint_sha256": digest, "scenes_file_sha256": file_hash(development),
                    "scenes_hash": pool["scenes_hash"], "audit_sha256": canonical_hash(audit),
                    "runtime": runtime, "evaluator_code_fingerprint": "test-evaluator", "runner_sha256": "test-runner"}
        result = {"schema": "pusht_results_v1", "evidence_kind": "formal", "policy": "unet",
                  "scene_split": "development", "scenes_hash": pool["scenes_hash"], "protocol": selection.PROTOCOL,
                  "provenance": provenance, "episodes": episodes, "environment": {"test": True},
                  "evaluator_code_fingerprint": "test-evaluator",
                  "runner": {"complete": True, "expected_scene_count": 50, "runtime": runtime,
                             "manifest_sha256": canonical_hash(manifest)}}
        result["summary"] = recompute(result)
        directory = root / f"step-{step:06d}"
        atomic_json(directory / "result.json", result)
        atomic_json(directory / "manifest.json", manifest)
    monkeypatch.setattr(selection, "load_diffusion_policy", lambda path, **kwargs:
                        SimpleNamespace(provenance=deepcopy(provenances[str(path)])))
    return Namespace(evaluation_dir=root, development=development, audit=audit_path,
                     output_dir=tmp_path / "selected", exclude=[], seed=5678)


def test_select_tie_freeze_and_reuse_without_redraw(evidence, monkeypatch):
    first = selection.run(evidence)
    assert first["chosen_step"] == 10000
    pool = selection.frozen_scenes(Path(first["final_scenes"]))
    assert len(pool["scenes"]) == 50
    assert len(set(record["environment_seed"] for record in pool["scenes"])) == 50
    assert pool["protocol"] == selection.FINAL_PROTOCOL
    assert pool["selection"]["video_example_rule"] == selection.VIDEO_RULE
    snapshot = Path(first["final_scenes"]).read_bytes()
    monkeypatch.setattr(selection, "create_pool", lambda *args: pytest.fail("Unexpected redraw"))
    evidence.seed = None
    second = selection.run(evidence)
    assert second["reused"] is True
    assert Path(first["final_scenes"]).read_bytes() == snapshot
    assert file_hash(first["best"]) == first["chosen_sha256"]


@pytest.mark.parametrize("mutation", ["partial", "summary", "order", "runtime", "checkpoint"])
def test_reject_corrupted_evidence(evidence, mutation):
    path = evidence.evaluation_dir / "step-020000/result.json"
    result = selection.read(path)
    if mutation == "partial":
        result["runner"]["complete"] = False
    elif mutation == "summary":
        result["summary"]["success_count_87"] += 1
    elif mutation == "order":
        result["episodes"].reverse()
    elif mutation == "runtime":
        result["runner"]["runtime"] = {"device": "different"}
    else:
        Path(result["provenance"]["checkpoint"]).write_bytes(b"corrupted")
    atomic_json(path, result)
    with pytest.raises(ValueError):
        selection.run(evidence)
    assert not (evidence.output_dir / "selection.json").exists()


def test_reject_requested_redraw(evidence):
    selection.run(evidence)
    evidence.seed += 1
    with pytest.raises(ValueError, match="refusing redraw"):
        selection.run(evidence)
