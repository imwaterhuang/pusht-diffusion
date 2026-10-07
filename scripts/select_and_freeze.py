"""Select from all eight development evaluations, then freeze the user's 50-scene final pool."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import shutil
import tempfile

import numpy as np
from importlib.metadata import version
import torch

from colab_evaluate import AMENDMENT, PROTOCOL, frozen_scenes
from pusht_diffusion.config import default_config
from pusht_diffusion.environment import random_scene, validate_scene
from pusht_diffusion.evaluation import checkpoint_score, recompute
from pusht_diffusion.policies import load_diffusion_policy
from pusht_diffusion.scenes import collect_exclusions, state_key
from pusht_diffusion.utils import atomic_json, canonical_hash, file_hash

STEPS = tuple(range(5000, 40001, 5000))
FINAL_PROTOCOL = {key: PROTOCOL[key] for key in ("max_steps", "execute_steps", "thresholds", "comparison")}
FINAL_PROTOCOL["sampling"] = default_config("unet")["sampling"]
VIDEO_RULE = {"success_example": "first scene_id in frozen order with success_95",
              "failure_example": "first scene_id in frozen order without success_87",
              "montage": "all 50 scenes in frozen order; never select only successes"}


def read(path):
    return json.loads(Path(path).read_text())


def candidates(root: Path, development: Path, audit: dict) -> tuple[list[dict], dict]:
    pool = frozen_scenes(development)
    if pool["split"] != "development" or pool["count"] != 50:
        raise ValueError("Selection requires the frozen development50 pool")
    records, common = [], None
    for step in STEPS:
        path = root / f"step-{step:06d}" / "result.json"
        result = read(path)
        runner, provenance = result.get("runner", {}), result.get("provenance", {})
        manifest = read(path.parent / "manifest.json")
        if (result.get("evidence_kind") != "formal" or result.get("policy") != "unet"
                or result.get("scene_split") != "development" or result.get("scenes_hash") != pool["scenes_hash"]
                or result.get("protocol") != PROTOCOL or runner.get("complete") is not True
                or runner.get("expected_scene_count") != 50 or len(result.get("episodes", [])) != 50
                or provenance.get("step") != step or provenance.get("weights") != "ema"
                or provenance.get("kind") != "trained_policy" or provenance.get("model_type") != "unet"
                or provenance.get("sampling") != default_config("unet")["sampling"]
                or provenance.get("config_fingerprint") != canonical_hash(default_config("unet"))
                or provenance.get("data_fingerprint") != audit["dataset_fingerprint"]):
            raise ValueError(f"Incomplete or incompatible development evidence: {path}")
        for episode, scene in zip(result["episodes"], pool["scenes"], strict=True):
            if any(episode.get(key) != value for key, value in scene.items()):
                raise ValueError(f"Development scene order/state mismatch: {path}")
        summary = recompute(result)
        if summary != result.get("summary"):
            raise ValueError(f"Cached summary differs from trace: {path}")
        if (runner.get("manifest_sha256") != canonical_hash(manifest)
                or manifest.get("checkpoint_sha256") != provenance["checkpoint_sha256"]
                or manifest.get("scenes_file_sha256") != file_hash(development)
                or manifest.get("scenes_hash") != pool["scenes_hash"]
                or manifest.get("audit_sha256") != canonical_hash(audit)
                or manifest.get("runtime") != runner.get("runtime")
                or manifest.get("evaluator_code_fingerprint") != result.get("evaluator_code_fingerprint")
                or not runner.get("runtime") or not result.get("evaluator_code_fingerprint")):
            raise ValueError(f"Runner manifest mismatch: {path}")
        identity = {key: result[key] for key in ("protocol", "environment", "evaluator_code_fingerprint")}
        identity["runtime"] = runner["runtime"]
        identity["runner_sha256"] = manifest["runner_sha256"]
        if common is not None and identity != common:
            raise ValueError("Candidate runtime/protocol/evaluator differs")
        common = identity
        checkpoint = Path(provenance["checkpoint"])
        digest = file_hash(checkpoint)
        if (digest != provenance["checkpoint_sha256"]
                or read(checkpoint.with_suffix(checkpoint.suffix + ".sha256.json")).get("sha256") != digest):
            raise ValueError(f"Candidate checkpoint checksum mismatch: {checkpoint}")
        records.append({"step": step, "result": str(path.resolve()), "result_sha256": file_hash(path),
                        "checkpoint_sha256": digest, "provenance": provenance, "summary": summary})
    return records, pool


def copy_frozen_checkpoint(source: Path, target: Path, digest: str):
    if target.exists():
        if file_hash(target) != digest:
            raise ValueError("Existing best.pt differs from selected checkpoint")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix="best.pt.", dir=target.parent)
        os.close(handle)
        try:
            shutil.copyfile(source, temporary)
            if file_hash(temporary) != digest:
                raise IOError("Selected checkpoint copy checksum mismatch")
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    atomic_json(target.with_suffix(".pt.sha256.json"), {"sha256": digest}, frozen=True)


def create_pool(selection: dict, exclusions: dict) -> dict:
    rng = np.random.default_rng(selection["final_generation_seed"])
    seeds, states = set(exclusions["seeds"]), {tuple(x) for x in exclusions["states"]}
    records = []
    for _ in range(15000):
        if len(records) == 50:
            break
        seed = int(rng.integers(0, 2**31 - 1))
        scene = random_scene(seed)
        try:
            validate_scene(scene)
        except ValueError:
            continue
        key = state_key(scene.to_dict())
        if seed in seeds or key in states:
            continue
        seeds.add(seed)
        states.add(key)
        records.append({"scene_id": f"diffusion-final50-{len(records):03d}",
                        "environment_seed": seed, "state": scene.to_dict()})
    if len(records) != 50:
        raise RuntimeError("Could not generate 50 unique legal final scenes")
    return {"schema": "pusht_scenes_v2", "split": "final", "count": 50,
            "protocol_amendment": AMENDMENT, "protocol": FINAL_PROTOCOL,
            "environment_version": version("gym-pusht"),
            "generation_seed": selection["final_generation_seed"], "scenes": records,
            "scenes_hash": canonical_hash(records), "exclusions": exclusions, "selection": selection}


def run(args):
    torch.set_num_threads(2)
    audit = read(args.audit)
    records, development = candidates(args.evaluation_dir, args.development, audit)
    winner = max(records, key=lambda record: checkpoint_score(record["summary"], record["step"]))
    out = args.output_dir.resolve()
    selection_path, pool_path = out / "selection.json", out / "final50-scenes.json"
    exclusions = collect_exclusions([args.development, *args.exclude])
    existing = read(selection_path) if selection_path.exists() else None
    if existing is None and pool_path.exists():
        raise ValueError("Final pool exists without selection; refusing to replace evidence")
    if existing is not None:
        if existing.get("lock_hash") != canonical_hash({k: v for k, v in existing.items() if k != "lock_hash"}):
            raise ValueError("Existing selection lock hash mismatch")
        seed = existing["final_generation_seed"]
        if args.seed is not None and args.seed != seed:
            raise ValueError("Final generation seed already frozen; refusing redraw")
    else:
        seed = args.seed if args.seed is not None else secrets.randbits(32)
    selected = {**winner["provenance"], "checkpoint": str(out / "best.pt")}
    selection = {"schema": "pusht_selection_v2", "models": {"unet": selected},
                 "sampling": default_config("unet")["sampling"],
                 "development_hash": file_hash(args.development), "development_scenes_hash": development["scenes_hash"],
                 "candidates": records, "score_order": ["success_count_87", "mean_max_coverage", "earlier_step"],
                 "chosen_sha256": winner["checkpoint_sha256"], "chosen_step": winner["step"],
                 "original_final_count": 100, "new_final_count": 50, "override_reason": "user_request",
                 "protocol_amendment": AMENDMENT, "final_generation_seed": seed,
                 "exclusions_hash": canonical_hash(exclusions), "video_example_rule": VIDEO_RULE,
                 "created_utc": existing["created_utc"] if existing else datetime.now(timezone.utc).isoformat()}
    selection["lock_hash"] = canonical_hash(selection)
    if existing is not None and existing != selection:
        raise ValueError("Selection inputs changed; refusing reselection or redraw")
    # 真实加载选中 EMA 权重；校验配置、数据和张量结构，不以 JSON 声明代替模型检查。
    policy = load_diffusion_policy(winner["provenance"]["checkpoint"], config=default_config("unet"),
                                   audit=audit, device="cpu")
    if policy.provenance != winner["provenance"]:
        raise ValueError("Selected checkpoint provenance differs from evaluated policy")
    del policy
    atomic_json(selection_path, selection, frozen=True)
    copy_frozen_checkpoint(Path(winner["provenance"]["checkpoint"]), out / "best.pt", winner["checkpoint_sha256"])
    if pool_path.exists():
        pool = frozen_scenes(pool_path)
        if (pool["selection"] != selection or pool["exclusions"] != exclusions
                or pool.get("generation_seed") != seed):
            raise ValueError("Existing final pool differs from locked inputs")
    else:
        atomic_json(pool_path, create_pool(selection, exclusions), frozen=True)
        pool = frozen_scenes(pool_path)
    return {"selection": str(selection_path), "final_scenes": str(pool_path), "best": str(out / "best.pt"),
            "chosen_step": winner["step"], "chosen_sha256": winner["checkpoint_sha256"],
            "final_scenes_hash": pool["scenes_hash"], "reused": existing is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("evaluation-dir", "development", "audit", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.seed is not None and not 0 <= args.seed < 2**32:
        parser.error("--seed must be an unsigned 32-bit integer")
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
