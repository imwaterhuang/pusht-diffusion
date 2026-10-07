"""Resume formal evaluation one frozen scene at a time; no model/training edits.

Example (run from repository root with PYTHONPATH=src):
  python scripts/colab_evaluate.py --checkpoint runs/step-040000.pt \
      --audit runs/audit.json --scenes scenes/development.json \
      --output-dir runs/evaluation/step-040000 --device cuda \
      --mirror-dir /content/drive/MyDrive/pusht/evaluation/step-040000

After completion, use the existing trace replay entry point for videos:
  python -m pusht_diffusion video --results runs/evaluation/step-040000/result.json \
      --scene diffusion-development-000 --output runs/example.mp4
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import tempfile
import time

import torch

from pusht_diffusion.config import default_config
from pusht_diffusion.environment import SceneState, validate_scene
from pusht_diffusion.evaluation import evaluate, recompute
from pusht_diffusion.policies import load_diffusion_policy
from pusht_diffusion.scenes import load_scenes, state_key
from pusht_diffusion.utils import atomic_json, canonical_hash, code_fingerprint, file_hash


AMENDMENT = "user_requested_final_50_v1"
PROTOCOL = {
    "max_steps": 300, "execute_steps": 4, "thresholds": [0.87, 0.95],
    "comparison": ">", "stop": "coverage > .95 or environment done or max_steps",
    "history_update": "every_environment_step",
}


def runtime_metadata(device: str) -> dict:
    """Resume only with the same physics, rendering, numerical and device runtime."""
    packages = {name: version(name) for name in (
        "gym-pusht", "gymnasium", "pymunk", "shapely", "pygame", "numpy", "Pillow",
    )}
    target = torch.device(device)
    return {
        "packages": packages, "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "device": str(target),
        "gpu": torch.cuda.get_device_name(target) if target.type == "cuda" else None,
    }


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def frozen_scenes(path: Path) -> dict:
    """Keep the frozen package intact; document the user's explicit 50-final amendment."""
    pool = read(path)
    if pool.get("protocol_amendment") != AMENDMENT:
        return load_scenes(path)
    if (pool.get("schema") != "pusht_scenes_v2" or pool.get("split") != "final"
            or pool.get("count") != 50 or len(pool.get("scenes", [])) != 50
            or pool.get("scenes_hash") != canonical_hash(pool["scenes"])
            or pool.get("environment_version") != version("gym-pusht")):
        raise ValueError("Invalid amended final-50 pool")
    expected_protocol = {
        "max_steps": 300, "execute_steps": 4, "thresholds": [0.87, 0.95],
        "comparison": ">", "sampling": default_config("unet")["sampling"],
    }
    if pool.get("protocol") != expected_protocol:
        raise ValueError("Final-50 v2 protocol differs from the approved evaluation protocol")
    selection = pool.get("selection") or {}
    if (not selection.get("models", {}).get("unet")
            or selection.get("lock_hash") != canonical_hash(
                {k: v for k, v in selection.items() if k != "lock_hash"})):
        raise ValueError("Final-50 pool requires a hashed selected-checkpoint lock")
    ids, seeds, states = set(), set(), set()
    excluded = pool.get("exclusions", {})
    excluded_seeds = set(excluded.get("seeds", []))
    excluded_states = {tuple(x) for x in excluded.get("states", [])}
    if not excluded.get("sources"):
        raise ValueError("Final-50 pool must identify development/debug exclusion sources")
    for record in pool["scenes"]:
        identifier, seed = record.get("scene_id"), record.get("environment_seed")
        if not isinstance(identifier, str) or type(seed) is not int or seed < 0:
            raise ValueError("Invalid final scene identifier/seed")
        validate_scene(SceneState(**record["state"]))
        state = state_key(record["state"])
        if (identifier in ids or seed in seeds or state in states
                or seed in excluded_seeds or state in excluded_states):
            raise ValueError("Duplicate or excluded final scene")
        ids.add(identifier)
        seeds.add(seed)
        states.add(state)
    return pool


def persist(path: Path, value: dict, mirror: Path | None, *, frozen: bool = True) -> None:
    atomic_json(path, value, frozen=frozen)
    if mirror is not None:
        atomic_json(mirror, value, frozen=frozen)


def validate_episode(envelope: dict, *, record: dict, pool: dict, checkpoint_hash: str,
                     evaluator_hash: str, runtime: dict) -> dict:
    if envelope.get("runtime") != runtime:
        raise ValueError("Per-scene runtime differs from current environment/device")
    result = envelope["result"]
    if envelope.get("result_sha256") != canonical_hash(result):
        raise ValueError("Per-scene result checksum mismatch")
    if (result.get("schema") != "pusht_results_v1" or result.get("evidence_kind") != "formal"
            or result.get("policy") != "unet" or result.get("protocol") != PROTOCOL
            or result.get("scenes_hash") != pool["scenes_hash"]
            or result.get("scene_split") != pool["split"]
            or result.get("provenance", {}).get("checkpoint_sha256") != checkpoint_hash
            or result.get("evaluator_code_fingerprint") != evaluator_hash
            or result.get("environment") != {
                "gym_pusht": version("gym-pusht"), "gymnasium": version("gymnasium")}
            or len(result.get("episodes", [])) != 1):
        raise ValueError("Per-scene evidence does not match checkpoint/pool/protocol/runtime")
    episode = result["episodes"][0]
    if any(episode.get(key) != value for key, value in record.items()):
        raise ValueError("Per-scene reset state does not match frozen pool")
    if result.get("summary") != recompute(result):
        raise ValueError("Per-scene summary does not match raw trace")
    return result


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.cpu_threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing silent CPU fallback")
    pool = frozen_scenes(args.scenes)
    if pool["split"] not in ("development", "final"):
        raise ValueError("Formal runner requires a frozen development or final pool")
    config = default_config("unet")
    audit = read(args.audit)
    checkpoint_hash = file_hash(args.checkpoint)
    evaluator_hash = code_fingerprint()
    runtime = runtime_metadata(args.device)
    if pool["split"] == "final":
        selected = (pool.get("selection") or {}).get("models", {}).get("unet", {})
        if selected.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError("Checkpoint differs from frozen final selection")

    root, mirror_root = args.output_dir, args.mirror_dir
    if mirror_root is not None and root.resolve() == mirror_root.resolve():
        raise ValueError("Local output and mirror directories must differ")

    # 身份清单不含绝对路径，因此 Colab 断线后可在新机器从 Drive 恢复。
    manifest = {
        "schema": "pusht_resumable_evaluation_v1", "checkpoint_sha256": checkpoint_hash,
        "scenes_file_sha256": file_hash(args.scenes), "scenes_hash": pool["scenes_hash"],
        "audit_sha256": canonical_hash(audit), "evaluator_code_fingerprint": evaluator_hash,
        "runner_sha256": file_hash(__file__), "config_sha256": canonical_hash(config),
        "scene_count": pool["count"], "protocol": PROTOCOL,
        "protocol_amendment": pool.get("protocol_amendment"),
        "runtime": runtime,
    }
    manifest_path = root / "manifest.json"
    mirror_manifest = mirror_root / "manifest.json" if mirror_root else None
    for candidate in (manifest_path, mirror_manifest):
        if candidate is not None and candidate.exists() and read(candidate) != manifest:
            raise ValueError(f"Different evaluation already occupies {candidate}")
    persist(manifest_path, manifest, mirror_manifest)

    # 仅加载一次 EMA 模型；每场景单独调用现有 evaluate，不改变停止/采样规则。
    policy = load_diffusion_policy(args.checkpoint, config=config, audit=audit, device=args.device)
    episodes = []
    result_template = None
    fresh_seconds = []
    for index, record in enumerate(pool["scenes"]):
        path = root / "episodes" / f"{index:04d}.json"
        mirror = mirror_root / "episodes" / path.name if mirror_root else None
        source = path if path.exists() else mirror if mirror is not None and mirror.exists() else None
        tick = time.monotonic()
        if source is not None:
            envelope = read(source)
            result = validate_episode(envelope, record=record, pool=pool,
                                      checkpoint_hash=checkpoint_hash, evaluator_hash=evaluator_hash, runtime=runtime)
            resumed = True
        else:
            # 临时文件只用于 evaluate 的原子输出；完成后才提交可续跑的场景证据。
            with tempfile.TemporaryDirectory(prefix="pusht-eval-") as scratch:
                result = evaluate(policy, {**pool, "count": 1, "scenes": [record]},
                                  Path(scratch) / "result.json", max_steps=300, evidence_kind="formal")
            envelope = {"result": result, "result_sha256": canonical_hash(result), "runtime": runtime}
            validate_episode(envelope, record=record, pool=pool,
                             checkpoint_hash=checkpoint_hash, evaluator_hash=evaluator_hash, runtime=runtime)
            fresh_seconds.append(time.monotonic() - tick)
            resumed = False
        # 每场景先写本地证据，再镜像；镜像失败即报错，重跑会补传同一场景。
        persist(path, envelope, mirror)
        result_template = result_template or result
        episodes.extend(result["episodes"])
        aggregate = {**result_template, "episodes": list(episodes)}
        aggregate["summary"] = recompute(aggregate)
        aggregate["runner"] = {
            "manifest_sha256": canonical_hash(manifest), "expected_scene_count": pool["count"],
            "complete": len(episodes) == pool["count"],
            "protocol_amendment": pool.get("protocol_amendment"),
            "runtime": runtime,
        }
        persist(root / "partial.json", aggregate,
                mirror_root / "partial.json" if mirror_root else None, frozen=False)
        metrics = aggregate["summary"]
        remaining = pool["count"] - len(episodes)
        eta = round(sum(fresh_seconds) / len(fresh_seconds) * remaining) if fresh_seconds else None
        print(json.dumps({
            "completed": len(episodes), "total": pool["count"], "scene_id": record["scene_id"],
            "resumed": resumed, "elapsed_scene_seconds": round(time.monotonic() - tick, 2),
            "estimated_remaining_seconds": eta,
            "success_87": metrics["success_count_87"], "success_95": metrics["success_count_95"],
            "mean_max_coverage": round(metrics["mean_max_coverage"], 6),
        }), flush=True)

    # 只有全部冻结场景都完成，才发布 result.json；指标始终从原始轨迹重算。
    persist(root / "result.json", aggregate, mirror_root / "result.json" if mirror_root else None)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("checkpoint", "audit", "scenes", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mirror-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    result = run(args)
    print(json.dumps({"complete": True, "summary": result["summary"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
