"""Audit every expert episode and freeze ACT training data identity/statistics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _sha256_files(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _stats(values: np.ndarray) -> dict[str, Any]:
    if not len(values) or values.shape[1:] != (2,) or not np.isfinite(values).all():
        raise ValueError("Position/action statistics need finite two-dimensional records")
    return {
        "count": int(len(values)),
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "mean": values.mean(axis=0, dtype=np.float64).tolist(),
        "std": values.std(axis=0, dtype=np.float64).tolist(),
    }


def build_act_audit(dataset_root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return all-episode manifest and audit; each valid source row counts once."""
    root = Path(dataset_root).expanduser().resolve()
    info_path = root / "meta/info.json"
    episode_paths = sorted((root / "meta/episodes").glob("**/*.parquet"))
    data_paths = sorted((root / "data").glob("**/*.parquet"))
    if not info_path.is_file() or not episode_paths or not data_paths:
        raise FileNotFoundError("Dataset needs meta/info.json, episode and data parquet files")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if (
        info["features"]["observation.image"]["shape"] != [96, 96, 3]
        or info["features"]["observation.state"]["shape"] != [2]
        or info["features"]["action"]["shape"] != [2]
    ):
        raise ValueError("ACT requires 96x96 RGB images and two-dimensional states/actions")
    episodes = pa.concat_tables(
        [
            pq.read_table(path, columns=["episode_index", "dataset_from_index", "dataset_to_index"])
            for path in episode_paths
        ]
    )
    ids = [int(value) for value in episodes["episode_index"]]
    starts = [int(value) for value in episodes["dataset_from_index"]]
    ends = [int(value) for value in episodes["dataset_to_index"]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Episode IDs must be nonempty and unique")
    records = sorted(zip(ids, starts, ends, strict=True))
    if any(end - start < 2 for _, start, end in records):
        raise ValueError("Every episode needs at least one valid transition")

    table = pa.concat_tables(
        [
            pq.read_table(path, columns=["index", "episode_index", "frame_index", "observation.state", "action"])
            for path in data_paths
        ]
    )
    order = np.argsort(table["index"].to_numpy())
    indices = np.asarray(table["index"].to_numpy(), dtype=np.int64)[order]
    episode_column = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)[order]
    frames = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)[order]
    if not np.array_equal(indices, np.arange(len(indices), dtype=np.int64)):
        raise ValueError("Global row indices must be contiguous from zero")
    mask = np.zeros(len(indices), dtype=bool)
    covered = np.zeros(len(indices), dtype=bool)
    for episode_id, start, end in records:
        if start < 0 or end > len(indices):
            raise ValueError(f"Episode {episode_id} range lies outside data")
        if not np.all(episode_column[start:end] == episode_id):
            raise ValueError(f"Episode {episode_id} range does not match rows")
        if not np.array_equal(frames[start:end], np.arange(end - start)):
            raise ValueError(f"Episode {episode_id} frame indices are not local and contiguous")
        if covered[start:end].any():
            raise ValueError("Episode ranges overlap")
        covered[start:end] = True
        mask[start : end - 1] = True
    if not covered.all():
        raise ValueError("Episode ranges do not cover every data row")

    positions = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)[order]
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)[order]
    position_stats = _stats(positions[mask])
    action_stats = _stats(actions[mask])
    fingerprint = _sha256_files(root, [info_path, *episode_paths, *data_paths])
    manifest = {
        "schema_version": 1,
        "repo_id": "lerobot/pusht_image",
        "dataset_fingerprint": fingerprint,
        "episode_count": len(records),
        "frame_count": len(indices),
        "valid_window_count": int(mask.sum()),
        "train": [episode_id for episode_id, _, _ in records],
    }
    audit = {
        "schema_version": 1,
        "repo_id": manifest["repo_id"],
        "dataset_fingerprint": fingerprint,
        "episode_count": len(records),
        "frame_count": len(indices),
        "valid_window_count": int(mask.sum()),
        "validity_rule": "local step 0 through episode length - 2; terminal row excluded",
        "training_normalization": {
            "agent_position": position_stats,
            "action": action_stats,
            "std_floor": 1e-6,
        },
    }
    return manifest, audit


def save_act_audit(
    dataset_root: str | Path,
    manifest_path: str | Path,
    audit_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write new files or verify identical existing files; never replace evidence."""
    manifest, audit = build_act_audit(dataset_root)
    targets = (
        (Path(manifest_path).expanduser().resolve(), manifest),
        (Path(audit_path).expanduser().resolve(), audit),
    )
    for path, payload in targets:
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != payload:
                raise FileExistsError(f"Refusing to overwrite different ACT audit data: {path}")
    for path, payload in targets:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest, audit
