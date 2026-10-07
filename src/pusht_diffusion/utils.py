"""Pure provenance helpers and atomic local persistence; no training algorithm."""

from __future__ import annotations
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def atomic_json(path: str | Path, payload: Any, *, frozen: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if frozen and path.exists():
        if json.loads(path.read_text()) != payload:
            raise FileExistsError(f'Refusing to overwrite frozen evidence: {path}')
        return
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def append_event(path: str | Path, event: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()


def code_fingerprint() -> str:
    root = Path(__file__).parent
    return canonical_hash({str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob('*.py'))})
