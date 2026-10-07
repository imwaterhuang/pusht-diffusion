"""Safe, complete state containers. No model/optimizer/EMA update code lives here."""

from __future__ import annotations
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import torch
from .config import validate_config
from .normalization import MinMaxNormalization
from .utils import canonical_hash, file_hash, atomic_json

SCHEMA = 'pusht_diffusion_checkpoint_v1'


def make_checkpoint(
    *,
    config: dict,
    data_fingerprint: str,
    normalization: dict,
    step: int,
    raw: dict,
    ema: dict,
    optimizer: dict,
    lr: dict,
    sampler: dict,
    rng: dict,
    code_fingerprint: str,
    extra: dict | None = None,
) -> dict:
    """Caller supplies already-updated states; generic serialization does not train."""
    c = validate_config(config)
    result = {
        'schema': SCHEMA,
        'model_type': c['model']['name'],
        'architecture': c['model'],
        'config': c,
        'config_fingerprint': canonical_hash(c),
        'data_fingerprint': data_fingerprint,
        'normalization': normalization,
        'step': step,
        'raw': raw,
        'ema': ema,
        'optimizer': optimizer,
        'lr': lr,
        'sampler': sampler,
        'rng': rng,
        'code_fingerprint': code_fingerprint,
        'extra': extra or {},
    }
    validate_checkpoint(result)
    return result


def validate_checkpoint(
    p: dict,
    *,
    expected_config: dict | None = None,
    expected_data: str | None = None,
    expected_normalization: dict | None = None,
) -> None:
    required = {
        'schema',
        'model_type',
        'architecture',
        'config',
        'config_fingerprint',
        'data_fingerprint',
        'normalization',
        'step',
        'raw',
        'ema',
        'optimizer',
        'lr',
        'sampler',
        'rng',
        'code_fingerprint',
    }
    if not isinstance(p, dict) or not required <= p.keys() or p['schema'] != SCHEMA:
        raise ValueError('Incomplete/incompatible checkpoint schema')
    c = validate_config(p['config'])
    if (
        p['model_type'] != c['model']['name']
        or p['architecture'] != c['model']
        or p['config_fingerprint'] != canonical_hash(c)
    ):
        raise ValueError('Checkpoint model/architecture/config fingerprint mismatch')
    for field in ('data_fingerprint', 'code_fingerprint'):
        if not isinstance(p[field], str) or len(p[field]) != 64 or any(x not in '0123456789abcdef' for x in p[field]):
            raise ValueError(f'Invalid {field}')
    MinMaxNormalization(**p['normalization'])
    if expected_config is not None and p['config_fingerprint'] != canonical_hash(validate_config(expected_config)):
        raise ValueError('Unexpected experiment configuration')
    if expected_data is not None and p['data_fingerprint'] != expected_data:
        raise ValueError('Unexpected data fingerprint')
    if expected_normalization is not None and p['normalization'] != expected_normalization:
        raise ValueError('Unexpected normalization')
    if type(p['step']) is not int or p['step'] < 0:
        raise ValueError('Invalid completed update count')
    for name in ('raw', 'ema'):
        if (
            not isinstance(p[name], dict)
            or not p[name]
            or any(
                not isinstance(k, str) or not isinstance(v, torch.Tensor) or not torch.isfinite(v).all()
                for k, v in p[name].items()
            )
        ):
            raise ValueError(f'Invalid {name} weights')
    if p['raw'].keys() != p['ema'].keys() or any(p['raw'][k].shape != p['ema'][k].shape for k in p['raw']):
        raise ValueError('Raw/EMA state shape mismatch')
    for name in ('optimizer', 'lr', 'sampler', 'rng'):
        if not isinstance(p[name], dict) or not p[name]:
            raise ValueError(f'Missing {name} resume state')
    if not {'python', 'numpy', 'torch'} <= p['rng'].keys():
        raise ValueError('Incomplete global random state')
    if not {'dataset_size', 'batch_size', 'seed', 'epoch', 'batch_in_epoch'} <= p['sampler'].keys():
        raise ValueError('Incomplete sampler position')


def save_checkpoint(path: str | Path, payload: dict, *, mirror: str | Path | None = None) -> dict:
    # 先校验完整状态，再写临时文件并原子替换。
    validate_checkpoint(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    digest = file_hash(path)
    atomic_json(path.with_suffix(path.suffix + '.sha256.json'), {'sha256': digest})

    # 可选镜像也先校验文件内容，再替换目标文件。
    if mirror is not None:
        destination = Path(mirror) / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=destination.name + '.', dir=destination.parent)
        os.close(fd)
        try:
            shutil.copyfile(path, tmp)
            if file_hash(tmp) != digest:
                raise IOError('Checkpoint mirror checksum mismatch')
            os.replace(tmp, destination)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        atomic_json(destination.with_suffix(destination.suffix + '.sha256.json'), {'sha256': digest})

    return {'path': str(path.resolve()), 'sha256': digest, 'step': payload['step']}


def load_checkpoint(path: str | Path, **expected) -> dict:
    import json

    path = Path(path)
    sidecar = path.with_suffix(path.suffix + '.sha256.json')
    if not sidecar.exists() or json.loads(sidecar.read_text()).get('sha256') != file_hash(path):
        raise ValueError('Missing or invalid checkpoint checksum sidecar')
    p = torch.load(path, map_location='cpu', weights_only=True)
    validate_checkpoint(p, **expected)
    return p
