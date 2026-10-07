"""All-episode windows, scalar-only audit, and bounded lazy image decoding."""

from __future__ import annotations
from collections import OrderedDict
import io
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import torch
from torch.utils.data import Dataset
from ._scalar_audit import build_act_audit, _sha256_files
from .normalization import MinMaxNormalization, image_tensor, IMAGENET_MEAN, IMAGENET_STD
from .utils import file_hash, atomic_json


def scalar_records(root: str | Path):
    root = Path(root)

    # 按原始 index 排列位置和动作，保证二者对应同一帧。
    paths = sorted((root / 'data').rglob('*.parquet'))
    table = pa.concat_tables([pq.read_table(p, columns=['index', 'observation.state', 'action']) for p in paths])
    order = np.argsort(table['index'].to_numpy())
    positions = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)[order]
    actions = np.asarray(table['action'].to_pylist(), dtype=np.float32)[order]

    # episode 的起止位置用于构造窗口，窗口不能跨 episode。
    eps = pa.concat_tables(
        [
            pq.read_table(p, columns=['episode_index', 'dataset_from_index', 'dataset_to_index'])
            for p in sorted((root / 'meta/episodes').rglob('*.parquet'))
        ]
    )
    episodes = sorted(zip(*[eps[k].to_pylist() for k in eps.column_names]))
    return positions, actions, episodes


def audit_data(root: str | Path, *, require_reference_counts: bool = True) -> dict:
    manifest, legacy = build_act_audit(root)  # no images decoded, source identity matches the ACT checkpoint
    positions, actions, episodes = scalar_records(root)

    # 位置统计包含末帧；动作统计排除没有有效动作标签的末帧。
    valid = np.concatenate([np.arange(start, end - 1) for _, start, end in episodes])
    norm = MinMaxNormalization(
        positions.min(0).tolist(),
        positions.max(0).tolist(),
        actions[valid].min(0).tolist(),
        actions[valid].max(0).tolist(),
    )
    if require_reference_counts and (
        manifest['episode_count'],
        manifest['frame_count'],
        manifest['valid_window_count'],
    ) != (206, 25650, 25444):
        raise ValueError('Reference dataset count mismatch')
    return {
        **manifest,
        'normalization': norm.to_dict(),
        'position_stat_count': len(positions),
        'action_stat_count': len(valid),
        'position_rule': 'all raw positions including terminal observations',
        'action_rule': 'raw nonterminal labels only',
        'image_decodes': 0,
        'legacy_act_statistics': legacy['training_normalization'],
    }


class LazyParquetImages:
    """Compressed image column read once per worker; at most 256 decoded frames retained."""

    def __init__(self, root: Path, max_decoded: int = 256):
        self.root = root
        self.max_decoded = max_decoded
        self._table = None
        self._cache = OrderedDict()
        self.decode_count = 0

    def __getstate__(self):
        return {**self.__dict__, '_table': None, '_cache': OrderedDict()}

    def __getitem__(self, index: int) -> torch.Tensor:
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        if self._table is None:
            table = pa.concat_tables(
                [
                    pq.read_table(p, columns=['index', 'observation.image'])
                    for p in sorted((self.root / 'data').rglob('*.parquet'))
                ]
            )
            self._table = table.take(pa.array(np.argsort(table['index'].to_numpy())))
        value = self._table['observation.image'][index].as_py()
        if value.get('bytes') is not None:
            with Image.open(io.BytesIO(value['bytes'])) as im:
                pixels = np.asarray(im.convert('RGB')).copy()
        else:
            path = (self.root / value['path']).resolve()
            if not path.is_relative_to(self.root.resolve()):
                raise ValueError('Image path escapes dataset root')
            with Image.open(path) as im:
                pixels = np.asarray(im.convert('RGB')).copy()
        tensor = image_tensor(pixels)
        self.decode_count += 1
        self._cache[index] = tensor
        while len(self._cache) > self.max_decoded:
            self._cache.popitem(last=False)
        return tensor


class ValidatedImageCache:
    """Read-only normalized legacy cache, validated by source/file checksums and pixel probes."""

    def __init__(self, path: str | Path, root: Path, count: int):
        self.path = Path(path)
        manifest = json.loads((self.path / 'manifest.json').read_text())
        identity = manifest['identity']
        expected = _sha256_files(root, sorted((root / 'data').rglob('*.parquet')))
        if identity['dataset_sha256'] != expected:
            raise ValueError('Cache dataset fingerprint mismatch')
        if not np.allclose(identity['mean'], IMAGENET_MEAN.flatten().tolist(), rtol=0, atol=1e-8) or not np.allclose(
            identity['std'], IMAGENET_STD.flatten().tolist(), rtol=0, atol=1e-8
        ):
            raise ValueError('Cache normalization mismatch')
        self.shape = (count, 3, 96, 96)
        if tuple(manifest['shape']) != self.shape:
            raise ValueError('Cache shape mismatch')
        frame = self.path / 'frames.npy'
        if frame.stat().st_size != manifest['file_bytes'] or file_hash(frame) != manifest['frames_sha256']:
            raise ValueError('Cache bytes/checksum mismatch')
        self._array = None
        self._open()
        reader = LazyParquetImages(root)
        for index in sorted({0, count // 2, count - 1}):
            if not np.array_equal(self._array[index], reader[index].numpy()):
                raise ValueError('Cache preprocessing differs from decoded source')

    def _open(self):
        if self._array is None:
            self._array = np.load(self.path / 'frames.npy', mmap_mode='r')
            if self._array.shape != self.shape or self._array.dtype != np.float32:
                raise ValueError('Cache dtype/shape mismatch')

    def __getstate__(self):
        return {**self.__dict__, '_array': None}

    def __getitem__(self, index):
        self._open()
        return torch.from_numpy(self._array[index].copy())


class PushTDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        *,
        image_cache: str | Path | None = None,
        audit: dict | None = None,
        require_reference_counts: bool = True,
    ):
        self.root = Path(root).expanduser().resolve()
        current = audit_data(self.root, require_reference_counts=require_reference_counts)
        if audit is not None and audit != current:
            raise ValueError('Supplied audit differs from current dataset identity/statistics')
        self.audit = current
        self.normalization = MinMaxNormalization(**self.audit['normalization'])
        self.positions, self.actions, self.episodes = scalar_records(self.root)

        # 每个窗口保存 episode、局部步数、全局帧号和 episode 边界。
        self.windows = [
            (int(e), int(t - start), int(t), int(start), int(end))
            for e, start, end in self.episodes
            for t in range(start, end - 1)
        ]
        self.images = (
            ValidatedImageCache(image_cache, self.root, len(self.positions))
            if image_cache
            else LazyParquetImages(self.root)
        )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        # 取相邻两帧；episode 首帧重复自身作为上一帧。
        episode, local, t, start, end = self.windows[index]
        previous = max(start, t - 1)
        stop = min(t + 16, end - 1)
        count = stop - t

        images = torch.stack([self.images[previous], self.images[t]])  # [2, 3, 96, 96]
        positions = self.normalization.normalize_position(
            torch.from_numpy(self.positions[[previous, t]].copy())
        )  # [2, 2]

        # 不足 16 步时重复最后一个有效动作，valid_mask 标记真实动作。
        raw = np.repeat(self.actions[stop - 1 : stop], 16, axis=0)
        raw[:count] = self.actions[t:stop]
        actions = self.normalization.normalize_action(torch.from_numpy(raw))  # [16, 2]

        return {
            'images': images,
            'positions': positions,
            'actions': actions,
            'valid_mask': torch.arange(16) < count,  # [16]
            'episode_id': torch.tensor(episode),
            'start_step': torch.tensor(local),
        }
