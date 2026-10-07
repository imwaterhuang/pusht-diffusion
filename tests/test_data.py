from copy import deepcopy
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from pusht_diffusion.data import PushTDataset, audit_data


def test_window_alignment_mask_padding_and_raw_stats(tiny_dataset):
    audit = audit_data(tiny_dataset, require_reference_counts=False)
    assert (
        audit['frame_count'],
        audit['valid_window_count'],
        audit['action_stat_count'],
        audit['position_stat_count'],
    ) == (7, 5, 5, 7)
    assert audit['normalization']['action_max'] == [25.0, 60.0]
    assert audit['normalization']['position_max'] == [60.0, 30.0]
    d = PushTDataset(tiny_dataset, audit=audit, require_reference_counts=False)
    first = d[0]
    tail = d[2]
    second_start = d[3]
    assert first['images'].shape == (2, 3, 96, 96) and first['positions'].shape == (2, 2)
    assert torch.equal(first['images'][0], first['images'][1])
    assert first['valid_mask'].tolist() == [True] * 3 + [False] * 13
    expected = d.normalization.normalize_action(torch.tensor([[20.0, 50.0], [21.0, 52.0], [22.0, 54.0]]))
    torch.testing.assert_close(first['actions'][:3], expected)
    torch.testing.assert_close(first['actions'][3:], expected[-1].expand(13, -1))
    assert tail['valid_mask'].sum() == 1
    assert second_start['episode_id'] == 1 and second_start['start_step'] == 0
    assert torch.equal(second_start['images'][0], second_start['images'][1])
    assert len(d) == 5
    decoded = d.images.decode_count
    d[0]
    assert d.images.decode_count == decoded


def test_supplied_audit_cannot_bypass_source_validation(tiny_dataset):
    audit = audit_data(tiny_dataset, require_reference_counts=False)
    stale = deepcopy(audit)
    stale['normalization']['action_min'][0] -= 1
    with pytest.raises(ValueError, match='Supplied audit'):
        PushTDataset(tiny_dataset, audit=stale, require_reference_counts=False)
    file = tiny_dataset / 'data/file.parquet'
    table = pq.read_table(file)
    rows = table.to_pylist()
    rows[0]['frame_index'] = 99
    pq.write_table(pa.Table.from_pylist(rows), file)
    with pytest.raises(ValueError, match='frame indices'):
        PushTDataset(tiny_dataset, audit=audit, require_reference_counts=False)


def test_minmax_roundtrip_unclipped(tiny_dataset):
    d = PushTDataset(tiny_dataset, require_reference_counts=False)
    raw = torch.tensor([[0.0, 100.0], [22.0, 55.0]])
    normalized = d.normalization.normalize_action(raw)
    assert normalized.abs().max() > 1
    torch.testing.assert_close(d.normalization.denormalize_action(normalized), raw)


def test_external_normalized_cache_validated_against_source(tiny_dataset, tmp_path):
    import json
    from pusht_diffusion.data import LazyParquetImages, ValidatedImageCache
    from pusht_diffusion._scalar_audit import _sha256_files
    from pusht_diffusion.normalization import IMAGENET_MEAN, IMAGENET_STD
    from pusht_diffusion.utils import file_hash

    cache = tmp_path / 'cache'
    cache.mkdir()
    reader = LazyParquetImages(tiny_dataset)
    frames = np.stack([reader[i].numpy() for i in range(7)])
    np.save(cache / 'frames.npy', frames)
    manifest = {
        'identity': {
            'dataset_sha256': _sha256_files(tiny_dataset, sorted((tiny_dataset / 'data').rglob('*.parquet'))),
            'mean': IMAGENET_MEAN.flatten().tolist(),
            'std': IMAGENET_STD.flatten().tolist(),
        },
        'shape': list(frames.shape),
        'file_bytes': (cache / 'frames.npy').stat().st_size,
        'frames_sha256': file_hash(cache / 'frames.npy'),
    }
    (cache / 'manifest.json').write_text(json.dumps(manifest))
    cached = PushTDataset(tiny_dataset, image_cache=cache, require_reference_counts=False)
    torch.testing.assert_close(cached[1]['images'], torch.stack([reader[0], reader[1]]))
    # Even a checksummed cache with wrong preprocessing must fail pixel probes.
    frames[0] += 1
    np.save(cache / 'frames.npy', frames)
    manifest['frames_sha256'] = file_hash(cache / 'frames.npy')
    (cache / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='preprocessing'):
        ValidatedImageCache(cache, tiny_dataset, 7)
