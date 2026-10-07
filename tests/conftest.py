from pathlib import Path
import io, json
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch


@pytest.fixture(autouse=True, scope='session')
def thread_budget():
    torch.set_num_threads(2)


@pytest.fixture
def tiny_dataset(tmp_path):
    root = tmp_path / 'data'
    (root / 'meta/episodes').mkdir(parents=True)
    (root / 'data').mkdir()
    info = {
        'features': {
            'observation.image': {'shape': [96, 96, 3]},
            'observation.state': {'shape': [2]},
            'action': {'shape': [2]},
        }
    }
    (root / 'meta/info.json').write_text(json.dumps(info))
    pq.write_table(
        pa.table({'episode_index': [0, 1], 'dataset_from_index': [0, 4], 'dataset_to_index': [4, 7]}),
        root / 'meta/episodes/file.parquet',
    )
    images = []
    for i in range(7):
        b = io.BytesIO()
        Image.fromarray(np.full((96, 96, 3), i * 25, dtype=np.uint8)).save(b, format='PNG')
        images.append({'bytes': b.getvalue(), 'path': None})
    # Deliberately non-monotonic physical row storage; index field is the source of truth.
    order = [4, 0, 6, 3, 1, 5, 2]
    cols = {
        'index': list(range(7)),
        'episode_index': [0, 0, 0, 0, 1, 1, 1],
        'frame_index': [0, 1, 2, 3, 0, 1, 2],
        'observation.state': [[10.0 * i, 5.0 * i] for i in range(7)],
        'action': [[20.0 + i, 50.0 + 2 * i] if i not in (3, 6) else [9999.0, 9999.0] for i in range(7)],
        'observation.image': images,
    }
    pq.write_table(pa.table({k: [v[i] for i in order] for k, v in cols.items()}), root / 'data/file.parquet')
    return root
