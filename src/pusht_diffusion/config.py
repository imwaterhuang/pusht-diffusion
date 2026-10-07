"""Frozen project contracts; paths live separately from experiment identity."""

from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
from .utils import canonical_hash

DEFAULT = {
    'schema_version': 1,
    'model': {
        'name': 'unet',
        'architecture': 'pusht_diffusion_unet_v1',
        'history': 2,
        'horizon': 16,
        'action_dim': 2,
        'condition_dim': 516,
        'vision': {
            'backbone': 'resnet18',
            'pretrained': 'IMAGENET1K_V1',
            'spatial': [512, 3, 3],
            'projection': 256,
            'freeze_bn_statistics': True,
        },
        'unet': {
            'down_dims': [128, 256, 512],
            'kernel_size': 5,
            'groups': 8,
            'timestep_dim': 128,
            'decoder': 'upsample_then_same_resolution_concat',
        },
        'dit': {
            'width': 256,
            'layers': 6,
            'heads': 8,
            'ff_dim': 1024,
            'position': 'fixed',
            'conditioning': 'adaLN-Zero',
            'causal': False,
        },
    },
    'data': {
        'episodes': 206,
        'frames': 25650,
        'windows': 25444,
        'image_shape': [3, 96, 96],
        'image_normalization': 'imagenet',
        'action_normalization': 'minmax',
        'position_records': 'all_raw_rows',
        'action_records': 'exclude_terminal_rows',
    },
    'diffusion': {
        'train_timesteps': 100,
        'beta_schedule': 'squaredcos_cap_v2',
        'prediction_type': 'epsilon',
        'clip_sample': False,
        'thresholding': False,
        'timestep_spacing': 'leading',
        'steps_offset': 0,
        'rescale_betas_zero_snr': False,
    },
    'sampling': {'method': 'ddim', 'steps': 20, 'eta': 0.0, 'noise_seed': 0},
    'training': {
        'seed': 0,
        'batch_size': 64,
        'steps': 40000,
        'optimizer': 'AdamW',
        'learning_rate': 1e-4,
        'weight_decay': 1e-6,
        'betas': [0.95, 0.999],
        'warmup_steps': 500,
        'lr_schedule': 'cosine',
        'gradient_clip_norm': 1.0,
        'precision': 'float32',
        'ema_max_decay': 0.999,
        'ema_rule': 'min(0.999,(1+s)/(10+s))',
        'save_every': 1000,
        'evaluate_every': 5000,
    },
    'evaluation': {
        'development_count': 50,
        'final_count': 100,
        'max_steps': 300,
        'execute_steps': 4,
        'thresholds': [0.87, 0.95],
        'selection': ['success_count_87', 'mean_max_coverage', 'earlier_step'],
    },
}


def default_config(model: str = 'unet') -> dict:
    if model not in ('unet', 'dit'):
        raise ValueError('model must be unet or dit')
    c = deepcopy(DEFAULT)
    c['model']['name'] = model
    c['model']['architecture'] = f'pusht_diffusion_{model}_v1'
    return c


def validate_config(config: dict) -> dict:
    """Reject unsupported/misspelled settings instead of silently ignoring them."""
    if not isinstance(config, dict):
        raise ValueError('config must be a mapping')
    model = config.get('model', {}).get('name')
    expected = default_config(model)
    # This v1 is intentionally frozen; budget changes require a documented new configuration version.
    if config != expected:
        raise ValueError(
            'Configuration differs from frozen v1 contract; update schema explicitly before changing the experiment'
        )
    canonical_hash(config)
    return deepcopy(config)


def load_config(path: str | Path) -> dict:
    return validate_config(json.loads(Path(path).read_text()))
