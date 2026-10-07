"""Colab L4 正式训练入口：校验 → 保存运行记录 → 调用你写的 train()。

从项目根目录运行；首次训练不传 --resume，断线恢复时明确传 Drive 的 last.pt。
此文件负责运行工程，噪声预测损失、反向传播和 EMA 仍在 learner/training.py。
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time

import torch

from pusht_diffusion.config import load_config
from pusht_diffusion.data import audit_data
from pusht_diffusion.learner.training import train
from pusht_diffusion.utils import atomic_json, canonical_hash, code_fingerprint


class TrainingLog:
    """屏幕仍是一行进度；文件保留每一步，便于之后画 loss 曲线。"""
    def __init__(self, screen, path):
        self.screen = screen
        self.file = path.open('a', encoding='utf-8', buffering=1)

    def write(self, text):
        self.screen.write(text)
        self.file.write(text.replace('\r', '\n'))
        return len(text)

    def flush(self):
        self.screen.flush()
        self.file.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mirror', required=True)
    parser.add_argument('--preflight', required=True)
    parser.add_argument('--resume')
    args = parser.parse_args()
    # 与 GPU 恢复检查采用相同的数值设置，便于比较恢复前后的结果。
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = load_config('configs/unet.json')
    output = Path(args.output).resolve()
    mirror = Path(args.mirror).resolve()
    output.mkdir(parents=True, exist_ok=True)
    mirror.mkdir(parents=True, exist_ok=True)

    # 重复点击同一训练单元格时，不允许启动第二个同目录训练进程。
    lock = (output / 'training.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not torch.cuda.is_available() or 'L4' not in torch.cuda.get_device_name(0):
        raise RuntimeError('本次训练要求 NVIDIA L4，请检查 Colab 运行时类型')
    if not Path('/content/drive/MyDrive').is_dir():
        raise RuntimeError('Google Drive 未挂载；先运行 notebook 的挂载单元格')
    if not mirror.is_relative_to(Path('/content/drive/MyDrive')):
        raise ValueError('--mirror 必须位于已挂载的 MyDrive 中')
    if args.resume is None and ((output / 'last.pt').exists() or (mirror / 'last.pt').exists()):
        raise FileExistsError('已有检查点。要继续训练，请明确使用 --resume，避免从零覆盖。')
    if args.resume is not None and not Path(args.resume).is_file():
        raise FileNotFoundError(args.resume)

    # 预检查必须对应本次代码、配置和数据；CPU smoke 不能替代 L4 预检查。
    audit = audit_data(args.dataset)
    expected = json.loads(Path('reports/data-audit.json').read_text())
    assert audit['dataset_fingerprint'] == expected['dataset_fingerprint']
    gate = json.loads(Path(args.preflight).read_text())
    assert gate['passed'], '预检查未通过'
    assert gate['code_fingerprint'] == code_fingerprint(), '预检查后源代码发生变化'
    assert gate['config_fingerprint'] == canonical_hash(config)
    assert gate['data_fingerprint'] == audit['dataset_fingerprint']
    if shutil.disk_usage(output).free < 6 * 1024**3:
        raise RuntimeError('本地存储不足 6 GiB，不能保证保存完整训练快照')

    # 数据读 Colab 本地磁盘；检查点每 1,000 步同步 Drive，每 5,000 步保留快照。
    paths = {'dataset_root': str(Path(args.dataset).resolve()),
             'output_dir': str(output), 'checkpoint_mirror': str(mirror)}
    atomic_json(output / 'paths.json', paths)
    atomic_json(output / 'config.json', config)
    atomic_json(output / 'data-audit.json', audit)
    environment = {
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'python': sys.version,
        'versions': {name: importlib.metadata.version(name) for name in
                     ('torchvision', 'diffusers', 'numpy', 'pyarrow', 'Pillow')},
        'code_fingerprint': code_fingerprint(),
        'config_fingerprint': canonical_hash(config),
        'data_fingerprint': audit['dataset_fingerprint'],
        'resume': args.resume, 'planned_total_updates': config['training']['steps'],
        'deterministic_algorithms': True, 'tf32': False,
        'num_workers': 8,
    }
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    atomic_json(output / f'environment-{stamp}.json', environment)
    for path in output.glob('*.json'):
        shutil.copy2(path, mirror / path.name)
    shutil.copy2(args.preflight, mirror / 'preflight.json')

    # 正式训练从 seed=0 重新初始化，不接着 smoke 或小批拟合后的参数训练。
    # 默认 40,000 步、batch=64、500 步预热；学习率计划保持 configs/unet.json。
    log = TrainingLog(sys.stdout, mirror / 'train.log')
    status = {'status': 'running', **environment}
    atomic_json(mirror / 'status.json', status)
    start = time.perf_counter()
    try:
        with redirect_stdout(log):
            print(f"正式训练：{environment['gpu']} | 40000 步 | batch 64", flush=True)
            train(config, paths, device='cuda', resume=args.resume)
        status.update(status='complete', elapsed_seconds=time.perf_counter() - start)
    except BaseException as error:
        status.update(status='failed', error=repr(error), elapsed_seconds=time.perf_counter() - start)
        raise
    finally:
        log.flush()
        log.file.close()
        atomic_json(mirror / 'status.json', status)
        lock.close()


if __name__ == '__main__':
    main()
