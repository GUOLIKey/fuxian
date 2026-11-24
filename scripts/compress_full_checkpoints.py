"""Verify full federated checkpoints, then compress unkept ones.

Every `epoch_*_full.pt` is loaded and structurally verified before any
compression: server and all 10 client state dicts must load strictly into the
author VaDE, the optimizer/scheduler/RNG/partition/config fields must be
present, and the round must match the filename.  Kept files stay raw; the
others become `.pt.gz` (gzip -1, ~61% of raw) to fit the 50 GB data disk.

A JSONL manifest records raw/compressed sizes and sha256 values.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path('/root/autodl-tmp/fedvade_project')
SOURCE = ROOT / 'fuxian-round2/FedVaDE_source'
sys.path.insert(0, str(SOURCE))
from model import VaDE  # noqa: E402

REQUIRED = ('round', 'server', 'clients', 'server_optimizer',
            'client_optimizers', 'server_scheduler', 'client_schedulers',
            'python_rng', 'numpy_rng', 'torch_rng', 'config', 'partition',
            'losses')
# Coarse full checkpoints are moved to the 30 GB system disk to keep the
# 50 GB data disk free for fine runs and the F01 raw checkpoints.  A symlink
# remains at the run-dir path, so every recorded path keeps working.
ARCHIVE_ROOT = Path('/root/round7_full_checkpoints')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def make_vade(k, alpha):
    return VaDE(SimpleNamespace(input_dim=31, hid_dim=10, nClusters=int(k),
                                alpha=float(alpha), cuda=False))


def verify(path, alpha):
    payload = torch.load(path, map_location='cpu')
    missing = [k for k in REQUIRED if k not in payload]
    if missing:
        raise ValueError(f'{path.name}: missing {missing}')
    epoch_from_name = int(path.name.split('_')[1])
    if int(payload['round']) != epoch_from_name:
        raise ValueError(f'{path.name}: round {payload["round"]} != {epoch_from_name}')
    if len(payload['clients']) != 10:
        raise ValueError(f'{path.name}: {len(payload["clients"])} clients')
    server = make_vade(len(payload['server']['rho_']), alpha)
    server.load_state_dict(payload['server'], strict=True)
    client_shapes = []
    for i, state in enumerate(payload['clients']):
        model = make_vade(len(state['rho_']), alpha)
        model.load_state_dict(state, strict=True)
        client_shapes.append(int(len(state['rho_'])))
    if len(payload['client_optimizers']) != 10:
        raise ValueError(f'{path.name}: client optimizer count')
    if len(payload['client_schedulers']) != 10:
        raise ValueError(f'{path.name}: client scheduler count')
    if len(payload['partition']) != 10:
        raise ValueError(f'{path.name}: partition count')
    return {
        'file': path.name,
        'round': int(payload['round']),
        'server_K': int(len(payload['server']['rho_'])),
        'client_K': client_shapes,
        'n_clients': len(payload['clients']),
        'has_optimizers': True, 'has_schedulers': True, 'has_rng': True,
        'partition_records': [int(len(p['y'])) for p in payload['partition']],
        'losses': payload.get('losses'),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--keep', action='append', default=[])
    ap.add_argument('--verify-only', action='store_true')
    args = ap.parse_args()
    run = args.run_dir.resolve()
    config = json.loads((run / 'config.json').read_text())
    alpha = float(config['alpha'])
    manifest = run / 'full_checkpoint_manifest.jsonl'
    files = sorted(run.glob('epoch_*_full.pt'))
    if not files:
        raise SystemExit(f'No epoch_*_full.pt in {run}')
    with manifest.open('a') as out:
        for path in files:
            record = verify(path, alpha)
            raw_size = path.stat().st_size
            record['raw_size'] = int(raw_size)
            record['raw_sha256'] = sha256(path)
            if args.verify_only or path.name in args.keep:
                record.update(kept=True, compressed=False,
                              load_path=str(path))
            else:
                gz = path.with_suffix(path.suffix + '.gz')
                if gz.exists():
                    gz.unlink()
                subprocess.run(['gzip', '-1', str(path)], check=True)
                record.update(kept=False, compressed=True,
                              gz_size=int(gz.stat().st_size),
                              gz_sha256=sha256(gz),
                              load_path=str(gz),
                              decompress=f'gunzip -c {gz} > {path.name}')
                if run.parent.name == 'coarse':
                    archive_dir = ARCHIVE_ROOT / 'coarse' / run.name
                    archive_dir.mkdir(parents=True, exist_ok=True)
                    target = archive_dir / gz.name
                    shutil.move(str(gz), str(target))
                    gz.symlink_to(target)
                    record['archive_path'] = str(target)
                path = gz
            out.write(json.dumps(record) + '\n')
            print(json.dumps({'file': record['file'],
                              'kept': record['kept'],
                              'raw_mb': round(raw_size / 1e6, 1),
                              'gz_mb': round(record.get('gz_size', 0) / 1e6, 1)}),
                  flush=True)


if __name__ == '__main__':
    main()
