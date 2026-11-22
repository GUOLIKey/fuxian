"""Stage 1: audit Round 10 checkpoints, data hashes and model provenance."""
import csv
import json
from pathlib import Path

import numpy as np
import torch

from common import BASE, R10, sha256, update_progress, data

RUNS = [
    ('fedvade_iid_seed0', R10 / 'formal/iid_seed0', 'iid', 0, 'fedvade'),
    ('fedvade_medium_seed0', R10 / 'formal/medium_seed0', 'medium', 0, 'fedvade'),
    ('fedvade_severe_seed0', R10 / 'formal/severe_seed0', 'severe', 0, 'fedvade'),
    ('fedvade_severe_seed1', R10 / 'formal/severe_seed1', 'severe', 1, 'fedvade'),
    ('fedvade_severe_seed2', R10 / 'formal/severe_seed2', 'severe', 2, 'fedvade'),
    ('local_A', R10 / 'references/local_A', 'severe', 0, 'reference'),
    ('centralized', R10 / 'references/centralized', 'severe', 0, 'reference'),
]


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    update_progress(stage='SOURCE_AND_CHECKPOINT_AUDIT', status='RUNNING',
                    current_model='all Round 10 models', current_experiment='audit',
                    progress_current=1, progress_total=12, github='PENDING', error=None)

    manifest = json.loads((R10 / 'dataset_manifest.json').read_text())
    data_ok = {}
    data_ok['data.npz'] = sha256(R10 / 'private_data/data.npz') == manifest['data_npz_sha256']
    for scenario in ('iid', 'medium', 'severe'):
        data_ok[f'{scenario}_partitions.npz'] = (
            sha256(R10 / 'private_data' / f'{scenario}_partitions.npz') ==
            manifest['partition_npz_sha256'][scenario]
        )
    if not all(data_ok.values()):
        raise RuntimeError(f'Round 10 data hash mismatch: {data_ok}')

    metrics = {r['model']: r for r in read_csv(R10 / 'noniid_baseline_metrics.csv')}
    rows = []
    for name, run, scenario, seed, kind in RUNS:
        selected = run / 'selected_server.pt'
        if not selected.is_file():
            raise FileNotFoundError(selected)
        digest = sha256(selected)
        expected = metrics.get(name, {}).get('checkpoint_sha256')
        if expected and digest != expected:
            raise RuntimeError(f'{name}: selected checkpoint hash mismatch')
        payload = torch.load(selected, map_location='cpu')
        state = payload.get('server', payload)
        global_k = len(state['rho_'])
        epoch = payload.get('round')
        if epoch is None:
            conf_path = run / 'config.json'
            res_path = run / 'result.json'
            if res_path.is_file():
                epoch = json.loads(res_path.read_text()).get('best_validation', {}).get('epoch')
            elif conf_path.is_file():
                epoch = json.loads(conf_path.read_text()).get('selected_epoch')
        local_ks = ''
        client_count = 0
        last_full = run / 'last_full.pt'
        if last_full.is_file():
            lf = torch.load(last_full, map_location='cpu')
            clients = lf.get('clients', [])
            client_count = len(clients)
            local_ks = ';'.join(str(len(c['rho_'])) for c in clients)
        rows.append({
            'model': name,
            'scenario': scenario,
            'training_seed': seed,
            'kind': kind,
            'checkpoint_path': str(selected),
            'checkpoint_sha256': digest,
            'selected_epoch': epoch if epoch is not None else '',
            'global_K': global_k,
            'client_count': client_count,
            'local_Ks': local_ks,
            'last_full_path': str(last_full) if last_full.is_file() else '',
            'state_source': 'server state (selected_server.pt)',
            'data_hashes_ok': True,
        })

    out = BASE / 'checkpoint_manifest.csv'
    with out.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Record the frozen-data provenance for the report.
    d = data()
    provenance = {
        'data_hashes_ok': data_ok,
        'train_shape': list(d['X_train'].shape),
        'val_shape': list(d['X_val'].shape),
        'test_shape': list(d['X_test'].shape),
        'class_counts': {
            'train': np.bincount(d['y_train'], minlength=3).tolist(),
            'val': np.bincount(d['y_val'], minlength=3).tolist(),
            'test': np.bincount(d['y_test'], minlength=3).tolist(),
        },
        'models': rows,
        'round10_report_sha256': sha256(R10 / 'ROUND10_REPORT.md'),
    }
    (BASE / 'audit_summary.json').write_text(json.dumps(provenance, indent=2) + '\n')
    update_progress(github='PENDING', best_result='audit complete')
    print('AUDIT_OK', len(rows), flush=True)


if __name__ == '__main__':
    main()
