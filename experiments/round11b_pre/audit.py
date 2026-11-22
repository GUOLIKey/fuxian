"""Stage 1: audit Round 10 checkpoints, Round 11A evaluator and data provenance."""
import csv
import json
from pathlib import Path

import torch

from common import (BASE, R10, R11A, sha256, update_progress, data, write_csv)


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    update_progress(stage='SOURCE_AUDIT', status='RUNNING',
                    current_model='all', current_config='audit',
                    completed=0, total=255, best_val_ll=None,
                    rare_mode_proxy=None, github='PENDING', error=None)
    manifest = json.loads((R10 / 'dataset_manifest.json').read_text())
    data_hashes = {
        'data.npz': sha256(R10 / 'private_data/data.npz'),
    }
    for scenario in ('iid', 'medium', 'severe'):
        data_hashes[f'{scenario}_partitions.npz'] = sha256(
            R10 / 'private_data' / f'{scenario}_partitions.npz')
    data_ok = {
        'data.npz': data_hashes['data.npz'] == manifest['data_npz_sha256'],
    }
    for scenario in ('iid', 'medium', 'severe'):
        data_ok[f'{scenario}_partitions.npz'] = (
            data_hashes[f'{scenario}_partitions.npz'] ==
            manifest['partition_npz_sha256'][scenario])
    if not all(data_ok.values()):
        raise RuntimeError(f'Round 10 data hash mismatch: {data_ok}')

    r11a_files = [
        'ROUND11A_REPORT.md', 'evaluation_protocol.md', 'evaluator_config.json',
        'gmm_k_comparison.csv', 'global_vs_union_mog.csv',
        'evaluator_calibration.csv', 'reconstruction_diagnostics.csv',
        'latent_separability.csv', 'summary.json',
    ]
    r11a_hashes = {}
    for name in r11a_files:
        path = R11A / name
        if not path.is_file():
            raise FileNotFoundError(path)
        r11a_hashes[name] = sha256(path)

    metrics = {r['model']: r for r in csv.DictReader(
        (R10 / 'noniid_baseline_metrics.csv').open(newline=''))}
    runs = [
        ('fedvade_severe_seed0', R10 / 'formal/severe_seed0', 0, 'fedvade'),
        ('fedvade_severe_seed1', R10 / 'formal/severe_seed1', 1, 'fedvade'),
        ('fedvade_severe_seed2', R10 / 'formal/severe_seed2', 2, 'fedvade'),
        ('centralized', R10 / 'references/centralized', 0, 'reference'),
    ]
    rows = []
    for name, run, seed, kind in runs:
        selected = run / 'selected_server.pt'
        last_full = run / 'last_full.pt'
        if not selected.is_file() or not last_full.is_file():
            raise FileNotFoundError(f'{name}: missing selected_server.pt or last_full.pt')
        digest = sha256(selected)
        expected = metrics.get(name, {}).get('checkpoint_sha256')
        if expected and digest != expected:
            raise RuntimeError(f'{name}: checkpoint hash mismatch')
        payload = torch.load(selected, map_location='cpu')
        state = payload.get('server', payload)
        lf = torch.load(last_full, map_location='cpu')
        clients = lf.get('clients', [])
        local_ks = ';'.join(str(len(c['rho_'])) for c in clients)
        rows.append({
            'model': name,
            'training_seed': seed,
            'kind': kind,
            'checkpoint_path': str(selected),
            'checkpoint_sha256': digest,
            'selected_epoch': payload.get('round', ''),
            'global_K': len(state['rho_']),
            'client_count': len(clients),
            'local_Ks': local_ks,
            'last_full_path': str(last_full),
        })
    write_csv(BASE / 'checkpoint_manifest.csv', rows)

    config = json.loads((R11A / 'evaluator_config.json').read_text())
    required_eval = {
        'primary_classifier': 'calibrated_isotonic',
        'tau': 0.5,
    }
    if config.get('primary_classifier') != required_eval['primary_classifier']:
        raise RuntimeError('Round 11A primary evaluator changed')
    if abs(float(config.get('tau')) - required_eval['tau']) > 1e-9:
        raise RuntimeError('Round 11A tau changed')
    if 'radius' not in config:
        raise RuntimeError('Round 11A radius missing')

    d = data()
    audit = {
        'data_hashes': data_hashes,
        'data_hashes_ok': data_ok,
        'r11a_hashes': r11a_hashes,
        'evaluator_config': {
            'primary_classifier': config['primary_classifier'],
            'tau': config['tau'],
            'radius': config['radius'],
            'tau_rule': config.get('tau_rule'),
            'radius_rule': config.get('radius_rule'),
        },
        'class_counts': {
            'train': torch.bincount(torch.tensor(d['y_train'])).tolist(),
            'val': torch.bincount(torch.tensor(d['y_val'])).tolist(),
            'test': torch.bincount(torch.tensor(d['y_test'])).tolist(),
        },
        'models': rows,
        'training': 'none; frozen Round 10 checkpoints only',
    }
    (BASE / 'source_checkpoint_audit.json').write_text(json.dumps(audit, indent=2) + '\n')

    lines = [
        '# Round 11B-pre source and checkpoint audit',
        '',
        'No neural network training is performed in this round.',
        '',
        '## Data',
        '',
    ]
    for name, ok in data_ok.items():
        lines.append(f'- `{name}`: {"OK" if ok else "MISMATCH"} (`{data_hashes[name]}`)')
    lines += ['', '## Fixed Round 11A evaluator', '',
              f'- Primary classifier: `{config["primary_classifier"]}`',
              f'- Tau: {config["tau"]}',
              f'- Radius: {config["radius"]}',
              f'- Tau rule: {config.get("tau_rule", "")}',
              f'- Radius rule: {config.get("radius_rule", "")}',
              '',
              '## Frozen models', '',
              '| Model | Seed | Kind | Selected epoch | Global K | Clients | Local Ks | SHA256 |',
              '| --- | ---: | --- | ---: | ---: | ---: | --- | --- |']
    for r in rows:
        lines.append(f"| {r['model']} | {r['training_seed']} | {r['kind']} | "
                     f"{r['selected_epoch']} | {r['global_K']} | {r['client_count']} | "
                     f"{r['local_Ks'] or '—'} | `{r['checkpoint_sha256'][:16]}…` |")
    lines += ['', '## Round 11A files reused', '']
    for name, digest in r11a_hashes.items():
        lines.append(f'- `{name}`: `{digest}`')
    (BASE / 'source_checkpoint_audit.md').write_text('\n'.join(lines) + '\n')
    update_progress(stage='CHECKPOINT_VALIDATION', status='RUNNING',
                    current_config='audit complete', completed=1)
    print('AUDIT_OK', len(rows), flush=True)


if __name__ == '__main__':
    main()
