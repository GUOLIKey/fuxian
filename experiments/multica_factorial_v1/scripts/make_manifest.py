"""Build the read-only experiment manifest from run status files."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path('/root/autodl-tmp/fedvade_project')
BASE = ROOT / 'experiments/multica_factorial_v1'
RUNS = ROOT / 'reproduction_runs/multica_factorial_v1'
EXPECTED = [(g, s) for g in 'ABCD' for s in (0, 1, 2)]


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def run_entry(group, seed):
    out = RUNS / 'stage_a' / f'{group}_seed{seed}'
    status = read_json(out / 'status.json', {})
    result = read_json(out / 'result.json', {})
    progress = read_json(out / 'progress.json', {})
    if not out.exists():
        state = 'PENDING'
    elif result.get('status') == 'ok':
        state = 'COMPLETED'
    elif result.get('status') == 'failed':
        state = 'FAILED'
    elif status.get('status') in ('RUNNING', 'INTERRUPTED'):
        state = status['status']
    else:
        state = 'INTERRUPTED'
    row = {
        'group': group, 'seed': seed, 'run_id': f'{group}_seed{seed}',
        'status': state,
        'round': int(progress.get('round', status.get('round', 0)) or 0),
        'rounds': 300,
        'latest_unique_ari': progress.get('unique_ari'),
        'latest_unique_nmi': progress.get('unique_nmi'),
        'latest_server_gmm_nll': progress.get('server_gmm_nll'),
        'n_local_components': progress.get('n_local_components'),
        'n_macro_components': progress.get('n_macro_components'),
        'n_merges': progress.get('n_merges'),
        'result_path': str(out / 'result.json'),
        'last_full_checkpoint': str(out / 'checkpoints' / 'round_300_full.pt'),
        'error': status.get('error') or result.get('error'),
    }
    if state == 'COMPLETED':
        last = result.get('last', {})
        row.update({
            'final_unique_ari': last.get('unique_ari'),
            'final_unique_nmi': last.get('unique_nmi'),
            'final_unique_purity': last.get('unique_purity'),
            'final_server_gmm_nll': last.get('server_gmm_nll'),
            'final_n_local_components': last.get('n_local_components'),
            'final_n_macro_components': last.get('n_macro_components'),
        })
    return row


def main():
    rows = [run_entry(g, s) for g, s in EXPECTED]
    completed = sum(r['status'] == 'COMPLETED' for r in rows)
    failed = sum(r['status'] == 'FAILED' for r in rows)
    running = sum(r['status'] == 'RUNNING' for r in rows)
    pretrain = {}
    for s in (0, 1, 2):
        p = RUNS / 'pretrain' / f'seed{s}' / 'pretrain.pt'
        pretrain[f'seed{s}'] = {
            'status': 'COMPLETED' if p.exists() else 'PENDING',
            'path': str(p),
            'sha256': read_json(RUNS / 'pretrain' / f'seed{s}' / 'config.json', {}).get('checkpoint_sha256'),
        }
    github = read_json(BASE / 'github_publish_status.json', {'status': 'NOT_PUSHED'})
    manifest = {
        'experiment': 'multica_factorial_v1_stage_a_bc',
        'status_label': 'STAGE_A_BC_2x2',
        'groups': {
            'A': 'Author Original + original aggregation',
            'B': 'Author Original + Multi-CA',
            'C': 'F01 + original aggregation',
            'D': 'F01 + Multi-CA',
        },
        'seeds': [0, 1, 2],
        'rounds': 300,
        'pretrain_rounds': 50,
        'capacity': 'Multi-CA target K=2; same capacity as author global GMM',
        'pretrain': pretrain,
        'runs': rows,
        'total_completed': completed,
        'total_failed': failed,
        'total_running': running,
        'total_expected': len(rows),
        'github': github,
        'notes': [
            'BC has no independent test split; metrics are author-protocol in-sample except explicitly labeled proxies.',
            'Multi-CA does not increase component count in Stage A; no capacity confound.',
            'Stage B/C are not started by this manifest.',
        ],
    }
    tmp = BASE / 'experiment_manifest.json.tmp'
    tmp.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    tmp.replace(BASE / 'experiment_manifest.json')
    print(json.dumps({'status': 'MANIFEST_OK', 'completed': completed,
                      'failed': failed, 'running': running}, indent=2))


if __name__ == '__main__':
    main()
