"""Collect every search run into one results table and JSON."""
from __future__ import annotations

import json
from pathlib import Path

E = Path('/root/autodl-tmp/fedvade_project/reproduction_runs/round7_f01_full_repro')


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def last_row(run):
    p = run / 'rounds.jsonl'
    if not p.exists():
        return None
    lines = [x for x in p.read_text().splitlines() if x.strip()]
    return json.loads(lines[-1]) if lines else None


def snapshot(summary, epoch):
    for s in (summary or {}).get('snapshots', []):
        if s.get('epoch') == epoch:
            return s
    return {}


def manifest_paths(run):
    out = {}
    p = run / 'full_checkpoint_manifest.jsonl'
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                out[item['file']] = item.get('load_path')
    return out


def collect(stage, run):
    cfg = read_json(run / 'config.json') or {}
    ev = read_json(run / 'evaluation_summary.json')
    row = last_row(run) or {}
    paths = manifest_paths(run)
    e100 = paths.get('epoch_100_full.pt') or paths.get('epoch_100_full.pt.gz')
    rec = {
        'stage': stage, 'run': run.name,
        'config_id': None,
        'lr': cfg.get('lr'), 'alpha': cfg.get('alpha'),
        'schedule': cfg.get('schedule'), 'sample': cfg.get('sample'),
        'rounds': cfg.get('rounds'), 'seed': cfg.get('seed'),
        'client_loss_last': row.get('client_loss'),
        'server_loss_last': row.get('server_loss'),
        'ari_last_loop': row.get('ari'),
        'purity_last_loop': row.get('purity'),
        'last_eval': (ev or {}).get('last'),
        'epoch100': snapshot(ev, 100),
        'epoch150': snapshot(ev, 150),
        'epoch300': snapshot(ev, 300),
        'best_unlabeled_epoch': ((ev or {}).get('best_unlabeled_loss') or {}).get('epoch'),
        'best_exploratory_epoch': ((ev or {}).get('best_exploratory_ari') or {}).get('epoch'),
        'epoch100_full': e100,
        'full_checkpoint_count': len(paths),
        'full_checkpoints': paths,
        'run_dir': str(run),
    }
    if rec['last_eval']:
        rec['ari_last'] = rec['last_eval'].get('ari_mean')
        rec['ari_sd_last'] = rec['last_eval'].get('ari_std')
        rec['purity_last'] = rec['last_eval'].get('purity_mean')
        rec['nmi_last'] = rec['last_eval'].get('nmi_mean')
        rec['acc_last'] = rec['last_eval'].get('acc_mean')
        rec['client_loss_last5'] = rec['last_eval'].get('client_loss_last5_mean')
    return rec


def main():
    rows = []
    for stage in ('coarse', 'fine'):
        for run in sorted((E / stage).glob('*')):
            if (run / 'result.json').exists():
                rows.append(collect(stage, run))
    # Attach the grid config id (coarse dirs are named by it; fine dirs by
    # config id too, except the priority F01 run).
    plan = read_json(E / 'search_plan.json') or {}
    fine_plan = read_json(E / 'fine_plan.json') or []
    for r in rows:
        r['config_id'] = r['run']
        if r['run'] == 'F01':
            r['stage'] = 'priority'
            r['config_id'] = 'F01'
    # Priority run is not in coarse/fine; include it explicitly.
    priority = E / 'fine' / 'F01'
    if priority.exists() and not any(r['run_dir'] == str(priority) for r in rows):
        rows.append(collect('priority', priority))
        rows[-1]['config_id'] = 'F01'
    out = E / 'results'
    out.mkdir(exist_ok=True)
    (out / 'all_runs.json').write_text(json.dumps(rows, indent=2) + '\n')
    keys = ['stage', 'config_id', 'lr', 'alpha', 'schedule', 'sample', 'rounds',
            'seed', 'ari_last', 'ari_sd_last', 'purity_last', 'nmi_last',
            'acc_last', 'ari_last_loop', 'client_loss_last', 'server_loss_last',
            'client_loss_last5', 'epoch100_ari', 'epoch100_purity',
            'epoch150_ari', 'epoch300_ari', 'best_unlabeled_epoch',
            'best_exploratory_epoch', 'epoch100_full', 'full_checkpoint_count',
            'run_dir']
    with (out / 'all_runs.csv').open('w') as f:
        f.write(','.join(keys) + '\n')
        for r in rows:
            f.write(','.join(str({
                'epoch100_ari': (r.get('epoch100') or {}).get('ari_mean'),
                'epoch100_purity': (r.get('epoch100') or {}).get('purity_mean'),
                'epoch150_ari': (r.get('epoch150') or {}).get('ari_mean'),
                'epoch300_ari': (r.get('epoch300') or {}).get('ari_mean'),
            }.get(k, r.get(k, ''))).replace(',', ';') for k in keys) + '\n')
    print(f'COLLECTED {len(rows)} runs')


if __name__ == '__main__':
    main()
