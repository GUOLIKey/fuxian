"""Aggregate the 15 frozen-client global runs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

EXP = Path('/root/autodl-tmp/fedvade_project/experiments/acd_f01_global_v1')
RUNS = EXP / 'results/runs'
GROUPS = ('A', 'B', 'C', 'D', 'E')
SEEDS = (0, 1, 2)
CONTRASTS = (('D', 'A'), ('B', 'A'), ('D', 'B'), ('D', 'C'), ('D', 'E'))
METRICS = ('ari', 'nmi', 'purity', 'global_nll', 'gen_mmd', 'gen_swd',
           'minority_gen_frac', 'minority_mmd', 'server_loss')
INITIAL_REF = 0.605326980742813


def mean_sd(vals):
    arr = np.asarray([v for v in vals if v is not None and np.isfinite(v)],
                     dtype=np.float64)
    if len(arr) == 0:
        return None, None
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def main():
    runs = []
    for p in sorted(RUNS.glob('*_seed*.json')):
        r = json.loads(p.read_text())
        if r.get('status') != 'DONE':
            raise SystemExit(f'Incomplete run: {p}')
        rounds = r['rounds']
        if len(rounds) != 11:
            raise SystemExit(f'Expected 11 records: {p}')
        final = rounds[-1]; initial = rounds[0]
        if abs(initial['ari'] - INITIAL_REF) > 1e-12:
            raise SystemExit(f'Round0 ARI mismatch {p}: {initial["ari"]}')
        runs.append({
            'group': r['group'], 'seed': r['seed'],
            'initial_ari': initial['ari'],
            'ari': final['ari'], 'nmi': final['nmi'],
            'purity': final['purity'],
            'final_ari': final['ari'],
            'final_ari_sd_inference': final['ari_sd_inference'],
            'final_nmi': final['nmi'], 'final_purity': final['purity'],
            'delta_ari': final['ari'] - initial['ari'],
            'global_nll': final['global_nll'],
            'gen_mmd': final['gen_mmd'], 'gen_mmd_sd': final['gen_mmd_sd'],
            'gen_swd': final['gen_swd'], 'gen_swd_sd': final['gen_swd_sd'],
            'minority_gen_frac': final['minority_gen_frac'],
            'minority_mmd': final['minority_mmd'],
            'minority_swd': final['minority_swd'],
            'server_loss': final['server_loss'],
            'server_lr': final['server_lr'],
            'local_K': r['config']['local_K'],
            'K_mean': float(np.mean(r['config']['local_K'])),
            'highest_ari': r['highest_exploratory']['ari'],
            'highest_round': r['highest_exploratory']['round'],
        })
    if len(runs) != 15:
        raise SystemExit(f'Expected 15 runs, got {len(runs)}')
    # Per-seed CSV.
    keys = list(runs[0].keys())
    with (EXP / 'results/per_seed.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in runs:
            row = dict(r); row['local_K'] = json.dumps(row['local_K'])
            w.writerow(row)
    # Group summaries.
    summary = []
    for g in GROUPS:
        rs = [r for r in runs if r['group'] == g]
        def ms(k):
            return mean_sd([r[k] for r in rs])
        ari_m, ari_sd_exp = ms('final_ari')
        summary.append({
            'group': g, 'n_seeds': len(rs),
            'initial_ari': INITIAL_REF,
            'final_ari_mean': ari_m, 'final_ari_sd_experiment': ari_sd_exp,
            'final_ari_sd_inference_mean': float(np.mean(
                [r['final_ari_sd_inference'] for r in rs])),
            'delta_ari_mean': ms('delta_ari')[0],
            'delta_ari_sd': ms('delta_ari')[1],
            'nmi_mean': ms('final_nmi')[0], 'nmi_sd': ms('final_nmi')[1],
            'purity_mean': ms('final_purity')[0],
            'purity_sd': ms('final_purity')[1],
            'global_nll_mean': ms('global_nll')[0],
            'global_nll_sd': ms('global_nll')[1],
            'gen_mmd_mean': ms('gen_mmd')[0], 'gen_mmd_sd': ms('gen_mmd')[1],
            'gen_swd_mean': ms('gen_swd')[0], 'gen_swd_sd': ms('gen_swd')[1],
            'minority_frac_mean': ms('minority_gen_frac')[0],
            'minority_mmd_mean': ms('minority_mmd')[0],
            'minority_swd_mean': ms('minority_swd')[0],
            'server_loss_mean': ms('server_loss')[0],
            'server_loss_sd': ms('server_loss')[1],
            'K_mean': ms('K_mean')[0], 'K_min': min(r['K_mean'] for r in rs),
            'K_max': max(r['K_mean'] for r in rs),
            'highest_ari_mean_exploratory': ms('highest_ari')[0],
            'highest_rounds': [r['highest_round'] for r in rs],
        })
    with (EXP / 'results/summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    # Paired deltas at round 10.
    by = {(r['group'], r['seed']): r for r in runs}
    pair_rows = []
    for a, b in CONTRASTS:
        for metric in METRICS:
            vals, details = [], []
            for seed in SEEDS:
                ra, rb = by[(a, seed)], by[(b, seed)]
                va, vb = ra.get(metric), rb.get(metric)
                if va is None or vb is None:
                    continue
                d = float(va - vb)
                vals.append(d)
                details.append({'seed': seed, 'left': va, 'right': vb, 'delta': d})
            m, sd = mean_sd(vals)
            pair_rows.append({'contrast': f'{a}-{b}', 'metric': metric,
                              'n': len(vals), 'mean_delta': m, 'sd_delta': sd,
                              'values': json.dumps(details)})
    with (EXP / 'results/paired.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        w.writeheader(); w.writerows(pair_rows)
    # Curves.
    curve_rows = []
    for g in GROUPS:
        rs = [r for r in runs if r['group'] == g]
        seeds_runs = [json.loads((RUNS / f'{g}_seed{s}.json').read_text())
                      for s in SEEDS]
        for rnd in range(11):
            for metric in ('ari', 'nmi', 'purity', 'global_nll', 'gen_mmd',
                           'gen_swd', 'server_loss'):
                vals = [run['rounds'][rnd].get(metric) for run in seeds_runs]
                m, sd = mean_sd(vals)
                curve_rows.append({'group': g, 'round': rnd, 'metric': metric,
                                   'mean': m, 'sd': sd, 'n': len(vals),
                                   'values': json.dumps(vals)})
    with (EXP / 'results/curves.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(curve_rows[0].keys()))
        w.writeheader(); w.writerows(curve_rows)
    # Highest exploratory.
    with (EXP / 'results/highest.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['group', 'seed', 'highest_ari',
                                          'highest_round', 'final_ari'])
        w.writeheader()
        for r in runs:
            w.writerow({'group': r['group'], 'seed': r['seed'],
                        'highest_ari': r['highest_ari'],
                        'highest_round': r['highest_round'],
                        'final_ari': r['final_ari']})
    # Coverage summary.
    cov_path = EXP / 'results/coverage.jsonl'
    cov_rows = []
    if cov_path.exists():
        raw = [json.loads(x) for x in cov_path.read_text().splitlines() if x.strip()]
        for g in GROUPS:
            for rnd in range(1, 11):
                sub = [x for x in raw if x['group'] == g and x['round'] == rnd]
                if not sub:
                    continue
                cov_rows.append({
                    'group': g, 'round': rnd, 'n': len(sub),
                    'mean_unique_sampled': float(np.mean(
                        [x['mean_unique_sampled'] for x in sub])),
                    'mean_low_support_sampled': float(np.mean(
                        [x['low_support_sampled'] for x in sub])),
                    'mean_low_support_total': float(np.mean(
                        [x['low_support_total'] for x in sub])),
                })
    with (EXP / 'results/coverage_summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['group', 'round', 'n',
                                          'mean_unique_sampled',
                                          'mean_low_support_sampled',
                                          'mean_low_support_total'])
        w.writeheader(); w.writerows(cov_rows)
    initial = json.loads((EXP / 'results/initial_check.json').read_text())
    out = {'initial_check': initial, 'groups': summary,
           'per_seed': runs, 'paired': pair_rows,
           'primary_endpoint_round': 10,
           'frozen_reference_ari': INITIAL_REF}
    (EXP / 'results/summary.json').write_text(json.dumps(
        out, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x)) + '\n')
    print(json.dumps({'groups': summary, 'paired': [
        p for p in pair_rows if p['metric'] == 'ari']}, indent=2,
        default=lambda x: x.item() if hasattr(x, 'item') else str(x)))


if __name__ == '__main__':
    main()
