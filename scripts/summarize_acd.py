"""Aggregate frozen ACD experiment units and write CSVs, JSON and PCA points."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

ROOT = Path('/root/autodl-tmp/fedvade_project')
CKPT = (ROOT / 'reproduction_runs/round7_f01_full_repro/fine/F01'
        / 'epoch_100_full.pt')
EXP = ROOT / 'experiments/acd_f01_frozen_v1'
METHODS = ('A0', 'A1', 'A2', 'A3', 'A4', 'A5')
CONTRASTS = (('A4', 'A5'), ('A4', 'A2'), ('A4', 'A3'), ('A1', 'A0'))


def mean_sd(vals):
    arr = np.asarray([v for v in vals if v is not None and np.isfinite(v)],
                     dtype=np.float64)
    if len(arr) == 0:
        return None, None
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def load_units():
    units = []
    for p in sorted((EXP / 'results/units').glob('client*.json')):
        units.append(json.loads(p.read_text()))
    return units


def load_ys():
    import torch
    ckpt = torch.load(CKPT, map_location='cpu')
    return [np.asarray(p['y'], dtype=int) for p in ckpt['partition']]


def main():
    units = load_units()
    ys = load_ys()
    rows, component_rows, stability_rows = [], [], []
    pred_by_client_method = {}
    for u in units:
        client = int(u['client'])
        seed = int(u['seed'])
        y = ys[client]
        n_classes = len(u.get('class_labels', [])) or len(np.unique(y))
        multi = n_classes > 1
        for method in METHODS:
            rec = u.get('methods', {}).get(method)
            if not rec:
                continue
            f = rec['full']; g = rec['generation']; h = rec['holdout']
            interface = f.get('interface') or h.get('interface')
            interface_ok = None if interface is None else bool(interface.get('ok'))
            rows.append({
                'method': method, 'client': client, 'seed': seed,
                'n': u['n'], 'n_classes': n_classes, 'multi_class': bool(multi),
                'original_K': u['original_K'], 'K': f['K'],
                'ari': f['ari'], 'nmi': f['nmi'], 'purity': f['purity'],
                'acc': f['acc'],
                'n_supported_5': f['n_supported_5'],
                'n_supported_20': f['n_supported_20'],
                'n_low5': f['n_low5'], 'n_low20': f['n_low20'],
                'n_ghost': f['n_ghost'],
                'mass_low5': f['mass_low5'], 'mass_low20': f['mass_low20'],
                'pi_low5': f['pi_low5'], 'pi_low20': f['pi_low20'],
                'holdout_K': h['K'], 'holdout_nll': h['nll'],
                'reference_only': bool(h['reference_only']),
                'interface_ok': interface_ok,
                'gen_mmd': g['mmd'], 'gen_swd': g['swd'],
                'minority_label': g['minority_label'],
                'minority_mass': g['minority_mass'],
                'minority_gen_frac': g['minority_gen_frac'],
                'minority_mmd': g['minority_mmd'],
                'minority_swd': g['minority_swd'],
                'diversity': g['diversity_mean_pairwise'],
                'entropy': g['component_entropy_norm'],
            })
            pred = f.get('pred')
            if pred is not None:
                pred = np.asarray(pred, dtype=int)
                pred_by_client_method.setdefault((client, method), {})[seed] = pred
                params = f['params']
                support = np.asarray(f['support'], dtype=int)
                pi = np.asarray(params['pi'], dtype=float)
                var = np.asarray(params['var'], dtype=float)
                for k in range(len(pi)):
                    idx = np.flatnonzero(pred == k)
                    majority = int(np.bincount(y[idx]).argmax()) if len(idx) else None
                    component_rows.append({
                        'method': method, 'client': client, 'seed': seed,
                        'component': k, 'pi': float(pi[k]),
                        'support': int(support[k]),
                        'low5': bool(0 < support[k] < 5),
                        'low20': bool(0 < support[k] < 20),
                        'ghost': bool(support[k] == 0),
                        'majority_label': majority,
                        'mean_norm': float(np.linalg.norm(params['mu'][k])),
                        'mean_logvar': float(np.mean(params['logvar'][k])),
                    })
    for (client, method), d in sorted(pred_by_client_method.items()):
        seeds = sorted(d)
        vals = []
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                vals.append(float(adjusted_rand_score(d[seeds[i]], d[seeds[j]])))
        m, sd = mean_sd(vals)
        stability_rows.append({'client': client, 'method': method,
                               'n_seeds': len(seeds),
                               'mean_pairwise_ari': m,
                               'sd_pairwise_ari': sd})

    # Paired unit-level comparisons.
    by_key = {}
    for r in rows:
        by_key.setdefault((r['client'], r['seed']), {})[r['method']] = r
    pair_rows = []
    metrics = ('K', 'ari', 'nmi', 'purity', 'holdout_nll', 'gen_mmd',
               'gen_swd', 'minority_gen_frac', 'minority_mmd', 'diversity')
    for a, b in CONTRASTS:
        for (client, seed), methods in sorted(by_key.items()):
            if a not in methods or b not in methods:
                continue
            for metric in metrics:
                va, vb = methods[a].get(metric), methods[b].get(metric)
                if va is None or vb is None or not np.isfinite(va) or not np.isfinite(vb):
                    continue
                pair_rows.append({'contrast': f'{a}-{b}', 'client': client,
                                  'seed': seed, 'metric': metric,
                                  'delta': float(va - vb),
                                  'multi_class': methods[a]['multi_class']})
    pair_summary = []
    for a, b in CONTRASTS:
        for metric in metrics:
            vals = [r['delta'] for r in pair_rows
                    if r['contrast'] == f'{a}-{b}' and r['metric'] == metric]
            vals_m = [r['delta'] for r in pair_rows
                      if r['contrast'] == f'{a}-{b}' and r['metric'] == metric
                      and r['multi_class']]
            m, sd = mean_sd(vals); mm, msd = mean_sd(vals_m)
            pair_summary.append({'contrast': f'{a}-{b}', 'metric': metric,
                                 'n': len(vals), 'mean_delta': m, 'sd_delta': sd,
                                 'n_multi': len(vals_m), 'mean_delta_multi': mm,
                                 'sd_delta_multi': msd,
                                 'positive': int(sum(v > 0 for v in vals)),
                                 'negative': int(sum(v < 0 for v in vals)),
                                 'values': json.dumps(vals)})

    summary = []
    for method in METHODS:
        rs = [r for r in rows if r['method'] == method]
        rm = [r for r in rs if r['multi_class']]
        stab = [r['mean_pairwise_ari'] for r in stability_rows
                if r['method'] == method]
        def ms(key, subset=None):
            return mean_sd([r[key] for r in (subset if subset is not None else rs)])
        summary.append({
            'method': method, 'n_units': len(rs), 'n_multi': len(rm),
            'K_mean': ms('K')[0], 'K_sd': ms('K')[1],
            'K_min': min((r['K'] for r in rs), default=None),
            'K_max': max((r['K'] for r in rs), default=None),
            'ari_mean': ms('ari')[0], 'ari_sd': ms('ari')[1],
            'ari_multi_mean': ms('ari', rm)[0], 'ari_multi_sd': ms('ari', rm)[1],
            'nmi_mean': ms('nmi')[0], 'nmi_multi_mean': ms('nmi', rm)[0],
            'purity_mean': ms('purity')[0], 'purity_multi_mean': ms('purity', rm)[0],
            'holdout_nll_mean': ms('holdout_nll')[0],
            'holdout_nll_sd': ms('holdout_nll')[1],
            'gen_mmd_mean': ms('gen_mmd')[0], 'gen_mmd_sd': ms('gen_mmd')[1],
            'gen_swd_mean': ms('gen_swd')[0], 'gen_swd_sd': ms('gen_swd')[1],
            'minority_frac_mean': ms('minority_gen_frac')[0],
            'minority_mmd_mean': ms('minority_mmd')[0],
            'minority_swd_mean': ms('minority_swd')[0],
            'diversity_mean': ms('diversity')[0],
            'entropy_mean': ms('entropy')[0],
            'low5_mean': ms('n_low5')[0], 'low20_mean': ms('n_low20')[0],
            'ghost_mean': ms('n_ghost')[0],
            'pi_low5_mean': ms('pi_low5')[0],
            'stability_mean': mean_sd(stab)[0],
            'stability_sd': mean_sd(stab)[1],
            'interface_ok_n': sum(1 for r in rs if r['interface_ok'] is True),
            'interface_check_n': sum(1 for r in rs if r['interface_ok'] is not None),
        })

    # Core descriptive answers.
    by_method = {m: [r for r in rows if r['method'] == m] for m in METHODS}
    multi_clients = sorted({r['client'] for r in rows if r['multi_class']})
    a4_gt_a2 = sum(1 for r in by_method['A4']
                   if any(x['client'] == r['client'] and x['seed'] == r['seed']
                          and x['method'] == 'A2' and r['K'] > x['K']
                          for x in rows))
    a4_gt_a3 = sum(1 for r in by_method['A4']
                   if any(x['client'] == r['client'] and x['seed'] == r['seed']
                          and x['method'] == 'A3' and r['K'] > x['K']
                          for x in rows))
    def pmean(contrast, metric, multi=False):
        r = next((x for x in pair_summary if x['contrast'] == contrast
                  and x['metric'] == metric), None)
        if not r:
            return None
        return r['mean_delta_multi'] if multi else r['mean_delta']
    core = {
        'n_units': len(units), 'n_clients': len({u['client'] for u in units}),
        'n_multi_class_clients': len(multi_clients),
        'multi_class_clients': multi_clients,
        'A4_K_gt_A2_units': a4_gt_a2,
        'A4_K_gt_A3_units': a4_gt_a3,
        'A4_vs_A5': {m: pmean('A4-A5', m, True) for m in
                     ('K', 'ari', 'nmi', 'purity', 'holdout_nll', 'gen_mmd',
                      'gen_swd', 'minority_gen_frac', 'minority_mmd', 'diversity')},
        'A4_vs_A2': {m: pmean('A4-A2', m, True) for m in
                     ('K', 'ari', 'nmi', 'purity', 'holdout_nll', 'gen_mmd',
                      'gen_swd', 'minority_gen_frac', 'minority_mmd', 'diversity')},
        'A4_vs_A3': {m: pmean('A4-A3', m, True) for m in
                     ('K', 'ari', 'nmi', 'purity', 'holdout_nll', 'gen_mmd',
                      'gen_swd', 'minority_gen_frac', 'minority_mmd', 'diversity')},
        'A4_interface_all_ok': all(r['interface_ok'] for r in by_method['A4']
                                  if r['interface_ok'] is not None),
        'failures': sum(len(u.get('failures', [])) for u in units),
    }

    out = EXP / 'results'
    def write_csv(name, rows_, keys):
        with (out / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows_:
                w.writerow({k: r.get(k) for k in keys})
    write_csv('per_client.csv', rows, list(rows[0].keys()))
    write_csv('summary.csv', summary, list(summary[0].keys()))
    write_csv('stability.csv', stability_rows,
              ['client', 'method', 'n_seeds', 'mean_pairwise_ari', 'sd_pairwise_ari'])
    write_csv('paired.csv', pair_summary, list(pair_summary[0].keys()))
    write_csv('per_component.csv', component_rows,
              list(component_rows[0].keys()))
    (out / 'summary.json').write_text(json.dumps({
        'methods': summary, 'core': core, 'paired': pair_summary,
        'n_units': len(units),
        'class_counts_by_client': {str(u['client']): u.get('class_counts')
                                   for u in units},
    }, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x)) + '\n')

    # PCA points for one informative multi-class client / seed 0.
    if multi_clients:
        client = max(multi_clients, key=lambda c: next(
            u['n'] for u in units if u['client'] == c))
        u = next(u for u in units if u['client'] == client and u['seed'] == 0)
        # Recompute z from the checkpoint encoder to avoid storing latents.
        import torch
        from types import SimpleNamespace
        import sys
        sys.path.insert(0, str(ROOT / 'fuxian-round2/FedVaDE_source'))
        from model import VaDE
        ckpt = torch.load(CKPT, map_location='cpu')
        state = ckpt['clients'][client]
        model = VaDE(SimpleNamespace(input_dim=31, hid_dim=10,
                                     nClusters=int(len(state['rho_'])),
                                     alpha=0.05, cuda=False))
        model.load_state_dict(state, strict=True); model.eval()
        X = np.asarray(ckpt['partition'][client]['x'], dtype=np.float64)
        y = np.asarray(ckpt['partition'][client]['y'], dtype=int)
        with torch.no_grad():
            z = model.encoder(torch.tensor(X, dtype=torch.float32))[0].numpy()
        pcs = PCA(n_components=2, random_state=0).fit_transform(z)
        pca_rows = []
        for method in ('A2', 'A4'):
            pred = u.get('methods', {}).get(method, {}).get('full', {}).get('pred')
            if pred is None:
                continue
            for i in range(len(y)):
                pca_rows.append({'client': client, 'seed': 0, 'method': method,
                                 'pc1': float(pcs[i, 0]), 'pc2': float(pcs[i, 1]),
                                 'cluster': int(pred[i]), 'true_label': int(y[i])})
        write_csv('pca_points.csv', pca_rows,
                  ['client', 'seed', 'method', 'pc1', 'pc2', 'cluster',
                   'true_label'])
    print(json.dumps(core, indent=2, default=lambda x: x.item()
                     if hasattr(x, 'item') else str(x)))


if __name__ == '__main__':
    main()
