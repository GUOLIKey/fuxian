"""Analyze the 12 Stage A runs and write summary/paired-effect CSVs."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

ROOT = Path('/root/autodl-tmp/fedvade_project')
SOURCE = ROOT / 'fuxian-round2/FedVaDE_source'
BASE = ROOT / 'experiments/multica_factorial_v1'
RUNS = ROOT / 'reproduction_runs/multica_factorial_v1'
sys.path.insert(0, str(BASE / 'scripts'))
sys.path.insert(0, str(SOURCE))

from analysis_utils import (  # noqa: E402
    clustering_metrics, diag_nll, distribution_metrics, encode_mean, gmm_state_metrics,
)
from multica_core import (  # noqa: E402
    multi_ca_merge, reconstruct_client_components, sample_macro_gmm,
)
from dataloader import make_noniid_clients  # noqa: E402
from model import VaDE  # noqa: E402

RARE = 0  # BC minority class
GROUPS = ('A', 'B', 'C', 'D')
METRICS = [
    'unique_ari', 'unique_nmi', 'unique_purity', 'unique_acc',
    'server_gmm_nll', 'author_client_ari', 'author_client_purity',
    'gen_overall_mmd', 'gen_overall_swd', 'gen_minority_fraction',
    'gen_minority_mmd', 'gen_minority_swd', 'gen_nn_mean',
    'agg_overall_mmd', 'agg_overall_swd', 'agg_minority_fraction',
    'agg_minority_mmd', 'agg_minority_swd', 'agg_nn_mean',
    'n_local_components', 'n_macro_components', 'n_merges',
    'n_unsafe_merges', 'max_merge_normalized_bures',
]


def make_model(state):
    from argparse import Namespace
    model = VaDE(Namespace(input_dim=31, hid_dim=10, nClusters=len(state['rho_']),
                           alpha=0.01, cuda=False))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def decode(server, z):
    with torch.no_grad():
        return server.decoder(torch.tensor(z, dtype=torch.float32)).cpu().numpy()


def sample_params(params, n, seed):
    return sample_macro_gmm(params, n, seed)[0]


def mean(vals):
    vals = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def sd(vals):
    vals = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def main():
    out = BASE / 'results'
    out.mkdir(parents=True, exist_ok=True)
    _, loaders, (X, y), sizes = make_noniid_clients(
        str(SOURCE / 'dataset/breast_cancer.csv'), 10, 200, 'breast_cancer', seed=42)
    X = X.numpy().astype(np.float32)
    y = y.numpy().astype(np.int64)
    sizes = np.asarray(sizes, dtype=np.float64)
    client_X = np.concatenate([dl.dataset.tensors[0].numpy().astype(np.float32)
                               for dl in loaders], axis=0)
    client_y = np.concatenate([dl.dataset.tensors[1].numpy().astype(np.int64)
                               for dl in loaders], axis=0)
    clf = RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample',
                                 random_state=42, n_jobs=1).fit(X, y)
    rare_mask = y == RARE

    per_seed = []
    training_rows = []
    missing = []
    for group in GROUPS:
        for seed in (0, 1, 2):
            run = RUNS / 'stage_a' / f'{group}_seed{seed}'
            ckpt = run / 'checkpoints' / 'round_300_full.pt'
            if not ckpt.exists():
                missing.append(f'{group}_seed{seed}')
                continue
            p = torch.load(ckpt, map_location='cpu')
            server = make_model(p['server'])
            client_states = p['clients']
            z = encode_mean(server, X)
            zc = encode_mean(server, client_X)
            sm, nll = gmm_state_metrics(server, z, y)
            scm, scnll = gmm_state_metrics(server, zc, client_y)
            server_params = {
                'weights': torch.softmax(p['server']['rho_'], 0).numpy().astype(np.float64),
                'means': p['server']['mu_c'].numpy().astype(np.float64),
                'log_sigma2': p['server']['log_sigma2_c'].numpy().astype(np.float64),
            }
            w, mu, lv, owner, _ = reconstruct_client_components(client_states, sizes)
            if group in ('B', 'D'):
                agg = multi_ca_merge(w, mu, lv, owner, target_components=2,
                                     max_components=2, min_components=2,
                                     merge_threshold=1.0, allow_early_stop=False)
                agg_params = {'weights': agg['weights'], 'means': agg['means'],
                              'log_sigma2': agg['log_sigma2']}
                agg_source = 'multi_ca'
                n_merges = agg['n_merges']
                n_unsafe = agg['n_unsafe_merges']
                max_bw = agg['max_merge_normalized_bures']
            else:
                agg_params = {'weights': w, 'means': mu, 'log_sigma2': lv}
                agg_source = 'client_union'
                n_merges = 0
                n_unsafe = 0
                max_bw = None
            # Generation from the trained server GMM (primary) and from the aggregation distribution (diagnostic).
            gen_metrics = {'overall_mmd': [], 'overall_swd': [], 'minority_fraction': [],
                           'minority_mmd': [], 'minority_swd': [], 'nn_mean': []}
            agg_metrics = {k: [] for k in gen_metrics}
            for inf_seed in range(5):
                for target, params, bucket in (
                        ('server', server_params, gen_metrics),
                        ('aggregation', agg_params, agg_metrics)):
                    zgen = sample_params(params, 2000, seed=70000 + inf_seed)
                    gen = decode(server, zgen)
                    pred = clf.predict(gen)
                    dm = distribution_metrics(X, gen, seed=80000 + inf_seed)
                    g_rare = gen[pred == RARE]
                    r_rare = X[rare_mask]
                    if len(g_rare) >= 2 and len(r_rare) >= 2:
                        dmr = distribution_metrics(r_rare, g_rare, seed=90000 + inf_seed)
                    else:
                        dmr = {'mmd_rbf': None, 'sliced_wasserstein': None}
                    bucket['overall_mmd'].append(dm.get('mmd_rbf'))
                    bucket['overall_swd'].append(dm.get('sliced_wasserstein'))
                    bucket['minority_fraction'].append(float(np.mean(pred == RARE)))
                    bucket['minority_mmd'].append(dmr.get('mmd_rbf'))
                    bucket['minority_swd'].append(dmr.get('sliced_wasserstein'))
                    bucket['nn_mean'].append(dm.get('generated_to_real_nn_mean'))
            rows = [json.loads(line) for line in (run / 'rounds.jsonl').read_text().splitlines()]
            last = rows[-1]
            for r in rows:
                training_rows.append({'group': group, 'seed': seed,
                                      'round': int(r['round']),
                                      'unique_ari': float(r['unique_ari']),
                                      'server_gmm_nll': float(r['server_gmm_nll']),
                                      'n_macro_components': r.get('n_macro_components'),
                                      'n_merges': r.get('n_merges')})
            per_seed.append({
                'group': group, 'seed': seed,
                'unique_ari': sm['ari'], 'unique_nmi': sm['nmi'],
                'unique_purity': sm['purity'], 'unique_acc': sm['acc'],
                'server_gmm_nll': nll,
                'client_record_ari': scm['ari'], 'client_record_nmi': scm['nmi'],
                'client_record_purity': scm['purity'], 'client_record_acc': scm['acc'],
                'author_client_ari': float(last['client_record_ari']),
                'author_client_purity': float(last['client_record_purity']),
                'n_local_components': int(sum(len(s['rho_']) for s in client_states)),
                'n_macro_components': 2,
                'n_merges': int(n_merges), 'n_unsafe_merges': int(n_unsafe),
                'max_merge_normalized_bures': max_bw,
                'aggregation_source': agg_source,
                'gen_overall_mmd': mean(gen_metrics['overall_mmd']),
                'gen_overall_swd': mean(gen_metrics['overall_swd']),
                'gen_minority_fraction': mean(gen_metrics['minority_fraction']),
                'gen_minority_mmd': mean(gen_metrics['minority_mmd']),
                'gen_minority_swd': mean(gen_metrics['minority_swd']),
                'gen_nn_mean': mean(gen_metrics['nn_mean']),
                'agg_overall_mmd': mean(agg_metrics['overall_mmd']),
                'agg_overall_swd': mean(agg_metrics['overall_swd']),
                'agg_minority_fraction': mean(agg_metrics['minority_fraction']),
                'agg_minority_mmd': mean(agg_metrics['minority_mmd']),
                'agg_minority_swd': mean(agg_metrics['minority_swd']),
                'agg_nn_mean': mean(agg_metrics['nn_mean']),
            })
            print(json.dumps({'status': 'analyzed', 'run': f'{group}_seed{seed}',
                              'ari': sm['ari'], 'nll': nll}), flush=True)

    if missing:
        print(json.dumps({'status': 'PARTIAL', 'missing': missing}), flush=True)
    if not per_seed:
        raise SystemExit('No complete runs to analyze')
    # CSV per seed
    with (out / 'per_seed_results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_seed[0]))
        writer.writeheader(); writer.writerows(per_seed)
    # Group mean/std
    summary = []
    for group in GROUPS:
        rs = [r for r in per_seed if r['group'] == group]
        if not rs:
            continue
        row = {'group': group, 'n': len(rs)}
        for metric in METRICS:
            vals = [r.get(metric) for r in rs]
            row[metric + '_mean'] = mean(vals)
            row[metric + '_sd'] = sd(vals)
        summary.append(row)
    with (out / 'four_group_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    # Paired effects
    by = {(r['group'], r['seed']): r for r in per_seed}
    effects = []
    for seed in (0, 1, 2):
        if not all((g, seed) in by for g in GROUPS):
            continue
        for metric in METRICS:
            a = by[('A', seed)].get(metric); b = by[('B', seed)].get(metric)
            c = by[('C', seed)].get(metric); d = by[('D', seed)].get(metric)
            def diff(x, y):
                return None if x is None or y is None else float(x) - float(y)
            effects.append({
                'seed': seed, 'metric': metric,
                'B_minus_A': diff(b, a), 'D_minus_C': diff(d, c),
                'C_minus_A': diff(c, a), 'D_minus_B': diff(d, b),
                'interaction_DC_minus_BA': None if diff(d, c) is None or diff(b, a) is None
                                         else diff(d, c) - diff(b, a),
            })
    effect_fields = ['seed', 'metric', 'B_minus_A', 'D_minus_C', 'C_minus_A',
                     'D_minus_B', 'interaction_DC_minus_BA']
    with (out / 'paired_effects.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=effect_fields)
        writer.writeheader(); writer.writerows(effects)
    # Aggregate paired effects
    agg_effects = []
    for metric in METRICS:
        rs = [e for e in effects if e['metric'] == metric]
        row = {'metric': metric, 'n_pairs': len(rs)}
        for key in ('B_minus_A', 'D_minus_C', 'C_minus_A', 'D_minus_B',
                    'interaction_DC_minus_BA'):
            vals = [e[key] for e in rs if e[key] is not None]
            row[key + '_mean'] = mean(vals)
            row[key + '_sd'] = sd(vals)
        agg_effects.append(row)
    with (out / 'paired_effects_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(agg_effects[0]))
        writer.writeheader(); writer.writerows(agg_effects)
    with (out / 'training_curves.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(training_rows[0]))
        writer.writeheader(); writer.writerows(training_rows)
    (out / 'ANALYSIS_NOTES.md').write_text(
        '# Stage A analysis notes\n\n'
        '- Primary final result: round 300.\n'
        '- BC has no independent test split; metrics are author-protocol in-sample.\n'
        '- `gen_*` is generation from the trained server GMM.\n'
        '- `agg_*` is generation from the final aggregation distribution: Multi-CA macro for B/D, raw client union for A/C.\n'
        '- Minority metrics use an in-sample classifier and are exploratory proxies.\n'
        '- Paired effects use same training seed: B-A, D-C, C-A, D-B, interaction.\n'
        '- No metric was used to select checkpoints.\n'
    )
    print(json.dumps({'status': 'ANALYSIS_OK' if not missing else 'ANALYSIS_PARTIAL',
                      'runs': len(per_seed), 'missing': missing,
                      'out': str(out)}, indent=2))


if __name__ == '__main__':
    main()
