"""Frozen-model proof of concept for capacity-matched Multi-CA.

Uses existing full checkpoints with real client states.  No training, no
modification of checkpoints.  This is an in-sample BC diagnostic, not a
generalization test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture

ROOT = Path('/root/autodl-tmp/fedvade_project')
SOURCE = ROOT / 'fuxian-round2/FedVaDE_source'
BASE = ROOT / 'experiments/multica_factorial_v1'
sys.path.insert(0, str(BASE / 'scripts'))
sys.path.insert(0, str(SOURCE))

from analysis_utils import (  # noqa: E402
    clustering_metrics, diag_nll, diag_responsibilities, distribution_metrics,
    encode_mean, gmm_state_metrics,
)
from multica_core import (  # noqa: E402
    multi_ca_merge, plain_wasserstein_merge, reconstruct_client_components,
    sample_macro_gmm,
)
from dataloader import make_noniid_clients  # noqa: E402
from model import VaDE  # noqa: E402

RARE = 0  # BC minority class


def load_state(path):
    payload = torch.load(path, map_location='cpu')
    return payload['server'], payload['clients'], payload.get('sizes'), payload.get('round')


def make_model(state):
    from argparse import Namespace
    model = VaDE(Namespace(input_dim=31, hid_dim=10, nClusters=len(state['rho_']),
                           alpha=0.01, cuda=False))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def params_from_state(state):
    return {
        'weights': torch.softmax(state['rho_'], 0).numpy().astype(np.float64),
        'means': state['mu_c'].numpy().astype(np.float64),
        'log_sigma2': state['log_sigma2_c'].numpy().astype(np.float64),
    }


def sample_union(weights, means, log_sigma2, n, seed):
    rng = np.random.default_rng(int(seed))
    w = np.asarray(weights, dtype=np.float64); w = w / w.sum()
    mu = np.asarray(means, dtype=np.float64)
    lv = np.asarray(log_sigma2, dtype=np.float64)
    labels = rng.choice(len(w), size=int(n), p=w)
    z = mu[labels] + rng.normal(size=(int(n), mu.shape[1])) * np.sqrt(np.exp(lv[labels]))
    return z


def eval_params(params, z, y):
    resp = diag_responsibilities(z, params['weights'], params['means'], params['log_sigma2'])
    metrics = clustering_metrics(y, resp.argmax(1))
    return {
        **metrics,
        'nll': diag_nll(z, params['weights'], params['means'], params['log_sigma2']),
        'n_components': int(len(params['weights'])),
        'weights': [float(x) for x in params['weights']],
    }


def decode(server, z):
    with torch.no_grad():
        return server.decoder(torch.tensor(z, dtype=torch.float32)).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=BASE / 'frozen_poc')
    p.add_argument('--union-fit-M', type=int, default=10000)
    p.add_argument('--generation-n', type=int, default=2000)
    p.add_argument('--inference-seeds', type=int, default=5)
    p.add_argument('--union-seed', type=int, default=2026)
    args = p.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    checkpoints = {
        'author_round300': ROOT / 'reproduction_runs/round5_broadcast/formal_author_seed0/round_300_full.pt',
        'f01_round300': ROOT / 'reproduction_runs/round7_hyperparameter_search/fine/F01/last_checkpoint.pt',
    }
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

    rows = []
    trees = {}
    for name, path in checkpoints.items():
        if not path.exists():
            raise FileNotFoundError(path)
        server_state, client_states, ckpt_sizes, round_no = load_state(path)
        server = make_model(server_state)
        z_all = encode_mean(server, X)
        z_clients = encode_mean(server, client_X)
        w, mu, lv, owner, _ = reconstruct_client_components(client_states, ckpt_sizes or sizes)
        methods = {
            'server_gmm': {
                **params_from_state(server_state),
                'n_input_components': int(len(server_state['rho_'])),
                'n_merges': 0,
                'n_unsafe_merges': 0,
                'max_merge_normalized_bures': None,
            },
            'flat_gmm_K2': None,
            'plain_wasserstein_K2': None,
            'multi_ca_K2': None,
        }
        z_union = sample_union(w, mu, lv, args.union_fit_M, args.union_seed)
        flat = GaussianMixture(n_components=2, covariance_type='diag', random_state=42,
                               n_init=5, max_iter=500, reg_covar=1e-5).fit(z_union)
        methods['flat_gmm_K2'] = {
            'weights': flat.weights_.astype(np.float64),
            'means': flat.means_.astype(np.float64),
            'log_sigma2': np.log(np.maximum(flat.covariances_.astype(np.float64), 1e-12)),
            'n_input_components': int(len(w)),
            'n_merges': None, 'n_unsafe_merges': 0,
            'max_merge_normalized_bures': None,
        }
        plain = plain_wasserstein_merge(w, mu, lv, owner, n_clusters=2)
        methods['plain_wasserstein_K2'] = {
            'weights': plain['weights'], 'means': plain['means'],
            'log_sigma2': plain['log_sigma2'],
            'n_input_components': plain['n_input_components'],
            'n_merges': plain['n_merges'], 'n_unsafe_merges': 0,
            'max_merge_normalized_bures': None,
        }
        multi = multi_ca_merge(w, mu, lv, owner, target_components=2,
                               max_components=2, min_components=2,
                               merge_threshold=1.0, allow_early_stop=False)
        methods['multi_ca_K2'] = {
            'weights': multi['weights'], 'means': multi['means'],
            'log_sigma2': multi['log_sigma2'],
            'n_input_components': multi['n_input_components'],
            'n_merges': multi['n_merges'],
            'n_unsafe_merges': multi['n_unsafe_merges'],
            'max_merge_normalized_bures': multi['max_merge_normalized_bures'],
        }
        trees[name] = {
            'plain_wasserstein_K2': plain['merge_events'],
            'multi_ca_K2': multi['merge_events'],
            'multi_ca_summary': {k: v for k, v in multi.items()
                                 if k not in ('weights', 'means', 'log_sigma2', 'merge_events', 'owners')},
        }
        for method, params in methods.items():
            m_all = eval_params(params, z_all, y)
            m_clients = eval_params(params, z_clients, client_y)
            gen_all = []
            for inf_seed in range(args.inference_seeds):
                z_gen, _ = sample_macro_gmm(params, args.generation_n, seed=10000 + inf_seed)
                gen = decode(server, z_gen)
                pred = clf.predict(gen)
                p_rare = clf.predict_proba(gen)[:, list(clf.classes_).index(RARE)]
                dm = distribution_metrics(X, gen, seed=20000 + inf_seed)
                g_rare = gen[pred == RARE]
                r_rare = X[y == RARE]
                if len(g_rare) >= 2 and len(r_rare) >= 2:
                    dm_rare = distribution_metrics(r_rare, g_rare, seed=30000 + inf_seed)
                else:
                    dm_rare = {'mmd_rbf': None, 'sliced_wasserstein': None}
                gen_all.append({
                    'rare_fraction_pred': float(np.mean(pred == RARE)),
                    'rare_prob_mean': float(np.mean(p_rare)),
                    'rare_count_pred': int(np.sum(pred == RARE)),
                    'rare_mmd': dm_rare.get('mmd_rbf'),
                    'rare_swd': dm_rare.get('sliced_wasserstein'),
                    'overall_mmd': dm.get('mmd_rbf'),
                    'overall_swd': dm.get('sliced_wasserstein'),
                    'nn_mean': dm.get('generated_to_real_nn_mean'),
                })
            def mean(key):
                vals = [x[key] for x in gen_all if x[key] is not None]
                return float(np.mean(vals)) if vals else None
            rows.append({
                'checkpoint': name, 'checkpoint_round': round_no,
                'method': method,
                'K': m_all['n_components'],
                'n_input_components': params['n_input_components'],
                'n_merges': params['n_merges'],
                'n_unsafe_merges': params['n_unsafe_merges'],
                'max_merge_normalized_bures': params['max_merge_normalized_bures'],
                'nll_all': m_all['nll'], 'ari_all': m_all['ari'],
                'nmi_all': m_all['nmi'], 'purity_all': m_all['purity'], 'acc_all': m_all['acc'],
                'weights_all': json.dumps(m_all['weights']),
                'nll_clients': m_clients['nll'], 'ari_clients': m_clients['ari'],
                'nmi_clients': m_clients['nmi'], 'purity_clients': m_clients['purity'],
                'acc_clients': m_clients['acc'],
                'rare_fraction_pred': mean('rare_fraction_pred'),
                'rare_prob_mean': mean('rare_prob_mean'),
                'rare_count_pred': mean('rare_count_pred'),
                'rare_mmd': mean('rare_mmd'), 'rare_swd': mean('rare_swd'),
                'overall_mmd': mean('overall_mmd'), 'overall_swd': mean('overall_swd'),
                'nn_mean': mean('nn_mean'),
                'note': 'FROZEN_IN_SAMPLE_BC; classifier is in-sample; no independent test split',
            })
        print(json.dumps({'status': 'checkpoint_done', 'checkpoint': name,
                          'rows': len(rows)}), flush=True)

    # Write CSV.
    fields = list(rows[0].keys())
    import csv
    with (out / 'FROZEN_POC_RESULTS.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out / 'FROZEN_MERGE_TREES.json').write_text(json.dumps(trees, indent=2) + '\n')
    print(json.dumps({'status': 'FROZEN_POC_OK', 'rows': len(rows), 'out': str(out)}, indent=2))


if __name__ == '__main__':
    main()
