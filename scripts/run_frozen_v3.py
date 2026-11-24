"""Frozen F01 + Multi-CA first-round aggregation experiment.

Loads the real F01 epoch-100 full federated checkpoint, keeps every weight
unchanged, and compares:

  A      original F01 global GMM (K=2)
  B      flat GMM built from the same 10 client local GMMs, same leaf count
  C      adaptive Macro-Micro Multi-CA GMM
  union  raw union of the 14 client local-GMM components (diagnostic)

No training, no client reconstruction, no VAE fine-tuning.  Seeds 0-9 are
inference/sampling seeds only; generation uses seeds 0,1,2.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path('/root/autodl-tmp/fedvade_project')
SOURCE = ROOT / 'fuxian-round2/FedVaDE_source'
DATA = SOURCE / 'dataset/breast_cancer.csv'
CKPT = (ROOT / 'reproduction_runs/round7_f01_full_repro/fine/F01'
        / 'epoch_100_full.pt')
EXP = ROOT / 'experiments/multica_f01_frozen_v3'
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(EXP / 'scripts'))
from dataloader import make_noniid_clients  # noqa: E402
from model import VaDE  # noqa: E402
import multica_core as core  # noqa: E402
import adaptive_multica as am  # noqa: E402

EXPECTED_SHA = '3be341c451a186d1f36673080164af208993d215f0c2cb03709407892f7d6f3e'
GEN_SEEDS = (0, 1, 2)
INF_SEEDS = tuple(range(10))
N_GEN = 5000
EPS = 1e-10


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, obj):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default) + '\n')
    tmp.replace(path)


def _json_default(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


def status(stage, message, done=False, **extra):
    payload = {'stage': stage, 'message': message, 'done': bool(done),
               'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    payload.update(extra)
    atomic_json(EXP / 'status.json', payload)


def make_server(state):
    model = VaDE(SimpleNamespace(input_dim=31, hid_dim=10,
                                 nClusters=int(len(state['rho_'])),
                                 alpha=0.05, cuda=False))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def make_client(state):
    model = VaDE(SimpleNamespace(input_dim=31, hid_dim=10,
                                 nClusters=int(len(state['rho_'])),
                                 alpha=0.05, cuda=False))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def cluster_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    d = int(max(y_true.max(), y_pred.max())) + 1
    conf = np.zeros((d, d), dtype=np.int64)
    for a, b in zip(y_true, y_pred):
        conf[int(a), int(b)] += 1
    r, c = linear_sum_assignment(-conf)
    return {
        'ari': float(adjusted_rand_score(y_true, y_pred)),
        'nmi': float(normalized_mutual_info_score(y_true, y_pred)),
        'purity': float(np.max(conf, axis=0).sum() / len(y_true)),
        'acc': float(conf[r, c].sum() / len(y_true)),
    }


def mmd2_rbf(a, b, sigma):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a2 = np.sum(a * a, 1)
    b2 = np.sum(b * b, 1)
    kxx = np.exp(-np.maximum(a2[:, None] + a2[None, :] - 2 * a @ a.T, 0) /
                 (2 * sigma * sigma))
    kyy = np.exp(-np.maximum(b2[:, None] + b2[None, :] - 2 * b @ b.T, 0) /
                 (2 * sigma * sigma))
    kxy = np.exp(-np.maximum(a2[:, None] + b2[None, :] - 2 * a @ b.T, 0) /
                 (2 * sigma * sigma))
    return float(max(kxx.mean() + kyy.mean() - 2 * kxy.mean(), 0.0))


def swd(a, b, directions):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    q = np.linspace(0, 1, 200)
    return float(np.mean(np.abs(np.quantile(a @ directions, q, axis=0) -
                                np.quantile(b @ directions, q, axis=0))))


def mahalanobis(points, data):
    points = np.asarray(points, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)
    d = data.shape[1]
    mean = data.mean(0)
    cov = np.cov(data.T) + 1e-6 * np.eye(d)
    inv = np.linalg.pinv(cov)
    diff = points - mean
    return np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', diff, inv, diff), 0))


def latent_compatibility(server, clients, client_states, Xu):
    with torch.no_grad():
        zs = server.encoder(torch.tensor(Xu, dtype=torch.float32))[0].numpy()
        zc = [c.encoder(torch.tensor(Xu, dtype=torch.float32))[0].numpy()
              for c in clients]
    sigma = float(np.sqrt(np.median(pdist(zs, metric='sqeuclidean'))))
    if sigma <= 0:
        sigma = 1.0
    rows = []
    for i, (m, state) in enumerate(zip(zc, client_states)):
        means = state['mu_c'].detach().cpu().numpy()
        mah_local = mahalanobis(means, m)
        mah_server = mahalanobis(means, zs)
        rows.append({
            'client': i,
            'latent_dim': int(m.shape[1]),
            'mean_l2_to_server': float(np.linalg.norm(zs.mean(0) - m.mean(0))),
            'cov_frobenius_to_server': float(np.linalg.norm(
                np.cov(zs.T) - np.cov(m.T))),
            'mmd_rbf_to_server': float(math.sqrt(max(mmd2_rbf(zs, m, sigma), 0))),
            'local_K': int(len(state['rho_'])),
            'gmm_mean_mahalanobis_local_mean': float(np.mean(mah_local)),
            'gmm_mean_mahalanobis_local_max': float(np.max(mah_local)),
            'gmm_mean_mahalanobis_server_mean': float(np.mean(mah_server)),
            'gmm_mean_mahalanobis_server_max': float(np.max(mah_server)),
        })
    server_mah = mahalanobis(
        server.mu_c.detach().cpu().numpy(), zs)
    return {
        'latent_dim': int(zs.shape[1]),
        'mmd_sigma': sigma,
        'server_latent_mean': zs.mean(0).tolist(),
        'max_mean_l2_to_server': max(r['mean_l2_to_server'] for r in rows),
        'max_cov_frobenius_to_server': max(r['cov_frobenius_to_server'] for r in rows),
        'max_mmd_rbf_to_server': max(r['mmd_rbf_to_server'] for r in rows),
        'max_gmm_mean_mahalanobis_local': max(
            r['gmm_mean_mahalanobis_local_max'] for r in rows),
        'max_gmm_mean_mahalanobis_server': max(
            r['gmm_mean_mahalanobis_server_max'] for r in rows),
        'server_gmm_mean_mahalanobis_mean': float(np.mean(server_mah)),
        'server_gmm_mean_mahalanobis_max': float(np.max(server_mah)),
        'clients': rows,
    }


def classification(server, models, macro_maps, loaders, seeds=INF_SEEDS):
    out = {name: {} for name in models}
    for seed in seeds:
        torch.manual_seed(int(seed))
        preds = {name: [] for name in models}
        ys = []
        with torch.no_grad():
            for loader in loaders:
                for x, y in loader:
                    mu, lv = server.encoder(x)
                    z = torch.randn_like(mu) * torch.exp(lv / 2) + mu
                    zn = z.numpy()
                    for name, model in models.items():
                        hard = core.model_resp(model, zn).argmax(1)
                        preds[name].append(macro_maps[name][hard])
                    ys.append(y.numpy())
        ytrue = np.concatenate(ys)
        for name in models:
            out[name][int(seed)] = cluster_metrics(
                ytrue, np.concatenate(preds[name]))
    return out


def generation(server, models, Xu, seeds=GEN_SEEDS, n=N_GEN):
    sigma = float(np.sqrt(np.median(pdist(Xu, metric='sqeuclidean'))))
    rng = np.random.default_rng(20261124)
    dirs = rng.normal(size=(Xu.shape[1], 64))
    dirs /= np.linalg.norm(dirs, axis=0, keepdims=True)
    out = {}
    for name, model in models.items():
        mmds, swds = [], []
        for seed in seeds:
            z, _ = core.sample_model(model, int(n), int(seed))
            with torch.no_grad():
                xg = server.decoder(torch.tensor(z, dtype=torch.float32)).numpy()
            mmds.append(math.sqrt(max(mmd2_rbf(xg, Xu, sigma), 0)))
            swds.append(swd(xg, Xu, dirs))
        out[name] = {
            'gen_seeds': [int(s) for s in seeds],
            'gen_n': int(n),
            'sigma': sigma,
            'gen_mmd_values': [float(x) for x in mmds],
            'gen_swd_values': [float(x) for x in swds],
            'gen_mmd_mean': float(np.mean(mmds)),
            'gen_mmd_sd': float(np.std(mmds, ddof=1)),
            'gen_swd_mean': float(np.mean(swds)),
            'gen_swd_sd': float(np.std(swds, ddof=1)),
        }
    return out


def per_micro(server, model, macro_map, Xu, y_unique, seeds=GEN_SEEDS):
    with torch.no_grad():
        zreal = server.encoder(torch.tensor(Xu, dtype=torch.float32))[0].numpy()
    resp = core.model_resp(model, zreal)
    hard = resp.argmax(1)
    weights = np.asarray(model['weights'], dtype=np.float64)
    sigma = float(np.sqrt(np.median(pdist(Xu, metric='sqeuclidean'))))
    rng = np.random.default_rng(20261124)
    dirs = rng.normal(size=(Xu.shape[1], 64))
    dirs /= np.linalg.norm(dirs, axis=0, keepdims=True)
    minority = int(np.argmin(np.bincount(y_unique)))
    rows = []
    for c in range(len(weights)):
        idx = np.flatnonzero(hard == c)
        support = int(len(idx))
        majority = int(np.bincount(y_unique[idx]).argmax()) if support else None
        row = {
            'micro': c, 'macro': int(macro_map[c]),
            'weight': float(weights[c]),
            'support': support,
            'empirical_mass': float(support / len(hard)),
            'majority_label': majority,
            'ghost': bool(support == 0),
            'low_mass': bool(weights[c] < 1.0 / (2 * len(weights))),
            'micro_mmd': None, 'micro_swd': None, 'self_rate': None,
        }
        if support >= 20:
            mmds, swds, rates = [], [], []
            for seed in seeds:
                zg, _ = core.sample_model(model, 500, int(seed), micro=c)
                with torch.no_grad():
                    xg = server.decoder(
                        torch.tensor(zg, dtype=torch.float32)).numpy()
                mmds.append(math.sqrt(max(mmd2_rbf(xg, Xu[idx], sigma), 0)))
                swds.append(swd(xg, Xu[idx], dirs))
                rates.append(float((core.model_resp(model, zg).argmax(1) == c).mean()))
            row['micro_mmd'] = float(np.mean(mmds))
            row['micro_swd'] = float(np.mean(swds))
            row['self_rate'] = float(np.mean(rates))
        rows.append(row)
    supported = [r for r in rows if r['micro_mmd'] is not None]
    nq = max(1, math.ceil(0.25 * len(supported))) if supported else 0
    rare = sorted(supported, key=lambda r: r['weight'])[:nq] if nq else []
    minority_micros = [r['micro'] for r in rows
                       if r['majority_label'] == minority]
    minority_mass = float(sum(weights[c] for c in minority_micros))
    minority_real_frac = float(np.mean(y_unique == minority))
    gen_fracs, gen_mmds, gen_swds = [], [], []
    real_minority = Xu[y_unique == minority]
    labels_by_micro = np.array([r['majority_label'] if r['majority_label'] is not None
                                else -1 for r in rows])
    for seed in seeds:
        zg, _ = core.sample_model(model, N_GEN, int(seed))
        hard_gen = core.model_resp(model, zg).argmax(1)
        gen_fracs.append(float(np.mean(labels_by_micro[hard_gen] == minority)))
        sel = np.isin(hard_gen, np.asarray(minority_micros, dtype=int))
        if int(sel.sum()) >= 20:
            with torch.no_grad():
                xg = server.decoder(
                    torch.tensor(zg[sel], dtype=torch.float32)).numpy()
            gen_mmds.append(math.sqrt(max(mmd2_rbf(xg, real_minority, sigma), 0)))
            gen_swds.append(swd(xg, real_minority, dirs))
    return {
        'rows': rows,
        'n_supported_20': int(sum(r['support'] >= 20 for r in rows)),
        'n_ghost': int(sum(r['support'] == 0 for r in rows)),
        'mass_unsupported': float(sum(weights[c] for c, r in enumerate(rows)
                                      if r['support'] < 20)),
        'rare_q25_n': int(len(rare)),
        'rare_q25_mass': float(sum(r['weight'] for r in rare)) if rare else 0.0,
        'rare_q25_mmd': float(np.mean([r['micro_mmd'] for r in rare])) if rare else None,
        'rare_q25_swd': float(np.mean([r['micro_swd'] for r in rare])) if rare else None,
        'minority_label': minority,
        'minority_real_frac': minority_real_frac,
        'minority_micros': [int(c) for c in minority_micros],
        'minority_mass': minority_mass,
        'minority_gen_frac_mean': float(np.mean(gen_fracs)),
        'minority_gen_frac_sd': float(np.std(gen_fracs, ddof=1)),
        'minority_mmd_mean': float(np.mean(gen_mmds)) if gen_mmds else None,
        'minority_mmd_sd': float(np.std(gen_mmds, ddof=1)) if len(gen_mmds) > 1 else None,
        'minority_swd_mean': float(np.mean(gen_swds)) if gen_swds else None,
        'minority_swd_sd': float(np.std(gen_swds, ddof=1)) if len(gen_swds) > 1 else None,
    }


def compact_model(model):
    return {
        'weights': [float(x) for x in model['weights']],
        'means': np.asarray(model['means']).tolist(),
        'log_sigma2': np.asarray(model['log_sigma2']).tolist(),
        'macro_of_micro': [int(x) for x in model['macro_of_micro']],
        'macro_weights': np.asarray(model['macro_weights']).tolist(),
        'macro_means': np.asarray(model['macro_means']).tolist(),
        'macro_log_sigma2': np.asarray(model['macro_log_sigma2']).tolist(),
        'n_micro': int(model['n_micro']), 'n_macro': int(model['n_macro']),
        'macro_allocation': model.get('macro_allocation'),
        'macro_sizes': model.get('macro_sizes'),
        'macro_masses': model.get('macro_masses'),
        'micro_members': model.get('micro_members'),
        'macro_events': model.get('macro_events'),
        'micro_events': model.get('micro_events'),
        'merge_events': model.get('merge_events'),
        'stop_info': model.get('stop_info'),
        'evaluation_macro_only': model.get('evaluation_macro_only', False),
        'adaptive': model.get('adaptive', False),
        'linkage': model.get('linkage'),
        'stop_threshold': model.get('stop_threshold'),
        'stop_pooled_median': model.get('stop_pooled_median'),
        'stop_pooled_mad': model.get('stop_pooled_mad'),
    }


def mean_sd(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def main():
    EXP.joinpath('results').mkdir(parents=True, exist_ok=True)
    status('verify', 'Verifying F01 checkpoint and loading frozen state')
    digest = sha256(CKPT)
    if digest != EXPECTED_SHA:
        raise RuntimeError(f'Checkpoint SHA256 mismatch: {digest}')
    ckpt = torch.load(CKPT, map_location='cpu')
    if int(ckpt['round']) != 100 or len(ckpt['clients']) != 10:
        raise RuntimeError('Unexpected checkpoint structure')
    server = make_server(ckpt['server'])
    clients = [make_client(st) for st in ckpt['clients']]
    client_states = ckpt['clients']
    sizes = np.asarray(ckpt['partition_sizes'], dtype=np.int64)

    _, loaders, (Xt, yt), sizes_author = make_noniid_clients(
        str(DATA), 10, 200, 'breast_cancer', seed=42)
    if list(map(int, sizes_author)) != sizes.tolist():
        raise RuntimeError('Partition sizes do not match checkpoint')
    Xu = Xt.numpy().astype(np.float64)
    y_unique = yt.numpy().astype(int)

    status('latent', 'Checking latent-space compatibility')
    compat = latent_compatibility(server, clients, client_states, Xu)
    atomic_json(EXP / 'results/latent_compatibility.json', compat)

    status('aggregate', 'Flattening the 10 real client local GMMs')
    components = core.flatten_clients(client_states, sizes)
    union = am.union_model(components)
    A = core.gmm_from_state(ckpt['server'])
    A['n_micro'] = int(len(A['weights']))
    A['n_macro'] = int(len(set(A['macro_of_micro'].tolist())))
    status('aggregate', 'Building adaptive Multi-CA Macro-Micro GMM')
    C = am.adaptive_hierarchical(components, k_macro=2, min_micro=2)
    k_leaf = int(C['n_micro'])
    status('aggregate', f'Building same-capacity flat GMM (K={k_leaf})')
    B = am.flat_fixed(components, target=k_leaf)
    if int(B['n_micro']) != k_leaf:
        raise RuntimeError('Flat/Multi-CA leaf count mismatch')

    models = {'A': A, 'B': B, 'C': C, 'union': union}
    b_part = sorted(tuple(sorted(x)) for x in B['micro_members'])
    c_part = sorted(tuple(sorted(x)) for x in C['micro_members'])
    b_c_equivalence = {
        'leaf_partitions_equal': bool(b_part == c_part),
        'weights_allclose': bool(np.allclose(B['weights'], C['weights'])),
        'means_allclose': bool(np.allclose(B['means'], C['means'])),
        'log_sigma2_allclose': bool(np.allclose(B['log_sigma2'], C['log_sigma2'])),
        'macro_map_equal': bool(np.array_equal(
            np.asarray(B['macro_of_micro']), np.asarray(C['macro_of_micro']))),
        'n_micro': int(k_leaf),
    }
    macro_maps = {
        'A': np.asarray(A['macro_of_micro'], dtype=int),
        'B': np.asarray(B['macro_of_micro'], dtype=int),
        'C': np.asarray(C['macro_of_micro'], dtype=int),
        'union': np.asarray(union['macro_of_micro'], dtype=int),
    }
    counts = {name: int(models[name]['n_micro']) for name in models}
    atomic_json(EXP / 'results/aggregation.json', {
        'checkpoint': {'path': str(CKPT), 'sha256': digest, 'round': 100,
                       'n_clients': 10, 'partition_sizes': sizes.tolist()},
        'components': {
            'weights': [float(x) for x in components['weights']],
            'means': components['means'].tolist(),
            'log_sigma2': components['log_sigma2'].tolist(),
            'component_clients': components['component_clients'].tolist(),
            'names': components['names'],
        },
        'models': {name: compact_model(models[name]) for name in models},
        'component_counts': counts,
        'adaptive_rule': ('complete-linkage normalized Bures merge; stop at '
                          'the largest relative jump in merge distance, '
                          'min 2 micros per macro'),
    })

    status('classification', 'Evaluating A/B/C/union on real client records',
           counts=counts)
    cls = classification(server, models, macro_maps, loaders, INF_SEEDS)

    # Validate A against the historical F01 evaluator per-seed values.
    hist_rows = []
    inf_path = CKPT.parent / 'inference.csv'
    if inf_path.exists():
        for row in csv.DictReader(inf_path.open()):
            if int(row['epoch']) == 100:
                hist_rows.append((int(row['inference_seed']), float(row['ari'])))
    hist_rows.sort()
    repro = [cls['A'][s]['ari'] for s in range(len(hist_rows))]
    match = (len(hist_rows) == len(repro) and
             max(abs(a - b) for a, b in zip(repro, [x[1] for x in hist_rows])) < 1e-9)

    with torch.no_grad():
        zmean = server.encoder(torch.tensor(Xu, dtype=torch.float32))[0].numpy()
    nll = {name: float(core.model_nll(models[name], zmean)) for name in models}

    status('generation', 'Generating from A/B/C/union with the frozen decoder',
           counts=counts)
    gen = generation(server, models, Xu)

    status('per-micro', 'Computing per-micro support, rare and minority diagnostics',
           counts=counts)
    micro = {name: per_micro(server, models[name], macro_maps[name], Xu, y_unique)
             for name in models}

    summary = {}
    for name in models:
        draws = [cls[name][s] for s in INF_SEEDS]
        ari02 = [cls[name][s]['ari'] for s in (0, 1, 2)]
        nmi02 = [cls[name][s]['nmi'] for s in (0, 1, 2)]
        pur02 = [cls[name][s]['purity'] for s in (0, 1, 2)]
        summary[name] = {
            'K': counts[name],
            'n_macro': int(models[name]['n_macro']),
            'ari_0_2_mean': mean_sd(ari02)[0], 'ari_0_2_sd': mean_sd(ari02)[1],
            'ari_0_9_mean': mean_sd([d['ari'] for d in draws])[0],
            'ari_0_9_sd': mean_sd([d['ari'] for d in draws])[1],
            'nmi_0_2_mean': mean_sd(nmi02)[0], 'nmi_0_2_sd': mean_sd(nmi02)[1],
            'purity_0_2_mean': mean_sd(pur02)[0],
            'purity_0_2_sd': mean_sd(pur02)[1],
            **gen[name], 'nll': nll[name],
            **{k: v for k, v in micro[name].items() if k != 'rows'},
        }

    a_to_bc = {}
    for key in ('ari_0_2_mean', 'ari_0_9_mean', 'nll',
                'gen_mmd_mean', 'gen_swd_mean',
                'minority_mmd_mean', 'minority_gen_frac_mean'):
        a_val = summary['A'].get(key)
        c_val = summary['C'].get(key)
        a_to_bc[key] = (None if a_val is None or c_val is None
                        else float(c_val - a_val))
    b_c_equivalence['metrics_identical'] = bool(
        abs(summary['B']['ari_0_9_mean'] - summary['C']['ari_0_9_mean']) < 1e-12
        and abs(summary['B']['nll'] - summary['C']['nll']) < 1e-12
        and abs(summary['B']['gen_mmd_mean'] - summary['C']['gen_mmd_mean']) < 1e-12
        and abs(summary['B']['gen_swd_mean'] - summary['C']['gen_swd_mean']) < 1e-12)
    metrics = {
        'checkpoint': {'path': str(CKPT), 'sha256': digest, 'round': 100},
        'latent_compatibility': compat,
        'component_counts': counts,
        'b_c_equivalence': b_c_equivalence,
        'a_to_bc_delta': a_to_bc,
        'classification': cls,
        'validation_A': {'historical_per_seed': hist_rows,
                         'reproduced_per_seed': repro,
                         'exact_1e-9': bool(match)},
        'nll': nll,
        'generation': gen,
        'per_model': {name: summary[name] for name in models},
        'micro': {name: micro[name] for name in models},
    }
    atomic_json(EXP / 'results/metrics.json', metrics)
    with (EXP / 'results/metrics.csv').open('w', newline='') as f:
        keys = ['model', 'K', 'n_macro', 'ari_0_2_mean', 'ari_0_2_sd',
                'ari_0_9_mean', 'nmi_0_2_mean', 'purity_0_2_mean', 'nll',
                'gen_mmd_mean', 'gen_mmd_sd', 'gen_swd_mean', 'gen_swd_sd',
                'n_supported_20', 'n_ghost', 'mass_unsupported',
                'minority_mass', 'minority_gen_frac_mean',
                'minority_mmd_mean', 'minority_swd_mean',
                'rare_q25_mass', 'rare_q25_mmd', 'rare_q25_swd']
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for name in models:
            row = {'model': name}
            row.update({k: summary[name].get(k) for k in keys if k != 'model'})
            w.writerow(row)
    with (EXP / 'results/per_micro.csv').open('w', newline='') as f:
        keys = ['model', 'micro', 'macro', 'weight', 'support', 'empirical_mass',
                'majority_label', 'ghost', 'low_mass', 'micro_mmd',
                'micro_swd', 'self_rate']
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for name in models:
            for row in micro[name]['rows']:
                item = {'model': name}
                item.update(row)
                w.writerow(item)
    atomic_json(EXP / 'results/merge_tree.json', {
        'adaptive': {'C': compact_model(C)},
        'flat': {'B': compact_model(B)},
        'union': {'union': compact_model(union)},
        'components': {
            'weights': [float(x) for x in components['weights']],
            'means': components['means'].tolist(),
            'log_sigma2': components['log_sigma2'].tolist(),
            'component_clients': components['component_clients'].tolist(),
            'names': components['names'],
        },
    })
    status('done', 'Frozen aggregation and evaluation complete', done=True,
           counts=counts, latest={name: {
               'ari_0_2': summary[name]['ari_0_2_mean'],
               'nll': summary[name]['nll'],
               'mmd': summary[name]['gen_mmd_mean']} for name in models})
    print(json.dumps({'counts': counts,
                      'A_validation_exact': match,
                      'B_C_equivalent': b_c_equivalence,
                      'summary': {name: {k: summary[name][k] for k in
                                         ('ari_0_2_mean', 'ari_0_9_mean',
                                          'nll', 'gen_mmd_mean', 'gen_swd_mean',
                                          'n_supported_20', 'n_ghost',
                                          'minority_gen_frac_mean')}
                                  for name in models}}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        Path(EXP / 'error.txt').write_text(traceback.format_exc())
        status('error', 'Run failed; see error.txt')
        raise
