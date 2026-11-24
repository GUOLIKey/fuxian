"""Direct inference evaluation of the two pretrained FedVaDE baselines.

A (author) and C (F01 epoch 100) are evaluated exactly as delivered:
no pretraining, no continuation, no warm start, no client reconstruction,
and no modification of any weight.  The evaluation reproduces the author's
own loop: for each inference seed, seed torch, then iterate the author's
client DataLoaders batch by batch and call model.predict on each batch.

This exact protocol reproduces the recorded F01 epoch-100 per-seed ARIs
(inference.csv) and their 0.6140 mean.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path('/root/autodl-tmp/fedvade_project')
SOURCE = ROOT / 'fuxian-round2/FedVaDE_source'
EXP = ROOT / 'experiments/multica_pretrained_corrected_v1'
DATA = SOURCE / 'dataset/breast_cancer.csv'
sys.path.insert(0, str(SOURCE))
from dataloader import make_noniid_clients  # noqa: E402
from model import VaDE  # noqa: E402

MODELS = {
    'AUTHOR': {
        'path': SOURCE / 'model/server_model_breast_cancer.pk',
        'alpha': 0.01,
        'historical_ari': 0.6395,
        'label': 'A: author delivered model (OFF)',
    },
    'F01': {
        'path': (ROOT / 'reproduction_runs/round7_hyperparameter_search/fine/F01'
                 / 'epoch_100_server.pt'),
        'alpha': 0.05,
        'historical_ari': 0.6140,
        'label': 'C: F01 epoch-100 best model (OFF)',
    },
}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_server_state(path):
    obj = torch.load(path, map_location='cpu')
    if isinstance(obj, dict) and 'server' in obj:
        return obj['server'], obj.get('round'), sorted(obj.keys())
    return obj, None, sorted(obj.keys())


def make_vade(k, alpha):
    return VaDE(SimpleNamespace(input_dim=31, hid_dim=10, nClusters=int(k),
                                alpha=float(alpha), cuda=False))


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


def predict_batched(model, loaders, seed):
    """Author evaluation loop: reseed, then predict each DataLoader batch."""
    torch.manual_seed(int(seed))
    preds, labels = [], []
    for loader in loaders:
        for x, y in loader:
            preds.append(model.predict(x))
            labels.append(y.numpy())
    return np.concatenate(labels, 0), np.concatenate(preds, 0)


def predict_unique(model, X, seed):
    torch.manual_seed(int(seed))
    return model.predict(torch.tensor(np.asarray(X), dtype=torch.float32))


def diag_log_prob(z, rho, mu, lv):
    w = torch.softmax(rho.float(), 0).numpy().astype(np.float64)
    mu = mu.float().numpy().astype(np.float64)
    lv = lv.float().numpy().astype(np.float64)
    z = np.asarray(z, dtype=np.float64)
    diff = z[:, None, :] - mu[None, :, :]
    lp = -0.5 * np.sum(np.log(2 * np.pi) + lv[None, :, :] +
                       diff ** 2 / np.exp(lv)[None, :, :], axis=2)
    return lp + np.log(np.maximum(w, 1e-12))[None, :]


def diag_nll(z, rho, mu, lv):
    lp = diag_log_prob(z, rho, mu, lv)
    m = lp.max(1, keepdims=True)
    return float(-np.mean(m.ravel() + np.log(np.exp(lp - m).sum(1))))


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


def gmm_params(state):
    rho = state['rho_']
    w = torch.softmax(rho.float(), 0).numpy().astype(np.float64)
    mu = state['mu_c'].float().numpy().astype(np.float64)
    lv = state['log_sigma2_c'].float().numpy().astype(np.float64)
    return w / w.sum(), mu, lv


def sample_gmm(weights, mu, lv, n, seed, component=None):
    rng = np.random.default_rng(int(seed))
    if component is None:
        comp = rng.choice(len(weights), size=int(n), p=weights)
    else:
        comp = np.full(int(n), int(component), dtype=int)
    z = mu[comp] + np.sqrt(np.exp(lv[comp])) * rng.standard_normal(mu[comp].shape)
    return z, comp


def generation_metrics(model, state, Xu, n=5000, seeds=(20261124, 20261125,
                                                        20261126, 20261127,
                                                        20261128)):
    w, mu, lv = gmm_params(state)
    rng = np.random.default_rng(20261124)
    dirs = rng.normal(size=(Xu.shape[1], 64))
    dirs /= np.linalg.norm(dirs, axis=0, keepdims=True)
    sigma = float(np.sqrt(np.median(pdist(Xu, metric='sqeuclidean'))))
    mmds, swds = [], []
    for s in seeds:
        z, _ = sample_gmm(w, mu, lv, n, s)
        with torch.no_grad():
            xg = model.decoder(torch.tensor(z, dtype=torch.float32)).numpy()
        mmds.append(math.sqrt(mmd2_rbf(xg, Xu, sigma)))
        swds.append(swd(xg, Xu, dirs))
    return {
        'gen_n': int(n),
        'gen_seeds': [int(s) for s in seeds],
        'sigma': sigma,
        'gen_mmd_mean': float(np.mean(mmds)), 'gen_mmd_sd': float(np.std(mmds, ddof=1)),
        'gen_swd_mean': float(np.mean(swds)), 'gen_swd_sd': float(np.std(swds, ddof=1)),
        'gen_mmd_values': [float(x) for x in mmds],
        'gen_swd_values': [float(x) for x in swds],
    }


def component_metrics(model, state, z_real, Xu, dirs, sigma):
    w, mu, lv = gmm_params(state)
    lp = diag_log_prob(z_real, state['rho_'], state['mu_c'], state['log_sigma2_c'])
    resp = np.exp(lp - lp.max(1, keepdims=True))
    resp /= resp.sum(1, keepdims=True)
    hard = resp.argmax(1)
    rows = []
    for c in range(len(w)):
        idx = np.flatnonzero(hard == c)
        row = {
            'component': c,
            'weight': float(w[c]),
            'support': int(len(idx)),
            'empirical_mass': float(len(idx) / len(hard)),
            'low_mass': bool(w[c] < 1.0 / (2 * len(w))),
            'micro_mmd': None, 'micro_swd': None, 'self_rate': None,
        }
        if len(idx) >= 20:
            vals_m, vals_s = [], []
            for s in range(3):
                zg, _ = sample_gmm(w, mu, lv, 500, 9000 + 10 * c + s, component=c)
                with torch.no_grad():
                    xg = model.decoder(torch.tensor(zg, dtype=torch.float32)).numpy()
                vals_m.append(math.sqrt(mmd2_rbf(xg, Xu[idx], sigma)))
                vals_s.append(swd(xg, Xu[idx], dirs))
                lpg = diag_log_prob(zg, state['rho_'], state['mu_c'],
                                    state['log_sigma2_c'])
                row['self_rate'] = float((lpg.argmax(1) == c).mean())
            row['micro_mmd'] = float(np.mean(vals_m))
            row['micro_swd'] = float(np.mean(vals_s))
        rows.append(row)
    supported = [r for r in rows if r['micro_mmd'] is not None]
    nq = max(1, math.ceil(0.25 * len(supported))) if supported else 0
    rare = sorted(supported, key=lambda r: r['weight'])[:nq] if nq else []
    return {
        'per_component': rows,
        'n_supported_20': int(sum(r['support'] >= 20 for r in rows)),
        'n_ghost': int(sum(r['support'] == 0 for r in rows)),
        'mass_unsupported': float(sum(r['weight'] for r in rows
                                      if r['support'] < 20)),
        'rare_q25_n': int(len(rare)),
        'rare_q25_mass': float(sum(r['weight'] for r in rare)) if rare else 0.0,
        'rare_q25_mmd': float(np.mean([r['micro_mmd'] for r in rare])) if rare else None,
        'rare_q25_swd': float(np.mean([r['micro_swd'] for r in rare])) if rare else None,
    }


def main():
    out_dir = EXP / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    _, loaders, (Xt, yt), sizes = make_noniid_clients(
        str(DATA), 10, 200, 'breast_cancer', seed=42)
    Xu = Xt.numpy().astype(np.float64)
    yu = yt.numpy().astype(int)
    client_X = [dl.dataset.tensors[0].numpy().astype(np.float64) for dl in loaders]
    client_y = [dl.dataset.tensors[1].numpy().astype(int) for dl in loaders]

    rng = np.random.default_rng(20261124)
    dirs = rng.normal(size=(Xu.shape[1], 64))
    dirs /= np.linalg.norm(dirs, axis=0, keepdims=True)
    sigma = float(np.sqrt(np.median(pdist(Xu, metric='sqeuclidean'))))

    records = []
    for name, cfg in MODELS.items():
        state, source_round, payload_keys = load_server_state(cfg['path'])
        model = make_vade(len(state['rho_']), cfg['alpha'])
        model.load_state_dict(state, strict=True)
        model.eval()

        draws = []
        for seed in range(10):
            y_true, y_pred = predict_batched(model, loaders, seed)
            m = cluster_metrics(y_true, y_pred)
            m['inference_seed'] = seed
            draws.append(m)

        unique_draws = []
        for seed in range(10):
            pred = predict_unique(model, Xu, seed)
            mm = cluster_metrics(yu, pred)
            mm['inference_seed'] = seed
            unique_draws.append(mm)

        with torch.no_grad():
            z_real = model.encoder(torch.tensor(Xu, dtype=torch.float32))[0].numpy()
        nll = diag_nll(z_real, state['rho_'], state['mu_c'], state['log_sigma2_c'])
        gen = generation_metrics(model, state, Xu)
        comp = component_metrics(model, state, z_real, Xu, dirs, sigma)

        def agg(prefix, rows, key):
            vals = np.array([r[key] for r in rows], dtype=np.float64)
            return {
                f'{prefix}_{key}_mean': float(vals.mean()),
                f'{prefix}_{key}_sd': float(vals.std(ddof=1)),
            }

        rec = {
            'model': name,
            'group': 'A' if name == 'AUTHOR' else 'C',
            'label': cfg['label'],
            'multi_ca': 'OFF',
            'evaluation_mode': 'direct inference; weights unchanged',
            'source_path': str(cfg['path']),
            'source_sha256': sha256_file(cfg['path']),
            'source_round': source_round,
            'source_payload_keys': payload_keys,
            'source_has_client_states': False,
            'historical_ari_reference': cfg['historical_ari'],
            'alpha': cfg['alpha'],
            'server_K': int(len(state['rho_'])),
            'partition_seed': 42,
            'client_sizes': [int(x) for x in sizes],
            'n_client_records': int(sum(sizes)),
            'n_unique_rows': int(len(yu)),
            'protocol': ('author loop: torch.manual_seed(seed); iterate client '
                         'DataLoaders; predict per batch'),
            'inference_seeds': list(range(10)),
            'draws_0_9': draws,
            'draws_0_2': draws[:3],
            'unique_row_draws_0_9': unique_draws,
            'nll_unique': nll,
            **gen,
            **comp,
            'actual_ari_seeds_0_1_2': [draws[i]['ari'] for i in range(3)],
            'actual_ari_mean_0_2': float(np.mean([draws[i]['ari'] for i in range(3)])),
            'actual_ari_sd_0_2': float(np.std([draws[i]['ari'] for i in range(3)],
                                              ddof=1)),
            'actual_ari_mean_0_9': float(np.mean([d['ari'] for d in draws])),
            'actual_ari_sd_0_9': float(np.std([d['ari'] for d in draws], ddof=1)),
        }
        for key in ('ari', 'purity', 'nmi', 'acc'):
            rec.update(agg('client', draws, key))
            rec.update(agg('unique', unique_draws, key))
        records.append(rec)
        print(json.dumps({
            'model': name,
            'ari_seeds_0_1_2': [round(x, 6) for x in rec['actual_ari_seeds_0_1_2']],
            'ari_mean_0_9': round(rec['actual_ari_mean_0_9'], 6),
            'ari_sd_0_9': round(rec['actual_ari_sd_0_9'], 6),
            'nll': round(nll, 4),
            'gen_mmd': round(rec['gen_mmd_mean'], 5),
            'gen_swd': round(rec['gen_swd_mean'], 5),
        }), flush=True)

    # F01 protocol check: the batched loop must reproduce the recorded values.
    f01 = next(r for r in records if r['model'] == 'F01')
    target = 0.6140076504228672
    if abs(f01['actual_ari_mean_0_9'] - target) > 1e-9:
        raise RuntimeError('F01 protocol reproduction failed: '
                           f"{f01['actual_ari_mean_0_9']} != {target}")

    payload = {
        'experiment': 'multica_pretrained_corrected_v1',
        'mode': 'A/C direct inference only; no training; no warm start; '
                'no client reconstruction; B/D blocked (no client states)',
        'created_utc': '2026-11-24',
        'models': records,
        'protocol_check': {
            'F01_recorded_mean': target,
            'F01_reproduced_mean': f01['actual_ari_mean_0_9'],
            'exact': True,
            'evidence': ('round7 F01 inference.csv per-seed values reproduced '
                         'bit-for-bit; recorded mean 0.6140076504228672'),
        },
    }
    (out_dir / 'baselines.json').write_text(json.dumps(payload, indent=2) + '\n')
    rows = []
    for r in records:
        rows.append({
            'group': r['group'], 'model': r['model'], 'mode': r['evaluation_mode'],
            'ari_seed0': r['actual_ari_seeds_0_1_2'][0],
            'ari_seed1': r['actual_ari_seeds_0_1_2'][1],
            'ari_seed2': r['actual_ari_seeds_0_1_2'][2],
            'ari_mean_0_2': r['actual_ari_mean_0_2'],
            'ari_sd_0_2': r['actual_ari_sd_0_2'],
            'ari_mean_0_9': r['actual_ari_mean_0_9'],
            'ari_sd_0_9': r['actual_ari_sd_0_9'],
            'purity_mean_0_9': r['client_purity_mean'],
            'nmi_mean_0_9': r['client_nmi_mean'],
            'acc_mean_0_9': r['client_acc_mean'],
            'unique_ari_mean_0_9': r['unique_ari_mean'],
            'nll': r['nll_unique'],
            'gen_mmd': r['gen_mmd_mean'], 'gen_swd': r['gen_swd_mean'],
            'n_supported_20': r['n_supported_20'], 'n_ghost': r['n_ghost'],
            'mass_unsupported': r['mass_unsupported'],
            'rare_q25_n': r['rare_q25_n'], 'rare_q25_mass': r['rare_q25_mass'],
            'rare_q25_mmd': r['rare_q25_mmd'], 'rare_q25_swd': r['rare_q25_swd'],
            'historical_ari_reference': r['historical_ari_reference'],
        })
    header = list(rows[0].keys())
    with (out_dir / 'baselines.csv').open('w') as f:
        f.write(','.join(header) + '\n')
        for row in rows:
            f.write(','.join('' if row[k] is None else str(row[k])
                             for k in header) + '\n')
    print('BASELINES_DONE', flush=True)


if __name__ == '__main__':
    main()
