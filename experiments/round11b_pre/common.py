"""Shared utilities for Round 11B-pre federated GMM factor diagnostics.

No neural network training or gradient updates are performed.
All GMM fits use frozen latent representations or synthetic union samples.
"""
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/root/autodl-tmp/fedvade_project')
BASE = ROOT / 'reproduction_runs/round11b_pre_gmm_factorization'
R10 = ROOT / 'reproduction_runs/round10_non_iid_generation'
R11A = ROOT / 'reproduction_runs/round11a_frozen_diagnostics'
SOURCE = ROOT / 'fuxian-round2/FedVaDE_source'

for _p in (str(SOURCE), str(R10)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import VaDE  # noqa: E402
from generation_tools import distribution_metrics  # noqa: E402

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

RARE = 2
CLASSES = (0, 1, 2)
DATA = None

MODELS = [
    ('fedvade_severe_seed0', 'severe', 0),
    ('fedvade_severe_seed1', 'severe', 1),
    ('fedvade_severe_seed2', 'severe', 2),
]
CENTRALIZED = ('centralized', 'severe', 0)


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def update_progress(**changes):
    path = BASE / 'progress.json'
    current = json.loads(path.read_text()) if path.exists() else {}
    current.update(changes)
    current['updated_at'] = time.time()
    atomic_json(path, current)


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text('')
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def data():
    global DATA
    if DATA is None:
        DATA = np.load(R10 / 'private_data/data.npz')
    return DATA


def split_data():
    d = data()
    return (d['X_train'], d['y_train'], d['X_val'], d['y_val'],
            d['X_test'], d['y_test'])


def make_vade(n_clusters, input_dim=21, hid_dim=10):
    from argparse import Namespace
    return VaDE(Namespace(input_dim=input_dim, hid_dim=hid_dim,
                          nClusters=n_clusters, alpha=.01, cuda=False))


def load_server(path):
    payload = torch.load(path, map_location='cpu')
    state = payload.get('server', payload)
    model = make_vade(len(state['rho_']))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, payload, state


def load_client(state):
    model = make_vade(len(state['rho_']))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def encode_mean(model, X):
    with torch.no_grad():
        mu, log_sigma2 = model.encoder(torch.tensor(np.asarray(X), dtype=torch.float32))
    return mu.numpy(), log_sigma2.numpy()


def decode(model, Z):
    with torch.no_grad():
        return model.decoder(torch.tensor(np.asarray(Z), dtype=torch.float32)).numpy()


def softmax_weights(rho):
    return torch.softmax(torch.as_tensor(rho, dtype=torch.float32), 0).numpy()


def diag_log_prob(z, weights, means, log_sigma2):
    z = np.asarray(z, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    log_sigma2 = np.asarray(log_sigma2, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    diff = z[:, None, :] - means[None, :, :]
    logp = -0.5 * np.sum(np.log(2 * np.pi) + log_sigma2[None, :, :] +
                         diff ** 2 / np.exp(log_sigma2)[None, :, :], axis=2)
    return logp + np.log(np.maximum(weights, 1e-12))[None, :]


def diag_responsibilities(z, weights, means, log_sigma2):
    logp = diag_log_prob(z, weights, means, log_sigma2)
    m = logp.max(axis=1, keepdims=True)
    e = np.exp(logp - m)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-300)


def diag_nll(z, weights, means, log_sigma2):
    logp = diag_log_prob(z, weights, means, log_sigma2)
    m = logp.max(axis=1, keepdims=True)
    logsum = (m + np.log(np.exp(logp - m).sum(axis=1, keepdims=True))).ravel()
    return float(-logsum.mean())


def _safe_cholesky(cov, jitter=1e-6):
    cov = np.asarray(cov, dtype=np.float64)
    d = cov.shape[0]
    eps = jitter
    for _ in range(8):
        try:
            return np.linalg.cholesky(cov + eps * np.eye(d)), eps
        except np.linalg.LinAlgError:
            eps *= 10.0
    raise np.linalg.LinAlgError('Cholesky failed after jitter escalation')


def full_log_prob(z, weights, means, covariances, jitter=1e-6):
    z = np.asarray(z, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    covariances = np.asarray(covariances, dtype=np.float64)
    n, d = z.shape
    K = len(weights)
    logp = np.empty((n, K), dtype=np.float64)
    for k in range(K):
        L, eps = _safe_cholesky(covariances[k], jitter=jitter)
        diff = (z - means[k]).T
        sol = np.linalg.solve(L, diff)
        logdet = 2.0 * np.log(np.diag(L)).sum()
        logp[:, k] = -0.5 * (d * np.log(2 * np.pi) + logdet + np.sum(sol ** 2, axis=0))
        logp[:, k] += np.log(max(weights[k], 1e-12))
    return logp


def full_responsibilities(z, weights, means, covariances):
    logp = full_log_prob(z, weights, means, covariances)
    m = logp.max(axis=1, keepdims=True)
    e = np.exp(logp - m)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-300)


def full_nll(z, weights, means, covariances):
    logp = full_log_prob(z, weights, means, covariances)
    m = logp.max(axis=1, keepdims=True)
    logsum = (m + np.log(np.exp(logp - m).sum(axis=1, keepdims=True))).ravel()
    return float(-logsum.mean())


def sample_diag_gmm(weights, means, log_sigma2, n, seed, component=None):
    rng = np.random.default_rng(seed)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    means = np.asarray(means)
    log_sigma2 = np.asarray(log_sigma2)
    if component is None:
        labels = rng.choice(len(weights), size=n, p=weights)
    else:
        labels = np.full(n, int(component), dtype=int)
    z = means[labels] + rng.normal(size=(n, means.shape[1])) * np.sqrt(np.exp(log_sigma2[labels]))
    return z, labels


def sample_full_gmm(weights, means, covariances, n, seed, component=None):
    rng = np.random.default_rng(seed)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    means = np.asarray(means)
    covariances = np.asarray(covariances)
    if component is None:
        labels = rng.choice(len(weights), size=n, p=weights)
    else:
        labels = np.full(n, int(component), dtype=int)
    z = np.zeros((n, means.shape[1]), dtype=np.float64)
    for k in range(len(weights)):
        idx = np.flatnonzero(labels == k)
        if len(idx):
            L, _ = _safe_cholesky(covariances[k])
            z[idx] = means[k] + rng.standard_normal((len(idx), means.shape[1])) @ L.T
    return z, labels


def n_eff(responsibilities):
    resp = np.asarray(responsibilities, dtype=np.float64)
    s = resp.sum(axis=0)
    ss = (resp ** 2).sum(axis=0)
    return s ** 2 / np.maximum(ss, 1e-12)


def shrink_covariances(covariances, lambdas, eps=1e-6):
    cov = np.asarray(covariances, dtype=np.float64)
    lambdas = np.asarray(lambdas, dtype=np.float64)
    if lambdas.ndim == 0:
        lambdas = np.full(len(cov), float(lambdas))
    K, d, _ = cov.shape
    out = np.empty_like(cov)
    for k in range(K):
        diag = np.diag(np.diag(cov[k]))
        out[k] = (1.0 - lambdas[k]) * cov[k] + lambdas[k] * diag + eps * np.eye(d)
    return out


def cov_metrics(covariances, jitter=1e-6):
    cov = np.asarray(covariances, dtype=np.float64)
    if cov.ndim == 2:
        cov = np.stack([np.diag(c) for c in cov], axis=0)
    out = []
    for k in range(len(cov)):
        try:
            vals = np.linalg.eigvalsh(cov[k])
            min_eig = float(vals.min())
            max_eig = float(vals.max())
            cond = float(max_eig / max(min_eig, 1e-12))
            try:
                _safe_cholesky(cov[k], jitter=jitter)
                chol_ok = True
            except Exception:
                chol_ok = False
            out.append({'min_eig': min_eig, 'max_eig': max_eig,
                        'cond': cond, 'chol_ok': chol_ok})
        except Exception:
            out.append({'min_eig': float('nan'), 'max_eig': float('nan'),
                        'cond': float('inf'), 'chol_ok': False})
    return out


def bures_diag(mu1, var1, mu2, var2):
    mu1 = np.asarray(mu1, dtype=np.float64)
    mu2 = np.asarray(mu2, dtype=np.float64)
    var1 = np.asarray(var1, dtype=np.float64)
    var2 = np.asarray(var2, dtype=np.float64)
    mean_term = float(np.sum((mu1 - mu2) ** 2))
    scale_term = float(np.sum((np.sqrt(np.maximum(var1, 1e-12)) -
                               np.sqrt(np.maximum(var2, 1e-12))) ** 2))
    return float(np.sqrt(max(mean_term + scale_term, 0.0)))


def normalized_bw(value, var1, var2):
    scale1 = float(np.sqrt(np.mean(np.maximum(var1, 1e-12))))
    scale2 = float(np.sqrt(np.mean(np.maximum(var2, 1e-12))))
    return float(value / max(scale1 + scale2, 1e-12))


def agglomerative_groups(distance, n_clusters):
    from sklearn.cluster import AgglomerativeClustering
    model = AgglomerativeClustering(n_clusters=int(n_clusters),
                                    metric='precomputed', linkage='average')
    return model.fit_predict(np.asarray(distance))


def moment_match(indices, weights, means, log_sigma2):
    indices = np.asarray(indices, dtype=int)
    w = np.asarray(weights, dtype=np.float64)[indices]
    w = w / max(w.sum(), 1e-12)
    mu_j = np.asarray(means, dtype=np.float64)[indices]
    var_j = np.exp(np.asarray(log_sigma2, dtype=np.float64)[indices])
    mu = np.sum(w[:, None] * mu_j, axis=0)
    d = mu_j.shape[1]
    cov = np.zeros((d, d), dtype=np.float64)
    for j in range(len(indices)):
        diff = mu_j[j] - mu
        cov += w[j] * (np.diag(var_j[j]) + np.outer(diff, diff))
    return w.sum(), mu, cov


def load_union(model_name='fedvade_severe_seed0', scenario='severe', seed=0):
    run = R10 / 'formal' / f'{scenario}_seed{seed}'
    server, payload, state = load_server(run / 'selected_server.pt')
    lf = torch.load(run / 'last_full.pt', map_location='cpu')
    client_states = lf.get('clients', [])
    if len(client_states) != 3:
        raise RuntimeError(f'{model_name}: expected 3 clients')
    client_models = [load_client(s) for s in client_states]
    partitions = np.load(R10 / 'private_data' / f'{scenario}_partitions.npz')
    sizes = np.array([len(partitions[f'client_{i}']) for i in range(3)], dtype=float)
    client_weights = sizes / sizes.sum()
    weights, means, log_sigma2, owner = [], [], [], []
    for i, cs in enumerate(client_states):
        pi = softmax_weights(cs['rho_'])
        weights.append(client_weights[i] * pi)
        means.append(cs['mu_c'].numpy())
        log_sigma2.append(cs['log_sigma2_c'].numpy())
        owner.extend([i] * len(pi))
    weights = np.concatenate(weights)
    weights = weights / weights.sum()
    means = np.concatenate(means, axis=0)
    log_sigma2 = np.concatenate(log_sigma2, axis=0)
    owner = np.asarray(owner, dtype=int)
    return {
        'model_name': model_name, 'scenario': scenario, 'seed': seed,
        'server': server, 'server_state': state, 'payload': payload,
        'client_states': client_states, 'client_models': client_models,
        'sizes': sizes, 'client_weights': client_weights,
        'weights': weights, 'means': means, 'log_sigma2': log_sigma2,
        'owner': owner, 'variances': np.exp(log_sigma2),
    }


def sample_union(union, n, seed, component=None):
    return sample_diag_gmm(union['weights'], union['means'],
                           union['log_sigma2'], n, seed, component=component)


def gmm_params(gmm):
    weights = np.asarray(gmm.weights_, dtype=np.float64)
    means = np.asarray(gmm.means_, dtype=np.float64)
    if gmm.covariance_type == 'diag':
        covs = np.asarray(gmm.covariances_, dtype=np.float64)
        log_sigma2 = np.log(np.maximum(covs, 1e-12))
        return weights, means, covs, log_sigma2
    covs = np.asarray(gmm.covariances_, dtype=np.float64)
    return weights, means, covs, None


def sample_from_params(weights, means, covs, covariance_type, n, seed, component=None):
    if covariance_type == 'diag':
        return sample_diag_gmm(weights, means, np.log(np.maximum(covs, 1e-12)),
                               n, seed, component=component)
    return sample_full_gmm(weights, means, covs, n, seed, component=component)


def evaluate_generation(generated, clf, nn_rare, radius, tau,
                        Xval, yval, seed=12345):
    from evaluator import predict_all
    generated = np.asarray(generated)
    pred, p_rare, conf = predict_all(clf, generated)
    dist = nn_rare.kneighbors(generated)[0][:, 0]
    trusted = (p_rare >= tau) & (dist <= radius)
    result = {
        'rare_proxy': float(np.mean(trusted)),
        'assigned_rare_fraction': float(np.mean(pred == RARE)),
        'high_conf_rare_fraction': float(np.mean(p_rare >= tau)),
        'classifier_rare_prob_mean': float(np.mean(p_rare)),
        'distance_to_rare_mean': float(np.mean(dist)),
        'distance_pass_fraction': float(np.mean(dist <= radius)),
        'class2_count': int(np.sum(pred == RARE)),
        'mmd_class2': None,
        'swd_class2': None,
    }
    g2 = generated[pred == RARE]
    r2 = np.asarray(Xval)[np.asarray(yval) == RARE]
    if len(g2) >= 2 and len(r2) >= 2:
        d2 = distribution_metrics(r2, g2, seed=seed + 2)
        result['mmd_class2'] = d2.get('mmd_rbf')
        result['swd_class2'] = d2.get('sliced_wasserstein')
    d = distribution_metrics(Xval, generated, seed=seed)
    result['mmd_rbf'] = d.get('mmd_rbf')
    result['sliced_wasserstein'] = d.get('sliced_wasserstein')
    return result


def load_evaluator_config():
    cfg = json.loads((R11A / 'evaluator_config.json').read_text())
    return cfg


def fit_and_score_gmm(union, M, sample_seed, K, covariance_type, init_seed,
                      n_init=5, max_iter=500, reg_covar=1e-5,
                      means_init=None, weights_init=None, precisions_init=None):
    from sklearn.mixture import GaussianMixture
    z_fit, _ = sample_union(union, M, sample_seed)
    kwargs = {}
    if means_init is not None:
        kwargs['means_init'] = np.asarray(means_init)
    if weights_init is not None:
        kwargs['weights_init'] = np.asarray(weights_init)
    if precisions_init is not None:
        kwargs['precisions_init'] = np.asarray(precisions_init)
    gmm = GaussianMixture(
        n_components=K, covariance_type=covariance_type,
        random_state=init_seed, n_init=1 if kwargs else n_init,
        max_iter=max_iter, reg_covar=reg_covar, **kwargs,
    ).fit(z_fit)
    return gmm, z_fit


def score_params_on(model, Z, weights, means, covs, covariance_type):
    if covariance_type == 'diag':
        return float(-diag_nll(Z, weights, means, np.log(np.maximum(covs, 1e-12))))
    return float(-full_nll(Z, weights, means, covs))


def parse_float(value):
    try:
        if value in ('', None, 'None', 'nan'):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
