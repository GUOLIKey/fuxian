"""Shared utilities for Round 11A frozen-model diagnostics.

All computations are read-only with respect to Round 1-10 artifacts.
No neural network training or gradient updates are performed here.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/root/autodl-tmp/fedvade_project')
BASE = ROOT / 'reproduction_runs/round11a_frozen_diagnostics'
R10 = ROOT / 'reproduction_runs/round10_non_iid_generation'
SOURCE = ROOT / 'fuxian-round2/FedVaDE_source'

for _p in (str(SOURCE), str(R10)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model import VaDE  # noqa: E402
from generation_tools import distribution_metrics, generate as generate_global  # noqa: E402

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

RARE = 2
CLASSES = (0, 1, 2)
DATA = None


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


def encode_sample(model, X, seed=0):
    mu, log_sigma2 = encode_mean(model, X)
    rng = np.random.default_rng(seed)
    eps = rng.normal(size=mu.shape)
    return mu + np.exp(0.5 * log_sigma2) * eps


def decode(model, Z):
    with torch.no_grad():
        return model.decoder(torch.tensor(np.asarray(Z), dtype=torch.float32)).numpy()


def softmax_weights(rho):
    return torch.softmax(torch.as_tensor(rho, dtype=torch.float32), 0).numpy()


def diag_log_prob(z, weights, means, log_sigma2):
    """Log mixture density for a diagonal-covariance GMM."""
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
            z[idx] = rng.multivariate_normal(means[k], covariances[k], size=len(idx))
    return z, labels


def generate_from_decoder(decoder_fn, z):
    with torch.no_grad():
        return decoder_fn(torch.tensor(np.asarray(z), dtype=torch.float32)).numpy()


def bootstrap_mean(values, n_boot=1000, seed=12345):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float('nan'), float('nan')
    return float(values.mean()), float(values.std(ddof=1) if len(values) > 1 else 0.0)


def cka_linear(X, Y):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    Kx = X @ X.T
    Ky = Y @ Y.T
    denom = np.sqrt((Kx * Kx).sum() * (Ky * Ky).sum())
    return float((Kx * Ky).sum() / denom) if denom > 0 else float('nan')


def knn_overlap(X, Y, k=10):
    from sklearn.neighbors import NearestNeighbors
    X = np.asarray(X)
    Y = np.asarray(Y)
    kx = min(k + 1, len(X))
    ky = min(k + 1, len(Y))
    nx = NearestNeighbors(n_neighbors=kx).fit(X).kneighbors(X)[1][:, 1:]
    ny = NearestNeighbors(n_neighbors=ky).fit(Y).kneighbors(Y)[1][:, 1:]
    scores = []
    for a, b in zip(nx, ny):
        sa, sb = set(a.tolist()), set(b.tolist())
        scores.append(len(sa & sb) / len(sa | sb))
    return float(np.mean(scores))


def cosine_mean(X, Y):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    num = (X * Y).sum(axis=1)
    den = np.linalg.norm(X, axis=1) * np.linalg.norm(Y, axis=1)
    return float(np.mean(num / np.maximum(den, 1e-12)))


def weight_entropy(weights):
    w = np.asarray(weights, dtype=np.float64)
    w = w / max(w.sum(), 1e-12)
    return float(-(w * np.log(np.maximum(w, 1e-12))).sum())


def effective_components(weights, threshold=0.01):
    return int(np.sum(np.asarray(weights) >= threshold))
