"""Read-only metrics for Multi-CA training and analysis."""
from __future__ import annotations

import json

import numpy as np


def diag_log_prob(z, weights, means, log_sigma2):
    z = np.asarray(z, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    log_sigma2 = np.asarray(log_sigma2, dtype=np.float64)
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


def clustering_metrics(y_true, y_pred):
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    purity = 0.0
    for c in np.unique(y_pred):
        counts = np.bincount(y_true[y_pred == c])
        purity += float(counts.max())
    purity /= max(n, 1)
    D = int(max(y_true.max(), y_pred.max())) + 1
    conf = np.zeros((D, D), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        conf[int(t), int(p)] += 1
    row, col = linear_sum_assignment(-conf)
    acc = float(conf[row, col].sum()) / max(n, 1)
    return {
        'n': int(n),
        'ari': float(adjusted_rand_score(y_true, y_pred)),
        'purity': float(purity),
        'acc': acc,
        'nmi': float(normalized_mutual_info_score(y_true, y_pred)),
    }


def gmm_state_metrics(model, z, y):
    import torch
    w = torch.softmax(model.rho_.detach().cpu(), 0).numpy()
    mu = model.mu_c.detach().cpu().numpy()
    lv = model.log_sigma2_c.detach().cpu().numpy()
    pred = np.argmax(diag_responsibilities(z, w, mu, lv), axis=1)
    return clustering_metrics(y, pred), diag_nll(z, w, mu, lv)


def encode_mean(model, X):
    import torch
    with torch.no_grad():
        mu, _ = model.encoder(torch.tensor(np.asarray(X), dtype=torch.float32))
    return mu.cpu().numpy()


def state_jsonable(obj):
    return json.loads(json.dumps(obj))


def distribution_metrics(real, generated, seed=123, subsample=400):
    from scipy.stats import wasserstein_distance
    from sklearn.metrics import pairwise_distances
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed)
    real = np.asarray(real, dtype=float)
    generated = np.asarray(generated, dtype=float)
    if len(real) < 2 or len(generated) < 2:
        return {'mmd_rbf': None, 'sliced_wasserstein': None,
                'generated_to_real_nn_mean': None}
    a = real[rng.choice(len(real), min(len(real), subsample), replace=False)]
    b = generated[rng.choice(len(generated), min(len(generated), subsample), replace=False)]
    aa = pairwise_distances(a, squared=True)
    bb = pairwise_distances(b, squared=True)
    ab = pairwise_distances(a, b, squared=True)
    upper = aa[np.triu_indices(len(a), 1)]
    gamma = max(float(np.median(upper[upper > 0])) if np.any(upper > 0) else 1.0, 1e-6)
    mmd = float(np.exp(-aa / gamma).mean() + np.exp(-bb / gamma).mean()
                - 2 * np.exp(-ab / gamma).mean())
    directions = rng.normal(size=(64, a.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    swd = float(np.mean([wasserstein_distance(a @ v, b @ v) for v in directions]))
    neighbor = NearestNeighbors(n_neighbors=1).fit(real)
    distance = neighbor.kneighbors(generated, return_distance=True)[0][:, 0]
    return {'mmd_rbf': mmd, 'sliced_wasserstein': swd,
            'generated_to_real_nn_mean': float(distance.mean())}
