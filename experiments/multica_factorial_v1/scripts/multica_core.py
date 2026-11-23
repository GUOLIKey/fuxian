"""Capacity-matched server-side Multi-CA v1.

Core operations:
- diagonal Gaussian 2-Wasserstein distance;
- deterministic agglomerative component matching;
- weighted moment matching;
- optional adaptive merge stopping with a normalized-Bures guard;
- macro GMM sampling and client-decoder ownership mapping.

No labels are accepted by any merge function.  Stage A uses target_K=2 to keep
the same capacity as the author global GMM and avoid confounding Multi-CA with
an increase in component count.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

EPS = 1e-8
LOGVAR_MIN = -30.0
LOGVAR_MAX = 20.0


def _as_float_array(x, ndim, name):
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != ndim:
        raise ValueError(f'{name} must have ndim={ndim}, got {a.shape}')
    if not np.isfinite(a).all():
        raise ValueError(f'{name} contains NaN/Inf')
    return a


def sanitize_gmm(weights, means, log_sigma2, weight_floor=0.0):
    """Validate and normalize a diagonal GMM."""
    w = _as_float_array(weights, 1, 'weights')
    mu = _as_float_array(means, 2, 'means')
    lv = _as_float_array(log_sigma2, 2, 'log_sigma2')
    if len(w) != len(mu) or len(w) != len(lv):
        raise ValueError('weights/means/log_sigma2 length mismatch')
    if np.any(w < 0):
        raise ValueError('weights must be non-negative')
    if w.sum() <= 0:
        raise ValueError('weights sum must be positive')
    if weight_floor > 0:
        w = np.maximum(w, weight_floor)
    w = w / w.sum()
    lv = np.clip(lv, LOGVAR_MIN, LOGVAR_MAX)
    return w, mu, lv


def w2_diag(mu1, var1, mu2, var2):
    """Squared 2-Wasserstein distance between diagonal Gaussians."""
    mu1 = _as_float_array(mu1, 1, 'mu1')
    mu2 = _as_float_array(mu2, 1, 'mu2')
    var1 = _as_float_array(var1, 1, 'var1')
    var2 = _as_float_array(var2, 1, 'var2')
    if np.any(var1 < 0) or np.any(var2 < 0):
        raise ValueError('variances must be non-negative')
    return float(np.sum((mu1 - mu2) ** 2 +
                        (np.sqrt(np.maximum(var1, EPS)) -
                         np.sqrt(np.maximum(var2, EPS))) ** 2))


def normalized_bures_diag(mu1, var1, mu2, var2):
    """Scale-normalized Bures-Wasserstein distance (diagnostic/merge guard)."""
    raw = w2_diag(mu1, var1, mu2, var2)
    # Use the median standard deviation scale; stable when components are narrow.
    s1 = float(np.median(np.sqrt(np.maximum(var1, EPS))))
    s2 = float(np.median(np.sqrt(np.maximum(var2, EPS))))
    return float(math.sqrt(raw) / max(s1 + s2, EPS))


@dataclass
class Cluster:
    weight: float
    mean: np.ndarray
    var: np.ndarray
    members: list[int]
    owner_weights: dict[int, float]

    @property
    def log_sigma2(self):
        return np.log(np.maximum(self.var, EPS))


def _merge_clusters(a: Cluster, b: Cluster) -> Cluster:
    w = a.weight + b.weight
    if w <= 0:
        raise ValueError('merged weight must be positive')
    mu = (a.weight * a.mean + b.weight * b.mean) / w
    var = (a.weight * (a.var + (a.mean - mu) ** 2) +
           b.weight * (b.var + (b.mean - mu) ** 2)) / w
    owners = dict(a.owner_weights)
    for k, v in b.owner_weights.items():
        owners[k] = owners.get(k, 0.0) + v
    return Cluster(float(w), mu, np.maximum(var, EPS),
                   sorted(a.members + b.members), owners)


def _pairwise(clusters, linkage='complete', comp_mu=None, comp_var=None):
    n = len(clusters)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            if linkage == 'complete' and comp_mu is not None and comp_var is not None:
                raw = 0.0
                norm = 0.0
                for a in clusters[i].members:
                    for b in clusters[j].members:
                        d = w2_diag(comp_mu[a], comp_var[a], comp_mu[b], comp_var[b])
                        dn = normalized_bures_diag(comp_mu[a], comp_var[a],
                                                   comp_mu[b], comp_var[b])
                        raw = max(raw, d)
                        norm = max(norm, dn)
            else:
                raw = w2_diag(clusters[i].mean, clusters[i].var,
                              clusters[j].mean, clusters[j].var)
                norm = normalized_bures_diag(clusters[i].mean, clusters[i].var,
                                             clusters[j].mean, clusters[j].var)
            out.append((raw, norm, i, j))
    return out


def multi_ca_merge(weights, means, log_sigma2, component_clients,
                   target_components=2, max_components=2, min_components=2,
                   merge_threshold=1.0, allow_early_stop=False, linkage='complete'):
    """Agglomerative Multi-CA merge.

    Parameters are capacity-matched by default.  `merge_threshold` is in the
    normalized-Bures scale and is used only to flag forced merges; early stop is
    disabled when max_components == target_components.
    """
    w, mu, lv = sanitize_gmm(weights, means, log_sigma2)
    component_clients = np.asarray(component_clients, dtype=int)
    if len(component_clients) != len(w):
        raise ValueError('component_clients length mismatch')
    target = int(target_components)
    max_k = int(max_components)
    min_k = int(min_components)
    if target < min_k or max_k < target:
        raise ValueError('require min_components <= target_components <= max_components')
    if target < 1:
        raise ValueError('target_components must be >= 1')

    clusters = [
        Cluster(float(w[i]), mu[i].copy(), np.exp(lv[i]).copy(), [i],
                {int(component_clients[i]): float(w[i])})
        for i in range(len(w))
    ]
    events = []
    forced = 0
    stopped_early = False

    while len(clusters) > target:
        pairs = _pairwise(clusters, linkage=linkage, comp_mu=mu, comp_var=np.exp(lv))
        if not pairs:
            break
        # Deterministic: raw distance, normalized distance, then index pair.
        raw, norm, i, j = min(pairs, key=lambda x: (x[1], x[0], x[2], x[3]))
        if (allow_early_stop and len(clusters) <= max_k and
                norm > merge_threshold):
            stopped_early = True
            break
        a, b = clusters[i], clusters[j]
        merged = _merge_clusters(a, b)
        events.append({
            'step': len(events) + 1,
            'n_before': len(clusters),
            'cluster_i': i,
            'cluster_j': j,
            'members': merged.members.copy(),
            'weight': float(merged.weight),
            'raw_w2': float(raw),
            'normalized_bures': float(norm),
            'unsafe_merge': bool(norm > merge_threshold),
            'merged_mean': merged.mean.tolist(),
            'merged_var': merged.var.tolist(),
        })
        if norm > merge_threshold:
            forced += 1
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)]
        clusters.append(merged)
        # Keep a deterministic canonical order: by minimum member index.
        clusters.sort(key=lambda c: min(c.members))

    # If early stopping would leave more than the allowed capacity, continue
    # deterministically.  Stage A disables early stopping anyway.
    while len(clusters) > max_k:
        pairs = _pairwise(clusters, linkage=linkage, comp_mu=mu, comp_var=np.exp(lv))
        raw, norm, i, j = min(pairs, key=lambda x: (x[1], x[0], x[2], x[3]))
        merged = _merge_clusters(clusters[i], clusters[j])
        events.append({
            'step': len(events) + 1, 'n_before': len(clusters),
            'cluster_i': i, 'cluster_j': j, 'members': merged.members.copy(),
            'weight': float(merged.weight), 'raw_w2': float(raw),
            'normalized_bures': float(norm),
            'unsafe_merge': bool(norm > merge_threshold),
            'merged_mean': merged.mean.tolist(),
            'merged_var': merged.var.tolist(),
        })
        if norm > merge_threshold:
            forced += 1
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)]
        clusters.append(merged)
        clusters.sort(key=lambda c: min(c.members))

    if len(clusters) != target:
        # Only possible with early stopping.  Keep an explicit status rather
        # than silently pretending the target was reached.
        status = 'EARLY_STOPPED'
    else:
        status = 'TARGET_REACHED'

    macro_w = np.array([c.weight for c in clusters], dtype=np.float64)
    macro_w = macro_w / macro_w.sum()
    macro_mu = np.stack([c.mean for c in clusters], axis=0)
    macro_lv = np.stack([c.log_sigma2 for c in clusters], axis=0)
    owners = []
    for c in clusters:
        total = sum(c.owner_weights.values())
        if total <= 0:
            raise ValueError('macro component has no owner mass')
        ow = {int(k): float(v / total) for k, v in c.owner_weights.items()}
        dominant = min(ow, key=lambda k: (-ow[k], k))
        owners.append({
            'owner_weights': ow,
            'dominant_client': int(dominant),
            'dominant_fraction': float(ow[dominant]),
            'members': c.members.copy(),
        })
    return {
        'status': status,
        'weights': macro_w,
        'means': macro_mu,
        'log_sigma2': macro_lv,
        'owners': owners,
        'merge_events': events,
        'n_input_components': int(len(w)),
        'n_macro_components': int(len(clusters)),
        'n_merges': int(len(events)),
        'n_unsafe_merges': int(forced),
        'stopped_early': bool(stopped_early),
        'merge_threshold': float(merge_threshold),
        'max_merge_distance': float(max((e['raw_w2'] for e in events), default=0.0)),
        'max_merge_normalized_bures': float(max((e['normalized_bures'] for e in events), default=0.0)),
        'min_merge_normalized_bures': float(min((e['normalized_bures'] for e in events), default=0.0)),
        'target_components': int(target),
        'max_components': int(max_k),
        'min_components': int(min_k),
        'allow_early_stop': bool(allow_early_stop),
        'linkage': str(linkage),
    }


def sample_macro_gmm(macro, n, seed, component=None):
    """Sample from a macro GMM with a dedicated RNG."""
    rng = np.random.default_rng(int(seed))
    w = np.asarray(macro['weights'], dtype=np.float64)
    w = w / w.sum()
    mu = np.asarray(macro['means'], dtype=np.float64)
    var = np.exp(np.asarray(macro['log_sigma2'], dtype=np.float64))
    if component is None:
        labels = rng.choice(len(w), size=int(n), p=w)
    else:
        labels = np.full(int(n), int(component), dtype=int)
    z = mu[labels] + rng.normal(size=(int(n), mu.shape[1])) * np.sqrt(var[labels])
    return z, labels


def decoder_layout(macro, n_clients):
    """Map macro components to deterministic client-decoder blocks.

    The author server loop expects `lengths[client]` adjacent component blocks.
    A macro component may contain multiple clients; the dominant owner client is
    used as the decoder for that macro component.  This is logged and is the
    minimal compatible adaptation of the original loop.
    """
    owners = macro['owners']
    groups = sorted(range(len(owners)),
                    key=lambda g: (owners[g]['dominant_client'], min(owners[g]['members'])))
    lengths = [0] * int(n_clients)
    macro_to_block = {}
    block = 0
    for g in groups:
        client = int(owners[g]['dominant_client'])
        if client < 0 or client >= n_clients:
            raise ValueError(f'macro owner client out of range: {client}')
        lengths[client] += 1
        macro_to_block[g] = block
        block += 1
    if sum(lengths) != len(groups):
        raise RuntimeError('decoder layout length mismatch')
    return {
        'lengths': lengths,
        'macro_to_block': macro_to_block,
        'block_client': [int(owners[g]['dominant_client']) for g in groups],
        'dominant_fractions': [float(owners[g]['dominant_fraction']) for g in groups],
    }


def sample_macro_for_author_loop(macro, n, seed, n_clients):
    """Sample macro GMM and return author-loop-compatible labels/lengths."""
    z, macro_labels = sample_macro_gmm(macro, n, seed)
    layout = decoder_layout(macro, n_clients)
    block_labels = np.array([layout['macro_to_block'][int(k)] for k in macro_labels],
                            dtype=np.int64)
    return z, block_labels, layout


def macrogmm_to_loglik_terms(macro):
    w = np.asarray(macro['weights'], dtype=np.float64)
    mu = np.asarray(macro['means'], dtype=np.float64)
    lv = np.asarray(macro['log_sigma2'], dtype=np.float64)
    return w, mu, lv


def summary_for_log(macro):
    return {
        'n_input_components': int(macro['n_input_components']),
        'n_macro_components': int(macro['n_macro_components']),
        'n_merges': int(macro['n_merges']),
        'n_unsafe_merges': int(macro['n_unsafe_merges']),
        'stopped_early': bool(macro['stopped_early']),
        'max_merge_distance': float(macro['max_merge_distance']),
        'max_merge_normalized_bures': float(macro['max_merge_normalized_bures']),
        'macro_weights': [float(x) for x in macro['weights']],
        'dominant_clients': [int(x['dominant_client']) for x in macro['owners']],
        'dominant_fractions': [float(x['dominant_fraction']) for x in macro['owners']],
        'status': macro['status'],
        'linkage': macro.get('linkage'),
    }


def reconstruct_client_components(clients_gmm_param, sizes):
    """Flatten client GMM params into global component weights and metadata."""
    sizes = np.asarray(sizes, dtype=np.float64)
    client_weights = sizes / sizes.sum()
    weights, means, logvars, clients = [], [], [], []
    for i, sd in enumerate(clients_gmm_param):
        rho = np.asarray(sd['rho_'], dtype=np.float64)
        pi = np.exp(rho - np.max(rho))
        pi = pi / pi.sum()
        mu = np.asarray(sd['mu_c'], dtype=np.float64)
        lv = np.asarray(sd['log_sigma2_c'], dtype=np.float64)
        weights.append(client_weights[i] * pi)
        means.append(mu)
        logvars.append(lv)
        clients.extend([i] * len(pi))
    w = np.concatenate(weights)
    w = w / w.sum()
    return (w, np.concatenate(means, axis=0), np.concatenate(logvars, axis=0),
            np.asarray(clients, dtype=int), client_weights)


def plain_wasserstein_merge(weights, means, log_sigma2, component_clients,
                            n_clusters=2):
    """Plain average-linkage Wasserstein agglomeration + moment matching.

    This is the non-adaptive comparison arm.  It uses no labels and no merge
    guard; labels are not accepted by this function.
    """
    from sklearn.cluster import AgglomerativeClustering
    w, mu, lv = sanitize_gmm(weights, means, log_sigma2)
    component_clients = np.asarray(component_clients, dtype=int)
    n = len(w)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = w2_diag(mu[i], np.exp(lv[i]), mu[j], np.exp(lv[j]))
    labels = AgglomerativeClustering(
        n_clusters=int(n_clusters), metric='precomputed', linkage='average'
    ).fit_predict(D)
    clusters = []
    for g in sorted(set(labels.tolist())):
        idx = np.flatnonzero(labels == g).tolist()
        ww = w[idx]
        m = np.sum(ww[:, None] * mu[idx], axis=0) / ww.sum()
        v = np.zeros(mu.shape[1])
        for k in idx:
            v += w[k] * (np.exp(lv[k]) + (mu[k] - m) ** 2)
        v /= ww.sum()
        owners = {}
        for k in idx:
            c = int(component_clients[k])
            owners[c] = owners.get(c, 0.0) + float(w[k])
        total = sum(owners.values())
        clusters.append({
            'weight': float(ww.sum()),
            'mean': m,
            'var': np.maximum(v, EPS),
            'members': idx,
            'owner_weights': {k: v / total for k, v in owners.items()},
        })
    total = sum(c['weight'] for c in clusters)
    owners_out = []
    for c in clusters:
        dominant = min(c['owner_weights'], key=lambda k: (-c['owner_weights'][k], k))
        owners_out.append({
            'owner_weights': c['owner_weights'],
            'dominant_client': int(dominant),
            'dominant_fraction': float(c['owner_weights'][dominant]),
            'members': c['members'],
        })
    return {
        'status': 'PLAIN_AGGLOMERATIVE',
        'weights': np.array([c['weight'] / total for c in clusters]),
        'means': np.stack([c['mean'] for c in clusters], axis=0),
        'log_sigma2': np.stack([np.log(c['var']) for c in clusters], axis=0),
        'owners': owners_out,
        'merge_events': [],
        'n_input_components': int(n),
        'n_macro_components': int(len(clusters)),
        'n_merges': int(n - len(clusters)),
        'n_unsafe_merges': 0,
        'stopped_early': False,
        'merge_threshold': None,
        'max_merge_distance': None,
        'max_merge_normalized_bures': None,
        'min_merge_normalized_bures': None,
        'target_components': int(n_clusters),
        'max_components': int(n_clusters),
        'min_components': int(n_clusters),
        'allow_early_stop': False,
    }
