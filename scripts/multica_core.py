"""True Macro-Micro Multi-CA v2: frozen, label-free aggregation.

The global generative model is a two-level mixture:
  macro component m (K_macro=2) -> micro Gaussian k (K_micro selected by BIC)
The micro layer is retained in the global distribution.  No labels are accepted
by any merge/selection function; labels are used only by the evaluator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

EPS = 1e-8
LOGVAR_MIN = -30.0
LOGVAR_MAX = 20.0


@dataclass
class Cluster:
    weight: float
    mean: np.ndarray
    var: np.ndarray
    members: list[int]
    owner_weights: dict[int, float] = field(default_factory=dict)


def _np(x):
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def sanitize(weights, means, log_sigma2, normalize=True):
    w = _np(weights)
    mu = _np(means)
    lv = _np(log_sigma2)
    if w.ndim != 1 or mu.ndim != 2 or lv.ndim != 2:
        raise ValueError('bad GMM shapes')
    if len(w) != len(mu) or len(w) != len(lv):
        raise ValueError('GMM length mismatch')
    if not np.isfinite(w).all() or not np.isfinite(mu).all() or not np.isfinite(lv).all():
        raise ValueError('non-finite GMM parameter')
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError('invalid weights')
    if normalize:
        w = w / w.sum()
    lv = np.clip(lv, LOGVAR_MIN, LOGVAR_MAX)
    return w, mu, lv


def w2_diag(mu1, var1, mu2, var2):
    return float(np.sum((np.asarray(mu1) - np.asarray(mu2)) ** 2 +
                        (np.sqrt(np.maximum(var1, EPS)) -
                         np.sqrt(np.maximum(var2, EPS))) ** 2))


def normalized_bures(mu1, var1, mu2, var2):
    """Scale-normalized Bures-Wasserstein diagnostic used only as a merge guard."""
    raw = w2_diag(mu1, var1, mu2, var2)
    s1 = float(np.median(np.sqrt(np.maximum(var1, EPS))))
    s2 = float(np.median(np.sqrt(np.maximum(var2, EPS))))
    return float(math.sqrt(raw) / max(s1 + s2, EPS))


def merge_two(a: Cluster, b: Cluster) -> Cluster:
    w = a.weight + b.weight
    if w <= 0:
        raise ValueError('non-positive merged weight')
    mu = (a.weight * a.mean + b.weight * b.mean) / w
    var = (a.weight * (a.var + (a.mean - mu) ** 2) +
           b.weight * (b.var + (b.mean - mu) ** 2)) / w
    owners = dict(a.owner_weights)
    for k, v in b.owner_weights.items():
        owners[k] = owners.get(k, 0.0) + v
    return Cluster(float(w), mu, np.maximum(var, EPS),
                   sorted(a.members + b.members), owners)


def _pair_distance(a: Cluster, b: Cluster, mu, var, linkage):
    pairs = [(i, j) for i in a.members for j in b.members]
    vals = [w2_diag(mu[i], var[i], mu[j], var[j]) for i, j in pairs]
    norms = [normalized_bures(mu[i], var[i], mu[j], var[j]) for i, j in pairs]
    if linkage == 'complete':
        return float(max(vals)), float(max(norms))
    if linkage == 'average':
        return float(np.mean(vals)), float(np.mean(norms))
    if linkage == 'centroid':
        return (w2_diag(a.mean, a.var, b.mean, b.var),
                normalized_bures(a.mean, a.var, b.mean, b.var))
    raise ValueError(f'unknown linkage {linkage}')


def agglomerate(weights, means, log_sigma2, component_clients, target,
                linkage='complete', guard_threshold=1.0, use_guard=True,
                stage='micro', normalize=True):
    """Deterministic agglomerative merge to `target` clusters."""
    w, mu, lv = sanitize(weights, means, log_sigma2, normalize=normalize)
    component_clients = np.asarray(component_clients, dtype=int)
    if len(component_clients) != len(w):
        raise ValueError('component_clients length mismatch')
    target = int(target)
    if target < 1 or target > len(w):
        raise ValueError(f'invalid target {target} for {len(w)} components')
    var = np.exp(lv)
    clusters = [Cluster(float(w[i]), mu[i].copy(), var[i].copy(), [i],
                        {int(component_clients[i]): float(w[i])})
                for i in range(len(w))]
    events = []
    while len(clusters) > target:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                raw, norm = _pair_distance(clusters[i], clusters[j], mu, var, linkage)
                key = (norm if use_guard else raw, raw, i, j)
                if best is None or key < best[0]:
                    best = (key, i, j, raw, norm)
        _, i, j, raw, norm = best
        a, b = clusters[i], clusters[j]
        merged = merge_two(a, b)
        events.append({
            'stage': stage, 'n_before': len(clusters),
            'members': merged.members.copy(), 'weight': float(merged.weight),
            'raw_w2': float(raw), 'normalized_bures': float(norm),
            'unsafe': bool(norm > guard_threshold),
        })
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)]
        clusters.append(merged)
        clusters.sort(key=lambda c: min(c.members))
    return clusters, events


def allocate_micro_budget(macro_sizes, macro_masses, total_micro, min_per_macro=2):
    """Allocate a fixed total micro budget across macros; label-free."""
    sizes = [int(x) for x in macro_sizes]
    masses = np.asarray(macro_masses, dtype=np.float64)
    masses = masses / masses.sum()
    total = int(total_micro)
    if total > sum(sizes):
        total = sum(sizes)
    if total < len(sizes):
        raise ValueError('micro budget smaller than number of macros')
    alloc = []
    for n in sizes:
        if n < 1:
            raise ValueError('empty macro')
        alloc.append(min(min_per_macro, n))
    if sum(alloc) > total:
        # Fall back to one micro per macro, then fill deterministically.
        alloc = [1] * len(sizes)
    remaining = total - sum(alloc)
    while remaining > 0:
        candidates = [i for i, n in enumerate(sizes) if alloc[i] < n]
        if not candidates:
            break
        # Mass-proportional marginal allocation, deterministic tie-break.
        i = max(candidates, key=lambda k: (masses[k] / (alloc[k] + 1.0), -k))
        alloc[i] += 1
        remaining -= 1
    if sum(alloc) != total:
        raise ValueError(f'could not allocate micro budget: {sum(alloc)} != {total}')
    return alloc


def flatten_clients(clients, sizes):
    """Flatten existing client local GMMs. No fitting and no labels."""
    sizes = np.asarray(sizes, dtype=np.float64)
    cw = sizes / sizes.sum()
    weights, means, logvars, comp_clients, local_ids, names = [], [], [], [], [], []
    for c, sd in enumerate(clients):
        rho = _np(sd['rho_'])
        pi = np.exp(rho - np.max(rho)); pi = pi / pi.sum()
        mu = _np(sd['mu_c'])
        lv = _np(sd['log_sigma2_c'])
        for k in range(len(pi)):
            weights.append(float(cw[c] * pi[k]))
            means.append(mu[k]); logvars.append(lv[k])
            comp_clients.append(c); local_ids.append(k)
            names.append(f'c{c}_k{k}')
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return {
        'weights': w, 'means': np.stack(means), 'log_sigma2': np.stack(logvars),
        'component_clients': np.asarray(comp_clients, dtype=int),
        'local_ids': np.asarray(local_ids, dtype=int), 'names': names,
        'client_weights': cw,
    }


def _clusters_to_model(clusters, macro_ids, stage_events):
    weights = np.array([c.weight for c in clusters], dtype=np.float64)
    weights = weights / weights.sum()
    means = np.stack([c.mean for c in clusters])
    logvars = np.stack([np.log(np.maximum(c.var, EPS)) for c in clusters])
    owners = []
    for c in clusters:
        total = sum(c.owner_weights.values())
        if total <= 0:
            owners.append({'dominant_client': None, 'dominant_fraction': None,
                           'owner_weights': {}})
        else:
            ow = {int(k): float(v / total) for k, v in c.owner_weights.items()}
            dom = min(ow, key=lambda k: (-ow[k], k))
            owners.append({'dominant_client': int(dom),
                           'dominant_fraction': float(ow[dom]),
                           'owner_weights': ow})
    macro_ids = np.asarray(macro_ids, dtype=int)
    macro_weights, macro_means, macro_logvars = [], [], []
    for m in sorted(set(macro_ids.tolist())):
        idx = np.flatnonzero(macro_ids == m)
        ww = weights[idx]; ww = ww / ww.sum()
        mm = np.sum(ww[:, None] * means[idx], axis=0)
        vv = np.zeros(means.shape[1])
        for j in idx:
            vv += weights[j] * (np.exp(logvars[j]) + (means[j] - mm) ** 2)
        vv /= weights[idx].sum()
        macro_weights.append(float(weights[idx].sum()))
        macro_means.append(mm)
        macro_logvars.append(np.log(np.maximum(vv, EPS)))
    return {
        'weights': weights, 'means': means, 'log_sigma2': logvars,
        'macro_of_micro': macro_ids, 'macro_weights': np.asarray(macro_weights),
        'macro_means': np.stack(macro_means),
        'macro_log_sigma2': np.stack(macro_logvars),
        'owners': owners, 'merge_events': stage_events,
        'n_unsafe_merges': int(sum(e['unsafe'] for e in stage_events)),
        'n_merges': int(len(stage_events)),
    }


def build_hierarchical(components, total_micro, k_macro=2,
                       linkage='complete', guard_threshold=1.0, use_guard=True):
    """Build macro groups, then merge within each macro to its micro budget."""
    w = components['weights']; mu = components['means']
    lv = components['log_sigma2']; cc = components['component_clients']
    macro_clusters, macro_events = agglomerate(
        w, mu, lv, cc, target=k_macro, linkage=linkage,
        guard_threshold=guard_threshold, use_guard=use_guard, stage='macro')
    macro_clusters.sort(key=lambda c: min(c.members))
    sizes = [len(c.members) for c in macro_clusters]
    masses = [c.weight for c in macro_clusters]
    if total_micro < k_macro * 2:
        raise ValueError('true macro-micro requires at least 2 micros per macro')
    alloc = allocate_micro_budget(sizes, masses, total_micro, min_per_macro=2)
    tagged = []; micro_events = []
    for m, parent in enumerate(macro_clusters):
        idx = parent.members
        sub_w = w[idx]; sub_mu = mu[idx]; sub_lv = lv[idx]; sub_cc = cc[idx]
        sub_clusters, ev = agglomerate(
            sub_w, sub_mu, sub_lv, sub_cc, target=alloc[m],
            linkage=linkage, guard_threshold=guard_threshold,
            use_guard=use_guard, stage=f'micro_m{m}', normalize=False)
        # Rebase member ids to the full component set.
        for c in sub_clusters:
            c.members = [idx[j] for j in c.members]
        tagged.extend((c, m) for c in sub_clusters)
        micro_events.extend(ev)
    # Canonical order: source-member order; keep each micro's macro tag aligned.
    tagged.sort(key=lambda item: (min(item[0].members), item[0].weight))
    micro_clusters = [c for c, _ in tagged]
    macro_ids = [m for _, m in tagged]
    if len(macro_ids) != len(micro_clusters):
        raise RuntimeError('macro/micro membership inconsistency')
    model = _clusters_to_model(micro_clusters, macro_ids, macro_events + micro_events)
    model['macro_allocation'] = [int(a) for a in alloc]
    model['macro_sizes'] = [int(s) for s in sizes]
    model['macro_masses'] = [float(x) for x in masses]
    model['linkage'] = linkage
    model['guard_threshold'] = float(guard_threshold) if use_guard else None
    model['n_input_components'] = int(len(w))
    model['n_micro'] = int(len(micro_clusters))
    model['n_macro'] = int(len(macro_clusters))
    model['micro_members'] = [c.members for c in micro_clusters]
    return model


def make_flat_model(z, k, seed=0):
    """Global single-level diag GMM fitted by EM as a capacity control."""
    from sklearn.mixture import GaussianMixture
    z = np.asarray(z, dtype=np.float64)
    gm = GaussianMixture(n_components=int(k), covariance_type='diag',
                         random_state=int(seed), n_init=20, max_iter=500,
                         reg_covar=1e-6).fit(z)
    w = np.asarray(gm.weights_, dtype=np.float64)
    w = w / w.sum()
    mu = np.asarray(gm.means_, dtype=np.float64)
    cov = np.asarray(gm.covariances_, dtype=np.float64)
    return {
        'weights': w, 'means': mu,
        'log_sigma2': np.log(np.maximum(cov, EPS)),
        'macro_of_micro': None, 'macro_weights': None,
        'macro_means': None, 'macro_log_sigma2': None,
        'owners': [{'dominant_client': None, 'dominant_fraction': None,
                    'owner_weights': {}} for _ in range(len(w))],
        'merge_events': [], 'n_unsafe_merges': 0, 'n_merges': 0,
        'linkage': 'global_em', 'n_micro': int(len(w)), 'n_macro': 0,
        'bic': float(gm.bic(z)), 'aic': float(gm.aic(z)),
    }


def gmm_from_state(state):
    rho = _np(state['rho_'])
    w = np.exp(rho - np.max(rho))
    w = w / w.sum()
    mu = _np(state['mu_c'])
    lv = _np(state['log_sigma2_c'])
    return {
        'weights': w,
        'means': mu,
        'log_sigma2': lv,
        'macro_of_micro': np.arange(len(w)), 'macro_weights': w.copy(),
        'macro_means': mu.copy(),
        'macro_log_sigma2': lv.copy(),
        'owners': [{'dominant_client': None, 'dominant_fraction': None,
                    'owner_weights': {}} for _ in range(len(w))],
        'merge_events': [], 'n_unsafe_merges': 0, 'n_merges': 0,
        'linkage': 'server_gmm', 'n_micro': int(len(w)), 'n_macro': int(len(w)),
    }


def eval_macro_grouping(model):
    """Evaluation-only macro grouping for models without a macro layer (Flat)."""
    if model.get('macro_of_micro') is not None:
        return np.asarray(model['macro_of_micro'], dtype=int)
    from sklearn.cluster import AgglomerativeClustering
    n = len(model['weights'])
    D = np.zeros((n, n))
    var = np.exp(model['log_sigma2'])
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = w2_diag(model['means'][i], var[i],
                                        model['means'][j], var[j])
    return AgglomerativeClustering(n_clusters=2, metric='precomputed',
                                   linkage='average').fit_predict(D)


def model_log_prob(model, z, micro=None, macro=None):
    z = np.asarray(z, dtype=np.float64)
    idx = np.arange(len(model['weights']))
    if micro is not None:
        idx = np.asarray([int(micro)], dtype=int)
    elif macro is not None:
        idx = np.flatnonzero(np.asarray(model['macro_of_micro']) == int(macro))
    w = model['weights'][idx]; w = w / w.sum()
    mu = model['means'][idx]; lv = model['log_sigma2'][idx]
    diff = z[:, None, :] - mu[None, :, :]
    lp = -0.5 * np.sum(np.log(2 * np.pi) + lv[None, :, :] +
                       diff ** 2 / np.exp(lv)[None, :, :], axis=2)
    lp += np.log(np.maximum(w, 1e-12))[None, :]
    return lp, idx


def model_nll(model, z, micro=None, macro=None):
    lp, _ = model_log_prob(model, z, micro=micro, macro=macro)
    m = lp.max(1, keepdims=True)
    return float(-np.mean(m.ravel() + np.log(np.exp(lp - m).sum(1))))


def model_resp(model, z):
    lp, idx = model_log_prob(model, z)
    m = lp.max(1, keepdims=True)
    p = np.exp(lp - m); p /= p.sum(1, keepdims=True)
    full = np.zeros((len(z), len(model['weights'])))
    full[:, idx] = p
    return full


def sample_model(model, n, seed, micro=None, macro=None):
    rng = np.random.default_rng(int(seed))
    idx = np.arange(len(model['weights']))
    if micro is not None:
        idx = np.asarray([int(micro)], dtype=int)
    elif macro is not None:
        if model.get('macro_of_micro') is None:
            raise ValueError('model has no macro layer')
        idx = np.flatnonzero(np.asarray(model['macro_of_micro']) == int(macro))
    if len(idx) == 0:
        raise ValueError('empty component selection')
    w = np.asarray(model['weights'][idx], dtype=np.float64)
    w = w / w.sum()
    comp = rng.choice(idx, size=int(n), p=w)
    mu = model['means'][comp]
    sd = np.sqrt(np.exp(model['log_sigma2'][comp]))
    z = mu + sd * rng.standard_normal(mu.shape)
    return z, comp


def macro_weights(model):
    if model.get('macro_of_micro') is None:
        return None
    ids = np.asarray(model['macro_of_micro'], dtype=int)
    return np.array([model['weights'][ids == m].sum()
                     for m in sorted(set(ids.tolist()))])
