"""Frozen ACD + FedVaDE local-GMM parameter adaptation experiment.

Loads the real F01 epoch-100 full federated checkpoint and compares six local
fitting methods per client:

  A0  original local GMM from the checkpoint
  A1  original K + ordinary GaussianMixture refit
  A2  BIC search, original range (2..14, capped by n_fit//5)
  A3  BIC search, extended range (2..30, capped by n_fit//5)
  A4  author ACD (Fed-HIRE FCPL) + moment-matched GMM parameters, no EM
  A5  ordinary GaussianMixture with exactly A4's K

No FedVaDE training, no client reconstruction, no ACD algorithm changes.
Labels are used only for post-hoc evaluation and component majority mapping.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture

ROOT = Path('/root/autodl-tmp/fedvade_project')
SOURCE = ROOT / 'fuxian-round2/FedVaDE_source'
CKPT = (ROOT / 'reproduction_runs/round7_f01_full_repro/fine/F01'
        / 'epoch_100_full.pt')
EXP = ROOT / 'experiments/acd_f01_frozen_v1'
VENDOR = EXP / 'vendor/FedHIRE'
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(VENDOR))
from model import VaDE  # noqa: E402
from MultiClustering import FCPL  # noqa: E402

EXPECTED_SHA = '3be341c451a186d1f36673080164af208993d215f0c2cb03709407892f7d6f3e'
ACD_RATE = 0.05
ACD_K0 = 0.5
REG_COVAR = 1e-6
PI_EPS = 1e-10
SEEDS = (0, 1, 2)
HOLDOUT_SEED = 20261125
HOLDOUT_FRAC = 0.2
N_GEN = 2000
DIV_N = 500
BIC_ORIG_MAX = 14
BIC_EXT_MAX = 30
MIN_SAMPLES_PER_COMP = 5
METHODS = ('A0', 'A1', 'A2', 'A3', 'A4', 'A5')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _json_default(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


def atomic_json(path, obj):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default) + '\n')
    tmp.replace(path)


def append_jsonl(path, obj):
    with Path(path).open('a') as f:
        f.write(json.dumps(obj, default=_json_default) + '\n')


def status(stage, message, done=False, **extra):
    payload = {'stage': stage, 'message': message, 'done': bool(done),
               'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    payload.update(extra)
    atomic_json(EXP / 'status.json', payload)


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


def diag_log_prob(z, pi, mu, var):
    z = np.asarray(z, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    var = np.maximum(np.asarray(var, dtype=np.float64), 1e-12)
    diff = z[:, None, :] - mu[None, :, :]
    lp = -0.5 * np.sum(np.log(2 * np.pi) + np.log(var)[None, :, :] +
                       diff ** 2 / var[None, :, :], axis=2)
    return lp + np.log(np.maximum(pi, 1e-12))[None, :]


def nll_params(z, params):
    lp = diag_log_prob(z, params['pi'], params['mu'], params['var'])
    m = lp.max(1, keepdims=True)
    return float(-np.mean(m.ravel() + np.log(np.exp(lp - m).sum(1))))


def assign_params(z, params):
    return np.argmax(diag_log_prob(z, params['pi'], params['mu'],
                                   params['var']), axis=1)


def sample_params(pi, mu, var, n, seed):
    rng = np.random.default_rng(int(seed))
    pi = np.asarray(pi, dtype=np.float64)
    pi = pi / pi.sum()
    comp = rng.choice(len(pi), size=int(n), p=pi)
    z = np.asarray(mu)[comp] + np.sqrt(np.asarray(var)[comp]) * \
        rng.standard_normal(np.asarray(mu)[comp].shape)
    return z, comp


def params_from_gmm(gm):
    pi = np.asarray(gm.weights_, dtype=np.float64)
    mu = np.asarray(gm.means_, dtype=np.float64)
    var = np.asarray(gm.covariances_, dtype=np.float64)
    if var.ndim == 1:
        var = var[:, None]
    var = np.maximum(var, REG_COVAR)
    return {'pi': pi / pi.sum(), 'mu': mu, 'var': var,
            'logvar': np.log(var), 'rho': np.log(pi / pi.sum() + PI_EPS),
            'reg_covar': REG_COVAR, 'source': 'sklearn_gmm'}


def params_from_labels(z, labels):
    labels = np.asarray(labels, dtype=int)
    if labels.min() < 0:
        raise ValueError('ACD returned unassigned points')
    K = int(labels.max()) + 1
    N = len(z)
    counts = np.bincount(labels, minlength=K).astype(np.float64)
    if np.any(counts <= 0):
        raise ValueError('empty ACD cluster')
    pi = counts / N
    mu = np.stack([np.asarray(z)[labels == k].mean(0) for k in range(K)])
    var = np.stack([np.asarray(z)[labels == k].var(0, ddof=0)
                    for k in range(K)])
    var = np.maximum(var, REG_COVAR)
    return {'pi': pi, 'mu': mu, 'var': var, 'logvar': np.log(var),
            'rho': np.log(pi + PI_EPS), 'reg_covar': REG_COVAR,
            'source': 'acd_moments'}


def params_from_original(state):
    rho = state['rho_'].detach().float().cpu().numpy()
    pi = np.exp(rho - rho.max())
    pi = pi / pi.sum()
    mu = state['mu_c'].detach().float().cpu().numpy()
    var = np.exp(state['log_sigma2_c'].detach().float().cpu().numpy())
    var = np.maximum(var, REG_COVAR)
    return {'pi': pi, 'mu': mu, 'var': var, 'logvar': np.log(var),
            'rho': np.log(pi + PI_EPS), 'reg_covar': REG_COVAR,
            'source': 'original_local_gmm'}


def params_to_json(p):
    return {'K': int(len(p['pi'])), 'pi': np.asarray(p['pi']).tolist(),
            'mu': np.asarray(p['mu']).tolist(),
            'var': np.asarray(p['var']).tolist(),
            'logvar': np.asarray(p['logvar']).tolist(),
            'rho': np.asarray(p['rho']).tolist(),
            'reg_covar': p.get('reg_covar'), 'source': p.get('source')}


def fit_gmm(z, K, seed):
    gm = GaussianMixture(n_components=int(K), covariance_type='diag',
                         random_state=int(seed), n_init=1, max_iter=100,
                         reg_covar=REG_COVAR)
    gm.fit(np.asarray(z, dtype=np.float64))
    return gm


def bic_search(z, seed, k_max):
    rows = []
    best = None
    for K in range(2, int(k_max) + 1):
        try:
            gm = fit_gmm(z, K, seed)
            bic = float(gm.bic(np.asarray(z, dtype=np.float64)))
            if not np.isfinite(bic):
                raise ValueError('non-finite BIC')
            rows.append({'K': K, 'bic': bic, 'success': True,
                         'converged': bool(gm.converged_),
                         'n_iter': int(gm.n_iter_)})
            if best is None or (bic, K) < (best['bic'], best['K']):
                best = {'K': K, 'bic': bic, 'gm': gm}
        except Exception as exc:
            rows.append({'K': K, 'success': False, 'error': repr(exc)})
    if best is None:
        raise RuntimeError('no BIC candidate fitted')
    return best, rows


def acd_run(z, seed):
    theta, label, K = FCPL(np.asarray(z, dtype=np.float64), ACD_RATE,
                           ACD_K0, random_state=int(seed))
    label = np.asarray(label, dtype=int)
    theta = np.asarray(theta, dtype=np.float64)
    K = int(K)
    if K < 1 or len(theta) != K:
        raise ValueError(f'invalid ACD K/theta: {K}/{len(theta)}')
    if label.min() < 0 or label.max() >= K:
        raise ValueError('invalid ACD label range')
    counts = np.bincount(label, minlength=K)
    if np.any(counts <= 0):
        raise ValueError('ACD produced an empty cluster')
    if not np.isfinite(theta).all():
        raise ValueError('non-finite ACD centers')
    return label, K, theta


def check_interface(client_model, params, x_batch, seed=0):
    K = int(len(params['pi']))
    state = {k: v.detach().clone() for k, v in
             client_model.state_dict().items()}
    state['rho_'] = torch.tensor(np.asarray(params['rho']),
                                 dtype=torch.float32)
    state['mu_c'] = torch.tensor(np.asarray(params['mu']),
                                 dtype=torch.float32)
    state['log_sigma2_c'] = torch.tensor(np.asarray(params['logvar']),
                                         dtype=torch.float32)
    model = VaDE(SimpleNamespace(input_dim=31, hid_dim=10, nClusters=K,
                                 alpha=0.05, cuda=False))
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        pred = model.predict(torch.tensor(np.asarray(x_batch, dtype=np.float32)))
        z, _ = sample_params(params['pi'], params['mu'], params['var'], 10, seed)
        xgen = model.decoder(torch.tensor(z, dtype=torch.float32)).numpy()
    ok = (np.isfinite(params['pi']).all() and np.isfinite(params['mu']).all()
          and np.isfinite(params['var']).all() and np.all(params['var'] > 0)
          and abs(float(params['pi'].sum()) - 1.0) < 1e-6
          and np.isfinite(z).all() and np.isfinite(xgen).all()
          and int(pred.min()) >= 0 and int(pred.max()) < K)
    return {'ok': bool(ok), 'K': K, 'pred_min': int(pred.min()),
            'pred_max': int(pred.max()), 'z_shape': list(z.shape),
            'x_shape': list(xgen.shape), 'finite': bool(np.isfinite(xgen).all())}


def split_indices(n, client_id):
    rng = np.random.default_rng(HOLDOUT_SEED + int(client_id))
    perm = rng.permutation(int(n))
    n_hold = max(2, int(round(HOLDOUT_FRAC * n)))
    n_hold = min(n_hold, n - 2)
    hold = np.sort(perm[:n_hold])
    fit = np.sort(perm[n_hold:])
    digest = hashlib.sha256(fit.tobytes() + hold.tobytes()).hexdigest()[:16]
    return fit, hold, digest


def full_metrics(y, pred, pi):
    pred = np.asarray(pred, dtype=int)
    K = int(len(pi))
    support = np.bincount(pred, minlength=K).astype(np.int64)
    low5 = (support > 0) & (support < 5)
    low20 = (support > 0) & (support < 20)
    ghost = support == 0
    m = cluster_metrics(y, pred)
    return {
        'K': K, **m,
        'support': support.tolist(),
        'n_supported_5': int(np.sum(support >= 5)),
        'n_supported_20': int(np.sum(support >= 20)),
        'n_low5': int(low5.sum()),
        'n_low20': int(low20.sum()),
        'n_ghost': int(ghost.sum()),
        'mass_low5': float(support[low5].sum() / len(pred)),
        'mass_low20': float(support[low20].sum() / len(pred)),
        'pi_low5': float(np.asarray(pi)[low5].sum()),
        'pi_low20': float(np.asarray(pi)[low20].sum()),
    }


def generation_metrics(client_model, X_real, y_real, params, pred_full, seed,
                       sigma, dirs, n=N_GEN):
    K = int(len(params['pi']))
    z, comp = sample_params(params['pi'], params['mu'], params['var'], n, seed)
    with torch.no_grad():
        xg = client_model.decoder(
            torch.tensor(z, dtype=torch.float32)).numpy()
    mmd = math.sqrt(max(mmd2_rbf(xg, X_real, sigma), 0.0))
    sw = swd(xg, X_real, dirs)
    pred_full = np.asarray(pred_full, dtype=int)
    support = np.bincount(pred_full, minlength=K)
    majority = []
    for k in range(K):
        idx = np.flatnonzero(pred_full == k)
        majority.append(int(np.bincount(np.asarray(y_real)[idx]).argmax())
                        if len(idx) else None)
    classes, counts = np.unique(y_real, return_counts=True)
    minority = int(classes[np.argmin(counts)]) if len(classes) > 1 else None
    minority_comps = ([k for k in range(K) if majority[k] == minority]
                      if minority is not None else [])
    minority_sel = np.isin(comp, np.asarray(minority_comps, dtype=int))
    minority_gen_frac = (float(minority_sel.mean())
                         if minority is not None else None)
    minority_mass = (float(np.asarray(params['pi'])[minority_comps].sum())
                     if minority is not None else None)
    min_mmd = min_sw = None
    if minority is not None:
        real_min = np.asarray(X_real)[np.asarray(y_real) == minority]
        gen_min = xg[minority_sel]
        if len(real_min) >= 20 and len(gen_min) >= 20:
            min_mmd = math.sqrt(max(mmd2_rbf(gen_min, real_min, sigma), 0.0))
            min_sw = swd(gen_min, real_min, dirs)
    base = xg[:min(DIV_N, len(xg))]
    div_mean = float(np.mean(pdist(base))) if len(base) > 1 else 0.0
    hist = np.bincount(comp, minlength=K).astype(np.float64)
    p = hist / hist.sum()
    ent = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
    ent_norm = float(ent / math.log(K)) if K > 1 else 0.0
    return {
        'gen_seed': int(seed), 'gen_n': int(n),
        'mmd': mmd, 'swd': sw,
        'minority_label': minority,
        'minority_mass': minority_mass,
        'minority_gen_frac': minority_gen_frac,
        'minority_mmd': min_mmd, 'minority_swd': min_sw,
        'diversity_mean_pairwise': div_mean,
        'component_entropy_norm': ent_norm,
        'component_hist': hist.astype(int).tolist(),
        'finite': bool(np.isfinite(xg).all() and np.isfinite(z).all()),
    }


def compute_full(method, z, y, X, client_model, seed, client_id,
                 original_params, sigma, dirs, K_override=None,
                 interface_batch=None):
    rows = []
    acd = None
    interface = None
    if method == 'A0':
        p = original_params
        pred = assign_params(z, p)
    elif method == 'A1':
        gm = fit_gmm(z, original_params['K'], seed)
        p = params_from_gmm(gm)
        pred = assign_params(z, p)
    elif method in ('A2', 'A3'):
        cap = max(2, len(z) // MIN_SAMPLES_PER_COMP)
        kmax = min(BIC_ORIG_MAX if method == 'A2' else BIC_EXT_MAX, cap)
        best, rows = bic_search(z, seed, kmax)
        p = params_from_gmm(best['gm'])
        pred = assign_params(z, p)
    elif method == 'A4':
        labels, K, theta = acd_run(z, seed)
        p = params_from_labels(z, labels)
        pred = labels
        batch = X[:8] if interface_batch is None else interface_batch
        interface = check_interface(client_model, p, batch, seed=seed)
        acd = {'K': K, 'counts': np.bincount(labels,
                                              minlength=K).tolist(),
               'theta_shape': list(theta.shape),
               'interface': interface, 'rate': ACD_RATE, 'k0': ACD_K0}
    elif method == 'A5':
        if K_override is None:
            raise ValueError('A5 requires A4 K')
        gm = fit_gmm(z, K_override, seed)
        p = params_from_gmm(gm)
        pred = assign_params(z, p)
    else:
        raise ValueError(method)
    fm = full_metrics(y, pred, p['pi'])
    fm['pred'] = np.asarray(pred, dtype=int).tolist()
    fm['assignment_source'] = ('acd_labels' if method == 'A4'
                               else ('original_gmm' if method == 'A0'
                                     else 'gmm_argmax'))
    fm['params'] = params_to_json(p)
    fm['interface'] = interface
    gen = generation_metrics(client_model, X, y, p, pred, seed, sigma, dirs)
    return fm, gen, p, rows, acd


def compute_hold(method, z_fit, z_hold, y_fit, client_model, seed, client_id,
                 original_params, K_override=None, interface_batch=None):
    rows = []
    acd = None
    interface = None
    if method == 'A0':
        p = original_params
        nll = nll_params(z_hold, p)
        reference = True
    elif method == 'A1':
        gm = fit_gmm(z_fit, original_params['K'], seed)
        p = params_from_gmm(gm)
        nll = nll_params(z_hold, p)
        reference = False
    elif method in ('A2', 'A3'):
        cap = max(2, len(z_fit) // MIN_SAMPLES_PER_COMP)
        kmax = min(BIC_ORIG_MAX if method == 'A2' else BIC_EXT_MAX, cap)
        best, rows = bic_search(z_fit, seed, kmax)
        p = params_from_gmm(best['gm'])
        nll = nll_params(z_hold, p)
        reference = False
    elif method == 'A4':
        labels, K, theta = acd_run(z_fit, seed)
        p = params_from_labels(z_fit, labels)
        nll = nll_params(z_hold, p)
        reference = False
        batch = z_fit[:8] if interface_batch is None else interface_batch
        interface = check_interface(client_model, p, batch, seed=seed)
        acd = {'K': K, 'counts': np.bincount(labels,
                                              minlength=K).tolist(),
               'theta_shape': list(theta.shape),
               'interface': interface, 'rate': ACD_RATE, 'k0': ACD_K0}
    elif method == 'A5':
        if K_override is None:
            raise ValueError('A5 requires A4 holdout K')
        gm = fit_gmm(z_fit, K_override, seed)
        p = params_from_gmm(gm)
        nll = nll_params(z_hold, p)
        reference = False
    else:
        raise ValueError(method)
    return ({'K': int(len(p['pi'])), 'nll': float(nll),
             'n_fit': int(len(z_fit)), 'n_hold': int(len(z_hold)),
             'reference_only': bool(reference),
             'params': params_to_json(p), 'interface': interface},
            rows, acd)


def load_data():
    digest = sha256(CKPT)
    if digest != EXPECTED_SHA:
        raise RuntimeError(f'Checkpoint SHA256 mismatch: {digest}')
    ckpt = torch.load(CKPT, map_location='cpu')
    if int(ckpt['round']) != 100 or len(ckpt['clients']) != 10:
        raise RuntimeError('Unexpected checkpoint structure')
    clients, Xs, ys, zs, states = [], [], [], [], []
    for i, state in enumerate(ckpt['clients']):
        model = VaDE(SimpleNamespace(input_dim=31, hid_dim=10,
                                     nClusters=int(len(state['rho_'])),
                                     alpha=0.05, cuda=False))
        model.load_state_dict(state, strict=True)
        model.eval()
        part = ckpt['partition'][i]
        X = np.asarray(part['x'], dtype=np.float64)
        y = np.asarray(part['y'], dtype=int)
        with torch.no_grad():
            z = model.encoder(torch.tensor(X, dtype=torch.float32))[0].numpy()
        clients.append(model)
        Xs.append(X)
        ys.append(y)
        zs.append(np.asarray(z, dtype=np.float64))
        states.append(state)
    sizes = [int(len(y)) for y in ys]
    if sizes != [int(x) for x in ckpt['partition_sizes']]:
        raise RuntimeError('Partition size mismatch')
    return {'ckpt': ckpt, 'clients': clients, 'X': Xs, 'y': ys, 'z': zs,
            'states': states, 'sizes': sizes, 'sha256': digest}


def original_params_for(data, i):
    p = params_from_original(data['states'][i])
    p['K'] = int(len(p['pi']))
    return p


def run_unit(data, client_id, seed, force=False):
    unit_path = EXP / 'results/units' / f'client{client_id:02d}_seed{seed}.json'
    if unit_path.exists() and not force:
        return json.loads(unit_path.read_text())
    X = data['X'][client_id]
    y = data['y'][client_id]
    z = data['z'][client_id]
    model = data['clients'][client_id]
    K0 = int(len(data['states'][client_id]['rho_']))
    fit_idx, hold_idx, split_hash = split_indices(len(y), client_id)
    z_fit, z_hold = z[fit_idx], z[hold_idx]
    x_batch = X[:8]
    sigma = float(np.sqrt(np.median(pdist(X, metric='sqeuclidean'))))
    if sigma <= 0:
        sigma = 1.0
    rng = np.random.default_rng(20261124 + client_id)
    dirs = rng.normal(size=(X.shape[1], 64))
    dirs /= np.linalg.norm(dirs, axis=0, keepdims=True)
    original = original_params_for(data, client_id)
    unit = {
        'client': int(client_id), 'seed': int(seed), 'n': int(len(y)),
        'class_counts': np.bincount(y, minlength=int(y.max()) + 1).tolist(),
        'class_labels': np.unique(y).tolist(),
        'original_K': K0,
        'holdout': {'seed': HOLDOUT_SEED, 'frac': HOLDOUT_FRAC,
                    'n_fit': int(len(fit_idx)), 'n_hold': int(len(hold_idx)),
                    'split_hash': split_hash},
        'methods': {}, 'failures': [],
    }
    a4_full_K = None
    a4_hold_K = None
    total_steps = len(data['clients']) * len(SEEDS) * len(METHODS)
    for method in METHODS:
        status('full', f'{method} client {client_id} seed {seed}',
               method=method, client=client_id, seed=seed, phase='full',
               total_steps=total_steps)
        try:
            if method == 'A5':
                if a4_full_K is None:
                    raise RuntimeError('A4 full failed; A5 cannot match K')
                full, gen, p, rows, acd = compute_full(
                    method, z, y, X, model, seed, client_id, original,
                    sigma, dirs, K_override=a4_full_K,
                    interface_batch=x_batch)
            else:
                full, gen, p, rows, acd = compute_full(
                    method, z, y, X, model, seed, client_id, original,
                    sigma, dirs, interface_batch=x_batch)
            if method == 'A4':
                a4_full_K = int(full['K'])
            for row in rows:
                append_jsonl(EXP / 'results/bic_search.jsonl',
                             {'client': client_id, 'seed': seed,
                              'split': 'full', 'method': method, **row})
            if acd is not None:
                append_jsonl(EXP / 'results/acd_runs.jsonl',
                             {'client': client_id, 'seed': seed,
                              'split': 'full', 'method': method, **acd})
            status('holdout', f'{method} client {client_id} seed {seed}',
                   method=method, client=client_id, seed=seed,
                   phase='holdout', total_steps=total_steps, latest={
                       'ari': full['ari'], 'K': full['K']})
            if method == 'A5':
                if a4_hold_K is None:
                    raise RuntimeError('A4 holdout failed; A5 cannot match K')
                hold, rows2, acd2 = compute_hold(
                    method, z_fit, z_hold, y[fit_idx], model, seed,
                    client_id, original, K_override=a4_hold_K,
                    interface_batch=x_batch)
            else:
                hold, rows2, acd2 = compute_hold(
                    method, z_fit, z_hold, y[fit_idx], model, seed,
                    client_id, original, interface_batch=x_batch)
            if method == 'A4':
                a4_hold_K = int(hold['K'])
            for row in rows2:
                append_jsonl(EXP / 'results/bic_search.jsonl',
                             {'client': client_id, 'seed': seed,
                              'split': 'holdout', 'method': method, **row})
            if acd2 is not None:
                append_jsonl(EXP / 'results/acd_runs.jsonl',
                             {'client': client_id, 'seed': seed,
                              'split': 'holdout', 'method': method, **acd2})
            status('generation', f'{method} client {client_id} seed {seed}',
                   method=method, client=client_id, seed=seed,
                   phase='generation', total_steps=total_steps, latest={
                       'mmd': gen['mmd'], 'swd': gen['swd']})
            unit['methods'][method] = {'full': full, 'generation': gen,
                                       'holdout': hold}
        except Exception as exc:
            tb = traceback.format_exc()
            fail = {'client': int(client_id), 'seed': int(seed),
                    'method': method, 'error': repr(exc),
                    'traceback': tb}
            unit['failures'].append(fail)
            append_jsonl(EXP / 'results/failures.jsonl', fail)
        status('unit', f'{method} client {client_id} seed {seed}',
               method=method, client=client_id, seed=seed,
               phase='done', total_steps=total_steps)
    atomic_json(unit_path, unit)
    return unit


def smoke():
    data = load_data()
    status('smoke', 'Running frozen ACD smoke test on client 0 seed 0')
    unit = run_unit(data, 0, 0, force=True)
    checks = {}
    passed = not unit['failures']
    for method in METHODS:
        rec = unit['methods'].get(method)
        if not rec:
            checks[method] = {'present': False}
            passed = False
            continue
        full = rec['full']
        gen = rec['generation']
        interface = full.get('interface')
        ok = (int(full['K']) >= 1 and np.isfinite(full['ari'])
              and np.isfinite(full['nmi']) and gen['finite'])
        if interface is not None:
            ok = ok and bool(interface['ok'])
        hold_if = (rec.get('holdout') or {}).get('interface')
        checks[method] = {'present': True, 'K': full['K'],
                          'ari': full['ari'], 'interface_ok':
                          None if interface is None else interface['ok'],
                          'holdout_interface_ok':
                          None if hold_if is None else hold_if['ok'],
                          'generation_finite': gen['finite'],
                          'ok': bool(ok)}
        if method == 'A4' and hold_if is not None:
            ok = ok and bool(hold_if['ok'])
            checks[method]['ok'] = bool(ok)
        passed = passed and ok
    report = {'passed': bool(passed), 'checkpoint_sha256': data['sha256'],
              'client': 0, 'seed': 0, 'checks': checks,
              'failures': unit['failures'],
              'holdout': unit['holdout']}
    atomic_json(EXP / 'results/smoke.json', report)
    lines = ['# Smoke test', '',
             f"- Checkpoint SHA256: `{data['sha256']}`",
             f"- Client 0, seed 0; holdout {unit['holdout']}",
             f"- Passed: **{passed}**", '',
             '| Method | K | ARI | Interface | Holdout interface | Gen finite | OK |',
             '|---|---:|---:|---|---|---|---|']
    for m in METHODS:
        c = checks[m]
        lines.append(f"| {m} | {c.get('K','—')} | "
                     f"{c.get('ari','—')} | {c.get('interface_ok','—')} | "
                     f"{c.get('holdout_interface_ok','—')} | "
                     f"{c.get('generation_finite','—')} | {c.get('ok','—')} |")
    if unit['failures']:
        lines += ['', '## Failures', '']
        for f in unit['failures']:
            lines.append(f"- {f['method']}: `{f['error']}`")
    (EXP / 'audit/SMOKE_TEST.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(report, indent=2, default=_json_default))
    return report


def full_run(force=False):
    smoke_path = EXP / 'results/smoke.json'
    if not smoke_path.exists() or not json.loads(
            smoke_path.read_text()).get('passed'):
        if not force:
            raise SystemExit('Smoke test not passed; run --smoke first')
    data = load_data()
    total_units = len(data['clients']) * len(SEEDS)
    status('full', 'Starting full frozen ACD experiment',
           total_units=total_units)
    done = 0
    for client_id in range(len(data['clients'])):
        for seed in SEEDS:
            unit_path = (EXP / 'results/units'
                         / f'client{client_id:02d}_seed{seed}.json')
            if unit_path.exists():
                done += 1
                status('full', f'unit {done}/{total_units} exists; skipping',
                       client=client_id, seed=seed, completed=done,
                       total_units=total_units)
                continue
            run_unit(data, client_id, seed)
            done += 1
            status('full', f'unit {done}/{total_units} complete',
                   client=client_id, seed=seed, completed=done,
                   total_units=total_units)
    status('done', 'Frozen ACD experiment complete', done=True,
           completed=done, total_units=total_units)
    print(json.dumps({'done': done, 'total_units': total_units}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    EXP.joinpath('results/units').mkdir(parents=True, exist_ok=True)
    EXP.joinpath('logs').mkdir(parents=True, exist_ok=True)
    config = {
        'experiment': 'acd_f01_frozen_v1',
        'checkpoint': str(CKPT), 'checkpoint_sha256': EXPECTED_SHA,
        'acd': {'implementation': 'vendor/FedHIRE/MultiClustering.py',
                'function': 'FCPL', 'sha256': sha256(VENDOR / 'MultiClustering.py'),
                'repo': 'https://github.com/hongdream/Fed-HIRE',
                'commit': 'c116c8354f988d4c3080bbd9b1ab685117e78787',
                'paper': 'https://arxiv.org/abs/2601.06404v2',
                'rate': ACD_RATE, 'k0': ACD_K0, 'max_iter': 100,
                'modified': False},
        'local_gmm': {'feature': 'client encoder mu (FedVaDE pre_train)',
                      'covariance': 'diag', 'reg_covar': REG_COVAR,
                      'pi_log_epsilon': PI_EPS,
                      'variance_estimator': 'population (ddof=0)'},
        'seeds': list(SEEDS), 'holdout_seed': HOLDOUT_SEED,
        'holdout_fraction': HOLDOUT_FRAC,
        'bic': {'original_range': [2, BIC_ORIG_MAX],
                'extended_range': [2, BIC_EXT_MAX],
                'sample_cap_per_component': MIN_SAMPLES_PER_COMP,
                'sample_cap_rule': 'k_max = min(range_max, n_fit // 5)',
                'source': 'prior project build_init.bic_select range 2..14'},
        'generation': {'n': N_GEN, 'diversity_n': DIV_N,
                       'decoder': 'client decoder', 'swd_directions': 64},
        'methods': {
            'A0': 'original local GMM from checkpoint',
            'A1': 'original K + ordinary GMM refit',
            'A2': 'BIC original range + ordinary GMM',
            'A3': 'BIC extended range + ordinary GMM',
            'A4': 'author ACD FCPL + moment-matched GMM, no EM',
            'A5': 'ordinary GMM with A4 K per client/seed/split'},
        'python': sys.version.split()[0], 'numpy': np.__version__,
        'torch': torch.__version__,
    }
    atomic_json(EXP / 'configs/config.json', config)
    if args.smoke:
        smoke()
    else:
        full_run(force=args.force)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        (EXP / 'error.txt').write_text(traceback.format_exc())
        status('error', 'Run failed; see error.txt')
        raise
