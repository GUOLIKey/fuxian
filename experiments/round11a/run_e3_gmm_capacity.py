"""E3: frozen-latent GMM capacity comparison (K=3/6/10, diagonal primary)."""
import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture

from common import (BASE, R10, RARE, split_data, update_progress, load_server, encode_mean,
                    decode, sample_diag_gmm, sample_full_gmm, distribution_metrics,
                    weight_entropy, effective_components, mean_std)
from evaluator import (fit_named_evaluator, distance_reference, evaluate_generated,
                      predict_all)

MODELS = [
    ('centralized', R10 / 'references/centralized/selected_server.pt', 'severe', 0),
    ('fedvade_severe_seed0', R10 / 'formal/severe_seed0/selected_server.pt', 'severe', 0),
    ('fedvade_severe_seed2', R10 / 'formal/severe_seed2/selected_server.pt', 'severe', 2),
]
KS = (3, 6, 10)
COVARIANCES = ('diag', 'full')


def write_rows(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def classwise(generated, clf, X, y, seed):
    pred, _, _ = predict_all(clf, generated)
    out = {}
    for c in (0, 1, 2):
        g = generated[pred == c]
        r = np.asarray(X)[np.asarray(y) == c]
        if len(g) >= 2 and len(r) >= 2:
            d = distribution_metrics(r, g, seed=seed + c)
            out[f'mmd_class{c}'] = d.get('mmd_rbf')
            out[f'swd_class{c}'] = d.get('sliced_wasserstein')
        else:
            out[f'mmd_class{c}'] = None
            out[f'swd_class{c}'] = None
    return out


def main():
    start = time.time()
    Xtr, ytr, Xval, yval, Xte, yte = split_data()
    cfg = json.loads((BASE / 'evaluator_config.json').read_text())
    tau = float(cfg['tau'])
    radius = float(cfg['radius'])
    clf = fit_named_evaluator(cfg['primary_classifier'], Xtr, ytr)
    nn_rare, _, _ = distance_reference(Xtr, ytr, quantile=0.95)
    rows = []
    best = -1.0

    for model_idx, (model_name, path, scenario, seed) in enumerate(MODELS):
        step = 9 + min(model_idx, 2)
        update_progress(stage='FROZEN_GMM_K_COMPARISON', status='RUNNING',
                        current_model=model_name, current_experiment='E3',
                        progress_current=step, progress_total=12)
        print('GMM_CAPACITY', model_name, flush=True)
        model, payload, state = load_server(path)
        Ztr, _ = encode_mean(model, Xtr)
        Zval, _ = encode_mean(model, Xval)
        Zte, _ = encode_mean(model, Xte)

        for cov in COVARIANCES:
            for K in KS:
                status = 'OK'
                note = ''
                try:
                    gmm = GaussianMixture(
                        n_components=K, covariance_type=cov, random_state=42,
                        n_init=5, max_iter=500,
                        reg_covar=1e-5 if cov == 'full' else 1e-6,
                    ).fit(Ztr)
                except Exception as exc:
                    rows.append({
                        'model': model_name, 'scenario': scenario, 'training_seed': seed,
                        'covariance': cov, 'K': K, 'mode': 'fit', 'component': '',
                        'inference_seed': '', 'eval_split': '', 'n': 0,
                        'train_ll': '', 'val_ll': '', 'bic': '', 'aic': '',
                        'effective_components': '', 'weight_entropy': '',
                        'min_weight': '', 'max_weight': '', 'occupancy_min': '',
                        'occupancy_max': '', 'rare_resp_max_component': '',
                        'rare_resp_max': '', 'rare_proxy': '', 'assigned_rare_fraction': '',
                        'mmd_rbf': '', 'sliced_wasserstein': '', 'mmd_class2': '',
                        'swd_class2': '', 'status': 'FAILED', 'note': repr(exc),
                    })
                    continue
                weights = gmm.weights_
                means = gmm.means_
                if cov == 'diag':
                    variances = gmm.covariances_
                    log_sigma2 = np.log(np.maximum(variances, 1e-12))
                else:
                    eig_min = min(float(np.linalg.eigvalsh(c).min()) for c in gmm.covariances_)
                    if eig_min <= 0:
                        status = 'INVALID_COVARIANCE'
                        note = f'min_eig={eig_min:.3e}'
                train_ll = float(gmm.score(Ztr))
                val_ll = float(gmm.score(Zval))
                bic = float(gmm.bic(Ztr))
                aic = float(gmm.aic(Ztr))
                eff = effective_components(weights, 0.01)
                went = weight_entropy(weights)
                occ_tr = np.bincount(gmm.predict(Ztr), minlength=K)
                rare_mask = np.asarray(yval) == RARE
                if np.any(rare_mask):
                    resp = gmm.predict_proba(Zval[rare_mask])
                    rare_mean = resp.mean(axis=0)
                    rare_max_component = int(np.argmax(rare_mean))
                    rare_max = float(rare_mean[rare_max_component])
                else:
                    rare_max_component = ''
                    rare_max = ''

                # Fit-level row.
                rows.append({
                    'model': model_name, 'scenario': scenario, 'training_seed': seed,
                    'covariance': cov, 'K': K, 'mode': 'fit', 'component': '',
                    'inference_seed': '', 'eval_split': '', 'n': len(Ztr),
                    'train_ll': train_ll, 'val_ll': val_ll, 'bic': bic, 'aic': aic,
                    'effective_components': eff, 'weight_entropy': went,
                    'min_weight': float(weights.min()), 'max_weight': float(weights.max()),
                    'occupancy_min': int(occ_tr.min()), 'occupancy_max': int(occ_tr.max()),
                    'rare_resp_max_component': rare_max_component,
                    'rare_resp_max': rare_max, 'rare_proxy': '', 'assigned_rare_fraction': '',
                    'mmd_rbf': '', 'sliced_wasserstein': '', 'mmd_class2': '',
                    'swd_class2': '', 'status': status, 'note': note,
                })
                if status != 'OK':
                    continue

                for inf_seed in range(5):
                    if cov == 'diag':
                        z, labels = sample_diag_gmm(weights, means, log_sigma2, 2000, inf_seed)
                    else:
                        z, labels = sample_full_gmm(weights, means, gmm.covariances_, 2000, inf_seed)
                    gen = decode(model, z)
                    ev = evaluate_generated(gen, clf, nn_rare, radius, tau)
                    cw = classwise(gen, clf, Xval, yval, seed=5000 + inf_seed)
                    for eval_split, Xr in (('val', Xval), ('test', Xte)):
                        dist = distribution_metrics(Xr, gen, seed=6000 + inf_seed)
                        rows.append({
                            'model': model_name, 'scenario': scenario, 'training_seed': seed,
                            'covariance': cov, 'K': K, 'mode': 'unconditional', 'component': '',
                            'inference_seed': inf_seed, 'eval_split': eval_split, 'n': len(gen),
                            'train_ll': train_ll, 'val_ll': val_ll, 'bic': bic, 'aic': aic,
                            'effective_components': eff, 'weight_entropy': went,
                            'min_weight': float(weights.min()), 'max_weight': float(weights.max()),
                            'occupancy_min': int(occ_tr.min()), 'occupancy_max': int(occ_tr.max()),
                            'rare_resp_max_component': rare_max_component,
                            'rare_resp_max': rare_max,
                            'rare_proxy': ev['rare_proxy'],
                            'assigned_rare_fraction': ev['assigned_rare_fraction'],
                            'mmd_rbf': dist.get('mmd_rbf'),
                            'sliced_wasserstein': dist.get('sliced_wasserstein'),
                            'mmd_class2': cw['mmd_class2'], 'swd_class2': cw['swd_class2'],
                            'status': status, 'note': 'test rows exploratory' if eval_split == 'test' else '',
                        })
                    best = max(best, ev['rare_proxy'])

                # Component-conditioned diagnostic, one fixed seed.
                for comp in range(K):
                    if cov == 'diag':
                        z, labels = sample_diag_gmm(weights, means, log_sigma2, 1000, 0, component=comp)
                    else:
                        z, labels = sample_full_gmm(weights, means, gmm.covariances_, 1000, 0, component=comp)
                    gen = decode(model, z)
                    ev = evaluate_generated(gen, clf, nn_rare, radius, tau)
                    cw = classwise(gen, clf, Xval, yval, seed=7000 + comp)
                    rows.append({
                        'model': model_name, 'scenario': scenario, 'training_seed': seed,
                        'covariance': cov, 'K': K, 'mode': 'component_conditioned', 'component': comp,
                        'inference_seed': 0, 'eval_split': 'val', 'n': len(gen),
                        'train_ll': train_ll, 'val_ll': val_ll, 'bic': bic, 'aic': aic,
                        'effective_components': eff, 'weight_entropy': went,
                        'min_weight': float(weights.min()), 'max_weight': float(weights.max()),
                        'occupancy_min': int(occ_tr.min()), 'occupancy_max': int(occ_tr.max()),
                        'rare_resp_max_component': rare_max_component,
                        'rare_resp_max': rare_max,
                        'rare_proxy': ev['rare_proxy'],
                        'assigned_rare_fraction': ev['assigned_rare_fraction'],
                        'mmd_rbf': '', 'sliced_wasserstein': '',
                        'mmd_class2': cw['mmd_class2'], 'swd_class2': cw['swd_class2'],
                        'status': status, 'note': 'component-conditioned diagnostic',
                    })

                update_progress(progress_current=step, rare_mode_proxy=float(best),
                                best_result=f'E3 best rare_proxy={best:.4f}')

        update_progress(progress_current=11, rare_mode_proxy=float(best),
                        best_result=f'E3 best rare_proxy={best:.4f}')

    write_rows(BASE / 'gmm_k_comparison.csv', rows)
    print('E3_OK', len(rows), 'elapsed', time.time() - start, flush=True)


if __name__ == '__main__':
    main()
