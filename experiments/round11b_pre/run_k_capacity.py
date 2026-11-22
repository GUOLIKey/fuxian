"""Stage C: K capacity and component quality audit at fixed M."""
import json
import time

import numpy as np

from common import (BASE, MODELS, RARE, write_csv, update_progress, load_evaluator_config,
                    split_data, load_union, fit_and_score_gmm, gmm_params, n_eff,
                    cov_metrics, sample_from_params, decode, evaluate_generation,
                    atomic_json, encode_mean)
from evaluator import fit_named_evaluator, distance_reference
from protocol import K_VALUES, INIT_SEEDS, INFERENCE_SEEDS, TOTAL_CONFIGS
from protocol import (LOW_SUPPORT_N_EFF, LOW_SUPPORT_WEIGHT, PSEUDO_RARE_PROXY,
                      PSEUDO_RARE_MMD, PSEUDO_RARE_SWD, PSEUDO_RARE_MIN_CLASS2)


def mean(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def main():
    start = time.time()
    Xtr, ytr, Xval, yval, Xte, yte = split_data()
    cfg = load_evaluator_config()
    tau = float(cfg['tau'])
    radius = float(cfg['radius'])
    clf = fit_named_evaluator(cfg['primary_classifier'], Xtr, ytr)
    nn_rare, _ = distance_reference(Xtr, ytr, quantile=0.95)
    selected_m = json.loads((BASE / 'selected_m.json').read_text())['selected_M']
    rows = []
    audit_rows = []
    fit_records = []
    best_val_ll = -1e9
    last_rare = None
    progress_path = BASE / 'progress.json'
    completed = json.loads(progress_path.read_text()).get('completed', 0) if progress_path.exists() else 0
    rare_mask = np.asarray(yval) == RARE

    for model_name, scenario, seed in MODELS:
        union = load_union(model_name, scenario, seed)
        Ztr, _ = encode_mean(union['server'], Xtr)
        Zval, _ = encode_mean(union['server'], Xval)
        for K in K_VALUES:
            for cov in ('diag', 'full'):
                for init_seed in INIT_SEEDS:
                    config = f'K={K} cov={cov} init={init_seed}'
                    update_progress(stage='K_CAPACITY', status='RUNNING',
                                    current_model=model_name, current_config=config,
                                    completed=completed, total=TOTAL_CONFIGS)
                    gmm, z_fit = fit_and_score_gmm(
                        union, selected_m, 0, K, cov, init_seed, n_init=5)
                    weights, means, covs, log_sigma2 = gmm_params(gmm)
                    resp = gmm.predict_proba(z_fit)
                    ne = n_eff(resp)
                    occ = np.bincount(gmm.predict(z_fit), minlength=K)
                    cm = cov_metrics(covs)
                    val_ll = float(gmm.score(Zval))
                    train_ll = float(gmm.score(Ztr))
                    bic = float(gmm.bic(z_fit))
                    aic = float(gmm.aic(z_fit))
                    stable = all(x['chol_ok'] for x in cm) and all(x['min_eig'] > 0 for x in cm)
                    best_val_ll = max(best_val_ll, val_ll)
                    fit_records.append({
                        'model': model_name, 'K': K, 'covariance': cov,
                        'init_seed': init_seed, 'val_ll': val_ll,
                        'bic': bic, 'stable': stable,
                    })
                    for inf_seed in INFERENCE_SEEDS:
                        z, _ = sample_from_params(weights, means, covs, cov, 2000, inf_seed)
                        gen = decode(union['server'], z)
                        ev = evaluate_generation(gen, clf, nn_rare, radius, tau,
                                                 Xval, yval, seed=11000 + inf_seed)
                        last_rare = ev['rare_proxy']
                        rows.append({
                            'model': model_name, 'training_seed': seed,
                            'M': selected_m, 'K': K, 'covariance': cov,
                            'init_seed': init_seed, 'n_fit': selected_m,
                            'train_ll': train_ll, 'val_ll': val_ll,
                            'bic_fit': bic, 'aic_fit': aic,
                            'min_eig': min(x['min_eig'] for x in cm),
                            'max_cond': max(x['cond'] for x in cm),
                            'chol_ok': all(x['chol_ok'] for x in cm),
                            'stable': stable,
                            'weight_min': float(np.min(weights)),
                            'weight_max': float(np.max(weights)),
                            'occupancy_min': int(np.min(occ)),
                            'occupancy_max': int(np.max(occ)),
                            'n_eff_min': float(np.min(ne)),
                            'n_eff_max': float(np.max(ne)),
                            'effective_K': int(np.sum(weights >= 0.01)),
                            'inference_seed': inf_seed,
                            'rare_proxy': ev['rare_proxy'],
                            'assigned_rare_fraction': ev['assigned_rare_fraction'],
                            'class2_count': ev['class2_count'],
                            'class2_mmd': ev['mmd_class2'], 'class2_swd': ev['swd_class2'],
                            'mmd_rbf': ev['mmd_rbf'], 'sliced_wasserstein': ev['sliced_wasserstein'],
                            'status': 'OK' if stable else 'UNSTABLE', 'note': '',
                        })
                    # Component-conditioned audit only for the first init seed.
                    if init_seed == 0:
                        rare_resp = gmm.predict_proba(Zval[rare_mask]).mean(axis=0) if np.any(rare_mask) else np.zeros(K)
                        for k in range(K):
                            z, _ = sample_from_params(weights, means, covs, cov, 1000, 0, component=k)
                            gen = decode(union['server'], z)
                            ev = evaluate_generation(gen, clf, nn_rare, radius, tau,
                                                     Xval, yval, seed=12000 + k)
                            flags = []
                            if ne[k] < LOW_SUPPORT_N_EFF or weights[k] < LOW_SUPPORT_WEIGHT:
                                flags.append('LOW_SUPPORT')
                            if ev['class2_count'] < PSEUDO_RARE_MIN_CLASS2:
                                flags.append('INSUFFICIENT_CLASS2')
                            elif (ev['rare_proxy'] >= PSEUDO_RARE_PROXY and
                                  ((ev['mmd_class2'] is not None and ev['mmd_class2'] >= PSEUDO_RARE_MMD) or
                                   (ev['swd_class2'] is not None and ev['swd_class2'] >= PSEUDO_RARE_SWD))):
                                flags.append('PSEUDO_RARE_RISK')
                            audit_rows.append({
                                'model': model_name, 'training_seed': seed, 'M': selected_m,
                                'K': K, 'covariance': cov, 'init_seed': init_seed,
                                'component': k, 'weight': float(weights[k]),
                                'n_eff': float(ne[k]), 'occupancy': int(occ[k]),
                                'min_eig': cm[k]['min_eig'], 'max_cond': cm[k]['cond'],
                                'chol_ok': cm[k]['chol_ok'],
                                'rare_responsibility_val': float(rare_resp[k]),
                                'component_rare_proxy': ev['rare_proxy'],
                                'assigned_rare_fraction': ev['assigned_rare_fraction'],
                                'class2_count': ev['class2_count'],
                                'class2_mmd': ev['mmd_class2'], 'class2_swd': ev['swd_class2'],
                                'mmd_rbf': ev['mmd_rbf'], 'sliced_wasserstein': ev['sliced_wasserstein'],
                                'flags': ';'.join(flags) if flags else 'OK',
                                'note': 'flags are diagnostic only; never used for fitting/selection',
                            })
                    completed += 1
                    update_progress(completed=completed, total=TOTAL_CONFIGS,
                                    best_val_ll=float(best_val_ll),
                                    rare_mode_proxy=float(last_rare))

    write_csv(BASE / 'k_capacity_ablation.csv', rows)
    write_csv(BASE / 'component_quality_audit.csv', audit_rows)

    summary = []
    for K in K_VALUES:
        for cov in ('diag', 'full'):
            recs = [r for r in fit_records if r['K'] == K and r['covariance'] == cov]
            stable = [r for r in recs if r['stable']]
            rr = [r for r in rows if r['K'] == K and r['covariance'] == cov]
            summary.append({
                'K': K, 'covariance': cov,
                'mean_val_ll': mean([r['val_ll'] for r in stable]),
                'mean_bic': mean([r['bic'] for r in stable]),
                'all_stable': len(stable) == len(recs),
                'mean_rare_proxy': mean([float(r['rare_proxy']) for r in rr]),
                'mean_class2_mmd': mean([float(r['class2_mmd']) for r in rr
                                         if r['class2_mmd'] not in ('', None)]),
                'mean_class2_swd': mean([float(r['class2_swd']) for r in rr
                                         if r['class2_swd'] not in ('', None)]),
                'mean_mmd': mean([float(r['mmd_rbf']) for r in rr]),
                'mean_swd': mean([float(r['sliced_wasserstein']) for r in rr]),
                'pseudo_rare_components': sum(1 for r in audit_rows
                                              if r['K'] == K and r['covariance'] == cov
                                              and 'PSEUDO_RARE_RISK' in r['flags']),
                'low_support_components': sum(1 for r in audit_rows
                                              if r['K'] == K and r['covariance'] == cov
                                              and 'LOW_SUPPORT' in r['flags']),
            })
    write_csv(BASE / 'k_capacity_summary.csv', summary)
    stable_summary = [r for r in summary if r['all_stable'] and r['mean_bic'] is not None]
    if stable_summary:
        best = min(stable_summary, key=lambda r: (r['mean_bic'], -r['mean_val_ll']))
        selected = {'K': best['K'], 'covariance': best['covariance'],
                    'mean_bic': best['mean_bic'], 'mean_val_ll': best['mean_val_ll'],
                    'rule': 'lowest mean BIC on fit samples among stable configs'}
    else:
        selected = {'K': 3, 'covariance': 'diag', 'mean_bic': None,
                    'mean_val_ll': None, 'rule': 'no stable config; fallback K=3 diag'}
    atomic_json(BASE / 'selected_k.json', selected)
    update_progress(stage='K_CAPACITY', status='RUNNING',
                    current_config=f"BIC selected K={selected['K']} {selected['covariance']}",
                    completed=completed, total=TOTAL_CONFIGS,
                    best_val_ll=float(best_val_ll),
                    rare_mode_proxy=float(last_rare))
    print('K_CAPACITY_OK', selected, 'elapsed', time.time() - start, flush=True)


if __name__ == '__main__':
    main()
