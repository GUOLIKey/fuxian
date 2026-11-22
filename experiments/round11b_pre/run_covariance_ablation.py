"""Stage B: covariance structure and shrinkage ablation at fixed M and K=3."""
import json
import time

import numpy as np

from common import (BASE, MODELS, write_csv, update_progress, load_evaluator_config,
                    split_data, load_union, fit_and_score_gmm, gmm_params, n_eff,
                    cov_metrics, shrink_covariances, sample_from_params, decode,
                    evaluate_generation, atomic_json, encode_mean, full_nll)
from evaluator import fit_named_evaluator, distance_reference
from protocol import COV_CONFIGS, INIT_SEEDS, INFERENCE_SEEDS, TOTAL_CONFIGS


def bump(completed, **changes):
    changes.setdefault('completed', completed)
    changes.setdefault('total', TOTAL_CONFIGS)
    update_progress(**changes)


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
    fit_records = []
    best_val_ll = -1e9
    last_rare = None
    progress_path = BASE / 'progress.json'
    completed = json.loads(progress_path.read_text()).get('completed', 0) if progress_path.exists() else 0

    for model_name, scenario, seed in MODELS:
        union = load_union(model_name, scenario, seed)
        Ztr, _ = encode_mean(union['server'], Xtr)
        Zval, _ = encode_mean(union['server'], Xval)
        for name, base_cov, fixed_lam, kappa in COV_CONFIGS:
            for init_seed in INIT_SEEDS:
                config = (f'{name}' + (f'_lambda={fixed_lam}' if fixed_lam is not None else '') +
                          (f'_kappa={kappa}' if kappa is not None else ''))
                update_progress(stage='COVARIANCE_ABLATION', status='RUNNING',
                                current_model=model_name, current_config=config,
                                completed=completed, total=TOTAL_CONFIGS)
                try:
                    gmm, z_fit = fit_and_score_gmm(
                        union, selected_m, 0, 3, base_cov, init_seed, n_init=5)
                    weights, means, covs, log_sigma2 = gmm_params(gmm)
                    resp = gmm.predict_proba(z_fit)
                    ne = n_eff(resp)
                    occ = np.bincount(gmm.predict(z_fit), minlength=3)
                    if base_cov == 'diag':
                        covs_used = covs
                        lambdas = np.zeros(3)
                        val_ll = float(gmm.score(Zval))
                        train_ll = float(gmm.score(Ztr))
                        bic = float(gmm.bic(z_fit))
                        aic = float(gmm.aic(z_fit))
                        cov_type_used = 'diag'
                    else:
                        if name == 'full':
                            lambdas = np.zeros(3)
                        elif name == 'shrink_fixed':
                            lambdas = np.full(3, float(fixed_lam))
                        else:
                            lambdas = float(kappa) / (ne + float(kappa))
                        covs_used = shrink_covariances(covs, lambdas)
                        val_ll = float(-full_nll(Zval, weights, means, covs_used))
                        train_ll = float(-full_nll(Ztr, weights, means, covs_used))
                        loglik_fit = -full_nll(z_fit, weights, means, covs_used) * len(z_fit)
                        p = 3 * (10 + 10 * 11 // 2) + 2
                        bic = float(-2.0 * loglik_fit + p * np.log(len(z_fit)))
                        aic = float(-2.0 * loglik_fit + 2.0 * p)
                        cov_type_used = 'full'
                    cm = cov_metrics(covs_used)
                    stable = all(x['chol_ok'] for x in cm) and all(x['min_eig'] > 0 for x in cm)
                    best_val_ll = max(best_val_ll, val_ll)
                    fit_records.append({
                        'model': model_name, 'M': selected_m, 'K': 3,
                        'config': name, 'base_covariance': base_cov,
                        'lambda': fixed_lam, 'kappa': kappa,
                        'init_seed': init_seed, 'val_ll': val_ll,
                        'max_cond': max(x['cond'] for x in cm),
                        'min_eig': min(x['min_eig'] for x in cm),
                        'chol_ok': all(x['chol_ok'] for x in cm),
                        'stable': stable,
                    })
                    for inf_seed in INFERENCE_SEEDS:
                        z, _ = sample_from_params(weights, means, covs_used,
                                                  cov_type_used, 2000, inf_seed)
                        gen = decode(union['server'], z)
                        ev = evaluate_generation(gen, clf, nn_rare, radius, tau,
                                                 Xval, yval, seed=10000 + inf_seed)
                        last_rare = ev['rare_proxy']
                        rows.append({
                            'model': model_name, 'training_seed': seed,
                            'M': selected_m, 'K': 3, 'config': name,
                            'base_covariance': base_cov,
                            'lambda': fixed_lam if fixed_lam is not None else '',
                            'kappa': kappa if kappa is not None else '',
                            'lambda_min': float(np.min(lambdas)),
                            'lambda_max': float(np.max(lambdas)),
                            'init_seed': init_seed, 'n_fit': selected_m,
                            'train_ll': train_ll, 'val_ll': val_ll,
                            'bic_fit': bic, 'aic_fit': aic,
                            'min_eig': min(x['min_eig'] for x in cm),
                            'max_cond': max(x['cond'] for x in cm),
                            'chol_ok': all(x['chol_ok'] for x in cm),
                            'stable': stable,
                            'n_eff_min': float(np.min(ne)), 'n_eff_max': float(np.max(ne)),
                            'weight_min': float(np.min(weights)), 'weight_max': float(np.max(weights)),
                            'occupancy_min': int(np.min(occ)), 'occupancy_max': int(np.max(occ)),
                            'inference_seed': inf_seed,
                            'rare_proxy': ev['rare_proxy'],
                            'assigned_rare_fraction': ev['assigned_rare_fraction'],
                            'class2_count': ev['class2_count'],
                            'class2_mmd': ev['mmd_class2'], 'class2_swd': ev['swd_class2'],
                            'mmd_rbf': ev['mmd_rbf'], 'sliced_wasserstein': ev['sliced_wasserstein'],
                            'status': 'OK' if stable else 'UNSTABLE',
                            'note': 'post-fit shrinkage; BIC approximate for shrinkage',
                        })
                except Exception as exc:
                    rows.append({
                        'model': model_name, 'training_seed': seed, 'M': selected_m, 'K': 3,
                        'config': name, 'base_covariance': base_cov,
                        'lambda': fixed_lam if fixed_lam is not None else '',
                        'kappa': kappa if kappa is not None else '',
                        'lambda_min': '', 'lambda_max': '', 'init_seed': init_seed,
                        'n_fit': selected_m, 'train_ll': '', 'val_ll': '', 'bic_fit': '',
                        'aic_fit': '', 'min_eig': '', 'max_cond': '', 'chol_ok': False,
                        'stable': False, 'n_eff_min': '', 'n_eff_max': '',
                        'weight_min': '', 'weight_max': '', 'occupancy_min': '',
                        'occupancy_max': '', 'inference_seed': '',
                        'rare_proxy': '', 'assigned_rare_fraction': '', 'class2_count': '',
                        'class2_mmd': '', 'class2_swd': '', 'mmd_rbf': '',
                        'sliced_wasserstein': '', 'status': 'FAILED', 'note': repr(exc),
                    })
                completed += 1
                bump(completed, best_val_ll=float(best_val_ll),
                     rare_mode_proxy=float(last_rare) if last_rare is not None else None)

    write_csv(BASE / 'covariance_ablation.csv', rows)
    summary_rows = []
    for name, base_cov, fixed_lam, kappa in COV_CONFIGS:
        recs = [r for r in fit_records if r['config'] == name and
                (fixed_lam is None or r['lambda'] == fixed_lam) and
                (kappa is None or r['kappa'] == kappa)]
        if not recs:
            continue
        rr = [r for r in rows if r['config'] == name and
              (fixed_lam is None or r['lambda'] == fixed_lam) and
              (kappa is None or r['kappa'] == kappa)]
        summary_rows.append({
            'config': name, 'base_covariance': base_cov,
            'lambda': fixed_lam if fixed_lam is not None else '',
            'kappa': kappa if kappa is not None else '',
            'mean_val_ll': mean([r['val_ll'] for r in recs]),
            'mean_min_eig': mean([r['min_eig'] for r in recs]),
            'max_cond': max([r['max_cond'] for r in recs]),
            'all_stable': all(r['stable'] for r in recs),
            'mean_rare_proxy': mean([float(r['rare_proxy']) for r in rr if r['rare_proxy'] != '']),
            'mean_class2_mmd': mean([float(r['class2_mmd']) for r in rr
                                     if r['class2_mmd'] not in ('', None)]),
            'mean_class2_swd': mean([float(r['class2_swd']) for r in rr
                                     if r['class2_swd'] not in ('', None)]),
            'mean_mmd': mean([float(r['mmd_rbf']) for r in rr if r['mmd_rbf'] != '']),
            'mean_swd': mean([float(r['sliced_wasserstein']) for r in rr
                              if r['sliced_wasserstein'] != '']),
        })
    write_csv(BASE / 'covariance_summary.csv', summary_rows)
    stable = [r for r in summary_rows if r['all_stable'] and r['mean_val_ll'] is not None]
    if stable:
        best = max(stable, key=lambda r: (r['mean_val_ll'], -r['max_cond']))
        selected = {
            'selected_config': best['config'],
            'base_covariance': best['base_covariance'],
            'lambda': best['lambda'],
            'kappa': best['kappa'],
            'mean_val_ll': best['mean_val_ll'],
            'max_cond': best['max_cond'],
            'rule': 'highest mean val LL among stable configs; tie-break lower max condition number',
        }
    else:
        selected = {
            'selected_config': 'diag', 'base_covariance': 'diag',
            'lambda': '', 'kappa': '',
            'mean_val_ll': None, 'max_cond': None,
            'rule': 'no stable full/shrinkage config; fallback diag',
        }
    atomic_json(BASE / 'selected_covariance.json', selected)
    update_progress(stage='COVARIANCE_ABLATION', status='RUNNING',
                    current_config=f"selected={selected['selected_config']}",
                    completed=completed, total=TOTAL_CONFIGS,
                    best_val_ll=float(best_val_ll),
                    rare_mode_proxy=float(last_rare) if last_rare is not None else None)
    print('COVARIANCE_OK', selected, 'elapsed', time.time() - start, flush=True)


if __name__ == '__main__':
    main()
