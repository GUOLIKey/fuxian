"""Stage A: M sweep with fixed K=3, diagonal covariance and sklearn EM."""
import json
import time

import numpy as np

from common import (BASE, MODELS, write_csv, update_progress, load_evaluator_config,
                    split_data, load_union, fit_and_score_gmm, gmm_params, n_eff,
                    cov_metrics, sample_from_params, decode, evaluate_generation,
                    atomic_json, encode_mean)
from evaluator import fit_named_evaluator, distance_reference
from protocol import M_VALUES, INIT_SEEDS, INFERENCE_SEEDS, M_SELECTION_DELTA, TOTAL_CONFIGS


def bump(n=1, **changes):
    path = BASE / 'progress.json'
    current = json.loads(path.read_text()) if path.exists() else {}
    changes.setdefault('completed', current.get('completed', 0) + n)
    changes.setdefault('total', TOTAL_CONFIGS)
    update_progress(**changes)


def main():
    start = time.time()
    Xtr, ytr, Xval, yval, Xte, yte = split_data()
    cfg = load_evaluator_config()
    tau = float(cfg['tau'])
    radius = float(cfg['radius'])
    clf = fit_named_evaluator(cfg['primary_classifier'], Xtr, ytr)
    nn_rare, _ = distance_reference(Xtr, ytr, quantile=0.95)
    rows = []
    main_ll = {}
    best_val_ll = -1e9
    last_rare = None
    completed = 0

    for model_name, scenario, seed in MODELS:
        update_progress(stage='M_SWEEP', status='RUNNING', current_model=model_name,
                        current_config='loading frozen model', completed=completed,
                        total=TOTAL_CONFIGS)
        union = load_union(model_name, scenario, seed)
        Ztr, _ = encode_mean(union['server'], Xtr)
        Zval, _ = encode_mean(union['server'], Xval)
        for M in M_VALUES:
            for sample_seed in (0, 1, 2):
                init_seeds = INIT_SEEDS if sample_seed == 0 else (0,)
                role = 'main' if sample_seed == 0 else 'sampling_variability'
                for init_seed in init_seeds:
                    config = f'M={M} sample={sample_seed} init={init_seed}'
                    update_progress(stage='M_SWEEP', status='RUNNING',
                                    current_model=model_name, current_config=config,
                                    completed=completed, total=TOTAL_CONFIGS)
                    gmm, z_fit = fit_and_score_gmm(
                        union, M, sample_seed, 3, 'diag', init_seed, n_init=5)
                    weights, means, covs, _ = gmm_params(gmm)
                    resp = gmm.predict_proba(z_fit)
                    ne = n_eff(resp)
                    occ = np.bincount(gmm.predict(z_fit), minlength=3)
                    cm = cov_metrics(covs)
                    val_ll = float(gmm.score(Zval))
                    train_ll = float(gmm.score(Ztr))
                    bic = float(gmm.bic(z_fit))
                    aic = float(gmm.aic(z_fit))
                    if sample_seed == 0:
                        main_ll[(model_name, M, init_seed)] = val_ll
                    best_val_ll = max(best_val_ll, val_ll)
                    for inf_seed in INFERENCE_SEEDS:
                        z, _ = sample_from_params(weights, means, covs, 'diag', 2000, inf_seed)
                        gen = decode(union['server'], z)
                        ev = evaluate_generation(gen, clf, nn_rare, radius, tau,
                                                 Xval, yval, seed=9000 + inf_seed)
                        last_rare = ev['rare_proxy']
                        rows.append({
                            'model': model_name, 'training_seed': seed,
                            'role': role, 'M': M, 'sample_seed': sample_seed,
                            'init_seed': init_seed, 'K': 3, 'covariance': 'diag',
                            'n_fit': M, 'train_ll': train_ll, 'val_ll': val_ll,
                            'bic_fit': bic, 'aic_fit': aic,
                            'min_eig': min(x['min_eig'] for x in cm),
                            'max_cond': max(x['cond'] for x in cm),
                            'chol_ok': all(x['chol_ok'] for x in cm),
                            'weight_min': float(np.min(weights)),
                            'weight_max': float(np.max(weights)),
                            'occupancy_min': int(np.min(occ)),
                            'occupancy_max': int(np.max(occ)),
                            'n_eff_min': float(np.min(ne)),
                            'n_eff_max': float(np.max(ne)),
                            'inference_seed': inf_seed,
                            'rare_proxy': ev['rare_proxy'],
                            'assigned_rare_fraction': ev['assigned_rare_fraction'],
                            'class2_count': ev['class2_count'],
                            'class2_mmd': ev['mmd_class2'],
                            'class2_swd': ev['swd_class2'],
                            'mmd_rbf': ev['mmd_rbf'],
                            'sliced_wasserstein': ev['sliced_wasserstein'],
                            'status': 'OK', 'note': 'BIC on union fit samples; not comparable across M',
                        })
                    completed += 1
                    bump(0, completed=completed, best_val_ll=float(best_val_ll),
                         rare_mode_proxy=float(last_rare))

    write_csv(BASE / 'm_sweep_results.csv', rows)

    mean_m = {}
    for M in M_VALUES:
        vals = [v for (m, mm, s), v in main_ll.items() if mm == M and m in [x[0] for x in MODELS]]
        mean_m[str(M)] = float(np.mean(vals)) if vals else None
    rule = (f'select 10000 only if mean val LL exceeds 1000 by > {M_SELECTION_DELTA}; '
            'otherwise 1000; 100 is baseline only')
    selected_m = 10000 if (mean_m['10000'] is not None and mean_m['1000'] is not None
                           and mean_m['10000'] > mean_m['1000'] + M_SELECTION_DELTA) else 1000
    atomic_json(BASE / 'selected_m.json', {
        'selected_M': selected_m,
        'mean_main_val_ll': mean_m,
        'delta': M_SELECTION_DELTA,
        'rule': rule,
        'finished_at': time.time(),
    })
    update_progress(stage='M_SWEEP', status='RUNNING', current_config=f'M selected={selected_m}',
                    completed=completed, total=TOTAL_CONFIGS,
                    best_val_ll=float(best_val_ll), rare_mode_proxy=float(last_rare))
    print('M_SWEEP_OK', selected_m, mean_m, 'elapsed', time.time() - start, flush=True)


if __name__ == '__main__':
    main()
