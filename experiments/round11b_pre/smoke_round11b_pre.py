"""Short smoke test for Round 11B-pre code paths. Writes only to smoke/."""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture

from common import (BASE, R10, MODELS, split_data, load_union, load_evaluator_config,
                    fit_and_score_gmm, gmm_params, n_eff, cov_metrics,
                    shrink_covariances, sample_from_params, decode,
                    evaluate_generation, full_nll, sample_union, encode_mean)
from evaluator import fit_named_evaluator, distance_reference
from run_federated_aggregation import build_moment_groups, group_params

OUT = BASE / 'smoke'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    Xtr, ytr, Xval, yval, Xte, yte = split_data()
    cfg = load_evaluator_config()
    tau = float(cfg['tau'])
    radius = float(cfg['radius'])
    clf = fit_named_evaluator(cfg['primary_classifier'], Xtr, ytr,
                              n_estimators=50, cv=3)
    nn_rare, _ = distance_reference(Xtr, ytr, quantile=0.95)
    union = load_union('fedvade_severe_seed0', 'severe', 0)

    # Diag fit.
    gmm_d, z_fit = fit_and_score_gmm(union, 100, 0, 3, 'diag', 0, n_init=2)
    wd, md, cd, _ = gmm_params(gmm_d)
    ev_d = evaluate_generation(
        decode(union['server'], sample_from_params(wd, md, cd, 'diag', 100, 0)[0]),
        clf, nn_rare, radius, tau, Xval, yval, seed=1)

    # Full fit and post-fit shrinkage.
    gmm_f, z_full = fit_and_score_gmm(union, 100, 0, 3, 'full', 0, n_init=2)
    wf, mf, cf, _ = gmm_params(gmm_f)
    resp = gmm_f.predict_proba(z_full)
    ne = n_eff(resp)
    lambdas = 50.0 / (ne + 50.0)
    cfs = shrink_covariances(cf, lambdas)
    val_ll = float(-full_nll(z_full, wf, mf, cfs))
    ev_f = evaluate_generation(
        decode(union['server'], sample_from_params(wf, mf, cfs, 'full', 100, 0)[0]),
        clf, nn_rare, radius, tau, Xval, yval, seed=2)
    cm = cov_metrics(cfs)

    # Moment matching and safe-K.
    groups, labels, D, Dn = build_moment_groups(union, 3)
    wg, mg, cg = group_params(groups)
    ev_g = evaluate_generation(
        decode(union['server'], sample_from_params(wg, mg, cg, 'full', 100, 0)[0]),
        clf, nn_rare, radius, tau, Xval, yval, seed=3)

    # D3-style moment init + EM.
    try:
        precisions = np.stack([np.linalg.inv(c + 1e-6 * np.eye(c.shape[0])) for c in cg])
        gmm_i = GaussianMixture(n_components=3, covariance_type='full',
                                means_init=mg, weights_init=wg,
                                precisions_init=precisions, n_init=1,
                                max_iter=200, reg_covar=1e-5,
                                random_state=0).fit(sample_union(union, 100, 0)[0])
        init_mode = 'means_weights_precisions'
    except Exception as exc:
        gmm_i = GaussianMixture(n_components=3, covariance_type='full',
                                means_init=mg, weights_init=wg, n_init=1,
                                max_iter=200, reg_covar=1e-5,
                                random_state=0).fit(sample_union(union, 100, 0)[0])
        init_mode = f'fallback_means_weights: {exc!r}'
    wi, mi, ci, _ = gmm_params(gmm_i)
    ev_i = evaluate_generation(
        decode(union['server'], sample_from_params(wi, mi, ci, 'full', 100, 0)[0]),
        clf, nn_rare, radius, tau, Xval, yval, seed=4)

    result = {
        'ok': True,
        'diag_val_ll': float(gmm_d.score(np.asarray(encode_mean(union['server'], Xval)[0]))),
        'full_shrunk_train_nll': val_ll,
        'diag_rare_proxy_100': ev_d['rare_proxy'],
        'full_shrunk_rare_proxy_100': ev_f['rare_proxy'],
        'moment_rare_proxy_100': ev_g['rare_proxy'],
        'moment_init_em_rare_proxy_100': ev_i['rare_proxy'],
        'moment_uneven_groups': [g['max_normalized_bw'] for g in groups],
        'min_eig': min(x['min_eig'] for x in cm),
        'max_cond': max(x['cond'] for x in cm),
        'chol_ok': all(x['chol_ok'] for x in cm),
        'init_mode': init_mode,
        'elapsed_seconds': time.time() - start,
    }
    (OUT / 'smoke_results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
