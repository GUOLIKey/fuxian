"""Stage D: federated statistics aggregation, moment matching and oracle gap inputs."""
import itertools
import json
import time

import numpy as np
from sklearn.mixture import GaussianMixture

from common import (BASE, MODELS, R10, RARE, write_csv, update_progress,
                    load_evaluator_config, split_data, load_union, load_server,
                    fit_and_score_gmm, gmm_params, n_eff, cov_metrics,
                    shrink_covariances, sample_from_params, decode,
                    evaluate_generation, atomic_json, encode_mean, full_nll,
                    full_responsibilities, diag_nll, bures_diag, normalized_bw,
                    agglomerative_groups, moment_match, sample_union)
from evaluator import fit_named_evaluator, distance_reference
from protocol import INIT_SEEDS, INFERENCE_SEEDS, TOTAL_CONFIGS, UNSAFE_MERGE_NORMALIZED_BW

FIELDS = ['model', 'training_seed', 'method', 'source', 'M', 'K', 'covariance',
          'init_seed', 'inference_seed', 'n_generated', 'val_ll', 'bic_fit',
          'min_eig', 'max_cond', 'chol_ok', 'rare_proxy', 'assigned_rare_fraction',
          'class2_count', 'class2_mmd', 'class2_swd', 'mmd_rbf', 'sliced_wasserstein',
          'status', 'note']


def mean(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def emit(rows, model_name, seed, method, source, M, K, covariance, init_seed,
         val_ll, bic, cm, weights, means, covs, cov_type, decoder,
         clf, nn_rare, radius, tau, Xval, yval, status='OK', note=''):
    for inf_seed in INFERENCE_SEEDS:
        z, _ = sample_from_params(weights, means, covs, cov_type, 2000, inf_seed)
        gen = decode(decoder, z)
        ev = evaluate_generation(gen, clf, nn_rare, radius, tau,
                                 Xval, yval, seed=13000 + inf_seed)
        rows.append({
            'model': model_name, 'training_seed': seed, 'method': method,
            'source': source, 'M': M, 'K': K, 'covariance': covariance,
            'init_seed': init_seed, 'inference_seed': inf_seed, 'n_generated': 2000,
            'val_ll': val_ll, 'bic_fit': bic,
            'min_eig': min(x['min_eig'] for x in cm),
            'max_cond': max(x['cond'] for x in cm),
            'chol_ok': all(x['chol_ok'] for x in cm),
            'rare_proxy': ev['rare_proxy'],
            'assigned_rare_fraction': ev['assigned_rare_fraction'],
            'class2_count': ev['class2_count'],
            'class2_mmd': ev['mmd_class2'], 'class2_swd': ev['swd_class2'],
            'mmd_rbf': ev['mmd_rbf'], 'sliced_wasserstein': ev['sliced_wasserstein'],
            'status': status, 'note': note,
        })


def fit_latent_gmm(Z, K, covariance, init_seed):
    return GaussianMixture(
        n_components=K, covariance_type=covariance, random_state=init_seed,
        n_init=5, max_iter=500, reg_covar=1e-5,
    ).fit(Z)


def build_moment_groups(union, n_clusters):
    means = union['means']
    vars_ = union['variances']
    weights = union['weights']
    n = len(weights)
    D = np.zeros((n, n), dtype=np.float64)
    Dn = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = bures_diag(means[i], vars_[i], means[j], vars_[j])
            D[i, j] = D[j, i] = d
            dn = normalized_bw(d, vars_[i], vars_[j])
            Dn[i, j] = Dn[j, i] = dn
    labels = agglomerative_groups(Dn, n_clusters)
    groups = []
    for g in sorted(set(labels.tolist())):
        idx = np.flatnonzero(labels == g)
        gw, mu, cov = moment_match(idx, weights, means, union['log_sigma2'])
        max_norm = max((Dn[i, j] for i, j in itertools.combinations(idx, 2)), default=0.0)
        mean_norm = float(np.mean([Dn[i, j] for i, j in itertools.combinations(idx, 2)])) if len(idx) > 1 else 0.0
        groups.append({
            'group': int(g), 'indices': idx.tolist(), 'weight_raw': float(gw),
            'mean': mu, 'cov': cov, 'max_normalized_bw': float(max_norm),
            'mean_normalized_bw': mean_norm,
            'unsafe': bool(max_norm > UNSAFE_MERGE_NORMALIZED_BW),
        })
    total = sum(g['weight_raw'] for g in groups)
    for g in groups:
        g['weight'] = g['weight_raw'] / max(total, 1e-12)
    return groups, labels, D, Dn


def safe_k(union, start=3):
    k = start
    while k < 9:
        groups, labels, D, Dn = build_moment_groups(union, k)
        if all(not g['unsafe'] for g in groups):
            return k, groups, labels, D, Dn
        k += 1
    groups, labels, D, Dn = build_moment_groups(union, 9)
    return 9, groups, labels, D, Dn


def group_params(groups):
    weights = np.array([g['weight'] for g in groups], dtype=np.float64)
    weights /= weights.sum()
    means = np.stack([g['mean'] for g in groups], axis=0)
    covs = np.stack([g['cov'] for g in groups], axis=0)
    return weights, means, covs


def main():
    start = time.time()
    Xtr, ytr, Xval, yval, Xte, yte = split_data()
    cfg = load_evaluator_config()
    tau = float(cfg['tau'])
    radius = float(cfg['radius'])
    clf = fit_named_evaluator(cfg['primary_classifier'], Xtr, ytr)
    nn_rare, _ = distance_reference(Xtr, ytr, quantile=0.95)
    selected_m = json.loads((BASE / 'selected_m.json').read_text())['selected_M']
    selected_cov = json.loads((BASE / 'selected_covariance.json').read_text())
    selected_k = json.loads((BASE / 'selected_k.json').read_text())
    rows = []
    moment_rows = []
    pairwise_rows = []
    best_val_ll = -1e9
    last_rare = None
    progress_path = BASE / 'progress.json'
    completed = json.loads(progress_path.read_text()).get('completed', 0) if progress_path.exists() else 0

    for model_name, scenario, seed in MODELS:
        update_progress(stage='FEDERATED_STATISTICS_AGGREGATION', status='RUNNING',
                        current_model=model_name, current_config='D0',
                        completed=completed, total=TOTAL_CONFIGS)
        union = load_union(model_name, scenario, seed)
        Ztr, _ = encode_mean(union['server'], Xtr)
        Zval, _ = encode_mean(union['server'], Xval)
        decoder = union['server']

        # D0: original server GMM.
        d0_w = union['weights']
        d0_means = union['means']
        d0_log = union['log_sigma2']
        d0_covs = np.exp(d0_log)
        d0_cm = cov_metrics(d0_covs)
        d0_val = float(-diag_nll(Zval, d0_w, d0_means, d0_log))
        emit(rows, model_name, seed, 'D0_original', 'server_global', selected_m, 3,
             'diag', '', d0_val, '', d0_cm, d0_w, d0_means, d0_covs, 'diag',
             decoder, clf, nn_rare, radius, tau, Xval, yval,
             status='ORIGINAL', note='original FedVaDE global GMM')
        best_val_ll = max(best_val_ll, d0_val)
        completed += 1
        update_progress(completed=completed, total=TOTAL_CONFIGS,
                        current_config='D0 complete')

        # D1: union sampling + EM, diag and full.
        for cov in ('diag', 'full'):
            for init_seed in INIT_SEEDS:
                gmm, z_fit = fit_and_score_gmm(
                    union, selected_m, 0, 3, cov, init_seed, n_init=5)
                w, mu, cv, _ = gmm_params(gmm)
                cm = cov_metrics(cv)
                val_ll = float(gmm.score(Zval))
                bic = float(gmm.bic(z_fit))
                best_val_ll = max(best_val_ll, val_ll)
                emit(rows, model_name, seed, f'D1_union_em_{cov}', 'union_samples',
                     selected_m, 3, cov, init_seed, val_ll, bic, cm, w, mu, cv,
                     cov, decoder, clf, nn_rare, radius, tau, Xval, yval)
                completed += 1
                update_progress(completed=completed, total=TOTAL_CONFIGS,
                                best_val_ll=float(best_val_ll),
                                current_config=f'D1 {cov} init={init_seed}',
                                rare_mode_proxy=float(rows[-1]['rare_proxy']))
                last_rare = rows[-1]['rare_proxy']

        # D2: moment matching K=3 full.
        groups3, labels3, D3, Dn3 = build_moment_groups(union, 3)
        w3, mu3, cv3 = group_params(groups3)
        cm3 = cov_metrics(cv3)
        val3 = float(-full_nll(Zval, w3, mu3, cv3))
        best_val_ll = max(best_val_ll, val3)
        emit(rows, model_name, seed, 'D2_moment_full', 'client_statistics',
             0, 3, 'full', '', val3, '', cm3, w3, mu3, cv3, 'full',
             decoder, clf, nn_rare, radius, tau, Xval, yval,
             note='moment-matched local Gaussian components; K=3')
        last_rare = rows[-1]['rare_proxy']
        completed += 1
        update_progress(completed=completed, total=TOTAL_CONFIGS,
                        current_config='D2 moment_full complete')
        for g in groups3:
            member = ';'.join(f"client{union['owner'][i]}:comp{i}" for i in g['indices'])
            moment_rows.append({
                'model': model_name, 'training_seed': seed, 'method': 'D2_moment_full',
                'K': 3, 'group': g['group'], 'members': member,
                'group_weight': g['weight'], 'max_normalized_bw': g['max_normalized_bw'],
                'mean_normalized_bw': g['mean_normalized_bw'], 'unsafe': g['unsafe'],
                'min_eig': float(np.linalg.eigvalsh(g['cov']).min()),
                'max_cond': float(np.linalg.cond(g['cov'])),
                'note': 'K=3 forced for fair comparison; unsafe means merge dispersion exceeds threshold',
            })
        for i in range(len(union['weights'])):
            for j in range(i + 1, len(union['weights'])):
                pairwise_rows.append({
                    'model': model_name, 'training_seed': seed,
                    'client_i': int(union['owner'][i]), 'component_i': i,
                    'client_j': int(union['owner'][j]), 'component_j': j,
                    'bures': float(D3[i, j]),
                    'normalized_bures': float(Dn3[i, j]),
                    'unsafe_pair': bool(Dn3[i, j] > UNSAFE_MERGE_NORMALIZED_BW),
                })

        # D2-shrunk using the Stage B selected shrinkage, when applicable.
        if selected_cov.get('base_covariance') == 'full' and selected_cov.get('selected_config') not in ('full', 'diag'):
            resp_union = full_responsibilities(
                np.asarray(sample_union(union, selected_m, 0)[0]), w3, mu3, cv3)
            ne = n_eff(resp_union)
            if selected_cov['selected_config'] == 'shrink_fixed':
                lambdas = np.full(len(w3), float(selected_cov['lambda']))
            else:
                lambdas = float(selected_cov['kappa']) / (ne + float(selected_cov['kappa']))
            cv3s = shrink_covariances(cv3, lambdas)
            cm3s = cov_metrics(cv3s)
            val3s = float(-full_nll(Zval, w3, mu3, cv3s))
            emit(rows, model_name, seed, 'D2_moment_shrunk', 'client_statistics',
                 0, 3, 'shrunk_full', '', val3s, '', cm3s, w3, mu3, cv3s, 'full',
                 decoder, clf, nn_rare, radius, tau, Xval, yval,
                 note=f"Stage B shrinkage {selected_cov['selected_config']}")
            last_rare = rows[-1]['rare_proxy']
            completed += 1
            update_progress(completed=completed, total=TOTAL_CONFIGS,
                            current_config='D2 moment_shrunk complete')

        # D2-safe: increase K until groups are safe.
        ksafe, gsafe, lsave, Dsafe, Dnsafe = safe_k(union, 3)
        if ksafe > 3:
            ws, mus, cvs = group_params(gsafe)
            cms = cov_metrics(cvs)
            vals = float(-full_nll(Zval, ws, mus, cvs))
            emit(rows, model_name, seed, 'D2_safe_full', 'client_statistics',
                 0, ksafe, 'full', '', vals, '', cms, ws, mus, cvs, 'full',
                 decoder, clf, nn_rare, radius, tau, Xval, yval,
                 note=f'K_safe={ksafe}; groups with max normalized BW <= {UNSAFE_MERGE_NORMALIZED_BW}')
            # Flat same-K control.
            gmm_s, z_s = fit_and_score_gmm(union, selected_m, 0, ksafe, 'full', 0, n_init=5)
            wsf, musf, cvsf, _ = gmm_params(gmm_s)
            cmsf = cov_metrics(cvsf)
            valsf = float(gmm_s.score(Zval))
            emit(rows, model_name, seed, 'D1_flat_Ksafe_control', 'union_samples',
                 selected_m, ksafe, 'full', 0, valsf, float(gmm_s.bic(z_s)), cmsf,
                 wsf, musf, cvsf, 'full', decoder, clf, nn_rare, radius, tau,
                 Xval, yval, note=f'flat same-K control for K_safe={ksafe}')
            last_rare = rows[-1]['rare_proxy']
            completed += 1
            update_progress(completed=completed, total=TOTAL_CONFIGS,
                            current_config='D2 safe complete')

        # D3: moment-matched init + EM, K=3 full.
        try:
            precisions = np.stack([np.linalg.inv(c + 1e-6 * np.eye(c.shape[0])) for c in cv3], axis=0)
            gmm3, z3 = fit_and_score_gmm(
                union, selected_m, 0, 3, 'full', 0, n_init=1,
                means_init=mu3, weights_init=w3, precisions_init=precisions)
            init_mode = 'means_weights_precisions'
        except Exception:
            gmm3, z3 = fit_and_score_gmm(
                union, selected_m, 0, 3, 'full', 0, n_init=1,
                means_init=mu3, weights_init=w3)
            init_mode = 'means_weights_only'
        w3e, mu3e, cv3e, _ = gmm_params(gmm3)
        cm3e = cov_metrics(cv3e)
        val3e = float(gmm3.score(Zval))
        emit(rows, model_name, seed, 'D3_moment_init_em_full', 'union_samples',
             selected_m, 3, 'full', 0, val3e, float(gmm3.bic(z3)), cm3e,
             w3e, mu3e, cv3e, 'full', decoder, clf, nn_rare, radius, tau,
             Xval, yval, note=f'init={init_mode}')
        last_rare = rows[-1]['rare_proxy']
        completed += 1
        update_progress(completed=completed, total=TOTAL_CONFIGS,
                        current_config='D3 complete')

        # D4: per-model centralized oracle on real training latents.
        for cov in ('diag', 'full'):
            for K in (3, 6):
                for init_seed in INIT_SEEDS:
                    gmm_o = fit_latent_gmm(Ztr, K, cov, init_seed)
                    w, mu, cv, _ = gmm_params(gmm_o)
                    cm = cov_metrics(cv)
                    val = float(gmm_o.score(Zval))
                    emit(rows, model_name, seed, f'D4_oracle_{cov}_K{K}', 'real_train_latents',
                         len(Ztr), K, cov, init_seed, val, float(gmm_o.bic(Ztr)), cm,
                         w, mu, cv, cov, decoder, clf, nn_rare, radius, tau,
                         Xval, yval, status='ORACLE',
                         note='centralized real-latent oracle; not federated')
                    best_val_ll = max(best_val_ll, val)
                    completed += 1
                    update_progress(completed=completed, total=TOTAL_CONFIGS,
                                    best_val_ll=float(best_val_ll),
                                    current_config=f'D4 {cov} K{K} init={init_seed}',
                                    rare_mode_proxy=float(rows[-1]['rare_proxy']))
                    last_rare = rows[-1]['rare_proxy']

        update_progress(stage='FEDERATED_STATISTICS_AGGREGATION', status='RUNNING',
                        completed=completed, total=TOTAL_CONFIGS,
                        best_val_ll=float(best_val_ll),
                        rare_mode_proxy=float(last_rare))

    # Centralized model oracle.
    c_model, _, c_state = load_server(R10 / 'references/centralized/selected_server.pt')
    Ztr_c, _ = encode_mean(c_model, Xtr)
    Zval_c, _ = encode_mean(c_model, Xval)
    for cov in ('diag', 'full'):
        for K in (3, 6):
            for init_seed in INIT_SEEDS:
                gmm_o = fit_latent_gmm(Ztr_c, K, cov, init_seed)
                w, mu, cv, _ = gmm_params(gmm_o)
                cm = cov_metrics(cv)
                val = float(gmm_o.score(Zval_c))
                emit(rows, 'centralized', 0, f'D4_centralized_{cov}_K{K}',
                     'real_train_latents', len(Ztr_c), K, cov, init_seed, val,
                     float(gmm_o.bic(Ztr_c)), cm, w, mu, cv, cov, c_model,
                     clf, nn_rare, radius, tau, Xval, yval, status='ORACLE',
                     note='centralized model real-latent oracle; not federated')
                best_val_ll = max(best_val_ll, val)
                completed += 1
                update_progress(completed=completed, total=TOTAL_CONFIGS,
                                best_val_ll=float(best_val_ll),
                                current_config=f'D4 centralized {cov} K{K} init={init_seed}',
                                rare_mode_proxy=float(rows[-1]['rare_proxy']))
                last_rare = rows[-1]['rare_proxy']

    write_csv(BASE / 'federated_statistics_aggregation.csv', rows)
    write_csv(BASE / 'moment_matching_diagnostics.csv', moment_rows)
    write_csv(BASE / 'moment_matching_pairwise.csv', pairwise_rows)
    update_progress(stage='FEDERATED_STATISTICS_AGGREGATION', status='RUNNING',
                    current_config='D0-D4 complete', completed=completed,
                    total=TOTAL_CONFIGS, best_val_ll=float(best_val_ll),
                    rare_mode_proxy=float(last_rare))
    print('FEDERATED_AGGREGATION_OK', len(rows), 'elapsed', time.time() - start, flush=True)


if __name__ == '__main__':
    main()
