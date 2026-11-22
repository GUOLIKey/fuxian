"""E2/D0: compare the server global GMM with the client-union MoG under a frozen decoder."""
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import (BASE, R10, RARE, split_data, update_progress, load_server, load_client,
                    encode_mean, decode, softmax_weights, diag_nll, sample_diag_gmm,
                    generate_global, distribution_metrics, cka_linear, cosine_mean, knn_overlap)
from evaluator import (fit_named_evaluator, distance_reference, evaluate_generated,
                      predict_all)

MODELS = [
    ('fedvade_severe_seed0', R10 / 'formal/severe_seed0', 'severe', 0),
    ('fedvade_severe_seed2', R10 / 'formal/severe_seed2', 'severe', 2),
    ('fedvade_severe_seed1', R10 / 'formal/severe_seed1', 'severe', 1),
]


def write_rows(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def classwise_metrics(generated, clf, Xval, yval, seed):
    pred, _, _ = predict_all(clf, generated)
    out = {}
    for c in (0, 1, 2):
        g = generated[pred == c]
        r = np.asarray(Xval)[np.asarray(yval) == c]
        if len(g) >= 2 and len(r) >= 2:
            d = distribution_metrics(r, g, seed=seed + c)
            out[f'mmd_class{c}'] = d.get('mmd_rbf')
            out[f'swd_class{c}'] = d.get('sliced_wasserstein')
        else:
            out[f'mmd_class{c}'] = None
            out[f'swd_class{c}'] = None
    return out


def decode_with_owner(z, labels, owner, client_models):
    out = np.zeros((len(z), 21), dtype=np.float64)
    for k in range(len(owner)):
        idx = np.flatnonzero(labels == k)
        if len(idx):
            out[idx] = decode(client_models[owner[k]], z[idx])
    return out


def evaluate_source(name, scenario, seed, source, decoder_name, mode, component, inference_seed,
                    generated, clf, nn_rare, radius, tau, Xval, yval, compatibility, cka_mean,
                    union_nll, global_nll, note):
    ev = evaluate_generated(generated, clf, nn_rare, radius, tau)
    dist = distribution_metrics(Xval, generated, seed=3000 + inference_seed)
    cw = classwise_metrics(generated, clf, Xval, yval, seed=4000 + inference_seed)
    return {
        'model': name, 'scenario': scenario, 'training_seed': seed,
        'source': source, 'decoder': decoder_name, 'mode': mode,
        'component': component if component is not None else '',
        'inference_seed': inference_seed, 'n': int(len(generated)),
        'rare_proxy': ev['rare_proxy'],
        'assigned_rare_fraction': ev['assigned_rare_fraction'],
        'high_conf_rare_fraction': ev['high_conf_rare_fraction'],
        'classifier_rare_prob_mean': ev['classifier_rare_prob_mean'],
        'distance_to_rare_mean': ev['distance_to_rare_mean'],
        'distance_pass_fraction': ev['distance_pass_fraction'],
        'mmd_rbf': dist.get('mmd_rbf'),
        'sliced_wasserstein': dist.get('sliced_wasserstein'),
        **cw,
        'compatibility_status': compatibility,
        'cka_mean': cka_mean,
        'union_nll': union_nll,
        'global_nll': global_nll,
        'note': note,
    }


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

    # Centralized global-only reference.
    update_progress(stage='GLOBAL_VS_UNION_MOG', status='RUNNING',
                    current_model='centralized', current_experiment='E2-global',
                    progress_current=6, progress_total=12)
    c_model, _, _ = load_server(R10 / 'references/centralized/selected_server.pt')
    for seed in range(5):
        gen, _, _, _ = generate_global(c_model, 2000, seed)
        row = evaluate_source('centralized', 'severe', 0, 'global', 'server',
                              'unconditional', None, seed, gen, clf, nn_rare, radius, tau,
                              Xval, yval, 'N/A', float('nan'), float('nan'), float('nan'),
                              'centralized reference; no client union')
        rows.append(row)
        best = max(best, row['rare_proxy'])
    for comp in range(3):
        gen, _, _, _ = generate_global(c_model, 1000, 0, component=comp)
        row = evaluate_source('centralized', 'severe', 0, 'global', 'server',
                              'component_conditioned', comp, 0, gen, clf, nn_rare, radius, tau,
                              Xval, yval, 'N/A', float('nan'), float('nan'), float('nan'),
                              'component-conditioned diagnostic')
        rows.append(row)

    for name, run, scenario, seed in MODELS:
        update_progress(stage='GLOBAL_VS_UNION_MOG', status='RUNNING',
                        current_model=name, current_experiment='E2-global-vs-union',
                        progress_current=7 if seed == 0 else 8, progress_total=12)
        print('UNION', name, flush=True)
        server, payload, state = load_server(run / 'selected_server.pt')
        lf = torch.load(run / 'last_full.pt', map_location='cpu')
        client_states = lf.get('clients', [])
        if len(client_states) != 3:
            raise RuntimeError(f'{name}: expected 3 clients, got {len(client_states)}')
        client_models = [load_client(s) for s in client_states]
        partitions = np.load(R10 / 'private_data' / f'{scenario}_partitions.npz')
        sizes = np.array([len(partitions[f'client_{i}']) for i in range(3)], dtype=float)
        client_weights = sizes / sizes.sum()

        weights, means, log_sigma2, owner = [], [], [], []
        for i, cs in enumerate(client_states):
            pi = softmax_weights(cs['rho_'])
            weights.append(client_weights[i] * pi)
            means.append(cs['mu_c'].numpy())
            log_sigma2.append(cs['log_sigma2_c'].numpy())
            owner.extend([i] * len(pi))
        weights = np.concatenate(weights)
        weights = weights / weights.sum()
        means = np.concatenate(means, axis=0)
        log_sigma2 = np.concatenate(log_sigma2, axis=0)
        owner = np.asarray(owner, dtype=int)

        Zval_server, _ = encode_mean(server, Xval)
        Zval_clients = [encode_mean(cm, Xval)[0] for cm in client_models]
        cka_values = [cka_linear(Zval_server, z) for z in Zval_clients]
        cosine_values = [cosine_mean(Zval_server, z) for z in Zval_clients]
        overlap_values = [knn_overlap(Zval_server, z, k=10) for z in Zval_clients]
        cka_mean = float(np.mean(cka_values))
        global_nll = diag_nll(Zval_server, softmax_weights(state['rho_']),
                              state['mu_c'].numpy(), state['log_sigma2_c'].numpy())
        union_nll = diag_nll(Zval_server, weights, means, log_sigma2)
        if cka_mean >= 0.80 and union_nll <= global_nll + 1.0:
            compatibility = 'OK'
        elif cka_mean >= 0.50 and union_nll <= global_nll + 5.0:
            compatibility = 'LIMITED'
        else:
            compatibility = 'INCOMPATIBLE'
        compat_note = (f'cka_mean={cka_mean:.4f}; union_nll={union_nll:.4f}; '
                       f'global_nll={global_nll:.4f}; client_cka={[round(v,4) for v in cka_values]}; '
                       f'client_cos={[round(v,4) for v in cosine_values]}; '
                       f'client_knn10={[round(v,4) for v in overlap_values]}')

        # Global GMM with server decoder.
        for inf_seed in range(5):
            gen, _, _, _ = generate_global(server, 2000, inf_seed)
            row = evaluate_source(name, scenario, seed, 'global', 'server',
                                  'unconditional', None, inf_seed, gen, clf, nn_rare, radius, tau,
                                  Xval, yval, compatibility, cka_mean, union_nll, global_nll,
                                  compat_note)
            rows.append(row)
            best = max(best, row['rare_proxy'])
        for comp in range(3):
            gen, _, _, _ = generate_global(server, 1000, 0, component=comp)
            rows.append(evaluate_source(name, scenario, seed, 'global', 'server',
                                        'component_conditioned', comp, 0, gen, clf, nn_rare,
                                        radius, tau, Xval, yval, compatibility, cka_mean,
                                        union_nll, global_nll, compat_note))

        # Union MoG with the same server decoder (required D0 comparison).
        for inf_seed in range(5):
            z, labels = sample_diag_gmm(weights, means, log_sigma2, 2000, inf_seed)
            gen = decode(server, z)
            row = evaluate_source(name, scenario, seed, 'union', 'server',
                                  'unconditional', None, inf_seed, gen, clf, nn_rare, radius, tau,
                                  Xval, yval, compatibility, cka_mean, union_nll, global_nll,
                                  compat_note)
            rows.append(row)
            best = max(best, row['rare_proxy'])
        for comp in range(len(weights)):
            z, labels = sample_diag_gmm(weights, means, log_sigma2, 1000, 0, component=comp)
            gen = decode(server, z)
            rows.append(evaluate_source(name, scenario, seed, 'union', 'server',
                                        'component_conditioned', comp, 0, gen, clf, nn_rare,
                                        radius, tau, Xval, yval, compatibility, cka_mean,
                                        union_nll, global_nll, compat_note))

        # Supplementary: union MoG with the matched client decoder.
        for inf_seed in range(5):
            z, labels = sample_diag_gmm(weights, means, log_sigma2, 2000, inf_seed)
            gen = decode_with_owner(z, labels, owner, client_models)
            row = evaluate_source(name, scenario, seed, 'union', 'client',
                                  'unconditional', None, inf_seed, gen, clf, nn_rare, radius, tau,
                                  Xval, yval, compatibility, cka_mean, union_nll, global_nll,
                                  compat_note + '; matched client decoder is supplementary')
            rows.append(row)
            best = max(best, row['rare_proxy'])
        for comp in range(len(weights)):
            z, labels = sample_diag_gmm(weights, means, log_sigma2, 1000, 0, component=comp)
            gen = decode_with_owner(z, labels, owner, client_models)
            rows.append(evaluate_source(name, scenario, seed, 'union', 'client',
                                        'component_conditioned', comp, 0, gen, clf, nn_rare,
                                        radius, tau, Xval, yval, compatibility, cka_mean,
                                        union_nll, global_nll, compat_note))

        update_progress(progress_current=8, rare_mode_proxy=float(best),
                        best_result=f'E2 best rare_proxy={best:.4f}')

    write_rows(BASE / 'global_vs_union_mog.csv', rows)
    print('E2_GLOBAL_UNION_OK', len(rows), 'elapsed', time.time() - start, flush=True)


if __name__ == '__main__':
    main()
