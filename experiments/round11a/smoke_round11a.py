"""Short smoke test for Round 11A code paths. Does not touch formal result files."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from common import (BASE, R10, RARE, split_data, load_server, load_client, encode_mean,
                    encode_sample, decode, softmax_weights, sample_diag_gmm, generate_global,
                    distribution_metrics, cka_linear)
from evaluator import (fit_primary_evaluator, distance_reference, select_threshold,
                       evaluate_generated, prob_rare)

OUT = BASE / 'smoke'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    Xtr, ytr, Xval, yval, Xte, yte = split_data()
    model, payload, state = load_server(R10 / 'formal/severe_seed0/selected_server.pt')
    clf = fit_primary_evaluator(Xtr, ytr, n_estimators=50, cv=3)
    p_val = prob_rare(clf, Xval)
    tau, rule, table, best = select_threshold(p_val, yval)
    nn_rare, radius, _ = distance_reference(Xtr, ytr, quantile=0.95)

    rare_idx = np.flatnonzero(yval == RARE)[:10]
    mu, logvar = encode_mean(model, Xval[rare_idx])
    recon = decode(model, mu)
    mse = float(np.mean((recon - Xval[rare_idx]) ** 2))
    sample = decode(model, encode_sample(model, Xval[rare_idx], seed=0))
    mse_sample = float(np.mean((sample - Xval[rare_idx]) ** 2))

    gen, _, _, _ = generate_global(model, 100, 0)
    ev_global = evaluate_generated(gen, clf, nn_rare, radius, tau)

    lf = torch.load(R10 / 'formal/severe_seed0/last_full.pt', map_location='cpu')
    clients = lf['clients']
    partitions = np.load(R10 / 'private_data/severe_partitions.npz')
    sizes = np.array([len(partitions[f'client_{i}']) for i in range(3)], dtype=float)
    cw = sizes / sizes.sum()
    weights, means, logs = [], [], []
    for i, cs in enumerate(clients):
        pi = softmax_weights(cs['rho_'])
        weights.append(cw[i] * pi)
        means.append(cs['mu_c'].numpy())
        logs.append(cs['log_sigma2_c'].numpy())
    weights = np.concatenate(weights)
    weights /= weights.sum()
    means = np.concatenate(means, axis=0)
    logs = np.concatenate(logs, axis=0)
    z, labels = sample_diag_gmm(weights, means, logs, 100, 0)
    ev_union = evaluate_generated(decode(model, z), clf, nn_rare, radius, tau)

    Ztr, _ = encode_mean(model, Xtr[:300])
    gmm = GaussianMixture(n_components=3, covariance_type='diag', random_state=42, n_init=2).fit(Ztr)
    val_ll = float(gmm.score(encode_mean(model, Xval)[0]))
    cka = cka_linear(encode_mean(model, Xval)[0], encode_mean(load_client(clients[0]), Xval)[0])

    result = {
        'ok': True,
        'tau': tau, 'tau_rule': rule, 'radius': radius,
        'recon_mse_10_rare_mean': mse,
        'recon_mse_10_rare_sample': mse_sample,
        'global_rare_proxy_100': ev_global['rare_proxy'],
        'union_server_rare_proxy_100': ev_union['rare_proxy'],
        'gmm3_val_ll_300train': val_ll,
        'cka_client0_server_val': cka,
        'elapsed_seconds': time.time() - t0,
    }
    (OUT / 'smoke_results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
