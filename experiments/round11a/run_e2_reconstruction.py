"""E2: frozen encoder/decoder reconstruction and latent-separability diagnostics."""
import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from common import (BASE, R10, RARE, split_data, update_progress, load_server,
                    load_client, encode_mean, encode_sample, decode,
                    distribution_metrics, mean_std)
from evaluator import (fit_named_evaluator, distance_reference,
                       evaluate_generated)

MODELS = [
    ('centralized', R10 / 'references/centralized/selected_server.pt', 'severe', 0),
    ('fedvade_severe_seed0', R10 / 'formal/severe_seed0/selected_server.pt', 'severe', 0),
    ('fedvade_severe_seed2', R10 / 'formal/severe_seed2/selected_server.pt', 'severe', 2),
    ('fedvade_severe_seed1', R10 / 'formal/severe_seed1/selected_server.pt', 'severe', 1),
]


def write_rows(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fisher_ratio(Z, y):
    Z = np.asarray(Z)
    y = np.asarray(y)
    overall = Z.mean(axis=0)
    between = 0.0
    within = 0.0
    for c in (0, 1, 2):
        m = y == c
        if not np.any(m):
            continue
        mu = Z[m].mean(axis=0)
        between += m.sum() * float(np.sum((mu - overall) ** 2))
        within += float(np.sum((Z[m] - mu) ** 2))
    return float(between / max(within, 1e-12))


def separability_row(Ztr, ytr, Z, y, model, scenario, seed, scope, split):
    sil = float(silhouette_score(Z, y)) if len(np.unique(y)) > 1 else float('nan')
    nn = NearestNeighbors(n_neighbors=1).fit(Ztr)
    pred = ytr[nn.kneighbors(Z, return_distance=False)[:, 0]]
    acc = float(np.mean(pred == y))
    centroids = {}
    for c in (0, 1, 2):
        centroids[c] = Z[y == c].mean(axis=0) if np.any(y == c) else None
    distances = {}
    for c in (0, 1, 2):
        if centroids[c] is None:
            distances[c] = float('nan')
        else:
            others = [centroids[k] for k in (0, 1, 2) if k != c and centroids[k] is not None]
            distances[c] = float(np.linalg.norm(centroids[c] - np.mean(others, axis=0))) if others else float('nan')
    return {
        'model': model, 'scenario': scenario, 'training_seed': seed,
        'scope': scope, 'split': split, 'n': int(len(y)),
        'silhouette': sil,
        'fisher_ratio': fisher_ratio(Z, y),
        'knn1_accuracy': acc,
        'centroid_distance_0': distances[0],
        'centroid_distance_1': distances[1],
        'centroid_distance_2': distances[2],
        'note': 'labels used for diagnostic evaluation only',
    }


def main():
    start = time.time()
    Xtr, ytr, Xval, yval, Xte, yte = split_data()
    cfg = json.loads((BASE / 'evaluator_config.json').read_text())
    tau = float(cfg['tau'])
    radius = float(cfg['radius'])
    clf = fit_named_evaluator(cfg['primary_classifier'], Xtr, ytr)
    nn_rare, _, _ = distance_reference(Xtr, ytr, quantile=0.95)

    recon_rows = []
    latent_rows = []
    for model_name, path, scenario, seed in MODELS:
        update_progress(stage='ORACLE_LATENT_DIAGNOSIS', status='RUNNING',
                        current_model=model_name, current_experiment='E2-reconstruction',
                        progress_current=4, progress_total=12)
        print('RECON', model_name, flush=True)
        model, payload, state = load_server(path)
        Ztr_mean, _ = encode_mean(model, Xtr)
        Zval_mean, _ = encode_mean(model, Xval)
        Zte_mean, _ = encode_mean(model, Xte)

        # Linear probe: does the latent retain linearly decodable feature information?
        ridge = Ridge(alpha=1.0)
        ridge.fit(Ztr_mean, Xtr)
        linear_val = ridge.predict(Zval_mean)
        linear_test = ridge.predict(Zte_mean)

        reconstructions = {}
        for split, X, Zmean, Zsample in (
                ('val', Xval, Zval_mean, encode_sample(model, Xval, seed=0)),
                ('test', Xte, Zte_mean, encode_sample(model, Xte, seed=0))):
            reconstructions[split] = {
                'encmean': decode(model, Zmean),
                'posterior': decode(model, Zsample),
            }
        reconstructions['val']['linear'] = linear_val
        reconstructions['test']['linear'] = linear_test

        for split, X, y in (('val', Xval, yval), ('test', Xte, yte)):
            for cls in (0, 1, 2, 'all'):
                mask = np.ones(len(y), dtype=bool) if cls == 'all' else (y == cls)
                real = np.asarray(X)[mask]
                for method, recon in (('encoder_mean', reconstructions[split]['encmean']),
                                      ('posterior_sample', reconstructions[split]['posterior']),
                                      ('linear_probe', reconstructions[split]['linear'])):
                    r = np.asarray(recon)[mask]
                    mse = float(np.mean((r - real) ** 2)) if len(real) else float('nan')
                    ev = evaluate_generated(r, clf, nn_rare, radius, tau)
                    dist = distribution_metrics(real, r, seed=1100 + cls if isinstance(cls, int) else 1104)
                    recon_rows.append({
                        'model': model_name, 'scenario': scenario, 'training_seed': seed,
                        'split': split, 'class': cls, 'method': method, 'n': int(len(real)),
                        'reconstruction_mse': mse,
                        'rare_proxy': ev['rare_proxy'] if cls == RARE else '',
                        'assigned_rare_fraction': ev['assigned_rare_fraction'] if cls == RARE else '',
                        'p_rare_mean': ev['classifier_rare_prob_mean'],
                        'distance_to_train_rare_mean': ev['distance_to_rare_mean'],
                        'distance_pass_fraction': ev['distance_pass_fraction'],
                        'mmd_rbf': dist.get('mmd_rbf'),
                        'sliced_wasserstein': dist.get('sliced_wasserstein'),
                        'note': 'test rows are exploratory' if split == 'test' else '',
                    })

        for split, Z, y in (('val', Zval_mean, yval), ('test', Zte_mean, yte)):
            latent_rows.append(separability_row(Ztr_mean, ytr, Z, y,
                                                model_name, scenario, seed, 'server', split))

        # Local client-C separability for FedVaDE severe models.
        if model_name.startswith('fedvade'):
            last_full = R10 / 'formal' / f'{scenario}_seed{seed}' / 'last_full.pt'
            if last_full.is_file():
                lf = __import__('torch').load(last_full, map_location='cpu')
                clients = lf.get('clients', [])
                if len(clients) > 2:
                    partitions = np.load(R10 / 'private_data' / f'{scenario}_partitions.npz')
                    ids = partitions['client_2']
                    client_model = load_client(clients[2])
                    Ztr_c, _ = encode_mean(client_model, np.asarray(Xtr)[ids])
                    ytr_c = np.asarray(ytr)[ids]
                    Zval_c, _ = encode_mean(client_model, Xval)
                    Zte_c, _ = encode_mean(client_model, Xte)
                    latent_rows.append(separability_row(Ztr_c, ytr_c, Zval_c, yval,
                                                       model_name, scenario, seed, 'client_C', 'val'))
                    latent_rows.append(separability_row(Ztr_c, ytr_c, Zte_c, yte,
                                                       model_name, scenario, seed, 'client_C', 'test'))

        # Progress / best result: use rare-class reconstruction MSE.
        rare_rows = [r for r in recon_rows if r['model'] == model_name and r['split'] == 'val'
                     and r['class'] == RARE and r['method'] == 'encoder_mean']
        best = rare_rows[0]['reconstruction_mse'] if rare_rows else None
        update_progress(progress_current=5, rare_mode_proxy=None,
                        best_result=f'E2 recon {model_name} rare_mse={best:.4f}' if best is not None else None)

    write_rows(BASE / 'reconstruction_diagnostics.csv', recon_rows)
    write_rows(BASE / 'latent_separability.csv', latent_rows)
    print('E2_RECON_OK', len(recon_rows), len(latent_rows), 'elapsed', time.time() - start, flush=True)


if __name__ == '__main__':
    main()
