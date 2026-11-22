"""Frozen author-weight and F01 generative comparison under author CSV protocol."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path('/root/autodl-tmp/fedvade_project')
OUT = ROOT / 'reproduction_runs/round10_non_iid_generation'
sys.path.insert(0, str(OUT))
from generation_tools import distribution_metrics, generate
sys.path.insert(0, str(ROOT / 'reproduction_runs/round6_checkpoint_diagnosis'))
from diagnose import PARAMS, VaDE, data


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    output = OUT / 'breast_cancer_generation.csv'
    if output.exists():
        raise SystemExit('Breast audit already exists')
    torch.set_num_threads(1)
    _, X, y, clients, _, _, _ = data()
    observed = np.concatenate([loader.dataset.tensors[0].numpy() for loader in clients])
    candidates = {
        'author_weight': ROOT / 'fuxian-round2/FedVaDE_source/model/server_model_breast_cancer.pk',
        'round7_F01_epoch100': ROOT / 'reproduction_runs/round7_hyperparameter_search/fine/F01/epoch_100_server.pt'}
    rows = []; latent = []
    for name, path in candidates.items():
        payload = torch.load(path, map_location='cpu')
        state = payload['server'] if 'server' in payload else payload
        model = VaDE(PARAMS).cpu(); model.load_state_dict(state, strict=True); model.eval()
        with torch.no_grad():
            mu, logvar = model.encoder(torch.from_numpy(X))
            recon = model.decoder(mu)
            posterior = torch.softmax(torch.log_softmax(model.rho_, 0).unsqueeze(0) +
                                      model.gaussian_pdfs_log(mu, model.mu_c, model.log_sigma2_c), 1)
        for label in np.unique(y):
            mask = y == label
            latent.append({'model': name, 'real_label': int(label), 'n': int(mask.sum()),
                           'reconstruction_mse': float(torch.mean((recon[mask] - torch.from_numpy(X[mask])) ** 2)),
                           'posterior_mean': json.dumps(posterior[mask].mean(0).tolist()),
                           'latent_centroid': json.dumps(mu[mask].mean(0).tolist()),
                           'latent_std': float(mu[mask].std())})
        for seed in range(5):
            gen, _, comp, weights = generate(model, 2000, seed)
            overall = distribution_metrics(observed, gen, seed)
            rows.append({'model': name, 'seed': seed, 'component': 'unconditional',
                         'n_generated': len(gen), 'weight': 1.0, 'checkpoint_sha256': sha(path),
                         'evaluation': 'author_protocol_diagnostic', **overall})
            for component in range(len(weights)):
                selected, _, _, _ = generate(model, 1000, seed, component)
                rows.append({'model': name, 'seed': seed, 'component': component,
                             'n_generated': len(selected), 'weight': float(weights[component]),
                             'checkpoint_sha256': sha(path),
                             'evaluation': 'component_conditioned_not_true_class',
                             **distribution_metrics(observed, selected, seed)})
    with output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (OUT / 'breast_latent_diagnostics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(latent[0])); writer.writeheader(); writer.writerows(latent)
    print('BREAST_GENERATION', len(rows), flush=True)


if __name__ == '__main__':
    main()
