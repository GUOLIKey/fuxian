"""Held-out CTG clustering and exploratory synthetic-sample diagnostics."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_rand_score, classification_report
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

ROOT = Path('/root/autodl-tmp/fedvade_project')
BASE = ROOT / 'reproduction_runs/round10_non_iid_generation'
sys.path.insert(0, str(BASE))
from generation_tools import distribution_metrics, generate
sys.path.insert(0, str(ROOT / 'fuxian-round2/FedVaDE_source'))
from model import VaDE
from argparse import Namespace


def csv_write(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def update_progress(**changes):
    path = BASE / 'progress.json'
    current = json.loads(path.read_text()) if path.exists() else {}
    current.update(changes, updated_at=time.time())
    temp = Path(str(path) + '.tmp')
    temp.write_text(json.dumps(current, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def load(path):
    payload = torch.load(path, map_location='cpu')
    state = payload['server']
    k = len(state['rho_'])
    model = VaDE(Namespace(input_dim=21, hid_dim=10, nClusters=k, alpha=.01, cuda=False)).cpu()
    model.load_state_dict(state, strict=True); model.eval()
    return model


def purity(true, pred):
    return sum(max(int(np.sum((pred == k) & (true == y))) for y in np.unique(true))
               for k in np.unique(pred)) / len(true)


def score_model(model, X, y):
    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32)
        mu, _ = model.encoder(x)
        posterior = torch.softmax(torch.log_softmax(model.rho_, 0).unsqueeze(0) +
                    model.gaussian_pdfs_log(mu, model.mu_c, model.log_sigma2_c), 1)
        pred = posterior.argmax(1).numpy()
        recon = model.decoder(mu).numpy()
    return {'ari_test': float(adjusted_rand_score(y, pred)),
            'purity_test': float(purity(y, pred)),
            'reconstruction_mse_test': float(np.mean((recon - X) ** 2))}, \
           mu.numpy(), posterior.numpy(), recon


def models():
    runs = []
    for scenario in ('iid', 'medium', 'severe'):
        seeds = (0, 1, 2) if scenario == 'severe' else (0,)
        for seed in seeds:
            runs.append((f'fedvade_{scenario}_seed{seed}',
                         BASE / f'formal/{scenario}_seed{seed}/selected_server.pt',
                         scenario, seed, 'label_blind_fedvade_baseline'))
    for kind in ('local_A', 'centralized'):
        runs.append((kind, BASE / f'references/{kind}/selected_server.pt',
                     'severe', 0, 'standalone_VaDE_reference'))
    return runs


def main():
    if (BASE / 'noniid_baseline_metrics.csv').exists():
        raise SystemExit('Evaluation exists; refusing overwrite')
    torch.set_num_threads(1)
    data = np.load(BASE / 'private_data/data.npz')
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    classifier = RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample',
                                         random_state=42, n_jobs=1).fit(X_train, y_train)
    class_pred = classifier.predict(X_test)
    reliability = classification_report(y_test, class_pred, labels=[0, 1, 2],
                                        output_dict=True, zero_division=0)
    (BASE / 'reference_classifier_test.json').write_text(json.dumps({
        'training_source': 'model-training split only', 'test_source': 'sealed grouped test split',
        'model': 'RandomForestClassifier(n_estimators=300,class_weight=balanced_subsample,random_state=42)',
        'metrics': reliability}, indent=2) + '\n')
    rare_reliable = (reliability['2']['precision'] >= .8 and reliability['2']['recall'] >= .8)
    rare_real = X_test[y_test == 2]
    self_distance = NearestNeighbors(n_neighbors=2).fit(rare_real).kneighbors(rare_real)[0][:, 1]
    rare_radius = float(np.quantile(self_distance, .95))
    nearest_rare = NearestNeighbors(n_neighbors=1).fit(rare_real)
    aggregate = []; inference = []; classwise = []; coverage = []; latent_rows = []
    for name, path, scenario, seed, kind in models():
        if not path.is_file():
            raise FileNotFoundError(f'Expected trained model missing: {path}')
        model = load(path)
        test, mu, posterior, recon = score_model(model, X_test, y_test)
        config_path = path.parent / 'config.json'
        conf = json.loads(config_path.read_text())
        aggregate.append({'model': name, 'scenario': scenario, 'training_seed': seed,
                          'kind': kind, 'selected_epoch': conf.get('best_validation', {}).get('epoch',
                              conf.get('selected_epoch')), 'checkpoint_sha256': sha(path),
                          'evaluation': 'held_out_test_fixed_unlabeled_selection', **test})
        for cls in (0, 1, 2):
            mask = y_test == cls
            centroid = mu[mask].mean(0)
            other = mu[~mask].mean(0)
            latent_rows.append({'model': name, 'class': cls, 'n_test': int(mask.sum()),
                                'latent_mean_norm': float(np.linalg.norm(centroid)),
                                'centroid_distance_to_others': float(np.linalg.norm(centroid - other)),
                                'reconstruction_mse': float(np.mean((recon[mask] - X_test[mask]) ** 2)),
                                'posterior_mean': json.dumps(posterior[mask].mean(0).tolist()),
                                'posterior_entropy_mean': float(np.mean(-np.sum(
                                     posterior[mask] * np.log(np.maximum(posterior[mask], 1e-12)), axis=1)))})
        for inference_seed in range(5):
            generated, _, component, weights = generate(model, 2000, inference_seed)
            probs = classifier.predict_proba(generated)
            labels = probs.argmax(1)
            confidence = probs.max(1)
            dist = distribution_metrics(X_test, generated, 100 + inference_seed)
            inference.append({'model': name, 'scenario': scenario, 'training_seed': seed,
                              'inference_seed': inference_seed, 'n_generated': len(generated),
                              'component_weights': json.dumps(weights.tolist()), **dist})
            for cls in (0, 1, 2):
                mask = labels == cls
                dist_cls = distribution_metrics(X_test[y_test == cls], generated[mask],
                                                200 + inference_seed)
                classwise.append({'model': name, 'scenario': scenario, 'training_seed': seed,
                                  'inference_seed': inference_seed, 'class': cls,
                                  'n_assigned': int(mask.sum()), 'assigned_fraction': float(mask.mean()),
                                  'high_conf_fraction_total': float(np.mean(mask & (confidence >= .8))),
                                  'reference_test_precision': reliability[str(cls)]['precision'],
                                  'reference_test_recall': reliability[str(cls)]['recall'],
                                  'class_reliable': (reliability[str(cls)]['precision'] >= .8 and
                                                     reliability[str(cls)]['recall'] >= .8), **dist_cls})
            distance_to_rare = nearest_rare.kneighbors(generated)[0][:, 0]
            assigned_rare = labels == 2
            trusted = assigned_rare & (confidence >= .8) & (distance_to_rare <= rare_radius)
            coverage.append({'model': name, 'scenario': scenario, 'training_seed': seed,
                             'inference_seed': inference_seed,
                             'test_rare_fraction': float(np.mean(y_test == 2)),
                             'assigned_rare_fraction': float(assigned_rare.mean()),
                             'high_conf_rare_fraction': float(np.mean(assigned_rare & (confidence >= .8))),
                             'rare_mode_recovery_proxy': float(trusted.mean()),
                             'rare_nearest_neighbor_radius_test_q95': rare_radius,
                             'generated_to_rare_nn_mean': float(distance_to_rare.mean()),
                             'reference_rare_precision': reliability['2']['precision'],
                             'reference_rare_recall': reliability['2']['recall'],
                             'rare_reference_reliable': rare_reliable,
                             'interpretation': 'proxy_uncertain' if not rare_reliable else 'classifier_and_distance_proxy'})
        current_coverage = [r for r in coverage if r['model'] == name]
        current_inference = [r for r in inference if r['model'] == name]
        update_progress(current_run=name,
                        rare_mode_coverage=float(np.mean([r['rare_mode_recovery_proxy']
                                                          for r in current_coverage])),
                        generation_mmd=float(np.mean([r['mmd_rbf'] for r in current_inference])))
        print('EVALUATED', name, test, flush=True)
    csv_write(BASE / 'noniid_baseline_metrics.csv', aggregate, list(aggregate[0]))
    csv_write(BASE / 'generation_inference_records.csv', inference, list(inference[0]))
    csv_write(BASE / 'classwise_generation_metrics.csv', classwise, list(classwise[0]))
    csv_write(BASE / 'rare_mode_coverage.csv', coverage, list(coverage[0]))
    csv_write(BASE / 'latent_diagnostics.csv', latent_rows, list(latent_rows[0]))
    severe = [r for r in coverage if r['model'] == 'fedvade_severe_seed0']
    rare_proxy = float(np.mean([r['rare_mode_recovery_proxy'] for r in severe]))
    weak = rare_reliable and rare_proxy < .5 * float(np.mean(y_test == 2))
    (BASE / 'simple_baseline_decision.json').write_text(json.dumps({
        'run': bool(weak), 'rule': 'run only if reference rare precision and recall >=0.8 and severe global recovery proxy < half test rare fraction',
        'reference_rare_reliable': bool(rare_reliable), 'rare_proxy': rare_proxy,
        'test_rare_fraction': float(np.mean(y_test == 2))}, indent=2) + '\n')
    if weak:
        frozen = load(BASE / 'formal/severe_seed0/selected_server.pt')
        with torch.no_grad():
            z_train = frozen.encoder(torch.tensor(X_train))[0].numpy()
        rows = []
        for components in (3, 6):
            gmm = GaussianMixture(n_components=components, covariance_type='diag',
                                  random_state=42, n_init=5).fit(z_train)
            frozen.rho_.data = torch.log(torch.tensor(gmm.weights_, dtype=torch.float32) + 1e-10)
            frozen.mu_c.data = torch.tensor(gmm.means_, dtype=torch.float32)
            frozen.log_sigma2_c.data = torch.log(torch.tensor(gmm.covariances_, dtype=torch.float32))
            for inference_seed in range(5):
                generated, _, _, _ = generate(frozen, 2000, inference_seed)
                probs = classifier.predict_proba(generated)
                pred = probs.argmax(1); confidence = probs.max(1)
                near = nearest_rare.kneighbors(generated)[0][:, 0]
                recovered = (pred == 2) & (confidence >= .8) & (near <= rare_radius)
                rows.append({'model': f'centralized_latent_refit_K{components}',
                             'K': components, 'inference_seed': inference_seed,
                             'rare_mode_recovery_proxy': float(recovered.mean()),
                             'assigned_rare_fraction': float(np.mean(pred == 2)),
                             'mmd_rbf': distribution_metrics(X_test, generated, inference_seed)['mmd_rbf'],
                             'privacy_note': 'centralized latent fit is a diagnostic upper bound, not federated deployable'})
        csv_write(BASE / 'simple_baseline_comparison.csv', rows, list(rows[0]))
    else:
        csv_write(BASE / 'simple_baseline_comparison.csv', [],
                  ['model', 'K', 'inference_seed', 'rare_mode_recovery_proxy',
                   'assigned_rare_fraction', 'mmd_rbf', 'privacy_note'])
    print('REFERENCE_RELIABLE', rare_reliable, 'SIMPLE_BASELINE', weak, flush=True)


if __name__ == '__main__':
    main()
