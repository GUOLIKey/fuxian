"""Build deterministic warm-start snapshots for the 2x2 continuation experiment.

The author and F01 epoch-100 checkpoints contain server weights only.  Exact
historical client-state continuation is therefore impossible.  This script
initializes clients deterministically from the delivered server weights (no
local-GMM fitting, no pretraining) and saves paired OFF/ON snapshots.  The
limitation is recorded explicitly in the feasibility report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

ROOT = Path('/root/autodl-tmp/fedvade_project')
SOURCE = ROOT / 'fuxian-round2/FedVaDE_source'
EXP = ROOT / 'experiments/multica_pretrained_2x2_v1'
DATA = SOURCE / 'dataset/breast_cancer.csv'
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(EXP / 'scripts'))
from dataloader import make_noniid_clients  # noqa: E402
from model import VaDE  # noqa: E402
import multica_core as core  # noqa: E402

MODELS = {
    'AUTHOR': {
        'path': SOURCE / 'model/server_model_breast_cancer.pk',
        'alpha': 0.01, 'schedule': 'author', 'historical_ari': 0.6395,
        'label': 'A/B author pretrained server',
    },
    'F01': {
        'path': ROOT / 'reproduction_runs/round7_hyperparameter_search/fine/F01/epoch_100_server.pt',
        'alpha': 0.05, 'schedule': 'epoch', 'historical_ari': 0.6140,
        'label': 'C/D F01 epoch-100 server',
    },
}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tensor_bytes(x):
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x).tobytes()


def state_hash(state, exclude_gmm=False):
    h = hashlib.sha256()
    for k in sorted(state):
        if exclude_gmm and k in ('rho_', 'mu_c', 'log_sigma2_c'):
            continue
        h.update(k.encode())
        h.update(tensor_bytes(state[k]))
    return h.hexdigest()


def nested_hash(states):
    h = hashlib.sha256()
    for s in states:
        h.update(state_hash(s).encode())
    return h.hexdigest()


def load_payload(path):
    obj = torch.load(path, map_location='cpu')
    if isinstance(obj, dict) and 'server' in obj:
        return obj['server'], obj.get('round'), obj
    return obj, None, {'server_only_state_dict': True}


def make_vade(k, alpha):
    return VaDE(SimpleNamespace(input_dim=31, hid_dim=10, nClusters=int(k),
                                alpha=float(alpha), cuda=False))


def encode_mean(model, x, batch=2048):
    out = []
    with torch.no_grad():
        x = torch.tensor(np.asarray(x, dtype=np.float32))
        for i in range(0, len(x), batch):
            mu, _ = model.encoder(x[i:i + batch])
            out.append(mu.numpy())
    return np.concatenate(out, 0)


def diag_log_prob(z, state):
    w = torch.softmax(state['rho_'].float(), 0).numpy().astype(np.float64)
    mu = state['mu_c'].float().numpy().astype(np.float64)
    lv = state['log_sigma2_c'].float().numpy().astype(np.float64)
    z = np.asarray(z, dtype=np.float64)
    diff = z[:, None, :] - mu[None, :, :]
    lp = -0.5 * np.sum(np.log(2 * np.pi) + lv[None, :, :] +
                       diff ** 2 / np.exp(lv)[None, :, :], axis=2)
    return lp + np.log(np.maximum(w, 1e-12))[None, :]


def diag_nll(z, state):
    lp = diag_log_prob(z, state)
    m = lp.max(1, keepdims=True)
    return float(-np.mean(m.ravel() + np.log(np.exp(lp - m).sum(1))))


def cluster_metrics(y_true, y_pred):
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    y_true = np.asarray(y_true, dtype=int); y_pred = np.asarray(y_pred, dtype=int)
    d = int(max(y_true.max(), y_pred.max())) + 1
    conf = np.zeros((d, d), dtype=np.int64)
    for a, b in zip(y_true, y_pred):
        conf[int(a), int(b)] += 1
    r, c = linear_sum_assignment(-conf)
    return {
        'ari': float(adjusted_rand_score(y_true, y_pred)),
        'nmi': float(normalized_mutual_info_score(y_true, y_pred)),
        'purity': float(np.max(conf, axis=0).sum() / len(y_true)),
        'acc': float(conf[r, c].sum() / len(y_true)),
        'n': int(len(y_true)),
    }


def macro_map_from_state(state):
    k = int(state['rho_'].numel())
    if k == 2:
        return np.array([0, 1], dtype=int)
    w = torch.softmax(state['rho_'].float(), 0).numpy()
    mu = state['mu_c'].float().numpy()
    lv = state['log_sigma2_c'].float().numpy()
    clusters, _ = core.agglomerate(w, mu, lv, np.zeros(k, dtype=int), target=2,
                                   linkage='complete', use_guard=True)
    out = np.zeros(k, dtype=int)
    for m, c in enumerate(clusters):
        for j in c.members:
            out[j] = m
    return out


def evaluate_server(model, X, y, macro_map=None, seeds=10):
    vals = []
    for s in range(seeds):
        torch.manual_seed(s)
        pred = model.predict(torch.tensor(np.asarray(X), dtype=torch.float32))
        if macro_map is not None:
            pred = np.asarray(macro_map)[np.asarray(pred, dtype=int)]
        vals.append(cluster_metrics(y, pred))
    return {k: {'mean': float(np.mean([v[k] for v in vals])),
                'sd': float(np.std([v[k] for v in vals], ddof=1))}
            for k in ('ari', 'purity', 'nmi', 'acc')}


def init_local_gmm(source_model, source_state, client_x, local_k):
    z = encode_mean(source_model, client_x)
    lp = diag_log_prob(z, source_state)
    lp -= lp.max(1, keepdims=True)
    resp = np.exp(lp); resp /= resp.sum(1, keepdims=True)
    mass = resp.sum(0)
    order = np.argsort(-mass)
    selected = np.sort(order[:local_k])
    w = torch.softmax(source_state['rho_'].float(), 0).numpy()[selected]
    w = w / w.sum()
    return {
        'rho_': torch.tensor(np.log(np.maximum(w, 1e-10)), dtype=torch.float32),
        'mu_c': source_state['mu_c'][selected].clone(),
        'log_sigma2_c': source_state['log_sigma2_c'][selected].clone(),
    }


def bic_select(source_model, X_unique):
    z = encode_mean(source_model, X_unique)
    rows = []
    for k in range(2, 15):
        gm = GaussianMixture(n_components=k, covariance_type='diag',
                             random_state=0, n_init=5, max_iter=500,
                             reg_covar=1e-6).fit(z)
        rows.append({'K': k, 'bic': float(gm.bic(z)), 'aic': float(gm.aic(z))})
    candidates = [r for r in rows if 4 <= r['K'] <= 14]
    best = min(candidates, key=lambda r: (r['bic'], r['K']))
    return int(best['K']), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='AUTHOR,F01')
    ap.add_argument('--seeds', default='0,1,2')
    args = ap.parse_args()
    EXP.joinpath('init').mkdir(parents=True, exist_ok=True)
    EXP.joinpath('configs').mkdir(parents=True, exist_ok=True)
    EXP.joinpath('audit').mkdir(parents=True, exist_ok=True)

    _, loaders, (X_t, y_t), sizes = make_noniid_clients(
        str(DATA), 10, 200, 'breast_cancer', seed=42)
    X = X_t.numpy().astype(np.float64); y = y_t.numpy().astype(int)
    client_X = [dl.dataset.tensors[0].numpy().astype(np.float64) for dl in loaders]
    client_y = [dl.dataset.tensors[1].numpy().astype(int) for dl in loaders]
    sizes = np.asarray(sizes, dtype=np.int64)
    local_K = [int(len(np.unique(cy))) for cy in client_y]
    weight = sizes / sizes.sum()
    partition_hash = hashlib.sha256()
    for cx, cy in zip(client_X, client_y):
        partition_hash.update(cx.tobytes()); partition_hash.update(cy.tobytes())

    manifest = {'models': {}, 'partition_hash': partition_hash.hexdigest(),
                'partition_seed': 42, 'client_sizes': sizes.tolist(),
                'local_K': local_K, 'n_local_components': int(sum(local_K)),
                'data_sha256': sha256_file(DATA), 'protocol': 'warm-start only'}
    for model_key in [x.strip() for x in args.models.split(',') if x.strip()]:
        cfg = MODELS[model_key]
        source_state, source_round, source_payload = load_payload(cfg['path'])
        source_model = make_vade(2, cfg['alpha'])
        source_model.load_state_dict(source_state, strict=True)
        source_model.eval()
        source_metrics = evaluate_server(source_model, np.concatenate(client_X),
                                         np.concatenate(client_y), seeds=10)
        k_micro, bic_rows = bic_select(source_model, X)
        (EXP / f'configs/bic_{model_key}.json').write_text(
            json.dumps({'selected_K_micro': k_micro, 'rows': bic_rows}, indent=2) + '\n')
        source_hash = sha256_file(cfg['path'])
        source_state_hash = state_hash(source_state)
        source_encoder_hash = state_hash(source_state, exclude_gmm=True)
        model_manifest = {
            'source_path': str(cfg['path']), 'source_sha256': source_hash,
            'source_round': source_round, 'historical_ari_reference': cfg['historical_ari'],
            'source_initial_metrics': source_metrics,
            'server_K': int(source_state['rho_'].numel()),
            'K_micro': int(k_micro), 'alpha': cfg['alpha'], 'schedule': cfg['schedule'],
            'seeds': {},
        }
        client_hashes = []
        for seed in [int(x) for x in args.seeds.split(',') if x.strip()]:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            local_states = []
            for c, (cx, ck) in enumerate(zip(client_X, local_K)):
                cstate = {k: v.clone() for k, v in source_state.items()
                          if k not in ('rho_', 'mu_c', 'log_sigma2_c')}
                cstate.update(init_local_gmm(source_model, source_state, cx, ck))
                local_states.append(cstate)
            client_hash = nested_hash(local_states)
            client_hashes.append(client_hash)
            data_seed = 20260000 + seed * 100
            base = {
                'model': model_key, 'seed': seed, 'source_path': str(cfg['path']),
                'source_sha256': source_hash, 'source_round': source_round,
                'alpha': cfg['alpha'], 'schedule': cfg['schedule'],
                'server_state': source_state, 'clients': local_states,
                'local_K': local_K, 'sizes': sizes, 'weight': weight,
                'partition_seed': 42, 'partition_hash': partition_hash.hexdigest(),
                'data_seed': data_seed,
                'client_hash': client_hash,
                'server_encoder_hash': source_encoder_hash,
                'source_state_hash': source_state_hash,
                'initial_metrics': source_metrics,
                'python_rng': random.getstate(),
                'numpy_rng': np.random.get_state(),
                'torch_rng': torch.get_rng_state(),
            }
            base_path = EXP / f'init/{model_key}_seed{seed}_base.pt'
            torch.save(base, base_path)

            comp = core.flatten_clients(local_states, sizes)
            hier = core.build_hierarchical(comp, total_micro=k_micro, k_macro=2,
                                           linkage='complete', use_guard=True,
                                           guard_threshold=1.0)
            on_state = {k: v.clone() for k, v in source_state.items()
                        if k not in ('rho_', 'mu_c', 'log_sigma2_c')}
            on_state['rho_'] = torch.tensor(
                np.log(np.maximum(hier['weights'], 1e-10)), dtype=torch.float32)
            on_state['mu_c'] = torch.tensor(hier['means'], dtype=torch.float32)
            on_state['log_sigma2_c'] = torch.tensor(hier['log_sigma2'], dtype=torch.float32)
            on_model = make_vade(k_micro, cfg['alpha'])
            on_model.load_state_dict(on_state, strict=True)
            on_model.eval()
            mmap = macro_map_from_state(on_state)
            on_metrics = evaluate_server(
                on_model, np.concatenate(client_X), np.concatenate(client_y),
                macro_map=mmap, seeds=10)
            on = {
                'model': model_key, 'seed': seed, 'source_path': str(cfg['path']),
                'source_sha256': source_hash, 'source_round': source_round,
                'alpha': cfg['alpha'], 'schedule': cfg['schedule'],
                'server_state': on_state, 'clients': local_states,
                'local_K': local_K, 'sizes': sizes, 'weight': weight,
                'K_micro': int(k_micro), 'K_macro': 2,
                'macro_allocation': hier.get('macro_allocation'),
                'macro_sizes': hier.get('macro_sizes'),
                'macro_masses': hier.get('macro_masses'),
                'macro_of_micro': hier['macro_of_micro'].tolist(),
                'n_unsafe_merges': int(hier.get('n_unsafe_merges', 0)),
                'merge_events': hier.get('merge_events', []),
                'partition_seed': 42, 'partition_hash': partition_hash.hexdigest(),
                'data_seed': data_seed, 'client_hash': client_hash,
                'server_encoder_hash': source_encoder_hash,
                'initial_metrics': on_metrics,
                'python_rng': base['python_rng'],
                'numpy_rng': base['numpy_rng'],
                'torch_rng': base['torch_rng'],
                'macro_map': mmap.tolist(),
            }
            on_path = EXP / f'init/{model_key}_seed{seed}_on.pt'
            torch.save(on, on_path)
            model_manifest['seeds'][str(seed)] = {
                'base': str(base_path), 'on': str(on_path),
                'base_sha256': sha256_file(base_path), 'on_sha256': sha256_file(on_path),
                'client_hash': client_hash,
                'server_encoder_hash_off': source_encoder_hash,
                'server_encoder_hash_on': state_hash(on_state, exclude_gmm=True),
                'initial_metrics_off': source_metrics,
                'initial_metrics_on': on_metrics,
                'macro_allocation': hier.get('macro_allocation'),
                'macro_sizes': hier.get('macro_sizes'),
                'macro_masses': hier.get('macro_masses'),
                'n_unsafe_merges': int(hier.get('n_unsafe_merges', 0)),
            }
            print(json.dumps({'model': model_key, 'seed': seed, 'K_micro': k_micro,
                              'client_hash': client_hash[:16],
                              'initial_ari': source_metrics['ari']['mean'],
                              'on_initial_ari': on_metrics['ari']['mean'],
                              'macro_allocation': hier.get('macro_allocation')}), flush=True)
        model_manifest['client_hashes_all_seeds'] = client_hashes
        manifest['models'][model_key] = model_manifest
    (EXP / 'init_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    lines = [
        '# Feasibility and Warm-Start Audit', '',
        'The author-delivered `.pk` and the F01 `epoch_100_server.pt` contain **server',
        'weights only**. They do not contain the 10 client models, local GMMs,',
        'optimizers, schedulers, or RNG states. Exact continuation of the historical',
        'federated trajectories is therefore **not possible**.', '',
        'This experiment uses a controlled warm start: the delivered server',
        'encoder/decoder and GMM are loaded; for each client, the encoder/decoder are',
        'copied and the local GMM is initialized by selecting the top local-class-count',
        'components of the same server GMM. No local GMM is fitted, no pretraining is',
        'run, and no retrained baseline is substituted. The result is a new 2x2',
        'warm-start experiment, not a reproduction of the historical author/F01 runs.', '',
        '| Model | Source | Server K | Initial ARI | Historical ref | K_micro |',
        '|---|---|---:|---:|---:|---:|']
    for model_key, m in manifest['models'].items():
        lines.append(f"| {model_key} | `{Path(m['source_path']).name}` | {m['server_K']} | "
                     f"{m['source_initial_metrics']['ari']['mean']:.4f} | "
                     f"{m['historical_ari_reference']:.4f} | {m['K_micro']} |")
    lines += ['', 'Client-state hashes are identical between OFF and ON snapshots; the',
              'ON server GMM differs by construction because it is the Multi-CA',
              'macro-micro mixture. See `init_manifest.json` for all hashes.', '',
              'BC has no independent test split; all metrics are author-protocol in-sample.', '']
    (EXP / 'audit/FEASIBILITY.md').write_text('\n'.join(lines))
    print('INIT_DONE')


if __name__ == '__main__':
    main()
