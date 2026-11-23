"""2x2 warm-start training runner. No pretraining, no zero-start baseline.

OFF: author FedVaDE loop with the author gmm_sampling.
ON:  Multi-CA macro-micro global mixture (K_macro=2, K_micro>2) sampled every
     round for server training; the server GMM is K_micro and micro-preserving.
All client/server encoder-decoder initialization is loaded from the same
snapshot for OFF and ON.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path('/root/autodl-tmp/fedvade_project')
SOURCE = ROOT / 'fuxian-round2/FedVaDE_source'
EXP = ROOT / 'experiments/multica_pretrained_2x2_v1'
DATA = SOURCE / 'dataset/breast_cancer.csv'
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(EXP / 'scripts'))
from dataloader import make_noniid_clients  # noqa: E402
from experiment import validation  # noqa: E402
from model import VaDE, gmm_sampling, vae_avg  # noqa: E402
import multica_core as core  # noqa: E402

EPS = 1e-10


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, obj):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    os.replace(tmp, path)


def make_vade(k, alpha):
    return VaDE(SimpleNamespace(input_dim=31, hid_dim=10, nClusters=int(k),
                                alpha=float(alpha), cuda=False))


def encode_mean(model, x):
    with torch.no_grad():
        mu, _ = model.encoder(torch.tensor(np.asarray(x), dtype=torch.float32))
    return mu.numpy()


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


def eval_predict(model, client_X, client_y, macro_map=None, seeds=10):
    vals = []
    state = torch.get_rng_state()
    try:
        for s in range(seeds):
            torch.manual_seed(1000 + s)
            pred = model.predict(torch.tensor(client_X, dtype=torch.float32))
            if macro_map is not None:
                pred = np.asarray(macro_map)[np.asarray(pred, dtype=int)]
            vals.append(cluster_metrics(client_y, pred))
    finally:
        torch.set_rng_state(state)
    return {k: {'mean': float(np.mean([v[k] for v in vals])),
                'sd': (float(np.std([v[k] for v in vals], ddof=1))
                       if len(vals) > 1 else 0.0)}
            for k in ('ari', 'purity', 'nmi', 'acc')}


def mmd2_rbf(A, B, sigma):
    A = np.asarray(A, dtype=np.float64); B = np.asarray(B, dtype=np.float64)
    if len(A) > 2000:
        A = A[:2000]
    a2 = np.sum(A * A, 1); b2 = np.sum(B * B, 1)
    Kxx = np.exp(-np.maximum(a2[:, None] + a2[None, :] - 2 * A @ A.T, 0) /
                 (2 * sigma * sigma))
    Kyy = np.exp(-np.maximum(b2[:, None] + b2[None, :] - 2 * B @ B.T, 0) /
                 (2 * sigma * sigma))
    Kxy = np.exp(-np.maximum(a2[:, None] + b2[None, :] - 2 * A @ B.T, 0) /
                 (2 * sigma * sigma))
    return float(max(Kxx.mean() + Kyy.mean() - 2 * Kxy.mean(), 0.0))


def swd(A, B, directions):
    A = np.asarray(A, dtype=np.float64); B = np.asarray(B, dtype=np.float64)
    pa = A @ directions; pb = B @ directions
    q = np.linspace(0, 1, 200)
    return float(np.mean(np.abs(np.quantile(pa, q, axis=0) -
                                np.quantile(pb, q, axis=0))))


def final_evaluation(server, init, client_X, client_y, X, y, out):
    state = {k: v.detach().cpu().clone() for k, v in server.state_dict().items()}
    macro = macro_map_from_state(state)
    metrics = eval_predict(server, client_X, client_y, macro_map=macro, seeds=10)
    z = encode_mean(server, X)
    nll = diag_nll(z, state)
    # Generation from the final server GMM (K=2 OFF, K_micro ON).
    gen_model = core.gmm_from_state(state)
    gen_model['macro_of_micro'] = macro
    rng = np.random.default_rng(20261124)
    dirs = rng.normal(size=(X.shape[1], 64))
    dirs /= np.linalg.norm(dirs, axis=0, keepdims=True)
    from scipy.spatial.distance import pdist
    sigma = float(np.sqrt(np.median(pdist(X, metric='sqeuclidean'))))
    resp = core.model_resp(gen_model, z)
    hard = resp.argmax(1)
    mmds, swds = [], []
    per_micro = []
    for s in range(5):
        zg, _ = core.sample_model(gen_model, 5000, seed=5000 + s)
        with torch.no_grad():
            xg = server.decoder(torch.tensor(zg, dtype=torch.float32)).numpy()
        mmds.append(math.sqrt(mmd2_rbf(xg, X, sigma)))
        swds.append(swd(xg, X, dirs))
    for c in range(len(gen_model['weights'])):
        idx = np.flatnonzero(hard == c)
        row = {'micro_id': c, 'macro_id': int(macro[c]),
               'weight': float(gen_model['weights'][c]), 'support': int(len(idx)),
               'emp_mass': float(len(idx) / len(hard)),
               'low_mass': bool(gen_model['weights'][c] < 1.0 / (2 * len(gen_model['weights']))),
               'micro_mmd': None, 'micro_swd': None, 'self_rate': None}
        if len(idx) >= 20:
            vals_m, vals_s = [], []
            for s in range(3):
                zg, _ = core.sample_model(gen_model, 500, seed=9000 + 10 * c + s, micro=c)
                with torch.no_grad():
                    xg = server.decoder(torch.tensor(zg, dtype=torch.float32)).numpy()
                vals_m.append(math.sqrt(mmd2_rbf(xg, X[idx], sigma)))
                vals_s.append(swd(xg, X[idx], dirs))
                r = core.model_resp(gen_model, zg)
                row['self_rate'] = float((r.argmax(1) == c).mean())
            row['micro_mmd'] = float(np.mean(vals_m))
            row['micro_swd'] = float(np.mean(vals_s))
        per_micro.append(row)
    ok = [r for r in per_micro if r['micro_mmd'] is not None]
    rare = [r for r in ok if r['low_mass']]
    out.update({
        'k_micro': int(len(gen_model['weights'])),
        'k_macro': int(len(set(macro.tolist()))),
        'nll_unique': nll,
        'client_record_metrics_10seed': metrics,
        'gen_mmd_mean': float(np.mean(mmds)), 'gen_mmd_sd': float(np.std(mmds, ddof=1)),
        'gen_swd_mean': float(np.mean(swds)), 'gen_swd_sd': float(np.std(swds, ddof=1)),
        'n_supported_20': int(sum(r['support'] >= 20 for r in per_micro)),
        'n_ghost': int(sum(r['support'] == 0 for r in per_micro)),
        'micro_mmd_mean': float(np.mean([r['micro_mmd'] for r in ok])) if ok else None,
        'micro_swd_mean': float(np.mean([r['micro_swd'] for r in ok])) if ok else None,
        'rare_n': len(rare),
        'rare_mmd_mean': float(np.mean([r['micro_mmd'] for r in rare])) if rare else None,
        'rare_swd_mean': float(np.mean([r['micro_swd'] for r in rare])) if rare else None,
        'per_micro': per_micro,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', required=True)
    ap.add_argument('--aggregation', required=True, choices=['off', 'on'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--rounds', type=int, default=300)
    ap.add_argument('--save-every', type=int, default=25)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'checkpoints').mkdir(exist_ok=True)
    init = torch.load(args.init, map_location='cpu')
    model_key = init['model']; seed = int(init['seed'])
    random.setstate(init['python_rng']); np.random.set_state(init['numpy_rng'])
    torch.set_rng_state(init['torch_rng'])
    _, loaders, (X_t, y_t), sizes = make_noniid_clients(
        str(DATA), 10, 200, 'breast_cancer', seed=42)
    X = X_t.numpy().astype(np.float64); y = y_t.numpy().astype(int)
    client_X = [dl.dataset.tensors[0].numpy().astype(np.float64) for dl in loaders]
    client_y = [dl.dataset.tensors[1].numpy().astype(int) for dl in loaders]
    sizes = np.asarray(sizes, dtype=np.int64)
    if sizes.tolist() != init['sizes'].tolist():
        raise RuntimeError('partition size mismatch')
    ds = int(init['data_seed'])
    train_loaders = [
        DataLoader(TensorDataset(torch.tensor(cx, dtype=torch.float32),
                                 torch.tensor(cy, dtype=torch.long)),
                   batch_size=200, shuffle=True, num_workers=0,
                   generator=torch.Generator().manual_seed(ds + c))
        for c, (cx, cy) in enumerate(zip(client_X, client_y))]
    client_Xc = np.concatenate(client_X, 0); client_yc = np.concatenate(client_y, 0)
    weight = torch.tensor(sizes / sizes.sum(), dtype=torch.float32)

    server_state = init['server_state']
    server = make_vade(len(server_state['rho_']), init['alpha'])
    server.load_state_dict(server_state, strict=True)
    clients = []
    for c, cstate in enumerate(init['clients']):
        vade = make_vade(len(cstate['rho_']), init['alpha'])
        vade.load_state_dict(cstate, strict=True)
        clients.append({'client_id': c, 'vade': vade, 'dataloader': train_loaders[c]})
    server_opt = Adam(
        (list(server.encoder.parameters()) + list(server.decoder.parameters()))
        if args.aggregation == 'on' else server.parameters(),
        lr=0.002)
    server_sch = StepLR(server_opt, step_size=10, gamma=0.85)
    for c in clients:
        c['optimizer'] = Adam(c['vade'].parameters(), lr=0.002)
        c['lr_s'] = StepLR(c['optimizer'], step_size=10, gamma=0.85)
    config = {
        'model': model_key, 'seed': seed, 'aggregation': args.aggregation,
        'init': str(args.init), 'init_sha256': sha256_file(args.init),
        'source': init['source_path'], 'source_sha256': init['source_sha256'],
        'alpha': init['alpha'], 'schedule': init['schedule'],
        'rounds': args.rounds, 'lr': 0.002, 'batch_size': 200, 'sample': 100,
        'clients': 10, 'local_K': init['local_K'],
        'server_K': int(server_state['rho_'].numel()),
        'K_micro': int(init.get('K_micro', server_state['rho_'].numel())),
        'server_gmm_update': ('multi_ca_aggregation_each_round'
                              if args.aggregation == 'on' else 'author_server_elbo'),
        'server_optimizer': ('encoder_decoder' if args.aggregation == 'on'
                             else 'all_server_parameters'),
        'macro_allocation': init.get('macro_allocation'),
        'macro_sizes': init.get('macro_sizes'),
        'macro_masses': init.get('macro_masses'),
        'partition_seed': 42, 'data_seed': ds,
        'test_status': 'NO_INDEPENDENT_BC_TEST_SPLIT; in-sample author protocol',
        'training_constraints': 'warm-start only; no pretraining; no retrained baseline',
    }
    atomic_json(out / 'config.json', config)
    rows_path = out / 'rounds.jsonl'
    rows_path.write_text('')
    t_start = time.time()
    for r in range(1, args.rounds + 1):
        t0 = time.time()
        clients_vae_param = []
        L_c = 0.0
        for c in clients:
            for x, _ in c['dataloader']:
                c['optimizer'].zero_grad()
                loss = c['vade'].ELBO_Loss(x)
                L_c += float(loss.item())
                loss.backward()
                c['optimizer'].step()
                if init['schedule'] == 'author':
                    c['lr_s'].step()
            if init['schedule'] == 'epoch':
                c['lr_s'].step()
            clients_vae_param.append(c['vade'].state_dict())
        L_c /= len(clients)
        server_vade_param = vae_avg(clients_vae_param, weight)
        server.load_state_dict(server_vade_param, strict=False)
        clients_gmm_param = [{
            'rho_': c['vade'].rho_.detach().cpu().clone(),
            'mu_c': c['vade'].mu_c.detach().cpu().clone(),
            'log_sigma2_c': c['vade'].log_sigma2_c.detach().cpu().clone(),
        } for c in clients]
        hier_info = None
        if args.aggregation == 'off':
            z_np, y_hat, lengths, _, _, _ = gmm_sampling(
                clients_gmm_param, weight, 100)
            sample_client = []
            for cidx, ln in enumerate(lengths):
                sample_client.extend([cidx] * int(ln))
            sample_client = np.asarray(sample_client, dtype=int)[np.asarray(y_hat, dtype=int)]
        else:
            comp = core.flatten_clients(clients_gmm_param, sizes)
            hier = core.build_hierarchical(
                comp, total_micro=int(config['K_micro']), k_macro=2,
                linkage='complete', use_guard=True, guard_threshold=1.0)
            z_np, comp_ids = core.sample_model(
                hier, 100, seed=seed * 100000 + r)
            own = [hier['owners'][int(cid)].get('dominant_client') for cid in comp_ids]
            own = [0 if o is None else int(o) for o in own]
            sample_client = np.asarray(own, dtype=int)
            hier_info = {
                'K_micro': int(hier['n_micro']), 'K_macro': int(hier['n_macro']),
                'macro_weights': [float(x) for x in hier['macro_weights']],
                'macro_allocation': hier.get('macro_allocation'),
                'macro_sizes': hier.get('macro_sizes'),
                'n_unsafe_merges': int(hier.get('n_unsafe_merges', 0)),
                'max_merge_normalized_bures': float(max(
                    [e['normalized_bures'] for e in hier.get('merge_events', [])] or [0.0])),
                'effective_micro': float(1.0 / np.sum(np.asarray(hier['weights']) ** 2)),
                'sample_client_counts': np.bincount(sample_client, minlength=10).tolist(),
            }
        Z_hat = torch.tensor(np.asarray(z_np), dtype=torch.float32)
        X_hat = server.decoder(Z_hat)
        decoded_X_hat = torch.zeros_like(X_hat)
        for cidx, c in enumerate(clients):
            idx = np.flatnonzero(sample_client == cidx)
            if len(idx):
                decoded_X_hat[idx] = c['vade'].decoder(Z_hat[idx])
        loss_s = server.ELBO_Loss_server(X_hat, decoded_X_hat)
        server_opt.zero_grad()
        loss_s.backward()
        server_opt.step()
        server_sch.step()
        if args.aggregation == 'on':
            # Multi-CA supplies the global GMM every round; the server GMM is
            # not allowed to collapse to a 2-component gradient solution.
            with torch.no_grad():
                server.rho_.data.copy_(torch.tensor(
                    np.log(np.maximum(hier['weights'], EPS)), dtype=torch.float32))
                server.mu_c.data.copy_(torch.tensor(hier['means'], dtype=torch.float32))
                server.log_sigma2_c.data.copy_(torch.tensor(
                    hier['log_sigma2'], dtype=torch.float32))
        for c in clients:
            c['vade'].load_state_dict(server_vade_param, strict=False)

        state = server.state_dict()
        macro = macro_map_from_state(state)
        ev = eval_predict(server, client_Xc, client_yc, macro_map=macro, seeds=1)
        z_unique = encode_mean(server, X)
        nll = diag_nll(z_unique, state)
        sw = torch.softmax(state['rho_'].float(), 0).numpy()
        row = {
            'round': r, 'model': model_key, 'seed': seed, 'aggregation': args.aggregation,
            'client_loss': L_c, 'server_loss': float(loss_s.item()),
            'ari': ev['ari']['mean'], 'purity': ev['purity']['mean'],
            'nmi': ev['nmi']['mean'], 'acc': ev['acc']['mean'],
            'server_nll': nll, 'server_K': int(len(sw)),
            'server_effective_components': float(1.0 / np.sum(sw ** 2)),
            'server_macro_weights': [float(x) for x in
                                     np.array([sw[macro == m].sum()
                                               for m in sorted(set(macro.tolist()))])],
            'n_local_components': int(sum(len(c['vade'].rho_) for c in clients)),
            'elapsed_round': float(time.time() - t0),
            'elapsed_total': float(time.time() - t_start),
        }
        if hier_info is not None:
            row['multi_ca'] = hier_info
        with rows_path.open('a') as f:
            f.write(json.dumps(row) + '\n')
        atomic_json(out / 'status.json', {
            'status': 'RUNNING', 'model': model_key, 'seed': seed,
            'aggregation': args.aggregation, 'round': r, 'rounds': args.rounds,
            'progress': f'{r}/{args.rounds}', 'latest': row,
            'last_error': None,
        })
        if r % args.save_every == 0:
            torch.save({'round': r, 'server': state,
                        'model': model_key, 'seed': seed,
                        'aggregation': args.aggregation},
                       out / 'checkpoints' / f'server_round_{r:03d}.pt')
        if r in (100, 300) or r == args.rounds:
            torch.save({
                'round': r, 'server': state,
                'clients': [c['vade'].state_dict() for c in clients],
                'server_optimizer': server_opt.state_dict(),
                'client_optimizers': [c['optimizer'].state_dict() for c in clients],
                'server_scheduler': server_sch.state_dict(),
                'client_schedulers': [c['lr_s'].state_dict() for c in clients],
                'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                'torch_rng': torch.get_rng_state(), 'config': config,
            }, out / 'checkpoints' / f'full_round_{r:03d}.pt')
        if r % 10 == 0 or r == args.rounds:
            print(json.dumps({'model': model_key, 'seed': seed,
                              'aggregation': args.aggregation, 'round': r,
                              'ari': round(ev['ari']['mean'], 4),
                              'nll': round(nll, 4),
                              'K': int(len(sw))}), flush=True)

    final = {'status': 'DONE', 'config': config, 'rounds': args.rounds,
             'elapsed_seconds': float(time.time() - t_start)}
    final_evaluation(server, init, client_Xc, client_yc, X, y, final)
    atomic_json(out / 'evaluation.json', final)
    atomic_json(out / 'status.json', {
        'status': 'DONE', 'model': model_key, 'seed': seed,
        'aggregation': args.aggregation, 'round': args.rounds,
        'rounds': args.rounds, 'progress': f'{args.rounds}/{args.rounds}',
        'latest': final, 'last_error': None,
    })
    print('DONE', model_key, seed, args.aggregation, flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
