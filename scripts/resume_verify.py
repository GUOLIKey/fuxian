"""Load a full checkpoint and reproduce the next formal round exactly.

Strong verification of the saved federated state: rebuild the author's
server/clients/optimizers/schedulers/DataLoaders from `epoch_XXX_full.pt`,
restore all three RNG states, run round X+1 with the author math, and compare
client loss, server loss, purity and ARI with that run's rounds.jsonl.

Usage:
  python resume_verify.py --checkpoint .../epoch_100_full.pt --next-round 101
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path('/root/autodl-tmp/fedvade_project')
SOURCE = ROOT / 'fuxian-round2/FedVaDE_source'
sys.path.insert(0, str(SOURCE))
from model import VaDE, gmm_sampling, vae_avg  # noqa: E402
from experiment import validation  # noqa: E402


def make_vade(k, alpha):
    return VaDE(SimpleNamespace(input_dim=31, hid_dim=10, nClusters=int(k),
                                alpha=float(alpha), cuda=False))


def build(ckpt):
    cfg = ckpt['config']
    alpha = float(ckpt['alpha'])
    lr = float(cfg['lr'])
    schedule = ckpt['schedule']
    sizes = np.asarray(ckpt['partition_sizes'], dtype=np.float64)
    weight = sizes / sizes.sum()

    loaders = [DataLoader(TensorDataset(p['x'], p['y']), batch_size=200,
                          shuffle=True) for p in ckpt['partition']]
    server = {'vade': make_vade(len(ckpt['server']['rho_']), alpha)}
    server['vade'].load_state_dict(ckpt['server'], strict=True)
    server['optimizer'] = Adam(server['vade'].parameters(), lr=lr)
    server['optimizer'].load_state_dict(ckpt['server_optimizer'])
    server['lr_s'] = StepLR(server['optimizer'], step_size=10, gamma=.85)
    server['lr_s'].load_state_dict(ckpt['server_scheduler'])

    clients = []
    for i, state in enumerate(ckpt['clients']):
        vade = make_vade(len(state['rho_']), alpha)
        vade.load_state_dict(state, strict=True)
        c = {'client_id': i, 'vade': vade, 'dataloader': loaders[i]}
        c['optimizer'] = Adam(vade.parameters(), lr=lr)
        c['optimizer'].load_state_dict(ckpt['client_optimizers'][i])
        c['lr_s'] = StepLR(c['optimizer'], step_size=10, gamma=.85)
        c['lr_s'].load_state_dict(ckpt['client_schedulers'][i])
        clients.append(c)
    return server, clients, weight, schedule, cfg


def run_round(server, clients, weight, schedule, cfg):
    clients_vae_param = []
    L_c = 0.0
    for c in clients:
        for x, _ in c['dataloader']:
            c['optimizer'].zero_grad()
            loss = c['vade'].ELBO_Loss(x)
            L_c += float(loss.item())
            loss.backward()
            c['optimizer'].step()
            if schedule == 'author':
                c['lr_s'].step()
        if schedule == 'epoch':
            c['lr_s'].step()
        clients_vae_param.append(c['vade'].state_dict())
    L_c /= len(clients)

    server_vade_param = vae_avg(clients_vae_param, weight)
    server['vade'].load_state_dict(server_vade_param, strict=False)

    clients_gmm_param = [{
        'rho_': c['vade'].rho_.data.clone().cpu(),
        'mu_c': c['vade'].mu_c.data.clone().cpu(),
        'log_sigma2_c': c['vade'].log_sigma2_c.data.clone().cpu(),
    } for c in clients]
    Z_hat, y_hat, lengths, _, _, _ = gmm_sampling(
        clients_gmm_param, weight, int(cfg['sample']))
    Z_hat = torch.from_numpy(Z_hat).float()
    X_hat = server['vade'].decoder(Z_hat)
    decoded_X_hat = torch.zeros_like(X_hat)
    start = 0
    for cid in range(len(clients)):
        for k in range(lengths[cid]):
            label = start + k
            idx = np.where(y_hat == label)[0]
            decoded_X_hat[idx] = clients[cid]['vade'].decoder(Z_hat[idx])
        start += lengths[cid]
    loss_s = server['vade'].ELBO_Loss_server(X_hat, decoded_X_hat)
    server['optimizer'].zero_grad()
    loss_s.backward()
    server['optimizer'].step()
    server['lr_s'].step()
    for c in clients:
        c['vade'].load_state_dict(server_vade_param, strict=False)

    pre, tru = [], []
    with torch.no_grad():
        for c in clients:
            for x, y in c['dataloader']:
                tru.append(y.numpy())
                pre.append(server['vade'].predict(x))
    purity, ari = validation(np.concatenate(tru, 0), np.concatenate(pre, 0))
    return {'client_loss': float(L_c), 'server_loss': float(loss_s.item()),
            'purity': float(purity), 'ari': float(ari)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--next-round', type=int, required=True)
    ap.add_argument('--run-dir', type=Path)
    args = ap.parse_args()
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    expected_round = int(args.checkpoint.name.split('_')[1])
    if int(ckpt['round']) != expected_round:
        raise SystemExit(f'round mismatch: {ckpt["round"]} vs {expected_round}')
    server, clients, weight, schedule, cfg = build(ckpt)
    # Restore RNG only after all model/optimizer/loader construction.
    random.setstate(ckpt['python_rng'])
    np.random.set_state(ckpt['numpy_rng'])
    torch.set_rng_state(ckpt['torch_rng'])
    got = run_round(server, clients, weight, schedule, cfg)
    run = args.run_dir or args.checkpoint.parent
    rows = [json.loads(x) for x in (run / 'rounds.jsonl').read_text().splitlines()
            if x.strip()]
    want = next(r for r in rows if r['round'] == args.next_round)
    keys = ('client_loss', 'server_loss', 'purity', 'ari')
    deltas = {k: got[k] - float(want[k]) for k in keys}
    exact = all(abs(v) <= 1e-6 for v in deltas.values())
    report = {
        'checkpoint': str(args.checkpoint), 'round': expected_round,
        'next_round': args.next_round,
        'got': got,
        'expected': {k: float(want[k]) for k in keys},
        'delta': deltas,
        'exact_1e-6': bool(exact),
        'n_clients': len(clients),
        'client_K': [int(len(c['vade'].rho_)) for c in clients],
        'optimizer_states': len(ckpt['client_optimizers']),
        'scheduler_states': len(ckpt['client_schedulers']),
        'partition_records': [int(len(p['y'])) for p in ckpt['partition']],
        'schedule': schedule,
    }
    out = args.checkpoint.parent / 'resume_verify.json'
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not exact:
        raise SystemExit('RESUME_MISMATCH')


if __name__ == '__main__':
    main()
