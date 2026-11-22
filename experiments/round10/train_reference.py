"""Standalone local-A and centralized VaDE references; remote server execution only."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path('/root/autodl-tmp/fedvade_project')
BASE = ROOT / 'reproduction_runs/round10_non_iid_generation'
sys.path.insert(0, str(ROOT / 'fuxian-round2/FedVaDE_source'))
from model import VaDE


def atomic_json(path, value):
    temp = Path(str(path) + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=('local_A', 'centralized'), required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--pretrain', type=int, default=50)
    p.add_argument('--epochs', type=int, default=150)
    args = p.parse_args()
    out = args.out.resolve()
    if out.exists(): raise SystemExit('Reference run exists; refusing overwrite')
    out.mkdir(parents=True)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    data = np.load(BASE / 'private_data/data.npz')
    x = data['X_train']
    if args.kind == 'local_A':
        ids = np.load(BASE / 'private_data/severe_partitions.npz')['client_0']
        x = x[ids]
    train = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32)),
                       batch_size=200, shuffle=True)
    val = torch.tensor(data['X_val'], dtype=torch.float32)
    from argparse import Namespace
    model = VaDE(Namespace(input_dim=x.shape[1], hid_dim=10, nClusters=3, alpha=.01, cuda=False))
    pre_optimizer = Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()))
    started = time.time(); rows = []
    for epoch in range(args.pretrain):
        for (batch,) in train:
            mu, _ = model.encoder(batch)
            loss = torch.nn.functional.mse_loss(model.decoder(mu), batch)
            pre_optimizer.zero_grad(); loss.backward(); pre_optimizer.step()
    model.encoder.log_sigma2_l.load_state_dict(model.encoder.mu_l.state_dict())
    with torch.no_grad():
        encoded = np.concatenate([model.encoder(batch)[0].numpy() for (batch,) in train])
    candidate = [GaussianMixture(n_components=k, covariance_type='diag', random_state=42).fit(encoded)
                 for k in (1, 2, 3)]
    selected = min(candidate, key=lambda m: m.bic(encoded))
    model.rho_.data = torch.log(torch.tensor(selected.weights_, dtype=torch.float32) + 1e-10)
    model.mu_c.data = torch.tensor(selected.means_, dtype=torch.float32)
    model.log_sigma2_c.data = torch.log(torch.tensor(selected.covariances_, dtype=torch.float32))
    optimizer = Adam(model.parameters(), lr=.002)
    scheduler = StepLR(optimizer, step_size=10, gamma=.85)
    best = {'mse': float('inf'), 'epoch': None}
    for epoch in range(1, args.epochs + 1):
        running = 0.
        for (batch,) in train:
            loss = model.ELBO_Loss(batch)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
            running += float(loss.detach())
        row = {'epoch': epoch, 'loss': running, 'lr': optimizer.param_groups[0]['lr']}
        if epoch % 25 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                mse = float(torch.mean((model.decoder(model.encoder(val)[0]) - val) ** 2))
            model.train()
            row['validation_reconstruction_mse'] = mse
            if mse < best['mse']:
                best.update(mse=mse, epoch=epoch)
                torch.save({'round': epoch, 'server': {k: v.detach().clone()
                           for k, v in model.state_dict().items()},
                           'selection': 'unlabeled validation reconstruction MSE'},
                           out / 'selected_server.pt')
        rows.append(row)
        with (out / 'rounds.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        progress = BASE / 'progress.json'
        status = json.loads(progress.read_text()) if progress.exists() else {}
        status.update(epoch=epoch, epoch_total=args.epochs, latest_ari=None,
                      latest_purity=None, updated_at=time.time())
        atomic_json(progress, status)
    torch.save({'round': args.epochs,
                'server': {k: v.detach().clone() for k, v in model.state_dict().items()},
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                'torch_rng': torch.get_rng_state()}, out / 'last_full.pt')
    atomic_json(out / 'config.json', {'kind': args.kind, 'seed': args.seed,
                'pretrain': args.pretrain, 'epochs': args.epochs, 'training_records': len(x),
                'global_K_max': 3, 'selected_K': len(model.rho_),
                'local_K_rule': 'unlabeled BIC on encoder means',
                'lr': .002, 'alpha': .01, 'batch_size': 200,
                'network': 'author VaDE, original encoder/decoder/ELBO',
                'selected_epoch': best['epoch'], 'selected_validation_mse': best['mse'],
                'command': [sys.executable, *sys.argv],
                'elapsed_seconds': time.time() - started})
    print(args.kind, 'selected K', len(model.rho_), 'epoch', best['epoch'], flush=True)


if __name__ == '__main__':
    main()
