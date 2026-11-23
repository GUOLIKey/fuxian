"""Create one shared author-protocol pretrain checkpoint per training seed.

All four groups for a seed load the same checkpoint; Author/F01 differ only in the
formal objective/scheduler, and Multi-CA ON/OFF differ only in server aggregation.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/root/autodl-tmp/fedvade_project')
SOURCE = ROOT / 'fuxian-round2/FedVaDE_source'
BASE = ROOT / 'experiments/multica_factorial_v1'
RUNS = ROOT / 'reproduction_runs/multica_factorial_v1'


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def state_cpu(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--pretrain', type=int, default=50)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = p.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f'Refusing existing pretrain output: {out}')
    out.mkdir(parents=True)

    sys.path.insert(0, str(args.source))
    from argparse import Namespace
    from dataloader import make_noniid_clients
    from model import VaDE, pre_train

    use_cuda = args.device == 'cuda' and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    n_clusters, loaders, (X, y), sizes = make_noniid_clients(
        str(args.source / 'dataset/breast_cancer.csv'), 10, 200,
        'breast_cancer', seed=42)
    X = X.numpy().astype(np.float32)
    y = y.numpy().astype(np.int64)
    sizes = np.asarray(sizes, dtype=np.int64)
    weight = sizes / sizes.sum()
    namespace = Namespace(
        input_dim=X.shape[1], hid_dim=10, nClusters=int(n_clusters),
        alpha=0.01, cuda=use_cuda, client_num=len(loaders),
        batch_size=200, sample=100,
    )
    server = {'vade': VaDE(namespace).to(device), 'optimizer': None, 'lr_s': None}
    clients = []
    for i, dl in enumerate(loaders):
        model = VaDE(namespace).to(device)
        model.load_state_dict(server['vade'].state_dict())
        clients.append({'client_id': i, 'dataloader': dl, 'vade': model,
                        'optimizer': None, 'lr_s': None})
    pre_train(namespace, server, clients, weight, pre_epoch=args.pretrain)

    checkpoint = {
        'round': 0,
        'seed': args.seed,
        'status': 'PRETRAINED_AUTHOR_PROTOCOL',
        'server': state_cpu(server['vade']),
        'clients': [state_cpu(c['vade']) for c in clients],
        'sizes': sizes.tolist(),
        'partition_sha256': [
            sha_bytes(dl.dataset.tensors[0].numpy()) + ':' + sha_bytes(dl.dataset.tensors[1].numpy())
            for dl in loaders
        ],
        'data_sha256': sha_bytes(X) + ':' + sha_bytes(y),
        'python_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.get_rng_state(),
        'torch_cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(checkpoint, out / 'pretrain.pt')
    config = {
        'seed': args.seed, 'pretrain_epochs': args.pretrain,
        'clients': 10, 'batch_size': 200, 'global_K': int(n_clusters),
        'local_K': 'author label-assisted len(unique(local_y))',
        'lr_pretrain': 0.001, 'sample_M': 100,
        'partition_seed': 42, 'gmm_seed': 42,
        'device': str(device), 'source': str(args.source),
        'source_sha256': {q.name: sha(q) for q in args.source.glob('*.py')},
        'data_sha256': sha(args.source / 'dataset/breast_cancer.csv'),
        'checkpoint_sha256': sha(out / 'pretrain.pt'),
        'note': 'Shared pretrain for A/B/C/D of this seed; pretrain is independent of alpha and post-pretrain scheduler.',
    }
    atomic_json(out / 'config.json', config)
    print(json.dumps({'status': 'PRETRAIN_OK', 'seed': args.seed,
                      'out': str(out), 'checkpoint_sha256': config['checkpoint_sha256'],
                      'partition_sizes': sizes.tolist()}, indent=2))


def sha_bytes(value):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


if __name__ == '__main__':
    main()
