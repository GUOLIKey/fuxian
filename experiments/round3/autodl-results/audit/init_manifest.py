"""Fingerprint the round-two FedVaDE initialization before pretraining or optimization."""
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import torch

source = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(source))
from dataloader import make_noniid_clients
from model import VaDE

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
clusters, loaders, (x, _), _ = make_noniid_clients(source / 'dataset/breast_cancer.csv', 10, 200, 'breast_cancer')
args = SimpleNamespace(nClusters=clusters, input_dim=x.shape[1], hid_dim=10, alpha=.01)


def fingerprint(model):
    return sha256(b''.join(t.detach().numpy().tobytes() for t in model.state_dict().values())).hexdigest()


results = {}
for seed in range(5):
    torch.manual_seed(seed)
    server = VaDE(args)
    clients = [VaDE(args) for _ in loaders]
    for model in clients:
        model.load_state_dict(server.state_dict())
    server_hash = fingerprint(server)
    assert all(fingerprint(model) == server_hash for model in clients)
    results[seed] = {'server_and_all_clients': server_hash,
                     'torch_cpu_rng_state_before_pretrain': sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}
print(json.dumps(results, indent=2))
