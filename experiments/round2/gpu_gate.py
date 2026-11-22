"""Compare author outputs and Adam updates for identical inputs and noise."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch


def main():
    root = Path(sys.argv[1]).resolve()
    csv = root / 'dataset/breast_cancer.csv'
    sys.path.insert(0, str(root))
    from dataloader import make_noniid_clients
    from model import VaDE

    torch.set_num_threads(1)
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError('PyTorch did not detect CUDA.')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    clusters, loaders, _, _ = make_noniid_clients(csv, 10, 200, 'breast_cancer')
    inputs = loaders[0].dataset.tensors[0][:200]
    model = VaDE(SimpleNamespace(nClusters=clusters, hid_dim=10, input_dim=inputs.shape[1], alpha=.01))
    with torch.no_grad():
        model.mu_c.copy_(torch.randn_like(model.mu_c) * .3)
    latent = torch.randn(100, 10)
    observations = {}
    for device in ('cpu', 'cuda'):
        working = deepcopy(model).to(device)
        optimizer = torch.optim.Adam(working.parameters(), lr=.002)
        data = inputs.to(device)
        latents = latent.to(device)
        generator = torch.Generator().manual_seed(123)

        def noise(tensor):
            return torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype).to(tensor.device)

        with patch.object(torch, 'randn_like', noise):
            embed, _ = working.encoder(data)
            rebuilt = working.decoder(latents)
            predictions = working.predict(data)
            client = working.ELBO_Loss(data)
            server = working.ELBO_Loss_server(rebuilt, rebuilt.detach() + .05)
            (client + server).backward()
        grads = {key: parameter.grad.detach().cpu() for key, parameter in working.named_parameters()}
        optimizer.step()
        observations[device] = {
            'encoded': embed.detach().cpu(), 'decoded': rebuilt.detach().cpu(),
            'label': torch.as_tensor(predictions), 'client_loss': client.detach().cpu(),
            'server_loss': server.detach().cpu(),
            **{f'gradient:{key}': value for key, value in grads.items()},
            **{f'adam:{key}': value.detach().cpu() for key, value in working.state_dict().items()},
        }
    differences = {}
    for key, reference in observations['cpu'].items():
        target = observations['cuda'][key]
        differences[key] = float((target - reference).abs().max())
    bad = [key for key, reference in observations['cpu'].items() if not torch.allclose(
        reference, observations['cuda'][key], rtol=.002, atol=2e-5, equal_nan=False)]
    print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda,
                      'device': torch.cuda.get_device_name(),
                      'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                      'absolute_max_errors': differences, 'outside_tolerance': bad,
                      'rtol': .002, 'atol': 2e-5}, indent=2, allow_nan=False))
    if bad:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
