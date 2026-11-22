"""Short replay of the round-two fixed-input Adam check, outside formal trials."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
from dataloader import make_noniid_clients
from model import VaDE

torch.set_num_threads(1)
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
clusters, loaders, _, _ = make_noniid_clients(root / 'dataset/breast_cancer.csv', 10, 200, 'breast_cancer')
inputs = loaders[0].dataset.tensors[0][:200]
model = VaDE(SimpleNamespace(nClusters=clusters, hid_dim=10, input_dim=inputs.shape[1], alpha=.01))
with torch.no_grad():
    model.mu_c.copy_(torch.randn_like(model.mu_c) * .3)
latent = torch.randn(100, 10)
name = 'decoder.decoder.2.weight'
initial = model.state_dict()[name].clone()
records = {}
for device in ('cpu', 'cuda'):
    working = deepcopy(model).to(device)
    optimizer = torch.optim.Adam(working.parameters(), lr=.002)
    data, latents = inputs.to(device), latent.to(device)
    rng = torch.Generator().manual_seed(123)

    def common_noise(tensor):
        return torch.randn(tensor.shape, generator=rng, dtype=tensor.dtype).to(tensor.device)

    rows = {}
    with patch.object(torch, 'randn_like', common_noise):
        for step in range(1, 51):
            optimizer.zero_grad()
            if step == 1:
                working.predict(data)  # Match the round-two gate's initial noise draw.
            generated = working.decoder(latents)
            client = working.ELBO_Loss(data)
            server = working.ELBO_Loss_server(generated, generated.detach() + .05)
            (client + server).backward()
            optimizer.step()
            rows[step] = (working.state_dict()[name].detach().cpu().clone(),
                          float(client.detach()), float(server.detach()))
    records[device] = rows

report = {}
for step in (1, 2, 5, 10, 20, 50):
    cpu, c_loss, s_loss = records['cpu'][step]
    gpu, g_loss, t_loss = records['cuda'][step]
    error = (cpu - gpu).abs()
    i = error.argmax()
    report[step] = {
        'absolute_max': float(error.max()),
        'cpu_abs_at_max_error': float(cpu.flatten()[i].abs()),
        'relative_at_max_abs_error': float(error.flatten()[i] / cpu.flatten()[i].abs()),
        'tensor_relative_l2': float(torch.linalg.norm(cpu - gpu) / torch.linalg.norm(cpu)),
        'relative_l2_to_cpu_change_since_initial': float(torch.linalg.norm(cpu - gpu) / torch.linalg.norm(cpu - initial)),
        'outside_original_tolerance_elements': int((error > 2e-5 + .002 * cpu.abs()).sum()),
        'client_loss_cpu': c_loss, 'client_loss_cuda': g_loss, 'server_loss_cpu': s_loss, 'server_loss_cuda': t_loss,
    }
print(json.dumps({'torch': torch.__version__, 'device': torch.cuda.get_device_name(),
                  'tensor': name, 'elements': initial.numel(), 'steps': report}, indent=2, allow_nan=False))
