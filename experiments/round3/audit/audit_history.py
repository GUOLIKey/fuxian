"""Inspect the supplied checkpoint and reconstruct historical partition fingerprints."""
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, pstdev
import sys

import numpy as np
import pandas as pd
import torch

WORKSPACE = Path(__file__).resolve().parents[3]
RUNS = WORKSPACE / 'reproduction_runs'
SOURCE = WORKSPACE / 'FedVaDE_source'
sys.path.insert(0, str(SOURCE))
from dataloader import make_noniid_clients

data = SOURCE / 'dataset/breast_cancer.csv'
checkpoint = SOURCE / 'model/server_model_breast_cancer.pk'
state = torch.load(checkpoint, map_location='cpu', weights_only=True)
_, loaders, (x, _), _ = make_noniid_clients(data, 10, 200, 'breast_cancer')
partition = [{'size': len(loader.dataset), 'batches': len(loader),
              'sha256': [sha256(tensor.numpy().tobytes()).hexdigest()
                         for tensor in loader.dataset.tensors]} for loader in loaders]
reference = json.loads((RUNS / 'round2/autodl-results/formal/B_author_seed0/result.json').read_text())
old_reads = RUNS / 'new_runs/2026-11-19/fedvade_author_300'
historical = []
for seed in range(10):
    run = old_reads / f'breast_cancer_seed{seed}'
    command = json.loads((run / 'command.json').read_text())
    result = json.loads((run / 'metrics.json').read_text())
    historical.append({'seed': seed, 'platform': command['platform'], 'versions': command['versions'],
                       'partition_seed': command['partition_seed'], 'GMM_seed': command['gmm_fit_seed'],
                       'source_hashes_match_round_two': command['sha256'] == reference['source_hashes'],
                       'client_sizes_match_round_two': result['client_sizes'] == [i['size'] for i in partition],
                       'scored_rounds': len(result['rows']), 'pretrain_rounds': result['pretrain_rounds'],
                       'config': result['author_config'], 'final': result['rows'][-1]})
aris = [row['final']['ari'] for row in historical]
author_table = pd.read_csv(data)
uci_table = pd.read_csv(RUNS / 'upstream/breast/wdbc.data', header=None)
labels = dict(zip(author_table.iloc[:, -1].astype(str), uci_table.iloc[:, 1]))
assert np.array_equal(author_table.iloc[:, -1].astype(str).map(labels), uci_table.iloc[:, 1])
public = {'checkpoint_sha256': sha256(checkpoint.read_bytes()).hexdigest(), 'checkpoint_type': type(state).__name__,
          'tensor_shapes': {name: list(value.shape) for name, value in state.items() if torch.is_tensor(value)},
          'non_tensor_keys': [name for name, value in state.items() if not torch.is_tensor(value)],
          'state_dict_metadata': getattr(state, '_metadata', None),
          'partition_reconstructed_matches_round_two': partition == reference['partition'],
          'reconstructed_manifest': partition, 'csv_rows': len(x), 'csv_features': x.shape[1],
          'historical_mean_ari': mean(aris), 'historical_population_sd_ari': pstdev(aris),
          'uci_wdbc_match': {
              'all_ids_match_A1': bool(np.array_equal(author_table.iloc[:, 0], uci_table.iloc[:, 0])),
              'all_30_measurements_match_A2_through_A31': bool(
                  np.array_equal(author_table.iloc[:, 1:-1], uci_table.iloc[:, 2:])),
              'label_mapping': labels,
          },
          'historical': historical}
Path(__file__).with_name('fedvade-history.json').write_text(json.dumps(public, indent=2) + '\n')
