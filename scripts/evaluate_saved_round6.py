"""Round 6 author-protocol evaluator, reimplemented on this server.

The original `reproduction_runs/round6_checkpoint_diagnosis/diagnose.py` was
not migrated to this server.  This script reproduces its evaluation protocol
and is validated against the recorded F01 epoch-100 per-seed values:

  torch.manual_seed(seed); iterate the author's client DataLoaders batch by
  batch; model.predict(batch); concatenate predictions and labels.

The model is constructed and loaded once per checkpoint, before the seed
loop.  Building the model after seeding would consume RNG and change the
predictions (this was verified during reconstruction).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path('/root/autodl-tmp/fedvade_project')
SOURCE = ROOT / 'fuxian-round2/FedVaDE_source'
DATA = SOURCE / 'dataset/breast_cancer.csv'
sys.path.insert(0, str(SOURCE))
from dataloader import make_noniid_clients  # noqa: E402
from model import VaDE  # noqa: E402


def cluster_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    d = int(max(y_true.max(), y_pred.max())) + 1
    conf = np.zeros((d, d), dtype=np.int64)
    for a, b in zip(y_true, y_pred):
        conf[int(a), int(b)] += 1
    r, c = linear_sum_assignment(-conf)
    return {
        'ari': float(adjusted_rand_score(y_true, y_pred)),
        'purity': float(np.max(conf, axis=0).sum() / len(y_true)),
        'nmi': float(normalized_mutual_info_score(y_true, y_pred)),
        'acc': float(conf[r, c].sum() / len(y_true)),
    }


def make_vade(k, alpha):
    return VaDE(SimpleNamespace(input_dim=31, hid_dim=10, nClusters=int(k),
                                alpha=float(alpha), cuda=False))


def build_loaders():
    _, loaders, (Xt, yt), sizes = make_noniid_clients(
        str(DATA), 10, 200, 'breast_cancer', seed=42)
    return loaders, [int(s) for s in sizes]


def evaluate_one(model, loaders, seed):
    torch.manual_seed(int(seed))
    preds, labels = [], []
    for loader in loaders:
        for x, y in loader:
            preds.append(model.predict(x))
            labels.append(y.numpy())
    return cluster_metrics(np.concatenate(labels, 0), np.concatenate(preds, 0))


def evaluate_checkpoint(path, alpha, loaders, seeds=range(10)):
    payload = torch.load(path, map_location='cpu')
    state = payload['server']
    epoch = int(payload.get('round', 0))
    # Construct before seeding: construction consumes RNG and must not affect
    # the per-seed prediction stream.
    model = make_vade(len(state['rho_']), alpha)
    model.load_state_dict(state, strict=True)
    model.eval()
    return epoch, [evaluate_one(model, loaders, s) for s in seeds]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--seeds', type=int, default=10)
    args = p.parse_args()
    run = args.run_dir.resolve()
    config = json.loads((run / 'config.json').read_text())
    alpha = float(config['alpha'])
    loaders, sizes = build_loaders()
    checkpoints = sorted(run.glob('epoch_*_server.pt'))
    if not checkpoints:
        raise SystemExit(f'No epoch_*_server.pt in {run}')
    rounds = [json.loads(x) for x in (run / 'rounds.jsonl').read_text().splitlines() if x.strip()]

    snapshots, all_draws = [], []
    for path in checkpoints:
        epoch, draws = evaluate_checkpoint(path, alpha, loaders, range(args.seeds))
        ari = np.array([d['ari'] for d in draws])
        purity = np.array([d['purity'] for d in draws])
        nmi = np.array([d['nmi'] for d in draws])
        acc = np.array([d['acc'] for d in draws])
        last5 = [r['client_loss'] for r in rounds[max(0, epoch - 5):epoch]]
        if len(last5) != min(5, epoch):
            raise ValueError(f'Missing last-five losses at epoch {epoch}')
        snapshots.append({
            'epoch': epoch, 'checkpoint': str(path.resolve()),
            'ari_mean': float(ari.mean()), 'ari_std': float(ari.std()),
            'purity_mean': float(purity.mean()), 'purity_std': float(purity.std()),
            'nmi_mean': float(nmi.mean()), 'nmi_std': float(nmi.std()),
            'acc_mean': float(acc.mean()), 'acc_std': float(acc.std()),
            'client_loss_last5_mean': float(np.mean(last5)),
            'n_inference_seeds': int(args.seeds),
        })
        for seed, score in enumerate(draws):
            all_draws.append({'epoch': epoch, 'inference_seed': seed,
                              'ari': score['ari'], 'purity': score['purity'],
                              'nmi': score['nmi'], 'acc': score['acc']})

    with (run / 'inference.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'inference_seed',
                                               'ari', 'purity', 'nmi', 'acc'])
        writer.writeheader()
        writer.writerows(all_draws)

    unsup = min(snapshots, key=lambda x: x['client_loss_last5_mean'])
    exploratory = max(snapshots, key=lambda x: x['ari_mean'])
    for record, filename in ((unsup, 'best_checkpoint.pt'),
                             (exploratory, 'best_exploratory_ari_checkpoint.pt')):
        payload = torch.load(record['checkpoint'], map_location='cpu')
        torch.save({'round': payload['round'], 'server': payload['server'],
                    'selection': ('minimum trailing-five client training loss'
                                  if filename == 'best_checkpoint.pt'
                                  else 'maximum evaluation ARI; exploratory, '
                                       'label-selected')}, run / filename)
    summary = {
        'protocol': 'round6 A_author, 10 fixed inference seeds per checkpoint',
        'training_seed': 0, 'inference_seeds': list(range(args.seeds)),
        'independent_test': False,
        'selection_warning': ('ARI-selected checkpoint and fine configuration '
                              'use evaluation labels; exploratory only.'),
        'best_checkpoint_rule': ('minimum trailing-five client training loss '
                                 'among saved epochs'),
        'partition_sizes': sizes,
        'snapshots': snapshots,
        'last': snapshots[-1],
        'best_unlabeled_loss': unsup,
        'best_exploratory_ari': exploratory,
    }
    (run / 'evaluation_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'run': run.name, 'last_epoch': snapshots[-1]['epoch'],
                      'last_ari': snapshots[-1]['ari_mean'],
                      'best_unlabeled_epoch': unsup['epoch'],
                      'best_exploratory_epoch': exploratory['epoch']}), flush=True)


if __name__ == '__main__':
    main()
