"""Evaluate fixed FedVaDE snapshots with the Round 6 author protocol."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROUND6 = Path('/root/autodl-tmp/fedvade_project/reproduction_runs/round6_checkpoint_diagnosis')
sys.path.insert(0, str(ROUND6))
from diagnose import PARAMS, VaDE, data, evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--external-checkpoint', type=Path)
    p.add_argument('--external-rounds', type=Path)
    a = p.parse_args()
    out = a.run_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'evaluation_summary.json').exists():
        raise SystemExit('Evaluation already exists; refusing overwrite')
    if a.external_checkpoint:
        checkpoints = [a.external_checkpoint]
        if not a.external_rounds:
            raise ValueError('External checkpoint requires historical rounds')
        historical = [json.loads(x) for x in a.external_rounds.read_text().splitlines()]
    else:
        checkpoints = sorted(out.glob('epoch_*_server.pt'))
        historical = [json.loads(x) for x in (out / 'rounds.jsonl').read_text().splitlines()]
    if not checkpoints:
        raise ValueError('No checkpoint found')
    _, X, y, clients, _, _, _ = data()
    results = []
    all_draws = []
    for path in checkpoints:
        payload = torch.load(path, map_location='cpu')
        epoch = int(payload['round'])
        model = VaDE(PARAMS).cpu()
        model.load_state_dict(payload['server'], strict=True)
        model.eval()
        draws = [evaluate(model, 'A_author', X, y, clients, seed) for seed in range(10)]
        ari = np.array([x['ari'] for x in draws])
        purity = np.array([x['purity'] for x in draws])
        last5 = [x['client_loss'] for x in historical[max(0, epoch-5):epoch]]
        if len(last5) != min(5, epoch):
            raise ValueError(f'Missing last five training losses for epoch {epoch}')
        results.append({'epoch': epoch, 'checkpoint': str(path.resolve()),
                        'ari_mean': float(ari.mean()), 'ari_std': float(ari.std()),
                        'purity_mean': float(purity.mean()), 'purity_std': float(purity.std()),
                        'client_loss_last5_mean': float(np.mean(last5))})
        for seed, score in enumerate(draws):
            all_draws.append({'epoch': epoch, 'inference_seed': seed,
                              'ari': score['ari'], 'purity': score['purity']})
    with (out / 'inference.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'inference_seed', 'ari', 'purity'])
        writer.writeheader()
        writer.writerows(all_draws)
    unsup = min(results, key=lambda x: x['client_loss_last5_mean'])
    exploratory = max(results, key=lambda x: x['ari_mean'])
    for record, filename in ((unsup, 'best_checkpoint.pt'),
                             (exploratory, 'best_exploratory_ari_checkpoint.pt')):
        payload = torch.load(record['checkpoint'], map_location='cpu')
        torch.save({'round': payload['round'], 'server': payload['server'],
                    'selection': 'minimum trailing-five client training loss' if filename == 'best_checkpoint.pt'
                                 else 'maximum evaluation ARI; exploratory, label-selected'}, out / filename)
    if a.external_checkpoint:
        (out / 'last_checkpoint.path').write_text(str(a.external_checkpoint.resolve()) + '\n')
    summary = {'protocol': 'round6 A_author, 10 fixed inference seeds per checkpoint',
               'training_seed': 0, 'inference_seeds': list(range(10)),
               'independent_test': False,
               'selection_warning': 'ARI-selected checkpoint and fine configuration use evaluation labels; exploratory only.',
               'best_checkpoint_rule': 'minimum trailing-five client training loss among saved epochs',
               'snapshots': results, 'last': results[-1],
               'best_unlabeled_loss': unsup, 'best_exploratory_ari': exploratory}
    (out / 'evaluation_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'last_epoch': results[-1]['epoch'], 'last_ari': results[-1]['ari_mean'],
                      'best_unlabeled_epoch': unsup['epoch'],
                      'best_exploratory_epoch': exploratory['epoch']}), flush=True)


if __name__ == '__main__':
    main()
