"""Run the author FedVaDE main loop with isolated label-blind CTG adapters on Linux."""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path('/root/autodl-tmp/fedvade_project')
BASE = ROOT / 'reproduction_runs/round10_non_iid_generation'
SOURCE = ROOT / 'fuxian-round2/FedVaDE_source'


def atomic_json(path, value):
    temp = Path(str(path) + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f'Expected one author-code match: {old!r}, got {text.count(old)}')
    return text.replace(old, new)


LOADER = '''import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def make_noniid_clients(csv_path, n_clients=3, batch_size=200, dataname='cardiotocography', seed=42):
    root = os.environ['FEDVADE_ROUND10_DATA']
    scenario = os.environ['FEDVADE_ROUND10_SCENARIO']
    data = np.load(os.path.join(root, 'data.npz'))
    partition = np.load(os.path.join(root, scenario + '_partitions.npz'))
    X = data['X_train']; y = data['y_train']
    loaders = []; sizes = []
    for client in range(3):
        ids = partition[f'client_{client}']
        ds = TensorDataset(torch.tensor(X[ids], dtype=torch.float32),
                           torch.tensor(y[ids], dtype=torch.long))
        loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True))
        sizes.append(len(ids))
    return 3, loaders, (torch.tensor(X), torch.tensor(y)), sizes
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--scenario', choices=('iid', 'medium', 'severe'), required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--rounds', type=int, default=150)
    p.add_argument('--pretrain', type=int, default=50)
    p.add_argument('--save-every', type=int, default=25)
    args = p.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f'Refusing existing run: {out}')
    out.mkdir(parents=True)
    copied = out / 'source'; copied.mkdir()
    for path in SOURCE.glob('*.py'):
        shutil.copy2(path, copied / path.name)
    (copied / 'dataset').mkdir()
    # The author main.py builds a CSV path but the isolated loader ignores it.
    author_main = (copied / 'main.py').read_text()
    author_model = (copied / 'model.py').read_text()
    author_loader = (copied / 'dataloader.py').read_text()
    main_code = replace_once(author_main, 'pre_epoch=50)', f'pre_epoch={args.pretrain})')
    main_code = replace_once(main_code, '            Result.append([L_c, L_s, purity, ari])',
                             '            Result.append([L_c, L_s, purity, ari])\n'
                             '            _observe(epoch + 1, server, clients, L_c, L_s, purity, ari)')
    # The author uses true local labels to choose K. BIC on encoder means is the
    # single explicit algorithm-protocol adaptation; ELBO and aggregation are unchanged.
    model_code = replace_once(author_model, '            K_l = len(np.unique(local_y))',
        "            K_l = min(range(1, min(3, len(local_mu) - 1) + 1), "
        "key=lambda k: GaussianMixture(n_components=k, covariance_type='diag', "
        "random_state=42).fit(local_mu).bic(local_mu))")
    model_code = replace_once(model_code, '                local_y.append(y)\n', '')
    model_code = replace_once(model_code, '            local_y = torch.cat(local_y, 0).cpu().numpy()\n', '')
    (copied / 'main.py').write_text(main_code)
    (copied / 'model.py').write_text(model_code)
    (copied / 'dataloader.py').write_text(LOADER)
    (out / 'source.diff').write_text(''.join(difflib.unified_diff(
        author_main.splitlines(True), main_code.splitlines(True), 'author/main.py', 'adapter/main.py')) +
        ''.join(difflib.unified_diff(author_model.splitlines(True), model_code.splitlines(True),
                                    'author/model.py', 'adapter/model.py')) +
        ''.join(difflib.unified_diff(author_loader.splitlines(True), LOADER.splitlines(True),
                                    'author/dataloader.py', 'adapter/dataloader.py')))
    partition = BASE / 'private_data' / f'{args.scenario}_partitions.npz'
    data_file = BASE / 'private_data/data.npz'
    config = {'kind': 'label_blind_fedvade_baseline', 'scenario': args.scenario,
              'seed': args.seed, 'rounds': args.rounds, 'pretrain': args.pretrain,
              'batch_size': 200, 'clients': 3, 'latent_dim': 10, 'global_K': 3,
              'local_K': 'unlabeled BIC over 1,2,3 encoder-mean GMM fits',
              'lr': .002, 'alpha': .01, 'sample': 100,
              'scheduler': 'author client per mini-batch/server per round',
              'partition_sha256': sha(partition), 'data_sha256': sha(data_file),
              'source_sha256': {q.name: sha(q) for q in SOURCE.glob('*.py')},
              'executed_sha256': {q.name: sha(q) for q in copied.glob('*.py')},
              'command': [sys.executable, *sys.argv], 'device': 'cpu',
              'selection': 'minimum deterministic validation reconstruction MSE among saved checkpoints; no labels'}
    atomic_json(out / 'config.json', config)
    os.environ['FEDVADE_ROUND10_DATA'] = str(BASE / 'private_data')
    os.environ['FEDVADE_ROUND10_SCENARIO'] = args.scenario
    os.chdir(copied); sys.path.insert(0, str(copied)); sys.dont_write_bytecode = True
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    val = np.load(data_file)['X_val'].astype(np.float32)
    val_tensor = torch.from_numpy(val)
    best = {'mse': float('inf'), 'epoch': None, 'checkpoint': None}
    rows = []
    started = time.time()
    original_parse = argparse.ArgumentParser.parse_args

    def cpu_parse(parser, *a, **kw):
        parsed = original_parse(parser, *a, **kw)
        if hasattr(parsed, 'cuda'): parsed.cuda = False
        return parsed

    def observe(epoch, server, clients, client_loss, server_loss, purity, ari):
        row = {'epoch': epoch, 'client_loss': float(client_loss), 'server_loss': float(server_loss),
               'training_partition_ari': float(ari), 'training_partition_purity': float(purity),
               'client_lr': [c['optimizer'].param_groups[0]['lr'] for c in clients],
               'server_lr': server['optimizer'].param_groups[0]['lr'],
               'server_gmm_weights': torch.softmax(server['vade'].rho_.detach(), 0).tolist(),
               'elapsed_seconds': time.time() - started}
        if epoch % args.save_every == 0 or epoch == args.rounds:
            with torch.no_grad():
                model = server['vade']; model.eval()
                mu, _ = model.encoder(val_tensor)
                decoded = model.decoder(mu)
                mse = float(torch.mean((decoded - val_tensor) ** 2))
                model.train()
            row['validation_reconstruction_mse'] = mse
            state = {k: v.detach().cpu().clone() for k, v in server['vade'].state_dict().items()}
            file = out / f'epoch_{epoch:03d}_server.pt'
            torch.save({'round': epoch, 'server': state}, file)
            if mse < best['mse']:
                best.update(mse=mse, epoch=epoch, checkpoint=str(file))
                torch.save({'round': epoch, 'server': state, 'selection': 'unlabeled validation reconstruction MSE'},
                           out / 'selected_server.pt')
        if epoch == args.rounds:
            torch.save({'round': epoch, 'server': {k: v.detach().cpu().clone()
                        for k, v in server['vade'].state_dict().items()},
                        'clients': [{k: v.detach().cpu().clone() for k, v in c['vade'].state_dict().items()}
                                    for c in clients],
                        'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                        'torch_rng': torch.get_rng_state()}, out / 'last_full.pt')
        rows.append(row)
        with (out / 'rounds.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        progress = BASE / 'progress.json'
        status = json.loads(progress.read_text()) if progress.exists() else {}
        status.update(epoch=epoch, epoch_total=args.rounds,
                      latest_ari=float(ari), latest_purity=float(purity), updated_at=time.time())
        atomic_json(progress, status)

    argv = [str(copied / 'main.py'), '--dataset', 'cardiotocography',
            '--epoch', str(args.rounds), '--client_num', '3', '--lr', '.002',
            '--alpha', '.01', '--sample', '100']
    result = {'status': 'running', 'config': config}
    atomic_json(out / 'result.json', result)
    try:
        with threadpool_limits(limits=1), patch.object(sys, 'argv', argv), \
             patch.object(argparse.ArgumentParser, 'parse_args', cpu_parse):
            exec(compile(main_code, str(copied / 'main.py'), 'exec'),
                 {'__name__': '__main__', '__file__': str(copied / 'main.py'), '_observe': observe})
        if len(rows) != args.rounds:
            raise RuntimeError(f'Incomplete rounds {len(rows)}/{args.rounds}')
        result.update(status='ok', best_validation=best, last=rows[-1],
                      elapsed_seconds=time.time() - started)
    except BaseException as error:
        result.update(status='failed', error=repr(error), elapsed_seconds=time.time() - started)
        raise
    finally:
        atomic_json(out / 'result.json', result)


if __name__ == '__main__':
    main()
