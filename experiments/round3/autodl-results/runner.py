"""Run exactly one author-code diagnostic in an isolated experiment directory."""

import argparse
import difflib
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import random
import runpy
import shutil
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def replace_once(source, original, updated):
    if source.count(original) != 1:
        raise ValueError(f'Expected one pinned source match: {original!r}')
    return source.replace(original, updated)


def edit(path, updated, destination):
    original = path.read_text()
    with destination.open('a') as stream:
        stream.writelines(difflib.unified_diff(
            original.splitlines(True), updated.splitlines(True),
            fromfile=f'author/{path.name}', tofile=f'copy/{path.name}',
        ))
    path.write_text(updated)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def dataset_manifest(clients):
    return [{'size': len(item['label']), 'classes': np.unique(item['label']).tolist(),
             'features': int(np.asarray(item['data']).shape[1]),
             'data_sha256': sha256(np.ascontiguousarray(item['data']).tobytes()).hexdigest(),
             'label_sha256': sha256(np.ascontiguousarray(item['label']).tobytes()).hexdigest()}
            for item in clients]


def classical(args, copy, result):
    author = runpy.run_path('main.py')
    if args.model == 'hire':
        clients = author['getData']('iris')
        result['partition'] = dataset_manifest(clients)
        result['params'] = {'rate': args.rate, 'init_k': .5, 'clients': 10,
                            'vcc_max': 3, 'fixed_author_local_vcc_seed': 42}
        if args.smoke:
            from MultiClustering import FCPL
            FCPL(clients[0]['data'], args.rate, .5)
            return
        prediction = author['fedClustering'](clients, args.rate, .5)
        values = author['validation'](prediction)
        result.update(zip(('purity', 'ari', 'nmi', 'acc'), map(float, values)))
        return

    with patch.object(sys, 'argv', ['main.py', '--dataset', 'EC', '--runs', '1']):
        defaults = author['parse_args']()
    clients = author['load_clients'](copy / defaults.data_dir / 'EC.npz')
    result['partition'] = dataset_manifest(clients)
    result['params'] = {**vars(defaults), 'clients': len(clients)}
    result['global_class_count'] = len(np.unique(np.concatenate([item['label'] for item in clients])))
    if args.smoke:
        from algorithm.MultiClustering import fine_cd
        fine_cd(clients[0]['data'], defaults.rate, defaults.k0)
        return
    prediction, centroids, status = author['fedClustering'](clients, defaults.rate, defaults.k0, defaults.k1)
    result['author_status'] = status
    if status != 'success':
        raise RuntimeError('Author FedHCA reported no valid clusters.')
    values = author['validation'](prediction)
    result.update(purity=float(values['Purity']), ari=float(values['ARI']),
                  global_clusters=len(centroids))


def fedvade_source(source, mode, smoke):
    if smoke:
        source = replace_once(source, 'pre_epoch=50)', 'pre_epoch=3)')
    marker = "                    c['lr_s'].step()\n"
    if mode == 'epoch':
        source = replace_once(source, marker, '')
        point = '                clients_vae_param.append(c[\'vade\'].state_dict())'
        source = replace_once(source, point, "                c['lr_s'].step()\n" + point)
    source = replace_once(source, '        for epoch in epoch_bar:\n',
                          '        for epoch in epoch_bar:\n            observed_lrs = []\n'
                          '            observed_started = _now()\n')
    point = "            for c in clients:\n                for x, _ in c['dataloader']:\n"
    source = replace_once(source, point, "            for c in clients:\n"
                          "                client_lrs = []\n                observed_lrs.append(client_lrs)\n"
                          "                for x, _ in c['dataloader']:\n"
                          "                    client_lrs.append(c['optimizer'].param_groups[0]['lr'])\n")
    source = replace_once(source, "            server['optimizer'].step()\n",
                          "            observed_server_lr = server['optimizer'].param_groups[0]['lr']\n"
                          "            server['optimizer'].step()\n")
    point = '            Result.append([L_c, L_s, purity, ari])'
    return replace_once(source, point, point + """
            _observe(round=epoch + 1, client_batch_lrs=observed_lrs,
                     client_end_lrs=[c['optimizer'].param_groups[0]['lr'] for c in clients],
                     server_lr=observed_server_lr, server_end_lr=server['optimizer'].param_groups[0]['lr'],
                     client_loss=float(L_c), server_loss=float(L_s), purity=float(purity), ari=float(ari),
                     seconds=_now() - observed_started)
""")


def fedvade(args, copy, result):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA device is not available.')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result['torch'] = torch.__version__
    result['cuda_version'] = torch.version.cuda
    result['device_name'] = torch.cuda.get_device_name() if args.device == 'cuda' else 'CPU'
    result['params'] = {'clients': 10, 'batch_size': 200, 'latent_dim': 10, 'initial_lr': .002,
                        'gamma': .85, 'stepsize': 10, 'alpha': .01, 'sample': 100,
                        'pretrain_rounds': 3 if args.smoke else 50,
                        'rounds': 12 if args.smoke else 300, 'partition_seed': 42,
                        'gmm_seed': 42, 'client_schedule': args.schedule}
    plan = result['params']
    entry = copy / 'main.py'
    edit(entry, fedvade_source(entry.read_text(), args.schedule, args.smoke), args.out / 'source.diff')
    configure = argparse.ArgumentParser.parse_args
    rows = []

    def author_args(parser, *parameters, **options):
        parsed = configure(parser, *parameters, **options)
        if hasattr(parsed, 'cuda'):
            parsed.cuda = args.device == 'cuda'
        return parsed

    def observe(**row):
        rows.append(row)
        with (args.out / 'rounds.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')

    def now():
        if args.device == 'cuda':
            torch.cuda.synchronize()
        return perf_counter()

    generator = torch.Generator(device='cpu').manual_seed(1000 + args.seed)

    def common_noise(tensor):
        return torch.randn(tensor.shape, dtype=tensor.dtype, generator=generator).to(tensor.device)

    result['author_argv'] = [str(entry), '--dataset', 'breast_cancer', '--epoch', str(plan['rounds'])]
    with patch.object(sys, 'argv', result['author_argv']), patch.object(
            argparse.ArgumentParser, 'parse_args', author_args):
        namespace = {'_observe': observe, '_now': now, '__name__': '__main__', '__file__': str(entry)}
        code = compile(entry.read_text(), str(entry), 'exec')
        if args.smoke:
            with patch.object(torch, 'randn_like', common_noise):
                exec(code, namespace)
        else:
            exec(code, namespace)
    if len(rows) != plan['rounds']:
        raise RuntimeError('Round trace is incomplete.')
    result['partition'] = [{'size': len(loader.dataset), 'batches': len(loader),
                            'sha256': [sha256(tensor.numpy().tobytes()).hexdigest()
                                       for tensor in loader.dataset.tensors]}
                           for loader in namespace['DL']]
    result['last'] = rows[-1]
    if not args.smoke:
        result['round_150'] = rows[149]
        torch.save({'round': plan['rounds'],
                    'server': {key: tensor.detach().cpu() for key, tensor in
                               namespace['server']['vade'].state_dict().items()}},
                   args.out / 'round_300_server.pt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=('hire', 'hca', 'fedvade'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rate', type=float, default=.0001)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--schedule', choices=('author', 'epoch'), default='author')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.source, args.out = args.source.resolve(), args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'result.json').exists() or (args.out / 'source').exists():
        raise SystemExit('Choose a new result directory.')
    sys.dont_write_bytecode = True
    copy = args.out / 'source'
    dataset = {'fedvade': 'breast_cancer.csv', 'hire': 'iris.npz', 'hca': 'EC.npz'}[args.model]
    copy.mkdir()
    source = args.source
    for file in source.glob('*.py'):
        shutil.copy2(file, copy)
    if args.model == 'hca':
        shutil.copytree(source / 'algorithm', copy / 'algorithm', ignore=shutil.ignore_patterns('__pycache__'))
    (copy / 'dataset').mkdir()
    shutil.copy2(source / 'dataset' / dataset, copy / 'dataset' / dataset)
    result = {'model': args.model, 'seed': args.seed, 'device': args.device,
              'smoke': args.smoke, 'rate': args.rate, 'schedule': args.schedule,
              'command': [sys.executable, *sys.argv], 'platform': platform.platform(),
              'python': platform.python_version(), 'numpy': np.__version__,
              'source_hashes': {str(file.relative_to(copy)): digest(file) for file in copy.rglob('*')
                                if file.is_file()}}
    write_json(args.out / 'command.json', result)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.model == 'hire':
        file = copy / 'MultiClustering.py'
        paired = replace_once(file.read_text(), 'rng = np.random.default_rng()',
                              f'rng = np.random.default_rng({args.seed})')
        paired = replace_once(paired, 'def _batch_cpl(X, rate, k, random_state=None):',
                              f'_paired_mcpl_rng = np.random.default_rng(np.random.SeedSequence([{args.seed}, 1]))\n\n'
                              'def _batch_cpl(X, rate, k, random_state=None):')
        paired = replace_once(paired, 'rng = np.random.default_rng(random_state)',
                              'rng = _paired_mcpl_rng if random_state is None else np.random.default_rng(random_state)')
        edit(file, paired, args.out / 'source.diff')
    os.chdir(copy)
    sys.path.insert(0, str(copy))
    started = perf_counter()
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            if args.model == 'fedvade':
                fedvade(args, copy, result)
            else:
                classical(args, copy, result)
        result['status'] = 'ok'
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        result['elapsed_seconds'] = perf_counter() - started
        write_json(args.out / 'result.json', result)


if __name__ == '__main__':
    main()
