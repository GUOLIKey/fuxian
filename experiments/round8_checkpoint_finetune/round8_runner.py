"""Weight-initialized FedVaDE fine-tuning; never described as exact resume."""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one author source match: {old!r}")
    return source.replace(old, new)


def atomic_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def instrument(source, schedule):
    if schedule == 'epoch':
        source = once(source, "                    c['lr_s'].step()\n", '')
        point = "                clients_vae_param.append(c['vade'].state_dict())"
        source = once(source, point, "                c['lr_s'].step()\n" + point)
    source = once(source, 'pre_train(args, server, clients, weight, pre_epoch=50)',
                  '_load_initial_state(args, server, clients, weight)')
    point = '            Result.append([L_c, L_s, purity, ari])'
    source = once(source, point, point + "\n            _observe(epoch + 1, server, clients, DL, L_c, L_s, purity, ari)")
    return source


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--initial-bundle', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--lr', type=float, required=True)
    p.add_argument('--alpha', type=float, required=True)
    p.add_argument('--schedule', choices=('author', 'epoch'), required=True)
    p.add_argument('--sample', type=int, default=100)
    p.add_argument('--rounds', type=int, required=True)
    p.add_argument('--save-every', type=int, default=25)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    assert args.seed == 0 and args.rounds > 0
    src, out = args.source.resolve(), args.out.resolve()
    bundle_path = args.initial_bundle.resolve()
    if not bundle_path.is_file():
        raise SystemExit(f'Initial bundle missing: {bundle_path}')
    if out.exists():
        raise SystemExit(f'Existing run directory: {out}')
    out.mkdir(parents=True)
    copy = out / 'source'
    copy.mkdir()
    for path in src.glob('*.py'):
        shutil.copy2(path, copy / path.name)
    (copy / 'dataset').mkdir()
    shutil.copy2(src / 'dataset/breast_cancer.csv', copy / 'dataset/breast_cancer.csv')
    if (copy / 'model/server_model_breast_cancer.pk').exists():
        raise RuntimeError('Author weights must not initialize a search run')
    entry = copy / 'main.py'
    original = entry.read_text()
    updated = instrument(original, args.schedule)
    (out / 'source.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True), updated.splitlines(True), 'author/main.py', 'copy/main.py')))
    entry.write_text(updated)
    source_hashes = {str(q.relative_to(copy)): sha(q) for q in copy.rglob('*') if q.is_file()}
    config = {'seed': 0, 'lr': args.lr, 'alpha': args.alpha, 'schedule': args.schedule,
              'sample': args.sample, 'pretrain': 0, 'rounds': args.rounds,
              'save_every': args.save_every, 'batch_size': 200, 'clients': 10,
              'latent_dim': 10, 'clusters': 2, 'partition_seed': 42, 'gmm_seed': 42,
              'gamma': .85, 'step_size': 10, 'device': 'cpu', 'source_dir': str(src),
              'initial_bundle': str(bundle_path), 'initial_bundle_sha256': sha(bundle_path),
              'mode': 'weight_initialization_not_exact_resume',
              'optimizer_reset': True, 'scheduler_reset': True,
              'author_source_hashes': {str(q.relative_to(copy)): sha(src / q.relative_to(copy)) for q in copy.rglob('*') if q.is_file() and (src / q.relative_to(copy)).exists()},
              'executed_source_hashes': source_hashes,
              'command': [sys.executable, *sys.argv], 'platform': platform.platform()}
    atomic_json(out / 'config.json', config)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.chdir(copy)
    sys.path.insert(0, str(copy))
    sys.dont_write_bytecode = True
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    bundle = torch.load(bundle_path, map_location='cpu')
    if len(bundle['clients']) != 10:
        raise RuntimeError('Initial bundle must contain ten client models')
    config['torch'] = torch.__version__
    config['numpy'] = np.__version__
    atomic_json(out / 'config.json', config)
    parse_args = argparse.ArgumentParser.parse_args
    rows = []
    started = perf_counter()

    def force_cpu(parser, *a, **kw):
        parsed = parse_args(parser, *a, **kw)
        if hasattr(parsed, 'cuda'):
            parsed.cuda = False
        return parsed

    def load_initial_state(author_args, server, clients, weight):
        server['vade'].load_state_dict(bundle['server'], strict=True)
        for client, state in zip(clients, bundle['clients']):
            # Author local GMMs have one or two components; resize before strict load.
            for key in ('rho_', 'mu_c', 'log_sigma2_c'):
                getattr(client['vade'], key).data = state[key].clone()
            client['vade'].load_state_dict(state, strict=True)
        all_states = [server['vade'].state_dict()] + [c['vade'].state_dict() for c in clients]
        if not all(all(torch.isfinite(v).all() for v in state.values()) for state in all_states):
            raise RuntimeError('Nonfinite initial state')
        got = [{'size': len(d.dataset),
                'tensor_sha256': [hashlib.sha256(t.numpy().tobytes()).hexdigest() for t in d.dataset.tensors]}
               for d in (c['dataloader'] for c in clients)]
        for actual, expected in zip(got, bundle['partition']):
            if actual['size'] != expected['size'] or actual['tensor_sha256'] != expected['tensor_sha256']:
                raise RuntimeError('Client partition differs from initial bundle')
        atomic_json(out / 'initial_state_verified.json',
                    {'initial_bundle_sha256': sha(bundle_path), 'client_gmm_components':
                     [len(c['vade'].rho_) for c in clients], 'partition_verified': True,
                     'optimizer_recreated_after_initial_load': True})

    def snapshot(epoch, server, clients, full=False):
        model_state = {k: v.detach().cpu().clone() for k, v in server['vade'].state_dict().items()}
        if full:
            data = {'round': epoch, 'seed': 0, 'server': model_state,
                    'clients': [{k: v.detach().cpu().clone() for k, v in c['vade'].state_dict().items()} for c in clients],
                    'server_optimizer': server['optimizer'].state_dict(),
                    'client_optimizers': [c['optimizer'].state_dict() for c in clients],
                    'server_scheduler': server['lr_s'].state_dict(),
                    'client_schedulers': [c['lr_s'].state_dict() for c in clients],
                    'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                    'torch_rng': torch.get_rng_state()}
            torch.save(data, out / 'last_checkpoint.pt')
        torch.save({'round': epoch, 'server': model_state}, out / f'epoch_{epoch:03d}_server.pt')

    def observe(epoch, server, clients, loaders, lc, ls, purity, ari):
        row = {'round': epoch, 'client_loss': float(lc), 'server_loss': float(ls),
               'purity': float(purity), 'ari': float(ari),
               'client_lr': [c['optimizer'].param_groups[0]['lr'] for c in clients],
               'server_lr': server['optimizer'].param_groups[0]['lr'],
               'elapsed_seconds': perf_counter() - started}
        rows.append(row)
        with (out / 'rounds.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        shared = out.parent / 'progress.json'
        prior = json.loads(shared.read_text()) if shared.exists() else {}
        atomic_json(shared, {'stage': 'FINETUNE', 'checkpoint_current': None,
                  'checkpoint_total': None, 'best_checkpoint': prior.get('best_checkpoint', 'F01_epoch100'),
                  'best_ari_so_far': prior.get('best_ari_so_far'),
                  'current_group': out.name, 'epoch': epoch, 'budget': args.rounds,
                  'latest_ari': float(ari), 'latest_purity': float(purity),
                  'status': 'RUNNING', 'updated_at': __import__('time').time()})
        if epoch % args.save_every == 0 or epoch == args.rounds:
            snapshot(epoch, server, clients, full=epoch == args.rounds)
        if epoch == 1:
            config['partition'] = [{'size': len(d.dataset), 'batches': len(d),
                                    'tensor_sha256': [hashlib.sha256(t.numpy().tobytes()).hexdigest() for t in d.dataset.tensors]}
                                   for d in loaders]
            atomic_json(out / 'config.json', config)

    author_argv = [str(entry), '--dataset', 'breast_cancer', '--epoch', str(args.rounds),
                   '--lr', str(args.lr), '--alpha', str(args.alpha), '--sample', str(args.sample)]
    result = {'status': 'running', 'config': config, 'author_argv': author_argv}
    atomic_json(out / 'result.json', result)
    try:
        with threadpool_limits(limits=1), patch.object(sys, 'argv', author_argv), patch.object(argparse.ArgumentParser, 'parse_args', force_cpu):
            namespace = {'__name__': '__main__', '__file__': str(entry), '_observe': observe,
                         '_load_initial_state': load_initial_state}
            exec(compile(updated, str(entry), 'exec'), namespace)
        if len(rows) != args.rounds:
            raise RuntimeError(f'Incomplete rounds: {len(rows)} / {args.rounds}')
        result.update(status='ok', last=rows[-1], saved_epochs=[r for r in range(1, args.rounds + 1) if r % args.save_every == 0 or r == args.rounds])
    except BaseException as exc:
        result.update(status='failed', error=repr(exc))
        raise
    finally:
        result['elapsed_seconds'] = perf_counter() - started
        atomic_json(out / 'result.json', result)


if __name__ == '__main__':
    main()
