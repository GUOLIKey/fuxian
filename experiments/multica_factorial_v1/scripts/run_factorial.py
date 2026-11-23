"""Run one group/seed of the 2x2 Multi-CA factorial experiment.

A: author config, original aggregation
B: author config, Multi-CA aggregation
C: F01 config, original aggregation
D: F01 config, Multi-CA aggregation

OFF paths preserve the author `gmm_sampling` call.  ON paths replace only the
server latent sampling distribution with the capacity-matched Multi-CA macro GMM.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import random
import shutil
import sys
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np

ROOT = Path('/root/autodl-tmp/fedvade_project')
SOURCE = ROOT / 'fuxian-round2/FedVaDE_source'
BASE = ROOT / 'experiments/multica_factorial_v1'
RUNS = ROOT / 'reproduction_runs/multica_factorial_v1'
sys.path.insert(0, str(BASE / 'scripts'))

from analysis_utils import clustering_metrics, diag_nll, encode_mean, gmm_state_metrics  # noqa: E402
from multica_core import (  # noqa: E402
    multi_ca_merge, reconstruct_client_components, sample_macro_for_author_loop,
    summary_for_log,
)

GROUPS = {
    'A': {'alpha': 0.01, 'schedule': 'author', 'aggregation': 'original'},
    'B': {'alpha': 0.01, 'schedule': 'author', 'aggregation': 'multi_ca'},
    'C': {'alpha': 0.05, 'schedule': 'epoch', 'aggregation': 'original'},
    'D': {'alpha': 0.05, 'schedule': 'epoch', 'aggregation': 'multi_ca'},
}


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


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f'Expected one author match: {old!r}, got {text.count(old)}')
    return text.replace(old, new)


def state_cpu(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save_payload(path, payload):
    import torch
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--group', choices=tuple(GROUPS), required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--pretrain', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=300)
    p.add_argument('--save-every', type=int, default=25)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = p.parse_args()
    if not args.pretrain.is_file():
        raise SystemExit(f'Missing pretrain checkpoint: {args.pretrain}')
    cfg = GROUPS[args.group]
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f'Refusing existing run: {out}')
    out.mkdir(parents=True)
    ensure = out / 'checkpoints'
    ensure.mkdir()

    copied = out / 'source'
    copied.mkdir()
    (copied / 'dataset').mkdir()
    for path in SOURCE.glob('*.py'):
        shutil.copy2(path, copied / path.name)
    shutil.copy2(SOURCE / 'dataset/breast_cancer.csv', copied / 'dataset/breast_cancer.csv')
    author_main = (copied / 'main.py').read_text()
    code = author_main

    # Shared pretrain: the author pretrain is identical for A/B/C/D of a seed.
    code = replace_once(code, 'pre_train(args, server, clients, weight, pre_epoch=50)',
                        '_load_pretrain(server, clients)')
    if cfg['schedule'] == 'epoch':
        # F01: client scheduler once per client per epoch, not per minibatch.
        code = replace_once(code, "                    c['lr_s'].step()\n", '')
        point = "                clients_vae_param.append(c['vade'].state_dict())"
        code = replace_once(code, point, "                c['lr_s'].step()\n" + point)
    if cfg['aggregation'] == 'multi_ca':
        code = replace_once(
            code,
            '            Z_hat, y_hat, lengths, _, _, _ = gmm_sampling(clients_gmm_param, weight, args.sample)',
            '            Z_hat, y_hat, lengths, _mc_info = _sample_union(\n'
            '                clients_gmm_param, weight, args.sample, epoch + 1, size)',
        )
        point = '            Result.append([L_c, L_s, purity, ari])'
        code = replace_once(code, point, point + """
            _observe(epoch + 1, server, clients, L_c, L_s, purity, ari, _mc_info)
""")
    else:
        point = '            Result.append([L_c, L_s, purity, ari])'
        code = replace_once(code, point, point + """
            _observe(epoch + 1, server, clients, L_c, L_s, purity, ari, None)
""")
    (copied / 'main.py').write_text(code)
    (out / 'source.diff').write_text(''.join(difflib.unified_diff(
        author_main.splitlines(True), code.splitlines(True),
        'author/main.py', 'executed/main.py')))

    sys.path.insert(0, str(SOURCE))
    from dataloader import make_noniid_clients
    _, loaders, (X_eval, y_eval), sizes = make_noniid_clients(
        str(SOURCE / 'dataset/breast_cancer.csv'), 10, 200, 'breast_cancer', seed=42)
    X_eval = X_eval.numpy().astype(np.float32)
    y_eval = y_eval.numpy().astype(np.int64)
    sizes = np.asarray(sizes, dtype=np.int64)
    client_X_eval = np.concatenate([dl.dataset.tensors[0].numpy().astype(np.float32)
                                    for dl in loaders], axis=0)
    client_y_eval = np.concatenate([dl.dataset.tensors[1].numpy().astype(np.int64)
                                    for dl in loaders], axis=0)

    import torch
    use_cuda = args.device == 'cuda' and torch.cuda.is_available()
    if args.device == 'cuda' and not use_cuda:
        raise SystemExit('CUDA requested but not available')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = {
        'group': args.group, 'seed': args.seed, 'rounds': args.rounds,
        'pretrain': str(args.pretrain),
        'pretrain_sha256': sha(args.pretrain),
        'alpha': cfg['alpha'], 'lr': 0.002, 'schedule': cfg['schedule'],
        'aggregation': cfg['aggregation'],
        'multi_ca': {
            'target_components': 2, 'max_components': 2, 'min_components': 2,
            'capacity': 'same as author global K (no component-count increase)',
            'distance': 'diagonal Gaussian W2 squared',
            'merge_guard_normalized_bures': 1.0,
            'allow_early_stop': False,
            'decoder_mapping': 'dominant owner client per macro component',
        },
        'sample_M': 100, 'batch_size': 200, 'clients': 10,
        'partition_seed': 42, 'gmm_seed': 42,
        'device': args.device,
        'data_sha256': sha(SOURCE / 'dataset/breast_cancer.csv'),
        'partition_sizes': sizes.tolist(),
        'source_hashes': {q.name: sha(q) for q in SOURCE.glob('*.py')},
        'executed_hashes': {q.name: sha(q) for q in copied.glob('*.py')},
        'command': [sys.executable, *sys.argv],
        'test_status': 'NO_INDEPENDENT_BC_TEST_SPLIT; author-protocol in-sample metrics only',
    }
    atomic_json(out / 'config.json', config)
    atomic_json(out / 'status.json', {'status': 'RUNNING', 'group': args.group,
                                       'seed': args.seed, 'round': 0,
                                       'rounds': args.rounds})

    os.chdir(copied)
    sys.path.insert(0, str(copied))
    sys.dont_write_bytecode = True
    import model as author_model
    import experiment as author_experiment

    def _load_pretrain(server, clients):
        ckpt = torch.load(str(args.pretrain), map_location='cpu')
        server['vade'].load_state_dict(ckpt['server'], strict=True)
        if len(clients) != len(ckpt['clients']):
            raise RuntimeError('pretrain client count mismatch')
        for c, state in zip(clients, ckpt['clients']):
            vade = c['vade']
            device = next(vade.parameters()).device
            # Local K can be 1 or 2; the author assigns local GMM parameters
            # with K_l rows during pretrain, so resize before loading.
            for key in ('rho_', 'mu_c', 'log_sigma2_c'):
                setattr(vade, key, torch.nn.Parameter(state[key].to(device).clone()))
            non_gmm = {k: v for k, v in state.items()
                       if k not in ('rho_', 'mu_c', 'log_sigma2_c')}
            vade.load_state_dict(non_gmm, strict=False)
        if 'sizes' in ckpt and list(ckpt['sizes']) != sizes.tolist():
            raise RuntimeError('pretrain partition sizes mismatch')
        random.setstate(ckpt['python_rng'])
        np.random.set_state(ckpt['numpy_rng'])
        torch.set_rng_state(ckpt['torch_rng'])
        if use_cuda and ckpt.get('torch_cuda_rng') is not None:
            torch.cuda.set_rng_state_all(ckpt['torch_cuda_rng'])

    def _sample_union(clients_gmm_param, weight, n_samples, round_no, client_sizes):
        if cfg['aggregation'] == 'original':
            z, y, lengths, _, _, _ = author_model.gmm_sampling(
                clients_gmm_param, weight, n_samples)
            return z, y, lengths, None
        w, mu, lv, owner, _ = reconstruct_client_components(clients_gmm_param, client_sizes)
        macro = multi_ca_merge(
            w, mu, lv, owner, target_components=2, max_components=2,
            min_components=2, merge_threshold=1.0, allow_early_stop=False)
        z, y, layout = sample_macro_for_author_loop(
            macro, int(n_samples), seed=args.seed * 100000 + int(round_no),
            n_clients=len(clients_gmm_param))
        info = summary_for_log(macro)
        info['macro_weights'] = [float(x) for x in macro['weights']]
        info['macro_means'] = macro['means'].tolist()
        info['macro_log_sigma2'] = macro['log_sigma2'].tolist()
        info['merge_events'] = macro['merge_events']
        info['layout'] = layout
        info['sample_seed'] = args.seed * 100000 + int(round_no)
        return z, y, layout['lengths'], info

    def _observe(epoch, server, clients, L_c, L_s, purity, ari, mc_info):
        model = server['vade']
        with torch.no_grad():
            z_eval_t, _ = model.encoder(torch.tensor(X_eval, dtype=torch.float32,
                                                      device=next(model.parameters()).device))
        z_eval = z_eval_t.cpu().numpy()
        metrics, nll = gmm_state_metrics(model, z_eval, y_eval)
        w_server = torch.softmax(model.rho_.detach().cpu(), 0).tolist()
        row = {
            'round': int(epoch), 'group': args.group, 'seed': args.seed,
            'aggregation': cfg['aggregation'], 'schedule': cfg['schedule'],
            'alpha': cfg['alpha'], 'client_loss': float(L_c), 'server_loss': float(L_s),
            'client_record_purity': float(purity), 'client_record_ari': float(ari),
            'unique_ari': metrics['ari'], 'unique_nmi': metrics['nmi'],
            'unique_purity': metrics['purity'], 'unique_acc': metrics['acc'],
            'server_gmm_nll': float(nll),
            'server_gmm_weights': [float(x) for x in w_server],
            'n_local_components': int(sum(len(c['vade'].rho_) for c in clients)),
            'n_macro_components': int(len(w_server) if mc_info is None
                                      else mc_info['n_macro_components']),
            'n_merges': int(0 if mc_info is None else mc_info['n_merges']),
            'n_unsafe_merges': int(0 if mc_info is None else mc_info['n_unsafe_merges']),
            'max_merge_distance': None if mc_info is None else mc_info['max_merge_distance'],
            'max_merge_normalized_bures': None if mc_info is None
                                          else mc_info['max_merge_normalized_bures'],
            'macro_weights': None if mc_info is None else mc_info['macro_weights'],
            'dominant_clients': None if mc_info is None else mc_info['dominant_clients'],
            'client_lr_end': [float(c['optimizer'].param_groups[0]['lr']) for c in clients],
            'server_lr': float(server['optimizer'].param_groups[0]['lr']),
            'elapsed_seconds': perf_counter() - started,
        }
        # At the fixed reporting rounds, also evaluate the repeated client records.
        if epoch in checkpoint_rounds:
            with torch.no_grad():
                zc_t, _ = model.encoder(torch.tensor(
                    client_X_eval, dtype=torch.float32,
                    device=next(model.parameters()).device))
            zc = zc_t.cpu().numpy()
            mclient, nll_client = gmm_state_metrics(model, zc, client_y_eval)
            row.update({'client_record_unique_ari': mclient['ari'],
                        'client_record_unique_nmi': mclient['nmi'],
                        'client_record_unique_purity': mclient['purity'],
                        'client_record_unique_acc': mclient['acc'],
                        'client_record_gmm_nll': float(nll_client)})
        rows.append(row)
        with (out / 'rounds.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        with (out / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        if epoch % args.save_every == 0 or epoch == args.rounds:
            save_payload(out / 'checkpoints' / f'round_{epoch:03d}_server.pt', {
                'round': epoch, 'server': state_cpu(model), 'group': args.group,
                'seed': args.seed, 'aggregation': cfg['aggregation'],
                'train_row': row,
            })
        if epoch in checkpoint_rounds:
            full = {
                'round': epoch, 'group': args.group, 'seed': args.seed,
                'aggregation': cfg['aggregation'], 'schedule': cfg['schedule'],
                'alpha': cfg['alpha'], 'server': state_cpu(model),
                'clients': [state_cpu(c['vade']) for c in clients],
                'server_optimizer': server['optimizer'].state_dict(),
                'client_optimizers': [c['optimizer'].state_dict() for c in clients],
                'server_scheduler': server['lr_s'].state_dict(),
                'client_schedulers': [c['lr_s'].state_dict() for c in clients],
                'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                'torch_rng': torch.get_rng_state(),
                'torch_cuda_rng': torch.cuda.get_rng_state_all() if use_cuda else None,
                'sizes': sizes.tolist(),
                'mc_info': mc_info,
            }
            save_payload(out / 'checkpoints' / f'round_{epoch:03d}_full.pt', full)
            if mc_info is not None:
                atomic_json(out / 'checkpoints' / f'round_{epoch:03d}_merge_tree.json', mc_info)
        atomic_json(out / 'progress.json', {
            'group': args.group, 'seed': args.seed, 'round': int(epoch),
            'rounds': args.rounds, 'unique_ari': metrics['ari'],
            'unique_nmi': metrics['nmi'], 'server_gmm_nll': float(nll),
            'n_local_components': row['n_local_components'],
            'n_macro_components': row['n_macro_components'],
            'n_merges': row['n_merges'],
            'updated_at': perf_counter(),
        })
        atomic_json(out / 'status.json', {
            'status': 'RUNNING', 'group': args.group, 'seed': args.seed,
            'round': int(epoch), 'rounds': args.rounds,
        })

    checkpoint_rounds = {100, 200, args.rounds}
    rows = []
    started = perf_counter()
    original_parse = argparse.ArgumentParser.parse_args

    def force_device(parser, *a, **kw):
        parsed = original_parse(parser, *a, **kw)
        parsed.cuda = bool(use_cuda)
        return parsed

    os.environ['MULTICA_PRETRAIN'] = str(args.pretrain)
    argv = [str(copied / 'main.py'), '--dataset', 'breast_cancer',
            '--epoch', str(args.rounds), '--lr', '0.002', '--alpha', str(cfg['alpha']),
            '--sample', '100', '--client_num', '10', '--batch_size', '200']
    result = {'status': 'running', 'config': config, 'argv': argv}
    atomic_json(out / 'result.json', result)
    try:
        with patch.object(sys, 'argv', argv), \
                patch.object(argparse.ArgumentParser, 'parse_args', force_device):
            exec(compile(code, str(copied / 'main.py'), 'exec'), {
                '__name__': '__main__', '__file__': str(copied / 'main.py'),
                '_load_pretrain': _load_pretrain,
                '_sample_union': _sample_union,
                '_observe': _observe,
            })
        if len(rows) != args.rounds:
            raise RuntimeError(f'Incomplete rounds {len(rows)}/{args.rounds}')
        result.update(status='ok', last=rows[-1], elapsed_seconds=perf_counter() - started)
    except BaseException as exc:
        result.update(status='failed', error=repr(exc),
                      elapsed_seconds=perf_counter() - started)
        raise
    finally:
        atomic_json(out / 'result.json', result)
        atomic_json(out / 'status.json', {
            'status': result.get('status'), 'group': args.group, 'seed': args.seed,
            'round': len(rows), 'rounds': args.rounds,
            'error': result.get('error'),
        })
        print(json.dumps({'status': result.get('status'), 'group': args.group,
                          'seed': args.seed, 'out': str(out),
                          'last': result.get('last')}, indent=2))


if __name__ == '__main__':
    main()
