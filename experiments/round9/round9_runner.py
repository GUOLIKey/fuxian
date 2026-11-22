"""Isolated Round 9 FedVaDE run; training is executed on the Linux server only."""
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
from time import perf_counter, time
from unittest.mock import patch

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temp = Path(str(path) + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Expected one source match: {old!r}, got {source.count(old)}')
    return source.replace(old, new)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--lr', type=float, required=True)
    p.add_argument('--alpha', type=float, required=True)
    p.add_argument('--schedule', choices=('author', 'epoch'), default='epoch')
    p.add_argument('--gamma', type=float, default=.85)
    p.add_argument('--stepsize', type=int, default=10)
    p.add_argument('--pretrain', type=int, default=50)
    p.add_argument('--gmm-seed', type=int, default=42)
    p.add_argument('--gmm-n-init', type=int, default=1)
    p.add_argument('--gmm-init', choices=('kmeans', 'k-means++', 'random'), default='kmeans')
    p.add_argument('--sample', type=int, default=100)
    p.add_argument('--rounds', type=int, required=True)
    p.add_argument('--save-every', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    if args.rounds < 1 or args.pretrain < 1 or args.gmm_n_init < 1:
        raise ValueError('Positive round, pretrain and GMM initialization counts required')
    src, out = args.source.resolve(), args.out.resolve()
    if out.exists():
        raise SystemExit(f'Existing run directory: {out}')
    out.mkdir(parents=True)
    copied = out / 'source'
    copied.mkdir()
    for path in src.glob('*.py'):
        shutil.copy2(path, copied / path.name)
    (copied / 'dataset').mkdir()
    shutil.copy2(src / 'dataset/breast_cancer.csv', copied / 'dataset/breast_cancer.csv')
    author_main = (copied / 'main.py').read_text()
    author_model = (copied / 'model.py').read_text()
    updated_main = author_main
    if args.schedule == 'epoch':
        updated_main = once(updated_main, "                    c['lr_s'].step()\n", '')
        point = "                clients_vae_param.append(c['vade'].state_dict())"
        updated_main = once(updated_main, point, "                c['lr_s'].step()\n" + point)
    updated_main = once(updated_main, 'pre_epoch=50)', f'pre_epoch={args.pretrain})')
    updated_main = once(updated_main, '        "adam优化器设置"',
                        '        _observe_initial(server, clients)\n\n        "adam优化器设置"')
    point = '            Result.append([L_c, L_s, purity, ari])'
    updated_main = once(updated_main, point,
                        point + '\n            _observe(epoch + 1, server, clients, DL, L_c, L_s, purity, ari)')
    gmm_old = "covariance_type='diag', random_state=42)"
    if author_model.count(gmm_old) != 2:
        raise ValueError('Expected two original local/global GMM constructors')
    gmm_new = ("covariance_type='diag', random_state=" + str(args.gmm_seed) +
               ', n_init=' + str(args.gmm_n_init) + ", init_params='" + args.gmm_init + "')")
    updated_model = author_model.replace(gmm_old, gmm_new)
    (out / 'source.diff').write_text(''.join(difflib.unified_diff(
        author_main.splitlines(True), updated_main.splitlines(True),
        'author/main.py', 'run/main.py')) + ''.join(difflib.unified_diff(
        author_model.splitlines(True), updated_model.splitlines(True),
        'author/model.py', 'run/model.py')))
    (copied / 'main.py').write_text(updated_main)
    (copied / 'model.py').write_text(updated_model)
    files = [q for q in copied.rglob('*') if q.is_file()]
    config = vars(args).copy()
    config.update(source=str(src), out=str(out), platform=platform.platform(),
                  device='cpu', source_sha256={str(q.relative_to(copied)): sha(src / q.relative_to(copied))
                                                 for q in files},
                  executed_sha256={str(q.relative_to(copied)): sha(q) for q in files},
                  command=[sys.executable, *sys.argv],
                  checkpoint_rule='server model every save_every; full state at final round only due 27GB available disk',
                  label_selection='exploratory only')
    for key in ('source', 'out'):
        config[key] = str(config[key])
    atomic_json(out / 'config.json', config)
    os.chdir(copied)
    sys.path.insert(0, str(copied))
    sys.dont_write_bytecode = True
    import torch
    from scipy.special import logsumexp
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config.update(torch=torch.__version__, numpy=np.__version__)
    atomic_json(out / 'config.json', config)
    parse_args = argparse.ArgumentParser.parse_args
    started = perf_counter()
    rows = []

    def force_cpu(parser, *a, **kw):
        result = parse_args(parser, *a, **kw)
        if hasattr(result, 'cuda'):
            result.cuda = False
        return result

    def initial_diagnostics(server, clients):
        records = []
        with torch.no_grad():
            for idx, client in enumerate(clients):
                model = client['vade']
                x = torch.cat([batch[0] for batch in client['dataloader']], 0)
                mu, logvar = model.encoder(x)
                reconstructed = model.decoder(mu)
                latent = mu.numpy()
                weight = torch.softmax(model.rho_, 0).numpy()
                mean = model.mu_c.detach().cpu().numpy()
                variance = np.exp(model.log_sigma2_c.detach().cpu().numpy())
                logprob = (-.5 * (np.log(2 * np.pi * variance)[None, :, :] +
                           (latent[:, None, :] - mean[None, :, :]) ** 2 / variance[None, :, :])).sum(-1)
                logprob += np.log(weight + 1e-12)[None, :]
                likelihood = logsumexp(logprob, axis=1)
                occupation = np.exp(logprob - likelihood[:, None]).mean(axis=0)
                k, d = mean.shape
                bic = -2 * likelihood.sum() + ((k - 1) + 2 * k * d) * np.log(len(latent))
                records.append({'client': idx, 'n': len(latent), 'k': k,
                                'reconstruction_mse': float(torch.mean((x - reconstructed) ** 2)),
                                'latent_mean': float(latent.mean()), 'latent_std': float(latent.std()),
                                'latent_logvar_mean': float(logvar.mean()),
                                'gmm_loglik_per_sample': float(likelihood.mean()),
                                'gmm_bic': float(bic),
                                'posterior_occupancy': occupation.tolist()})
        atomic_json(out / 'initialization.json', {'clients': records,
                     'reconstruction_mse_weighted': float(np.average(
                         [r['reconstruction_mse'] for r in records],
                         weights=[r['n'] for r in records])),
                     'gmm_loglik_weighted': float(np.average(
                         [r['gmm_loglik_per_sample'] for r in records],
                         weights=[r['n'] for r in records])),
                     'gmm_bic_total': float(sum(r['gmm_bic'] for r in records)),
                     'server_mixture_weights': torch.softmax(server['vade'].rho_.detach().cpu(), 0).numpy().tolist(),
                     'component_counts_determined_by_unique_client_labels': True})

    def snapshot(epoch, server, clients):
        model = {k: v.detach().cpu().clone() for k, v in server['vade'].state_dict().items()}
        torch.save({'round': epoch, 'server': model}, out / f'epoch_{epoch:03d}_server.pt')
        if epoch == args.rounds:
            torch.save({'round': epoch, 'seed': args.seed, 'server': model,
                        'clients': [{k: v.detach().cpu().clone() for k, v in c['vade'].state_dict().items()}
                                    for c in clients],
                        'server_optimizer': server['optimizer'].state_dict(),
                        'client_optimizers': [c['optimizer'].state_dict() for c in clients],
                        'server_scheduler': server['lr_s'].state_dict(),
                        'client_schedulers': [c['lr_s'].state_dict() for c in clients],
                        'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                        'torch_rng': torch.get_rng_state()}, out / 'last_checkpoint.pt')

    def observe(epoch, server, clients, loaders, lc, ls, purity, ari):
        row = {'round': epoch, 'client_loss': float(lc), 'server_loss': float(ls),
               'ari': float(ari), 'purity': float(purity),
               'client_lr': [c['optimizer'].param_groups[0]['lr'] for c in clients],
               'server_lr': server['optimizer'].param_groups[0]['lr'],
               'alpha': args.alpha, 'gmm_weights': torch.softmax(server['vade'].rho_.detach().cpu(), 0).tolist(),
               'elapsed_seconds': perf_counter() - started}
        rows.append(row)
        with (out / 'rounds.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        if epoch % args.save_every == 0 or epoch == args.rounds:
            snapshot(epoch, server, clients)
        if epoch == 1:
            config['partition'] = [{'size': len(loader.dataset),
                    'tensor_sha256': [hashlib.sha256(t.numpy().tobytes()).hexdigest()
                                      for t in loader.dataset.tensors]} for loader in loaders]
            atomic_json(out / 'config.json', config)
        status = out.parents[1] / 'progress.json'
        previous = json.loads(status.read_text()) if status.exists() else {}
        previous.update(epoch=epoch, epoch_total=args.rounds, latest_ari=float(ari),
                        latest_purity=float(purity), updated_at=time())
        atomic_json(status, previous)

    argv = [str(copied / 'main.py'), '--dataset', 'breast_cancer', '--epoch', str(args.rounds),
            '--lr', str(args.lr), '--alpha', str(args.alpha), '--sample', str(args.sample),
            '--gamma', str(args.gamma), '--stepsize', str(args.stepsize)]
    result = {'status': 'running', 'author_argv': argv, 'seed': args.seed}
    atomic_json(out / 'result.json', result)
    try:
        with threadpool_limits(limits=1), patch.object(sys, 'argv', argv), \
             patch.object(argparse.ArgumentParser, 'parse_args', force_cpu):
            exec(compile(updated_main, str(copied / 'main.py'), 'exec'),
                 {'__name__': '__main__', '__file__': str(copied / 'main.py'),
                  '_observe_initial': initial_diagnostics, '_observe': observe})
        if len(rows) != args.rounds:
            raise RuntimeError(f'Incomplete training: {len(rows)} / {args.rounds}')
        result.update(status='ok', last=rows[-1], saved_epochs=[r for r in range(1, args.rounds + 1)
                       if r % args.save_every == 0 or r == args.rounds])
    except BaseException as exc:
        result.update(status='failed', error=repr(exc))
        raise
    finally:
        result['elapsed_seconds'] = perf_counter() - started
        atomic_json(out / 'result.json', result)


if __name__ == '__main__':
    main()
