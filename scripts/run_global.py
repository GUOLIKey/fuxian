"""Frozen-client FedVaDE server-only intervention for five local-GMM builders.

All five groups start from the same F01 epoch-100 full checkpoint and run the
original server update for 10 rounds.  Client encoders/decoders and local GMMs
are frozen; client training and broadcast are skipped.  The only intervention
is the local GMM parameterization.

Global ARI/NMI/Purity use the historical F01 author-protocol evaluator and
10 inference seeds per round.  Generation uses the global server GMM and the
server decoder; server training uses client local GMMs and client decoders.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial.distance import pdist
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

ROOT = Path('/root/autodl-tmp/fedvade_project')
SOURCE = ROOT / 'fuxian-round2/FedVaDE_source'
CKPT = (ROOT / 'reproduction_runs/round7_f01_full_repro/fine/F01'
        / 'epoch_100_full.pt')
ACD_UNITS = ROOT / 'experiments/acd_f01_frozen_v1/results/units'
EXP = ROOT / 'experiments/acd_f01_global_v1'
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(ROOT / 'reproduction_runs/round7_f01_full_repro/scripts'))
from dataloader import make_noniid_clients  # noqa: E402
from model import VaDE, gmm_sampling  # noqa: E402
from evaluate_saved_round6 import evaluate_one  # noqa: E402

EXPECTED_SHA = '3be341c451a186d1f36673080164af208993d215f0c2cb03709407892f7d6f3e'
GROUPS = ('A', 'B', 'C', 'D', 'E')
GROUP_METHOD = {'A': 'A0', 'B': 'A1', 'C': 'A2', 'D': 'A4', 'E': 'A5'}
SEEDS = (0, 1, 2)
ROUNDS = 10
INF_SEEDS = tuple(range(10))
GEN_SEEDS = (0, 1, 2)
N_GEN = 5000
SAMPLE = 100
REG_COVAR = 1e-6
PI_EPS = 1e-10


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, obj):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=_default) + '\n')
    tmp.replace(path)


def append_jsonl(path, obj):
    with Path(path).open('a') as f:
        f.write(json.dumps(obj, default=_default) + '\n')


def _default(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


def status(stage, message, done=False, **extra):
    payload = {'stage': stage, 'message': message, 'done': bool(done),
               'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    payload.update(extra)
    atomic_json(EXP / 'status.json', payload)


def load_context():
    digest = sha256(CKPT)
    if digest != EXPECTED_SHA:
        raise RuntimeError('Checkpoint SHA mismatch')
    ckpt = torch.load(CKPT, map_location='cpu')
    if int(ckpt['round']) != 100 or len(ckpt['clients']) != 10:
        raise RuntimeError('Bad checkpoint structure')
    clients = []
    for state in ckpt['clients']:
        m = VaDE(SimpleNamespace(input_dim=31, hid_dim=10,
                                 nClusters=int(len(state['rho_'])),
                                 alpha=0.05, cuda=False))
        m.load_state_dict(state, strict=True)
        m.eval()
        clients.append(m)
    _, loaders, (Xt, yt), sizes = make_noniid_clients(
        str(SOURCE / 'dataset/breast_cancer.csv'), 10, 200,
        'breast_cancer', seed=42)
    Xu = Xt.numpy().astype(np.float64)
    yu = yt.numpy().astype(int)
    prior = {}
    for p in sorted(ACD_UNITS.glob('client*_seed*.json')):
        u = json.loads(p.read_text())
        prior[(int(u['client']), int(u['seed']))] = u
    if len(prior) != 30:
        raise RuntimeError(f'Expected 30 prior units, got {len(prior)}')
    return {'ckpt': ckpt, 'clients': clients, 'loaders': loaders,
            'Xu': Xu, 'yu': yu, 'sizes': [int(s) for s in sizes],
            'prior': prior, 'sha256': digest}


def make_server(ctx):
    state = copy.deepcopy(ctx['ckpt']['server'])
    server = VaDE(SimpleNamespace(input_dim=31, hid_dim=10, nClusters=2,
                                  alpha=0.05, cuda=False))
    server.load_state_dict(state, strict=True)
    server.train()
    opt = Adam(server.parameters(), lr=0.002)
    opt.load_state_dict(copy.deepcopy(ctx['ckpt']['server_optimizer']))
    sch = StepLR(opt, step_size=10, gamma=0.85)
    sch.load_state_dict(copy.deepcopy(ctx['ckpt']['server_scheduler']))
    return server, opt, sch


def prior_full(ctx, client, seed, group):
    method = GROUP_METHOD[group]
    u = ctx['prior'][(client, seed)]
    rec = u['methods'][method]['full']
    p = rec['params']
    return {
        'rho': np.asarray(p['rho'], dtype=np.float64),
        'mu': np.asarray(p['mu'], dtype=np.float64),
        'logvar': np.asarray(p['logvar'], dtype=np.float64),
        'pi': np.asarray(p['pi'], dtype=np.float64),
        'K': int(p['K']),
        'support': np.asarray(rec['support'], dtype=np.int64),
    }


def original_full(ctx, client):
    state = ctx['ckpt']['clients'][client]
    rho = state['rho_'].detach().float().numpy()
    pi = np.exp(rho - rho.max()); pi /= pi.sum()
    mu = state['mu_c'].detach().float().numpy()
    var = np.exp(state['log_sigma2_c'].detach().float().numpy())
    var = np.maximum(var, REG_COVAR)
    u = ctx['prior'][(client, 0)]
    support = np.asarray(u['methods']['A0']['full']['support'], dtype=np.int64)
    return {'rho': rho, 'mu': mu,
            'logvar': np.log(var), 'pi': pi, 'K': int(len(pi)),
            'support': support}


def group_gmms(ctx, group, seed):
    out = []
    for client in range(len(ctx['clients'])):
        if group == 'A':
            p = original_full(ctx, client)
        else:
            p = prior_full(ctx, client, seed, group)
        out.append(p)
    if group in ('D', 'E'):
        for client in range(len(ctx['clients'])):
            kd = out[client]['K']
            # The prior A5 was matched to A4 per client/seed.
            if GROUP_METHOD[group] == 'A5' and kd != prior_full(
                    ctx, client, seed, 'D')['K']:
                raise RuntimeError('D/E K mismatch')
    return out


def gmm_states(params):
    return [{'rho_': torch.tensor(p['rho'], dtype=torch.float32),
             'mu_c': torch.tensor(p['mu'], dtype=torch.float32),
             'log_sigma2_c': torch.tensor(p['logvar'], dtype=torch.float32)}
            for p in params]


def diag_log_prob(z, rho, mu, logvar):
    z = np.asarray(z, dtype=np.float64)
    w = torch.softmax(rho.detach().float(), 0).numpy().astype(np.float64)
    mu = mu.detach().float().numpy().astype(np.float64)
    lv = logvar.detach().float().numpy().astype(np.float64)
    diff = z[:, None, :] - mu[None, :, :]
    lp = -0.5 * np.sum(np.log(2 * np.pi) + lv[None, :, :] +
                       diff ** 2 / np.exp(lv)[None, :, :], axis=2)
    return lp + np.log(np.maximum(w, 1e-12))[None, :]


def diag_nll(z, rho, mu, logvar):
    lp = diag_log_prob(z, rho, mu, logvar)
    m = lp.max(1, keepdims=True)
    return float(-np.mean(m.ravel() + np.log(np.exp(lp - m).sum(1))))


def sample_gmm(rho, mu, logvar, n, seed):
    rng = np.random.default_rng(int(seed))
    w = torch.softmax(rho.detach().float(), 0).numpy().astype(np.float64)
    w = w / w.sum()
    m = mu.detach().float().numpy().astype(np.float64)
    lv = logvar.detach().float().numpy().astype(np.float64)
    comp = rng.choice(len(w), size=int(n), p=w)
    z = m[comp] + np.sqrt(np.exp(lv[comp])) * \
        rng.standard_normal(m[comp].shape)
    return z, comp


def mmd2_rbf(a, b, sigma):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    a2 = np.sum(a * a, 1); b2 = np.sum(b * b, 1)
    kxx = np.exp(-np.maximum(a2[:, None] + a2[None, :] - 2 * a @ a.T, 0) /
                 (2 * sigma * sigma))
    kyy = np.exp(-np.maximum(b2[:, None] + b2[None, :] - 2 * b @ b.T, 0) /
                 (2 * sigma * sigma))
    kxy = np.exp(-np.maximum(a2[:, None] + b2[None, :] - 2 * a @ b.T, 0) /
                 (2 * sigma * sigma))
    return float(max(kxx.mean() + kyy.mean() - 2 * kxy.mean(), 0.0))


def swd(a, b, directions):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    q = np.linspace(0, 1, 200)
    return float(np.mean(np.abs(np.quantile(a @ directions, q, axis=0) -
                                np.quantile(b @ directions, q, axis=0))))


def initial_check(ctx):
    server, _, _ = make_server(ctx)
    server.eval()
    draws = [evaluate_one(server, ctx['loaders'], s) for s in INF_SEEDS]
    ari = np.array([d['ari'] for d in draws])
    pur = np.array([d['purity'] for d in draws])
    nmi = np.array([d['nmi'] for d in draws])
    target = 0.605326980742813
    report = {'checkpoint_sha256': ctx['sha256'], 'target': target,
              'per_seed_ari': ari.tolist(), 'ari_mean': float(ari.mean()),
              'ari_sd_inference': float(ari.std(ddof=1)),
              'purity_mean': float(pur.mean()), 'nmi_mean': float(nmi.mean()),
              'exact_1e-12': bool(abs(float(ari.mean()) - target) < 1e-12)}
    atomic_json(EXP / 'results/initial_check.json', report)
    (EXP / 'audit/INITIAL_CHECK.md').write_text(
        '# Initial F01 @100 Global ARI reload check\n\n'
        f"- Checkpoint SHA256: `{ctx['sha256']}`\n"
        f"- 10 inference seeds: {[round(float(x), 6) for x in ari]}\n"
        f"- Mean ARI: **{ari.mean():.12f}** (target {target:.12f})\n"
        f"- Inference SD: {ari.std(ddof=1):.6f}\n"
        f"- Exact at 1e-12: **{report['exact_1e-12']}**\n")
    if not report['exact_1e-12']:
        raise RuntimeError('Initial Global ARI does not reproduce 0.6053')
    return report


def evaluate_global(ctx, server):
    server.eval()
    draws = [evaluate_one(server, ctx['loaders'], s) for s in INF_SEEDS]
    server.train()
    ari = np.array([d['ari'] for d in draws])
    pur = np.array([d['purity'] for d in draws])
    nmi = np.array([d['nmi'] for d in draws])
    return {'ari': float(ari.mean()), 'ari_sd_inference': float(ari.std(ddof=1)),
            'purity': float(pur.mean()), 'nmi': float(nmi.mean()),
            'per_seed_ari': ari.tolist()}


def generation_global(ctx, server):
    with torch.no_grad():
        z = server.encoder(torch.tensor(ctx['Xu'], dtype=torch.float32))[0].numpy()
    nll = diag_nll(z, server.rho_, server.mu_c, server.log_sigma2_c)
    Xu = ctx['Xu']; yu = ctx['yu']
    sigma = float(np.sqrt(np.median(pdist(Xu, metric='sqeuclidean'))))
    rng = np.random.default_rng(20261124)
    dirs = rng.normal(size=(Xu.shape[1], 64))
    dirs /= np.linalg.norm(dirs, axis=0, keepdims=True)
    resp = diag_log_prob(z, server.rho_, server.mu_c, server.log_sigma2_c)
    hard = resp.argmax(1)
    w = torch.softmax(server.rho_.detach().float(), 0).numpy().astype(np.float64)
    K = len(w)
    majority = []
    for k in range(K):
        idx = np.flatnonzero(hard == k)
        majority.append(int(np.bincount(yu[idx]).argmax()) if len(idx) else None)
    classes, counts = np.unique(yu, return_counts=True)
    minority = int(classes[np.argmin(counts)]) if len(classes) > 1 else None
    minority_comps = ([k for k in range(K) if majority[k] == minority]
                      if minority is not None else [])
    mmds, swds, fracs, min_mmds, min_swds = [], [], [], [], []
    real_min = Xu[yu == minority] if minority is not None else None
    for seed in GEN_SEEDS:
        zg, comp = sample_gmm(server.rho_, server.mu_c,
                              server.log_sigma2_c, N_GEN, seed)
        with torch.no_grad():
            xg = server.decoder(torch.tensor(zg, dtype=torch.float32)).numpy()
        mmds.append(math.sqrt(max(mmd2_rbf(xg, Xu, sigma), 0)))
        swds.append(swd(xg, Xu, dirs))
        if minority is not None:
            sel = np.isin(comp, np.asarray(minority_comps, dtype=int))
            fracs.append(float(sel.mean()))
            if len(real_min) >= 20 and int(sel.sum()) >= 20:
                min_mmds.append(math.sqrt(max(
                    mmd2_rbf(xg[sel], real_min, sigma), 0)))
                min_swds.append(swd(xg[sel], real_min, dirs))
    return {
        'global_nll': float(nll),
        'gen_mmd': float(np.mean(mmds)), 'gen_mmd_sd': float(np.std(mmds, ddof=1)),
        'gen_swd': float(np.mean(swds)), 'gen_swd_sd': float(np.std(swds, ddof=1)),
        'minority_label': minority,
        'minority_mass': float(w[minority_comps].sum()) if minority is not None else None,
        'minority_gen_frac': float(np.mean(fracs)) if fracs else None,
        'minority_mmd': float(np.mean(min_mmds)) if min_mmds else None,
        'minority_swd': float(np.mean(min_swds)) if min_swds else None,
        'gen_finite': bool(np.isfinite(xg).all()),
    }


def coverage_row(y_hat, lengths, params):
    y_hat = np.asarray(y_hat, dtype=int)
    lengths = [int(x) for x in lengths]
    total = sum(lengths)
    component_client = np.repeat(np.arange(len(lengths)), lengths)
    sample_client = component_client[y_hat]
    rows = []
    start = 0
    low_total = low_sampled = unique_total = 0
    for cid, n in enumerate(lengths):
        support = params[cid]['support']
        local = y_hat[sample_client == cid] - start
        uniq = np.unique(local)
        cnt = np.bincount(local, minlength=n)
        low = support < 5
        low_total += int(low.sum())
        low_sampled += int(sum(1 for j in uniq if j < len(low) and low[j]))
        unique_total += int(len(uniq))
        rows.append({'client': cid, 'components': n,
                     'sampled_components_unique': int(len(uniq)),
                     'sampled_counts': cnt.tolist(),
                     'low_support_total': int(low.sum()),
                     'low_support_sampled': int(sum(
                         1 for j in uniq if j < len(low) and low[j]))})
        start += n
    return {'mean_unique_sampled': (unique_total / len(lengths)
                                    if lengths else 0.0),
            'unique_sampled_total': unique_total,
            'low_support_total': low_total,
            'low_support_sampled': low_sampled,
            'clients': rows}


def server_round(server, opt, sch, clients, gmm_states, weight, pair_seed):
    np.random.seed(int(pair_seed))
    torch.manual_seed(int(pair_seed))
    Z, y_hat, lengths, _, _, _ = gmm_sampling(gmm_states, weight, SAMPLE)
    if len(y_hat) != SAMPLE:
        raise RuntimeError(f'sampling budget mismatch: {len(y_hat)} != {SAMPLE}')
    Z = torch.from_numpy(Z).float()
    X_hat = server.decoder(Z)
    decoded = torch.zeros_like(X_hat)
    start = 0
    for cid in range(len(clients)):
        for k in range(int(lengths[cid])):
            label = start + k
            idx = np.where(y_hat == label)[0]
            if len(idx):
                decoded[idx] = clients[cid].decoder(Z[idx])
        start += int(lengths[cid])
    loss = server.ELBO_Loss_server(X_hat, decoded)
    opt.zero_grad()
    loss.backward()
    opt.step()
    sch.step()
    return float(loss.item()), y_hat, lengths


def run_one(ctx, group, seed, rounds=ROUNDS, out_dir=None, save_final=True,
            log_global=True):
    out_dir = out_dir or (EXP / 'results/runs')
    out_dir.mkdir(parents=True, exist_ok=True)
    final_dir = EXP / 'results/final_states'
    gmm_dir = EXP / 'results/group_gmms'
    final_dir.mkdir(parents=True, exist_ok=True)
    gmm_dir.mkdir(parents=True, exist_ok=True)
    server, opt, sch = make_server(ctx)
    params = group_gmms(ctx, group, seed)
    gstates = gmm_states(params)
    weight = torch.tensor(np.asarray(ctx['sizes'], dtype=np.float64) /
                          np.sum(ctx['sizes']), dtype=torch.float32)
    rounds_rec = []
    initial = evaluate_global(ctx, server)
    initial.update(generation_global(ctx, server))
    initial.update({'round': 0, 'server_loss': None,
                    'server_lr': float(opt.param_groups[0]['lr'])})
    rounds_rec.append(initial)
    for r in range(1, int(rounds) + 1):
        status('train', f'{group} seed {seed} round {r}', group=group,
               seed=seed, round=r, rounds=int(rounds))
        loss, y_hat, lengths = server_round(
            server, opt, sch, ctx['clients'], gstates, weight,
            seed * 100000 + r)
        ev = evaluate_global(ctx, server)
        gen = generation_global(ctx, server)
        cov = coverage_row(y_hat, lengths, params)
        rec = {'round': r, 'server_loss': float(loss),
               'server_lr': float(opt.param_groups[0]['lr']), **ev, **gen,
               'coverage': cov}
        rounds_rec.append(rec)
        if log_global:
            append_jsonl(EXP / 'results/rounds.jsonl',
                         {'group': group, 'seed': seed, **rec})
            append_jsonl(EXP / 'results/coverage.jsonl',
                         {'group': group, 'seed': seed, 'round': r, **cov})
        atomic_json(out_dir / f'{group}_seed{seed}.json', {
            'group': group, 'seed': seed, 'status': 'PARTIAL',
            'local_method': GROUP_METHOD[group], 'rounds': rounds_rec,
            'config': {'rounds': int(rounds), 'sample': SAMPLE,
                       'client_weights': (np.asarray(ctx['sizes']) /
                                          np.sum(ctx['sizes'])).tolist(),
                       'local_K': [p['K'] for p in params],
                       'restored_server_optimizer': True,
                       'restored_server_scheduler': True,
                       'client_frozen': True}})
        status('train', f'{group} seed {seed} round {r} done', group=group,
               seed=seed, round=r, rounds=int(rounds),
               latest={'ari': ev['ari'], 'nmi': ev['nmi'],
                       'purity': ev['purity'], 'loss': loss})
    high = max(rounds_rec, key=lambda x: x['ari'])
    run = {'group': group, 'seed': seed, 'status': 'DONE',
           'local_method': GROUP_METHOD[group], 'rounds': rounds_rec,
           'final': rounds_rec[-1],
           'highest_exploratory': {'round': high['round'], 'ari': high['ari']},
           'config': {'rounds': int(rounds), 'sample': SAMPLE,
                      'client_weights': (np.asarray(ctx['sizes']) /
                                         np.sum(ctx['sizes'])).tolist(),
                      'local_K': [p['K'] for p in params],
                      'restored_server_optimizer': True,
                      'restored_server_scheduler': True,
                      'client_frozen': True,
                      'primary_endpoint': int(rounds)}}
    atomic_json(out_dir / f'{group}_seed{seed}.json', run)
    atomic_json(gmm_dir / f'{group}_seed{seed}.json', {
        'group': group, 'seed': seed,
        'clients': [{'client': c, 'K': p['K'], 'pi': p['pi'].tolist(),
                     'mu': p['mu'].tolist(), 'logvar': p['logvar'].tolist()}
                    for c, p in enumerate(params)]})
    if save_final:
        torch.save({'group': group, 'seed': seed, 'round': int(rounds),
                    'server': {k: v.detach().cpu().clone()
                               for k, v in server.state_dict().items()},
                    'server_optimizer': opt.state_dict(),
                    'server_scheduler': sch.state_dict(),
                    'python_rng': random.getstate(),
                    'numpy_rng': np.random.get_state(),
                    'torch_rng': torch.get_rng_state(),
                    'config': run['config']},
                   final_dir / f'{group}_seed{seed}.pt')
    return run


def smoke(ctx):
    initial = initial_check(ctx)
    out = EXP / 'results/smoke_runs'
    reports = []
    passed = True
    for group in GROUPS:
        run = run_one(ctx, group, 0, rounds=1, out_dir=out,
                      save_final=False, log_global=False)
        rec = run['rounds'][-1]
        ok = (np.isfinite(rec['server_loss']) and np.isfinite(rec['ari'])
              and np.isfinite(rec['gen_mmd']) and rec['gen_finite'])
        reports.append({'group': group, 'K': len(run['config']['local_K']),
                        'local_K': run['config']['local_K'],
                        'round1_ari': rec['ari'],
                        'server_loss': rec['server_loss'],
                        'gen_finite': rec['gen_finite'], 'ok': bool(ok)})
        passed = passed and ok
    # D/E K equality
    for c in range(10):
        kd = ctx['prior'][(c, 0)]['methods']['A4']['full']['K']
        ke = ctx['prior'][(c, 0)]['methods']['A5']['full']['K']
        passed = passed and (kd == ke)
    report = {'passed': bool(passed), 'initial': initial,
              'groups': reports}
    atomic_json(EXP / 'results/smoke.json', report)
    lines = ['# Smoke test', '',
             f"- Initial Global ARI: **{initial['ari_mean']:.12f}**",
             f"- Exact 1e-12: **{initial['exact_1e-12']}**",
             f"- Passed: **{passed}**", '',
             '| Group | Local K | Round-1 ARI | Server Loss | Gen finite | OK |',
             '|---|---|---|---|---|']
    for r in reports:
        lines.append(f"| {r['group']} | {r['local_K']} | {r['round1_ari']:.6f} | "
                     f"{r['server_loss']:.6f} | {r['gen_finite']} | {r['ok']} |")
    (EXP / 'audit/SMOKE_TEST.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(report, indent=2, default=_default))
    return report


def full(ctx):
    smoke_path = EXP / 'results/smoke.json'
    if not smoke_path.exists() or not json.loads(
            smoke_path.read_text()).get('passed'):
        raise SystemExit('Smoke not passed; run --smoke first')
    initial_check(ctx)
    total = len(GROUPS) * len(SEEDS)
    done = 0
    for group in GROUPS:
        for seed in SEEDS:
            run_path = EXP / 'results/runs' / f'{group}_seed{seed}.json'
            if run_path.exists():
                try:
                    if json.loads(run_path.read_text()).get('status') == 'DONE':
                        done += 1
                        status('run', f'{group} seed {seed} exists', group=group,
                               seed=seed, completed=done, total=total)
                        continue
                except Exception:
                    pass
            run_one(ctx, group, seed)
            done += 1
            status('run', f'{group} seed {seed} done', group=group, seed=seed,
                   completed=done, total=total)
    status('done', 'Frozen-client global experiment complete', done=True,
           completed=done, total=total)
    print(json.dumps({'done': done, 'total': total}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    EXP.joinpath('results').mkdir(parents=True, exist_ok=True)
    EXP.joinpath('results/runs').mkdir(parents=True, exist_ok=True)
    EXP.joinpath('audit').mkdir(parents=True, exist_ok=True)
    ctx = load_context()
    if args.smoke:
        smoke(ctx)
    else:
        full(ctx)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        (EXP / 'error.txt').write_text(traceback.format_exc())
        status('error', 'Run failed; see error.txt')
        raise
