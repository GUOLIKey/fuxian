"""Round 7 search reproduction with full federated checkpoints.

Grid, seeds, partitions, pretraining, formal rounds, optimizer, scheduler and
evaluation protocol are identical to `round7_hyperparameter_search`.

Two forced deviations, both caused by artifacts missing on this server:
  * The original coarse C01 reused `round5_broadcast/formal_author_seed0/
    round_150_full.pt`; that file is absent here, so C01 is retrained for
    50+150 rounds under the identical config and protocol.
  * The original fine reuse for the author/epoch baselines used checkpoints
    that are absent or server-only here, so selected fine configs are trained
    for 50+300 rounds to obtain full client states.

Every run saves `epoch_XXX_full.pt` at every 25-round server checkpoint.
Unkept full checkpoints are verified and gzip-compressed after evaluation.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path('/root/autodl-tmp/fedvade_project')
HERE = ROOT / 'reproduction_runs/round7_f01_full_repro'
SOURCE = ROOT / 'fuxian-round2/FedVaDE_source'
PYTHON = ROOT / '.venv-cpu/bin/python'
RUNNER = HERE / 'scripts/round7_runner_full.py'
EVAL = HERE / 'scripts/evaluate_saved_round6.py'
COMPRESS = HERE / 'scripts/compress_full_checkpoints.py'
LOGS = HERE / 'logs'

MIN_FREE_BYTES = 3_000_000_000


def configs():
    values = [dict(id='C01', lr=.002, alpha=.01, schedule='author', sample=100),
              dict(id='C02', lr=.002, alpha=.01, schedule='epoch', sample=100)]
    for lr in (.0005, .001, .005):
        for schedule in ('author', 'epoch'):
            values.append(dict(id=f'C{len(values)+1:02d}', lr=lr, alpha=.01,
                               schedule=schedule, sample=100))
    for alpha in (.001, .005, .05):
        for schedule in ('author', 'epoch'):
            values.append(dict(id=f'C{len(values)+1:02d}', lr=.002,
                               alpha=alpha, schedule=schedule, sample=100))
    for schedule in ('author', 'epoch'):
        values.append(dict(id=f'C{len(values)+1:02d}', lr=.001, alpha=.005,
                           schedule=schedule, sample=100))
    assert len(values) == 16
    return values


def is_f01(config):
    return (float(config['lr']) == .002 and float(config['alpha']) == .05
            and config['schedule'] == 'epoch' and int(config['sample']) == 100)


def progress(stage, position, total, config, path, status):
    value = '|'.join(map(str, (stage, position, total, config, path, status))) + '\n'
    tmp = HERE / 'progress.tmp'
    tmp.write_text(value)
    tmp.replace(HERE / 'progress.txt')


def check_disk():
    free = shutil.disk_usage(str(HERE)).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f'Only {free/1e9:.1f} GB free; refusing to continue')


def command(args, log):
    with log.open('a') as stream:
        stream.write('\n$ ' + ' '.join(map(str, args)) + '\n')
        stream.flush()
        subprocess.run([str(x) for x in args], cwd=HERE, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def train(config, out, rounds, stage, position, total):
    if out.exists():
        result = out / 'result.json'
        if not result.exists():
            raise RuntimeError(f'Existing incomplete run {out}; inspect logs '
                               'and remove it before retrying')
        status = json.loads(result.read_text()).get('status')
        if status != 'ok':
            raise RuntimeError(f'Existing failed run {out} (status={status}); '
                               'inspect logs before retrying')
        progress(stage, position, total, config['id'], out, 'RUNNING')
        return
    progress(stage, position, total, config['id'], out, 'RUNNING')
    command([PYTHON, RUNNER, '--source', SOURCE, '--out', out,
             '--lr', config['lr'], '--alpha', config['alpha'],
             '--schedule', config['schedule'], '--sample', config['sample'],
             '--rounds', rounds, '--pretrain', 50, '--save-every', 25,
             '--seed', 0], LOGS / f'{stage}_{config["id"]}.log')


def evaluate(run):
    if (run / 'evaluation_summary.json').exists():
        return json.loads((run / 'evaluation_summary.json').read_text())
    command([PYTHON, EVAL, '--run-dir', run], run / 'evaluation.log')
    return json.loads((run / 'evaluation_summary.json').read_text())


def compress(run, keep):
    args = [PYTHON, COMPRESS, '--run-dir', run]
    for name in keep:
        args += ['--keep', name]
    command(args, run / 'compress.log')


def epoch_metrics(summary, epoch):
    for snap in summary['snapshots']:
        if snap['epoch'] == epoch:
            return snap
    return None


def record(config, stage, run, summary):
    ckpt = run / 'full_checkpoint_manifest.jsonl'
    files = {}
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                files[item['file']] = item['load_path']
    return {
        'stage': stage, 'config_id': config['id'],
        'lr': config['lr'], 'alpha': config['alpha'],
        'schedule': config['schedule'], 'sample': config['sample'],
        'rounds': 150 if stage == 'coarse' else 300,
        'seed': 0, 'run_dir': str(run),
        'epoch150': epoch_metrics(summary, 150),
        'epoch100': epoch_metrics(summary, 100),
        'epoch300': epoch_metrics(summary, 300),
        'last': summary['last'],
        'best_unlabeled_epoch': summary['best_unlabeled_loss']['epoch'],
        'best_exploratory_epoch': summary['best_exploratory_ari']['epoch'],
        'full_checkpoints': files,
        'epoch100_full': files.get('epoch_100_full.pt')
        or files.get('epoch_100_full.pt.gz'),
    }


def write_results(coarse, fine):
    rows = []
    for item in coarse + fine:
        r = item['record']
        last = r['last']
        rows.append({
            'stage': r['stage'], 'config_id': r['config_id'],
            'lr': r['lr'], 'alpha': r['alpha'], 'schedule': r['schedule'],
            'sample': r['sample'], 'rounds': r['rounds'], 'seed': r['seed'],
            'ari_last': last['ari_mean'], 'ari_sd_last': last['ari_std'],
            'purity_last': last['purity_mean'], 'nmi_last': last['nmi_mean'],
            'acc_last': last['acc_mean'],
            'ari_epoch100': (r['epoch100'] or {}).get('ari_mean'),
            'ari_epoch150': (r['epoch150'] or {}).get('ari_mean'),
            'ari_epoch300': (r['epoch300'] or {}).get('ari_mean'),
            'client_loss_last5': last['client_loss_last5_mean'],
            'best_unlabeled_epoch': r['best_unlabeled_epoch'],
            'best_exploratory_epoch': r['best_exploratory_epoch'],
            'epoch100_full': r['epoch100_full'],
            'run_dir': r['run_dir'],
        })
    out = HERE / 'results'
    out.mkdir(exist_ok=True)
    keys = list(rows[0].keys())
    with (out / 'search_results.csv').open('w') as f:
        f.write(','.join(keys) + '\n')
        for row in rows:
            f.write(','.join('' if row[k] is None else str(row[k])
                             for k in keys) + '\n')
    (out / 'search_results.json').write_text(json.dumps(rows, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    plan = configs()
    protocol = {
        'coarse': plan, 'coarse_epochs': 150, 'fine_epochs': 300,
        'fine_rule': ('Top 4 coarse configurations by fixed epoch-150 '
                      'author-protocol ARI mean; exploratory.'),
        'training_seed': 0, 'inference_seeds': list(range(10)),
        'checkpoint_rule': ('full federated state at every 25-round server '
                            'checkpoint; epoch_XXX_full.pt'),
        'forced_deviations': [
            'C01 retrained (round_150_full.pt absent on this server)',
            'selected fine configs retrained for full client states '
            '(original reuse artifacts absent or server-only)',
        ],
    }
    if args.plan_only:
        print(json.dumps(protocol, indent=2))
        return
    LOGS.mkdir(parents=True, exist_ok=True)
    (HERE / 'search_plan.json').write_text(json.dumps(protocol, indent=2) + '\n')
    try:
        coarse = []
        for position, config in enumerate(plan, 1):
            check_disk()
            out = HERE / 'coarse' / config['id']
            train(config, out, 150, 'coarse', position, len(plan))
            summary = evaluate(out)
            compress(out, keep=[])
            coarse.append({'config': config,
                           'record': record(config, 'coarse', out, summary)})
            (HERE / 'coarse_rankings.json').write_text(
                json.dumps(coarse, indent=2) + '\n')
        top = sorted(coarse, key=lambda x: x['record']['last']['ari_mean'],
                     reverse=True)[:4]
        fine_plan = [dict(item['config']) for item in top]
        (HERE / 'fine_plan.json').write_text(json.dumps(fine_plan, indent=2) + '\n')
        fine = []
        for position, config in enumerate(fine_plan, 1):
            check_disk()
            out = HERE / 'fine' / config['id']
            if is_f01(config):
                priority = HERE / 'fine' / 'F01'
                if not (priority / 'result.json').exists():
                    raise RuntimeError('Priority F01 run is missing; run it first')
                out = priority
            train(config, out, 300, 'fine', position, len(fine_plan))
            summary = evaluate(out)
            keep = ['epoch_100_full.pt']
            if is_f01(config):
                keep.append('epoch_300_full.pt')
            compress(out, keep=keep)
            fine.append({'rank': position, 'config': config,
                         'record': record(config, 'fine', out, summary)})
            (HERE / 'fine_rankings.json').write_text(
                json.dumps(fine, indent=2) + '\n')
        write_results(coarse, fine)
        progress('fine', len(fine_plan), len(fine_plan),
                 fine[-1]['config']['id'], fine[-1]['record']['run_dir'], 'FINISHED')
    except BaseException:
        (HERE / 'error.txt').write_text(traceback.format_exc())
        if (HERE / 'progress.txt').exists():
            values = (HERE / 'progress.txt').read_text().strip().split('|')
            progress(*values[:5], 'FAILED')
        raise


if __name__ == '__main__':
    main()
