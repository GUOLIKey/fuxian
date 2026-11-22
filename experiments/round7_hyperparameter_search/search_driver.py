"""Bounded staged search. All ARI-based choices are exploratory, not test claims."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path('/root/autodl-tmp/fedvade_project')
HERE = ROOT / 'reproduction_runs/round7_hyperparameter_search'
SOURCE = ROOT / 'fuxian-round2/FedVaDE_source'
PYTHON = ROOT / '.venv-cpu/bin/python'


def configs():
    values = [dict(id='C01', lr=.002, alpha=.01, schedule='author', sample=100, reused=True),
              dict(id='C02', lr=.002, alpha=.01, schedule='epoch', sample=100)]
    for lr in (.0005, .001, .005):
        for schedule in ('author', 'epoch'):
            values.append(dict(id=f'C{len(values)+1:02d}', lr=lr, alpha=.01,
                               schedule=schedule, sample=100))
    for alpha in (.001, .005, .05):
        for schedule in ('author', 'epoch'):
            values.append(dict(id=f'C{len(values)+1:02d}', lr=.002, alpha=alpha,
                               schedule=schedule, sample=100))
    for schedule in ('author', 'epoch'):
        values.append(dict(id=f'C{len(values)+1:02d}', lr=.001, alpha=.005,
                           schedule=schedule, sample=100))
    assert len(values) == 16
    return values


def progress(stage, position, total, config, path, status):
    # Pipe fields contain no user data; Bash monitor parses without Python.
    value = '|'.join(map(str, (stage, position, total, config, path, status))) + '\n'
    tmp = HERE / 'progress.tmp'
    tmp.write_text(value)
    tmp.replace(HERE / 'progress.txt')


def command(args, log):
    with log.open('w') as stream:
        subprocess.run([str(x) for x in args], cwd=HERE, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def train(config, out, rounds, stage, position, total):
    if out.exists():
        raise RuntimeError(f'Will not overwrite {out}')
    progress(stage, position, total, config['id'], out, 'RUNNING')
    command([PYTHON, HERE / 'round7_runner.py', '--source', SOURCE, '--out', out,
             '--lr', config['lr'], '--alpha', config['alpha'], '--schedule', config['schedule'],
             '--sample', config['sample'], '--rounds', rounds, '--pretrain', 50,
             '--save-every', 25, '--seed', 0], HERE / f"{stage}_{config['id']}.log")


def evaluate(run, external=None, external_rounds=None):
    args = [PYTHON, HERE / 'evaluate_saved.py', '--run-dir', run]
    if external:
        args += ['--external-checkpoint', external, '--external-rounds', external_rounds]
    command(args, run / 'evaluation.log')
    return json.loads((run / 'evaluation_summary.json').read_text())


def reuse(out, checkpoint, rounds, config, stage, position, total):
    if out.exists():
        raise RuntimeError(f'Will not overwrite {out}')
    out.mkdir(parents=True)
    (out / 'reused_config.json').write_text(json.dumps(config, indent=2) + '\n')
    progress(stage, position, total, config['id'], out, 'RUNNING')
    return evaluate(out, checkpoint, rounds)


def known_full(config):
    if config['lr'] != .002 or config['alpha'] != .01 or config['sample'] != 100:
        return None
    if config['schedule'] == 'author':
        run = ROOT / 'reproduction_runs/round5_broadcast/formal_author_seed0'
        return run / 'round_300_full.pt', run / 'rounds.jsonl'
    if config['schedule'] == 'epoch':
        run = ROOT / 'fuxian-round2/formal/B_epoch_seed0'
        return run / 'round_300_server.pt', run / 'rounds.jsonl'
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    plan = configs()
    protocol = {'coarse': plan, 'coarse_epochs': 150, 'fine_epochs': 300,
                'fine_rule': 'Top 4 coarse configurations by fixed epoch-150 author-protocol ARI mean; all exploratory because evaluation labels select configs. Sample count remains 100.',
                'checkpoint_rule': 'Within each run, minimum trailing-five client training loss among saved epochs; no evaluation labels.',
                'exploratory_checkpoint_rule': 'Maximum fixed-checkpoint inference ARI, labeled exploratory, never independent test.',
                'inference_seeds': list(range(10)), 'training_seed': 0,
                'evaluation_reference': str(ROOT / 'reproduction_runs/round6_checkpoint_diagnosis/evaluation_protocol.md'),
                'author_source': str(SOURCE),
                'no_independent_validation': True}
    if args.plan_only:
        print(json.dumps(protocol, indent=2))
        return
    HERE.mkdir(parents=True, exist_ok=True)
    if (HERE / 'progress.txt').exists():
        raise SystemExit('Search state already exists; refusing duplicate launch')
    (HERE / 'search_plan.json').write_text(json.dumps(protocol, indent=2) + '\n')
    try:
        coarse = []
        for position, config in enumerate(plan, 1):
            out = HERE / 'coarse' / config['id']
            if config.get('reused'):
                ref = ROOT / 'reproduction_runs/round5_broadcast/formal_author_seed0'
                summary = reuse(out, ref / 'round_150_full.pt', ref / 'rounds.jsonl',
                                config, 'coarse', position, len(plan))
            else:
                train(config, out, 150, 'coarse', position, len(plan))
                summary = evaluate(out)
            coarse.append({'config': config, 'summary': summary})
            (HERE / 'coarse_rankings.json').write_text(json.dumps(coarse, indent=2) + '\n')
        top = sorted(coarse, key=lambda x: x['summary']['last']['ari_mean'], reverse=True)[:4]
        fine = [dict(item['config']) for item in top]
        assert len(fine) == 4
        (HERE / 'fine_plan.json').write_text(json.dumps(fine, indent=2) + '\n')
        fine_results = []
        for position, config in enumerate(fine, 1):
            config['id'] = f'F{position:02d}'
            out = HERE / 'fine' / config['id']
            known = known_full(config)
            if known:
                summary = reuse(out, *known, config, 'fine', position, len(fine))
            else:
                train(config, out, 300, 'fine', position, len(fine))
                summary = evaluate(out)
            fine_results.append({'config': config, 'summary': summary})
            (HERE / 'fine_rankings.json').write_text(json.dumps(fine_results, indent=2) + '\n')
        progress('fine', len(fine), len(fine), fine[-1]['id'], HERE / 'fine' / fine[-1]['id'], 'FINISHED')
    except BaseException:
        (HERE / 'error.txt').write_text(traceback.format_exc())
        if (HERE / 'progress.txt').exists():
            values = (HERE / 'progress.txt').read_text().strip().split('|')
            progress(*values[:5], 'FAILED')
        else:
            progress('coarse', 0, len(plan), 'none', HERE, 'FAILED')
        raise


if __name__ == '__main__':
    main()
