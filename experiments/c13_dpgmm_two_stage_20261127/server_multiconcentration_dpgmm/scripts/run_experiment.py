"""Isolated runner for author-client GMM plus server multigranularity DPGMM."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METHOD = 'C_server_dpgmm'
DATASETS = ('iris', 'breast_cancer', 'banknote_authentication')


def validate_run(dataset, seed, alpha):
    if dataset not in DATASETS:
        raise ValueError('unsupported dataset')
    if type(seed) is not int or seed not in range(5):
        raise ValueError('training seed must be 0 through 4')
    expected = 0.01 if dataset == 'iris' else 0.05
    if alpha != expected:
        raise ValueError('%s fixes VaDE alpha=%s' % (dataset, expected))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_server_config(path):
    from protocol import validate_server_config
    resolved = Path(path).resolve()
    return resolved, validate_server_config(json.loads(resolved.read_text()))


def parser():
    result = argparse.ArgumentParser()
    result.add_argument('--dataset', required=True, choices=DATASETS)
    result.add_argument('--seed', type=int, required=True)
    result.add_argument('--alpha', type=float, required=True)
    result.add_argument('--data-dir', type=Path, required=True)
    result.add_argument('--out', type=Path, required=True)
    result.add_argument('--server-config', type=Path, required=True)
    result.add_argument('--smoke', action='store_true')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    validate_run(args.dataset, args.seed, args.alpha)
    manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text())
    for relative, expected in manifest.items():
        if digest(ROOT / relative) != expected:
            raise ValueError('frozen source changed: %s' % relative)
    sys.path.insert(0, str(ROOT / 'scripts'))
    sys.path.insert(0, str(ROOT / 'vendor/audited_scripts'))
    from protocol import (
        instrument_main_sampling,
        instrument_model_sampling,
        validate_server_config,
    )
    import server_runtime
    from common import train as trainer

    args.server_config, server_config = load_server_config(args.server_config)
    args.out = args.out.resolve()
    args.data_dir = args.data_dir.resolve()
    server_runtime.configure(args.out, server_config, args.seed)

    original_instrument_main = trainer.instrument_main

    def instrument(source, pretrain):
        model_path = args.out / 'source/model.py'
        before = model_path.read_text()
        after = instrument_model_sampling(before)
        model_path.write_text(after)
        (args.out / 'server_model.diff').write_text(''.join(difflib.unified_diff(
            before.splitlines(True), after.splitlines(True),
            'author/model.py', 'copy/model.py')))
        return instrument_main_sampling(original_instrument_main(source, pretrain))

    trainer.instrument_main = instrument
    train_args = trainer.build_parser().parse_args([
        '--source', str(ROOT / 'vendor/author'),
        '--out', str(args.out),
        '--data-dir', str(args.data_dir),
        '--dataset', args.dataset,
        '--mode', 'author',
        '--local-gmm', 'author',
        '--device', 'cpu',
        '--seed', str(args.seed),
        '--alpha', str(args.alpha),
        '--lr', '0.002',
        '--sample', '100',
        '--rounds', '3' if args.smoke else '300',
        '--pretrain', '2' if args.smoke else '50',
        '--save-every', '3' if args.smoke else '300',
        '--batch-size', '200',
        '--clients', '10',
        '--hid-dim', '10',
        '--schedule', 'author',
    ])
    try:
        trainer._run_in_process(train_args)
    finally:
        if args.out.exists():
            record = {
                'method': METHOD,
                'dataset': args.dataset,
                'training_seed': args.seed,
                'alpha': args.alpha,
                'smoke': args.smoke,
                'server_config': server_config,
                'server_config_sha256': digest(args.server_config),
                'method_code_hashes': {
                    path.name: digest(path) for path in sorted((ROOT / 'scripts').glob('*.py'))
                },
            }
            (args.out / 'experiment.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
