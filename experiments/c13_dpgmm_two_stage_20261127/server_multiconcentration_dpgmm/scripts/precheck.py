"""Label-free structural gate for server multigranularity DPGMM."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


LEVEL_NAMES = ['coarse', 'medium', 'fine']
CONCENTRATIONS = [0.1, 1.0, 10.0]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _reject_forbidden_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if 'label' in lowered or lowered == 'ari':
                raise ValueError('precheck must not use labels or ARI')
            _reject_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden_keys(child)


def _check_structure(row):
    _reject_forbidden_keys(row)
    if row.get('method') != 'C_server_dpgmm':
        raise ValueError('wrong method')
    if type(row.get('round_index')) is not int or row['round_index'] < -1:
        raise ValueError('invalid round index')
    levels = row.get('levels')
    if not isinstance(levels, list) or len(levels) != 3:
        raise ValueError('three levels required')
    if [level.get('name') for level in levels] != LEVEL_NAMES:
        raise ValueError('level names/order mismatch')
    if [level.get('concentration') for level in levels] != CONCENTRATIONS:
        raise ValueError('level concentrations mismatch')
    candidate = row.get('candidate_sha256')
    if not candidate or any(level.get('input_sha256') != candidate for level in levels):
        raise ValueError('levels did not fit the same candidate pool')
    if not all(level.get('converged') is True for level in levels):
        raise ValueError('non-converged level')
    active = [level.get('active_k') for level in levels]
    if not all(type(value) is int and value >= 1 for value in active):
        raise ValueError('invalid active K')
    return active, candidate


def check_row(row):
    active, candidate = _check_structure(row)
    if row.get('round_index') != -1:
        raise ValueError('single-row precheck requires initialization record')
    distinct = len(set(active))
    if distinct < 2:
        raise ValueError('DPGMM settings did not produce multiple granularities')
    return {'active_k': active, 'distinct_active_k': distinct,
            'candidate_sha256': candidate}


def check_rows(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError('precheck log is empty')
    expected_rounds = [-1] + list(range(len(rows) - 1))
    actual_rounds = [row.get('round_index') for row in rows]
    if actual_rounds != expected_rounds:
        raise ValueError('precheck log round sequence mismatch')
    active_by_call, candidate_hashes = [], []
    for row in rows:
        active, candidate = _check_structure(row)
        active_by_call.append(active)
        candidate_hashes.append(candidate)
    multigranular_calls = sum(len(set(active)) >= 2 for active in active_by_call)
    if multigranular_calls < 1:
        raise ValueError('no smoke call produced multiple active granularities')
    return {
        'active_k': active_by_call[0],
        'distinct_active_k': len(set(active_by_call[0])),
        'active_k_by_call': active_by_call,
        'multigranular_calls': multigranular_calls,
        'candidate_sha256': candidate_hashes[0],
        'candidate_sha256_by_call': candidate_hashes,
    }


def create_receipt(dataset, log, config, manifest, output):
    rows = [json.loads(line) for line in Path(log).read_text().splitlines() if line.strip()]
    summary = check_rows(rows)
    receipt = {
        'schema': 'server_dpgmm_precheck_v1',
        'method': 'C_server_dpgmm',
        'dataset': dataset,
        'server_config_sha256': digest(config),
        'source_manifest_sha256': digest(manifest),
        'log_sha256': digest(log),
        **summary,
    }
    with Path(output).open('x') as handle:
        json.dump(receipt, handle, indent=2)
        handle.write('\n')
    return receipt


def verify_receipt(dataset, receipt, config, manifest):
    value = json.loads(Path(receipt).read_text())
    expected = {
        'schema': 'server_dpgmm_precheck_v1',
        'method': 'C_server_dpgmm',
        'dataset': dataset,
        'server_config_sha256': digest(config),
        'source_manifest_sha256': digest(manifest),
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise ValueError('precheck receipt mismatch: %s' % key)
    if value.get('multigranular_calls', 0) < 1:
        raise ValueError('precheck receipt lacks a multigranular smoke call')
    return value


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('create')
    create.add_argument('--dataset', required=True)
    create.add_argument('--log', type=Path, required=True)
    create.add_argument('--config', type=Path, required=True)
    create.add_argument('--manifest', type=Path, required=True)
    create.add_argument('--output', type=Path, required=True)
    verify = sub.add_parser('verify')
    verify.add_argument('--dataset', required=True)
    verify.add_argument('--receipt', type=Path, required=True)
    verify.add_argument('--config', type=Path, required=True)
    verify.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'create':
        create_receipt(args.dataset, args.log, args.config, args.manifest, args.output)
    else:
        verify_receipt(args.dataset, args.receipt, args.config, args.manifest)


if __name__ == '__main__':
    main()
