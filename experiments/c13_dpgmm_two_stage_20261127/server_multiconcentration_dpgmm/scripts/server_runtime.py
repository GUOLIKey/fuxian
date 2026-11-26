"""Stateful adapter from author sampling calls to the pure server module."""
from __future__ import annotations

import json
from pathlib import Path

from protocol import validate_server_config
from server_multigranularity import (
    component_client_mass,
    fit_levels,
    sample_client_mixture_pool,
    sample_multigranular,
)


METHOD = 'C_server_dpgmm'
_CONTEXT = {}


def configure(out, config, training_seed):
    _CONTEXT.clear()
    _CONTEXT.update(
        out=Path(out).resolve(),
        config=validate_server_config(config),
        training_seed=int(training_seed),
    )


def _seeds(round_index):
    base = _CONTEXT['training_seed'] * 1_000_003 + (int(round_index) + 2) * 10_007
    return {'candidate': base + 101, 'fit': base + 211, 'final': base + 307}


def _level_record(level, client_mass):
    return {
        'name': level['name'],
        'concentration': level['concentration'],
        'fit_seed': level['fit_seed'],
        'input_sha256': level['input_sha256'],
        'active_k': level['active_k'],
        'weights': level['weights'].tolist(),
        'active_indices': level['active_indices'].tolist(),
        'retained_mass': level['retained_mass'],
        'removed_mass': level['removed_mass'],
        'converged': level['converged'],
        'n_iter': level['n_iter'],
        'lower_bound': level['lower_bound'],
        'warnings': level['warnings'],
        'client_mass': client_mass.tolist(),
    }


def sample(state_dicts, client_weights, nums, round_index):
    if not _CONTEXT:
        raise RuntimeError('server_runtime.configure must be called first')
    seeds = _seeds(round_index)
    pool = sample_client_mixture_pool(
        state_dicts, client_weights, int(nums), seed=seeds['candidate'])
    levels = fit_levels(pool['z'], _CONTEXT['config'], seed=seeds['fit'])
    n_clients = len(state_dicts)
    client_masses = [
        component_client_mass(level, pool['source_client'], n_clients)
        for level in levels
    ]
    final = sample_multigranular(levels, client_masses, int(nums), seed=seeds['final'])
    row = {
        'method': METHOD,
        'round_index': int(round_index),
        'candidate_count': int(nums),
        'candidate_sha256': levels[0]['input_sha256'],
        'seeds': seeds,
        'level_counts': final['level_counts'],
        'routing_counts': [int((final['client_id'] == i).sum()) for i in range(n_clients)],
        'levels': [_level_record(level, mass)
                   for level, mass in zip(levels, client_masses)],
    }
    log = _CONTEXT['out'] / 'server_multigranularity.jsonl'
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as handle:
        handle.write(json.dumps(row, sort_keys=True) + '\n')
    return final['z'], final['client_id'], [1] * n_clients, None, None, None
