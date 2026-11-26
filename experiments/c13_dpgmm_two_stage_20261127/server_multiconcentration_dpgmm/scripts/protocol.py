"""Strict experiment configuration and exact source hooks."""
from __future__ import annotations

import math


CONFIG_KEYS = {
    'schema', 'levels', 'upper_bound', 'prune', 'covariance_type',
    'prior_type', 'max_iter', 'tol', 'reg_covar',
}
LEVEL_NAMES = ('coarse', 'medium', 'fine')
LEVEL_CONCENTRATIONS = (0.1, 1.0, 10.0)


def once(source, old, new):
    count = source.count(old)
    if count != 1:
        raise ValueError('source hook mismatch for %r: found %d' % (old[:70], count))
    return source.replace(old, new)


def _positive_finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(name)


def validate_server_config(raw):
    if not isinstance(raw, dict) or set(raw) != CONFIG_KEYS:
        raise ValueError('server config keys must be exactly %s' % sorted(CONFIG_KEYS))
    if raw['schema'] != 'server_dpgmm_multigranularity_v1':
        raise ValueError('schema')
    if raw['covariance_type'] != 'diag':
        raise ValueError('covariance_type')
    if raw['prior_type'] != 'dirichlet_process':
        raise ValueError('prior_type')
    if type(raw['upper_bound']) is not int or raw['upper_bound'] < 1:
        raise ValueError('upper_bound')
    if type(raw['max_iter']) is not int or raw['max_iter'] < 1:
        raise ValueError('max_iter')
    if isinstance(raw['prune'], bool) or not isinstance(raw['prune'], (int, float)):
        raise ValueError('prune')
    if not math.isfinite(raw['prune']) or not 0 <= raw['prune'] < 1:
        raise ValueError('prune')
    _positive_finite(raw['tol'], 'tol')
    _positive_finite(raw['reg_covar'], 'reg_covar')
    levels = raw['levels']
    if not isinstance(levels, list) or len(levels) != 3:
        raise ValueError('levels')
    names, concentrations = [], []
    for level in levels:
        if not isinstance(level, dict) or set(level) != {'name', 'concentration'}:
            raise ValueError('level keys')
        names.append(level['name'])
        _positive_finite(level['concentration'], 'concentration')
        concentrations.append(float(level['concentration']))
    if tuple(names) != LEVEL_NAMES or tuple(concentrations) != LEVEL_CONCENTRATIONS:
        raise ValueError('levels must be fixed coarse/medium/fine concentrations')
    return raw


def instrument_main_sampling(source):
    source = once(
        source,
        'from model import VaDE, vae_avg, pre_train, gmm_sampling\n',
        'from model import VaDE, vae_avg, pre_train, gmm_sampling\n'
        'from server_runtime import sample as _server_multigranular_sample\n',
    )
    old = ('            Z_hat, y_hat, lengths, _, _, _ = '
           'gmm_sampling(clients_gmm_param, weight, args.sample)')
    new = ('            Z_hat, y_hat, lengths, _, _, _ = '
           '_server_multigranular_sample(clients_gmm_param, weight, args.sample, '
           'round_index=epoch)')
    return once(source, old, new)


def instrument_model_sampling(source):
    old = ('    Z_hat, _, _, _, _, _ = '
           'gmm_sampling(clients_gmm_param, weights, args.sample)')
    new = ('    from server_runtime import sample as _server_multigranular_sample\n'
           '    Z_hat, _, _, _, _, _ = '
           '_server_multigranular_sample(clients_gmm_param, weights, args.sample, '
           'round_index=-1)')
    return once(source, old, new)
