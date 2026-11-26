"""Server-side multigranularity DPGMM operations with explicit provenance."""
from __future__ import annotations

import hashlib
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import BayesianGaussianMixture


def _numpy(value, name, ndim):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or not np.isfinite(array).all():
        raise ValueError('%s must be a finite %d-D array' % (name, ndim))
    return array


def _softmax(logits):
    shifted = logits - np.max(logits)
    weights = np.exp(shifted)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError('invalid local mixture weights')
    return weights / total


def sample_client_mixture_pool(state_dicts, client_weights, count, seed):
    """Sample latent points while retaining source client and component IDs."""
    if type(count) is not int or count < 1:
        raise ValueError('count must be a positive integer')
    if not isinstance(state_dicts, (list, tuple)) or not state_dicts:
        raise ValueError('state_dicts must be non-empty')
    cw = _numpy(client_weights, 'client_weights', 1)
    if len(cw) != len(state_dicts) or np.any(cw < 0) or cw.sum() <= 0:
        raise ValueError('invalid client weights')
    cw = cw / cw.sum()

    flat_weights, means, variances = [], [], []
    source_client, source_component = [], []
    component_offset = 0
    latent_dim = None
    for client_id, state in enumerate(state_dicts):
        if not isinstance(state, dict) or not {'rho_', 'mu_c', 'log_sigma2_c'} <= set(state):
            raise ValueError('invalid client mixture state')
        rho = _numpy(state['rho_'], 'rho_', 1)
        mu = _numpy(state['mu_c'], 'mu_c', 2)
        logvar = _numpy(state['log_sigma2_c'], 'log_sigma2_c', 2)
        if len(rho) != len(mu) or mu.shape != logvar.shape or not len(rho):
            raise ValueError('client mixture shape mismatch')
        if latent_dim is None:
            latent_dim = mu.shape[1]
        if mu.shape[1] != latent_dim:
            raise ValueError('latent dimension mismatch')
        var = np.exp(logvar)
        if not np.isfinite(var).all() or np.any(var <= 0):
            raise ValueError('invalid client variances')
        local_weights = _softmax(rho)
        flat_weights.extend((cw[client_id] * local_weights).tolist())
        means.append(mu)
        variances.append(var)
        source_client.extend([client_id] * len(rho))
        source_component.extend(range(component_offset, component_offset + len(rho)))
        component_offset += len(rho)

    weights = np.asarray(flat_weights, dtype=np.float64)
    weights /= weights.sum()
    means = np.concatenate(means, axis=0)
    variances = np.concatenate(variances, axis=0)
    component_clients = np.asarray(source_client, dtype=np.int64)
    component_ids = np.asarray(source_component, dtype=np.int64)

    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(len(weights), size=count, p=weights)
    z = means[chosen] + rng.normal(size=(count, latent_dim)) * np.sqrt(variances[chosen])
    return {
        'z': z,
        'source_client': component_clients[chosen],
        'source_component': component_ids[chosen],
        'component_weights': weights,
    }


def _array_hash(array):
    array = np.ascontiguousarray(array)
    payload = str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()
    return hashlib.sha256(payload).hexdigest()


def fit_levels(z, config, seed):
    """Fit independent DPGMM levels to one identical latent candidate pool."""
    z = _numpy(z, 'z', 2)
    if len(z) < 2 or config['upper_bound'] > len(z):
        raise ValueError('upper_bound exceeds available latent samples')
    input_hash = _array_hash(z)
    output = []
    for level_index, declared in enumerate(config['levels']):
        fit_seed = int(seed) + 1009 * level_index
        model = BayesianGaussianMixture(
            n_components=int(config['upper_bound']),
            covariance_type=config['covariance_type'],
            weight_concentration_prior_type=config['prior_type'],
            weight_concentration_prior=float(declared['concentration']),
            random_state=fit_seed,
            max_iter=int(config['max_iter']),
            tol=float(config['tol']),
            reg_covar=float(config['reg_covar']),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(z)
        if not model.converged_:
            raise RuntimeError('%s DPGMM did not converge' % declared['name'])
        keep = np.flatnonzero(model.weights_ > float(config['prune']))
        if not len(keep):
            raise RuntimeError('%s DPGMM pruning removed every component' % declared['name'])
        weights = np.asarray(model.weights_[keep], dtype=np.float64)
        retained_mass = float(weights.sum())
        weights /= retained_mass
        means = np.asarray(model.means_[keep], dtype=np.float64)
        variances = np.asarray(model.covariances_[keep], dtype=np.float64)
        responsibilities = np.asarray(model.predict_proba(z)[:, keep], dtype=np.float64)
        row_mass = responsibilities.sum(axis=1, keepdims=True)
        if np.any(row_mass <= 0):
            raise RuntimeError('%s active responsibility mass is zero' % declared['name'])
        responsibilities /= row_mass
        arrays = (weights, means, variances, responsibilities)
        if not all(np.isfinite(x).all() for x in arrays) or np.any(variances <= 0):
            raise RuntimeError('%s DPGMM produced invalid values' % declared['name'])
        output.append({
            'name': declared['name'],
            'concentration': float(declared['concentration']),
            'fit_seed': fit_seed,
            'input_sha256': input_hash,
            'active_indices': keep,
            'active_k': int(len(keep)),
            'weights': weights,
            'means': means,
            'variances': variances,
            'responsibilities': responsibilities,
            'retained_mass': retained_mass,
            'removed_mass': float(1.0 - retained_mass),
            'converged': bool(model.converged_),
            'n_iter': int(model.n_iter_),
            'lower_bound': float(model.lower_bound_),
            'warnings': [str(item.message) for item in caught
                         if issubclass(item.category, ConvergenceWarning)],
        })
    return output


def component_client_mass(level, source_client, n_clients):
    """Aggregate candidate posterior responsibility into client routing mass."""
    responsibilities = _numpy(level['responsibilities'], 'responsibilities', 2)
    source_client = np.asarray(source_client)
    if source_client.ndim != 1 or len(source_client) != len(responsibilities):
        raise ValueError('source_client length mismatch')
    if type(n_clients) is not int or n_clients < 1:
        raise ValueError('n_clients')
    if not np.issubdtype(source_client.dtype, np.integer):
        raise ValueError('source_client must contain integers')
    if np.any(source_client < 0) or np.any(source_client >= n_clients):
        raise ValueError('source_client out of range')
    mass = np.zeros((responsibilities.shape[1], n_clients), dtype=np.float64)
    for client_id in range(n_clients):
        mass[:, client_id] = responsibilities[source_client == client_id].sum(axis=0)
    totals = mass.sum(axis=1, keepdims=True)
    if np.any(totals <= 0) or not np.isfinite(mass).all():
        raise RuntimeError('component has no finite client responsibility mass')
    mass /= totals
    return mass


def balanced_level_counts(total, n_levels):
    if type(total) is not int or total < 1:
        raise ValueError('total must be positive')
    if type(n_levels) is not int or n_levels < 1:
        raise ValueError('n_levels must be positive')
    base, remainder = divmod(total, n_levels)
    return [base + (1 if index < remainder else 0) for index in range(n_levels)]


def sample_multigranular(levels, client_masses, total, seed):
    """Sample equally across levels and route by soft component-client mass."""
    if not levels or len(levels) != len(client_masses):
        raise ValueError('levels/client_masses mismatch')
    counts = balanced_level_counts(total, len(levels))
    rng = np.random.default_rng(int(seed))
    z_parts, client_parts, level_parts, component_parts = [], [], [], []
    latent_dim = None
    for level_id, (level, client_mass, count) in enumerate(zip(levels, client_masses, counts)):
        weights = _numpy(level['weights'], 'weights', 1)
        means = _numpy(level['means'], 'means', 2)
        variances = _numpy(level['variances'], 'variances', 2)
        client_mass = _numpy(client_mass, 'client_mass', 2)
        if len(weights) != len(means) or means.shape != variances.shape:
            raise ValueError('level mixture shape mismatch')
        if client_mass.shape[0] != len(weights):
            raise ValueError('client mass component mismatch')
        if latent_dim is None:
            latent_dim = means.shape[1]
        if means.shape[1] != latent_dim or np.any(variances <= 0):
            raise ValueError('invalid level variances/dimension')
        weights = weights / weights.sum()
        if count == 0:
            continue
        components = rng.choice(len(weights), size=count, p=weights)
        z = means[components] + rng.normal(size=(count, latent_dim)) * np.sqrt(variances[components])
        clients = np.empty(count, dtype=np.int64)
        for index, component in enumerate(components):
            probabilities = client_mass[component]
            probabilities = probabilities / probabilities.sum()
            clients[index] = rng.choice(client_mass.shape[1], p=probabilities)
        z_parts.append(z)
        client_parts.append(clients)
        level_parts.append(np.full(count, level_id, dtype=np.int64))
        component_parts.append(components.astype(np.int64))
    result = {
        'z': np.concatenate(z_parts, axis=0),
        'client_id': np.concatenate(client_parts),
        'level_id': np.concatenate(level_parts),
        'component_id': np.concatenate(component_parts),
        'level_counts': counts,
    }
    if len(result['z']) != total or not np.isfinite(result['z']).all():
        raise RuntimeError('invalid final multigranular sample')
    return result
