"""Fixed-budget generation and tabular distribution metrics for remote evaluation."""
import numpy as np
import torch
from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


def generate(model, n, seed, component=None):
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        weights = torch.softmax(model.rho_.detach().cpu(), 0).numpy()
        means = model.mu_c.detach().cpu().numpy()
        variances = np.exp(model.log_sigma2_c.detach().cpu().numpy())
        labels = (rng.choice(len(weights), n, p=weights) if component is None
                  else np.full(n, component, dtype=int))
        z = means[labels] + rng.normal(size=(n, means.shape[1])) * np.sqrt(variances[labels])
        x = model.decoder(torch.tensor(z, dtype=torch.float32)).detach().cpu().numpy()
    return x, z, labels, weights


def distribution_metrics(real, generated, seed=123, subsample=400):
    rng = np.random.default_rng(seed)
    real = np.asarray(real, dtype=float)
    generated = np.asarray(generated, dtype=float)
    if len(real) < 2 or len(generated) < 2:
        return {'mmd_rbf': None, 'sliced_wasserstein': None,
                'generated_to_real_nn_mean': None, 'feature_range_01_99_fraction': None}
    a = real[rng.choice(len(real), min(len(real), subsample), replace=False)]
    b = generated[rng.choice(len(generated), min(len(generated), subsample), replace=False)]
    aa = pairwise_distances(a, squared=True)
    bb = pairwise_distances(b, squared=True)
    ab = pairwise_distances(a, b, squared=True)
    upper = aa[np.triu_indices(len(a), 1)]
    gamma = max(float(np.median(upper[upper > 0])) if np.any(upper > 0) else 1., 1e-6)
    mmd = float(np.exp(-aa / gamma).mean() + np.exp(-bb / gamma).mean()
                - 2 * np.exp(-ab / gamma).mean())
    directions = rng.normal(size=(64, a.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    swd = float(np.mean([wasserstein_distance(a @ v, b @ v) for v in directions]))
    neighbor = NearestNeighbors(n_neighbors=1).fit(real)
    distance = neighbor.kneighbors(generated, return_distance=True)[0][:, 0]
    low, high = np.quantile(real, [.01, .99], axis=0)
    in_range = np.mean((generated >= low) & (generated <= high), axis=1)
    return {'mmd_rbf': mmd, 'sliced_wasserstein': swd,
            'generated_to_real_nn_mean': float(distance.mean()),
            'feature_range_01_99_fraction': float(in_range.mean())}
