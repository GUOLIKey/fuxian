import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import multica_core as mc


def test_w2_identity_symmetry_nonnegative():
    mu = np.array([0.1, -0.2, 0.3])
    var = np.array([0.5, 1.0, 2.0])
    assert mc.w2_diag(mu, var, mu, var) == 0.0
    mu2 = np.array([1.0, 0.5, -0.5])
    var2 = np.array([1.5, 0.2, 0.7])
    a = mc.w2_diag(mu, var, mu2, var2)
    b = mc.w2_diag(mu2, var2, mu, var)
    assert a > 0 and abs(a - b) < 1e-12


def test_moment_matching_formula_and_mass():
    w = np.array([0.2, 0.3, 0.5])
    mu = np.array([[0.0, 1.0], [2.0, -1.0], [3.0, 2.0]])
    lv = np.log(np.array([[1.0, 0.5], [0.4, 2.0], [0.8, 1.2]]))
    macro = mc.multi_ca_merge(w, mu, lv, [0, 1, 2], target_components=1,
                              max_components=1, min_components=1,
                              allow_early_stop=False)
    assert abs(macro['weights'].sum() - 1.0) < 1e-12
    # Compare with the closed-form weighted moment match.
    m = w.sum()
    mn = np.sum(w[:, None] * mu, axis=0) / m
    vn = np.zeros(2)
    for k in range(3):
        vn += w[k] * (np.exp(lv[k]) + (mu[k] - mn) ** 2)
    vn /= m
    assert np.allclose(macro['means'][0], mn, atol=1e-10)
    assert np.allclose(np.exp(macro['log_sigma2'][0]), vn, atol=1e-10)
    assert macro['n_unsafe_merges'] >= 0


def test_merge_to_two_is_deterministic():
    w = np.array([0.4, 0.6, 0.2, 0.3, 0.5])
    mu = np.array([[0.0, 0.0], [0.1, 0.2], [5.0, 5.0], [5.2, 4.9], [2.0, 2.0]])
    lv = np.log(np.array([[1.0, 1.0], [0.8, 0.9], [1.2, 1.1], [0.9, 1.0], [1.3, 0.7]]))
    a = mc.multi_ca_merge(w, mu, lv, [0, 0, 1, 1, 2], target_components=2,
                          max_components=2, min_components=2,
                          allow_early_stop=False)
    b = mc.multi_ca_merge(w, mu, lv, [0, 0, 1, 1, 2], target_components=2,
                          max_components=2, min_components=2,
                          allow_early_stop=False)
    assert np.allclose(a['weights'], b['weights'])
    assert np.allclose(a['means'], b['means'])
    assert np.allclose(a['log_sigma2'], b['log_sigma2'])
    assert a['n_macro_components'] == 2
    assert a['n_merges'] == 3
    assert len(a['merge_events']) == 3


def test_invalid_values_rejected():
    w = np.array([0.5, 0.5])
    mu = np.zeros((2, 2))
    lv = np.zeros((2, 2))
    bad = lv.copy(); bad[0, 0] = np.nan
    for args in [
        (w, mu, bad),
        (np.array([-0.1, 0.9]), mu, lv),
        (np.array([0.0, 0.0]), mu, lv),
    ]:
        try:
            mc.multi_ca_merge(*args, [0, 1], target_components=2,
                              max_components=2, min_components=2)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid GMM was accepted')


def test_sampling_layout_and_seed():
    w = np.array([0.3, 0.7])
    mu = np.array([[0.0, 0.0], [2.0, 2.0]])
    lv = np.zeros((2, 2))
    macro = mc.multi_ca_merge(w, mu, lv, [0, 1], target_components=2,
                              max_components=2, min_components=2,
                              allow_early_stop=False)
    z1, l1 = mc.sample_macro_gmm(macro, 5000, seed=7)
    z2, l2 = mc.sample_macro_gmm(macro, 5000, seed=7)
    assert np.array_equal(z1, z2) and np.array_equal(l1, l2)
    z, blocks, layout = mc.sample_macro_for_author_loop(macro, 5000, seed=9, n_clients=3)
    assert z.shape == (5000, 2)
    assert sum(layout['lengths']) == 2
    assert set(np.unique(blocks)).issubset(set(range(2)))
    assert layout['lengths'][0] >= 0


def test_reconstruct_components():
    clients = [
        {'rho_': np.log(np.array([1.0])), 'mu_c': np.array([[0.0, 0.0]]),
         'log_sigma2_c': np.zeros((1, 2))},
        {'rho_': np.log(np.array([0.25, 0.75])),
         'mu_c': np.array([[1.0, 0.0], [2.0, 0.0]]),
         'log_sigma2_c': np.zeros((2, 2))},
    ]
    w, mu, lv, owner, cw = mc.reconstruct_client_components(clients, [100, 300])
    assert len(w) == 3 and len(mu) == 3 and len(lv) == 3
    assert owner.tolist() == [0, 1, 1]
    assert abs(w.sum() - 1.0) < 1e-12
    assert abs(cw.sum() - 1.0) < 1e-12


def test_early_stop_capacity():
    w = np.array([0.25, 0.25, 0.25, 0.25])
    mu = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0], [15.0, 15.0]])
    lv = np.zeros((4, 2))
    macro = mc.multi_ca_merge(w, mu, lv, [0, 1, 2, 3], target_components=2,
                              max_components=4, min_components=2,
                              merge_threshold=0.01, allow_early_stop=True)
    assert macro['stopped_early']
    assert macro['n_macro_components'] >= 2
    assert macro['n_unsafe_merges'] == 0
    assert macro['status'] == 'EARLY_STOPPED'


def test_normalized_distance_finite():
    mu1 = np.zeros(3); v1 = np.array([1e-12, 1e-8, 1.0])
    mu2 = np.ones(3); v2 = np.array([1e-12, 1e-8, 1.0])
    d = mc.normalized_bures_diag(mu1, v1, mu2, v2)
    assert math.isfinite(d) and d > 0
