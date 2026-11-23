"""Unit tests for true Macro-Micro Multi-CA v2 core. No training."""
from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import multica_core as core  # noqa: E402


def synthetic_components(n=14, dim=5, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[-3.0] * dim, [3.0] * dim])
    means = []
    for i in range(n):
        g = i % 2
        means.append(centers[g] + 0.35 * rng.normal(size=dim))
    means = np.stack(means)
    weights = np.ones(n) / n
    logvars = np.log(np.full((n, dim), 0.5))
    clients = np.arange(n) % 10
    return weights, means, logvars, clients


class TestMultiCAV2(unittest.TestCase):
    def test_moment_matching(self):
        w = np.array([0.25, 0.75])
        mu = np.array([[0.0, 0.0], [2.0, 1.0]])
        lv = np.log(np.array([[1.0, 2.0], [3.0, 4.0]]))
        clusters, events = core.agglomerate(w, mu, lv, [0, 0], 1)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(events), 1)
        c = clusters[0]
        expected_mu = 0.25 * mu[0] + 0.75 * mu[1]
        self.assertTrue(np.allclose(c.mean, expected_mu))
        expected_var = (0.25 * (np.exp(lv[0]) + (mu[0] - expected_mu) ** 2) +
                        0.75 * (np.exp(lv[1]) + (mu[1] - expected_mu) ** 2))
        self.assertTrue(np.allclose(c.var, expected_var, atol=1e-12))
        self.assertAlmostEqual(c.weight, 1.0)

    def test_agglomerate_target(self):
        w, mu, lv, cc = synthetic_components()
        for target in (2, 4, 8, 14):
            clusters, _ = core.agglomerate(w, mu, lv, cc, target)
            self.assertEqual(len(clusters), target)
            self.assertAlmostEqual(sum(c.weight for c in clusters), 1.0)

    def test_allocation_budget(self):
        alloc = core.allocate_micro_budget([9, 5], [0.7, 0.3], 8, 2)
        self.assertEqual(sum(alloc), 8)
        self.assertTrue(all(a >= 2 for a in alloc))
        self.assertTrue(all(a <= n for a, n in zip(alloc, [9, 5])))
        alloc = core.allocate_micro_budget([14], [1.0], 10, 2)
        self.assertEqual(alloc, [10])

    def test_hierarchy_true_micro(self):
        w, mu, lv, cc = synthetic_components()
        comp = {
            'weights': w, 'means': mu, 'log_sigma2': lv,
            'component_clients': cc,
        }
        for linkage in ('complete', 'average'):
            m = core.build_hierarchical(comp, total_micro=8, k_macro=2,
                                        linkage=linkage)
            self.assertEqual(m['n_micro'], 8)
            self.assertEqual(m['n_macro'], 2)
            self.assertEqual(len(m['macro_of_micro']), 8)
            self.assertAlmostEqual(float(m['weights'].sum()), 1.0, places=12)
            for macro in (0, 1):
                self.assertGreaterEqual(int((m['macro_of_micro'] == macro).sum()), 2)
            covered = set()
            for members in m['micro_members']:
                covered.update(members)
            self.assertEqual(covered, set(range(14)))
            self.assertAlmostEqual(float(m['macro_weights'].sum()), 1.0, places=12)
            for macro, mass in zip(range(2), m['macro_masses']):
                self.assertAlmostEqual(
                    float(m['weights'][m['macro_of_micro'] == macro].sum()),
                    float(mass), places=10)
                self.assertAlmostEqual(float(m['macro_weights'][macro]),
                                       float(mass), places=10)

    def test_specific_micro_sampling(self):
        w, mu, lv, cc = synthetic_components()
        comp = {'weights': w, 'means': mu, 'log_sigma2': lv, 'component_clients': cc}
        m = core.build_hierarchical(comp, total_micro=8, k_macro=2)
        z, ids = core.sample_model(m, 1000, seed=4, micro=3)
        self.assertEqual(z.shape, (1000, mu.shape[1]))
        self.assertTrue(np.all(ids == 3))
        self.assertTrue(np.isfinite(z).all())
        z2, ids2 = core.sample_model(m, 1000, seed=4, macro=1)
        self.assertTrue(set(ids2.tolist()).issubset(
            set(np.flatnonzero(m['macro_of_micro'] == 1).tolist())))
        self.assertEqual(z.shape, z2.shape)

    def test_flat_gmm_capacity(self):
        rng = np.random.default_rng(0)
        z = np.concatenate([rng.normal(-2, 0.4, (200, 3)),
                            rng.normal(0, 0.4, (200, 3)),
                            rng.normal(2, 0.4, (200, 3))])
        m = core.make_flat_model(z, 3, seed=0)
        self.assertEqual(m['n_micro'], 3)
        self.assertAlmostEqual(float(m['weights'].sum()), 1.0, places=12)
        self.assertGreater(m['bic'], 0)

    def test_nll_and_resp_finite(self):
        w, mu, lv, cc = synthetic_components()
        comp = {'weights': w, 'means': mu, 'log_sigma2': lv, 'component_clients': cc}
        m = core.build_hierarchical(comp, total_micro=8, k_macro=2)
        z, _ = core.sample_model(m, 100, seed=1)
        self.assertTrue(np.isfinite(core.model_nll(m, z)))
        p = core.model_resp(m, z)
        self.assertEqual(p.shape, (100, 8))
        self.assertTrue(np.allclose(p.sum(1), 1.0))

    def test_guard_flags_unsafe(self):
        w = np.ones(2) / 2
        mu = np.array([[0.0], [100.0]])
        lv = np.log(np.array([[0.01], [0.01]]))
        _, events = core.agglomerate(w, mu, lv, [0, 1], 1,
                                     guard_threshold=1.0, use_guard=True)
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]['unsafe'])

    def test_merge_functions_do_not_accept_labels(self):
        for fn in (core.agglomerate, core.build_hierarchical,
                   core.allocate_micro_budget, core.make_flat_model,
                   core.flatten_clients):
            params = set(inspect.signature(fn).parameters)
            self.assertFalse({'y', 'labels', 'y_true', 'target_labels'} & params,
                             msg=f'{fn.__name__} accepts labels: {params}')


if __name__ == '__main__':
    unittest.main()
