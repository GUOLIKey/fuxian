import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def synthetic_clients():
    return [
        {
            'rho_': np.log([0.75, 0.25]),
            'mu_c': np.array([[-3.0, 0.0], [-1.0, 0.0]]),
            'log_sigma2_c': np.log(np.array([[0.2, 0.3], [0.4, 0.5]])),
        },
        {
            'rho_': np.log([1.0]),
            'mu_c': np.array([[3.0, 0.0]]),
            'log_sigma2_c': np.log(np.array([[0.3, 0.2]])),
        },
    ]


class CandidatePoolTests(unittest.TestCase):
    def test_candidate_pool_is_deterministic_and_retains_source(self):
        from server_multigranularity import sample_client_mixture_pool
        one = sample_client_mixture_pool(synthetic_clients(), [0.6, 0.4], 200, seed=17)
        two = sample_client_mixture_pool(synthetic_clients(), [6.0, 4.0], 200, seed=17)
        np.testing.assert_array_equal(one['z'], two['z'])
        np.testing.assert_array_equal(one['source_client'], two['source_client'])
        np.testing.assert_array_equal(one['source_component'], two['source_component'])
        self.assertEqual(one['z'].shape, (200, 2))
        self.assertTrue(set(np.unique(one['source_client'])).issubset({0, 1}))
        self.assertTrue(set(np.unique(one['source_component'])).issubset({0, 1, 2}))
        self.assertTrue(np.isfinite(one['z']).all())
        self.assertAlmostEqual(sum(one['component_weights']), 1.0)

    def test_sampler_does_not_mutate_global_numpy_rng(self):
        from server_multigranularity import sample_client_mixture_pool
        np.random.seed(123)
        expected = np.random.random(3)
        np.random.seed(123)
        sample_client_mixture_pool(synthetic_clients(), [0.6, 0.4], 10, seed=9)
        actual = np.random.random(3)
        np.testing.assert_array_equal(actual, expected)

    def test_malformed_client_mixtures_are_rejected(self):
        from server_multigranularity import sample_client_mixture_pool
        bad_cases = [
            (synthetic_clients(), [0.0, 0.0], 10),
            (synthetic_clients(), [1.0], 10),
            (synthetic_clients(), [0.6, 0.4], 0),
        ]
        broken = synthetic_clients()
        broken[0]['mu_c'][0, 0] = np.nan
        bad_cases.append((broken, [0.6, 0.4], 10))
        for clients, weights, count in bad_cases:
            with self.subTest(count=count, weights=weights):
                with self.assertRaises(ValueError):
                    sample_client_mixture_pool(clients, weights, count, seed=1)


class LevelFitTests(unittest.TestCase):
    def config(self):
        return {
            'schema': 'server_dpgmm_multigranularity_v1',
            'levels': [
                {'name': 'coarse', 'concentration': 0.1},
                {'name': 'medium', 'concentration': 1.0},
                {'name': 'fine', 'concentration': 10.0},
            ],
            'upper_bound': 6,
            'prune': 0.01,
            'covariance_type': 'diag',
            'prior_type': 'dirichlet_process',
            'max_iter': 1000,
            'tol': 0.001,
            'reg_covar': 1e-6,
        }

    def latent_data(self):
        rng = np.random.default_rng(3)
        return np.r_[rng.normal([-4, 0], 0.25, (80, 2)),
                     rng.normal([0, 4], 0.25, (80, 2)),
                     rng.normal([4, 0], 0.25, (80, 2))]

    def test_all_levels_fit_identical_input_with_declared_concentrations(self):
        from server_multigranularity import fit_levels
        levels = fit_levels(self.latent_data(), self.config(), seed=41)
        self.assertEqual([x['name'] for x in levels], ['coarse', 'medium', 'fine'])
        self.assertEqual([x['concentration'] for x in levels], [0.1, 1.0, 10.0])
        self.assertEqual(len({x['input_sha256'] for x in levels}), 1)
        for level in levels:
            self.assertTrue(level['converged'])
            self.assertGreaterEqual(level['active_k'], 1)
            self.assertAlmostEqual(float(level['weights'].sum()), 1.0)
            self.assertTrue(np.isfinite(level['means']).all())
            self.assertTrue(np.isfinite(level['variances']).all())
            self.assertTrue((level['variances'] > 0).all())
            self.assertEqual(level['responsibilities'].shape,
                             (len(self.latent_data()), level['active_k']))
            np.testing.assert_allclose(level['responsibilities'].sum(axis=1), 1.0)

    def test_nonfinite_and_impossible_fit_are_rejected(self):
        from server_multigranularity import fit_levels
        bad = self.latent_data()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            fit_levels(bad, self.config(), seed=1)
        cfg = self.config()
        cfg['upper_bound'] = len(self.latent_data()) + 1
        with self.assertRaises(ValueError):
            fit_levels(self.latent_data(), cfg, seed=1)


class OwnershipAndSamplingTests(unittest.TestCase):
    def test_component_client_mass_uses_soft_responsibilities(self):
        from server_multigranularity import component_client_mass
        level = {
            'responsibilities': np.array([
                [0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]
            ])
        }
        mass = component_client_mass(level, np.array([0, 1, 1, 1]), n_clients=2)
        np.testing.assert_allclose(mass.sum(axis=1), 1.0)
        np.testing.assert_allclose(mass[0], [0.45, 0.55])
        np.testing.assert_allclose(mass[1], [0.05, 0.95])
        self.assertTrue((mass > 0).all(), 'routing must not collapse to dominant client')

    def test_level_allocation_is_balanced_and_exact(self):
        from server_multigranularity import balanced_level_counts
        self.assertEqual(balanced_level_counts(100, 3), [34, 33, 33])
        self.assertEqual(balanced_level_counts(2, 3), [1, 1, 0])
        with self.assertRaises(ValueError):
            balanced_level_counts(0, 3)

    def test_final_sampling_is_deterministic_and_routes_valid_clients(self):
        from server_multigranularity import (
            component_client_mass, fit_levels, sample_multigranular,
        )
        helper = LevelFitTests()
        z = helper.latent_data()
        levels = fit_levels(z, helper.config(), seed=41)
        source_client = np.tile([0, 1, 2], len(z) // 3)
        masses = [component_client_mass(level, source_client, 3) for level in levels]
        one = sample_multigranular(levels, masses, total=100, seed=57)
        two = sample_multigranular(levels, masses, total=100, seed=57)
        np.testing.assert_array_equal(one['z'], two['z'])
        np.testing.assert_array_equal(one['client_id'], two['client_id'])
        np.testing.assert_array_equal(one['level_id'], two['level_id'])
        self.assertEqual(one['z'].shape, (100, 2))
        self.assertEqual(one['level_counts'], [34, 33, 33])
        self.assertTrue(set(np.unique(one['client_id'])).issubset({0, 1, 2}))
        self.assertTrue(np.isfinite(one['z']).all())


if __name__ == '__main__':
    unittest.main()
