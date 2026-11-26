import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


class InstrumentationTests(unittest.TestCase):
    def setUp(self):
        self.main = (ROOT / 'vendor/author/main.py').read_text()
        self.model = (ROOT / 'vendor/author/model.py').read_text()

    def test_sampling_hooks_are_exact_and_client_gmm_is_unchanged(self):
        import protocol
        new_main = protocol.instrument_main_sampling(self.main)
        new_model = protocol.instrument_model_sampling(self.model)
        self.assertEqual(new_main.count('_server_multigranular_sample('), 1)
        self.assertEqual(new_model.count('_server_multigranular_sample('), 1)
        self.assertIn('round_index=epoch', new_main)
        self.assertIn('round_index=-1', new_model)
        client_block = (
            "gmm = GaussianMixture(n_components=K_l, covariance_type='diag', random_state=42)\n"
            "            gmm.fit(local_mu)"
        )
        self.assertIn(client_block, new_model)
        self.assertNotIn('AgglomerativeClustering', new_main + new_model)
        self.assertNotIn('complete-linkage', new_main + new_model)

    def test_missing_or_duplicate_hook_points_are_rejected(self):
        import protocol
        with self.assertRaises(ValueError):
            protocol.instrument_main_sampling(self.main + self.main)
        with self.assertRaises(ValueError):
            protocol.instrument_model_sampling(self.model.replace(
                'Z_hat, _, _, _, _, _ = gmm_sampling(clients_gmm_param, weights, args.sample)',
                'Z_hat = None',
            ))


if __name__ == '__main__':
    unittest.main()
