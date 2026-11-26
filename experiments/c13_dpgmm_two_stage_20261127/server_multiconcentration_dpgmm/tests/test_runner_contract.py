import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def config():
    return {
        'schema': 'server_dpgmm_multigranularity_v1',
        'levels': [
            {'name': 'coarse', 'concentration': 0.1},
            {'name': 'medium', 'concentration': 1.0},
            {'name': 'fine', 'concentration': 10.0},
        ],
        'upper_bound': 5,
        'prune': 0.01,
        'covariance_type': 'diag',
        'prior_type': 'dirichlet_process',
        'max_iter': 1000,
        'tol': 0.001,
        'reg_covar': 1e-6,
    }


def clients():
    return [
        {'rho_': np.log([1.0]), 'mu_c': np.array([[-3.0, 0.0]]),
         'log_sigma2_c': np.log(np.array([[0.2, 0.2]]))},
        {'rho_': np.log([1.0]), 'mu_c': np.array([[3.0, 0.0]]),
         'log_sigma2_c': np.log(np.array([[0.2, 0.2]]))},
    ]


class RunnerContractTests(unittest.TestCase):
    def test_method_and_run_selection_are_strict(self):
        import run_experiment
        for dataset, alpha in [('iris', 0.01), ('breast_cancer', 0.05),
                               ('banknote_authentication', 0.05)]:
            run_experiment.validate_run(dataset, 4, alpha)
        for args in [('yeast', 0, 0.05), ('iris', 5, 0.01), ('iris', 0, 0.05)]:
            with self.assertRaises(ValueError):
                run_experiment.validate_run(*args)
        self.assertEqual(run_experiment.METHOD, 'C_server_dpgmm')

    def test_runtime_returns_author_contract_and_logs_provenance(self):
        import server_runtime
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            server_runtime.configure(out, config(), training_seed=2)
            result = server_runtime.sample(clients(), [0.5, 0.5], 100, round_index=7)
            self.assertEqual(len(result), 6)
            z, client_id, lengths, _, _, _ = result
            self.assertEqual(z.shape, (100, 2))
            self.assertEqual(client_id.shape, (100,))
            self.assertEqual(lengths, [1, 1])
            self.assertTrue(set(np.unique(client_id)).issubset({0, 1}))
            rows = [json.loads(line) for line in
                    (out / 'server_multigranularity.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['method'], 'C_server_dpgmm')
            self.assertEqual(rows[0]['round_index'], 7)
            self.assertEqual(rows[0]['level_counts'], [34, 33, 33])
            self.assertEqual([x['name'] for x in rows[0]['levels']],
                             ['coarse', 'medium', 'fine'])
            self.assertNotIn('labels', rows[0])
            self.assertNotIn('ari', rows[0])

    def test_instrumented_sources_import_runtime_and_keep_author_local_gmm(self):
        import protocol
        main = protocol.instrument_main_sampling((ROOT / 'vendor/author/main.py').read_text())
        model = protocol.instrument_model_sampling((ROOT / 'vendor/author/model.py').read_text())
        self.assertIn('from server_runtime import sample as _server_multigranular_sample', main)
        lazy_import = '    from server_runtime import sample as _server_multigranular_sample'
        self.assertIn(lazy_import, model)
        self.assertNotIn('\nfrom server_runtime import sample as _server_multigranular_sample', model)
        self.assertLess(model.index(lazy_import), model.index(
            '_server_multigranular_sample(clients_gmm_param'))
        self.assertIn('K_l = len(np.unique(local_y))', model)

    def test_relative_server_config_is_resolved_before_training_chdir(self):
        import run_experiment
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'server.json'
            source.write_text(json.dumps(config()))
            elsewhere = root / 'elsewhere'
            elsewhere.mkdir()
            os.chdir(root)
            try:
                resolved, loaded = run_experiment.load_server_config(Path('server.json'))
                os.chdir(elsewhere)
                self.assertTrue(resolved.is_absolute())
                self.assertEqual(run_experiment.digest(resolved),
                                 run_experiment.digest(source))
                self.assertEqual(loaded['schema'], 'server_dpgmm_multigranularity_v1')
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    unittest.main()
