import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def row(active=(2, 3, 5), round_index=-1):
    return {
        'method': 'C_server_dpgmm',
        'round_index': round_index,
        'candidate_sha256': 'a' * 64,
        'level_counts': [34, 33, 33],
        'levels': [
            {'name': name, 'concentration': concentration, 'active_k': k,
             'input_sha256': 'a' * 64, 'converged': True}
            for name, concentration, k in
            zip(('coarse', 'medium', 'fine'), (0.1, 1.0, 10.0), active)
        ],
    }


class PrecheckTests(unittest.TestCase):
    def test_distinct_active_counts_pass_without_labels_or_ari(self):
        import precheck
        result = precheck.check_row(row())
        self.assertEqual(result['active_k'], [2, 3, 5])
        self.assertEqual(result['distinct_active_k'], 3)

    def test_same_granularity_is_rejected(self):
        import precheck
        with self.assertRaises(ValueError):
            precheck.check_row(row((3, 3, 3)))

    def test_label_or_metric_leakage_is_rejected_recursively(self):
        import precheck
        for key in ('labels', 'ari', 'true_labels'):
            changed = row()
            changed['nested'] = {key: [0, 1]}
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    precheck.check_row(changed)

    def test_wrong_level_identity_or_candidate_is_rejected(self):
        import precheck
        changed = row()
        changed['levels'][1]['name'] = 'fine'
        with self.assertRaises(ValueError):
            precheck.check_row(changed)

    def test_full_smoke_passes_when_later_round_produces_distinct_k(self):
        import precheck
        result = precheck.check_rows([
            row((2, 2, 2), -1),
            row((3, 3, 6), 0),
            row((3, 6, 6), 1),
        ])
        self.assertEqual(result['active_k_by_call'], [[2, 2, 2], [3, 3, 6], [3, 6, 6]])
        self.assertEqual(result['multigranular_calls'], 2)

    def test_full_smoke_rejects_when_every_call_has_one_k(self):
        import precheck
        with self.assertRaises(ValueError):
            precheck.check_rows([
                row((2, 2, 2), -1),
                row((3, 3, 3), 0),
            ])
        changed = row()
        changed['levels'][2]['input_sha256'] = 'b' * 64
        with self.assertRaises(ValueError):
            precheck.check_row(changed)


if __name__ == '__main__':
    unittest.main()
