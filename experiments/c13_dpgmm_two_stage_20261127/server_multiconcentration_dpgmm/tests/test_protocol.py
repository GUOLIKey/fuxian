import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_protocol():
    path = ROOT / 'scripts/protocol.py'
    spec = importlib.util.spec_from_file_location('server_protocol', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProtocolTests(unittest.TestCase):
    def test_formal_config_is_explicit_and_valid(self):
        module = load_protocol()
        raw = json.loads((ROOT / 'configs/server_dpgmm.json').read_text())
        cfg = module.validate_server_config(raw)
        self.assertEqual([x['name'] for x in cfg['levels']], ['coarse', 'medium', 'fine'])
        self.assertEqual([x['concentration'] for x in cfg['levels']], [0.1, 1.0, 10.0])
        self.assertEqual(cfg['upper_bound'], 10)
        self.assertEqual(cfg['prune'], 0.01)
        self.assertEqual(cfg['covariance_type'], 'diag')

    def test_unknown_or_missing_keys_are_rejected(self):
        module = load_protocol()
        raw = json.loads((ROOT / 'configs/server_dpgmm.json').read_text())
        for changed in ({**raw, 'mystery': 1}, {k: v for k, v in raw.items() if k != 'tol'}):
            with self.assertRaises(ValueError):
                module.validate_server_config(changed)

    def test_invalid_levels_and_numbers_are_rejected(self):
        module = load_protocol()
        raw = json.loads((ROOT / 'configs/server_dpgmm.json').read_text())
        variants = []
        duplicate = {**raw, 'levels': [dict(x) for x in raw['levels']]}
        duplicate['levels'][1]['name'] = 'coarse'
        variants.append(duplicate)
        wrong_order = {**raw, 'levels': list(reversed(raw['levels']))}
        variants.append(wrong_order)
        for key, value in [('upper_bound', 0), ('prune', 1.0), ('max_iter', 0),
                           ('tol', 0.0), ('reg_covar', float('nan'))]:
            variants.append({**raw, key: value})
        for changed in variants:
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    module.validate_server_config(changed)


if __name__ == '__main__':
    unittest.main()
