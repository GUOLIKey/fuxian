import importlib.util
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

class ProtocolTests(unittest.TestCase):
    def load(self):
        p=ROOT/'scripts/protocol.py'
        self.assertTrue(p.exists(), 'protocol implementation missing')
        s=importlib.util.spec_from_file_location('protocol',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
    def test_hook_precedes_gmm(self):
        m=self.load();src=(ROOT/'vendor/author/model.py').read_text()
        new=m.instrument_model(src)
        self.assertLess(new.index('_paired_boundary('),new.index('_paired_local(c['))
        self.assertIn('range(0 if _paired_restored else pre_epoch)',new)
        self.assertIn('gmm_global = GaussianMixture(n_components=args.nClusters',new)
    def test_rejects_duplicate_hook(self):
        m=self.load();src=(ROOT/'vendor/author/model.py').read_text()
        with self.assertRaises(ValueError):m.instrument_model(src+src)
    def test_dp_config_required_and_validated(self):
        m=self.load()
        for cfg in ({},{'upper_bound':0,'prior':.1,'prune':0,'fit_seed':42,'max_iter':500,'tol':.001,'reg_covar':1e-6}):
            with self.assertRaises(ValueError):m.validate_dp(cfg)
    def test_no_silent_dataset_or_group(self):
        m=self.load()
        with self.assertRaises(ValueError):m.validate_selection('yeast','A')
        with self.assertRaises(ValueError):m.validate_selection('iris','multica')
    def test_samek_does_not_refit_dp(self):
        m=self.load();src=(ROOT/'vendor/author/model.py').read_text();new=m.instrument_model(src)
        self.assertIn('_paired_local(c[',new)
        self.assertNotIn('K_l = len(np.unique(local_y))',new)
if __name__=='__main__':unittest.main()
