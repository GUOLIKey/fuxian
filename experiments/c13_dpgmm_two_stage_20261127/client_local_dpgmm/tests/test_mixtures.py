import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
class MixtureTests(unittest.TestCase):
    def test_fit_without_torch_dependency(self):
        import paired_runtime as r
        x=np.random.RandomState(0).normal(size=(40,2));y=np.repeat([0,1],20)
        w,m,v,meta=r.fit_mixture(x,y,'A',None)
        self.assertEqual(len(w),2);self.assertAlmostEqual(w.sum(),1)
    def test_dp_and_samek(self):
        import paired_runtime as r
        x=np.r_[np.random.RandomState(0).normal(-4,.2,(30,2)),np.random.RandomState(1).normal(4,.2,(30,2))]
        cfg=dict(upper_bound=3,prior=.1,prune=.01,fit_seed=42,max_iter=1000,tol=.001,reg_covar=1e-6)
        w,m,v,b=r.fit_mixture(x,None,'B',cfg)
        wc,mc,vc,c=r.fit_mixture(x,None,'C',cfg,k=b['k'])
        self.assertEqual(b['k'],c['k']);self.assertTrue((v>0).all());self.assertAlmostEqual(w.sum(),1)
    def test_invalid_input_and_all_pruned_rejected(self):
        import paired_runtime as r
        with self.assertRaises(ValueError):r.fit_mixture(np.array([[np.nan],[0.]]),[0,1],'A',None)
        cfg=dict(upper_bound=2,prior=.1,prune=.999999,fit_seed=42,max_iter=1000,tol=.001,reg_covar=1e-6)
        with self.assertRaises(ValueError):r.fit_mixture(np.random.RandomState(0).normal(size=(60,2)),None,'B',cfg)
if __name__=='__main__':unittest.main()
