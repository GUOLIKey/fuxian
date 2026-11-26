"""Runtime hooks for a pre-GMM paired fork. Used only by the isolated runner."""
import hashlib,json,random,warnings
from pathlib import Path
import numpy as np
from sklearn.mixture import GaussianMixture,BayesianGaussianMixture
from sklearn.exceptions import ConvergenceWarning
CTX={}
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_hash(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def write(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,indent=2);f.write('\n')
def states(server,clients):
    return [{k:v.detach().cpu().clone() for k,v in x['vade'].state_dict().items()} for x in [server,*clients]]
def rng():
    import torch
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state()}
def set_rng(r):
    import torch
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'])
def identity():
    cfg=json.loads((CTX['out']/'config.json').read_text())
    keys=('dataset','seed','alpha','lr','schedule','sample','pretrain','batch_size','clients','latent_dim','device','data_sha256','client_data_hashes','author_source_hashes','runtime_packages')
    return {k:cfg[k] for k in keys}
def restore(server,clients):
    import torch
    if CTX['group']=='prepare':return False
    p=CTX['prepared']
    receipt=json.loads((p.parent/'prepared.json').read_text())
    if digest(p)!=receipt['checkpoint_sha256']:raise ValueError('prepared checkpoint checksum mismatch')
    saved=torch.load(p,map_location='cpu')
    if saved['identity']!=identity():raise ValueError('prepared identity/data/source/environment mismatch')
    if len(saved['states'])!=len(clients)+1:raise ValueError('client count mismatch')
    for obj,state in zip([server,*clients],saved['states']):obj['vade'].load_state_dict(state,strict=True)
    CTX['saved_rng']=saved['rng']
    return True

def boundary(server,clients):
    import torch
    if CTX['group']=='prepare':
        p=CTX['out']/'pre_gmm.pt'
        if p.exists():raise FileExistsError(p)
        torch.save({'identity':identity(),'states':states(server,clients),'rng':rng(),
                    'stage':'after AE training, before local GMM; not full-training resume'},p)
        write(CTX['out']/'prepared.json',{'checkpoint_sha256':digest(p),'identity':identity()})
        raise SystemExit(0)
    set_rng(CTX['saved_rng'])
    write(CTX['out']/'fork.json',{'prepared_sha256':digest(CTX['prepared']),'identity':identity()})

def fit_mixture(x,y,group,cfg,k=None):
    if not np.isfinite(x).all() or len(x)<2:raise ValueError('invalid local representations')
    if group=='A':
        g=GaussianMixture(n_components=len(np.unique(y)),covariance_type='diag',random_state=42)
    elif group=='B':
        if cfg['upper_bound']>len(x):raise ValueError('DP upper_bound exceeds client sample count')
        g=BayesianGaussianMixture(n_components=cfg['upper_bound'],covariance_type='diag',weight_concentration_prior_type='dirichlet_process',weight_concentration_prior=cfg['prior'],random_state=cfg['fit_seed'],max_iter=cfg['max_iter'],tol=cfg['tol'],reg_covar=cfg['reg_covar'])
    elif group=='C':
        if k is None or not 1<=k<=len(x):raise ValueError('invalid paired K')
        # Match A's ordinary-GMM fitting settings; only its component count changes.
        g=GaussianMixture(n_components=k,covariance_type='diag',random_state=42)
    else:raise ValueError(group)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');g.fit(x)
    if not g.converged_:raise RuntimeError('mixture did not converge; retain failed run')
    keep=np.flatnonzero(g.weights_>cfg['prune']) if group=='B' else np.arange(len(g.weights_))
    if not len(keep):raise ValueError('pruning removed every component')
    w=g.weights_[keep];mass=float(w.sum());w=w/mass
    mu=g.means_[keep];var=g.covariances_[keep]
    if not all(np.isfinite(a).all() for a in (w,mu,var)) or (var<=0).any():raise ValueError('invalid mixture parameters')
    meta={'k':len(w),'weights':w.tolist(),'raw_weights':g.weights_.tolist(),'removed_mass':1-mass,'converged':bool(g.converged_),'n_iter':int(g.n_iter_),'lower_bound':float(g.lower_bound_),'warnings':[str(z.message) for z in caught]}
    return w,mu,var,meta

def local(cid,x,y):
    group=CTX['group'];cfg=CTX.get('dp');k=None
    if group=='C':
        b=CTX['b_run'];fork=json.loads((b/'fork.json').read_text())
        if fork['prepared_sha256']!=digest(CTX['prepared']):raise ValueError('B used different prepared state')
        meta=json.loads((b/('client_%02d.json'%cid)).read_text())
        if meta['input_sha256']!=array_hash(x):raise ValueError('B/C local representation/order mismatch')
        k=meta['k']
    w,m,v,meta=fit_mixture(x,y,group,cfg,k)
    meta.update(group=group,client_id=int(cid),input_sha256=array_hash(x),dp_config=cfg)
    write(CTX['out']/('client_%02d.json'%cid),meta)
    return w,m,v
