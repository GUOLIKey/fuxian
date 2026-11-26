"""Source hooks and strict selection validation; no numerical imports."""
import math
DATASETS=('iris','breast_cancer','banknote_authentication')
def once(s,old,new):
    if s.count(old)!=1:raise ValueError('source hook mismatch: '+old[:70])
    return s.replace(old,new)
def validate_selection(dataset,group):
    if dataset not in DATASETS or group not in ('prepare','A','B','C'):raise ValueError('unsupported dataset/group')
def validate_dp(c):
    required={'upper_bound','prior','prune','fit_seed','max_iter','tol','reg_covar'}
    if set(c)!=required:raise ValueError('DP config must explicitly specify '+str(sorted(required)))
    for k in ('upper_bound','max_iter','fit_seed'):
        if type(c[k]) is not int or c[k] < (0 if k=='fit_seed' else 1):raise ValueError(k)
    for k in ('prior','tol','reg_covar'):
        if not math.isfinite(c[k]) or c[k]<=0:raise ValueError(k)
    if not math.isfinite(c['prune']) or not 0<=c['prune']<1:raise ValueError('prune')
    return c

def instrument_model(s):
    s=once(s,'    Loss = nn.MSELoss()','    from paired_runtime import restore as _paired_restore, boundary as _paired_boundary, local as _paired_local\n    _paired_restored = _paired_restore(server, clients)\n    Loss = nn.MSELoss()')
    s=once(s,'tqdm(range(pre_epoch))','tqdm(range(0 if _paired_restored else pre_epoch))')
    marker="    for c in clients:\n        c['vade'].encoder.log_sigma2_l.load_state_dict"
    s=once(s,marker,"    _paired_boundary(server, clients)\n\n"+marker)
    start="            K_l = len(np.unique(local_y))"
    end="            clients_gmm_param.append({"
    if s.count(start)!=1 or s.count(end)!=1:raise ValueError('local block mismatch')
    a=s.index(start);b=s.index(end,a)
    replacement="""            _w, _m, _v = _paired_local(c['client_id'], local_mu, local_y)
            c['vade'].rho_.data = torch.log(torch.tensor(_w, dtype=torch.float, device=device) + epsilon)
            c['vade'].mu_c.data = torch.tensor(_m, dtype=torch.float, device=device)
            c['vade'].log_sigma2_c.data = torch.log(torch.tensor(_v, dtype=torch.float, device=device))

"""
    return s[:a]+replacement+s[b:]
