"""Local review implementation. CPU only; no server launching or automatic sweep."""
import argparse,json,sys,difflib,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
from protocol import validate_selection,validate_dp,instrument_model

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--group',choices=('prepare','A','B','C'),required=True)
    p.add_argument('--dataset',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--alpha',type=float,required=True,help='0.01 for Iris; 0.05 for BC/Banknote')
    p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--prepared',type=Path);p.add_argument('--dp-config',type=Path);p.add_argument('--b-run',type=Path)
    p.add_argument('--smoke',action='store_true',help='local engineering only: 2 pretrain / 3 rounds')
    a=p.parse_args();validate_selection(a.dataset,a.group)
    manifest=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
    for name,expected in manifest.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            p.error('frozen source changed: '+name)
    if a.seed not in range(5):p.error('protocol fixes seeds 0 through 4')
    expected_alpha=0.01 if a.dataset=='iris' else 0.05
    if a.alpha!=expected_alpha:p.error(f'{a.dataset} protocol fixes alpha={expected_alpha}')
    if a.group!='prepare' and a.prepared is None:p.error('--prepared is required')
    if a.group=='B' and a.dp_config is None:p.error('explicit --dp-config required; no research default')
    if a.group=='C' and a.b_run is None:p.error('--b-run required')
    if a.out.exists():p.error('output already exists')
    a.data_dir=a.data_dir.resolve();a.out=a.out.resolve()
    # Dataset-only path prevents an author model.py shadowing the execution copy.
    if (a.data_dir/'model.py').exists():p.error('use dataset/ directory, not author source root')
    sys.path.insert(0,str(ROOT/'vendor/audited_scripts'))
    from common import train as T
    import paired_runtime as R
    dp=validate_dp(json.loads(a.dp_config.read_text())) if a.dp_config else None
    if a.group=='C':
        b=json.loads((a.b_run/'experiment.json').read_text())
        if b['group']!='B':p.error('--b-run must be group B')
        if json.loads((a.b_run/'result.json').read_text())['status']!='ok':p.error('B did not complete')
        dp=validate_dp(b['dp_config'])
    R.CTX.update(group=a.group,out=a.out,prepared=a.prepared.resolve() if a.prepared else None,dp=dp,b_run=a.b_run.resolve() if a.b_run else None)
    original=T.instrument_main
    def inject(text,pretrain):
        model=a.out/'source/model.py';before=model.read_text();after=instrument_model(before);model.write_text(after)
        (a.out/'paired_model.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True))))
        return original(text,pretrain)
    T.instrument_main=inject
    args=T.build_parser().parse_args(['--source',str(ROOT/'vendor/author'),'--out',str(a.out),'--data-dir',str(a.data_dir),'--dataset',a.dataset,'--mode','author','--local-gmm','author','--device','cpu','--seed',str(a.seed),'--alpha',str(a.alpha),'--rounds','3' if a.smoke else '300','--pretrain','2' if a.smoke else '50','--save-every','300'])
    try:T._run_in_process(args)
    except SystemExit as e:
        if a.group!='prepare' or e.code not in (None,0) or not (a.out/'pre_gmm.pt').exists():raise
        result=json.loads((a.out/'result.json').read_text());result['status']='prepared';(a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    finally:
        if a.out.exists():
            R.write(a.out/'experiment.json',{'group':a.group,'dataset':a.dataset,'seed':a.seed,'alpha':a.alpha,'dp_config':dp,'smoke':a.smoke,'method_code_hashes':{q.name:R.digest(q) for q in sorted((ROOT/'scripts').glob('*.py'))},'note':'config.local_gmm=author names base transport; actual method is this group and paired_model.diff'})
if __name__=='__main__':main()
