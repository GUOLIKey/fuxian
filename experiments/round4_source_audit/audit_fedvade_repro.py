import ast,csv,difflib,hashlib,json,os,platform,subprocess,sys,zipfile
from pathlib import Path
from statistics import mean,pstdev
R=Path('/root/autodl-tmp/fedvade_project'); O=R/'reproduction_runs/round4_source_audit'; S=R/'fuxian-round2/FedVaDE_source'; F=R/'fuxian-round3/formal'; G=R/'fuxian-round2/formal'; M=R/'reports/FEDVADE_MIGRATION_FILES.tsv'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return json.loads(p.read_text())
def save(name,obj): (O/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+'\n')
source={str(p.relative_to(S)):sha(p) for p in S.rglob('*') if p.is_file() and (p.suffix in ('.py','.csv','.pk','.md'))}
zipfiles=[p for p in R.glob('**/*.zip') if 'FedVaDE' in p.name or 'fedvade' in p.name.lower()]
archives={str(p):{'sha256':sha(p),'members':[{'name':m.filename,'sha256':hashlib.sha256(zipfile.ZipFile(p).read(m)).hexdigest()} for m in zipfile.ZipFile(p).infolist() if m.filename.endswith(('.py','.json','.yaml','.yml','.ini','.toml','.cfg'))]} for p in zipfiles}
original={p.name:p for p in S.glob('*.py')}; tree={n:ast.parse(p.read_text()) for n,p in original.items()}
functions={n:[{'name':x.name,'line':x.lineno,'end':x.end_lineno} for x in t.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))] for n,t in tree.items()}
checks={}
for label,folder in [('r2_author',G/'B_author_seed0'),('r2_epoch',G/'B_epoch_seed0'),('r3_author',F/'B_author_seed1'),('r3_epoch',F/'B_epoch_seed1')]:
 d=load(folder/'result.json'); src=folder/'source'; changed={}
 for name,p in original.items():
  q=src/name
  if q.exists() and sha(p)!=sha(q):
   diff=''.join(difflib.unified_diff(p.read_text().splitlines(True),q.read_text().splitlines(True),fromfile='source/'+name,tofile=label+'/'+name))
   (O/'source_diff'/f'{label}_{name}.diff').write_text(diff)
   changed[name]={'diff':f'source_diff/{label}_{name}.diff','functions':[x for x in functions[name] if any(x['line']<=k<=x['end'] for k in range(1,len(q.read_text().splitlines())+1) if k<=len(p.read_text().splitlines()) and p.read_text().splitlines()[k-1]!=q.read_text().splitlines()[k-1])]}
 checks[label]={'command':d.get('command'),'author_argv':d.get('author_argv'),'source_hashes_match':all(d['source_hashes'].get(n)==sha(p) for n,p in original.items()),'changed':changed,'status':d.get('status'),'partition':d.get('partition'),'metrics':{k:d.get('last',{}).get(k) for k in ('purity','ari')},'rounds':d.get('params',{}).get('rounds')}
manifest=[]
with M.open() as f:
 for row in csv.reader(f,delimiter='\t'):
  if len(row)>=7 and row[1].startswith(str(R)) and Path(row[1]).exists() and any(key in row[1] for key in ('FedVaDE_source/','/formal/B_')): manifest.append((row[1],row[6],sha(Path(row[1]))))
matched=sum(a==b for _,a,b in manifest)
code=(S/'main.py').read_text().splitlines(); scheduler=[{'line':i+1,'text':v.strip()} for i,v in enumerate(code) if 'StepLR(' in v or "['lr_s'].step()" in v or "['optimizer'].step()" in v]
rows=[]
for label,base in [('author','B_author_seed'),('epoch','B_epoch_seed')]:
 for seed in range(5):
  folder=(G if seed==0 else F)/f'{base}{seed}'; d=load(folder/'result.json'); trace=[json.loads(s) for s in (folder/'rounds.jsonl').read_text().splitlines()]; rows.append({'schedule':label,'seed':seed,'ari':d['last']['ari'],'purity':d['last']['purity'],'ari_150':d['round_150']['ari'],'rounds':len(trace),'batch_counts':[len(x) for x in trace[0]['client_batch_lrs']],'lr_round5':trace[4]['client_end_lrs'],'lr_round10':trace[9]['client_end_lrs'],'partition_sha':hashlib.sha256(json.dumps(d['partition'],sort_keys=True).encode()).hexdigest()})
with (O/'paper_code_execution_matrix.csv').open('w') as f:
 w=csv.writer(f); w.writerow(['item','paper','author_source','historical_execution','status'])
 for item,paper,auth,hist,status in [
 ('initial_lr','0.002 IV-A','main.py:42 0.002','0.002','MATCH'),('latent_dim','10 IV-A','main.py:37 10','10','MATCH'),('client_count','10 IV-A','main.py:39 10','10','MATCH'),('training_rounds','Table II unspecified; Fig5 150','main.py:46 300','300','UNKNOWN'),('pretrain','undocumented','main.py:120 50 MSE','50','UNDOCUMENTED'),('batch_size','undocumented','main.py:33 200','200','UNDOCUMENTED'),('scheduler','factor .85 each 10 epochs IV-A','main.py:124,128,147 client per batch','author and epoch variants','MISMATCH'),('server_sample','100 IV-A','main.py:49 100','100','MATCH'),('alpha','0.01 IV-A','main.py:48 0.01','0.01','MATCH'),('client_partition','class-informed IV-A','dataloader.py:9-44 class subset, seed42, overlap','partition hash recorded','MATCH'),('global_broadcast','updated server model III-C','main.py:212-213 pre-server-update average','same','MISMATCH'),('input_dim','31 Table I','dataloader.py:14 all 31 CSV columns','31','MATCH'),('dataset_source','UCI IV-A','dataset/breast_cancer.csv','same hash','MATCH'),('preprocessing','undocumented','dataloader.py:16 StandardScaler full data','same','UNDOCUMENTED'),('GMM_components','target 2 Table I','model.py:176 true-label local k','same','MISMATCH'),('GMM_init','undocumented','model.py:177,194 sklearn GMM seed42','same','UNDOCUMENTED'),('ELBO_reconstruction','Gaussian log likelihood Eq5-8','model.py:284 MSE','same','MISMATCH'),('alignment','decoder MSE Eq15','model.py:317 MSE','same','MATCH'),('checkpoint_selection','Table II unspecified','main.py:84 loads existing file; no save','300-round checkpoint by wrapper','UNKNOWN'),('evaluation','Purity and ARI IV-A','main.py:220-230 concatenated client predictions','same','MATCH'),('prediction','mixture responsibility Eq9','model.py:260-271 samples z','same','MATCH')]: w.writerow([item,paper,auth,hist,status])
save('audit_findings.json',{'zip':archives or 'MISSING','paper_pdf':'MISSING','source_hashes':source,'migration_hashes':{'checked':len(manifest),'matched':matched},'functions':functions,'executions':checks,'scheduler':scheduler,'runs':rows,'aggregate':{x:{'ari_mean':mean(r['ari'] for r in rows if r['schedule']==x),'ari_std_population':pstdev(r['ari'] for r in rows if r['schedule']==x)} for x in ('author','epoch')},'manual_review_required':['original ZIP unavailable','paper PDF and appendix unavailable on new server','historical import runtime cannot be proven from logs alone']})
env={'host':platform.node(),'python':sys.version,'platform':platform.platform(),'executable':sys.executable,'cuda_toolkit':subprocess.getoutput('nvcc --version'),'nvidia':subprocess.getoutput('nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader'),'packages':subprocess.getoutput('/root/autodl-tmp/fedvade_project/.venv-cpu/bin/python -m pip show torch numpy scikit-learn scipy pandas'),'pythonpath':os.environ.get('PYTHONPATH')}
save('environment_manifest.json',env)
print(json.dumps({'zip_count':len(archives),'migration_checked':len(manifest),'migration_matched':matched,'author_mean':mean(r['ari'] for r in rows if r['schedule']=='author'),'epoch_mean':mean(r['ari'] for r in rows if r['schedule']=='epoch'),'source_matches':{k:v['source_hashes_match'] for k,v in checks.items()}},indent=2))
import numpy as np,pandas as pd,torch
from collections import Counter
D=pd.read_csv(S/'dataset/breast_cancer.csv')
labels=D.iloc[:,-1]
x=D.iloc[:,:-1]
y=labels.to_numpy()
assert D.shape==(569,32)
sys.path.insert(0,str(S))
from dataloader import make_noniid_clients
_,dl,_,sizes=make_noniid_clients(S/'dataset/breast_cancer.csv',10,200,'breast_cancer')
parts=[(np.asarray(i.dataset.tensors[0]),np.asarray(i.dataset.tensors[1])) for i in dl]
records=np.concatenate([i[0] for i in parts])
unique=len({tuple(v) for v in records})
save('data_audit.json',{'shape':D.shape,'sha256':sha(S/'dataset/breast_cancer.csv'),'sizes':sizes,'client_records':len(records),'unique_scaled_feature_vectors':unique,'duplicates_beyond_first':len(records)-unique,'classes':dict(Counter(map(int,y))),'A1_unique':int(x.iloc[:,0].nunique()),'feature_columns':list(x.columns),'scale_method':'StandardScaler.fit_transform on all rows','label_used_for_partition':True,'label_used_for_local_gmm_component_count':True,'label_column_in_features':False})
paths={'author':S/'model/server_model_breast_cancer.pk','retrained_seed0':G/'B_author_seed0/round_300_server.pt','retrained_seed1':F/'B_author_seed1/round_300_server.pt'}
states={}
summaries={}
for name,path in paths.items():
 obj=torch.load(path,map_location='cpu',weights_only=True)
 state=obj.get('server',obj)
 states[name]=state
 weights=torch.softmax(state['rho_'],0).numpy()
 variance=torch.exp(state['log_sigma2_c']).numpy()
 summaries[name]={'sha256':sha(path),'bytes':path.stat().st_size,'keys':list(state),'shapes':{k:list(v.shape) for k,v in state.items()},'nonfinite':{k:int((~torch.isfinite(v)).sum()) for k,v in state.items()},'mixture_weights':weights.tolist(),'mu_c':state['mu_c'].tolist(),'variance_min_max':[float(variance.min()),float(variance.max())],'checkpoint_metadata':list(obj.keys()) if isinstance(obj,dict) else []}
for name in ('retrained_seed0','retrained_seed1'):
 summaries[name]['parameter_relative_l2_to_author']={k:float(torch.linalg.vector_norm(states[name][k]-states['author'][k])/torch.linalg.vector_norm(states['author'][k])) for k in states['author']}
save('checkpoint_audit.json',summaries)
