import csv,hashlib,json,zipfile
from pathlib import Path
R=Path('/root/autodl-tmp/fedvade_project'); O=R/'reproduction_runs/round4_source_audit'; S=R/'fuxian-round2/FedVaDE_source'
z=zipfile.ZipFile(O/'author_delivery.zip')
files={}
for item in z.infolist():
 name=item.filename
 if name.startswith('FedVaDE_source/') and not item.is_dir() and not name.startswith('FedVaDE_source/__pycache__/') and not name.startswith('FedVaDE_source/.idea/') and not name.endswith('.DS_Store'):
  relative=name.removeprefix('FedVaDE_source/')
  p=S/relative
  files[relative]={'zip_sha256':hashlib.sha256(z.read(item)).hexdigest(),'working_sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None,'match':p.exists() and z.read(item)==p.read_bytes()}
round3={}
for kind in ('author','epoch'):
 p=R/f'fuxian-round3/formal/B_{kind}_seed1/source'
 round3[kind]={name: {'same_as_baseline':(p/name).exists() and (p/name).read_bytes()==(S/name).read_bytes()} for name in ('main.py','model.py','dataloader.py','experiment.py','dataset/breast_cancer.csv')}
summary={'zip_sha256':hashlib.sha256((O/'author_delivery.zip').read_bytes()).hexdigest(),'source_delivery_provenance':'user confirms extracted FedVaDE_source as author delivery; ZIP from WeChat received file directory','file_comparison':files,'third_round':round3,'paper_sha256':hashlib.sha256((O/'FedVaDE.pdf').read_bytes()).hexdigest(),'paper_pages':6,'appendix_in_pdf':False,'appendix_in_zip':False}
(O/'originals_comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
with (O/'paper_code_execution_matrix.originals.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['item','paper_evidence','author_code_evidence','historical_execution','status'])
 w.writerows([
 ['learning rate','p4 IV-A 0.002; ×0.85 every 10 epochs','main.py:42-44,123-128,147 client per mini-batch','author and epoch modes, five seeds each','MISMATCH'],
 ['VAE structure','p4 IV-A 500,500,2000 encoder; 2000,500,500 decoder; z=10','model.py:205-235','same source hash','MATCH'],
 ['client number','p4 IV-A ten','main.py:39','ten','MATCH'],
 ['pretraining epochs','not specified in six-page PDF or available appendix','main.py:120 pre_epoch=50','50','UNDOCUMENTED'],
 ['pretraining loss','not specified in six-page PDF','model.py:125 MSE','same','UNDOCUMENTED'],
 ['local GMM','p3 Eq9-10 MoG; fitting details absent','model.py:166-183 GMM on encoded means, true-label class count, seed42','same','UNDOCUMENTED'],
 ['global GMM','p3 Eq11-13 all local components weighted by sample count; p4 server samples M=100','model.py:48-101 sampling; 192-200 global GMM fit','same','UNKNOWN'],
 ['aggregation','p3 Eq12 sample-size weighted VAE','model.py:30-45 exclude GMM; main.py:153-155','same','MATCH'],
 ['broadcast','p4 end III-C updated global encoder and decoder to clients','main.py:212-213 sends pre-server-update average','same','MISMATCH'],
 ['reconstruction','p3 Eq5-8 Gaussian observation log-likelihood','model.py:284 and 328 MSE with deterministic decoder','same','MISMATCH'],
 ['alignment','p4 Eq15 local-global decoder discrepancy','model.py:316-317 MSE','same','MATCH'],
 ['Table II ARI','p5 Table II BC 0.635, ten-run mean per p4','main.py:46 300 rounds; no save; loads provided weight if present','five seed retrain; saved final by wrapper','UNKNOWN'],
 ])
print(sum(v['match'] for v in files.values()),len(files),round3)
