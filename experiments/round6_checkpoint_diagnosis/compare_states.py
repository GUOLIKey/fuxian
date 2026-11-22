import json, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/root/autodl-tmp/fedvade_project')
OUT=ROOT/'reproduction_runs/round6_checkpoint_diagnosis'
SOURCE=ROOT/'fuxian-round2/FedVaDE_source'
sys.path.insert(0,str(SOURCE))
from model import VaDE
from argparse import Namespace
from sklearn.preprocessing import StandardScaler
import pandas as pd
torch.set_num_threads(1)
args=Namespace(input_dim=31,hid_dim=10,nClusters=2,alpha=.01,cuda=False)
raw=pd.read_csv(SOURCE/'dataset/breast_cancer.csv')
X=torch.tensor(StandardScaler().fit_transform(raw.iloc[:,:-1].to_numpy()),dtype=torch.float32)
rows={}
for arm in ('author','post_server'):
 for rnd in (150,300):
  path=ROOT/f'reproduction_runs/round5_broadcast/formal_{arm}_seed0/round_{rnd}_full.pt'
  ck=torch.load(path,map_location='cpu')
  server=ck['server']
  keys=[k for k in server if k not in ('rho_','mu_c','log_sigma2_c')]
  server_model=VaDE(args); server_model.load_state_dict(server);server_model.eval()
  with torch.no_grad(): server_mu=server_model.encoder(X)[0]
  param_deltas=[];latent_deltas=[];client_masses=[]
  for state in ck['clients']:
   param_deltas.append(max((state[k]-server[k]).abs().max().item() for k in keys))
   model=VaDE(Namespace(input_dim=31,hid_dim=10,nClusters=len(state["rho_"]),alpha=.01,cuda=False));model.load_state_dict(state);model.eval()
   with torch.no_grad(): mu=model.encoder(X)[0]
   latent_deltas.append(float(torch.sqrt(torch.mean((mu-server_mu)**2))))
   client_masses.append(torch.softmax(state['rho_'],0).tolist())
  rows[f'{arm}_{rnd}']={'server_client_vae_max_abs_min':min(param_deltas),
                       'server_client_vae_max_abs_max':max(param_deltas),
                       'server_client_latent_mu_rms_mean':float(np.mean(latent_deltas)),
                       'client_gmm_weights':client_masses,
                       'server_gmm_weights':torch.softmax(server['rho_'],0).tolist(),
                       'optimizer_state_lengths':[len(o['state']) for o in ck['client_optimizers']]}
  del ck,server_model
p2=torch.load(ROOT/'fuxian-round2/formal/B_author_seed0/round_300_server.pt',map_location='cpu')['server']
p5=torch.load(ROOT/'reproduction_runs/round5_broadcast/formal_author_seed0/round_300_server.pt',map_location='cpu')['server']
rows['round2_author_seed0_vs_round5_A300']={'same_keys':list(p2)==list(p5),'max_tensor_abs_diff':max((p2[k]-p5[k]).abs().max().item() for k in p2)}
(OUT/'client_server_diagnostics.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps({k:{x:v for x,v in row.items() if x!='client_gmm_weights'} for k,row in rows.items()},indent=2))
