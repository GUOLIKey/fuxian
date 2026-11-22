"""Inventory saved models and build a separate, explicit weight-initialization bundle."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

ROOT = Path('/root/autodl-tmp/fedvade_project')
OUT = ROOT / 'reproduction_runs/round8_checkpoint_finetune'
R7 = ROOT / 'reproduction_runs/round7_hyperparameter_search'
SOURCE = ROOT / 'fuxian-round2/FedVaDE_source'
sys.path.insert(0, str(ROOT / 'reproduction_runs/round6_checkpoint_diagnosis'))
from diagnose import PARAMS, VaDE, data, evaluate


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def atomic_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def progress(stage, current=0, total=0, checkpoint='', group='', epoch=None, budget=None,
             ari=None, purity=None, status='RUNNING'):
    atomic_json(OUT / 'progress.json', dict(stage=stage, checkpoint_current=current,
                checkpoint_total=total, best_checkpoint=checkpoint, current_group=group,
                epoch=epoch, budget=budget, latest_ari=ari, latest_purity=purity,
                status=status))


def read_state(path):
    payload = torch.load(path, map_location='cpu')
    if 'server' in payload:
        state = payload['server']
        epoch = payload['round']
        full = 'clients' in payload
    else:
        state = payload
        epoch = None
        full = False
    model = VaDE(PARAMS).cpu()
    model.load_state_dict(state, strict=True)
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise RuntimeError(f'Nonfinite model: {path}')
    return {k:v.detach().cpu().clone() for k,v in state.items()}, epoch, full


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'checkpoint_inventory.csv').exists():
        raise SystemExit('Inventory exists; refusing overwrite')
    candidates = []
    for stage in ('coarse', 'fine'):
        for run in sorted((R7 / stage).glob('*')):
            summary_file = run / 'evaluation_summary.json'
            if not summary_file.exists():
                continue
            summary = json.loads(summary_file.read_text())
            config_file = run / 'config.json'
            config = json.loads(config_file.read_text()) if config_file.exists() else json.loads((run/'reused_config.json').read_text())
            for entry in summary['snapshots']:
                candidates.append(dict(source=f'R7_{stage}_{run.name}', epoch=entry['epoch'],
                                       path=Path(entry['checkpoint']), ari=entry['ari_mean'],
                                       ari_std=entry['ari_std'], purity=entry['purity_mean'],
                                       purity_std=entry['purity_std'], config=config,
                                       run=str(run)))
    r6 = json.loads((ROOT/'reproduction_runs/round6_checkpoint_diagnosis/round6_findings.json').read_text())
    for arm in ('author','post_server'):
        run = ROOT/f'reproduction_runs/round5_broadcast/formal_{arm}_seed0'
        for epoch in (150,300):
            name=f'r5_{arm}_{epoch}'
            score=r6['summary'][name]['A_author']
            candidates.append(dict(source=f'R5_{arm}',epoch=epoch,path=run/f'round_{epoch}_full.pt',
                ari=score['ari_mean'],ari_std=score['ari_std'],purity=score['purity_mean'],
                purity_std=score['purity_std'],config=json.loads((run/'result.json').read_text())['params'],run=str(run)))
    inventory=[]
    for i,c in enumerate(candidates,1):
        progress('CHECKPOINT_SCAN',i,len(candidates),c['source'])
        path=c['path']
        if not path.is_file():
            raise RuntimeError(f'Ranked checkpoint missing: {path}')
        state, embedded_epoch, full=read_state(path)
        if embedded_epoch != c['epoch']:
            raise RuntimeError(f'Checkpoint epoch mismatch: {path}')
        inventory.append(dict(source=c['source'],epoch=c['epoch'],path=str(path.resolve()),
                              sha256=sha(path),bytes=path.stat().st_size,
                              strict_load=True,full_train_state=full,
                              ari_mean=c['ari'],ari_std=c['ari_std'],
                              purity_mean=c['purity'],purity_std=c['purity_std'],
                              config_path=str(Path(c['run'])/'config.json'),
                              inference_protocol='round6 A_author seeds 0-9',
                              selection='exploratory_label_selected'))
        del state
    write_csv(OUT/'checkpoint_inventory.csv',inventory)
    # Fine F01 is the canonical copy when its state ties coarse C14.
    rank_ari=sorted(range(len(inventory)),key=lambda i:(inventory[i]['ari_mean'],inventory[i]['source']=='R7_fine_F01'),reverse=True)
    rank_purity=sorted(range(len(inventory)),key=lambda i:(inventory[i]['purity_mean'],inventory[i]['source']=='R7_fine_F01'),reverse=True)
    for n,i in enumerate(rank_ari,1):inventory[i]['ari_rank']=n
    for n,i in enumerate(rank_purity,1):inventory[i]['purity_rank']=n
    write_csv(OUT/'checkpoint_ranking.csv',inventory)
    best=inventory[rank_ari[0]]
    if best['source'] != 'R7_fine_F01' or best['epoch'] != 100:
        raise RuntimeError(f'Unexpected best checkpoint; inspect before tuning: {best}')
    best_dir=OUT/'best_initial_checkpoint'
    best_dir.mkdir()
    original_state, epoch, _=read_state(Path(best['path']))
    torch.save({'round':epoch,'server':original_state,'source_path':best['path'],
                'source_sha256':best['sha256'],'selection':'exploratory_label_selected'},
               best_dir/'server.pt')
    (best_dir/'provenance.json').write_text(json.dumps(best,indent=2)+'\n')
    progress('CHECKPOINT_EVALUATION',len(candidates),len(candidates),best['source'])
    ft00=OUT/'FT00'
    ft00.mkdir()
    command=[str(ROOT/'.venv-cpu/bin/python'),str(OUT/'evaluate_saved.py'),
             '--run-dir',str(ft00),'--external-checkpoint',str(best_dir/'server.pt'),
             '--external-rounds',str(R7/'fine/F01/rounds.jsonl')]
    with (ft00/'evaluation.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    recheck=json.loads((ft00/'evaluation_summary.json').read_text())['last']
    if abs(recheck['ari_mean']-best['ari_mean'])>1e-8 or abs(recheck['purity_mean']-best['purity_mean'])>1e-8:
        raise RuntimeError('Selected checkpoint did not reproduce original ten-seed evaluation')
    progress('RESUME_VALIDATION',checkpoint=best['source'])
    # Reconstruct client-local GMMs with the author's own fitting rule, not a stateful resume.
    _,_,_,clients,sizes,_,_=data()
    client_states=[]; partition=[]
    for loader in clients:
        model=VaDE(PARAMS).cpu()
        model.load_state_dict(original_state,strict=True)
        means=[]; labels=[]
        with torch.no_grad():
            for x,y in loader:
                mu,_=model.encoder(x)
                means.append(mu.numpy());labels.append(y.numpy())
        latent=np.concatenate(means)
        y=np.concatenate(labels)
        components=len(np.unique(y))
        gmm=GaussianMixture(n_components=components,covariance_type='diag',random_state=42)
        gmm.fit(latent)
        model.rho_.data=torch.log(torch.tensor(gmm.weights_,dtype=torch.float32)+1e-10)
        model.mu_c.data=torch.tensor(gmm.means_,dtype=torch.float32)
        model.log_sigma2_c.data=torch.log(torch.tensor(gmm.covariances_,dtype=torch.float32))
        state={k:v.detach().clone() for k,v in model.state_dict().items()}
        if not all(torch.isfinite(v).all() for v in state.values()):
            raise RuntimeError('Nonfinite local GMM initialization')
        client_states.append(state)
        partition.append({'size':len(loader.dataset),'tensor_sha256':[
            hashlib.sha256(t.numpy().tobytes()).hexdigest() for t in loader.dataset.tensors]})
    bundle={'mode':'weight_initialization_not_exact_resume','server':original_state,
            'clients':client_states,'partition':partition,'source_checkpoint':best['path'],
            'source_sha256':best['sha256'],'source_epoch':100,
            'local_gmm_method':'author local GaussianMixture on encoder means, K=unique label count, seed 42',
            'optimizer_state':'new','scheduler_state':'new','rng_state':'new seed 0'}
    torch.save(bundle,OUT/'initial_bundle.pt')
    config=json.loads((R7/'fine/F01/config.json').read_text())
    rows=[json.loads(x) for x in (R7/'fine/F01/rounds.jsonl').read_text().splitlines()]
    current_lr=float(rows[99]['server_lr'])
    assert all(abs(x-current_lr)<1e-12 for x in rows[99]['client_lr'])
    plans=[dict(id='FT00',epochs=0,lr=0,alpha=config['alpha'],schedule=config['schedule'],sample=100,
                initialization='selected_server_checkpoint_no_training')]
    for i,ratio in enumerate((1,.5,.2,.1),1):
        plans.append(dict(id=f'FT{i:02d}',epochs=20,lr=current_lr*ratio,
                          lr_ratio=ratio,alpha=config['alpha'],schedule=config['schedule'],sample=100,
                          initialization='same_reconstructed_weight_initialization_bundle'))
    plans.append(dict(id='FT05',conditional=True,epochs=50,
                      rule='Only if a 20-epoch arm improves FT00 ARI by >0.005 without Purity dropping >0.002; 50-round replay from same initial bundle, not exact continuation of FT20.'))
    plans.append(dict(id='FT06',conditional=True,rule='Only if FT05 further improves by >0.005; one alpha=0.025 20-round arm from common bundle at the chosen LR.'))
    atomic_json(OUT/'finetune_configs.json',plans)
    atomic_json(OUT/'resume_audit.json',{
        'strict_resume_possible':False,'reason':'Epoch-100 checkpoint contains only server state_dict; no client, optimizer, scheduler, RNG state.',
        'weight_initialization_safe_to_smoke':True,'source_checkpoint':best['path'],
        'source_checkpoint_sha256':best['sha256'],'source_epoch':100,
        'initial_bundle_sha256':sha(OUT/'initial_bundle.pt'),
        'client_component_counts':[len(x['rho_']) for x in client_states],
        'client_partition_sizes':sizes,'current_server_lr_at_epoch100':current_lr,
        'repeated_evaluation_ari':recheck['ari_mean'],'repeated_evaluation_purity':recheck['purity_mean']})
    progress('RESUME_VALIDATION',checkpoint=best['source'],ari=recheck['ari_mean'],purity=recheck['purity_mean'])
    print(json.dumps({'candidate_count':len(inventory),'best':best['source'],'epoch':best['epoch'],
                      'ari':best['ari_mean'],'purity':best['purity_mean'],'current_lr':current_lr,
                      'strict_resume':False,'bundle':str(OUT/'initial_bundle.pt')}),flush=True)


if __name__=='__main__':
    main()
