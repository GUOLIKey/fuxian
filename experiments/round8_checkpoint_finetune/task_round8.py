"""Bounded exploratory FedVaDE weight-initialized fine-tuning queue."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path('/root/autodl-tmp/fedvade_project')
HERE = ROOT / 'reproduction_runs/round8_checkpoint_finetune'
SOURCE = ROOT / 'fuxian-round2/FedVaDE_source'
PY = ROOT / '.venv-cpu/bin/python'


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def progress(stage, group='', epoch=None, budget=None, ari=None, purity=None, status='RUNNING'):
    old = json.loads((HERE/'progress.json').read_text()) if (HERE/'progress.json').exists() else {}
    atomic_json(HERE / 'progress.json', dict(stage=stage, checkpoint_current=None,
                checkpoint_total=None, best_checkpoint=old.get('best_checkpoint','F01_epoch100'),
                best_ari_so_far=old.get('best_ari_so_far'),
                current_group=group, epoch=epoch, budget=budget,
                latest_ari=ari, latest_purity=purity, status=status,
                updated_at=time.time()))


def execute(argv, logfile):
    with logfile.open('w') as output:
        subprocess.run([str(a) for a in argv], cwd=HERE, stdout=output,
                       stderr=subprocess.STDOUT, check=True,
                       env={**os.environ, 'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
                            'OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})


def train_and_evaluate(plan):
    group = plan['id']
    out = HERE / group
    if out.exists():
        raise RuntimeError(f'Existing result directory: {out}')
    progress('FINETUNE', group, 0, plan['epochs'])
    execute([PY, HERE/'round8_runner.py', '--source',SOURCE,
             '--initial-bundle',HERE/'initial_bundle.pt','--out',out,
             '--lr',plan['lr'],'--alpha',plan['alpha'],
             '--schedule',plan['schedule'],'--sample',plan['sample'],
             '--rounds',plan['epochs'],'--save-every',5,'--seed',0],
            HERE/f'{group}.log')
    progress('FINAL_EVALUATION', group, plan['epochs'], plan['epochs'])
    execute([PY,HERE/'evaluate_saved.py','--run-dir',out],out/'evaluation.log')
    summary=json.loads((out/'evaluation_summary.json').read_text())
    chosen=summary['best_exploratory_ari']
    current=json.loads((HERE/'progress.json').read_text())
    if current.get('best_ari_so_far') is None or chosen['ari_mean']>current['best_ari_so_far']:
        current['best_ari_so_far']=chosen['ari_mean']
        current['best_checkpoint']=f"{group}_epoch{chosen['epoch']}"
        atomic_json(HERE/'progress.json',current)
    print(group,'last',summary['last']['ari_mean'],'best',summary['best_exploratory_ari']['ari_mean'],flush=True)
    return summary


def aggregate(groups):
    metrics=[];draws=[]
    for group in groups:
        folder=HERE/group
        summary=json.loads((folder/'evaluation_summary.json').read_text())
        for row in summary['snapshots']:
            metrics.append({'group':group,'selection':'exploratory_label_selected',**row})
        with (folder/'inference.csv').open() as stream:
            for row in csv.DictReader(stream):
                draws.append({'group':group,**row})
    for filename,rows in (('finetune_metrics.csv',metrics),('inference_repeats.csv',draws)):
        with (HERE/filename).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    return metrics


def final_report(plans, groups, decisions, metrics):
    baseline=json.loads((HERE/'FT00/evaluation_summary.json').read_text())['last']
    choices=[row for row in metrics if row['group']!='FT00']
    top=max(choices,key=lambda x:x['ari_mean']) if choices else None
    best_dir=HERE/'best_finetuned_checkpoint'
    best_dir.mkdir(exist_ok=False)
    if top:
        src=Path(top['checkpoint'])
        if not src.is_file():
            raise RuntimeError(f'Best selected checkpoint missing: {src}')
        dest=best_dir/'server.pt'
        shutil.copy2(src,dest)
        digest=hashlib.sha256(dest.read_bytes()).hexdigest()
        atomic_json(best_dir/'provenance.json',{'source':str(src),'copied_path':str(dest),
                    'sha256':digest,'group':top['group'],'epoch':top['epoch'],
                    'ari_mean':top['ari_mean'],'purity_mean':top['purity_mean'],
                    'selection':'exploratory_label_selected'})
    lines=['# FedVaDE Round 8 weight-initialized fine-tuning','',
           'This is not exact continuation from Round 7 epoch 100: its checkpoint has only the server state. Local client GMMs were refitted in that latent space with the author method; all optimizers, schedulers and RNGs were newly initialized and shared across arms. No author source or prior checkpoint was modified.','',
           'All ARI-selected checkpoints use the same evaluation labels; these values are exploratory, not independent test performance. Ten inference seeds are repeats of one fixed model, not ten training seeds.','',
           '| Group | LR | Alpha | Epoch | ARI mean±sd | Purity mean±sd |',
           '|---|---:|---:|---:|---:|---:|']
    plan_map={p['id']:p for p in plans if 'lr' in p}
    for row in metrics:
        plan=plan_map.get(row['group'],{})
        lines.append(f"| {row['group']} | {plan.get('lr','--')} | {plan.get('alpha','--')} | {row['epoch']} | {row['ari_mean']:.4f}±{row['ari_std']:.4f} | {row['purity_mean']:.4f}±{row['purity_std']:.4f} |")
    lines += ['',f"FT00 reference: ARI {baseline['ari_mean']:.4f}±{baseline['ari_std']:.4f}, Purity {baseline['purity_mean']:.4f}±{baseline['purity_std']:.4f}.",
              'Conditional decisions: '+json.dumps(decisions,ensure_ascii=False),
              'Author supplied weight reference (Round 6): ARI approximately 0.6295 under the same author-protocol evaluation. Checkpoint selection on current labels biases comparison. Generation quality was not measured.','',
              'For a strict evaluation, construct train/validation/test splits before model or hyperparameter selection, fit preprocessing on training data only, and report performance once on untouched test records across multiple training seeds.','']
    (HERE/'ROUND8_REPORT.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    plans=json.loads((HERE/'finetune_configs.json').read_text())
    audit=json.loads((HERE/'resume_audit.json').read_text())
    if audit['strict_resume_possible'] or not (HERE/'initial_bundle.pt').exists():
        raise RuntimeError('Unexpected resume audit or missing initial bundle')
    smoke=json.loads((HERE/'smoke_weight_init/result.json').read_text())
    if smoke.get('status')!='ok' or smoke['last']['round']!=2 \
       or not (HERE/'smoke_weight_init/evaluation_summary.json').exists():
        raise RuntimeError('Smoke test did not pass training and evaluation gates')
    if args.dry_run:
        print(json.dumps({'base_lr':audit['current_server_lr_at_epoch100'],
              'groups':[(p['id'],p.get('lr'),p.get('epochs')) for p in plans]},indent=2))
        return
    if (HERE/'task_exit_status.json').exists() or (HERE/'FT01').exists():
        raise RuntimeError('Task appears previously launched')
    atomic_json(HERE/'stage_history.json',{'CHECKPOINT_SCAN':'143 files strictly loaded',
                'CHECKPOINT_EVALUATION':'FT00 ten-seed score reproduced',
                'RESUME_VALIDATION':'exact resume impossible; common weight-initialization bundle validated',
                'SMOKE_TEST':'2 rounds and saved-checkpoint evaluation passed'})
    groups=['FT00'];decisions=[]
    baseline=json.loads((HERE/'FT00/evaluation_summary.json').read_text())['last']
    prior=json.loads((HERE/'progress.json').read_text())
    prior['best_ari_so_far']=baseline['ari_mean']
    prior['best_checkpoint']='F01_epoch100'
    atomic_json(HERE/'progress.json',prior)
    try:
        first=[]
        for plan in plans[1:5]:
            summary=train_and_evaluate(plan)
            groups.append(plan['id'])
            first.append((plan,summary['best_exploratory_ari']))
        eligible=[x for x in first if x[1]['ari_mean']>baseline['ari_mean']+.005
                  and x[1]['purity_mean']>=baseline['purity_mean']-.002]
        if eligible:
            winner=max(eligible,key=lambda x:x[1]['ari_mean'])
            plan=dict(id='FT05',lr=winner[0]['lr'],alpha=winner[0]['alpha'],
                      schedule=winner[0]['schedule'],sample=100,epochs=50,
                      initialization='replay_from_common_bundle_not_FT20_resume')
            plans[5]=plan
            atomic_json(HERE/'finetune_configs.json',plans)
            summary=train_and_evaluate(plan)
            groups.append('FT05')
            decisions.append({'FT05':'run','basis':winner[0]['id']})
            if summary['best_exploratory_ari']['ari_mean']>winner[1]['ari_mean']+.005:
                plan2=dict(id='FT06',lr=winner[0]['lr'],alpha=.025,
                           schedule=winner[0]['schedule'],sample=100,epochs=20,
                           initialization='same_common_bundle_alpha_probe')
                plans[6]=plan2
                atomic_json(HERE/'finetune_configs.json',plans)
                train_and_evaluate(plan2)
                groups.append('FT06')
                decisions.append({'FT06':'run','basis':'FT05 gained >0.005 ARI over best 20-round arm'})
            else:
                decisions.append({'FT06':'skipped','reason':'FT05 did not exceed best FT20 ARI by >0.005'})
        else:
            decisions.append({'FT05':'skipped','reason':'No FT01-FT04 checkpoint met the prespecified improvement/purity gate'})
            decisions.append({'FT06':'skipped','reason':'FT05 not run'})
        progress('FINAL_EVALUATION','all')
        metrics=aggregate(groups)
        final_report(plans,groups,decisions,metrics)
        progress('FINAL_EVALUATION','all',status='RUNNING')
        print('REPORT_READY',HERE/'ROUND8_REPORT.md',flush=True)
    except BaseException:
        (HERE/'error.txt').write_text(traceback.format_exc())
        progress('FAILED',status='FAILED')
        raise


if __name__=='__main__':
    main()
