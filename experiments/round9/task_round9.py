"""Bounded Round 9 experiment queue and report generation; run only on the server."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path('/root/autodl-tmp/fedvade_project')
HERE = ROOT / 'reproduction_runs/round9_fedvade_optimization'
SOURCE = ROOT / 'fuxian-round2/FedVaDE_source'
R7 = ROOT / 'reproduction_runs/round7_hyperparameter_search'
PYTHON = ROOT / '.venv-cpu/bin/python'
REFERENCE = 0.6140076504228672
AUTHOR_WEIGHT = 0.6295


def atomic_json(path, data):
    temp = Path(str(path) + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def csv_write(path, rows, columns):
    temp = Path(str(path) + '.tmp')
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def base(id, stage, **changes):
    value = dict(id=id, stage=stage, lr=.002, alpha=.05, schedule='epoch', gamma=.85,
                 stepsize=10, pretrain=50, gmm_seed=42, gmm_n_init=1,
                 gmm_init='kmeans', sample=100)
    value.update(changes)
    return value


def plan():
    initialization = [
        base('I01', 'GMM_INIT_SEARCH', reused='Round7 F01 epoch150, author GMM'),
        base('I02', 'GMM_INIT_SEARCH', gmm_seed=7),
        base('I03', 'GMM_INIT_SEARCH', gmm_seed=123),
        base('I04', 'GMM_INIT_SEARCH', gmm_n_init=5),
        base('I05', 'GMM_INIT_SEARCH', gmm_n_init=10, gmm_init='random'),
        base('I06', 'PRETRAIN_SEARCH', pretrain=30),
        base('I07', 'PRETRAIN_SEARCH', pretrain=75),
        base('I08', 'PRETRAIN_SEARCH', pretrain=100),
        base('I09', 'GMM_INIT_SEARCH', gmm_n_init=5, gmm_init='k-means++')]
    hyper = [
        base('H01', 'HYPERPARAM_SEARCH', alpha=.02),
        base('H02', 'HYPERPARAM_SEARCH', alpha=.03),
        base('H03', 'HYPERPARAM_SEARCH', alpha=.075),
        base('H04', 'HYPERPARAM_SEARCH', lr=.0015),
        base('H05', 'HYPERPARAM_SEARCH', lr=.0008),
        base('H06', 'HYPERPARAM_SEARCH', lr=.0005),
        base('H07', 'HYPERPARAM_SEARCH', lr=.0003),
        base('H08', 'HYPERPARAM_SEARCH', gamma=.95)]
    return initialization, hyper


def progress(**changes):
    path = HERE / 'progress.json'
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(changes, updated_at=time.time())
    atomic_json(path, old)


def launch(config, rounds, category, position, total, seed=0):
    name = config['id'] if seed == 0 else f"{config['id']}_seed{seed}"
    out = HERE / category / name
    if out.exists():
        raise RuntimeError(f'Refusing existing experiment directory {out}')
    stage = ('MULTISEED_VALIDATION' if seed else
             'EXTENDED_TRAINING' if rounds == 300 else config['stage'])
    progress(stage=stage, current_config=name, config_current=position,
             config_total=total, epoch=0, epoch_total=rounds, latest_ari=None,
             latest_purity=None, status='RUNNING')
    argv = [str(PYTHON), str(HERE / 'round9_runner.py'), '--source', str(SOURCE),
            '--out', str(out), '--lr', str(config['lr']), '--alpha', str(config['alpha']),
            '--schedule', config['schedule'], '--gamma', str(config['gamma']),
            '--stepsize', str(config['stepsize']), '--pretrain', str(config['pretrain']),
            '--gmm-seed', str(config['gmm_seed']), '--gmm-n-init', str(config['gmm_n_init']),
            '--gmm-init', config['gmm_init'], '--sample', str(config['sample']),
            '--rounds', str(rounds), '--save-every', '10', '--seed', str(seed)]
    with (HERE / f'{name}_{rounds}.log').open('w') as stream:
        subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, check=True)
    with (HERE / f'{name}_{rounds}_evaluation.log').open('w') as stream:
        subprocess.run([str(PYTHON), str(HERE / 'evaluate_saved.py'), '--run-dir', str(out)],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    return out, json.loads((out / 'evaluation_summary.json').read_text())


def referenced_baseline():
    run = R7 / 'fine/F01'
    result = json.loads((run / 'evaluation_summary.json').read_text())
    snapshot = next(row for row in result['snapshots'] if row['epoch'] == 150)
    if not Path(snapshot['checkpoint']).is_file():
        raise RuntimeError('Round 7 F01 epoch150 checkpoint missing')
    return snapshot


def row_for(config, category, summary=None, out=None, reused=False):
    if reused:
        point = referenced_baseline()
        return dict(id=config['id'], category=category, stage=config['stage'],
                    seed=0, epochs=150, lr=config['lr'], alpha=config['alpha'],
                    pretrain=50, gmm_seed=42, gmm_n_init=1, gmm_init='kmeans',
                    gamma=.85, sample=100, fixed_ari=point['ari_mean'],
                    fixed_purity=point['purity_mean'], best_ari=point['ari_mean'],
                    best_epoch=150, best_checkpoint=point['checkpoint'],
                    unlabeled_epoch=None, unlabeled_ari=None,
                    init_reconstruction_mse=None, init_gmm_loglik=None,
                    init_gmm_bic=None,
                    source='reused_Round7_F01_epoch150', status='reused')
    best = summary['best_exploratory_ari']
    unlabeled = summary['best_unlabeled_loss']
    initial = json.loads((out / 'initialization.json').read_text())
    return dict(id=config['id'], category=category, stage=config['stage'], seed=summary['training_seed'],
                epochs=summary['last']['epoch'], lr=config['lr'], alpha=config['alpha'],
                pretrain=config['pretrain'], gmm_seed=config['gmm_seed'],
                gmm_n_init=config['gmm_n_init'], gmm_init=config['gmm_init'],
                gamma=config['gamma'], sample=config['sample'],
                fixed_ari=summary['last']['ari_mean'], fixed_purity=summary['last']['purity_mean'],
                best_ari=best['ari_mean'], best_epoch=best['epoch'],
                best_checkpoint=best['checkpoint'], unlabeled_epoch=unlabeled['epoch'],
                unlabeled_ari=unlabeled['ari_mean'],
                init_reconstruction_mse=initial['reconstruction_mse_weighted'],
                init_gmm_loglik=initial['gmm_loglik_weighted'],
                init_gmm_bic=initial['gmm_bic_total'],
                source=str(out), status='ok')


COLS = ['id', 'category', 'stage', 'seed', 'epochs', 'lr', 'alpha', 'pretrain',
        'gmm_seed', 'gmm_n_init', 'gmm_init', 'gamma', 'sample', 'fixed_ari',
        'fixed_purity', 'best_ari', 'best_epoch', 'best_checkpoint',
        'unlabeled_epoch', 'unlabeled_ari', 'init_reconstruction_mse',
        'init_gmm_loglik', 'init_gmm_bic', 'source', 'status']


def save_tables(rows):
    csv_write(HERE / 'initialization_comparison.csv',
              [r for r in rows if r['category'] == 'initialization'], COLS)
    csv_write(HERE / 'hyperparameter_comparison.csv',
              [r for r in rows if r['category'] == 'hyperparameter'], COLS)
    csv_write(HERE / 'multiseed_validation.csv',
              [r for r in rows if r['category'] == 'multiseed'], COLS)


def gather(rows):
    metrics, draws, inventory = [], [], []
    for record in rows:
        if record['status'] == 'reused':
            continue
        run = Path(record['source'])
        summary_path = run / 'evaluation_summary.json'
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text())
        for point in summary['snapshots']:
            item = {'config': record['id'], 'category': record['category'],
                    'training_seed': record['seed'], **point}
            metrics.append(item)
            checkpoint = Path(point['checkpoint'])
            inventory.append(dict(config=record['id'], training_seed=record['seed'],
                                  epoch=point['epoch'], path=str(checkpoint),
                                  sha256=sha(checkpoint), size_bytes=checkpoint.stat().st_size,
                                  checkpoint_type='server_only'))
        with (run / 'inference.csv').open(newline='') as stream:
            for item in csv.DictReader(stream):
                draws.append({'config': record['id'], 'training_seed': record['seed'], **item})
        full = run / 'last_checkpoint.pt'
        inventory.append(dict(config=record['id'], training_seed=record['seed'],
                              epoch=record['epochs'], path=str(full), sha256=sha(full),
                              size_bytes=full.stat().st_size, checkpoint_type='full_final'))
    metric_cols = ['config', 'category', 'training_seed', 'epoch', 'checkpoint',
                   'ari_mean', 'ari_std', 'purity_mean', 'purity_std', 'client_loss_last5_mean']
    csv_write(HERE / 'all_checkpoint_metrics.csv', metrics, metric_cols)
    csv_write(HERE / 'inference_repeats.csv', draws,
              ['config', 'training_seed', 'epoch', 'inference_seed', 'ari', 'purity'])
    csv_write(HERE / 'checkpoint_inventory.csv', inventory,
              ['config', 'training_seed', 'epoch', 'path', 'sha256', 'size_bytes', 'checkpoint_type'])
    ranking = sorted(metrics, key=lambda x: x['ari_mean'], reverse=True)
    for n, item in enumerate(ranking, 1):
        item['ari_rank'] = n
        item['selection'] = 'exploratory_label_selected'
    csv_write(HERE / 'best_checkpoint_ranking.csv', ranking,
              ['ari_rank', 'config', 'training_seed', 'epoch', 'ari_mean', 'ari_std',
               'purity_mean', 'purity_std', 'checkpoint', 'selection'])
    return ranking


def report(rows, ranking, plan_info, errors=None):
    seeds0 = [r for r in ranking if r['training_seed'] == 0]
    best = seeds0[0] if seeds0 else None
    lines = ['# FedVaDE Round 9 optimization', '',
             'All processing and training ran on the remote Linux server. No author source, data,',
             'or historical checkpoint was modified. This is an exploratory search on the author',
             'Breast Cancer evaluation records, **not independent test performance**.', '',
             f"Seed-0 plan: {len(plan_info['initialization'])} initialization/pretraining groups",
             f"({sum('reused' not in c for c in plan_info['initialization'])} new, one reused),",
             f"{len(plan_info['hyperparameter'])} hyperparameter groups; at most 2 full-300",
             'reruns and at most 6 extra training-seed validation runs. M search is not planned',
             'because this already reaches the 24-new-training upper budget.', '',
             'The author code sets each client GMM component count using the number of unique',
             'true labels. This count remains fixed by the inherited protocol; labels are never',
             'used to rank GMM starts. Ten inference seeds repeat one fixed checkpoint and are',
             'not ten independent training runs.', '',
             '| Config | Stage | Seed | LR | Alpha | Pretrain | GMM init | Fixed ARI | Exploratory best ARI | Best epoch |',
             '| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['id']} | {r['category']} | {r['seed']} | {r['lr']} | {r['alpha']} | "
                     f"{r['pretrain']} | {r['gmm_init']} seed {r['gmm_seed']} n_init={r['gmm_n_init']} | "
                     f"{r['fixed_ari']:.4f} | {r['best_ari']:.4f} | {r['best_epoch']} |")
    lines += ['', f'Round 7 F01 epoch100 exploratory reference: ARI {REFERENCE:.4f}.',
              f'Author supplied weight reference: approximately {AUTHOR_WEIGHT:.4f}.',
              'Publication comparison is exploratory: configuration and epoch selection reused',
              'the evaluation labels. No held-out test set was created.', '',
              'Server-only checkpoints were saved every 10 formal rounds. Full optimizer,',
              'scheduler, client and RNG state was saved at each run final round. More frequent',
              'full checkpoints would exceed the available 27 GB disk budget.', '',
              'The 300-round extensions are fresh runs with the same seed/configuration, not',
              'exact continuation of their 150-round runs. Pretraining length and GMM starts',
              'are changed one factor at a time relative to F01; the eight H groups vary only',
              'training hyperparameters relative to F01.', '']
    if best:
        lines += [f"Best seed-0 checkpoint: {best['checkpoint']}",
                  f"ARI {best['ari_mean']:.4f} ± {best['ari_std']:.4f}; "
                  f"Purity {best['purity_mean']:.4f} ± {best['purity_std']:.4f}; "
                  f"epoch {best['epoch']}; config {best['config']}.",
                  f"Change vs F01 epoch100: {best['ari_mean']-REFERENCE:+.4f} ARI.", '']
        best_dir = HERE / 'best_checkpoint'
        best_dir.mkdir(exist_ok=True)
        dest = best_dir / 'server.pt'
        if not dest.exists():
            shutil.copy2(best['checkpoint'], dest)
        atomic_json(best_dir / 'provenance.json', {'selection': 'exploratory_label_selected',
                    'source': best['checkpoint'], 'sha256': sha(dest),
                    'ari_mean': best['ari_mean'], 'ari_std': best['ari_std'],
                    'purity_mean': best['purity_mean'], 'purity_std': best['purity_std']})
    new_seed0 = [r for r in rows if r['status'] == 'ok' and r['seed'] == 0
                 and r['epochs'] == 150]
    if new_seed0:
        strongest = max(new_seed0, key=lambda r: r['best_ari'])
        lines += [f"Best 150-round new configuration: {strongest['id']}, "
                  f"exploratory ARI {strongest['best_ari']:.4f} at epoch {strongest['best_epoch']}.",
                  f"Its initial reconstruction MSE is {strongest['init_reconstruction_mse']:.6f}, "
                  f"GMM log likelihood per client record {strongest['init_gmm_loglik']:.6f}.",
                  'The reused F01 baseline lacks an initial-state likelihood snapshot, so the',
                  'initial likelihood comparison to that baseline remains unavailable.', '']
        try:
            import numpy as np
            gmm = np.array([r['init_gmm_loglik'] for r in new_seed0])
            final = np.array([r['fixed_ari'] for r in new_seed0])
            correlation = float(np.corrcoef(gmm, final)[0, 1])
            lines += [f'Across the {len(new_seed0)} new seed-0 phase-1 configurations, the',
                      f'Pearson correlation of initial GMM log likelihood and fixed-150 ARI is {correlation:.4f}.',
                      'This association does not establish causality.', '']
        except BaseException:
            lines += ['Initial likelihood / ARI correlation unavailable.', '']
        late = [(r['id'], r['best_ari'] - r['fixed_ari']) for r in new_seed0]
        lines += [f'{sum(gap > .02 for _, gap in late)} of {len(late)} phase-1 runs lost more',
                  'than 0.02 ARI between their exploratory best saved point and epoch 150.', '']
    multi = [r for r in rows if r['category'] == 'multiseed' and r['status'] == 'ok']
    if multi:
        lines += ['Independent training seeds 1–3 were run for the selected configurations;',
                  'their fixed-endpoint scores are in `multiseed_validation.csv`.', '']
    else:
        lines += ['No additional training-seed validation passed the predeclared improvement',
                  'gate; training stability beyond seed 0 remains unverified.', '']
    if errors:
        lines += ['## Errors', '', '```text', str(errors), '```', '']
    lines += ['## Files', '',
              'See `experiment_plan.json`, `experiment_configs.csv`,',
              '`initialization_comparison.csv`, `hyperparameter_comparison.csv`,',
              '`all_checkpoint_metrics.csv`, `inference_repeats.csv`,',
              '`multiseed_validation.csv`, `checkpoint_inventory.csv`,',
              'per-run `config.json`, `initialization.json`, `rounds.jsonl`, and raw `.log` files.',
              'The server retains all model weights. GitHub contains only text artifacts.', '']
    (HERE / 'ROUND9_REPORT.md').write_text('\n'.join(lines))


def main():
    if (HERE / 'experiment_plan.json').exists():
        raise SystemExit('Round 9 already started; refusing duplicate launch')
    initial, hyper = plan()
    full_plan = {'initialization': initial, 'hyperparameter': hyper,
                 'reference_checkpoint': str(R7 / 'fine/F01/epoch_100_server.pt'),
                 'reference_checkpoint_sha256': sha(R7 / 'fine/F01/epoch_100_server.pt'),
                 'training_seed': 0, 'inference_seeds': list(range(10)),
                 'phase1_rounds': 150, 'extended_rounds': 300,
                 'checkpoint_every': 10, 'new_phase1_runs': 16,
                 'max_extended_runs': 2, 'max_multiseed_runs': 6,
                 'max_new_runs': 24,
                 'extension_rule': 'top two new seed0 groups by max checkpoint ARI among epoch <=150 with fixed-150 Purity >=.88, if ARI >=.58; explicitly exploratory label selection',
                 'multiseed_rule': 'top two seed0 configurations with exploratory best ARI >= F01 epoch100 +.005; train seeds 1,2,3 at selected full budget; else skip',
                 'M_search': 'skipped by predeclared 24-run cap',
                 'author_protocol_caveats': ['full-data scaler', 'ID feature', 'overlapping client records',
                                             'local GMM component count uses unique true labels']}
    atomic_json(HERE / 'experiment_plan.json', full_plan)
    csv_write(HERE / 'experiment_configs.csv', initial + hyper,
              ['id', 'stage', 'lr', 'alpha', 'schedule', 'gamma', 'stepsize',
               'pretrain', 'gmm_seed', 'gmm_n_init', 'gmm_init', 'sample', 'reused'])
    rows = []
    progress(stage='PREFLIGHT', current_config='none', config_current=0,
             config_total=17, epoch=0, epoch_total=150, latest_ari=None,
             latest_purity=None, best_ari_so_far=REFERENCE,
             best_checkpoint=str(R7 / 'fine/F01/epoch_100_server.pt'),
             github='PENDING', status='RUNNING')
    error = None
    try:
        referenced_baseline()
        for position, config in enumerate(initial + hyper, 1):
            category = 'initialization' if config in initial else 'hyperparameter'
            if config.get('reused'):
                item = row_for(config, category, reused=True)
            else:
                out, summary = launch(config, 150, 'phase1', position, 17)
                item = row_for(config, category, summary, out)
            rows.append(item)
            save_tables(rows)
            if item['best_ari'] > json.loads((HERE / 'progress.json').read_text())['best_ari_so_far']:
                progress(best_ari_so_far=item['best_ari'],
                         best_checkpoint=item['best_checkpoint'])
        top = sorted([r for r in rows if r['status'] == 'ok' and r['best_ari'] >= .58
                      and r['fixed_purity'] >= .88], key=lambda x: x['best_ari'], reverse=True)[:2]
        atomic_json(HERE / 'extension_decision.json', {'selected': [r['id'] for r in top],
                    'rule': full_plan['extension_rule'], 'label_selected': True})
        by_id = {c['id']: c for c in initial + hyper}
        for position, item in enumerate(top, 1):
            config = by_id[item['id']]
            out, summary = launch(config, 300, 'extended', position, len(top))
            row = row_for(config, 'extended', summary, out)
            rows.append(row)
            save_tables(rows)
            if row['best_ari'] > json.loads((HERE / 'progress.json').read_text())['best_ari_so_far']:
                progress(best_ari_so_far=row['best_ari'], best_checkpoint=row['best_checkpoint'])
        candidates = sorted([r for r in rows if r['status'] == 'ok'
                             and r['best_ari'] >= REFERENCE + .005],
                            key=lambda x: x['best_ari'], reverse=True)
        chosen = []
        for item in candidates:
            if item['id'] not in chosen:
                chosen.append(item['id'])
            if len(chosen) == 2:
                break
        atomic_json(HERE / 'multiseed_decision.json', {'selected': chosen,
                    'rule': full_plan['multiseed_rule'], 'label_selected': True})
        for config_id in chosen:
            config = by_id[config_id]
            full = next((r for r in rows if r['id'] == config_id and r['epochs'] == 300), None)
            budget = 300 if full else 150
            for seed in (1, 2, 3):
                out, summary = launch(config, budget, 'multiseed', seed, 3, seed=seed)
                rows.append(row_for(config, 'multiseed', summary, out))
                save_tables(rows)
        progress(stage='FINALIZING', current_config='none', epoch=None, epoch_total=None)
        ranking = gather(rows)
        report(rows, ranking, full_plan)
        atomic_json(HERE / 'task_exit_status.json', {'exit_code': 0,
                    'completed_at': time.time(), 'new_runs': sum(r['status'] == 'ok' for r in rows),
                    'reused_runs': sum(r['status'] == 'reused' for r in rows)})
    except BaseException:
        error = traceback.format_exc()
        (HERE / 'error.log').write_text(error)
        try:
            ranking = gather(rows)
            report(rows, ranking, full_plan, error)
        except BaseException:
            (HERE / 'finalize_error.log').write_text(traceback.format_exc())
        atomic_json(HERE / 'task_exit_status.json', {'exit_code': 1,
                    'completed_at': time.time(), 'error': error})
    progress(stage='GITHUB_PUBLISH', github='PUSHING',
             status='FAILED' if error else 'RUNNING')
    try:
        subprocess.run([str(PYTHON), str(HERE / 'publish_round9.py')],
                       cwd=HERE, check=True, stdout=(HERE / 'github_publish.log').open('w'),
                       stderr=subprocess.STDOUT)
        github = json.loads((HERE / 'github_publish_status.json').read_text())
        progress(github='PUSHED' if github['status'] == 'PUSHED' else 'FAILED')
    except BaseException:
        (HERE / 'github_publish_error.log').write_text(traceback.format_exc())
        progress(github='FAILED')
    progress(stage='COMPLETE' if error is None else 'FAILED',
             status='FINISHED' if error is None else 'FAILED')
    if error:
        raise RuntimeError(error)


if __name__ == '__main__':
    main()
