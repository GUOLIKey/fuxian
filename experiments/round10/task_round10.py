"""Bounded Round 10 server workflow; stops after final report and GitHub publication."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path('/root/autodl-tmp/fedvade_project')
BASE = ROOT / 'reproduction_runs/round10_non_iid_generation'
PYTHON = ROOT / '.venv-cpu/bin/python'


def atomic_json(path, value):
    temp = Path(str(path) + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def progress(**changes):
    path = BASE / 'progress.json'
    current = json.loads(path.read_text()) if path.exists() else {}
    current.update(changes, updated_at=time.time())
    atomic_json(path, current)


def run(argv, name):
    with (BASE / name).open('w') as stream:
        subprocess.run([str(x) for x in argv], cwd=BASE, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def table(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def mean(rows, key):
    return float(np.mean([float(r[key]) for r in rows])) if rows else float('nan')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_report(error=None):
    dataset = json.loads((BASE / 'dataset_manifest.json').read_text())
    breast = table(BASE / 'breast_cancer_generation.csv')
    summaries = {}
    for model in ('author_weight', 'round7_F01_epoch100'):
        rows = [r for r in breast if r['model'] == model and r['component'] == 'unconditional']
        summaries[model] = {'mmd': mean(rows, 'mmd_rbf'),
                            'swd': mean(rows, 'sliced_wasserstein'),
                            'nn': mean(rows, 'generated_to_real_nn_mean')}
    metrics = table(BASE / 'noniid_baseline_metrics.csv') if (BASE / 'noniid_baseline_metrics.csv').exists() else []
    coverage = table(BASE / 'rare_mode_coverage.csv') if (BASE / 'rare_mode_coverage.csv').exists() else []
    latent = table(BASE / 'latent_diagnostics.csv') if (BASE / 'latent_diagnostics.csv').exists() else []
    classifier = json.loads((BASE / 'reference_classifier_test.json').read_text()) if (BASE / 'reference_classifier_test.json').exists() else None
    decision = json.loads((BASE / 'simple_baseline_decision.json').read_text()) if (BASE / 'simple_baseline_decision.json').exists() else None
    lines = ['# FedVaDE Round 10: Non-IID generation and rare-mode audit', '',
             'This round tests a generative-coverage hypothesis rather than optimizing Breast Cancer ARI.',
             'All data processing, model training, inference and evaluation ran on the remote server.',
             'No original author material or previous run was modified. Generated samples are not assigned',
             'true clinical labels; all generated-class results below are reference-classifier proxies.', '',
             '## Phase A: frozen Breast Cancer models', '',
             'The author protocol contains the ID feature, full-data StandardScaler and repeated client records.',
             'These are in-sample distribution diagnostics, **not held-out generation tests**.', '',
             '| Weight | MMD (lower) | Sliced Wasserstein (lower) | Generated-to-real NN mean (lower) |',
             '| --- | ---: | ---: | ---: |']
    for model, item in summaries.items():
        lines.append(f"| {model} | {item['mmd']:.4f} | {item['swd']:.4f} | {item['nn']:.4f} |")
    comparisons = [key for key in ('mmd', 'swd', 'nn')
                   if summaries['author_weight'][key] < summaries['round7_F01_epoch100'][key]]
    if len(comparisons) == 3:
        phase_a = 'On these three in-sample distances, the author weight is better.'
    elif len(comparisons) == 0:
        phase_a = 'On these three in-sample distances, F01 is better.'
    else:
        phase_a = 'The three distances disagree; neither weight is unambiguously better at generation.'
    lines += ['', phase_a, 'Neither ARI nor GMM-component assignment establishes the true class of a generated sample.', '',
              '## Dataset and protocol', '',
              f"Official [UCI Cardiotocography](https://archive.ics.uci.edu/dataset/193/cardiotocography), "
              f"DOI {dataset['doi']}, CC BY 4.0. Raw workbook SHA256: `{dataset['workbook_sha256']}`.",
              f"Rows {dataset['raw_rows']}; NSP class counts {dataset['raw_class_counts']}; "
              f"exact feature duplicates {dataset['feature_duplicate_rows']}; "
              f"duplicate groups with conflicting labels {dataset['duplicate_groups_with_conflicting_labels']}.",
              f"Train/validation/test sizes: {[dataset['split_counts'][s]['n'] for s in ('train','validation','test')]}. "
              'Exact duplicate feature vectors are kept within one split. CLASS, expert pattern indicators',
              'and record identifiers are excluded. The scaler is fitted on model-training records only.',
              'The new model is `label_blind_fedvade_baseline`: global K=3 is fixed in advance; local K',
              'is selected by unlabeled BIC on encoder means. The author source originally uses true local',
              'labels to determine K, so this is a documented protocol adaptation, not a literal source rerun.',
              'IID, medium and severe partitions use the same model-training records; only assignment to clients changes.',
              'In the severe case A and B contain no NSP=3 records and C contains all rare-class training records.', '',
              '## Fixed-model held-out test and generation', '',
              'Checkpoint selection uses validation reconstruction MSE only; the sealed test split is used once',
              'after all runs finish. Test ARI/Purity and reconstruction are held-out metrics. Generated-class',
              'coverage is **exploratory** because generated samples have no true labels.', '',
              '| Model | Selected epoch | Test ARI | Test Purity | Test recon MSE | Rare proxy mean |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in metrics:
        own = [r for r in coverage if r['model'] == row['model']]
        lines.append(f"| {row['model']} | {row['selected_epoch']} | {float(row['ari_test']):.4f} | "
                     f"{float(row['purity_test']):.4f} | {float(row['reconstruction_mse_test']):.4f} | "
                     f"{mean(own, 'rare_mode_recovery_proxy'):.4f} |")
    if classifier:
        rare = classifier['metrics']['2']
        lines += ['', f"Independent reference-classifier rare-class test precision {rare['precision']:.3f}, "
                  f"recall {rare['recall']:.3f} on {int(rare['support'])} held-out records.",
                  'A rare-mode proxy requires classifier rare probability >=0.8 and a generated-to-real',
                  'nearest-neighbor distance within the 95th percentile of real rare-class nearest-neighbor distances.',
                  'This is a conservative, classifier-dependent proxy, not true synthetic precision or recall.', '']
    severe = [r for r in coverage if r['model'] == 'fedvade_severe_seed0']
    iid = [r for r in coverage if r['model'] == 'fedvade_iid_seed0']
    medium = [r for r in coverage if r['model'] == 'fedvade_medium_seed0']
    local = [r for r in coverage if r['model'] == 'local_A']
    if severe and iid and medium:
        lines += [f"Rare-mode proxy means at training seed 0: IID {mean(iid,'rare_mode_recovery_proxy'):.4f}, "
                  f"medium {mean(medium,'rare_mode_recovery_proxy'):.4f}, "
                  f"severe {mean(severe,'rare_mode_recovery_proxy'):.4f}; "
                  f"local A {mean(local,'rare_mode_recovery_proxy'):.4f}.",
                  'Only severe has three independent training seeds. The IID-to-severe comparison is therefore',
                  'a controlled single-seed scenario contrast, not a stable trend estimate.', '']
    rare_latent = [r for r in latent if r['model'] == 'fedvade_severe_seed0' and r['class'] == '2']
    if rare_latent:
        r = rare_latent[0]
        lines += [f"Severe-seed0 rare test latent centroid distance to other classes "
                  f"{float(r['centroid_distance_to_others']):.4f}; rare reconstruction MSE "
                  f"{float(r['reconstruction_mse']):.4f}; mean GMM responsibility {r['posterior_mean']}.",
                  'These diagnostics help separate encoder overlap, prior mass, sampling frequency and decoder',
                  'quality but do not alone identify a causal failure point.', '']
    if decision:
        lines += ['## Simple baseline gate', '',
                  f"Reference reliable={decision['reference_rare_reliable']}; "
                  f"severe rare proxy={decision['rare_proxy']:.4f}; "
                  f"test rare fraction={decision['test_rare_fraction']:.4f}; "
                  f"flat-GMM comparison executed={decision['run']}.",
                  'If executed, the K=3/K=6 fits use pooled training latent vectors with the encoder/decoder frozen.',
                  'This is a centralized diagnostic upper bound, not a privacy-preserving federated method.', '']
    # Conservative decision: a proxy cannot establish true rare-mode generation.
    conclusion = 'E: evidence remains uncertain; next work should validate the generated-class evaluator and repeat scenario controls across training seeds before choosing ACD, Multi-CA or decoder changes.'
    if classifier and classifier['metrics']['2']['precision'] >= .8 and classifier['metrics']['2']['recall'] >= .8 and severe:
        test_fraction = float(np.mean([float(r['test_rare_fraction']) for r in severe]))
        if mean(severe, 'rare_mode_recovery_proxy') >= .8 * test_fraction:
            conclusion = 'D: the prespecified rare-mode proxy did not reveal a large severe-scenario coverage deficit; the proposed rare-loss hypothesis is not supported by this diagnostic.'
    lines += ['## Research decision', '', conclusion,
              'ACD and Multi-CA were not implemented. Current evidence does not justify either mechanism',
              'as a proven remedy. If rare-mode weakness is confirmed with a reliable evaluator, inspect latent',
              'class separation and reconstruction first; only if representations are sound but mixture mass',
              'or server sampling fails should server probability aggregation become the first modification.',
              'Simple flat GMM fitting and central/local references are diagnostics, not evidence of novelty.', '',
              '## Evidence and limits', '',
              '- `source_audit.md`: author source/paper method and protocol differences.',
              '- `dataset_manifest.json`, `partition_protocol.md`, `partition_manifest.csv`: source, splits and clients.',
              '- `breast_cancer_generation.csv`, `breast_latent_diagnostics.csv`: author-protocol frozen-weight diagnostics.',
              '- `noniid_baseline_metrics.csv`: held-out fixed-checkpoint test clustering/reconstruction.',
              '- `generation_inference_records.csv`, `classwise_generation_metrics.csv`, `rare_mode_coverage.csv`: fixed-budget exploratory generation proxies.',
              '- `latent_diagnostics.csv`, `simple_baseline_comparison.csv`: mechanistic checks and gated simple baseline.',
              '- Per-run `result.json`, `config.json`, `rounds.jsonl`, `*.log` and server-only checkpoints retain failed attempts and provenance.',
              '- Generation validity is judged in standardized tabular feature space. No image metric is used.',
              '- Severe vs IID/medium has only one common training seed; missing-mode recovery has no ground-truth labels for synthetic samples.',
              '- The supplied six-page paper PDF lacks a full Appendix; Table II protocol remains partly unknown.', '']
    if error:
        lines += ['## Failure', '', '```text', error, '```', '']
    (BASE / 'ROUND10_REPORT.md').write_text('\n'.join(lines))


def main():
    if (BASE / 'task_started.json').exists():
        raise SystemExit('Round 10 already launched; refusing duplicate start')
    atomic_json(BASE / 'task_started.json', {'started_at': time.time(),
                'formal_runs': ['iid_seed0', 'medium_seed0', 'severe_seed0', 'severe_seed1', 'severe_seed2'],
                'references': ['local_A', 'centralized'], 'training_budget': '50+150 per run'})
    progress(stage='SOURCE_AUDIT', current_dataset='Cardiotocography', current_scenario='--',
             current_run='--', epoch=None, epoch_total=None, latest_ari=None, latest_purity=None,
             rare_mode_coverage=None, generation_mmd=None, github='PENDING', status='RUNNING')
    error = None
    try:
        if not (BASE / 'source_audit.md').is_file():
            run([PYTHON, BASE / 'source_audit.py'], 'source_audit.log')
        progress(stage='BREAST_CANCER_GENERATION_AUDIT', current_dataset='Breast Cancer')
        if not (BASE / 'breast_cancer_generation.csv').is_file():
            run([PYTHON, BASE / 'breast_generation.py'], 'breast_generation.log')
        progress(stage='DATASET_PREPARATION', current_dataset='Cardiotocography')
        if not (BASE / 'dataset_manifest.json').is_file():
            run([PYTHON, BASE / 'prepare_data.py'], 'dataset_prepare.log')
        manifest = json.loads((BASE / 'dataset_manifest.json').read_text())
        if sha(BASE / 'raw/cardiotocography.zip') != manifest['archive_sha256'] or \
           sha(BASE / 'raw/CTG.xls') != manifest['workbook_sha256'] or \
           sha(BASE / 'private_data/data.npz') != manifest['data_npz_sha256']:
            raise RuntimeError('Official data or split hash changed after preparation')
        progress(stage='NONIID_PARTITION')
        if not (BASE / 'partition_manifest.csv').is_file():
            raise RuntimeError('Partition manifest missing')
        progress(stage='BASELINE_SMOKE_TEST', current_scenario='severe')
        smoke = json.loads((BASE / 'smoke/severe_seed0/result.json').read_text())
        if smoke['status'] != 'ok' or not (BASE / 'smoke/severe_seed0/selected_server.pt').is_file():
            raise RuntimeError('FedVaDE short smoke did not complete')
        ordered = [('iid', 0), ('medium', 0), ('severe', 0), ('severe', 1), ('severe', 2)]
        for index, (scenario, seed) in enumerate(ordered, 1):
            name = f'{scenario}_seed{seed}'
            progress(stage='FEDVADE_BASELINE_TRAIN', current_dataset='Cardiotocography',
                     current_scenario=scenario, current_run=f'Seed {seed} / 2 ({index} / 5)',
                     epoch=0, epoch_total=150, latest_ari=None, latest_purity=None)
            run([PYTHON, BASE / 'train_fedvade.py', '--out', BASE / 'formal' / name,
                 '--scenario', scenario, '--seed', seed, '--rounds', 150,
                 '--pretrain', 50, '--save-every', 25], f'{name}.log')
            result = json.loads((BASE / 'formal' / name / 'result.json').read_text())
            if result['status'] != 'ok' or not (BASE / 'formal' / name / 'selected_server.pt').is_file():
                raise RuntimeError(f'Formal model missing/incomplete: {name}')
        for kind in ('local_A', 'centralized'):
            progress(stage='FEDVADE_BASELINE_TRAIN', current_scenario='severe',
                     current_run=kind, epoch=0, epoch_total=150,
                     latest_ari=None, latest_purity=None)
            run([PYTHON, BASE / 'train_reference.py', '--kind', kind,
                 '--out', BASE / 'references' / kind, '--seed', 0,
                 '--pretrain', 50, '--epochs', 150], f'{kind}.log')
        progress(stage='RARE_MODE_EVALUATION', current_run='all frozen models',
                 epoch=None, epoch_total=None, latest_ari=None, latest_purity=None)
        run([PYTHON, BASE / 'evaluate_noniid.py'], 'generation_evaluation.log')
        progress(stage='LATENT_DIAGNOSIS')
        if not (BASE / 'latent_diagnostics.csv').is_file():
            raise RuntimeError('Latent diagnostics missing')
        progress(stage='OPTIONAL_SIMPLE_BASELINE')
        if not (BASE / 'simple_baseline_comparison.csv').is_file():
            raise RuntimeError('Simple-baseline gate did not produce a record')
        progress(stage='FINAL_REPORT')
        write_report()
        required = ['source_audit.md', 'dataset_manifest.json', 'partition_manifest.csv',
                    'breast_cancer_generation.csv', 'noniid_baseline_metrics.csv',
                    'classwise_generation_metrics.csv', 'rare_mode_coverage.csv',
                    'latent_diagnostics.csv', 'ROUND10_REPORT.md']
        missing = [name for name in required if not (BASE / name).is_file()]
        if missing: raise RuntimeError(f'Missing required results: {missing}')
        atomic_json(BASE / 'task_exit_status.json', {'exit_code': 0,
                    'finished_at': time.time(), 'formal_runs': len(ordered),
                    'reference_runs': 2, 'all_required_results': True})
    except BaseException:
        error = traceback.format_exc()
        (BASE / 'error.log').write_text(error)
        try: write_report(error)
        except BaseException: (BASE / 'report_error.log').write_text(traceback.format_exc())
        atomic_json(BASE / 'task_exit_status.json', {'exit_code': 1,
                    'finished_at': time.time(), 'error': error})
    progress(stage='GITHUB_PUBLISH', github='PUSHING', status='RUNNING' if not error else 'FAILED')
    try:
        run([PYTHON, BASE / 'publish_round10.py'], 'github_publish.log')
        github = json.loads((BASE / 'github_publish_status.json').read_text())
        progress(github=github['status'])
    except BaseException:
        (BASE / 'github_publish_error.log').write_text(traceback.format_exc())
        progress(github='FAILED')
    progress(stage='FINISHED' if not error else 'FAILED',
             status='FINISHED' if not error else 'FAILED')
    if error: raise RuntimeError(error)


if __name__ == '__main__':
    main()
