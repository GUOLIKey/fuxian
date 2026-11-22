"""Round 11A orchestration: frozen-model diagnostics only, no neural training."""
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

from common import BASE, atomic_json, update_progress

STAGES = [
    'SOURCE_AND_CHECKPOINT_AUDIT',
    'EVALUATOR_CALIBRATION',
    'ORACLE_LATENT_DIAGNOSIS',
    'GLOBAL_VS_UNION_MOG',
    'FROZEN_GMM_K_COMPARISON',
    'FINAL_ANALYSIS',
    'GITHUB_PUBLISH',
]


def run(script, log_name):
    started = time.time()
    print('START', script, flush=True)
    with (BASE / log_name).open('w') as stream:
        subprocess.run([sys.executable, str(BASE / script)], cwd=BASE,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    print('DONE', script, f'{time.time() - started:.1f}s', flush=True)


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    started_path = BASE / 'task_started.json'
    if started_path.exists():
        raise SystemExit('Round 11A already launched; refusing duplicate start')
    atomic_json(started_path, {
        'started_at': time.time(),
        'command': [sys.executable, str(BASE / 'task_round11a.py')],
        'stages': STAGES,
        'training_budget': '0 new VaDE/FedVaDE training; frozen checkpoints only',
        'python': sys.version,
    })
    update_progress(stage='SOURCE_AND_CHECKPOINT_AUDIT', status='RUNNING',
                    current_dataset='Cardiotocography', current_model='--',
                    current_experiment='--', progress_current=0, progress_total=12,
                    rare_mode_proxy=None, best_result='--', github='PENDING', error=None)

    error = None
    publish_ok = False
    try:
        run('audit.py', 'audit.log')

        update_progress(stage='EVALUATOR_CALIBRATION', status='RUNNING',
                        current_model='reference classifier', current_experiment='E1')
        run('run_e1_evaluator.py', 'evaluator.log')

        update_progress(stage='ORACLE_LATENT_DIAGNOSIS', status='RUNNING',
                        current_model='frozen models', current_experiment='E2-reconstruction')
        run('run_e2_reconstruction.py', 'reconstruction.log')

        update_progress(stage='GLOBAL_VS_UNION_MOG', status='RUNNING',
                        current_model='frozen models', current_experiment='E2-global-vs-union')
        run('run_e2_global_vs_union.py', 'global_vs_union.log')

        update_progress(stage='FROZEN_GMM_K_COMPARISON', status='RUNNING',
                        current_model='frozen models', current_experiment='E3')
        run('run_e3_gmm_capacity.py', 'gmm_capacity.log')

        update_progress(stage='FINAL_ANALYSIS', status='RUNNING',
                        current_model='all', current_experiment='report')
        run('report.py', 'report.log')

        update_progress(stage='GITHUB_PUBLISH', status='RUNNING', github='PUSHING')
        run('publish_round11a.py', 'publish.log')
        publish_status = json.loads((BASE / 'github_publish_status.json').read_text())
        publish_ok = publish_status.get('status') == 'PUSHED'
        update_progress(stage='FINISHED', status='FINISHED' if publish_ok else 'FAILED',
                        github='PUSHED' if publish_ok else 'FAILED',
                        progress_current=12, progress_total=12)
    except BaseException:
        error = traceback.format_exc()
        (BASE / 'error.log').write_text(error)
        update_progress(stage='FAILED', status='FAILED', error=error,
                        github='FAILED' if (BASE / 'github_publish_status.json').exists() else 'PENDING')
    finally:
        status = {
            'exit_code': 0 if (error is None and publish_ok) else 1,
            'finished_at': time.time(),
            'diagnostics_complete': (BASE / 'ROUND11A_REPORT.md').is_file() and error is None,
            'github': 'PUSHED' if publish_ok else 'FAILED',
            'error': error,
        }
        if (BASE / 'github_publish_status.json').exists():
            status['github_publish_status'] = json.loads(
                (BASE / 'github_publish_status.json').read_text())
        atomic_json(BASE / 'task_exit_status.json', status)
        print('TASK_EXIT', status['exit_code'], status['github'], flush=True)


if __name__ == '__main__':
    main()
