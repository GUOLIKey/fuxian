"""Round 11B-pre orchestration: frozen GMM factor diagnostics only."""
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

from common import BASE, atomic_json, update_progress
from protocol import TOTAL_CONFIGS

STAGES = [
    'SOURCE_AUDIT',
    'CHECKPOINT_VALIDATION',
    'PROTOCOL_LOCK',
    'SMOKE_TEST',
    'M_SWEEP',
    'COVARIANCE_ABLATION',
    'K_CAPACITY',
    'FEDERATED_STATISTICS_AGGREGATION',
    'ORACLE_COMPARISON',
    'REPORT_GENERATION',
    'GITHUB_PUBLISH',
    'FINISHED',
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
        raise SystemExit('Round 11B-pre already launched; refusing duplicate start')
    atomic_json(started_path, {
        'started_at': time.time(),
        'command': [sys.executable, str(BASE / 'task_round11b_pre.py')],
        'stages': STAGES,
        'training': 'none; frozen Round 10 checkpoints only',
        'total_configs': TOTAL_CONFIGS,
        'python': sys.version,
    })
    update_progress(stage='SOURCE_AUDIT', status='RUNNING', current_model='all',
                    current_config='audit', completed=0, total=TOTAL_CONFIGS,
                    best_val_ll=None, rare_mode_proxy=None,
                    github='PENDING', error=None)

    compute_error = None
    publish_ok = False
    try:
        run('audit.py', 'audit.log')
        update_progress(stage='CHECKPOINT_VALIDATION', status='RUNNING',
                        current_config='checkpoint hashes verified')
        run('protocol.py', 'protocol.log')
        run('smoke_round11b_pre.py', 'smoke.log')

        update_progress(stage='M_SWEEP', status='RUNNING')
        run('run_m_sweep.py', 'm_sweep.log')

        update_progress(stage='COVARIANCE_ABLATION', status='RUNNING')
        run('run_covariance_ablation.py', 'covariance.log')

        update_progress(stage='K_CAPACITY', status='RUNNING')
        run('run_k_capacity.py', 'k_capacity.log')

        update_progress(stage='FEDERATED_STATISTICS_AGGREGATION', status='RUNNING')
        run('run_federated_aggregation.py', 'federated_aggregation.log')

        update_progress(stage='ORACLE_COMPARISON', status='RUNNING')
        run('oracle_gap.py', 'oracle_gap.log')

        update_progress(stage='REPORT_GENERATION', status='RUNNING')
        run('report.py', 'report.log')

        update_progress(stage='GITHUB_PUBLISH', status='RUNNING', github='PUSHING')
        try:
            run('publish_round11b_pre.py', 'publish.log')
            status = json.loads((BASE / 'github_publish_status.json').read_text())
            publish_ok = status.get('status') == 'PUSHED'
        except BaseException:
            publish_ok = False
            (BASE / 'github_publish_error.log').write_text(traceback.format_exc())
    except BaseException:
        compute_error = traceback.format_exc()
        (BASE / 'error.log').write_text(compute_error)

    if compute_error is None:
        final_status = 'FINISHED' if publish_ok else 'COMPUTE_FINISHED_PUBLISH_FAILED'
        update_progress(stage='FINISHED' if publish_ok else 'COMPUTE_FINISHED_PUBLISH_FAILED',
                        status=final_status, completed=TOTAL_CONFIGS, total=TOTAL_CONFIGS,
                        github='PUSHED' if publish_ok else 'FAILED', error=None)
    else:
        update_progress(stage='FAILED', status='FAILED', error=compute_error,
                        github='FAILED' if (BASE / 'github_publish_status.json').exists() else 'PENDING')

    status = {
        'exit_code': 0 if compute_error is None else 1,
        'compute_exit_code': 0 if compute_error is None else 1,
        'publish_status': 'PUSHED' if publish_ok else 'FAILED',
        'status': ('FINISHED' if publish_ok else 'COMPUTE_FINISHED_PUBLISH_FAILED')
        if compute_error is None else 'FAILED',
        'finished_at': time.time(),
        'error': compute_error,
        'training': 'none',
        'total_configs': TOTAL_CONFIGS,
    }
    if (BASE / 'github_publish_status.json').exists():
        status['github_publish_status'] = json.loads(
            (BASE / 'github_publish_status.json').read_text())
    atomic_json(BASE / 'task_exit_status.json', status)
    print('TASK_EXIT', status['exit_code'], status['status'], flush=True)


if __name__ == '__main__':
    main()
