"""Budgeted operator for the nine pre-registered diagnostics."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import subprocess
from statistics import mean, median, pstdev
import sys
import time


ROOT = Path(__file__).resolve().parent
CPU = '/root/autodl-tmp/fuxian-run/reproduction_runs/env-fedvade/bin/python'
HCA = '/root/autodl-tmp/fuxian-smoke/py312-hca/bin/python'
GPU = ROOT / 'env-gpu/bin/python'
MODELS = {'hire': 'reproduction_runs/upstream/Fed-HIRE/opensource',
          'hca': 'reproduction_runs/upstream/FedHCA', 'fedvade': 'FedVaDE_source'}
ENV = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
       'MKL_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'}


def load(path):
    return json.loads(path.read_text())


def rounds(path):
    return [json.loads(line) for line in (path / 'rounds.jsonl').read_text().splitlines()]


def run_job(python, model, folder, identifier, deadline, limit, *args):
    out = ROOT / folder / identifier
    if out.exists():
        raise RuntimeError(f'Would overwrite {out}')
    out.mkdir(parents=True)
    options = [str(python), '-u', 'runner.py', model, '--source', MODELS[model], '--out', str(out), *args]
    if time.time() + limit > deadline:
        value = {'model': model, 'status': 'not_started_budget', 'command': options}
    else:
        print(f'Start {identifier}', flush=True)
        with (out / 'stdout.log').open('w') as stream:
            proc = subprocess.run(options, cwd=ROOT, env=ENV, stdout=stream, stderr=subprocess.STDOUT)
        value = load(out / 'result.json') if (out / 'result.json').is_file() else {'model': model, 'status': 'failed'}
        value['exit_code'] = proc.returncode
        if proc.returncode != 0:
            value['status'] = 'failed'
    (out / 'status.json').write_text(json.dumps(value, indent=2) + '\n')
    print(f"{identifier}: {value['status']}", flush=True)
    return value


def summary():
    fields = ['run', 'model', 'status', 'seed', 'device', 'rate', 'schedule', 'elapsed_seconds',
              'purity', 'ari', 'nmi', 'acc', 'purity_150', 'ari_150', 'client_loss', 'server_loss', 'error']
    records = []
    for path in sorted((ROOT / 'formal').glob('*/status.json')):
        data = load(path)
        row = {key: data.get(key) for key in fields}
        row['run'] = path.parent.name
        actual = data.get('params', {})
        if row['model'] == 'hca':
            row['rate'] = actual.get('rate')
        if row['model'] == 'fedvade':
            row['rate'] = actual.get('initial_lr')
        for key in ('purity', 'ari', 'client_loss', 'server_loss'):
            row[key] = data.get('last', {}).get(key, row[key])
        for key in ('purity', 'ari'):
            row[key + '_150'] = data.get('round_150', {}).get(key)
        records.append(row)
    with (ROOT / 'results.csv.tmp').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    (ROOT / 'results.csv.tmp').replace(ROOT / 'results.csv')
    groups = {}
    for rate in (.0001, .03, .05):
        good = [item for item in records if item['model'] == 'hire'
                and item['rate'] == rate and item['status'] == 'ok']
        groups[str(rate)] = {'n': len(good)}
        for key in ('purity', 'ari', 'nmi', 'acc', 'elapsed_seconds'):
            values = [item[key] for item in good]
            if values:
                groups[str(rate)][key] = {'mean': mean(values), 'std': pstdev(values)}
    (ROOT / 'hire_group_stats.json').write_text(json.dumps(groups, indent=2) + '\n')


def valid_schedule():
    original = rounds(ROOT / 'preflight/cpu-author')
    revised = rounds(ROOT / 'preflight/cpu-epoch')
    differing = [first['round'] for first, second in zip(original, revised)
                 if first['client_batch_lrs'] != second['client_batch_lrs']]
    if not differing or any(
        first['server_end_lr'] != second['server_end_lr'] for first, second in zip(original, revised)
    ):
        raise RuntimeError('Client trajectories must differ and server trajectories must match.')
    if (load(ROOT / 'preflight/cpu-author/result.json')['partition'] != load(
        ROOT / 'preflight/cpu-epoch/result.json')['partition']
    ):
        raise RuntimeError('The two schedule variants did not use the same partition.')
    return differing[0]


def gpu_check(deadline):
    gate = ROOT / 'preflight/gpu-core.json'
    with gate.open('w') as stream, (ROOT / 'preflight/gpu-core.stderr.log').open('w') as errors:
        if subprocess.run([str(GPU), 'gpu_gate.py', 'FedVaDE_source'], env=ENV,
                          stdout=stream, stderr=errors).returncode:
            return False, 'Fixed-input numerical comparison failed.'
    sample = run_job(GPU, 'fedvade', 'preflight', 'gpu-author', deadline, 60,
                     '--device', 'cuda', '--smoke')
    if sample.get('status') != 'ok':
        return False, 'End-to-end CUDA short run failed.'
    cpu_path = ROOT / 'preflight/cpu-author'
    if load(cpu_path / 'result.json')['partition'] != sample['partition']:
        return False, 'CPU/GPU partition mismatch.'
    cpu_rows = rounds(cpu_path)
    gpu_rows = rounds(ROOT / 'preflight/gpu-author')
    differences = {}
    tolerances = {'client_loss': (.02, .01), 'server_loss': (.02, .01),
                  'purity': (.0, .03), 'ari': (.0, .08)}
    invalid = []
    for key, (rtol, atol) in tolerances.items():
        errors = [abs(left[key] - right[key]) for left, right in zip(cpu_rows, gpu_rows)]
        differences[key] = max(errors)
        if any(error > atol + rtol * abs(row[key]) for error, row in zip(errors, cpu_rows)):
            invalid.append(key)
    if any(left['client_batch_lrs'] != right['client_batch_lrs']
           for left, right in zip(cpu_rows, gpu_rows)):
        invalid.append('learning_rate')
    cpu_speed = median(row['seconds'] for row in cpu_rows[2:])
    gpu_speed = median(row['seconds'] for row in gpu_rows[2:])
    verdict = {'max_absolute_difference': differences, 'tolerances_rtol_atol': tolerances,
               'outside_tolerance': invalid, 'cpu_seconds_per_round': cpu_speed,
               'gpu_seconds_per_round': gpu_speed, 'cpu_over_gpu': cpu_speed / gpu_speed}
    (ROOT / 'preflight/gpu-vs-cpu.json').write_text(json.dumps(verdict, indent=2) + '\n')
    return not invalid, ('CUDA gate passed.' if not invalid else 'End-to-end CPU/GPU tolerance failed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--gpu-wait-until', type=float, required=True)
    parser.add_argument('--hire-wait-until', type=float, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    first_divergence = valid_schedule()
    estimated = median(row['seconds'] for row in rounds(ROOT / 'preflight/cpu-author')[2:]) * 350
    print(f'First differing client update: round {first_divergence}; estimated FedVaDE CPU per run {estimated:.1f}s.')
    ready = ROOT / 'hire-ready'
    print('Waiting for the pinned Fed-HIRE environment.', flush=True)
    while not ready.exists() and time.time() < args.hire_wait_until:
        time.sleep(5)
    hire = ready.read_text().strip() if ready.exists() else None
    if hire:
        smoke = run_job(hire, 'hire', 'preflight', 'hire-local-smoke', args.deadline, 10, '--smoke')
        if smoke['status'] != 'ok':
            print('Fed-HIRE short check failed; other models will continue.', flush=True)
            hire = None
    for rate in (.0001, .03, .05):
        if not hire:
            for seed in (0, 1):
                out = ROOT / 'formal' / f'A_lr{rate}_seed{seed}'
                out.mkdir(parents=True)
                (out / 'status.json').write_text(json.dumps({
                    'model': 'hire', 'status': 'not_started_environment', 'rate': rate, 'seed': seed,
                }) + '\n')
            summary()
            continue
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = [pool.submit(run_job, hire, 'hire', 'formal', f'A_lr{rate}_seed{seed}',
                                   args.deadline, 10, '--rate', str(rate), '--seed', str(seed))
                       for seed in (0, 1)]
            for completed in as_completed(pending):
                completed.result()
                summary()
    print('Waiting for the isolated CUDA build; formal FedVaDE remains unstarted.', flush=True)
    while not (ROOT / 'gpu-ready').exists() and time.time() < args.gpu_wait_until:
        time.sleep(5)
    if (ROOT / 'gpu-ready').exists():
        try:
            passed, reason = gpu_check(args.deadline)
        except Exception as error:
            passed, reason = False, repr(error)
    else:
        passed, reason = False, 'CUDA wheel provisioning did not finish by its preflight cutoff.'
    (ROOT / 'device_decision.json').write_text(json.dumps({'gpu_passed': passed, 'reason': reason,
                                                         'device': 'cuda' if passed else 'cpu',
                                                         'first_schedule_divergence_round': first_divergence},
                                                        indent=2) + '\n')
    seconds = max(estimated * 1.5, 60.)
    if args.deadline - time.time() > 2 * seconds + 60:
        with ThreadPoolExecutor(max_workers=1) as pool:
            hca = pool.submit(run_job, HCA, 'hca', 'formal', 'C_ecoli_default_seed0', args.deadline, 30)
            for schedule in ('author', 'epoch'):
                run_job(GPU if passed else CPU, 'fedvade', 'formal', f'B_{schedule}_seed0',
                        args.deadline, seconds, '--device', 'cuda' if passed else 'cpu', '--schedule', schedule)
                summary()
            hca.result()
    else:
        for schedule in ('author', 'epoch'):
            run_job(GPU if passed else CPU, 'fedvade', 'formal', f'B_{schedule}_seed0',
                    args.deadline, seconds, '--device', 'cuda' if passed else 'cpu', '--schedule', schedule)
            summary()
        run_job(HCA, 'hca', 'formal', 'C_ecoli_default_seed0', args.deadline, 30)
    summary()
    print('Round 2 queue finished.', flush=True)


if __name__ == '__main__':
    main()
