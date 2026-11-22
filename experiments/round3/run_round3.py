"""Run the 32 approved trials using the frozen round-two runner and environments."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'fuxian-round2'
ENV = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
CPU = '/root/autodl-tmp/fuxian-run/reproduction_runs/env-fedvade/bin/python'
RUNNER_HASH = '2cba959cc7353598cdea24475d69bae6d2a52a4ca39ec7cbb79981d70df84048'


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(model, interpreter, source, reference, freeze):
    previous = read(PREVIOUS / 'formal' / reference / 'result.json')
    if any(sha(source / name) != value for name, value in previous['source_hashes'].items()):
        raise ValueError(f'{model}: author source or data mismatch')
    packages = subprocess.check_output([interpreter, '-m', 'pip', 'freeze'], env=ENV)
    actual = packages.decode().splitlines()
    expected = (PREVIOUS / freeze).read_text().splitlines()
    (ROOT / f'{model}-freeze.txt').write_bytes(packages)
    if sorted(actual) != sorted(expected):
        raise ValueError(f'{model}: package versions differ from round two')
    return previous


def run(job, deadline):
    model, interpreter, source, name, seed, options, reference = job
    out = ROOT / 'formal' / name
    out.mkdir(parents=True)
    command = [interpreter, '-u', str(ROOT / 'runner.py'), model, '--source',
               str(source), '--out', str(out), '--seed', str(seed), *options]
    status = {'model': model, 'seed': seed, 'command': command}
    if time.time() + (650 if model == 'fedvade' else 15) > deadline:
        status['status'] = 'not_started_budget'
    else:
        print(f'Start {name}', flush=True)
        with (out / 'stdout.log').open('w') as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=ENV, stdout=stream,
                                       stderr=subprocess.STDOUT)
            save(out / 'process.json', {'pid': process.pid, 'started': time.time()})
            status['exit_code'] = process.wait()
        path = out / 'result.json'
        if status['exit_code'] == 0 and path.is_file():
            result = read(path)
            for key in ('params', 'partition', 'source_hashes', 'python', 'numpy', 'device', 'torch'):
                if result.get(key) != reference.get(key):
                    status['error'] = f'Differs from round two: {key}'
                    break
            status['status'] = result['status'] if 'error' not in status else 'protocol_mismatch'
        else:
            status['status'] = 'failed'
    save(out / 'status.json', status)
    print(f'{name}: {status["status"]}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deadline', type=float, required=True)
    options = parser.parse_args()
    if sha(ROOT / 'runner.py') != RUNNER_HASH or (ROOT / 'formal').exists():
        raise SystemExit('Wrong runner or existing output directory.')
    hire = (PREVIOUS / 'hire-ready').read_text().strip()
    hire_source = PREVIOUS / 'reproduction_runs/upstream/Fed-HIRE/opensource'
    vade_source = PREVIOUS / 'FedVaDE_source'
    jobs = []
    for model, interpreter, source, freeze, pairs, seeds in [
        ('hire', hire, hire_source, 'hire-actual-freeze.txt',
         [(f'A_lr{rate}', ['--rate', rate]) for rate in ('0.0001', '0.03', '0.05')], range(2, 10)),
        ('fedvade', CPU, vade_source, 'fedvade-cpu-freeze.txt',
         [(f'B_{kind}', ['--schedule', kind]) for kind in ('author', 'epoch')], range(1, 5)),
    ]:
        try:
            references = {base: verify(model, interpreter, source, f'{base}_seed0', freeze)
                          for base, _ in pairs}
        except (ValueError, OSError, subprocess.CalledProcessError) as error:
            print(f'Blocked {model}: {error}', flush=True)
            continue
        jobs.extend((model, interpreter, source, f'{base}_seed{seed}', seed, args, references[base])
                    for seed in seeds for base, args in pairs)
    save(ROOT / 'queue-plan.json', {'planned': 32, 'verified': len(jobs),
                                    'deadline': options.deadline, 'runner_sha256': RUNNER_HASH})
    with ThreadPoolExecutor(max_workers=4) as workers:
        for _ in workers.map(lambda job: run(job, options.deadline), jobs):
            pass
    print('Round 3 queue finished.', flush=True)


if __name__ == '__main__':
    main()
