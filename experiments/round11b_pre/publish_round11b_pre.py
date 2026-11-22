"""Publish text-only Round 11B-pre artifacts to GitHub without force-push."""
import json
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path

BASE = Path('/root/autodl-tmp/fedvade_project/reproduction_runs/round11b_pre_gmm_factorization')
WORK = BASE / 'publish_worktree'
KEY = Path('/root/.ssh/fuxian_round9_deploy')
REPO = 'git@github.com:GUOLIKey/fuxian.git'
SAFE = {'.py', '.sh', '.md', '.json', '.jsonl', '.csv', '.log', '.txt'}
EXCLUDED = {'raw', 'private_data', 'source', 'publish_worktree', '__pycache__', '.git', 'smoke'}


def atomic_json(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def call(argv, cwd=None, env=None):
    run = subprocess.run(argv, cwd=cwd, env=env, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if run.returncode:
        raise RuntimeError(f'{argv!r} failed: {run.stdout}')
    return run.stdout.strip()


def main():
    status_path = BASE / 'github_publish_status.json'
    atomic_json(status_path, {'status': 'PUSHING', 'started_at': time.time()})
    try:
        if not KEY.is_file() or KEY.stat().st_mode & 0o077:
            raise RuntimeError('Dedicated repository deploy key absent or unsafe permissions')
        if WORK.exists():
            raise RuntimeError('Publication clone already exists; preserve for review')
        env = os.environ.copy()
        env['GIT_SSH_COMMAND'] = (f'ssh -i {KEY} -o IdentitiesOnly=yes '
                                  '-o BatchMode=yes -o StrictHostKeyChecking=yes')
        env['GIT_CONFIG_COUNT'] = '0'
        call(['git', 'clone', '--single-branch', '--branch', 'main', REPO, str(WORK)], env=env)
        previous = call(['git', 'rev-parse', 'HEAD'], cwd=WORK, env=env)
        target = WORK / 'experiments/round11b_pre'
        if target.exists():
            raise RuntimeError('Round 11B-pre is already published')
        target.mkdir(parents=True)
        copied = []
        for source in BASE.rglob('*'):
            if not source.is_file() or source.suffix not in SAFE:
                continue
            rel = source.relative_to(BASE)
            if any(part in EXCLUDED for part in rel.parts):
                continue
            if source.name.endswith('.diff') or source.name.endswith('.command'):
                continue
            if source.name in ('github_publish_status.json', 'retry_publish.sh'):
                continue
            dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            copied.append(str(rel))
        required = ['ROUND11B_PRE_REPORT.md', 'experiment_protocol.md',
                    'source_checkpoint_audit.md', 'm_sweep_results.csv',
                    'covariance_ablation.csv', 'k_capacity_ablation.csv',
                    'component_quality_audit.csv',
                    'federated_statistics_aggregation.csv',
                    'moment_matching_diagnostics.csv', 'oracle_gap_analysis.csv',
                    'evaluation_protocol.md']
        missing = [name for name in required if not (target / name).is_file()]
        if missing:
            raise RuntimeError(f'Required Round 11B-pre outputs missing: {missing}')

        summary = json.loads((BASE / 'summary.json').read_text())
        section = (
            '\n## Round 11B-pre：联邦 GMM 因素分离与统计量聚合诊断\n\n'
            '本轮不训练神经网络，只在冻结 latent 上比较 M、协方差结构、K、'
            'Union Sampling EM、矩匹配与统计量初始化，并给出集中式 full GMM oracle gap。'
            f"选定 M={summary['selected_M']['selected_M']}，"
            f"协方差={summary['selected_covariance'].get('selected_config')}，"
            f"BIC 选定 K={summary['selected_K'].get('K')} {summary['selected_K'].get('covariance')}；"
            f"暂定决策 {summary.get('decision')}：{summary.get('decision_text', '')}。"
            '全部比较为验证集探索性诊断，不使用测试集选择模型。详见 '
            '[Round 11B-pre 报告](experiments/round11b_pre/ROUND11B_PRE_REPORT.md)、'
            '[实验协议](experiments/round11b_pre/experiment_protocol.md)、'
            '[M sweep](experiments/round11b_pre/m_sweep_results.csv)、'
            '[协方差消融](experiments/round11b_pre/covariance_ablation.csv)、'
            '[K 容量](experiments/round11b_pre/k_capacity_ablation.csv)、'
            '[联邦统计量聚合](experiments/round11b_pre/federated_statistics_aggregation.csv)。\n'
        )
        readme = WORK / 'README.md'
        old = readme.read_text()
        if '## Round 11B-pre' in old:
            raise RuntimeError('README already has Round 11B-pre section')
        readme.write_text(old.rstrip() + '\n' + section)

        call(['git', 'config', 'user.name', 'GUOLIKey'], cwd=WORK, env=env)
        call(['git', 'config', 'user.email', '190384412+GUOLIKey@users.noreply.github.com'],
             cwd=WORK, env=env)
        call(['git', 'add', 'README.md', 'experiments/round11b_pre'], cwd=WORK, env=env)
        staged = call(['git', 'diff', '--cached', '--name-only'], cwd=WORK, env=env).splitlines()
        if not staged or any(p != 'README.md' and not p.startswith('experiments/round11b_pre/')
                             for p in staged):
            raise RuntimeError('Unexpected staged path')
        call(['git', 'commit', '-m', 'Publish FedVaDE Round 11B-pre GMM diagnostics'],
             cwd=WORK, env=env)
        commit = call(['git', 'rev-parse', 'HEAD'], cwd=WORK, env=env)
        call(['git', 'fetch', 'origin', 'main'], cwd=WORK, env=env)
        if call(['git', 'rev-parse', 'origin/main'], cwd=WORK, env=env) != previous:
            raise RuntimeError('Remote branch changed during publication; local commit retained')
        call(['git', 'push', 'origin', 'HEAD:main'], cwd=WORK, env=env)
        verified = call(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=WORK, env=env).split()[0]
        if verified != commit:
            raise RuntimeError('Remote SHA verification failed')
        link = f'https://github.com/GUOLIKey/fuxian/commit/{commit}'
        atomic_json(status_path, {'status': 'PUSHED', 'commit': commit,
                                  'verified_remote_sha': verified, 'commit_url': link,
                                  'published_files': len(copied), 'finished_at': time.time()})
        (BASE / 'github_publish.log').write_text(link + '\n')
        print(link, flush=True)
    except BaseException:
        atomic_json(status_path, {'status': 'FAILED', 'error': traceback.format_exc(),
                                  'local_commit_dir': str(WORK), 'finished_at': time.time()})
        (BASE / 'github_publish_error.log').write_text(traceback.format_exc())
        (BASE / 'retry_publish.sh').write_text(
            '#!/bin/bash\nset -euo pipefail\n'
            f'cd {BASE}\n'
            '# Inspect/rename publish_worktree first if it exists; do not force push.\n'
            '/root/autodl-tmp/fedvade_project/.venv-cpu/bin/python publish_round11b_pre.py\n'
        )
        raise


if __name__ == '__main__':
    main()
