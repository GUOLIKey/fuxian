"""Publish text-only Round 11A artifacts to GitHub without force-push."""
import json
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path

BASE = Path('/root/autodl-tmp/fedvade_project/reproduction_runs/round11a_frozen_diagnostics')
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
    status = BASE / 'github_publish_status.json'
    atomic_json(status, {'status': 'PUSHING', 'started_at': time.time()})
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
        target = WORK / 'experiments/round11a'
        if target.exists():
            raise RuntimeError('Round 11A is already published')
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
        required = ['ROUND11A_REPORT.md', 'evaluation_protocol.md', 'checkpoint_manifest.csv',
                    'evaluator_calibration.csv', 'reconstruction_diagnostics.csv',
                    'global_vs_union_mog.csv', 'gmm_k_comparison.csv']
        missing = [name for name in required if not (target / name).is_file()]
        if missing:
            raise RuntimeError(f'Required Round 11A outputs missing: {missing}')

        summary = json.loads((BASE / 'summary.json').read_text())
        section = (
            '\n## Round 11A：冻结模型生成瓶颈诊断\n\n'
            '本轮不训练新的 VaDE/FedVaDE，只复用 Round 10 冻结权重，审计稀有评价器，并比较真实重建、'
            '全局 GMM 与客户端并集 MoG、以及 K=3/6/10 的冻结 latent GMM 容量。'
            f"评价器主规则 tau={summary['tau']:.2f}、radius={summary['radius']:.4f}；"
            f"验证稀有正对照 {summary.get('evaluator_positive_val_joint')}，"
            f"常见负对照 {summary.get('evaluator_negative_val_joint')}。"
            f"探索性结论：{summary.get('recommendation', '证据不足')}。"
            '测试集比较仅为探索性，不构成新的独立测试成绩。详见 '
            '[Round 11A 报告](experiments/round11a/ROUND11A_REPORT.md)、'
            '[评价协议](experiments/round11a/evaluation_protocol.md)、'
            '[评价器校准](experiments/round11a/evaluator_calibration.csv)、'
            '[全局 vs 并集](experiments/round11a/global_vs_union_mog.csv)、'
            '[GMM 容量](experiments/round11a/gmm_k_comparison.csv)。\n'
        )
        readme = WORK / 'README.md'
        old = readme.read_text()
        if '## Round 11A' in old:
            raise RuntimeError('README already has Round 11A section')
        readme.write_text(old.rstrip() + '\n' + section)

        call(['git', 'config', 'user.name', 'GUOLIKey'], cwd=WORK, env=env)
        call(['git', 'config', 'user.email', '190384412+GUOLIKey@users.noreply.github.com'], cwd=WORK, env=env)
        call(['git', 'add', 'README.md', 'experiments/round11a'], cwd=WORK, env=env)
        staged = call(['git', 'diff', '--cached', '--name-only'], cwd=WORK, env=env).splitlines()
        if not staged or any(p != 'README.md' and not p.startswith('experiments/round11a/') for p in staged):
            raise RuntimeError('Unexpected staged path')
        call(['git', 'commit', '-m', 'Publish FedVaDE Round 11A frozen diagnostics'], cwd=WORK, env=env)
        commit = call(['git', 'rev-parse', 'HEAD'], cwd=WORK, env=env)
        call(['git', 'fetch', 'origin', 'main'], cwd=WORK, env=env)
        if call(['git', 'rev-parse', 'origin/main'], cwd=WORK, env=env) != previous:
            raise RuntimeError('Remote branch changed during publication; local commit retained')
        call(['git', 'push', 'origin', 'HEAD:main'], cwd=WORK, env=env)
        verified = call(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=WORK, env=env).split()[0]
        if verified != commit:
            raise RuntimeError('Remote SHA verification failed')
        link = f'https://github.com/GUOLIKey/fuxian/commit/{commit}'
        atomic_json(status, {'status': 'PUSHED', 'commit': commit,
                             'verified_remote_sha': verified, 'commit_url': link,
                             'published_files': len(copied), 'finished_at': time.time()})
        (BASE / 'github_publish.log').write_text(link + '\n')
        print(link, flush=True)
    except BaseException:
        atomic_json(status, {'status': 'FAILED', 'error': traceback.format_exc(),
                             'local_commit_dir': str(WORK), 'finished_at': time.time()})
        (BASE / 'github_publish_error.log').write_text(traceback.format_exc())
        (BASE / 'retry_publish.sh').write_text(
            '#!/bin/bash\nset -euo pipefail\n'
            f'cd {BASE}\n'
            '# Inspect/rename publish_worktree first if it exists; do not force push.\n'
            '/root/autodl-tmp/fedvade_project/.venv-cpu/bin/python publish_round11a.py\n'
        )
        raise


if __name__ == '__main__':
    main()
