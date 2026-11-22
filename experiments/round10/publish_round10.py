"""Safely publish text-only Round 10 outputs from the remote server to GitHub."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

BASE = Path('/root/autodl-tmp/fedvade_project/reproduction_runs/round10_non_iid_generation')
WORK = BASE / 'publish_worktree'
KEY = Path('/root/.ssh/fuxian_round9_deploy')
REPO = 'git@github.com:GUOLIKey/fuxian.git'
SAFE = {'.py', '.sh', '.md', '.json', '.jsonl', '.csv', '.log', '.txt'}
EXCLUDED = {'raw', 'private_data', 'source', 'publish_worktree', '__pycache__', '.git'}


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
        if WORK.exists(): raise RuntimeError('Publication clone already exists; preserve for review')
        env = os.environ.copy()
        env['GIT_SSH_COMMAND'] = (f'ssh -i {KEY} -o IdentitiesOnly=yes '
                                  '-o BatchMode=yes -o StrictHostKeyChecking=yes')
        env['GIT_CONFIG_COUNT'] = '0'
        call(['git', 'clone', '--single-branch', '--branch', 'main', REPO, str(WORK)], env=env)
        previous = call(['git', 'rev-parse', 'HEAD'], cwd=WORK, env=env)
        target = WORK / 'experiments/round10'
        if target.exists(): raise RuntimeError('Round 10 is already published')
        target.mkdir(parents=True)
        copied = []
        for source in BASE.rglob('*'):
            if not source.is_file() or source.suffix not in SAFE:
                continue
            rel = source.relative_to(BASE)
            if any(part in EXCLUDED for part in rel.parts) or source.name.endswith('.diff'):
                continue
            if source.name == 'github_publish_status.json':
                continue
            dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            copied.append(str(rel))
        if not (target / 'ROUND10_REPORT.md').is_file():
            raise RuntimeError('Final report missing')
        result = json.loads((BASE / 'task_exit_status.json').read_text())
        with (BASE / 'noniid_baseline_metrics.csv').open(newline='') as stream:
            scored = list(csv.DictReader(stream))
        with (BASE / 'rare_mode_coverage.csv').open(newline='') as stream:
            rare = list(csv.DictReader(stream))
        severe = next((r for r in scored if r['model'] == 'fedvade_severe_seed0'), None)
        rare_severe = [float(r['rare_mode_recovery_proxy']) for r in rare
                       if r['model'] == 'fedvade_severe_seed0']
        section = ('\n## Round 10：Non-IID 生成与稀有模式诊断\n\n'
                   '本轮在 UCI Cardiotocography 三分类数据上固定训练、验证、测试划分，'
                   '检查 IID、中度及稀有类缺失的严重 Non-IID 场景。'
                   f"正式 FedVaDE 训练 {result.get('formal_runs', 0)} 次，"
                   f"本地与集中式参考训练 {result.get('reference_runs', 0)} 次。")
        if severe:
            section += (f"严重场景 seed=0 的固定 checkpoint 测试 ARI "
                        f"{float(severe['ari_test']):.4f}、Purity {float(severe['purity_test']):.4f}；")
        if rare_severe:
            section += f'稀有类生成覆盖代理均值 {sum(rare_severe)/len(rare_severe):.4f}。'
        section += ('生成样本没有真实类别标签，覆盖结果依赖独立参考分类器和距离门槛，'
                    '不能视为真实临床类别生成率。测试集聚类与重构指标使用预先封存的测试记录。'
                    '详见 [Round 10 报告](experiments/round10/ROUND10_REPORT.md)、'
                    '[原始训练日志](experiments/round10/severe_seed0.log)、'
                    '[逐类别生成指标](experiments/round10/classwise_generation_metrics.csv)、'
                    '[稀有类覆盖](experiments/round10/rare_mode_coverage.csv)、'
                    '[数据清单](experiments/round10/dataset_manifest.json)。'
                    '原始医疗数据、作者源码/权重、完整 checkpoint 和密钥仅保存在服务器。\n')
        readme = WORK / 'README.md'
        old = readme.read_text()
        if '## Round 10：Non-IID 生成与稀有模式诊断' in old:
            raise RuntimeError('README already has Round 10 section')
        readme.write_text(old.rstrip() + '\n' + section)
        call(['git', 'config', 'user.name', 'GUOLIKey'], cwd=WORK, env=env)
        call(['git', 'config', 'user.email', '190384412+GUOLIKey@users.noreply.github.com'], cwd=WORK, env=env)
        call(['git', 'add', 'README.md', 'experiments/round10'], cwd=WORK, env=env)
        staged = call(['git', 'diff', '--cached', '--name-only'], cwd=WORK, env=env).splitlines()
        if not staged or any(p != 'README.md' and not p.startswith('experiments/round10/')
                             for p in staged):
            raise RuntimeError('Unexpected staged path')
        call(['git', 'commit', '-m', 'Publish FedVaDE Round 10 generation diagnostics'], cwd=WORK, env=env)
        commit = call(['git', 'rev-parse', 'HEAD'], cwd=WORK, env=env)
        call(['git', 'fetch', 'origin', 'main'], cwd=WORK, env=env)
        if call(['git', 'rev-parse', 'origin/main'], cwd=WORK, env=env) != previous:
            raise RuntimeError('Remote branch changed during publication; local commit retained')
        call(['git', 'push', 'origin', 'HEAD:main'], cwd=WORK, env=env)
        verified = call(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=WORK, env=env).split()[0]
        if verified != commit: raise RuntimeError('Remote SHA verification failed')
        link = f'https://github.com/GUOLIKey/fuxian/commit/{commit}'
        atomic_json(status, {'status': 'PUSHED', 'commit': commit,
                    'verified_remote_sha': verified, 'commit_url': link,
                    'published_files': len(copied), 'finished_at': time.time()})
        with (BASE / 'ROUND10_REPORT.md').open('a') as stream:
            stream.write(f'\nGitHub publication verified: {link}\n')
        print(link, flush=True)
    except BaseException:
        atomic_json(status, {'status': 'FAILED', 'error': traceback.format_exc(),
                    'local_commit_dir': str(WORK), 'finished_at': time.time()})
        raise


if __name__ == '__main__':
    main()
