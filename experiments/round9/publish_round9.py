"""Publish only curated text artifacts after Round 9; run on the remote server."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

ROOT = Path('/root/autodl-tmp/fedvade_project')
HERE = ROOT / 'reproduction_runs/round9_fedvade_optimization'
WORK = HERE / 'publish_worktree'
REPO = 'git@github.com:GUOLIKey/fuxian.git'
KEY = Path('/root/.ssh/fuxian_round9_deploy')
SAFE = {'.py', '.sh', '.md', '.json', '.jsonl', '.csv', '.log', '.txt'}
EXCLUDE = {'source', 'publish_worktree', '__pycache__', '.git'}
SSH = f'ssh -i {KEY} -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes'


def atomic_json(path, value):
    temp = Path(str(path) + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def run(argv, cwd=None, env=None):
    result = subprocess.run(argv, cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{argv!r} failed ({result.returncode}): {result.stdout}')
    return result.stdout.strip()


def main():
    status_path = HERE / 'github_publish_status.json'
    atomic_json(status_path, {'status': 'PUSHING', 'started_at': time.time()})
    try:
        if not KEY.is_file() or KEY.stat().st_mode & 0o077:
            raise RuntimeError('Dedicated deploy key missing or permissions too broad')
        env = os.environ.copy()
        env['GIT_SSH_COMMAND'] = SSH
        env['GIT_CONFIG_COUNT'] = '0'
        if WORK.exists():
            raise RuntimeError(f'Publish worktree already exists: {WORK}')
        run(['git', 'clone', '--branch', 'main', '--single-branch', REPO, str(WORK)], env=env)
        prior = run(['git', 'rev-parse', 'HEAD'], cwd=WORK, env=env)
        dest = WORK / 'experiments/round9'
        if dest.exists():
            raise RuntimeError('Round 9 already exists in remote repository')
        dest.mkdir(parents=True)
        copied = []
        for path in HERE.rglob('*'):
            if not path.is_file() or path.suffix not in SAFE:
                continue
            relative = path.relative_to(HERE)
            if any(part in EXCLUDE for part in relative.parts):
                continue
            if path.name == 'github_publish_status.json' or path.name.endswith('.diff'):
                continue
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied.append(str(relative))
        if not (dest / 'ROUND9_REPORT.md').is_file():
            raise RuntimeError('Report missing; refusing incomplete publication')
        import csv
        ranking_path = HERE / 'best_checkpoint_ranking.csv'
        best = None
        if ranking_path.is_file():
            with ranking_path.open(newline='') as stream:
                best = next((row for row in csv.DictReader(stream)
                             if row['training_seed'] == '0'), None)
        readme = WORK / 'README.md'
        original = readme.read_text()
        marker = '## Round 9：初始化与训练稳定性搜索'
        if marker in original:
            raise RuntimeError('README already contains Round 9 section')
        exit_status = json.loads((HERE / 'task_exit_status.json').read_text())
        run_count = exit_status.get('new_runs', 0)
        best_line = ('最佳 seed=0 探索性 checkpoint：ARI '
                     f"{float(best['ari_mean']):.4f} ± {float(best['ari_std']):.4f}，"
                     f"Purity {float(best['purity_mean']):.4f} ± {float(best['purity_std']):.4f}，"
                     f"{best['config']} 第 {best['epoch']} 轮。") if best else '没有可报告的完整 checkpoint。'
        section = ('\n' + marker + '\n\n'
                   '本轮在服务器运行 9 组初始化/预训练配置（其中 1 组复用旧结果）和 '
                   f'8 组超参数配置；已完整记录的新增训练 {run_count} 次。'
                   + ('有实验失败，见报告和原始日志。' if exit_status['exit_code'] else '')
                   + best_line + '\n\n'
                   'Round 7 F01 第 100 轮参考 ARI 为 0.6140；作者权重参考约 0.6295。'
                   '所有按评价标签挑选的配置或 checkpoint 均为探索性结果，'
                   '10 个推断种子不是 10 个独立训练种子。'
                   '完整配置、逐轮记录、原始日志和失败情况见 '
                   '[Round 9 报告](experiments/round9/ROUND9_REPORT.md)、'
                   '[实验计划](experiments/round9/experiment_plan.json)、'
                   '[Checkpoint 排名](experiments/round9/best_checkpoint_ranking.csv) 和 '
                   '[任务日志](experiments/round9/task.log)。'
                   '权重、数据、作者源码和密钥仅留在服务器。\n')
        readme.write_text(original.rstrip() + '\n' + section)
        run(['git', 'config', 'user.name', 'GUOLIKey'], cwd=WORK, env=env)
        run(['git', 'config', 'user.email',
             '190384412+GUOLIKey@users.noreply.github.com'], cwd=WORK, env=env)
        run(['git', 'add', 'README.md', 'experiments/round9'], cwd=WORK, env=env)
        staged = run(['git', 'diff', '--cached', '--name-only'], cwd=WORK, env=env).splitlines()
        if not staged or any(not (name == 'README.md' or name.startswith('experiments/round9/'))
                             for name in staged):
            raise RuntimeError('Unexpected staged path')
        run(['git', 'commit', '-m', 'Publish FedVaDE Round 9 optimization results'], cwd=WORK, env=env)
        commit = run(['git', 'rev-parse', 'HEAD'], cwd=WORK, env=env)
        run(['git', 'fetch', 'origin', 'main'], cwd=WORK, env=env)
        remote_before = run(['git', 'rev-parse', 'origin/main'], cwd=WORK, env=env)
        if remote_before != prior:
            raise RuntimeError('Remote main changed during publication; keep local commit for review')
        run(['git', 'push', 'origin', 'HEAD:main'], cwd=WORK, env=env)
        remote_after = run(['git', 'ls-remote', 'origin', 'refs/heads/main'],
                           cwd=WORK, env=env).split()[0]
        if remote_after != commit:
            raise RuntimeError(f'Push verification mismatch {remote_after} != {commit}')
        link = f'https://github.com/GUOLIKey/fuxian/commit/{commit}'
        atomic_json(status_path, {'status': 'PUSHED', 'commit': commit,
                    'verified_remote_sha': remote_after, 'commit_url': link,
                    'files': len(copied), 'finished_at': time.time()})
        with (HERE / 'ROUND9_REPORT.md').open('a') as stream:
            stream.write(f'\nGitHub publication verified: {link}\n')
        print(link, flush=True)
    except BaseException:
        error = traceback.format_exc()
        atomic_json(status_path, {'status': 'FAILED', 'error': error,
                    'local_commit_dir': str(WORK), 'finished_at': time.time()})
        raise


if __name__ == '__main__':
    main()
