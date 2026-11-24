"""Read-only panel for the Round 7 F01 full-checkpoint reproduction."""
from __future__ import annotations

import json
import sys
from pathlib import Path

E = Path(sys.argv[1])


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def last_row(run):
    p = Path(run) / 'rounds.jsonl'
    if not p.exists():
        return None
    lines = [x for x in p.read_text().splitlines() if x.strip()]
    return json.loads(lines[-1]) if lines else None


def main():
    prog = (E / 'progress.txt')
    if prog.exists():
        parts = prog.read_text().strip().split('|')
    else:
        parts = ['none', '0', '0', '--', '', 'PENDING']
    stage, pos, total, cfg, path, status = (parts + [''] * 6)[:6]
    run = Path(path) if path else None

    print('Round 7 F01 full-checkpoint reproduction | NEW server')
    print(f'Stage: {stage} | config {cfg} | {pos}/{total} | {status}')
    if run:
        print(f'Run dir: {run}')

    row = last_row(run) if run else None
    rounds = None
    if run:
        result = read_json(run / 'result.json')
        rounds = (result or {}).get('config', {}).get('rounds')
    if row:
        print(f"Round: {row['round']} / {rounds or '?'} | "
              f"ARI {row['ari']:.4f} | purity {row['purity']:.4f} | "
              f"client_loss {row['client_loss']:.4f} | "
              f"server_loss {row['server_loss']:.4f}")
    elif rounds:
        print(f'Round: 0 / {rounds} | no formal round yet (pretraining)')
    else:
        print('Round: --')

    evaluated = []
    for sub in ('coarse', 'fine'):
        for d in sorted((E / sub).glob('*')):
            ev = read_json(d / 'evaluation_summary.json')
            if ev:
                best = ev.get('best_exploratory_ari', {})
                evaluated.append((best.get('ari_mean', -1), d.name,
                                  best.get('epoch'), 'best'))
                last = ev.get('last', {})
                evaluated.append((last.get('ari_mean', -1), d.name,
                                  last.get('epoch'), 'last'))
    if evaluated:
        best = max(evaluated)
        print(f'Best evaluated ARI (10 seeds): {best[0]:.4f} '
              f'({best[1]} epoch {best[2]}, {best[3]})')
        top = sorted({(round(v[0], 6), v[1]) for v in evaluated
                      if v[3] == 'best'}, reverse=True)[:4]
        print('Top evaluated runs: ' + ' | '.join(f'{n}={a:.4f}' for a, n in top))
    else:
        print('Best evaluated ARI (10 seeds): --')

    completed = 0
    for sub in ('coarse', 'fine'):
        for d in (E / sub).glob('*'):
            r = read_json(d / 'result.json')
            if r and r.get('status') == 'ok':
                completed += 1
    print(f'Completed run dirs: {completed}')

    if run and run.exists():
        raw = sorted(run.glob('epoch_*_full.pt'))
        gz = sorted(run.glob('epoch_*_full.pt.gz'))
        expected = (int(rounds) // 25) if rounds else 0
        e100 = run / 'epoch_100_full.pt'
        e100gz = run / 'epoch_100_full.pt.gz'
        if e100.exists():
            e100txt = f'raw {e100.stat().st_size / 1e6:.0f} MB'
        elif e100gz.exists():
            e100txt = f'compressed {e100gz.stat().st_size / 1e6:.0f} MB'
        else:
            e100txt = 'not saved yet'
        print(f'Full checkpoints (active): {len(raw)} raw, {len(gz)} compressed '
              f'| expected {expected} | epoch100: {e100txt}')
    priority = E / 'fine' / 'F01' / 'epoch_100_full.pt'
    if priority.exists():
        print(f'Priority F01 epoch100_full: {priority} '
              f'({priority.stat().st_size / 1e6:.0f} MB, raw)')
    elif (E / 'fine' / 'F01' / 'epoch_100_full.pt.gz').exists():
        print(f"Priority F01 epoch100_full: "
              f"{E / 'fine' / 'F01' / 'epoch_100_full.pt.gz'} (compressed)")

    err = E / 'error.txt'
    if err.exists():
        print('DRIVER ERROR: ' + err.read_text().strip().splitlines()[-1])


if __name__ == '__main__':
    main()
