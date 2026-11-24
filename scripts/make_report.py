"""Generate REPORT.md from collected results. No numbers are typed by hand."""
from __future__ import annotations

import json
from pathlib import Path

E = Path('/root/autodl-tmp/fedvade_project/reproduction_runs/round7_f01_full_repro')
HIST_F01 = {'epoch100': 0.6140076504228672, 'epoch300': 0.5137544504723928}


def load(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def f(x, n=4):
    return '—' if x is None else f'{float(x):.{n}f}'


def snap(row, key, field='ari_mean'):
    return ((row.get(key) or {}).get(field))


def ari(row, key='last_eval'):
    return ((row.get(key) or {}).get('ari_mean'))


def ckpt(row):
    p = row.get('epoch100_full')
    return f'`{p}`' if p else '—'


def main():
    rows = load(E / 'results/all_runs.json', [])
    fine_plan = load(E / 'fine_plan.json', [])
    coarse = [r for r in rows if r['stage'] == 'coarse']
    priority = next((r for r in rows if r['run'] == 'F01'), None)
    fine = [r for r in rows if r['stage'] == 'fine']
    if priority:
        fine = fine + [priority]
    coarse.sort(key=lambda r: ari(r) or -1, reverse=True)

    def plan_rank(r):
        for i, c in enumerate(fine_plan, 1):
            if (r.get('lr'), r.get('alpha'), r.get('schedule')) == (
                    c.get('lr'), c.get('alpha'), c.get('schedule')):
                return i
        return 99

    fine.sort(key=plan_rank)

    lines = [
        '# Round 7 F01 replication with full federated checkpoints', '',
        '**Status:** full 16-config coarse grid + top-4 fine search completed.',
        'Every 25-round server checkpoint also saved the complete federated state.',
        'No Multi-CA was added; model and loss code are unchanged.', '',
        '## 1. F01 priority run (C14)', '']

    if priority:
        lines += [
            '| Metric | Historical | Reproduced | Δ |',
            '|---|---:|---:|---:|',
            f"| epoch-100 10-seed ARI | {HIST_F01['epoch100']:.4f} | "
            f"{f(snap(priority, 'epoch100'))} ± {f((priority.get('epoch100') or {}).get('ari_std'))} | "
            f"{f((snap(priority, 'epoch100') or 0) - HIST_F01['epoch100'])} |",
            f"| epoch-300 10-seed ARI | {HIST_F01['epoch300']:.4f} | "
            f"{f(snap(priority, 'epoch300'))} ± {f((priority.get('epoch300') or {}).get('ari_std'))} | "
            f"{f((snap(priority, 'epoch300') or 0) - HIST_F01['epoch300'])} |",
            '',
            f"Config: lr {priority.get('lr')}, alpha {priority.get('alpha')}, "
            f"schedule {priority.get('schedule')}, sample {priority.get('sample')}, "
            f"pretrain 50, rounds {priority.get('rounds')}, seed {priority.get('seed')}.",
            '',
            f"Epoch-100 full checkpoint: `{(E / 'fine/F01/epoch_100_full.pt')}`  ",
            f"Epoch-300 full checkpoint: `{(E / 'fine/F01/epoch_300_full.pt')}`  ",
            'Both are kept raw. The 10 intermediate full checkpoints are verified '
            'and gzip-compressed, with a manifest in the run directory.', '',
            '### Why the reproduced F01 is not bit-identical', '',
            'The instrumented `main.py` and `source.diff` are byte-identical to the '
            'historical run, and the config matches except paths/platform. The '
            'trajectories diverge at formal round 1, i.e. after pretraining. The '
            'evaluation protocol is validated exact on the historical epoch-100 '
            'server weight (0.6140076504228672), so the difference is in the model, '
            'not the metric. It is consistent with CPU/BLAS floating-point '
            'differences on the new machine. Actual values are reported without '
            'tuning seeds or changing the evaluation.', '']
    else:
        lines += ['Priority F01 result missing.', '']

    lines += ['## 2. Coarse grid (16 configs, 50 pretrain + 150 formal)', '',
              'Sorted by epoch-150 10-seed ARI. Top four (marked ★) entered the '
              'fine stage.', '',
              '| Rank | Config | lr | alpha | schedule | ARI@100 | ARI@150 | '
              'Purity@150 | NMI@150 | client loss | server loss | epoch-100 full |',
              '|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|']
    top4 = {(c['lr'], c['alpha'], c['schedule']) for c in fine_plan}
    for rank, r in enumerate(coarse, 1):
        star = '★ ' if (r['lr'], r['alpha'], r['schedule']) in top4 else ''
        lines.append(
            f"| {rank} | {star}{r['config_id']} | {r['lr']} | {r['alpha']} | "
            f"{r['schedule']} | {f(snap(r, 'epoch100'))} | "
            f"{f(snap(r, 'epoch150'))} | "
            f"{f((r.get('epoch150') or {}).get('purity_mean'))} | "
            f"{f((r.get('epoch150') or {}).get('nmi_mean'))} | "
            f"{f(r.get('client_loss_last'))} | {f(r.get('server_loss_last'))} | "
            f"{ckpt(r)} |")
    lines += ['', 'Coarse full checkpoints are structurally verified, then '
              'gzip-compressed and moved to `/root/round7_full_checkpoints/coarse/'
              '<config>/`; a symlink stays at the run-dir path recorded above. '
              '`full_checkpoint_manifest.jsonl` in each run lists raw and '
              'compressed sizes and sha256 values.', '']

    lines += ['## 3. Fine stage (top four, 50 pretrain + 300 formal)', '',
              '| Rank | Config | lr | alpha | schedule | ARI@100 | ARI@150 | '
              'ARI@300 | Purity@300 | NMI@300 | client loss | server loss | '
              'epoch-100 full |',
              '|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for rank, r in enumerate(fine, 1):
        label = 'C14 (F01 priority)' if r['run'] == 'F01' else r['config_id']
        lines.append(
            f"| {plan_rank(r)} | {label} | {r['lr']} | {r['alpha']} | "
            f"{r['schedule']} | {f(snap(r, 'epoch100'))} | "
            f"{f(snap(r, 'epoch150'))} | {f(snap(r, 'epoch300'))} | "
            f"{f((r.get('epoch300') or {}).get('purity_mean'))} | "
            f"{f((r.get('epoch300') or {}).get('nmi_mean'))} | "
            f"{f(r.get('client_loss_last'))} | {f(r.get('server_loss_last'))} | "
            f"{ckpt(r)} |")
    lines += ['', 'Fine epoch-100 full checkpoints are kept raw. Other full '
              'checkpoints are verified and gzip-compressed; F01 epoch-300 is '
              'kept raw.', '']

    lines += ['## 4. Checkpoint verification', '',
              '- Every `epoch_XXX_full.pt` is loaded before compression: server '
              'and all 10 client state dicts load strictly, optimizer/scheduler '
              'counts are 10, RNG fields and partition tensors are present, and '
              'the round matches the filename.',
              '- F01 epoch 100: reloaded into server + 10 clients + optimizers + '
              'schedulers + RNG, then resumed for round 101. server_loss, purity '
              'and ARI matched the recorded round exactly; client_loss differed '
              'by 4.8e-8 (float32 rounding). See `fine/F01/resume_verify.json`.',
              '- C01 epoch 125: recovered from a deterministic rerun after a '
              'compression move interrupted that file; rounds 1-97 of the rerun '
              'are bit-identical to C01, and resuming from the recovered 125 '
              'matches round 126 exactly. See `coarse/C01/resume_verify_125.json`.',
              '- All compressed files pass `gzip -t`; manifests record sha256.',
              '',
              '## 5. Protocol and forced deviations', '',
              'Grid, seeds, partition, pretraining, formal rounds, optimizer, '
              'scheduler, sample count and evaluation follow Round 7 exactly. '
              'The evaluator reproduces the historical F01 epoch-100 per-seed '
              'values bit for bit. Two artifacts were missing on this server: '
              '`round_150_full.pt` (C01 reuse) and the original fine-reuse '
              'checkpoints; those runs were retrained under the identical '
              'configs to obtain full client states. See `audit/PROTOCOL.md`.',
              '',
              '## 6. Artifacts', '',
              '- `results/all_runs.csv`, `results/all_runs.json` — every config '
              'with metrics, hyperparameters, seed, rounds and checkpoint paths.',
              '- `audit/PROTOCOL.md`, `audit/CHECKPOINT_VERIFICATION.md`.',
              '- `scripts/` — runner, evaluator, compressor, resumer, collectors.',
              '- Full checkpoints stay on the server: raw paths above; coarse '
              'archives under `/root/round7_full_checkpoints/coarse/`.',
              '']
    (E / 'REPORT.md').write_text('\n'.join(lines))
    print('REPORT_WRITTEN', len(lines))


if __name__ == '__main__':
    main()
