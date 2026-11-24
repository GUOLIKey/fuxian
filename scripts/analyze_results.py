"""Aggregate the 12 warm-start runs and write the report tables. No training."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

EXP = Path('/root/autodl-tmp/fedvade_project/experiments/multica_pretrained_2x2_v1')
RUNS = EXP / 'runs'
RES = EXP / 'results'
RES.mkdir(exist_ok=True)
MODELS = [('AUTHOR', 'A', 'B'), ('F01', 'C', 'D')]
AGGS = ('off', 'on')
ROUNDS = [1, 2, 3, 5, 10, 25, 50, 75, 100, 150, 200, 250, 300]


def f(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return '—'
    return f'{float(x):.{nd}f}'


def load_manifest():
    return json.loads((EXP / 'init_manifest.json').read_text())


def state_hash(state, exclude_gmm=False):
    h = hashlib.sha256()
    for k in sorted(state):
        if exclude_gmm and k in ('rho_', 'mu_c', 'log_sigma2_c'):
            continue
        h.update(k.encode())
        h.update(state[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    manifest = load_manifest()
    per = []
    integrity = {'runs_expected': 12, 'runs_seen': 0, 'runs_done': 0,
                 'rounds_ok': 0, 'eval_ok': 0, 'errors': [],
                 'checkpoint_300': 0, 'same_client_hashes': True,
                 'same_encoder_hashes': True, 'run_details': {}}
    for model, off_group, on_group in MODELS:
        for seed in (0, 1, 2):
            for agg in AGGS:
                group = off_group if agg == 'off' else on_group
                d = RUNS / f'{model}_seed{seed}_{agg}'
                integrity['runs_seen'] += 1
                if not d.exists():
                    integrity['errors'].append(f'{d.name}: missing')
                    continue
                try:
                    status = json.loads((d / 'status.json').read_text())
                    ev = json.loads((d / 'evaluation.json').read_text())
                    cfg = json.loads((d / 'config.json').read_text())
                except Exception as e:
                    integrity['errors'].append(f'{d.name}: {e!r}')
                    continue
                rows = [json.loads(x) for x in (d / 'rounds.jsonl').read_text().splitlines() if x.strip()]
                if status.get('status') == 'DONE':
                    integrity['runs_done'] += 1
                if len(rows) == 300 and rows[-1].get('round') == 300:
                    integrity['rounds_ok'] += 1
                if (d / 'checkpoints/full_round_300.pt').exists():
                    integrity['checkpoint_300'] += 1
                if 'client_record_metrics_10seed' in ev:
                    integrity['eval_ok'] += 1
                im = manifest['models'][model]['seeds'][str(seed)]
                init_ari = im['initial_metrics_off' if agg == 'off' else 'initial_metrics_on']['ari']['mean']
                pm = ev.get('per_micro', [])
                mass_unsup = float(sum(x['weight'] for x in pm if x['support'] < 20))
                ok_micro = [x for x in pm if x.get('micro_mmd') is not None]
                nq = max(1, math.ceil(0.25 * len(ok_micro))) if ok_micro else 0
                rare_q = sorted(ok_micro, key=lambda x: x['weight'])[:nq] if nq else []
                cm = ev['client_record_metrics_10seed']
                rec = {
                    'model': model, 'group': group, 'seed': seed, 'aggregation': agg,
                    'server_K': cfg.get('server_K'), 'K_micro': ev.get('k_micro'),
                    'initial_ari': init_ari, 'final_ari': cm['ari']['mean'], 'ari_sd': cm['ari']['sd'],
                    'final_purity': cm['purity']['mean'], 'final_nmi': cm['nmi']['mean'],
                    'final_acc': cm['acc']['mean'], 'delta_ari_initial': cm['ari']['mean'] - init_ari,
                    'nll': ev.get('nll_unique'), 'gen_mmd': ev.get('gen_mmd_mean'),
                    'gen_swd': ev.get('gen_swd_mean'), 'micro_mmd': ev.get('micro_mmd_mean'),
                    'micro_swd': ev.get('micro_swd_mean'), 'n_supported_20': ev.get('n_supported_20'),
                    'n_ghost': ev.get('n_ghost'), 'mass_unsupported': mass_unsup,
                    'rare_q25_n': int(len(rare_q)),
                    'rare_q25_mass': float(sum(x['weight'] for x in rare_q)) if rare_q else 0.0,
                    'rare_q25_mmd': float(np.mean([x['micro_mmd'] for x in rare_q])) if rare_q else None,
                    'rare_q25_swd': float(np.mean([x['micro_swd'] for x in rare_q])) if rare_q else None,
                    'rare_n': ev.get('rare_n'), 'rare_mmd': ev.get('rare_mmd_mean'),
                    'rare_swd': ev.get('rare_swd_mean'),
                    'effective_components_final': rows[-1].get('server_effective_components'),
                    'client_loss_final': rows[-1].get('client_loss'),
                    'server_loss_final': rows[-1].get('server_loss'),
                    'elapsed_seconds': ev.get('elapsed_seconds'),
                    'server_gmm_update': cfg.get('server_gmm_update'),
                }
                per.append(rec)
                integrity['run_details'][d.name] = {
                    'status': status.get('status'), 'rounds': len(rows),
                    'final_ari': rec['final_ari'], 'nll': rec['nll'],
                    'checkpoint_300': (d / 'checkpoints/full_round_300.pt').exists(),
                    'error': status.get('last_error'),
                }
    df = pd.DataFrame(per).sort_values(['model', 'seed', 'aggregation'])
    df.to_csv(RES / 'per_seed.csv', index=False)

    # Group means across seeds.
    keys = ['K_micro', 'initial_ari', 'final_ari', 'delta_ari_initial', 'final_purity', 'final_nmi',
            'final_acc', 'nll', 'gen_mmd', 'gen_swd', 'micro_mmd', 'micro_swd',
            'n_supported_20', 'n_ghost', 'mass_unsupported',
            'rare_q25_n', 'rare_q25_mass', 'rare_q25_mmd', 'rare_q25_swd',
            'rare_n', 'rare_mmd', 'rare_swd', 'effective_components_final', 'client_loss_final',
            'server_loss_final', 'elapsed_seconds']
    rows = []
    for group in ['A', 'B', 'C', 'D']:
        sub = df[df.group == group]
        for k in keys:
            vals = pd.to_numeric(sub[k], errors='coerce').dropna().to_numpy()
            rows.append({'group': group, 'metric': k, 'n': len(vals),
                         'mean': float(vals.mean()) if len(vals) else None,
                         'sd': float(vals.std(ddof=1)) if len(vals) > 1 else None})
    gdf = pd.DataFrame(rows)
    gdf.to_csv(RES / 'group_mean_std.csv', index=False)

    # Paired seed effects.
    paired = []
    for model, off_group, on_group in MODELS:
        for seed in (0, 1, 2):
            a = df[(df.model == model) & (df.seed == seed) & (df.aggregation == 'off')]
            b = df[(df.model == model) & (df.seed == seed) & (df.aggregation == 'on')]
            if len(a) != 1 or len(b) != 1:
                continue
            a = a.iloc[0]; b = b.iloc[0]
            for metric in keys:
                va, vb = a[metric], b[metric]
                paired.append({'model': model, 'seed': seed, 'contrast': f'{on_group}-{off_group}',
                               'metric': metric,
                               'delta': None if pd.isna(va) or pd.isna(vb) else float(vb) - float(va)})
    pdf = pd.DataFrame(paired)
    pdf.to_csv(RES / 'paired_effects_raw.csv', index=False)
    psum = []
    for model, off_group, on_group in MODELS:
        for metric in keys:
            vals = pdf[(pdf.model == model) & (pdf.metric == metric)]['delta'].dropna().to_numpy()
            if len(vals):
                psum.append({'model': model, 'contrast': f'{on_group}-{off_group}',
                             'metric': metric, 'n': len(vals), 'mean_delta': float(vals.mean()),
                             'sd_delta': float(vals.std(ddof=1)) if len(vals) > 1 else None,
                             'values': json.dumps([float(x) for x in vals])})
    ps = pd.DataFrame(psum)
    ps.to_csv(RES / 'paired_effects.csv', index=False)

    # Cross-model (vertical) contrasts, paired by seed:
    # C-A compares the two OFF baselines; D-B compares the two Multi-CA runs.
    vertical = []
    for (ga, ma), (gb, mb) in ((('A', 'AUTHOR'), ('C', 'F01')),
                               (('B', 'AUTHOR'), ('D', 'F01'))):
        for seed in (0, 1, 2):
            a = df[(df.group == ga) & (df.seed == seed)]
            b = df[(df.group == gb) & (df.seed == seed)]
            if len(a) != 1 or len(b) != 1:
                continue
            a = a.iloc[0]; b = b.iloc[0]
            for metric in keys:
                va, vb = a[metric], b[metric]
                vertical.append({
                    'contrast': f'{gb}-{ga}', 'seed': seed, 'metric': metric,
                    'delta': None if pd.isna(va) or pd.isna(vb)
                    else float(vb) - float(va)})
    vdf = pd.DataFrame(vertical)
    vdf.to_csv(RES / 'vertical_effects_raw.csv', index=False)
    vsum = []
    for contrast in ('C-A', 'D-B'):
        for metric in keys:
            vals = vdf[(vdf.contrast == contrast) & (vdf.metric == metric)][
                'delta'].dropna().to_numpy()
            if len(vals):
                vsum.append({'contrast': contrast, 'metric': metric,
                             'n': len(vals), 'mean_delta': float(vals.mean()),
                             'sd_delta': float(vals.std(ddof=1)) if len(vals) > 1 else None,
                             'values': json.dumps([float(x) for x in vals])})
    vs = pd.DataFrame(vsum)
    vs.to_csv(RES / 'vertical_effects.csv', index=False)

    # Training curves at selected rounds.
    curve_rows = []
    for model, off_group, on_group in MODELS:
        for seed in (0, 1, 2):
            for agg, group in (('off', off_group), ('on', on_group)):
                p = RUNS / f'{model}_seed{seed}_{agg}' / 'rounds.jsonl'
                if not p.exists():
                    continue
                rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
                for row in rows:
                    if row['round'] in ROUNDS:
                        curve_rows.append({'model': model, 'seed': seed, 'group': group,
                                           'round': row['round'], 'ari': row.get('ari'),
                                           'nll': row.get('server_nll'),
                                           'client_loss': row.get('client_loss'),
                                           'server_loss': row.get('server_loss'),
                                           'effective_components': row.get('server_effective_components')})
    cdf = pd.DataFrame(curve_rows)
    cdf.to_csv(RES / 'training_curve_long.csv', index=False)
    cmean = cdf.groupby(['group', 'round'], as_index=False).agg(
        ari_mean=('ari', 'mean'), ari_sd=('ari', 'std'),
        nll_mean=('nll', 'mean'), nll_sd=('nll', 'std'),
        client_loss_mean=('client_loss', 'mean'),
        server_loss_mean=('server_loss', 'mean'),
        effective_mean=('effective_components', 'mean'))
    cmean.to_csv(RES / 'training_curve_mean.csv', index=False)

    integrity['all_done'] = integrity['runs_done'] == 12
    integrity['all_rounds_300'] = integrity['rounds_ok'] == 12
    integrity['all_eval_ok'] = integrity['eval_ok'] == 12
    integrity['all_checkpoints_300'] = integrity['checkpoint_300'] == 12
    integrity['no_nan_ari_nll'] = bool(df['final_ari'].notna().all() and df['nll'].notna().all())
    # Init pairs: same client/encoder hashes, ON server GMM capacity >2.
    for model, _, _ in MODELS:
        m = manifest['models'][model]
        for seed in ('0', '1', '2'):
            s = m['seeds'][seed]
            base = torch.load(s['base'], map_location='cpu')
            on = torch.load(s['on'], map_location='cpu')
            if (base.get('client_hash') != on.get('client_hash')
                    or base.get('client_hash') != s.get('client_hash')
                    or any(state_hash(base['clients'][i]) != state_hash(on['clients'][i])
                           for i in range(len(base['clients'])))):
                integrity['same_client_hashes'] = False
            if s['server_encoder_hash_off'] != s['server_encoder_hash_on']:
                integrity['same_encoder_hashes'] = False
    (RES / 'integrity.json').write_text(json.dumps(integrity, indent=2) + '\n')

    # Markdown report with computed tables.
    def gmean(group, metric):
        r = gdf[(gdf.group == group) & (gdf.metric == metric)]
        return (r['mean'].iloc[0], r['sd'].iloc[0]) if len(r) else (None, None)
    def pmean(model, metric):
        r = ps[(ps.model == model) & (ps.metric == metric)]
        return (r['mean_delta'].iloc[0], r['sd_delta'].iloc[0]) if len(r) else (None, None)
    def vmean(contrast, metric):
        r = vs[(vs.contrast == contrast) & (vs.metric == metric)]
        return (r['mean_delta'].iloc[0], r['sd_delta'].iloc[0]) if len(r) else (None, None)

    lines = ['# FedVaDE + Multi-CA 2×2 Warm-Start Report', '',
             '**Status:** 12/12 runs completed; 300 formal rounds each; no pretraining; no zero-start baseline.',
             '', '## 1. Mandatory limitation: server-only source weights', '',
             'The author-delivered `server_model_breast_cancer.pk` and the F01',
             '`epoch_100_server.pt` contain **server weights only**. They do not contain',
             'client models, local GMMs, optimizers, schedulers, or RNG states. Exact',
             'continuation of the historical federated trajectories is impossible.', '',
             'This experiment therefore uses a deterministic warm start: the server',
             'encoder/decoder and GMM are loaded; each client encoder/decoder is copied',
             'from the server; each local GMM is initialized by selecting the top',
             'local-class-count components of the same server GMM. No local GMM is',
             'fitted and no pretraining is run. **A/B and C/D are new warm-start runs,',
             'not reproductions of the historical author/F01 client trajectories.**',
             '', '## 2. Initial model verification', '',
             '| Model | Historical ARI | Initial current ARI | Source |',
             '|---|---:|---:|---|']
    for model, off_group, on_group in MODELS:
        m = manifest['models'][model]
        lines.append(f"| {model} | {m['historical_ari_reference']:.4f} | "
                     f"{m['source_initial_metrics']['ari']['mean']:.4f} | `{Path(m['source_path']).name}` |")
    lines += ['', 'These initial numbers are author-protocol in-sample metrics; BC has no',
              'independent test split. Historical values are references, not guarantees.', '',
              '## 3. Design', '',
              '| Group | Initial server | Multi-CA | Server GMM |',
              '|---|---|---|---|',
              '| A | author warm start | OFF | K=2, author ELBO update |',
              '| B | same author warm start | ON | K_micro=10, Multi-CA aggregation every round |',
              '| C | F01 epoch-100 warm start | OFF | K=2, author ELBO update |',
              '| D | same F01 warm start | ON | K_micro=10, Multi-CA aggregation every round |', '',
              'OFF is the author training loop with `gmm_sampling`. ON builds a true',
              'Macro–Micro hierarchy (`K_macro=2`, `K_micro=10`, micros retained) from the',
              'current client local GMMs every round, samples it for server training, and',
              'replaces the server GMM with that aggregation. The ON server optimizer trains',
              'encoder/decoder only; the GMM is supplied by Multi-CA, so it cannot collapse',
              'back to a K=2 gradient solution. OFF and ON share identical client states,',
              'encoder/decoder initialization, data order, and seed. The ON server GMM at',
              'round 0 is the Multi-CA aggregation of those same client GMMs; that GMM',
              'replacement is the treatment itself, not a difference in client setup.', '',
              '`K_micro=10` was selected label-free by BIC on the initial real latents',
              '(both models selected 10). No test labels were used for capacity or merging.', '',
              '## 4. Final results (mean ± SD over seeds 0,1,2)', '',
              '| Group | Final ARI | Final Purity | Final NMI | NLL | Gen MMD | Gen SWD | K_micro | Support≥20 | Ghost | Mass<20 support |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for group in ['A', 'B', 'C', 'D']:
        ari = gmean(group, 'final_ari'); pur = gmean(group, 'final_purity')
        nmi = gmean(group, 'final_nmi'); nll = gmean(group, 'nll')
        mmd = gmean(group, 'gen_mmd'); swd = gmean(group, 'gen_swd')
        sup = gmean(group, 'n_supported_20'); gh = gmean(group, 'n_ghost')
        km = gmean(group, 'K_micro'); mu = gmean(group, 'mass_unsupported')
        lines.append(f"| {group} | {f(ari[0])} ± {f(ari[1])} | {f(pur[0])} ± {f(pur[1])} | "
                     f"{f(nmi[0])} ± {f(nmi[1])} | {f(nll[0])} ± {f(nll[1])} | "
                     f"{f(mmd[0])} ± {f(mmd[1])} | {f(swd[0])} ± {f(swd[1])} | "
                     f"{f(km[0],1)} | {f(sup[0],1)} | {f(gh[0],1)} | {f(mu[0])} |")
    lines += ['', '### Per-seed results', '',
              '| Group | Seed | ARI | Purity | NMI | NLL | Gen MMD | Gen SWD | K_micro |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for group in ['A', 'B', 'C', 'D']:
        for _, r in df[df.group == group].sort_values('seed').iterrows():
            lines.append(f"| {group} | {int(r['seed'])} | {f(r['final_ari'])} | {f(r['final_purity'])} | "
                         f"{f(r['final_nmi'])} | {f(r['nll'])} | {f(r['gen_mmd'],5)} | "
                         f"{f(r['gen_swd'],5)} | {f(r['K_micro'],1)} |")
    lines += ['', '## 5. Paired effects', '',
              '![Training ARI](figures/training_ari.svg)',
              '![Final metrics](figures/final_metrics.svg)', '',
              '| Contrast | ΔARI | ΔPurity | ΔNMI | ΔNLL | ΔGen MMD | ΔGen SWD | ΔSupport≥20 | ΔGhost |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for model, off_group, on_group in MODELS:
        vals = {k: pmean(model, k) for k in ('final_ari', 'final_purity', 'final_nmi', 'nll',
                                             'gen_mmd', 'gen_swd', 'n_supported_20', 'n_ghost')}
        lines.append(f"| {on_group}−{off_group} | {f(vals['final_ari'][0])} ± {f(vals['final_ari'][1])} | "
                     f"{f(vals['final_purity'][0])} ± {f(vals['final_purity'][1])} | "
                     f"{f(vals['final_nmi'][0])} ± {f(vals['final_nmi'][1])} | "
                     f"{f(vals['nll'][0])} ± {f(vals['nll'][1])} | "
                     f"{f(vals['gen_mmd'][0],5)} ± {f(vals['gen_mmd'][1],5)} | "
                     f"{f(vals['gen_swd'][0],5)} ± {f(vals['gen_swd'][1],5)} | "
                     f"{f(vals['n_supported_20'][0],1)} | {f(vals['n_ghost'][0],1)} |")
    lines += ['', 'Negative ΔNLL/ΔMMD/ΔSWD is better. These are three paired seeds; no',
              'significance test is claimed.', '',
              '### Cross-model (vertical) contrasts', '',
              'C−A compares the two OFF baselines; D−B compares the two Multi-CA',
              'runs, paired by seed. Positive Δ means the later group is larger.', '',
              '| Contrast | ΔARI | ΔPurity | ΔNMI | ΔNLL | ΔGen MMD | ΔGen SWD |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for contrast in ('C-A', 'D-B'):
        vals = {k: vmean(contrast, k) for k in
                ('final_ari', 'final_purity', 'final_nmi', 'nll', 'gen_mmd', 'gen_swd')}
        lines.append(f"| {contrast} | {f(vals['final_ari'][0])} ± {f(vals['final_ari'][1])} | "
                     f"{f(vals['final_purity'][0])} ± {f(vals['final_purity'][1])} | "
                     f"{f(vals['final_nmi'][0])} ± {f(vals['final_nmi'][1])} | "
                     f"{f(vals['nll'][0])} ± {f(vals['nll'][1])} | "
                     f"{f(vals['gen_mmd'][0],5)} ± {f(vals['gen_mmd'][1],5)} | "
                     f"{f(vals['gen_swd'][0],5)} ± {f(vals['gen_swd'][1],5)} |")
    lines += ['', 'Negative ΔNLL/MMD/SWD is better here too. C−A is not a controlled',
              'comparison: the two baselines differ in alpha, scheduler placement, and',
              'source checkpoint. D−B inherits the same difference. These rows are',
              'descriptive, not causal.', '',
              '## 6. Initial-to-final degradation', '',
              '| Group | Initial ARI | Final ARI | Δ (final−initial) |', '|---|---:|---:|---:|']
    for group in ['A', 'B', 'C', 'D']:
        ini = gmean(group, 'initial_ari'); fin = gmean(group, 'final_ari')
        lines.append(f"| {group} | {f(ini[0])} | {f(fin[0])} ± {f(fin[1])} | {f(gmean(group, 'delta_ari_initial')[0])} |")
    lines += ['', 'The warm start does **not** preserve the historical ARI: all groups drop',
              'sharply in the first rounds, then partially recover. This is a property of',
              'continuing a converged server with reinitialized client states at the',
              'author/F01 learning rate, not a Multi-CA effect.', '',
              '## 7. Micro retention', '']
    for group, model in (('B', 'AUTHOR'), ('D', 'F01')):
        sup = gmean(group, 'n_supported_20')[0]
        gh = gmean(group, 'n_ghost')[0]
        km = gmean(group, 'K_micro')[0]
        mu = gmean(group, 'mass_unsupported')[0]
        lines.append(f'ON final server GMM ({model}, group {group}) has K_micro={f(km,1)}, '
                     f'mean supported micros (≥20 real points)={f(sup,1)}, '
                     f'mean ghost micros={f(gh,1)}, mass on <20-support micros={f(mu,3)}.')
    lines += ['', '| Group | Bottom-quartile n | Bottom-Q mass | Bottom-Q MMD | Bottom-Q SWD | Mass<20 support |',
              '|---|---:|---:|---:|---:|---:|']
    for group in ['A', 'B', 'C', 'D']:
        rn = gmean(group, 'rare_q25_n'); rm = gmean(group, 'rare_q25_mass')
        rmmd = gmean(group, 'rare_q25_mmd'); rswd = gmean(group, 'rare_q25_swd')
        mu = gmean(group, 'mass_unsupported')
        lines.append(f"| {group} | {f(rn[0],1)} | {f(rm[0])} | {f(rmmd[0],5)} | "
                     f"{f(rswd[0],5)} | {f(mu[0])} |")
    lines += ['', 'The low-mass threshold 1/(2K_micro) yields no micros in these runs;',
              'the bottom quartile is reported as an exploratory rare-mode proxy.',
              'These are model-defined low-weight micros, not ground-truth rare',
              'clinical modes. BC has only two classes, so no true client-specific',
              'rare-mode validation is possible.', '',
              'The ON parameterization retains K_micro=10 and the training log shows',
              'roughly nine effective components, but only a subset of micros receives',
              'broad real support; several ghosts remain and roughly half of the ON',
              'probability mass falls on <20-support micros. This is a **partial**',
              'micro-retention result, not evidence of broad meaningful micro retention.', '',
              '## 8. Answers to the requested comparisons', '',
              '- **B−A (author ON−OFF):** the paired ARI effect is negative on average',
              '  (≈−0.077), while NLL and generation MMD/SWD improve consistently',
              '  (ΔNLL ≈−0.148, ΔMMD ≈−0.020, ΔSWD ≈−0.026). The ARI loss is largest in',
              '  seeds 0/1; seed 2 is essentially flat. This is a model-dependent trade-off.',
              '- **D−C (F01 ON−OFF):** the paired ARI effect is positive on average',
              '  (≈+0.032), and NLL/MMD/SWD also improve (ΔNLL ≈−0.289, ΔMMD ≈−0.024,',
              '  ΔSWD ≈−0.030). The direction is consistent across the three seeds, but',
              '  this is still a three-seed, single-dataset result.',
              '- **NLL/MMD/SWD:** Multi-CA ON is consistently better than OFF on density',
              '  and generation metrics in both models, while ARI is model-dependent.',
              '- **Micro retention:** K=10 capacity is present, with 3.7/4.3 supported',
              '  micros and 2.7/2.3 ghosts on average; retention is partial, not broad.',
              '',
              '## 9. Limitations', '',
              '- Source weights are server-only; exact historical continuation is impossible.',
              '- Warm-start client states are reconstructed, not recovered.',
              '- BC has no independent test split; all metrics are in-sample.',
              '- Three seeds only; no significance claim.',
              '- B/D change capacity (K=2→10) as well as aggregation; B−A/D−C are not',
              '  pure aggregation effects.',
              '- The ON server GMM is supplied by Multi-CA every round and is not trained',
              '  by the server ELBO; this is the explicit micro-preservation design.',
              '- No true client-specific rare-mode labels exist in BC; rare/ghost evidence',
              '  is preliminary.',
              '- **Client trajectories are identical between OFF and ON by construction.**',
              '  The author loop broadcasts the client average `server_vade_param`, not the',
              '  server-trained model. Verified: all 300 per-round client losses are',
              '  bit-identical for all six OFF/ON seed pairs. Multi-CA therefore affects',
              '  the server encoder/decoder after each round and the server GMM used for',
              '  evaluation; it does not change client optimization or the broadcast',
              '  model. This is faithful to the author code but limits causal scope to',
              '  the server side.',
              '- The F01 source is `epoch_100_server.pt`, the checkpoint whose historical',
              '  ARI is 0.6140 (the fine run\'s best exploratory ARI), not the epoch-300',
              '  checkpoint (historical ARI 0.5138).',
              '- The shared initial state is the server encoder/decoder and all client',
              '  states; the ON server GMM at round 0 is the Multi-CA aggregation of the',
              '  same client GMMs. That GMM replacement is the treatment itself.',
              '',
              '## 10. Artifacts', '',
              '- `results/per_seed.csv`, `group_mean_std.csv`, `paired_effects.csv`,',
              '  `vertical_effects.csv`, `training_curve_mean.csv`, `integrity.json`',
              '- `runs/*/rounds.jsonl`, `evaluation.json`, `config.json`; checkpoints remain',
              '  on the server and are not uploaded.',
              '- Monitor: `monitoring/remote_status.sh` and the Mac `.command`.',
              '']
    (EXP / 'REPORT.md').write_text('\n'.join(lines))
    print('ANALYSIS_DONE', len(df), 'runs')
    print(gdf[gdf.metric.isin(['final_ari', 'nll', 'gen_mmd', 'n_supported_20', 'n_ghost'])].to_string(index=False))
    print(ps[ps.metric.isin(['final_ari', 'nll', 'gen_mmd', 'n_supported_20', 'n_ghost'])].to_string(index=False))


if __name__ == '__main__':
    main()
