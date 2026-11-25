"""Plots for the frozen-client global experiment."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

EXP = Path('/root/autodl-tmp/fedvade_project/experiments/acd_f01_global_v1')
RES = EXP / 'results'
FIG = EXP / 'figures'
FIG.mkdir(exist_ok=True)
GROUPS = ['A', 'B', 'C', 'D', 'E']
COLORS = {'A': '#555555', 'B': '#0072B2', 'C': '#009E73',
          'D': '#D55E00', 'E': '#E69F00'}
REF = 0.605326980742813


def read_csv(name):
    with (RES / name).open() as f:
        return list(csv.DictReader(f))


def fl(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color='#dddddd', linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)


def main():
    curves = read_csv('curves.csv')
    summary = {r['group']: r for r in read_csv('summary.csv')}
    per = read_csv('per_seed.csv')
    paired = read_csv('paired.csv')

    def curve(metric):
        out = {}
        for g in GROUPS:
            rows = sorted([r for r in curves if r['group'] == g
                           and r['metric'] == metric], key=lambda r: int(r['round']))
            x = np.array([int(r['round']) for r in rows])
            y = np.array([fl(r['mean']) if fl(r['mean']) is not None else np.nan
                          for r in rows], dtype=float)
            sd = np.array([fl(r['sd']) if fl(r['sd']) is not None else 0.0
                           for r in rows], dtype=float)
            out[g] = (x, y, sd)
        return out

    # Global ARI curve.
    fig, ax = plt.subplots(figsize=(7.4, 4.3), dpi=160)
    for g in GROUPS:
        x, y, sd = curve('ari')[g]
        ax.plot(x, y, marker='o', markersize=3.5, color=COLORS[g], label=g)
        ax.fill_between(x, y - sd, y + sd, color=COLORS[g], alpha=0.10, linewidth=0)
    ax.axhline(REF, color='black', linestyle='--', linewidth=0.9,
               label='frozen reference 0.6053')
    ax.set_xlabel('Server update round'); ax.set_ylabel('Global ARI (mean ± SD, 3 seeds)')
    ax.set_title('Global ARI under frozen-client local-GMM interventions', loc='left',
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    style(ax); fig.tight_layout(); fig.savefig(FIG / 'global_ari_curve.png'); plt.close(fig)

    # Final ARI bars at round 10.
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=160)
    vals = [fl(summary[g]['final_ari_mean']) for g in GROUPS]
    errs = [fl(summary[g]['final_ari_sd_experiment']) or 0 for g in GROUPS]
    ax.bar(np.arange(len(GROUPS)), vals, yerr=errs, capsize=3,
           color=[COLORS[g] for g in GROUPS], width=0.65,
           error_kw={'elinewidth': 0.8})
    ax.axhline(REF, color='black', linestyle='--', linewidth=0.9,
               label='frozen 0.6053')
    ax.set_xticks(np.arange(len(GROUPS))); ax.set_xticklabels(GROUPS)
    ax.set_ylabel('Global ARI at round 10'); ax.set_title('Round-10 global ARI', loc='left')
    ax.set_ylim(min(vals + [REF]) - 0.05, max(vals + [REF]) + 0.03)
    ax.legend(frameon=False, fontsize=8); style(ax)
    fig.tight_layout(); fig.savefig(FIG / 'final_ari.png'); plt.close(fig)

    # NMI / Purity / NLL / generation / loss curves.
    specs = [('nmi', 'Global NMI', 'global_nmi_curve.png'),
             ('purity', 'Global Purity', 'global_purity_curve.png'),
             ('global_nll', 'Global GMM NLL', 'global_nll_curve.png'),
             ('gen_mmd', 'Global generation MMD', 'global_mmd_curve.png'),
             ('gen_swd', 'Global generation SWD', 'global_swd_curve.png'),
             ('server_loss', 'Server loss', 'server_loss_curve.png')]
    for metric, title, name in specs:
        fig, ax = plt.subplots(figsize=(7.4, 3.8), dpi=160)
        for g in GROUPS:
            x, y, sd = curve(metric)[g]
            mask = np.isfinite(y)
            ax.plot(x[mask], y[mask], marker='o', markersize=3,
                    color=COLORS[g], label=g)
            ax.fill_between(x[mask], y[mask] - sd[mask], y[mask] + sd[mask],
                            color=COLORS[g], alpha=0.08, linewidth=0)
        ax.set_xlabel('Server update round'); ax.set_ylabel(title)
        ax.set_title(title + ' (mean ± SD, 3 seeds)', loc='left', fontsize=10)
        ax.legend(frameon=False, fontsize=8, ncol=3)
        style(ax); fig.tight_layout(); fig.savefig(FIG / name); plt.close(fig)

    # Paired ARI deltas at round 10.
    contrasts = ['D-A', 'B-A', 'D-B', 'D-C', 'D-E']
    fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=160)
    means, sds, pts = [], [], []
    for c in contrasts:
        r = next(x for x in paired if x['contrast'] == c and x['metric'] == 'ari')
        means.append(fl(r['mean_delta']))
        sds.append(fl(r['sd_delta']) or 0)
        vals = [d['delta'] for d in json.loads(r['values'])]
        pts.append(vals)
    x = np.arange(len(contrasts))
    ax.bar(x, means, yerr=sds, capsize=3, color=['#D55E00', '#0072B2',
                                                  '#009E73', '#56B4E9',
                                                  '#CC79A7'], width=0.62,
           error_kw={'elinewidth': 0.8})
    for i, vals in enumerate(pts):
        ax.scatter(np.full(len(vals), i), vals, color='black', s=16, zorder=3)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(contrasts)
    ax.set_ylabel('Δ Global ARI at round 10 (left − right)')
    ax.set_title('Paired arm differences; dots are the three experiment seeds',
                 loc='left', fontsize=9.5)
    style(ax); fig.tight_layout(); fig.savefig(FIG / 'paired_ari.png'); plt.close(fig)

    # Coverage: mean unique sampled components.
    cov_path = RES / 'coverage_summary.csv'
    if cov_path.exists():
        cov = read_csv('coverage_summary.csv')
        fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=160)
        for g in GROUPS:
            rows = sorted([r for r in cov if r['group'] == g],
                          key=lambda r: int(r['round']))
            if not rows:
                continue
            ax.plot([int(r['round']) for r in rows],
                    [fl(r['mean_unique_sampled']) for r in rows],
                    marker='o', markersize=3, color=COLORS[g], label=g)
        ax.set_xlabel('Server update round')
        ax.set_ylabel('Mean unique local components sampled')
        ax.set_title('Actual sampled coverage (100 samples total)', loc='left')
        ax.legend(frameon=False, fontsize=8, ncol=3)
        style(ax); fig.tight_layout(); fig.savefig(FIG / 'coverage.png'); plt.close(fig)
    print('FIGURES_DONE')


if __name__ == '__main__':
    main()
