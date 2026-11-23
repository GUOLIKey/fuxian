"""Create the two report figures from the experiment CSVs."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXP = Path('/Users/cheeseowl/Desktop/科研/fuxian/experiments/multica_pretrained_2x2_v1')
RES = EXP / 'results'
FIG = EXP / 'figures'
FIG.mkdir(exist_ok=True)
COLORS = {'A': '#0072B2', 'B': '#E69F00', 'C': '#009E73', 'D': '#D55E00'}
LABELS = {'A': 'A: Author, original', 'B': 'B: Author, Multi-CA',
          'C': 'C: F01, original', 'D': 'D: F01, Multi-CA'}
GROUPS = ['A', 'B', 'C', 'D']


def style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color='#d9d9d9', linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors='#333333', labelsize=9)


def main():
    per = pd.read_csv(RES / 'per_seed.csv')
    curve = pd.read_csv(RES / 'training_curve_mean.csv')
    # Training ARI: mean line plus seed SD band.
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=160)
    for g in GROUPS:
        x = curve[curve.group == g].sort_values('round')
        ax.plot(x['round'], x['ari_mean'], color=COLORS[g], linewidth=2.0,
                marker='o', markersize=3.2, label=LABELS[g])
        ax.fill_between(x['round'], x['ari_mean'] - x['ari_sd'].fillna(0),
                        x['ari_mean'] + x['ari_sd'].fillna(0),
                        color=COLORS[g], alpha=0.12, linewidth=0)
    ax.set_xlabel('Formal round', fontsize=10)
    ax.set_ylabel('Client-record ARI (mean ± SD, 3 seeds)', fontsize=10)
    ax.set_title('Warm-start training: client-record ARI', fontsize=12, loc='left')
    ax.set_xlim(0, 305)
    ax.axhline(0, color='#777777', linewidth=0.8)
    ax.legend(frameon=False, fontsize=8.3, loc='lower right', ncol=2)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / 'training_ari.svg', bbox_inches='tight')
    fig.savefig(FIG / 'training_ari.png', bbox_inches='tight', dpi=220)
    plt.close(fig)

    # Final small multiples. Each panel has one measure and one axis.
    metrics = [
        ('final_ari', 'Final ARI', 'ARI'),
        ('nll', 'Final latent GMM NLL', 'NLL'),
        ('gen_mmd', 'Generated vs real MMD', 'MMD'),
        ('gen_swd', 'Generated vs real SWD', 'SWD'),
        ('n_supported_20', 'Micros with ≥20 real points', 'count'),
        ('n_ghost', 'Ghost micros (0 real points)', 'count'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.4), dpi=160)
    for ax, (metric, title, ylabel) in zip(axes.ravel(), metrics):
        means, sds = [], []
        for g in GROUPS:
            vals = pd.to_numeric(per[per.group == g][metric], errors='coerce').dropna()
            means.append(vals.mean() if len(vals) else np.nan)
            sds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
        x = np.arange(len(GROUPS))
        ax.bar(x, means, color=[COLORS[g] for g in GROUPS], width=0.66,
               yerr=sds, capsize=3, error_kw={'elinewidth': 0.9, 'ecolor': '#444444'})
        ax.set_xticks(x); ax.set_xticklabels(GROUPS)
        ax.set_title(title, fontsize=9.5, loc='left', pad=10)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.set_xlabel('Group', fontsize=8.5)
        top = max((m + (s if np.isfinite(s) else 0)) for m, s in zip(means, sds)
                  if np.isfinite(m))
        ax.set_ylim(0, top * 1.28 if top > 0 else 1.0)
        for i, (m, s) in enumerate(zip(means, sds)):
            if np.isfinite(m):
                ax.text(i, m + (s if np.isfinite(s) else 0) + 0.035 * ax.get_ylim()[1],
                        f'{m:.3g}', ha='center', va='bottom', fontsize=7.5, color='#333333')
        style(ax)
    fig.suptitle('Final frozen-evaluation metrics (mean ± SD, 3 seeds)', fontsize=11.5, x=0.02, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.6, w_pad=2.0)
    fig.savefig(FIG / 'final_metrics.svg', bbox_inches='tight')
    fig.savefig(FIG / 'final_metrics.png', bbox_inches='tight', dpi=220)
    plt.close(fig)
    print('FIGURES_DONE')


if __name__ == '__main__':
    main()
