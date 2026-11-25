"""Plots for the frozen ACD experiment. Run with a matplotlib-capable Python."""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

EXP = Path('/root/autodl-tmp/fedvade_project/experiments/acd_f01_frozen_v1')
RES = EXP / 'results'
FIG = EXP / 'figures'
FIG.mkdir(exist_ok=True)
METHODS = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']
COLORS = {'A0': '#555555', 'A1': '#0072B2', 'A2': '#009E73',
          'A3': '#56B4E9', 'A4': '#D55E00', 'A5': '#E69F00'}
MULTI = {2, 4, 6, 9}


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
    pc = read_csv('per_client.csv')
    sm = {r['method']: r for r in read_csv('summary.csv')}
    # K by client (mean over seeds).
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=160)
    clients = sorted({int(r['client']) for r in pc})
    width = 0.13
    for j, m in enumerate(METHODS):
        vals, errs = [], []
        for c in clients:
            xs = [fl(r['K']) for r in pc if int(r['client']) == c and r['method'] == m]
            xs = [x for x in xs if x is not None]
            vals.append(np.mean(xs) if xs else np.nan)
            errs.append(np.std(xs) if len(xs) > 1 else 0)
        x = np.arange(len(clients)) + (j - 2.5) * width
        ax.bar(x, vals, width, yerr=errs, capsize=2,
               color=COLORS[m], label=m, error_kw={'elinewidth': 0.7})
    ax.set_yscale('log')
    ax.set_xticks(np.arange(len(clients)))
    ax.set_xticklabels([f'c{c}' for c in clients])
    ax.set_ylabel('K (mean over seeds, log)')
    ax.set_title('Local K by client', loc='left')
    ax.legend(frameon=False, ncol=6, fontsize=8)
    style(ax)
    fig.tight_layout(); fig.savefig(FIG / 'k_by_client.png'); plt.close(fig)

    # ARI on multi-class clients.
    fig, ax = plt.subplots(figsize=(8, 4.1), dpi=160)
    mcs = sorted(MULTI)
    for j, m in enumerate(METHODS):
        vals, errs = [], []
        for c in mcs:
            xs = [fl(r['ari']) for r in pc if int(r['client']) == c
                  and r['method'] == m and r['multi_class'] == 'True']
            xs = [x for x in xs if x is not None]
            vals.append(np.mean(xs) if xs else np.nan)
            errs.append(np.std(xs) if len(xs) > 1 else 0)
        ax.bar(np.arange(len(mcs)) + (j - 2.5) * width, vals, width,
               yerr=errs, capsize=2, color=COLORS[m], label=m,
               error_kw={'elinewidth': 0.7})
    ax.set_xticks(np.arange(len(mcs)))
    ax.set_xticklabels([f'c{c}' for c in mcs])
    ax.set_ylabel('ARI (multi-class clients)')
    ax.set_title('Local clustering ARI', loc='left')
    ax.legend(frameon=False, ncol=6, fontsize=8)
    style(ax)
    fig.tight_layout(); fig.savefig(FIG / 'ari_by_client_multi.png'); plt.close(fig)

    # Holdout NLL by method.
    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    vals, errs = [], []
    for m in METHODS:
        xs = [fl(r['holdout_nll']) for r in pc if r['method'] == m
              and r['reference_only'] == 'False']
        xs = [x for x in xs if x is not None]
        vals.append(np.mean(xs) if xs else np.nan)
        errs.append(np.std(xs) if len(xs) > 1 else 0)
    ax.bar(np.arange(len(METHODS)), vals, yerr=errs, capsize=3,
           color=[COLORS[m] for m in METHODS], width=0.65,
           error_kw={'elinewidth': 0.8})
    ax.set_xticks(np.arange(len(METHODS))); ax.set_xticklabels(METHODS)
    ax.set_ylabel('Holdout NLL (lower is better)')
    ax.set_title('Holdout NLL; A0 is a full-data reference', loc='left')
    style(ax)
    fig.tight_layout(); fig.savefig(FIG / 'holdout_nll.png'); plt.close(fig)

    # Generation MMD / SWD.
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), dpi=160)
    for ax, key, title in ((axes[0], 'gen_mmd_mean', 'Generation MMD'),
                           (axes[1], 'gen_swd_mean', 'Generation SWD')):
        vals = [fl(sm[m][key]) for m in METHODS]
        ax.bar(np.arange(len(METHODS)), vals,
               color=[COLORS[m] for m in METHODS], width=0.65)
        ax.set_xticks(np.arange(len(METHODS)))
        ax.set_xticklabels(METHODS)
        ax.set_title(title, loc='left')
        style(ax)
    fig.tight_layout(); fig.savefig(FIG / 'gen_mmd_swd.png'); plt.close(fig)

    # A4/A5 support composition by client.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=160, sharey=True)
    for ax, m in zip(axes, ('A4', 'A5')):
        sup20, mid, low5, ghost = [], [], [], []
        for c in clients:
            rs = [r for r in pc if int(r['client']) == c and r['method'] == m]
            def avg(k):
                xs = [fl(r[k]) for r in rs]
                xs = [x for x in xs if x is not None]
                return np.mean(xs) if xs else 0
            s20 = avg('n_supported_20')
            l20 = avg('n_low20')
            l5 = avg('n_low5')
            g = avg('n_ghost')
            sup20.append(s20); mid.append(max(l20 - l5, 0))
            low5.append(l5); ghost.append(g)
        x = np.arange(len(clients))
        ax.bar(x, sup20, color='#009E73', label='support >= 20')
        ax.bar(x, mid, bottom=sup20, color='#E69F00', label='5 <= support < 20')
        ax.bar(x, low5, bottom=np.array(sup20) + np.array(mid),
               color='#D55E00', label='0 < support < 5')
        ax.bar(x, ghost, bottom=np.array(sup20) + np.array(mid) + np.array(low5),
               color='#555555', label='ghost')
        ax.set_xticks(x); ax.set_xticklabels([f'c{c}' for c in clients])
        ax.set_title(f'{m} component support', loc='left')
        style(ax)
    axes[0].set_ylabel('components (mean over seeds)')
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / 'support_a4_a5.png'); plt.close(fig)

    # A4 vs A5 ARI scatter.
    fig, ax = plt.subplots(figsize=(5.4, 4.6), dpi=160)
    by = defaultdict(dict)
    for r in pc:
        if r['multi_class'] == 'True' and r['method'] in ('A4', 'A5'):
            by[(int(r['client']), int(r['seed']))][r['method']] = fl(r['ari'])
    for (c, s), d in by.items():
        if d.get('A4') is not None and d.get('A5') is not None:
            ax.scatter(d['A4'], d['A5'], s=34, label=f'c{c}')
    lim = [0, max([max(v.values()) for v in by.values() if len(v) == 2] + [0.05]) * 1.1]
    ax.plot(lim, lim, color='#777777', linewidth=1, linestyle='--')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('A4 ACD ARI'); ax.set_ylabel('A5 same-K GMM ARI')
    ax.set_title('ACD vs same-K ordinary GMM', loc='left')
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout(); fig.savefig(FIG / 'acd_vs_a5_ari.png'); plt.close(fig)

    # PCA visualization.
    pca_path = RES / 'pca_points.csv'
    if pca_path.exists():
        points = read_csv('pca_points.csv')
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), dpi=160)
        for ax, m in zip(axes, ('A2', 'A4')):
            sub = [r for r in points if r['method'] == m]
            if not sub:
                continue
            pc1 = np.array([float(r['pc1']) for r in sub])
            pc2 = np.array([float(r['pc2']) for r in sub])
            cl = np.array([int(r['cluster']) for r in sub])
            yt = np.array([int(r['true_label']) for r in sub])
            cmap = plt.get_cmap('tab20', max(len(np.unique(cl)), 1))
            ax.scatter(pc1, pc2, c=cl, cmap=cmap, s=16, alpha=0.85)
            for lab, marker in ((0, 'o'), (1, '^')):
                idx = yt == lab
                if idx.any():
                    ax.scatter(pc1[idx], pc2[idx], facecolors='none',
                               edgecolors='black', marker=marker, s=38,
                               linewidths=0.6, label=f'true {lab}')
            ax.set_title(f'{m} clustering (client {sub[0]["client"]})', loc='left')
            ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
            ax.legend(frameon=False, fontsize=8)
            style(ax)
        fig.tight_layout(); fig.savefig(FIG / 'pca_clusters.png'); plt.close(fig)
    print('FIGURES_DONE')


if __name__ == '__main__':
    main()
