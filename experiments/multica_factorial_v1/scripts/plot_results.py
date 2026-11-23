"""Static SVG figures for the Stage A report (no matplotlib dependency)."""
from __future__ import annotations

import csv
import html
import math
from pathlib import Path

BASE = Path('/root/autodl-tmp/fedvade_project/experiments/multica_factorial_v1')
RESULTS = BASE / 'results'
FIGURES = BASE / 'figures'
COLORS = {'A': '#2a78d6', 'B': '#eb6834', 'C': '#1baf7a', 'D': '#eda100'}
LABELS = {
    'A': 'A Author / original',
    'B': 'B Author / Multi-CA',
    'C': 'C F01 / original',
    'D': 'D F01 / Multi-CA',
}
TEXT = '#0b0b0b'
MUTED = '#52514e'
GRID = '#deddd8'
SURFACE = '#fcfcfb'


def read_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def fnum(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def esc(value):
    return html.escape(str(value))


def svg_line_chart(path, rows, metric, title, ylabel, digits=3, note=''):
    groups = 'ABCD'
    by = {}
    for row in rows:
        group = row['group']; rnd = int(row['round']); value = fnum(row.get(metric))
        if value is None:
            continue
        by.setdefault((group, rnd), []).append(value)
    all_values = [v for vals in by.values() for v in vals]
    if not all_values:
        return
    # Per-round mean and min/max bands.
    series = {}
    for group in groups:
        points = []
        for rnd in sorted({r for g, r in by if g == group}):
            vals = by[(group, rnd)]
            points.append((rnd, sum(vals) / len(vals), min(vals), max(vals)))
        if points:
            series[group] = points
    ymin = min(min(p[2] for p in pts) for pts in series.values())
    ymax = max(max(p[3] for p in pts) for pts in series.values())
    pad = (ymax - ymin) * 0.08 or 0.01
    ymin -= pad; ymax += pad
    xmin, xmax = 1, max(p[0] for pts in series.values() for p in pts)
    W, H = 980, 560
    L, R, T, B = 78, 176, 70, 62
    pw, ph = W - L - R, H - T - B

    def sx(x): return L + (x - xmin) / max(xmax - xmin, 1) * pw
    def sy(y): return T + (ymax - y) / max(ymax - ymin, 1e-12) * ph
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{esc(title)}</title>',
        f'<desc id="desc">{esc(note)}</desc>',
        f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
        f'<text x="{L}" y="36" font-family="Inter, Arial, sans-serif" font-size="20" font-weight="600" fill="{TEXT}">{esc(title)}</text>',
        f'<text x="{L}" y="57" font-family="Inter, Arial, sans-serif" font-size="12" fill="{MUTED}">BC, 3 seeds; line = mean, band = min/max. {esc(note)}</text>',
    ]
    # Horizontal grid + y ticks.
    for k in range(5):
        y = ymin + (ymax - ymin) * k / 4
        yy = sy(y)
        out.append(f'<line x1="{L}" y1="{yy:.1f}" x2="{L+pw}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{L-10}" y="{yy+4:.1f}" text-anchor="end" font-family="Inter, Arial, sans-serif" font-size="11" fill="{MUTED}">{y:.{digits}f}</text>')
    # X ticks.
    for rnd in (1, 50, 100, 150, 200, 250, 300):
        if rnd > xmax:
            continue
        xx = sx(rnd)
        out.append(f'<text x="{xx:.1f}" y="{H-B+22}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="11" fill="{MUTED}">{rnd}</text>')
    out.append(f'<text x="{L+pw/2:.1f}" y="{H-18}" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="12" fill="{MUTED}">formal round</text>')
    out.append(f'<text x="20" y="{T+ph/2:.1f}" transform="rotate(-90 20 {T+ph/2:.1f})" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="12" fill="{MUTED}">{esc(ylabel)}</text>')
    # Series bands and lines.
    for group, pts in series.items():
        color = COLORS[group]
        upper = ' '.join(f'{sx(x):.1f},{sy(hi):.1f}' for x, _, _, hi in pts)
        lower = ' '.join(f'{sx(x):.1f},{sy(lo):.1f}' for x, _, lo, _ in reversed(pts))
        out.append(f'<polygon points="{upper} {lower}" fill="{color}" fill-opacity="0.12" stroke="none"/>')
        d = ' '.join(f'{sx(x):.1f},{sy(mean):.1f}' for x, mean, _, _ in pts)
        out.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><title>{esc(LABELS[group])}</title></polyline>')
        x, mean, _, _ = pts[-1]
        out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(mean):.1f}" r="4" fill="{color}" stroke="{SURFACE}" stroke-width="2"/>')
        out.append(f'<text x="{sx(x)+10:.1f}" y="{sy(mean)+4:.1f}" font-family="Inter, Arial, sans-serif" font-size="12" fill="{TEXT}">{esc(group)}</text>')
    # Legend.
    ly = T + 8
    for i, group in enumerate(groups):
        yy = ly + i * 24
        out.append(f'<line x1="{L+pw+20}" y1="{yy}" x2="{L+pw+48}" y2="{yy}" stroke="{COLORS[group]}" stroke-width="3"/>')
        out.append(f'<text x="{L+pw+56}" y="{yy+4}" font-family="Inter, Arial, sans-serif" font-size="12" fill="{TEXT}">{esc(LABELS[group])}</text>')
    out.append('</svg>')
    Path(path).write_text('\n'.join(out) + '\n')


def svg_paired_effects(path, rows):
    metrics = [('unique_ari', 'Unique-row ARI'), ('server_gmm_nll', 'Server GMM NLL')]
    effects = [('B_minus_A', 'B − A'), ('D_minus_C', 'D − C'),
               ('C_minus_A', 'C − A'), ('D_minus_B', 'D − B'),
               ('interaction_DC_minus_BA', 'Interaction')]
    W, H = 1000, 620
    L, R, T, B = 80, 40, 72, 66
    panel_w = (W - L - R - 60) / 2
    panel_h = H - T - B
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
        '<title id="title">Paired effects, same training seed</title>',
        '<desc id="desc">Stage A paired differences with error bars (SD across three same-seed paired contrasts).</desc>',
        f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
        f'<text x="{L}" y="34" font-family="Inter, Arial, sans-serif" font-size="20" font-weight="600" fill="{TEXT}">Paired effects, same training seed</text>',
        f'<text x="{L}" y="55" font-family="Inter, Arial, sans-serif" font-size="12" fill="{MUTED}">Mean ± SD across seeds 0/1/2. Positive ARI = better; negative NLL = better.</text>',
    ]
    by = {(r['metric'], r['comparison']): r for r in rows} if rows and 'comparison' in rows[0] else {}
    # The summary CSV stores one row per metric with effect_mean/_sd columns.
    summary = {r['metric']: r for r in rows}
    for panel, (metric, title) in enumerate(metrics):
        px = L + panel * (panel_w + 60)
        py = T
        vals = []
        for key, _ in effects:
            r = summary.get(metric, {})
            vals.append((fnum(r.get(key + '_mean')), fnum(r.get(key + '_sd'))))
        finite = [v for v, _ in vals if v is not None]
        lo = min(finite + [0.0]); hi = max(finite + [0.0])
        pad = (hi - lo) * 0.15 or 0.01
        lo -= pad; hi += pad
        def sy(v): return py + (hi - v) / max(hi - lo, 1e-12) * panel_h
        out.append(f'<text x="{px}" y="{py-18}" font-family="Inter, Arial, sans-serif" font-size="14" font-weight="600" fill="{TEXT}">{esc(title)}</text>')
        # zero line + grid
        out.append(f'<line x1="{px}" y1="{sy(0):.1f}" x2="{px+panel_w}" y2="{sy(0):.1f}" stroke="{MUTED}" stroke-width="1" stroke-dasharray="4 4"/>')
        for k in range(3):
            v = lo + (hi - lo) * k / 2
            yy = sy(v)
            out.append(f'<line x1="{px}" y1="{yy:.1f}" x2="{px+panel_w}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="1"/>')
            out.append(f'<text x="{px-8}" y="{yy+4:.1f}" text-anchor="end" font-family="Inter, Arial, sans-serif" font-size="10" fill="{MUTED}">{v:.3f}</text>')
        bar_h = panel_h / (len(effects) * 1.7)
        for i, ((key, label), (mean, sd)) in enumerate(zip(effects, vals)):
            yy = py + (i + 0.5) * panel_h / len(effects)
            color = '#2a78d6' if mean is not None and mean >= 0 else '#eb6834'
            if mean is None:
                color = MUTED
            x0 = px + panel_w * 0.5
            x1 = x0 + (mean or 0) / max(abs(hi), abs(lo), 1e-12) * panel_w * 0.36
            out.append(f'<line x1="{px}" y1="{yy:.1f}" x2="{px+panel_w}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="1" stroke-opacity="0.45"/>')
            out.append(f'<rect x="{min(x0,x1):.1f}" y="{yy-bar_h/2:.1f}" width="{abs(x1-x0):.1f}" height="{bar_h:.1f}" rx="4" fill="{color}" fill-opacity="0.9"><title>{esc(label)}: {mean}</title></rect>')
            if sd is not None:
                sx0 = x0 + (mean - sd) / max(abs(hi), abs(lo), 1e-12) * panel_w * 0.36
                sx1 = x0 + (mean + sd) / max(abs(hi), abs(lo), 1e-12) * panel_w * 0.36
                out.append(f'<line x1="{sx0:.1f}" y1="{yy:.1f}" x2="{sx1:.1f}" y2="{yy:.1f}" stroke="{TEXT}" stroke-width="1.2"/>')
                out.append(f'<line x1="{sx0:.1f}" y1="{yy-4:.1f}" x2="{sx0:.1f}" y2="{yy+4:.1f}" stroke="{TEXT}" stroke-width="1.2"/>')
                out.append(f'<line x1="{sx1:.1f}" y1="{yy-4:.1f}" x2="{sx1:.1f}" y2="{yy+4:.1f}" stroke="{TEXT}" stroke-width="1.2"/>')
            out.append(f'<text x="{px+6}" y="{yy+4:.1f}" font-family="Inter, Arial, sans-serif" font-size="11" fill="{TEXT}">{esc(label)}</text>')
            if mean is not None:
                out.append(f'<text x="{px+panel_w-4}" y="{yy+4:.1f}" text-anchor="end" font-family="Inter, Arial, sans-serif" font-size="11" fill="{MUTED}">{mean:.3f}</text>')
    out.append('</svg>')
    Path(path).write_text('\n'.join(out) + '\n')


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    curves = read_rows(RESULTS / 'training_curves.csv')
    effects = read_rows(RESULTS / 'paired_effects_summary.csv')
    svg_line_chart(FIGURES / 'training_ari_by_group.svg', curves, 'unique_ari',
                   'Training trajectory: unique-row ARI', 'ARI',
                   note='Author-protocol in-sample; fixed round 300 is primary.')
    svg_line_chart(FIGURES / 'training_gmm_nll_by_group.svg', curves, 'server_gmm_nll',
                   'Training trajectory: server GMM NLL', 'NLL (lower is better)',
                   note='NLL on unique rows; no independent BC test split.')
    svg_paired_effects(FIGURES / 'paired_effects.svg', effects)
    print('FIGURES_OK')


if __name__ == '__main__':
    main()
