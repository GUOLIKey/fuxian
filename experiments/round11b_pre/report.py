"""Generate ROUND11B_PRE_REPORT.md from controlled diagnostic results."""
import csv
import json
import math
import time
from collections import defaultdict

import numpy as np

from common import BASE, update_progress


def read(name):
    path = BASE / name
    if not path.is_file():
        return []
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def num(value):
    try:
        if value in ('', None, 'None', 'nan'):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def avg(rows, key):
    vals = [num(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    return float(np.mean(vals)) if vals else None


def fmt(value, nd=4):
    if isinstance(value, str):
        value = num(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return '—'
    return f'{value:.{nd}f}'


def main():
    m_rows = read('m_sweep_results.csv')
    cov_rows = read('covariance_summary.csv')
    k_rows = read('k_capacity_summary.csv')
    audit_rows = read('component_quality_audit.csv')
    agg_rows = read('oracle_gap_analysis.csv')
    moment_rows = read('moment_matching_diagnostics.csv')
    selected_m = json.loads((BASE / 'selected_m.json').read_text())
    selected_cov = json.loads((BASE / 'selected_covariance.json').read_text())
    selected_k = json.loads((BASE / 'selected_k.json').read_text())

    main_m = [r for r in m_rows if r['role'] == 'main']
    m_summary = []
    for M in ('100', '1000', '10000'):
        rs = [r for r in main_m if r['M'] == M]
        m_summary.append({
            'M': int(M), 'n_rows': len(rs),
            'mean_val_ll': avg(rs, 'val_ll'),
            'mean_rare': avg(rs, 'rare_proxy'),
            'mean_class2_mmd': avg(rs, 'class2_mmd'),
            'mean_class2_swd': avg(rs, 'class2_swd'),
            'mean_mmd': avg(rs, 'mmd_rbf'),
            'mean_swd': avg(rs, 'sliced_wasserstein'),
        })
    m_by = {str(r['M']): r for r in m_summary}
    m_gain = (m_by['1000']['mean_val_ll'] - m_by['100']['mean_val_ll']
              if m_by['1000']['mean_val_ll'] is not None and m_by['100']['mean_val_ll'] is not None else None)
    m_gain2 = (m_by['10000']['mean_val_ll'] - m_by['1000']['mean_val_ll']
               if m_by['10000']['mean_val_ll'] is not None and m_by['1000']['mean_val_ll'] is not None else None)

    cov_by = {(r['config'], str(r['lambda']), str(r['kappa'])): r for r in cov_rows}
    diag_cov = next((r for r in cov_rows if r['config'] == 'diag'), None)
    stable_covs = [r for r in cov_rows
                   if str(r['all_stable']).lower() == 'true' and r['mean_val_ll'] is not None]
    best_cov = max(stable_covs, key=lambda r: (num(r['mean_val_ll']) or -1e9,
                                               -(num(r['max_cond']) or float('inf')))) if stable_covs else None
    full_cov = next((r for r in cov_rows if r['config'] == 'full'), None)
    cov_gain = (num(best_cov['mean_val_ll']) - num(diag_cov['mean_val_ll'])
                if best_cov and diag_cov and num(best_cov['mean_val_ll']) is not None
                and num(diag_cov['mean_val_ll']) is not None else None)

    k_by = {(r['K'], r['covariance']): r for r in k_rows}
    k3 = k_by.get(('3', 'diag'))
    k6 = k_by.get(('6', 'diag'))
    k10 = k_by.get(('10', 'diag'))
    k3_full = k_by.get(('3', 'full'))
    k6_full = k_by.get(('6', 'full'))
    k10_full = k_by.get(('10', 'full'))
    pseudo = [r for r in audit_rows if 'PSEUDO_RARE_RISK' in r['flags']]
    low = [r for r in audit_rows if 'LOW_SUPPORT' in r['flags']]
    unsafe = [r for r in moment_rows if str(r['unsafe']).lower() == 'true']

    def agg_method(method):
        rs = [r for r in agg_rows if r['method'] == method]
        if not rs:
            return None
        return {
            'mean_val_ll': avg(rs, 'mean_val_ll'),
            'mean_rare': avg(rs, 'mean_rare_proxy'),
            'mean_class2_mmd': avg(rs, 'mean_class2_mmd'),
            'mean_class2_swd': avg(rs, 'mean_class2_swd'),
            'mean_mmd': avg(rs, 'mean_mmd'),
            'mean_swd': avg(rs, 'mean_swd'),
            'gap_val_ll_K3': avg(rs, 'gap_val_ll_full_K3'),
            'gap_rare_K3': avg(rs, 'gap_rare_full_K3'),
            'gap_val_ll_K6': avg(rs, 'gap_val_ll_full_K6'),
            'gap_rare_K6': avg(rs, 'gap_rare_full_K6'),
        }
    methods = {m: agg_method(m) for m in sorted(set(r['method'] for r in agg_rows))}
    d0 = methods.get('D0_original')
    d1d = methods.get('D1_union_em_diag')
    d1f = methods.get('D1_union_em_full')
    d2 = methods.get('D2_moment_full')
    d2s = methods.get('D2_moment_shrunk')
    d2safe = methods.get('D2_safe_full')
    d3 = methods.get('D3_moment_init_em_full')
    d4_3 = methods.get('D4_oracle_full_K3')
    d4_6 = methods.get('D4_oracle_full_K6')

    d2_gain = (d2['mean_rare'] - d1f['mean_rare']
               if d2 and d1f and d2['mean_rare'] is not None and d1f['mean_rare'] is not None else None)
    d3_gain = (d3['mean_rare'] - d1f['mean_rare']
               if d3 and d1f and d3['mean_rare'] is not None and d1f['mean_rare'] is not None else None)
    d2_oracle_gap = d2['gap_rare_K3'] if d2 else None
    d3_oracle_gap = d3['gap_rare_K3'] if d3 else None

    # Provisional pre-registered decision rules.
    decision = 'E'
    decision_text = '现有结果不足以选择创新方向（INCONCLUSIVE）。'
    if (m_gain is not None and m_gain >= 0.10 and
            (cov_gain is None or cov_gain < 0.05) and
            (d2_gain is None or d2_gain < 0.005)):
        decision = 'A'
        decision_text = '简单改善 GMM 初始化已经足够：M 的收益明显，而协方差/统计量聚合没有显著独立增益。'
    elif (cov_gain is not None and cov_gain >= 0.05 and best_cov is not None and best_cov['all_stable']):
        decision = 'B'
        decision_text = '优先开发 Full-Covariance / Shrinkage 服务器 GMM：在固定 M、K、来源后仍有稳定增益。'
    elif (d2_gain is not None and d2_gain >= 0.005 and
          d2_oracle_gap is not None and abs(d2_oracle_gap) <= 0.01):
        decision = 'C'
        decision_text = '优先开发基于局部统计量的概率聚合：矩匹配/统计量初始化接近集中式 full GMM，并优于普通 Union Sampling EM。'
    elif (unsafe or pseudo):
        decision = 'D'
        decision_text = '需要开发安全分量合并 + 稀有模式保护：矩匹配出现不安全合并，或分量审计发现可信的稀有富集低支持分量。'
    else:
        decision = 'E'
        decision_text = '现有结果不足以选择创新方向（INCONCLUSIVE）。'

    lines = [
        '# FedVaDE Round 11B-pre: Federated GMM factor diagnosis',
        '',
        '**No VaDE/FedVaDE neural training was performed.** All results are frozen-model '
        'GMM/statistics diagnostics. The CTG test split was not used for selection.',
        '',
        '## Protocol',
        '',
        f'- Selected M: **{selected_m["selected_M"]}**',
        f'- Selected covariance: **{selected_cov.get("selected_config")}** '
        f'(base={selected_cov.get("base_covariance")}, lambda={selected_cov.get("lambda")}, kappa={selected_cov.get("kappa")})',
        f'- Selected K by BIC: **{selected_k.get("K")} {selected_k.get("covariance")}**',
        f'- Evaluator: fixed Round 11A calibrated_isotonic, tau=0.50, radius fixed.',
        '',
        '## Stage A: M sweep (K=3, diag, union samples)',
        '',
        '| M | mean val LL | rare proxy | class-2 MMD | class-2 SWD | overall MMD | overall SWD |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for r in m_summary:
        lines.append(f"| {r['M']} | {fmt(r['mean_val_ll'])} | {fmt(r['mean_rare'])} | "
                     f"{fmt(r['mean_class2_mmd'])} | {fmt(r['mean_class2_swd'])} | "
                     f"{fmt(r['mean_mmd'])} | {fmt(r['mean_swd'])} |")
    lines += [
        '',
        f'- Validation-LL gain M=1000 vs M=100: **{fmt(m_gain)} nats/sample**.',
        f'- Validation-LL gain M=10000 vs M=1000: **{fmt(m_gain2)} nats/sample**.',
        f'- Selection rule: {selected_m["rule"]}; selected M={selected_m["selected_M"]}.',
        '- BIC is on union fit samples and is not comparable across different M.',
        '',
        '## Stage B: covariance ablation (fixed M, K=3, union samples)',
        '',
        '| Config | lambda | kappa | mean val LL | max cond | stable | rare proxy | class-2 MMD | class-2 SWD |',
        '| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |',
    ]
    for r in cov_rows:
        lines.append(f"| {r['config']} | {r['lambda'] or '—'} | {r['kappa'] or '—'} | "
                     f"{fmt(r['mean_val_ll'])} | {fmt(r['max_cond'], 1)} | "
                     f"{r['all_stable']} | {fmt(r['mean_rare_proxy'])} | "
                     f"{fmt(r['mean_class2_mmd'])} | {fmt(r['mean_class2_swd'])} |")
    lines += [
        '',
        f'- Best stable covariance: **{selected_cov.get("selected_config")}**; val-LL gain over diag: **{fmt(cov_gain)} nats/sample**.',
        '- Shrinkage is post-fit, not a newly trained MAP-EM model; BIC for shrinkage is approximate.',
        '',
        '## Stage C: K capacity and component audit',
        '',
        '| K | covariance | mean val LL | mean BIC | stable | rare proxy | class-2 MMD | class-2 SWD | PSEUDO_RARE components | LOW_SUPPORT components |',
        '| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for r in k_rows:
        lines.append(f"| {r['K']} | {r['covariance']} | {fmt(r['mean_val_ll'])} | "
                     f"{fmt(r['mean_bic'], 1)} | {r['all_stable']} | {fmt(r['mean_rare_proxy'])} | "
                     f"{fmt(r['mean_class2_mmd'])} | {fmt(r['mean_class2_swd'])} | "
                     f"{r['pseudo_rare_components']} | {r['low_support_components']} |")
    lines += [
        '',
        f'- BIC-selected K: **{selected_k.get("K")} {selected_k.get("covariance")}** (rule: {selected_k.get("rule")}).',
        f'- PSEUDO_RARE_RISK components: **{len(pseudo)}**; LOW_SUPPORT components: **{len(low)}**.',
        '- Flags are diagnostic only; they were never used to filter or reweight components.',
        '',
        '## Stage D: federated statistics aggregation and oracle gap',
        '',
        '| Method | mean val LL | rare proxy | class-2 MMD | class-2 SWD | gap val-LL vs D4 full K3 | gap rare vs D4 full K3 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for name in ('D0_original', 'D1_union_em_diag', 'D1_union_em_full',
                 'D2_moment_full', 'D2_moment_shrunk', 'D2_safe_full',
                 'D3_moment_init_em_full', 'D4_oracle_full_K3', 'D4_oracle_full_K6'):
        r = methods.get(name)
        if r:
            lines.append(f"| {name} | {fmt(r['mean_val_ll'])} | {fmt(r['mean_rare'])} | "
                         f"{fmt(r['mean_class2_mmd'])} | {fmt(r['mean_class2_swd'])} | "
                         f"{fmt(r['gap_val_ll_K3'])} | {fmt(r['gap_rare_K3'])} |")
    lines += [
        '',
        f'- Moment matching vs union EM full: rare-proxy change **{fmt(d2_gain)}**; '
        f'moment-init EM vs union EM full: **{fmt(d3_gain)}**.',
        f'- D2 oracle rare gap: **{fmt(d2_oracle_gap)}**; D3 oracle rare gap: **{fmt(d3_oracle_gap)}**.',
        f'- Unsafe K=3 moment merges: **{len(unsafe)} group rows** across models.',
        '- D4 is a centralized real-latent oracle, not a federated method.',
        '',
        '## Answers',
        '',
        '1. **Is M=100 too small?** See the Stage A val-LL/rare table. '
        f'M=1000 vs 100 val-LL gain is {fmt(m_gain)} nats/sample; M=10000 vs 1000 is {fmt(m_gain2)}. '
        'The report selects M based on the locked validation rule, not rare proxy.',
        '',
        '2. **Does Full Covariance still help under controlled M/K/source?** See Stage B. '
        f'Best stable covariance is {selected_cov.get("selected_config")} with val-LL gain {fmt(cov_gain)} over diag. '
        'If the gain is small or unstable, the answer is NO/INCONCLUSIVE.',
        '',
        '3. **Is shrinkage more stable?** See Stage B max condition numbers and stability column. '
        'Shrinkage is useful if it improves val LL or reduces condition numbers without losing rare/class-2 quality.',
        '',
        '4. **Is K=6 better than K=3?** See Stage C BIC/val LL and rare/class-2 metrics. '
        f'BIC selected K={selected_k.get("K")} {selected_k.get("covariance")}. '
        'Do not select K by rare proxy.',
        '',
        '5. **Does K=10 create untrustworthy small components?** '
        f'PSEUDO_RARE_RISK components={len(pseudo)}, LOW_SUPPORT components={len(low)}. '
        'See component_quality_audit.csv for occupancy, n_eff, class-2 MMD/SWD and flags.',
        '',
        '6. **Can client statistics approach centralized full GMM?** Compare D2/D3 with D4 in Stage D. '
        f'D2 rare gap to D4 full K3 is {fmt(d2_oracle_gap)}; D3 is {fmt(d3_oracle_gap)}. '
        'A small negative gap (federated below oracle) is expected; a large gap means the statistical aggregation is insufficient.',
        '',
        '7. **Is moment matching better than ordinary Union Sampling + EM?** '
        f'D2 rare-proxy change vs D1 full is {fmt(d2_gain)}; D3 is {fmt(d3_gain)}. '
        'Positive is better; near-zero means no advantage.',
        '',
        '8. **Which components should not be merged?** See moment_matching_diagnostics.csv and '
        'moment_matching_pairwise.csv. Unsafe groups are those with maximum normalized '
        'Bures-Wasserstein dispersion above the locked threshold; their members are listed.',
        '',
        '9. **Is label-free rare-mode protection still needed?** '
        'Yes if PSEUDO_RARE_RISK or credible rare-rich LOW_SUPPORT components appear, or if '
        'D2 K=3 has unsafe merges. The flags are diagnostic; they were not used to merge or filter.',
        '',
        '10. **Next priority: simple server Full-Covariance GMM or Macro/Micro?** '
        'See the provisional decision below. Do not start ACD or Multi-CA before a controlled '
        'same-capacity baseline.',
        '',
        '## Provisional decision (pre-registered rules)',
        '',
        f'**{decision}. {decision_text}**',
        '',
        'This is a rule-based reading of the controlled validation diagnostics, not a claim of a '
        'new independent test result. If the evidence is contradictory, the report says E/INCONCLUSIVE.',
        '',
        '## Evidence limits',
        '',
        '- No neural training; frozen Round 10 checkpoints.',
        '- Union samples come from local GMMs; they are not raw patient data.',
        '- Full covariance is post-fit in Stage B; it is not yet integrated into the VaDE ELBO.',
        '- Shrinkage BIC is approximate; covariance selection uses validation LL and stability.',
        '- Moment matching is a first-two-moment Gaussian approximation and can collapse multimodal groups; '
        'the diagnostics mark unsafe merges.',
        '- Rare proxy is a fixed classifier+distance proxy, not a true generated-label metric.',
        '- Test split was not used.',
        '',
    ]
    (BASE / 'ROUND11B_PRE_REPORT.md').write_text('\n'.join(lines))
    summary = {
        'selected_M': selected_m, 'selected_covariance': selected_cov,
        'selected_K': selected_k, 'm_summary': m_summary,
        'm_gain_1000_vs_100': m_gain, 'm_gain_10000_vs_1000': m_gain2,
        'cov_gain_best_vs_diag': cov_gain,
        'd2_gain_vs_d1_full': d2_gain, 'd3_gain_vs_d1_full': d3_gain,
        'd2_oracle_rare_gap': d2_oracle_gap, 'd3_oracle_rare_gap': d3_oracle_gap,
        'unsafe_merge_groups': len(unsafe), 'pseudo_rare_components': len(pseudo),
        'low_support_components': len(low), 'decision': decision,
        'decision_text': decision_text,
    }
    (BASE / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    update_progress(stage='REPORT_GENERATION', status='RUNNING',
                    current_config='report generated', completed=255, total=255,
                    best_val_ll=None, rare_mode_proxy=None)
    print('REPORT_OK', decision, flush=True)


if __name__ == '__main__':
    main()
