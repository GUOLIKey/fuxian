"""Generate ROUND11A_REPORT.md and summary.json from diagnostic CSVs."""
import csv
import json
import math
import time
from pathlib import Path

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
        if value in ('', None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def avg(rows, key):
    vals = [num(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    return float(np.mean(vals)) if vals else None


def fmt(value, nd=4):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return '—'
    return f'{value:.{nd}f}'


def first(rows, **kwargs):
    for r in rows:
        if all(str(r.get(k)) == str(v) for k, v in kwargs.items()):
            return r
    return None


def main():
    eval_cfg = json.loads((BASE / 'evaluator_config.json').read_text())
    tau = float(eval_cfg['tau'])
    radius = float(eval_cfg['radius'])
    ev_rows = read('evaluator_calibration.csv')
    recon_rows = read('reconstruction_diagnostics.csv')
    latent_rows = read('latent_separability.csv')
    union_rows = read('global_vs_union_mog.csv')
    gmm_rows = read('gmm_k_comparison.csv')
    manifest = read('checkpoint_manifest.csv')

    # E1 primary controls.
    def control(split, group, metric='joint_pass'):
        r = first(ev_rows, section='primary_control', split=split, group=group, metric=metric)
        return num(r['value']) if r else None
    eval_pos = control('val', 'positive')
    eval_neg = control('val', 'negative')
    eval_pos_test = control('test', 'positive')
    eval_neg_test = control('test', 'negative')
    r10_pos = num((first(ev_rows, section='round10_reproduction', split='test', group='positive') or {}).get('value'))
    r10_neg = num((first(ev_rows, section='round10_reproduction', split='test', group='negative') or {}).get('value'))
    primary_name = eval_cfg['primary_classifier']
    cls_val = first(ev_rows, section='classifier_metrics', classifier=primary_name,
                    split='val', **{'class': '2'})
    cls_test = first(ev_rows, section='classifier_metrics', classifier=primary_name,
                     split='test', **{'class': '2'})
    unc_test = first(ev_rows, section='classifier_metrics', classifier='uncalibrated',
                     split='test', **{'class': '2'})

    # E2 reconstruction (validation primary).
    def recon(model, cls, method='encoder_mean'):
        r = first(recon_rows, model=model, split='val', **{'class': str(cls), 'method': method})
        return r
    recon_models = sorted(set(r['model'] for r in recon_rows if r['model']))
    recon_table = []
    for m in recon_models:
        rare = recon(m, 2)
        common = recon(m, 0)
        intermediate = recon(m, 1)
        rare_mse = num(rare['reconstruction_mse']) if rare else None
        common_mse = num(common['reconstruction_mse']) if common else None
        ratio = (rare_mse / common_mse) if rare_mse is not None and common_mse not in (None, 0) else None
        recon_table.append((m, rare_mse, common_mse, intermediate and num(intermediate['reconstruction_mse']), ratio,
                            num(rare['rare_proxy']) if rare else None))

    def latent(model):
        r = first(latent_rows, model=model, scope='server', split='val')
        return r
    latent_table = []
    for m in recon_models:
        r = latent(m)
        latent_table.append((m, num(r['silhouette']) if r else None,
                             num(r['fisher_ratio']) if r else None,
                             num(r['knn1_accuracy']) if r else None,
                             num(r['centroid_distance_2']) if r else None))

    # E2 global vs union means.
    def union_mean(model, source, decoder):
        rows = [r for r in union_rows if r['model'] == model and r['source'] == source
                and r['decoder'] == decoder and r['mode'] == 'unconditional']
        return avg(rows, 'rare_proxy'), avg(rows, 'assigned_rare_fraction'), avg(rows, 'mmd_class2'), avg(rows, 'swd_class2')
    union_models = sorted(set(r['model'] for r in union_rows if r['model'] != 'centralized'))
    union_table = []
    for m in union_models:
        rows = [r for r in union_rows if r['model'] == m]
        compat = rows[0]['compatibility_status'] if rows else ''
        cka = num(rows[0]['cka_mean']) if rows and rows[0].get('cka_mean') else None
        unll = num(rows[0]['union_nll']) if rows and rows[0].get('union_nll') else None
        gnll = num(rows[0]['global_nll']) if rows and rows[0].get('global_nll') else None
        union_table.append((m, compat, cka, unll, gnll,
                            union_mean(m, 'global', 'server'),
                            union_mean(m, 'union', 'server'),
                            union_mean(m, 'union', 'client')))
    central_global = union_mean('centralized', 'global', 'server')

    # E3 GMM capacity means (diag, val).
    def gmm_fit(model, K):
        return first(gmm_rows, model=model, covariance='diag', K=str(K), mode='fit')
    def gmm_gen(model, K):
        rows = [r for r in gmm_rows if r['model'] == model and r['covariance'] == 'diag'
                and r['K'] == str(K) and r['mode'] == 'unconditional' and r['eval_split'] == 'val']
        return avg(rows, 'rare_proxy'), avg(rows, 'assigned_rare_fraction'), avg(rows, 'mmd_class2'), avg(rows, 'swd_class2')
    gmm_models = sorted(set(r['model'] for r in gmm_rows if r['mode'] == 'fit' and r['covariance'] == 'diag'))
    gmm_table = []
    for m in gmm_models:
        for K in (3, 6, 10):
            fit = gmm_fit(m, K)
            proxy, assigned, mmd2, swd2 = gmm_gen(m, K)
            gmm_table.append((m, K, num(fit['val_ll']) if fit else None,
                              num(fit['bic']) if fit else None,
                              num(fit['effective_components']) if fit else None,
                              proxy, assigned, mmd2, swd2))

    # Provisional recommendation rules (explicitly exploratory).
    k3 = [row for row in gmm_table if row[1] == 3]
    k6 = [row for row in gmm_table if row[1] == 6]
    k10 = [row for row in gmm_table if row[1] == 10]
    def best_of(rows, idx):
        vals = [r[idx] for r in rows if r[idx] is not None]
        return max(vals) if vals else None
    k3_proxy = best_of(k3, 5)
    k_best_proxy = max([v for v in (best_of(k6, 5), best_of(k10, 5)) if v is not None], default=None)
    global_proxy = max([v[5][0] for v in union_table if v[5][0] is not None], default=None)
    union_server_proxy = max([v[6][0] for v in union_table if v[6][0] is not None], default=None)
    union_client_proxy = max([v[7][0] for v in union_table if v[7][0] is not None], default=None)
    union_gains = [v[6][0] - v[5][0] for v in union_table
                   if v[5][0] is not None and v[6][0] is not None]
    union_gain_min = min(union_gains) if union_gains else None
    flat_gain = (k_best_proxy - k3_proxy) if (k_best_proxy is not None and k3_proxy is not None) else None
    recon_ratios = [row[4] for row in recon_table if row[4] is not None]
    recon_ratio_min = min(recon_ratios) if recon_ratios else None
    latent_knn_min = min([row[3] for row in latent_table if row[3] is not None], default=None)
    eval_usable = (eval_pos is not None and eval_neg is not None
                   and eval_pos >= 0.75 and eval_neg <= 0.05)

    recommendation = '证据不足，暂时不能决定'
    recommendation_reason = '评价器控制或机制对照尚不足以排除评价器假阴性。'
    if not eval_usable:
        recommendation = '证据不足，暂时不能决定'
        recommendation_reason = '评价器在验证正/负对照上仍不够稳定，先修评价器再比较模块。'
    elif union_gains and union_gain_min is not None and union_gain_min >= 0.002:
        recommendation = '优先改善服务器概率聚合 / Multi-CA'
        recommendation_reason = (
            f'相同 server Decoder 下，客户端并集 MoG 在 3 个 severe seed 上均高于全局 GMM'
            f'（增益 {", ".join(f"{g:+.4f}" for g in union_gains)}）；'
            f'但集中式扁平 K=6/10 的稀有代理也提高约 {flat_gain:+.4f}，'
            '因此 Multi-CA 必须与同分量数扁平 GMM 正面对照，不能只靠“分量更多”。'
        )
    elif flat_gain is not None and flat_gain >= 0.01:
        recommendation = '优先改善局部 GMM / ACD 或混合容量'
        recommendation_reason = (
            f'冻结 latent 上 K=6/10 的稀有代理比 K=3 提高约 {flat_gain:+.4f}；'
            '当前所有客户端 local K 都饱和在 3，提示局部/全局混合容量不足。'
            '但这是集中式诊断，ACD 仍需与 BIC 和 Fine-CD 正面对照。'
        )
    elif (recon_ratio_min is not None and recon_ratio_min > 1.5
          and latent_knn_min is not None and latent_knn_min > 0.8):
        recommendation = '优先改善基础 VaDE（Encoder/Decoder）'
        recommendation_reason = '真实稀有重建误差高于常见类，且潜在近邻结构仍可分辨；提示 decoder 生成保真度是瓶颈。'

    summary = {
        'tau': tau,
        'radius': radius,
        'evaluator_positive_val_joint': eval_pos,
        'evaluator_negative_val_joint': eval_neg,
        'evaluator_positive_test_joint': eval_pos_test,
        'round10_reproduction_positive_test_joint': r10_pos,
        'eval_usable': eval_usable,
        'k3_best_rare_proxy': k3_proxy,
        'k6_k10_best_rare_proxy': k_best_proxy,
        'flat_k_gain': flat_gain,
        'global_best_rare_proxy': global_proxy,
        'union_server_best_rare_proxy': union_server_proxy,
        'union_client_best_rare_proxy': union_client_proxy,
        'union_gains': union_gains,
        'union_gain_min': union_gain_min,
        'recon_rare_common_ratio_min': recon_ratio_min,
        'latent_knn1_min': latent_knn_min,
        'recommendation': recommendation,
        'recommendation_reason': recommendation_reason,
    }
    (BASE / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    lines = []
    lines += [
        '# FedVaDE Round 11A: Frozen-model diagnostics',
        '',
        '**No VaDE/FedVaDE training was performed in this round.** All results use frozen Round 10 checkpoints. '
        'The CTG test split has been used in prior exploratory analysis; test rows below are exploratory, not new independent test scores.',
        '',
        '## Status',
        '',
        f'- Primary evaluator: {primary_name}, tau={tau:.2f}, radius={radius:.6f} (train-rare self-NN reference).',
        f'- Validation positive control (real rare): joint pass {fmt(eval_pos)}.',
        f'- Validation negative control (real common): joint pass {fmt(eval_neg)}.',
        f'- Test positive control (exploratory): joint pass {fmt(eval_pos_test)}.',
        f'- Round 10-style test positive control (uncalibrated, tau=0.80, test-rare q95 radius): {fmt(r10_pos)}.',
        '',
        '## E1: evaluator calibration',
        '',
        '| Classifier | Split | Rare precision | Rare recall | Rare support |',
        '| --- | --- | ---: | ---: | ---: |',
    ]
    for row in ([cls_val, cls_test, unc_test]):
        if row:
            lines.append(f"| {row['classifier']} | {row['split']} | {fmt(num(row['precision']))} | {fmt(num(row['recall']))} | {row['support']} |")
    lines += [
        '',
        f'- Chosen tau rule: {eval_cfg["tau_rule"]}.',
        f'- Chosen radius rule: {eval_cfg["radius_rule"]}.',
        f'- Real rare validation pass at the primary rule: {fmt(eval_pos)}; real common false-positive pass: {fmt(eval_neg)}.',
        '- The evaluator remains a classifier + distance proxy, not a true synthetic-label metric. It is conservative; the positive-control pass is the ceiling reference.',
        '',
        '## E2: real reconstruction and latent separability',
        '',
        '| Model | Rare recon MSE | Common recon MSE | Rare/common | Rare recon proxy | Latent silhouette | Latent 1-NN | Rare centroid distance |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for (m, rm, cm, im, ratio, rp), (lm, sil, fisher, knn, cd2) in zip(recon_table, latent_table):
        lines.append(f'| {m} | {fmt(rm)} | {fmt(cm)} | {fmt(ratio)} | {fmt(rp)} | {fmt(sil)} | {fmt(knn)} | {fmt(cd2)} |')
    lines += [
        '',
        '- `rare recon proxy` is the same calibrated evaluator applied to reconstructed real rare samples; it is not a true label.',
        '- A high rare MSE with high latent separability points toward the decoder; low latent separability points toward the encoder/representation.',
        '',
        '## E2/D0: global GMM vs client-union MoG',
        '',
        '| Model | Compatibility | CKA mean | Union NLL | Global NLL | Global proxy | Union/server proxy | Union/client proxy |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for m, compat, cka, unll, gnll, g, us, uc in union_table:
        lines.append(f'| {m} | {compat} | {fmt(cka)} | {fmt(unll)} | {fmt(gnll)} | {fmt(g[0])} | {fmt(us[0])} | {fmt(uc[0])} |')
    lines += [
        '',
        '- `Union/server` decodes client-union MoG samples with the same server decoder (required D0 comparison).',
        '- `Union/client` decodes each sampled component with its matched client decoder (supplementary compatibility check).',
        '- If compatibility is INCOMPATIBLE or LIMITED, the union comparison must not be interpreted as evidence that aggregation loses rare modes.',
        '',
        '## E3: frozen-latent GMM capacity (diagonal, centralized diagnostic)',
        '',
        '| Model | K | Val log-lik | BIC | Effective K | Rare proxy | Assigned rare | Class-2 MMD | Class-2 SWD |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for m, K, ll, bic, eff, proxy, assigned, mmd2, swd2 in gmm_table:
        lines.append(f'| {m} | {K} | {fmt(ll)} | {fmt(bic, 1)} | {fmt(eff, 0)} | {fmt(proxy)} | {fmt(assigned)} | {fmt(mmd2)} | {fmt(swd2)} |')
    lines += [
        '',
        '- GMM fits use frozen training latents only; true labels are never used for fitting or K selection.',
        '- These are centralized mixture-capacity diagnostics, not deployable federated algorithms.',
        '',
        '## Answers',
        '',
        '1. **Is the evaluator trustworthy?** See validation positive/negative controls above. '
        f'The primary rule gives positive {fmt(eval_pos)} and negative {fmt(eval_neg)}. '
        'It is usable as a conservative diagnostic if the positive control is high and the negative control is low; otherwise the evaluator must be revised first.',
        '',
        '2. **Can real rare samples be reconstructed?** See the rare vs common reconstruction MSE and rare recon proxy. '
        'High rare MSE alone is not proof of decoder failure; compare with latent separability and the linear-probe rows in the CSV.',
        '',
        '3. **Global GMM vs client-union MoG?** Compare the three proxy columns. '
        'A union/client gain with a union/server loss suggests cross-space decoding or compatibility problems; '
        'a union/server gain over global supports the aggregation hypothesis.',
        '',
        '4. **K=3 vs K=6 vs K=10?** Compare validation log-likelihood/BIC and rare proxy/class-2 distances. '
        'If K=6/10 improves rare proxy without worsening class-2 MMD/SWD, mixture capacity is a plausible bottleneck. '
        'If only assigned rare fraction rises while class-2 quality does not, the gain is mainly sampling proportion.',
        '',
        '5. **Can we prove the problem is server aggregation?** Only if the union/server comparison is compatible and clearly better than global, '
        'or if a flat same-K GMM beats the original global GMM in a controlled way. The current design cannot prove it if compatibility is LIMITED/INCOMPATIBLE.',
        '',
        '6. **What evidence supports ACD?** All Round 10 local K values saturated at the K=3 cap; the frozen-latent K capacity test is indirect evidence about local mixture capacity, '
        'but this round did not implement ACD. ACD is not yet proven necessary.',
        '',
        '7. **What evidence supports Multi-CA?** The union vs global comparison is the relevant evidence. '
        'Even if flat K=6/10 helps, Multi-CA must still beat a flat GMM with the same number of components and the same parameters.',
        '',
        '8. **Should the base VaDE be optimized first?** If real rare reconstruction is poor and latent separability is weak, or if IID/centralized references also fail, '
        'the base encoder/decoder/ELBO is the first target. Round 10 already showed IID and centralized rare coverage near zero.',
        '',
        '9. **Which conclusions are exploratory?** All test-split comparisons, the classifier proxy, the union compatibility interpretation, '
        'the single-seed IID/medium evidence, and the component-conditioned diagnostics.',
        '',
        '10. **Minimal next training experiment?** Start with the direction selected below, using 3 seeds, the same partitions, the same evaluator, '
        'and one same-capacity control. Do not start with ACD+Multi-CA combined.',
        '',
        '## Provisional recommendation',
        '',
        f'**{recommendation}.** {recommendation_reason}',
        '',
        'The recommendation is a rule-based reading of this exploratory round, not a confirmed causal claim. '
        'The server-generated CSVs remain the source of truth; no test-based model selection was used.',
        '',
        '## Evidence limits',
        '',
        '- No new training; no gradient updates.',
        '- Test split is not independent for new comparisons; all test rows are exploratory.',
        '- Rare coverage is a classifier + distance proxy, not true synthetic class accuracy.',
        '- The client-union MoG may not share the server decoder latent space; compatibility is audited and reported.',
        '- Reference runs have no intermediate checkpoints; formal runs do.',
        '- Single training seed for IID/medium; severe has 3 seeds in Round 10 but this round focuses on severe0/2.',
        '',
    ]
    (BASE / 'ROUND11A_REPORT.md').write_text('\n'.join(lines))
    update_progress(stage='FINAL_ANALYSIS', status='RUNNING', progress_current=12,
                    progress_total=12, best_result=recommendation)
    print('REPORT_OK', recommendation, flush=True)


if __name__ == '__main__':
    main()
