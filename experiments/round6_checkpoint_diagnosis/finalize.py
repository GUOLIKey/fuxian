import json
from pathlib import Path
import numpy as np
ROOT=Path('/root/autodl-tmp/fedvade_project/reproduction_runs/round6_checkpoint_diagnosis')
d=json.loads((ROOT/'round6_findings.json').read_text())
cs=json.loads((ROOT/'client_server_diagnostics.json').read_text())
s=d['summary']
a=s['author_weight']['A_author']
r=s['r5_author_300']['A_author']
b=s['r5_post_server_300']['A_author']
au=s['author_weight']['B_unique_diagnostic']
ru=s['r5_author_300']['B_unique_diagnostic']
seed_means=np.array([s[f'rround2_B_author_seed0' if i==0 else f'rround3_B_author_seed{i}']['A_author']['ari_mean'] for i in range(5)])
d['client_server']=cs
d['training_seed_summary_5_author']={'means':seed_means.tolist(),'mean':float(seed_means.mean()),'population_std':float(seed_means.std()),'min':float(seed_means.min()),'max':float(seed_means.max())}
d['gap_author_vs_A300_A_protocol']=a['ari_mean']-r['ari_mean']
d['gap_author_vs_A300_B_protocol']=au['ari_mean']-ru['ari_mean']
d['evidence_levels']=[
 {'finding':'Author checkpoint reaches near-paper ARI using same current 31-column CSV, scaler, model.predict and validation; training provenance remains missing.',
  'level':'CONFIRMED','evidence':'unified_evaluation.csv; checkpoint_manifest.csv; evaluation_protocol.md'},
 {'finding':'The author-vs-retrained gap remains nearly unchanged when the evaluation set changes from 2129 duplicated client records to 569 original CSV rows.',
  'level':'CONFIRMED','evidence':'unified_evaluation.csv; round6_findings.json'},
 {'finding':'Fifth-round post-server broadcast lowers ARI, degrades latent class separation and reconstruction, while giving exactly matching client/server VAE states.',
  'level':'CONFIRMED','evidence':'round5_broadcast/formal_*/rounds.jsonl; latent_diagnostics.csv; client_server_diagnostics.json'},
 {'finding':'Training seed variation is larger than stochastic inference variation across five author-schedule retrain checkpoints.',
  'level':'CONFIRMED','evidence':'unified_evaluation.csv; round2/round3 result.json and checkpoint_manifest.csv'},
 {'finding':'Author checkpoint has a more favorable data-to-mixture alignment than A300; mixture separation alone and reconstruction error do not explain the gap.',
  'level':'SUPPORTED','evidence':'gmm_diagnostics.csv; latent_diagnostics.csv; unified_evaluation.csv confusion matrices'},
 {'finding':'Author checkpoint may come from a different training endpoint, partition, preprocessing or checkpoint selection protocol.',
  'level':'UNRESOLVED','evidence':'author state_dict lacks training seed, round, partition/scaler and selection metadata'},
 {'finding':'GMM initialization, data partition and optimizer dynamics may contribute causally to the gap.',
  'level':'HYPOTHESIS','evidence':'source audit and score variation; no controlled causal test of these factors'},
 {'finding':'A1 ID input, all-data scaler, label-assisted client partition and local GMM component count, and overlapping client records are present in author code.',
  'level':'CONFIRMED','evidence':'dataloader.py:12-45; model.py:176-183; evaluation_protocol.md'},
]
d['decision']='B: keep the reproducible fixed-seed/multi-seed baseline and move to model innovation; no single causal training-protocol discrepancy has been isolated.'
(ROOT/'round6_findings.json').write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
names=['author_weight','r5_author_150','r5_author_300','r5_post_server_150','r5_post_server_300']
lines=[
 '# FedVaDE 第六轮：同协议权重复评及根因诊断','',
 '本轮只读取已有权重、数据和日志并进行 CPU 推理；没有训练模型，也没有改动作者源码。所有均值±标准差来自固定 checkpoint 的 10 次随机潜在变量推理（总体标准差），不是 10 次训练。','',
 '## 协议与可比性','',
 'A：作者 CSV 的 A1–A31（包括 A1 ID），全 569 行拟合 StandardScaler；用源码 seed=42 划分并拼接十个客户端的 2129 条记录（包含重复），batch=200、shuffle=True；调用原版 VaDE.predict 的随机潜在采样和 GMM posterior argmax，再调用原版 validation（labelMapping、Purity、ARI）。每份 checkpoint 用相同的推理 seed 0–9，在 CPU float32 计算。标签在本轮评价中只进入指标；作者训练源码使用标签构造客户端分区并确定局部 GMM 分量数。作者权重原训练预处理未知。','',
 'B：同一 CSV、列序、全数据 scaler、模型预测和指标，改为原始 CSV 的 569 行各评一次、shuffle=False。B 只诊断重复计权；不是严格无泄漏的独立测试。不能跨 A/B 协议混排成绩。','',
 '## 统一评价（A/B 分开）','',
 '| 权重 | A ARI | A Purity | B ARI | B Purity |',
 '|---|---:|---:|---:|---:|',
]
for name in names:
 x=s[name]['A_author'];y=s[name]['B_unique_diagnostic']
 lines.append(f"| {name} | {x['ari_mean']:.4f}±{x['ari_std']:.4f} | {x['purity_mean']:.4f}±{x['purity_std']:.4f} | {y['ari_mean']:.4f}±{y['ari_std']:.4f} | {y['purity_mean']:.4f}±{y['purity_std']:.4f} |")
lines += [
 '',
 f"作者权重与 A-300 的 ARI 差距：A 协议 {a['ari_mean']-r['ari_mean']:.4f}；B 协议 {au['ari_mean']-ru['ari_mean']:.4f}。仅去掉评价重复计权几乎不改变差距。第五轮训练日志的单次值 A-300=0.4778、B-300=-0.0260；固定权重重新抽样产生轻微变化。作者历史约 0.6315 落在本轮 A 协议 10 次推理范围内。",
 '',
 '## Checkpoint 与模型角色','',
 '作者文件是裸 state_dict，可严格加载到 31 输入、10 维潜变量、2 个服务器 GMM 分量的同一 VaDE（共 2,597,093 参数）；其训练种子、轮次、分区、scaler、模型选择和训练时客户端/优化器状态均 UNKNOWN。第五轮 150/300 是服务器 Adam 更新后模型；完整 checkpoint 另含客户端、优化器、scheduler 和 RNG。第三轮最终文件是服务器模型，仅有 round 元数据，seed 来自关联 result.json。逐文件 SHA256、参数键、形状、dtype、源码及数据路径见 checkpoint_manifest.csv。','',
 f"第三轮作者调度五个训练种子的同协议 ARI 均值（各 checkpoint 再均值 10 次推理）为 {seed_means.mean():.4f}±{seed_means.std():.4f}，范围 {seed_means.min():.4f}–{seed_means.max():.4f}；推理波动不能替代训练种子波动。第三轮 seed0 的服务器最终张量与第五轮 A-300 完全相同（最大绝对差 0）。",
 '',
 '## 内部状态','',
 f"作者/A-300/B-300 的服务器混合权重分别为 {d['gmm']['author_weight']['mixture_weights']}、{d['gmm']['r5_author_300']['mixture_weights']}、{d['gmm']['r5_post_server_300']['mixture_weights']}。分量标签可互换，不能把权重顺序不同当成原因；三者 GMM 高斯间 Bhattacharyya overlap 均约 0.10，未见权重或方差的完全坍缩。",
 f"独立 CSV 的潜在类中心距离/类内 RMS：作者 {d['latent']['author_weight']['class_separation_ratio']:.3f}，A-300 {d['latent']['r5_author_300']['class_separation_ratio']:.3f}，B-300 {d['latent']['r5_post_server_300']['class_separation_ratio']:.3f}。重构 MSE 分别为 {d['latent']['author_weight']['reconstruction_mse_unique']:.4f}、{d['latent']['r5_author_300']['reconstruction_mse_unique']:.4f}、{d['latent']['r5_post_server_300']['reconstruction_mse_unique']:.4f}。A 的重构甚至好于作者，说明单看重构不能解释 ARI。",
 f"第五轮 A-300 服务器与客户端 VAE 最大参数差 {cs['author_300']['server_client_vae_max_abs_max']:.6g}，潜在均值 RMS 差 {cs['author_300']['server_client_latent_mu_rms_mean']:.4f}；B-300 两者均为零，但聚类成绩更差。B 的 posterior 虽更自信，其潜在类分离和重构均退化，不能将其负 ARI 归结为客户端/服务器 VAE 未对齐。",
 '',
 '## 根因分级','',
]
for item in d['evidence_levels']:
 lines.append(f"- **{item['level']}** {item['finding']} 证据：{item['evidence']}。")
lines += [
 '',
 '仍无法确认作者权重与我们复训的初始权重、训练分区、scaler、迭代终点或模型选择是否相同。作者权重接近论文值只证明该文件在当前评价路径下表现接近，不能反推出它的生成协议；权重差异本身也不是因果证明。',
 '',
 '## 决策','',
 '选择 B：暂不进行无依据的参数追分或追加多种子修正实验。保留作者源码逻辑的固定协议、多种子结果和明确的失败案例作为研究基线，进入生成式联邦聚类创新研究。未来所有方法使用相同数据、分区、训练预算、评价协议和多个训练种子，分别报告聚类与生成质量。若能取得作者权重的训练日志和 checkpoint 选择规则，再针对具体证据设计最小验证。',''
]
(ROOT/'round6_report.md').write_text('\n'.join(lines))
print("FINALIZED",ROOT/'round6_report.md')
