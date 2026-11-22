# FedVaDE 第六轮：同协议权重复评及根因诊断

本轮只读取已有权重、数据和日志并进行 CPU 推理；没有训练模型，也没有改动作者源码。所有均值±标准差来自固定 checkpoint 的 10 次随机潜在变量推理（总体标准差），不是 10 次训练。

## 协议与可比性

A：作者 CSV 的 A1–A31（包括 A1 ID），全 569 行拟合 StandardScaler；用源码 seed=42 划分并拼接十个客户端的 2129 条记录（包含重复），batch=200、shuffle=True；调用原版 VaDE.predict 的随机潜在采样和 GMM posterior argmax，再调用原版 validation（labelMapping、Purity、ARI）。每份 checkpoint 用相同的推理 seed 0–9，在 CPU float32 计算。标签在本轮评价中只进入指标；作者训练源码使用标签构造客户端分区并确定局部 GMM 分量数。作者权重原训练预处理未知。

B：同一 CSV、列序、全数据 scaler、模型预测和指标，改为原始 CSV 的 569 行各评一次、shuffle=False。B 只诊断重复计权；不是严格无泄漏的独立测试。不能跨 A/B 协议混排成绩。

## 统一评价（A/B 分开）

| 权重 | A ARI | A Purity | B ARI | B Purity |
|---|---:|---:|---:|---:|
| author_weight | 0.6295±0.0084 | 0.8973±0.0026 | 0.6159±0.0231 | 0.8930±0.0073 |
| r5_author_150 | 0.4697±0.0034 | 0.8429±0.0012 | 0.4662±0.0043 | 0.8418±0.0016 |
| r5_author_300 | 0.4720±0.0049 | 0.8438±0.0018 | 0.4590±0.0083 | 0.8392±0.0031 |
| r5_post_server_150 | -0.0267±0.0008 | 0.6407±0.0000 | -0.0254±0.0007 | 0.6274±0.0000 |
| r5_post_server_300 | -0.0258±0.0007 | 0.6407±0.0000 | -0.0235±0.0014 | 0.6274±0.0000 |

作者权重与 A-300 的 ARI 差距：A 协议 0.1575；B 协议 0.1570。仅去掉评价重复计权几乎不改变差距。第五轮训练日志的单次值 A-300=0.4778、B-300=-0.0260；固定权重重新抽样产生轻微变化。作者历史约 0.6315 落在本轮 A 协议 10 次推理范围内。

## Checkpoint 与模型角色

作者文件是裸 state_dict，可严格加载到 31 输入、10 维潜变量、2 个服务器 GMM 分量的同一 VaDE（共 2,597,093 参数）；其训练种子、轮次、分区、scaler、模型选择和训练时客户端/优化器状态均 UNKNOWN。第五轮 150/300 是服务器 Adam 更新后模型；完整 checkpoint 另含客户端、优化器、scheduler 和 RNG。第三轮最终文件是服务器模型，仅有 round 元数据，seed 来自关联 result.json。逐文件 SHA256、参数键、形状、dtype、源码及数据路径见 checkpoint_manifest.csv。

第三轮作者调度五个训练种子的同协议 ARI 均值（各 checkpoint 再均值 10 次推理）为 0.3628±0.1501，范围 0.0964–0.4936；推理波动不能替代训练种子波动。第三轮 seed0 的服务器最终张量与第五轮 A-300 完全相同（最大绝对差 0）。

## 内部状态

作者/A-300/B-300 的服务器混合权重分别为 [0.29991888999938965, 0.7000810503959656]、[0.7216567993164062, 0.27834323048591614]、[0.696640133857727, 0.30335989594459534]。分量标签可互换，不能把权重顺序不同当成原因；三者 GMM 高斯间 Bhattacharyya overlap 均约 0.10，未见权重或方差的完全坍缩。
独立 CSV 的潜在类中心距离/类内 RMS：作者 1.196，A-300 1.139，B-300 0.652。重构 MSE 分别为 0.0752、0.0699、0.2325。A 的重构甚至好于作者，说明单看重构不能解释 ARI。
第五轮 A-300 服务器与客户端 VAE 最大参数差 3.23057e-05，潜在均值 RMS 差 0.0166；B-300 两者均为零，但聚类成绩更差。B 的 posterior 虽更自信，其潜在类分离和重构均退化，不能将其负 ARI 归结为客户端/服务器 VAE 未对齐。

## 根因分级

- **CONFIRMED** Author checkpoint reaches near-paper ARI using same current 31-column CSV, scaler, model.predict and validation; training provenance remains missing. 证据：unified_evaluation.csv; checkpoint_manifest.csv; evaluation_protocol.md。
- **CONFIRMED** The author-vs-retrained gap remains nearly unchanged when the evaluation set changes from 2129 duplicated client records to 569 original CSV rows. 证据：unified_evaluation.csv; round6_findings.json。
- **CONFIRMED** Fifth-round post-server broadcast lowers ARI, degrades latent class separation and reconstruction, while giving exactly matching client/server VAE states. 证据：round5_broadcast/formal_*/rounds.jsonl; latent_diagnostics.csv; client_server_diagnostics.json。
- **CONFIRMED** Training seed variation is larger than stochastic inference variation across five author-schedule retrain checkpoints. 证据：unified_evaluation.csv; round2/round3 result.json and checkpoint_manifest.csv。
- **SUPPORTED** Author checkpoint has a more favorable data-to-mixture alignment than A300; mixture separation alone and reconstruction error do not explain the gap. 证据：gmm_diagnostics.csv; latent_diagnostics.csv; unified_evaluation.csv confusion matrices。
- **UNRESOLVED** Author checkpoint may come from a different training endpoint, partition, preprocessing or checkpoint selection protocol. 证据：author state_dict lacks training seed, round, partition/scaler and selection metadata。
- **HYPOTHESIS** GMM initialization, data partition and optimizer dynamics may contribute causally to the gap. 证据：source audit and score variation; no controlled causal test of these factors。
- **CONFIRMED** A1 ID input, all-data scaler, label-assisted client partition and local GMM component count, and overlapping client records are present in author code. 证据：dataloader.py:12-45; model.py:176-183; evaluation_protocol.md。

仍无法确认作者权重与我们复训的初始权重、训练分区、scaler、迭代终点或模型选择是否相同。作者权重接近论文值只证明该文件在当前评价路径下表现接近，不能反推出它的生成协议；权重差异本身也不是因果证明。

## 决策

选择 B：暂不进行无依据的参数追分或追加多种子修正实验。保留作者源码逻辑的固定协议、多种子结果和明确的失败案例作为研究基线，进入生成式联邦聚类创新研究。未来所有方法使用相同数据、分区、训练预算、评价协议和多个训练种子，分别报告聚类与生成质量。若能取得作者权重的训练日志和 checkpoint 选择规则，再针对具体证据设计最小验证。
