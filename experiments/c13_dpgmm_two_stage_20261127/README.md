# C13 DPGMM 两轮实验（2026-11-27）

统一表格、组名和结论以 [两轮实验统一结果](COMBINED_RESULTS_20261127.md) 为准。下列内容保留较短的发布摘要，分实验目录保留详细证据。

本目录记录两轮相互独立的 FedVaDE/C13 改动。第一轮改变客户端 Local GMM；第二轮恢复客户端普通 GMM，只改变服务器潜变量采样。两轮不能合并解释为“客户端和服务器同时使用 DPGMM”的联合实验。

## 两轮实验

| 轮次 | 客户端 | 服务器 | 研究问题 | 主要结论 |
| --- | --- | --- | --- | --- |
| 1. `client_local_dpgmm` | Local GMM 改为 DPGMM | 保持 C13/FedVaDE 流程 | 自适应局部分量数能否改善最终聚类？ | 在 Iris、Breast Cancer、Banknote 的三个预设训练 seed 均值上，B 组没有改善 A 组 ARI。 |
| 2. `server_multiconcentration_dpgmm` | 保持普通 Local GMM | 对同一潜变量候选池拟合浓度 0.1/1.0/10.0 的三个 DPGMM | 不使用 complete-linkage，能否通过不同浓度产生有用的多粒度划分？ | 能产生不同 K，但逐轮 K 不稳定有序；Iris 无改善，Breast Cancer 仅在重叠评价总体上接近基线，Banknote seed1 的改善在 seed4 反向。 |

第一轮内部还包含 `C_matched_k_gmm`：用 B 组得到的客户端 K 拟合普通 GMM，用于检查“只改变分量数”是否足够。第二轮的 `C_server_dpgmm` 是另一个实验组。两个 C 的含义不同，结果表中均使用完整名称。

## 实验协议

- 数据集：Iris、Breast Cancer、Banknote Authentication。
- 10 个客户端，潜变量维度 10，batch size 200，学习率 0.002。
- 50 轮 AE 预训练，300 轮联邦训练，round 300 固定 checkpoint 评价。
- Iris 的 VaDE `alpha=0.01`；Breast Cancer 和 Banknote 的 `alpha=0.05`。
- 每个固定 checkpoint 使用 cluster-assignment inference seed 0–9；表中的“±”是这 10 次推理的标准差，不是独立训练标准差。
- `unique_full` 对每条原始记录评价一次；`legacy_clients` 按作者存在跨客户端重复的客户端记录评价。两者都不是独立测试集。

客户端 DPGMM 的数据集特定参数、选择过程和三训练 seed 结果见 [第一轮正式报告](client_local_dpgmm/docs/FORMAL_RESULTS_20261127.md)。服务器多浓度 DPGMM 的 seed1 主结果、seed4 敏感性表与结构统计见 [第二轮正式报告](server_multiconcentration_dpgmm/docs/C_SERVER_DPGMM_RESULTS_20261127.md)。

## 结果摘要

### 第一轮：客户端 DPGMM，三个训练 seed 的 ARI 均值

| 数据集 | 评价总体 | A_baseline | B_client_dpgmm | B − A |
| --- | --- | ---: | ---: | ---: |
| Iris | unique_full | 0.491326 | 0.463676 | −0.027650 |
| Breast Cancer | unique_full | 0.492831 | 0.466702 | −0.026128 |
| Banknote | unique_full | 0.139134 | 0.006557 | −0.132577 |
| Iris | legacy_clients | 0.513558 | 0.492537 | −0.021021 |
| Breast Cancer | legacy_clients | 0.512261 | 0.480248 | −0.032014 |
| Banknote | legacy_clients | 0.181059 | 0.011520 | −0.169538 |

这三次训练只能给出有限的训练 seed 稳定性证据，不能写成统计显著性结论。

### 第二轮：服务器多浓度 DPGMM

seed1 下，C 相对 A 的 ARI 差值为：

| 数据集 | unique_full：C − A | legacy_clients：C − A |
| --- | ---: | ---: |
| Iris | −0.037826 | −0.042531 |
| Breast Cancer | −0.016054 | −0.000834 |
| Banknote | +0.087467 | +0.007655 |

Iris 和 Banknote 又使用按既有 A 组表现选出的 seed4 做敏感性检查。Iris 仍低于 A；Banknote 的差值变为 −0.459119（`unique_full`）和 −0.480762（`legacy_clients`）。因此 Banknote seed1 不能作为稳定收益。

三档浓度在每条运行的 301 次服务器调用中，有 254–284 次产生至少两个不同的活跃 K；但满足 `K(0.1) ≤ K(1.0) ≤ K(10.0)` 的只有 98–136 次。准确描述是“三个浓度条件下的 DPGMM 划分”，不是稳定的粗—中—细层次，也不是原 Multi-CA 竞争惩罚学习的复现。

## 文件清单

| 内容 | 位置 |
| --- | --- |
| 两轮实验统一表格与最终结论 | `COMBINED_RESULTS_20261127.md` |
| 第一轮方案、参数、实现与测试 | `client_local_dpgmm/docs/`、`configs/`、`scripts/`、`tests/` |
| 第一轮 seed0–2 日志与 JSON/JSONL | `client_local_dpgmm/results/formal_seed0_2/` |
| 第一轮 seed3–4 补充日志与 JSON/JSONL | `client_local_dpgmm/results/formal_seed3_4/` |
| 第一轮结果表与结论 | `client_local_dpgmm/docs/FORMAL_RESULTS_20261127.md`、`FIXED_CHECKPOINT_INFERENCE_RESULTS_20261127.md` |
| 第二轮方案、参数、实现与测试 | `server_multiconcentration_dpgmm/docs/`、`configs/`、`scripts/`、`tests/` |
| 第二轮 seed1 日志与 JSON/JSONL | `server_multiconcentration_dpgmm/results/seed1/` |
| 第二轮 seed4 敏感性日志与 JSON/JSONL | `server_multiconcentration_dpgmm/results/seed4_sensitivity/` |
| 第二轮结果表、结构统计与结论 | `server_multiconcentration_dpgmm/docs/C_SERVER_DPGMM_RESULTS_20261127.md` |

## 发布边界

本分支不包含作者交付源码、原始数据、模型权重、checkpoint、运行时源码副本、缓存、密钥或登录信息。JSON 和日志保留了服务器绝对路径、环境版本、命令参数、源码哈希与 checkpoint 哈希，用于确认实验来源；这些记录不包含 checkpoint 本体。

因此，本目录支持审查方案、参数、执行状态、指标和结论，但不是一个脱离作者源码与数据即可重新训练的自包含包。精确重跑需要另行取得与清单哈希一致的源码和数据。
