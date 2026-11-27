# C13 DPGMM 两轮实验统一结果

记录日期：2026-11-27
文档状态：**两轮实验的统一结论入口**。分实验报告和原始 JSON/JSONL 保留为明细证据；若表述冲突，以本文件的组名、评价口径和结论边界为准。

## 1. 研究问题

两轮实验依次回答两个不同问题：

1. **客户端问题**：把 C13 客户端的普通 Local GMM 改成 DPGMM，让局部分量数自适应变化，能否提高 FedVaDE 最终聚类 ARI？
2. **服务器问题**：恢复客户端普通 Local GMM，仅在服务器对同一潜变量候选池拟合三档浓度 DPGMM，能否形成有用的多粒度划分并提高 ARI？

本次没有执行“客户端 DPGMM + 服务器 DPGMM”的联合实验，因此不能从现有结果推断二者组合后的协同或抵消效果。

## 2. 统一组名

| 统一名称 | 原实验名称 | 客户端 Local GMM | 服务器 | 作用 |
| --- | --- | --- | --- | --- |
| `A_baseline` | 第一轮 A | 按原 C13 规则固定 K 的普通 GMM | 原 FedVaDE/C13 采样 | 共同基线 |
| `B_client_dpgmm` | 第一轮 B | DPGMM 自适应拟合 | 与 A 相同 | 检查客户端 DPGMM |
| `K_matched_gmm` | 第一轮 C | 使用 B 得到的同一客户端 K，重新拟合普通 GMM | 与 A 相同 | 检查“只改变局部分量数”是否足够 |
| `S_server_dpgmm` | 第二轮 C | 与 A 相同的普通 GMM | 浓度 `0.1/1.0/10.0` 的三个 DPGMM | 检查服务器多浓度划分 |

`K_matched_gmm` 和 `S_server_dpgmm` 在原记录中都曾被简称为 C，但它们不是同一个实验。本文不再单独使用“C 组”称呼它们。

## 3. 统一实验协议

| 项目 | 统一设置 |
| --- | --- |
| 数据集 | Iris、Breast Cancer、Banknote Authentication |
| 客户端数 | 10 |
| 潜变量维度 | 10 |
| Batch size | 200 |
| 学习率 | 0.002 |
| AE 预训练 | 50 轮 |
| 联邦训练 | 300 轮 |
| 服务器潜变量采样数 | 100 |
| VaDE alpha | Iris `0.01`；Breast Cancer、Banknote `0.05` |
| 客户端 DPGMM | prior `0.1`，prune `0.05`；上限 Iris/BC/Banknote 分别为 `6/5/7` |
| 服务器 DPGMM | prior `0.1/1.0/10.0`，上限 `10`，prune `0.01`，最终采样 `34/33/33` |
| 固定 checkpoint | round 300 |
| 每个 checkpoint 的推理 | cluster-assignment inference seed 0–9，共 10 次 |
| 主要指标 | ARI |

评价总体必须分开解释：

- `unique_full`：每条原始记录评价一次；
- `legacy_clients`：按作者存在跨客户端重复的客户端记录评价，Iris 为 150→532、Breast Cancer 为 569→2129、Banknote 为 1372→4318。

两种评价都来自训练数据来源，不是独立测试集。表中的“均值 ± 标准差”是同一训练 checkpoint 的 10 次推理结果，不是 10 次独立训练。

## 4. 第一轮结果：客户端方法

每个训练 seed 都先对 10 次推理求平均，下面再对这些固定 checkpoint 均值取算术平均。seed0–2 是原正式配对实验；seed3–4 是后来补充的 A/B 训练，没有运行 `K_matched_gmm`。因此 A/B 可以汇总五个训练 seed，而三方法对照必须限定在共同的 seed0–2。

### 4.1 A/B 的五训练 seed 汇总

| 数据集 | 评价总体 | A_baseline | B_client_dpgmm | B − A |
| --- | --- | ---: | ---: | ---: |
| Iris | unique_full | 0.568678 | 0.515052 | −0.053626 |
| Iris | legacy_clients | 0.593323 | 0.545052 | −0.048271 |
| Breast Cancer | unique_full | 0.499947 | 0.503315 | +0.003368 |
| Breast Cancer | legacy_clients | 0.518705 | 0.522010 | +0.003305 |
| Banknote | unique_full | 0.200166 | 0.009513 | −0.190652 |
| Banknote | legacy_clients | 0.262146 | 0.000714 | −0.261432 |

五 seed 汇总中，B 在 Iris 和 Banknote 明显低于 A；Breast Cancer 的均值只高约 0.0033。Breast Cancer 的配对差值随训练 seed 改变方向：seed0、3、4 为正，seed1、2 为负，其中 seed2 明显下降。这个结果不支持“稳定改善”，也不能用 +0.0033 宣称有效。

### 4.2 A/B/K 的共同三个训练 seed 汇总

| 数据集 | 评价总体 | A_baseline | B_client_dpgmm | B − A | K_matched_gmm | K − A |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Iris | unique_full | 0.491326 | 0.463676 | −0.027650 | 0.396034 | −0.095291 |
| Iris | legacy_clients | 0.513558 | 0.492537 | −0.021021 | 0.410228 | −0.103329 |
| Breast Cancer | unique_full | 0.492831 | 0.466702 | −0.026128 | 0.223795 | −0.269035 |
| Breast Cancer | legacy_clients | 0.512261 | 0.480248 | −0.032014 | 0.231504 | −0.280757 |
| Banknote | unique_full | 0.139134 | 0.006557 | −0.132577 | 0.004612 | −0.134523 |
| Banknote | legacy_clients | 0.181059 | 0.011520 | −0.169538 | 0.013154 | −0.167904 |

在共同的 seed0–2 上，`B_client_dpgmm` 和 `K_matched_gmm` 在三个数据集、两种评价总体上都低于 A。单纯把普通 GMM 的局部分量数改成 B 保留的 K 没有解决问题。B 与 K 使用相同潜在输入和相同 K，但拟合算法不同；B−K 不能被单独解释为纯粹的“贝叶斯先验贡献”。

## 5. 统一 seed1 对照：四种方法使用同一训练 seed

每个单元格格式为“ARI 均值 ± 推理标准差（相对 A 的差值）”。seed1 是服务器实验覆盖三个数据集的共同训练 seed，因此用于把两轮方法放在同一张表中；它不是三个数据集各自的最佳 seed。

| 数据集 | 评价总体 | A_baseline | B_client_dpgmm | K_matched_gmm | S_server_dpgmm |
| --- | --- | ---: | ---: | ---: | ---: |
| Iris | unique_full | 0.532524 ± 0.007004 | 0.496343 ± 0.004829（−0.036181） | 0.482685 ± 0.015888（−0.049839） | 0.494698 ± 0.018023（−0.037826） |
| Iris | legacy_clients | 0.551635 ± 0.003457 | 0.528108 ± 0.000740（−0.023527） | 0.507485 ± 0.007929（−0.044149） | 0.509104 ± 0.010695（−0.042531） |
| Breast Cancer | unique_full | 0.626278 ± 0.012545 | 0.611066 ± 0.015320（−0.015212） | 0.593883 ± 0.012376（−0.032395） | 0.610224 ± 0.016694（−0.016054） |
| Breast Cancer | legacy_clients | 0.638135 ± 0.007298 | 0.622993 ± 0.009492（−0.015142） | 0.602114 ± 0.008502（−0.036022） | 0.637301 ± 0.005758（−0.000834） |
| Banknote | unique_full | 0.407751 ± 0.013747 | 0.028930 ± 0.004133（−0.378821） | 0.013799 ± 0.000925（−0.393952） | 0.495218 ± 0.019861（+0.087467） |
| Banknote | legacy_clients | 0.511751 ± 0.010248 | 0.075124 ± 0.003503（−0.436627） | 0.039252 ± 0.002846（−0.472499） | 0.519406 ± 0.008791（+0.007655） |

seed1 下，`S_server_dpgmm` 在 Iris 低于 A；Breast Cancer 仅在 `legacy_clients` 上与 A 数值接近，但 `unique_full` 仍低 0.016054；Banknote 在两种评价总体上高于 A。由于这只是一次训练，不能据此把 Breast Cancer 写成等价，也不能把 Banknote 写成稳定提升。

## 6. seed4 敏感性检查

Iris 和 Banknote 的 seed4 是依据既有 `A_baseline` 在训练 seed 0–4 中的表现选出的强基线案例，因此属于事后敏感性检查，不是预先规定的多训练 seed 总体估计。`K_matched_gmm` 没有运行 seed4。

| 数据集 | 评价总体 | A_baseline | B_client_dpgmm | S_server_dpgmm |
| --- | --- | ---: | ---: | ---: |
| Iris | unique_full | 0.815349 ± 0.017206 | 0.653795 ± 0.021963（−0.161554） | 0.783920 ± 0.030585（−0.031429） |
| Iris | legacy_clients | 0.848989 ± 0.016186 | 0.690446 ± 0.007190（−0.158543） | 0.806490 ± 0.019909（−0.042499） |
| Banknote | unique_full | 0.531342 ± 0.012969 | −0.007604 ± 0.001573（−0.538946） | 0.072223 ± 0.004311（−0.459119） |
| Banknote | legacy_clients | 0.648270 ± 0.012003 | −0.066317 ± 0.001194（−0.714587） | 0.167508 ± 0.008263（−0.480762） |

Iris seed4 的 S 仍低于 A。Banknote seed1 的正向结果没有保持：seed4 的 S 比 A 低 0.459119（`unique_full`）和 0.480762（`legacy_clients`）。Banknote seed4 的推理标准差很小，因此低值不是由某一次 cluster-assignment 抽样造成的宽波动；它反映的是固定训练 checkpoint 本身较差。

## 7. 服务器三档浓度的结构结果

`不同 K` 表示一次服务器调用中至少两档浓度保留的活跃分量数不同；`有序` 要求当次调用满足 `K(0.1) ≤ K(1.0) ≤ K(10.0)`。

| 训练 seed | 数据集 | 调用数 | 不同 K 调用 | 有序调用 | 平均 K：0.1 / 1.0 / 10.0 | 最后一轮 K | warning 调用 |
| ---: | --- | ---: | ---: | ---: | --- | --- | ---: |
| 1 | Iris | 301 | 262 | 135 | 4.425 / 4.654 / 4.844 | 4 / 8 / 8 | 0 |
| 1 | Breast Cancer | 301 | 278 | 98 | 5.442 / 5.588 / 5.635 | 2 / 2 / 5 | 0 |
| 1 | Banknote | 301 | 284 | 107 | 4.146 / 4.349 / 4.714 | 6 / 3 / 10 | 0 |
| 4 | Iris | 301 | 254 | 136 | 4.332 / 4.515 / 4.708 | 7 / 7 / 5 | 0 |
| 4 | Banknote | 301 | 270 | 135 | 3.977 / 3.977 / 4.525 | 6 / 3 / 3 | 0 |

浓度上升时平均 K 总体增加，但逐次调用满足有序关系的只有 98/301 至 136/301。最后一轮也出现 `6/3/10`、`7/7/5` 和 `6/3/3`。因此当前方法应称为“三个浓度条件下的 DPGMM 划分”，不能称为稳定的粗—中—细层次，也不是原 Multi-CA 竞争惩罚学习的复现。

## 8. 统一结论

1. **客户端 DPGMM 没有显示稳定收益。** 在共同的 seed0–2 上，B 在三个数据集、两种评价总体上全部低于 A；扩展到 seed0–4 后，Iris 和 Banknote 仍明显下降，Breast Cancer 仅高约 0.0033，而且配对差值随 seed 改变方向。
2. **只改变局部分量数也没有解决问题。** `K_matched_gmm` 的三 seed 平均结果同样全部低于 A，而且在 Iris、Breast Cancer 上下降更大。
3. **服务器多浓度 DPGMM 没有稳定收益。** Iris 在 seed1 和 seed4 均低于 A；Breast Cancer seed1 只在有重复记录的 `legacy_clients` 上接近 A；Banknote seed1 的正向结果在 seed4 大幅反向。
4. **服务器方法对训练 seed 敏感。** Banknote 的 `unique_full` 从 seed1 的 0.495218 变为 seed4 的 0.072223，不能把单个 seed 的改善写成方法结论。
5. **“多浓度”不等于稳定多粒度层次。** 三档 DPGMM 经常产生不同 K，但逐轮 K 顺序不可靠。当前结果没有证明学长所说的 Multi-CA 式多粒度结构已经实现。
6. **现有结果不支持组合协同。** S 在若干案例中明显高于 B，只能说明服务器改法没有出现客户端 B 那么严重的下降；由于没有联合组，不能推断客户端和服务器同时使用 DPGMM 会有效。
7. **负面结果不是明显的数值崩溃。** 服务器方法的五条正式训练均完成 300 轮，`status=ok`、`nan_events=0`，结构日志没有 warning，DPGMM 路径确实被调用。
8. **结论范围有限。** 客户端 A/B 有五个训练 seed，但 `K_matched_gmm` 只有 seed0–2；服务器主比较只有 seed1，seed4 又是基线选出的敏感性案例。10 个推理 seed 不是独立训练。不能宣称统计显著、独立测试泛化或“DPGMM 普遍无效”。

最窄且准确的总结是：**在当前 C13、作者重叠划分和已记录参数下，客户端 DPGMM 没有表现出跨数据集、跨训练 seed 的稳定收益，按其 K 重拟合普通 GMM 在共同的三个 seed 上也没有改善；服务器三浓度 DPGMM 能产生不同划分，但没有形成稳定有序层次，其 ARI 收益对训练 seed 不稳定。**

## 9. 后续决定

当前结果适合按“负面且不稳定的机制实验”汇报。下一步不应直接把客户端和服务器 DPGMM 叠加后再寻找单个高分 seed；应先向学长确认：他要求的是任意多个浓度条件下的非嵌套划分，还是必须存在稳定的粗—中—细关系或竞争惩罚学习机制。这个选择会改变下一轮方法定义和评价标准。

## 10. 证据来源

- 客户端方法详细报告：本地 `../c13_dpgmm_3datasets_v1/docs/FORMAL_RESULTS_20261127.md`；GitHub 发布包 `client_local_dpgmm/docs/FORMAL_RESULTS_20261127.md`。
- 固定 checkpoint 补充表：本地 `../c13_dpgmm_3datasets_v1/docs/FIXED_CHECKPOINT_INFERENCE_RESULTS_20261127.md`；GitHub 发布包 `client_local_dpgmm/docs/FIXED_CHECKPOINT_INFERENCE_RESULTS_20261127.md`。
- 服务器方法详细报告：本地 `../c13_server_dpgmm_multigranularity_v1/docs/C_SERVER_DPGMM_RESULTS_20261127.md`；GitHub 发布包 `server_multiconcentration_dpgmm/docs/C_SERVER_DPGMM_RESULTS_20261127.md`。
- 客户端 seed0–2 原始结果：本地 `../c13_dpgmm_3datasets_v1/server_formal/`；GitHub 发布包 `client_local_dpgmm/results/formal_seed0_2/`。
- 服务器 seed1 与 seed4 原始日志、评价 JSON 和结构 JSONL 保留在服务器，并已发布到 GitHub 实验分支；模型权重和数据未上传。
