# C13 与客户端 DPGMM：固定 checkpoint 的 10 推理 seed 结果

记录日期：2026-11-27。

本文中的“±”是固定 round300 checkpoint 后，10 个 cluster-assignment inference seed 的 ARI 标准差，不是 10 次独立训练的标准差。A 是原 C13 客户端普通 GMM 基线，B 是客户端 DPGMM。`unique_full` 对原始数据的每条记录评价一次；`legacy_clients` 按作者有重叠的客户端记录评价。

## Breast Cancer：固定训练 seed 1

选择 seed 1 是为了对齐历史 C13-s1 的 `legacy_clients` ARI 0.6381，不是根据本次 DPGMM 结果选 seed。

| 评估数据 | 原基线 A ARI | DPGMM B ARI | B − A |
| --- | ---: | ---: | ---: |
| unique_full | 0.6263 ± 0.0125 | 0.6111 ± 0.0153 | −0.0152 |
| legacy_clients | 0.6381 ± 0.0073 | 0.6230 ± 0.0095 | −0.0151 |

在这个固定训练模型上，DPGMM 的 ARI 在两种评价总体上均下降约 0.015。

## Iris 与 Banknote：基线 A 的最佳训练 seed

在现有训练 seed 0–4 中，只根据原基线 A 的 ARI 选择每个数据集的最佳固定模型，不使用 B 的结果选 seed。Iris 和 Banknote 在两种评价总体上选中的都是 seed 4。

| 数据集 | 评估数据 | 最佳训练 seed | 原基线 A ARI | DPGMM B ARI | B − A |
| --- | --- | ---: | ---: | ---: | ---: |
| Iris | unique_full | 4 | 0.8153 ± 0.0172 | 0.6538 ± 0.0220 | −0.1616 |
| Iris | legacy_clients | 4 | 0.8490 ± 0.0162 | 0.6904 ± 0.0072 | −0.1585 |
| Banknote | unique_full | 4 | 0.5313 ± 0.0130 | −0.0076 ± 0.0016 | −0.5389 |
| Banknote | legacy_clients | 4 | 0.6483 ± 0.0120 | −0.0663 ± 0.0012 | −0.7146 |

这张表是事后选取基线 A 最佳 seed 的案例展示，不是跨训练 seed 的总体效果估计。在所选固定模型上，DPGMM 在 Iris 的 ARI 下降约 0.16，在 Banknote 上接近零或为负。

## 结果来源

- seed 0–2：服务器 `formal_cv_selected_20261126/`
- seed 3–4：服务器 `formal_ab_seeds34_20261127/`
- 评估轮次：round 300
- 每个固定 checkpoint 的推理 seed 数：10
