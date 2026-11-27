# K-matched GMM 单独消融报告

记录日期：2026-11-27

本文只记录 `K_matched_gmm` 消融，不把它列入准备向学长展示的 A/B/S 主报告。该消融在原实验中曾称为 C；它与服务器端的 `S_server_dpgmm` 无关。

## 1. 消融问题

客户端 DPGMM 会同时改变两件事：保留的局部分量数 K，以及具体的拟合算法。为了检查“只改变 K 是否足够”，本消融读取 `B_client_dpgmm` 在同一客户端保留的实际 K，再用普通 GMM 对同一潜在表示重新拟合。

| 组 | 客户端方法 | 服务器 |
| --- | --- | --- |
| `A_baseline` | 按原 C13 规则固定 K 的普通 GMM | 原流程 |
| `B_client_dpgmm` | DPGMM 自适应拟合 | 原流程 |
| `K_matched_gmm` | 使用 B 的实际 K，重新拟合普通 GMM | 原流程 |

A、B、K 共享相同数据、训练 seed、预训练起点和联邦训练协议。K 只完成了训练 seed0–2，没有 seed3–4 结果。

## 2. seed0–2 的三训练 seed 均值

每个训练 seed 先对固定 round300 checkpoint 的10个推理 seed 求平均，再对训练 seed0–2 取算术平均。

| 数据集 | 评价总体 | A_baseline | B_client_dpgmm | K_matched_gmm | K − A | K − B |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Iris | unique_full | 0.491326 | 0.463676 | 0.396034 | −0.095291 | −0.067642 |
| Iris | legacy_clients | 0.513558 | 0.492537 | 0.410228 | −0.103329 | −0.082308 |
| Breast Cancer | unique_full | 0.492831 | 0.466702 | 0.223795 | −0.269035 | −0.242907 |
| Breast Cancer | legacy_clients | 0.512261 | 0.480248 | 0.231504 | −0.280757 | −0.248744 |
| Banknote | unique_full | 0.139134 | 0.006557 | 0.004612 | −0.134523 | −0.001945 |
| Banknote | legacy_clients | 0.181059 | 0.011520 | 0.013154 | −0.167904 | +0.001634 |

K 在三个数据集、两种评价总体上全部低于 A。Iris 和 Breast Cancer 的 K 也明显低于 B；Banknote 的 B、K 都接近零，二者差异很小。

## 3. seed1 固定 checkpoint 对照

每个单元格为“ARI 均值 ± 同一 checkpoint 的10个推理 seed 标准差”。

| 数据集 | 评价总体 | A_baseline | B_client_dpgmm | K_matched_gmm | K − A | K − B |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Iris | unique_full | 0.532524 ± 0.007004 | 0.496343 ± 0.004829 | 0.482685 ± 0.015888 | −0.049839 | −0.013658 |
| Iris | legacy_clients | 0.551635 ± 0.003457 | 0.528108 ± 0.000740 | 0.507485 ± 0.007929 | −0.044149 | −0.020623 |
| Breast Cancer | unique_full | 0.626278 ± 0.012545 | 0.611066 ± 0.015320 | 0.593883 ± 0.012376 | −0.032395 | −0.017184 |
| Breast Cancer | legacy_clients | 0.638135 ± 0.007298 | 0.622993 ± 0.009492 | 0.602114 ± 0.008502 | −0.036022 | −0.020880 |
| Banknote | unique_full | 0.407751 ± 0.013747 | 0.028930 ± 0.004133 | 0.013799 ± 0.000925 | −0.393952 | −0.015131 |
| Banknote | legacy_clients | 0.511751 ± 0.010248 | 0.075124 ± 0.003503 | 0.039252 ± 0.002846 | −0.472499 | −0.035872 |

seed1 的六个评价中，K 也全部低于 A 和 B。

## 4. 结论边界

1. 在当前 C13 参数、作者重叠划分和训练 seed0–2 下，仅把普通 GMM 的局部分量数改成客户端 DPGMM 保留的 K，没有改善最终 ARI。
2. K 在 Iris、Breast Cancer 上比 B 下降更多；Banknote 的 B、K 都接近失败区间。这说明结果不能简化为“选出更合适的 K 就能改善 FedVaDE”。
3. B 与 K 虽共享同一客户端 K 和潜在输入，但一个是 DPGMM、一个是普通 GMM，两者的目标函数、正则化和参数估计不同。因此 K−B 不是单独一个纯贝叶斯先验效应。
4. 本消融只有三个训练 seed，且评价数据来自训练数据来源，不能推广为其他划分或数据上的普遍结论。

最窄结论是：**在本实验设置中，客户端 DPGMM 得到的局部分量数 K 本身不足以改善 C13；按该 K 重新拟合普通 GMM 反而整体下降。**

## 5. 证据来源

- 详细原实验报告：本地 `../c13_dpgmm_3datasets_v1/docs/FORMAL_RESULTS_20261127.md`；GitHub 发布包 `client_local_dpgmm/docs/FORMAL_RESULTS_20261127.md`。
- 原始 seed0–2 评价：本地 `../c13_dpgmm_3datasets_v1/server_formal/runs/`；GitHub 发布包 `client_local_dpgmm/results/formal_seed0_2/runs/`。
