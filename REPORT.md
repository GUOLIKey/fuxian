# F01 + Multi-CA 第一轮冻结聚合实验

**状态：完成。** 仅做冻结模型聚合、推断与生成；未训练 VAE、未重建客户端、未修改任何权重、未启动联邦训练。

## 1. 冻结 Checkpoint

| 项 | 值 |
|---|---|
| 文件 | `reproduction_runs/round7_f01_full_repro/fine/F01/epoch_100_full.pt` |
| SHA256 | `3be341c451a186d1f36673080164af208993d215f0c2cb03709407892f7d6f3e` |
| 轮次 | 100 |
| 客户端 | 10 个真实 client state；local K `[1,1,2,1,2,1,2,1,1,2]` |
| 客户端记录 | 2129；分区 sizes `[115,209,326,56,218,155,350,239,138,323]` |
| Server GMM | K=2 |

10 个 client 的 encoder 完全相同（hash `092282c80bddd19a`），decoder 也完全相同（`29e5472708611b11`）；10 个 local GMM 各不相同。这是作者训练循环每轮把 client 平均模型广播给所有 client 的结果。

## 2. 潜在空间兼容性（先检查）

| 检查 | 结果 |
|---|---|
| 客户端 encoder/decoder 之间 | 10/10 完全相同（共享 client latent space） |
| Client vs server encoder | 不同：server encoder `b4b3146a18e62752`，decoder `853b84614a7c3f15` |
| Client vs server latent 均值 L2 | 0.6175（所有 client 相同） |
| Client vs server latent 协方差 Frobenius | 1.4253 |
| Client vs server latent RBF-MMD | 0.0715 |
| Client GMM 均值到 server latent 的 Mahalanobis | mean 最大 21.92；到自身 client latent 最大 22.04 |
| Server GMM 均值到 server latent 的 Mahalanobis | mean 8.38，max 15.70 |

**结论：** 10 个 client local GMM 共享同一个 client latent space，但该空间与 server latent space **不完全相同**。因此：

- **B vs C 是受控的同空间比较**（两者都由同一组 client local GMM 构造）。
- **A 是 server 空间的原始 F01 GMM**；A 与 B/C 的横向比较是**跨空间**的描述性结果，不能把差异全部归因于容量或层次结构。
- 未做任何权重迁移、重编码或客户端重建。

## 3. 方法

- **A**：原始 F01 server GMM，K=2。
- **Union**：10 个 client local GMM 的原始并集，14 个分量（client 大小 × local π 加权），不合并；作为诊断。
- **C（Multi-CA）**：先把 14 个分量用 complete-linkage normalized-Bures 2-Wasserstein 聚成 2 个 Macro（加权矩匹配）；再在每个 Macro 内做同样的合并，自适应停止：
  - 合并距离序列（每个 Macro 去掉最后一次合并到 1 的末端点）；
  - `tau = median + 1.4826 × MAD = 0.2400`（pooled median 0.1782，MAD 0.0417）；
  - 当下一合并距离 > tau 时停止；每个 Macro 至少 2 个 Micro。
  - 结果：Macro 0 合并 3 次 → 4 个 Micro；Macro 1 合并 4 次 → 3 个 Micro；**K=7**；Macro 质量 0.3651 / 0.6349。
- **B（Flat 同容量对照）**：同样的 14 个分量、同样的 complete-linkage Bures 距离和加权矩匹配，但不分 Macro，直接合并到 **K=7**。
- B/C 使用相同 client 权重、相同 latent 输入、相同采样预算（5000）和相同 server Decoder。
- 推断种子 0–9（ARI/NMI/Purity）；生成种子 0、1、2。

## 4. 主结果（ARI/NMI/Purity 为 seeds 0–2；括号为 SD）

| 模型 | K | ARI 0–2 | ARI 0–9 | NMI | Purity | NLL | Gen MMD | Gen SWD | 支持≥20 | Ghost | 质量<20 | Minority mass | Minority gen frac | Minority MMD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 2 | 0.6010 (0.0015) | 0.6053 (0.0048) | 0.4717 | 0.8887 | 17.5638 | 0.1879 | 0.2704 | 2 | 0 | 0.0000 | 0.2938 | 0.2913 | 0.3119 |
| B | 7 | 0.6168 (0.0031) | 0.6192 (0.0064) | 0.4893 | 0.8939 | 17.1290 | 0.16065 | 0.23166 | 5 | 1 | 0.0832 | 0.3957 | 0.3602 | 0.2912 |
| C | 7 | 0.6168 (0.0031) | 0.6192 (0.0064) | 0.4893 | 0.8939 | 17.1290 | 0.16065 | 0.23166 | 5 | 1 | 0.0832 | 0.3957 | 0.3602 | 0.2912 |
| Union | 14 | 0.6144 (0.0038) | 0.6201 (0.0060) | 0.4865 | 0.8931 | 17.1164 | 0.16030 | 0.23178 | 7 | 1 | 0.4219 | 0.4401 | 0.3707 | 0.2863 |

A 的 10 个 per-seed ARI 与历史 F01 `inference.csv` **逐位一致**（`exact_1e-9=true`），0–9 均值 0.6053，确认评价协议与历史一致。这同时验证了 B/C 使用的采样/推断流程。

### 生成 per-seed（MMD / SWD）

| 模型 | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| A | 0.18674 / 0.26961 | 0.18950 / 0.27149 | 0.18746 / 0.27016 |
| B/C | 0.15987 / 0.23139 | 0.15994 / 0.23037 | 0.16215 / 0.23321 |
| Union | 0.16002 / 0.23167 | 0.16085 / 0.23240 | 0.16003 / 0.23127 |

## 5. 核心结论：Multi-CA 与同容量 Flat GMM 完全相同

B 与 C 在 K=7 时：

- 叶分量的成员集合完全相同：
  `[[0,7], [1,10,11], [2,12], [3,8,13], [4], [5], [6,9]]`
- 权重、均值、log-variance 全部 `allclose`；
- Macro 映射完全相同；
- 所有 ARI/NMI/Purity/NLL/MMD/SWD/minority 指标逐位相同；
- `metrics_identical = true`。

**原因**：Macro 划分与 Macro 内 Micro 合并使用的是**同一棵 complete-linkage normalized-Bures 树**。Macro 层只是这棵树在 K=2 处的一个切分；在 K=7 处，两个 Macro 内的合并集合恰好等于全局 Flat 合并的前 7 个叶。因此层次结构只是同一组叶的**重新标注**，没有产生不同的 GMM。

**因此：Multi-CA 没有优于同容量 Flat GMM；B 与 C 是同一个模型。** 之前 elbow 停止规则（K=4）下 B 与 C 也完全相同，说明这一等价性不依赖于具体的自适应阈值。B 单独就实现了 B/C 相对 A 的全部差异；层次结构没有额外贡献。

A 与 B/C 的横向差异（ARI +0.0158、NLL −0.4347、MMD −0.0273）只能作为描述性参考：A 在 server 空间，B/C 在 client 空间；且 B=C。因此既不能把差异归因于 Multi-CA，也不能把它当作干净的容量效应估计。

## 6. Union 诊断

Union（K=14）的 10-seed ARI 0.6201 略高于 B/C 的 0.6192，NLL 17.116 vs 17.129，但：

- 7 个支持 Micro 中只有 7 个达到 ≥20 支持（1 个 ghost），**42.19% 的概率质量落在 <20 支持的 Micro 上**（B/C 为 8.32%）；
- 多出的容量主要形成低支持/ghost 分量，不是一个更稳健的模型。

## 7. B/C 的每-Micro 支持

| Micro | Macro | 权重 | 真实支持 | 多数类 | Ghost | 自生成 MMD |
|---:|---:|---:|---:|---:|---|---:|
| 0 | 0 | 0.127 | 101 | 0 (minority) | 否 | 0.295 |
| 1 | 1 | 0.275 | 130 | 1 | 否 | 0.1455 |
| 2 | 0 | 0.121 | 66 | 0 | 否 | 0.5275 |
| 3 | 1 | 0.303 | 244 | 1 | 否 | 0.1534 |
| 4 | 0 | 0.026 | 0 | — | **是** | — |
| 5 | 1 | 0.057 | 4 | 0 | 否 | — |
| 6 | 0 | 0.090 | 24 | 0 | 否 | 0.5306 |

5 个 Micro 达到 ≥20 支持；1 个 ghost（Micro 4）；1 个仅 4 个真实点（Micro 5）；质量<20 = 0.0832。Macro 0 有 4 个 Micro，Macro 1 有 3 个。

## 8. Minority / rare 诊断（初步）

Minority 类为 0，真实比例 0.3726。

| 模型 | Minority 质量 | 生成 Minority 比例 | Minority MMD | Minority SWD | Rare MMD | Rare SWD |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.2938 | 0.2913 | 0.3119 | 0.4340 | 0.3806 | 0.5614 |
| B/C | 0.3957 | 0.3602 | 0.2912 | 0.4107 | 0.5291 | 0.8122 |
| Union | 0.4401 | 0.3707 | 0.2863 | 0.3981 | 0.5237 | 0.8051 |

B/C 的 minority 质量与生成比例比 A 更接近真实比例，但这是容量/空间差异；B/C 的 bottom-quartile rare Micro 生成 MMD/SWD 反而更差（0.529/0.812 vs A 0.381/0.561）。BC 只有 2 个类，**这些只是初步诊断，不能宣称验证了客户端独有稀有模式**。

## 9. 合并树

C：

- Macro 质量 0.3651 / 0.6349；Macro 大小 7 / 7；Micro 分配 4 / 3。
- 停止阈值 tau = 0.2400（pooled median 0.1782，MAD 0.0417）。
- Macro 0 距离 `[0.1501, 0.1766, 0.1798, 0.3168, 0.3193, 0.6735]` → 保留 4 个叶。
- Macro 1 距离 `[0.1221, 0.1228, 0.1739, 0.1902, 0.3243, 0.6513]` → 保留 3 个叶。

B：Flat 目标 K=7，叶集合与 C 完全相同。

完整合并事件、成员索引、权重和距离在 `results/merge_tree.json` 与 `results/aggregation.json`。

## 10. 验证与限制

- A 的 per-seed ARI 与历史 `inference.csv` 完全一致（0–9 均值 0.6053）。
- 未训练、未重建客户端、未修改权重；checkpoint SHA256 校验通过。
- 潜在空间不完全兼容：client 共享空间 vs server 空间；A vs B/C 是跨空间描述，B vs C 才是受控同空间比较。
- BC 无独立测试集；所有指标为样本内。
- Minority/rare 指标仅初步诊断；局部标签只用于评价和 Micro 的多数类映射，不参与聚合。
- Multi-CA 与 Flat 的等价性是“同一距离/同一 linkage 的同一棵树的不同切法”导致的；若改用不同的 Macro linkage 或允许跨 Macro 合并，可能得到不同结果，但那是另一种方法，本次未改。

## 11. 产物

- `results/metrics.json`、`results/metrics.csv`
- `results/aggregation.json`、`results/merge_tree.json`
- `results/latent_compatibility.json`
- `results/per_micro.csv`
- `logs/run_final.log`
- 监控：`/Users/cheeseowl/Desktop/F01_MultiCA_冻结聚合监控.command`
- 代码：`scripts/multica_core.py`、`scripts/adaptive_multica.py`、`scripts/run_frozen_v3.py`
