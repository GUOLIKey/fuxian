# FedVaDE + ACD 全局性能验证实验

**状态：完成。** 五组 × 三个实验种子 × 10 个服务器更新轮次，共 15 组短程实验。
主指标为 **Global ARI**（round 10 预先确定），不是 Local ARI。未做 300 轮训练。

## 1. 初始参考与 A 对照的区别

| 项 | Global ARI (10 inference seeds) |
|---|---:|
| F01 @100 冻结参考 | **0.605326980742813** |
| A 组（同预算 10 轮服务器更新的对照） | 0.5572 ± 0.0114 |
| B 组 | 0.5282 ± 0.0156 |
| C 组 | 0.5451 ± 0.0062 |
| D 组（ACD） | 0.5580 ± 0.0112 |
| E 组（同 K GMM） | 0.5596 ± 0.0289 |

冻结参考 0.6053 是单独的参考值；A 组是相同 10 轮服务器预算的对照，二者不混淆。
**五组在 10 轮后都低于冻结参考**；A 对照本身下降 0.0481，说明“冻结客户端 + 仅服务器
更新”这一短程干预本身就会改变/削弱全局性能。

## 2. 源码与服务器机制核查

- FedVaDE 源码哈希：
  `main.py 33bd9927…`、`model.py 6cdef32b…`、`dataloader.py 28fc737a…`、
  `experiment.py 979739da…`；与 F01 复现配置一致。
- Checkpoint SHA256 `3be341c4…`，round 100，10 个真实 client，直接使用 partition。
- 原始服务器阶段：
  - `gmm_sampling(client_gmms, weight, sample=100)` 从各 client local GMM 采样；
    **100 是总样本数**，不是每 client/每分量 100。
  - `Z_hat` 由 **server decoder** 解码；`decoded_X_hat` 按采样分量的 client 归属，
    分别由对应 **client decoder** 解码。
  - `server.ELBO_Loss_server(X_hat, decoded_X_hat)`；server optimizer 包含
    server encoder/decoder **和 server GMM**；Adam lr=0.002；StepLR(10,0.85)
    每轮 step 一次。
  - 服务器 optimizer/scheduler 从 F01 @100 Checkpoint 的
    `server_optimizer`/`server_scheduler` 恢复；客户端优化器不使用。
  - 本实验跳过客户端训练和广播覆盖，只执行服务器更新；client encoder/decoder
    与 local GMM 冻结。**没有把 Client GMM 直接作为 Server GMM，也没有用 Server
    Decoder 解码 Client latent。**
- 采样随机数：同一实验种子和轮次下，五组使用相同的 NumPy/Torch seed
  （`seed*100000 + round`），每轮重新设种子；不恢复 Checkpoint RNG（这是新的配对干预）。
- Global ARI/NMI/Purity：`evaluate_saved_round6.evaluate_one`，2129 条 author
  client records，10 个 inference seeds 0–9。
- 初始重载：均值 **0.605326980742813**，与 0.6053 精确一致。
- 第 101 轮完整流程恢复检查：`exact_1e-6=true`；server_loss/purity/ARI delta=0；
  client_loss 仅 4.77e-8。该检查不计入正式结果。

## 3. 五组 Local GMM 来源

直接复用上一轮 `acd_f01_frozen_v1/results/units/` 的**完整本地数据**参数（非 holdout）：

| 组 | 上一轮 | Local GMM |
|---|---|---|
| A | A0 | Checkpoint 原始 Local GMM |
| B | A1 | 原始 K + ordinary GMM 重拟合 |
| C | A2 | BIC 原始范围 + ordinary GMM |
| D | A4 | 作者 ACD（FCPL）+ 矩估计，无 EM |
| E | A5 | 与 D 逐客户端/逐种子相同 K 的 ordinary GMM |

D/E 的逐客户端 K 在 30/30 个 `(client, seed)` 上完全一致。D 的参数转换沿用上一轮：
`pi=n/N`、population variance、`var=max(var,1e-6)`、`rho=log(pi+1e-10)`、
`log_sigma2=log(var)`。ACD 为作者源码版本
（`hongdream/Fed-HIRE`, commit `c116c835…`, `FCPL`）；论文/源码差异见上一轮
`acd_f01_frozen_v1/audit/ACD_SOURCE.md`，本次没有按论文重写或追加淘汰/EM/限制 K。

## 4. Round-10 结果

均值 ± 实验种子 SD（3 个 experiment seeds）；括号中的 inference SD 是同一 run 内
10 个 inference seeds 的平均 SD，**与实验种子 SD 分开**，不混合。

| 组 | Local K mean | Global ARI | Global NMI | Global Purity | Global GMM NLL | Gen MMD | Gen SWD | Minority frac | Minority MMD | Minority SWD | Server loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 1.4 | 0.5572 ± 0.0114 (0.0052) | 0.4300 | 0.8742 | 19.1253 | 0.2094 | 0.3122 | 0.2946 | 0.3386 | 0.4777 | 2.2805 |
| B | 1.4 | 0.5282 ± 0.0156 (0.0049) | 0.4011 | 0.8647 | 17.9267 | 0.1990 | 0.3114 | 0.2949 | 0.2966 | 0.4088 | 2.3370 |
| C | 6.7 | 0.5451 ± 0.0062 (0.0042) | 0.4181 | 0.8707 | 17.6126 | 0.1966 | 0.3062 | 0.2949 | 0.2921 | 0.4030 | 2.1501 |
| **D** | **91.2** | **0.5580 ± 0.0112 (0.0062)** | **0.4300** | **0.8748** | **17.4779** | **0.1958** | **0.3025** | **0.2949** | **0.2887** | **0.3986** | **2.2988** |
| E | 91.2 | 0.5596 ± 0.0289 (0.0063) | 0.4317 | 0.8753 | 17.4741 | 0.1980 | 0.3065 | 0.2949 | 0.2877 | 0.3979 | 2.2303 |

所有组 round 0 都是 0.6053；所有组的最高 ARI 都是 **round 0**（exploratory），
没有任何组在 10 轮内超过冻结参考。正式比较固定使用 round 10。

### 逐实验种子

| 组 | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| A | 0.5638 | 0.5440 | 0.5638 |
| B | 0.5373 | 0.5371 | 0.5102 |
| C | 0.5507 | 0.5384 | 0.5463 |
| D | 0.5454 | 0.5621 | 0.5666 |
| E | 0.5587 | 0.5889 | 0.5311 |

## 5. 配对差值（round 10，D 或 B 减对照）

| 对比 | ΔGlobal ARI | ΔNMI | ΔPurity | ΔNLL | ΔMMD | ΔSWD | ΔMinority frac | ΔServer loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D−A | **+0.0008 ± 0.0184** | −0.0001 | +0.0006 | −1.6474 | −0.01355 | −0.00967 | +0.00029 | +0.0183 |
| B−A | **−0.0290 ± 0.0234** | −0.0289 | −0.0095 | −1.1986 | −0.01038 | −0.00076 | +0.00029 | +0.0565 |
| D−B | +0.0298 ± 0.0246 | +0.0288 | +0.0101 | −0.4488 | −0.00317 | −0.00891 | 0.00000 | −0.0382 |
| D−C | +0.0129 ± 0.0159 | +0.0119 | +0.0042 | −0.1347 | −0.00080 | −0.00367 | +0.00002 | +0.1488 |
| D−E | −0.0016 ± 0.0328 | −0.0018 | −0.0005 | +0.0038 | −0.00215 | −0.00396 | 0.00000 | +0.0685 |

D−A 的 ARI 逐 seed 为 −0.0185、+0.0181、+0.0029：方向不一致，均值接近 0。
D−E 的 ARI 逐 seed 为 −0.0134、−0.0269、+0.0355：同样方向不一致。
Negative ΔNLL/MMD/SWD 更好。

## 6. 实际采样覆盖（round 10）

`sample=100` 是总采样数（10 个客户端合计），每客户端约 10 个样本。
下表 unique 为每客户端平均被采样到的局部分量数；low 为每客户端平均低支持
（<5 真实样本）分量数。

| 组 | Local K mean | Unique sampled/client | Low-support sampled/client | Low-support total/client | 采样 unique 中低支持占比 |
|---|---:|---:|---:|---:|---:|
| A | 1.4 | 1.4 | 0.0 | 0.0 | 0% |
| B | 1.4 | 1.4 | 0.0 | 0.0 | 0% |
| C | 6.7 | 4.2 | 0.1 | 1.2 | ~2% |
| **D** | **91.2** | **9.3** | **8.0** | **85.0** | **~86%** |
| E | 91.2 | 9.3 | 6.2 | 78.6 | ~66% |

D 只采样到约 **9.3/91.2（10%）** 的局部分量，其中约 **86%** 是支持<5 的低支持分量；
E 约 66%。在原始 `sample=100` 总预算下，ACD 的高 K 使服务器看到的本地结构非常稀疏，
这也是 D/E 的全局性能无法超过 A 的重要机制。

## 7. 核心判断

1. **D vs A：ACD 是否改善原始 FedVaDE 的服务器表现？**
   **没有可重复的 Global ARI 改善。** 均值仅 +0.0008（实验种子 SD 0.0184），
   逐 seed 方向不一致（−0.0185/+0.0181/+0.0029）。NLL/MMD/SWD 一致更好
   （ΔNLL −1.65、ΔMMD −0.0136、ΔSWD −0.0097），但主指标 Global ARI 不支持改善。
2. **B vs A：重新拟合 GMM 本身是否有效？**
   否。B−A 的 Global ARI −0.0290（三个种子全负），尽管 NLL/MMD 变好；
   在原始 K 上重新拟合普通 GMM 反而损害全局聚类。
3. **D vs B：ACD 是否优于保持原始 K 的重新拟合？**
   是相对 B 的 +0.0298（三个种子全正），但 B 本身已经比 A 差；
   D 相对 A 仍接近 0。因此这不是支持 ACD 的独立证据。
4. **D vs C：ACD 是否优于普通 BIC 自适应聚类？**
   +0.0129 ± 0.0159；逐 seed 为 −0.0054/+0.0236/+0.0204，只有 2/3 为正；
   NLL/MMD 略好。方向偏正但不稳定，不能宣称稳健优势。
5. **D vs E：ACD 是否具有独立于分量数量的收益？**
   否。E 与 D 逐客户端/逐种子 K 完全一致；D−E ARI = −0.0016 ± 0.0328，
   逐 seed 方向混合；E 的最终均值 ARI 0.5596 略高于 D 的 0.5580，
   E 的 NLL 也略好；MMD/SWD D 略好。没有 ACD 独立收益。
6. **所有组都低于冻结 0.6053**；A 对照本身下降 0.0481，说明 10 轮服务器-only
   干预本身就会降低全局性能；D/E 的下降最小（−0.0473/−0.0458），但 D 与 A 持平，
   不是 ACD 的恢复效果。最高 ARI 都是 round 0，仅作探索性记录。

**结论：在这次冻结客户端、仅服务器 10 轮更新的短程实验中，作者 ACD 的 Local GMM
没有带来稳健的 Global ARI 改善。D 的 NLL/MMD/SWD 较好，但同 K 的 E 给出了相近结果，
因此不能把这些生成/密度收益归因于 ACD；所有组都没有超过未修改的冻结参考 0.6053。
这是一次负面/中性结果。**

## 8. 限制

- 冻结客户端、跳过客户端训练与广播，只执行服务器更新；不是完整端到端联邦训练。
- A 组是相同服务器预算的对照，不是未修改的 0.6053 冻结参考。
- ACD 的 K 很大（本地平均 91），在 `sample=100` 总预算下采样覆盖极稀疏；
  低支持分量占采样到的 unique 分量约 86%（D）。
- D/E 的 Local GMM 参数来自上一轮完整数据拟合；未使用 holdout 子集。
- 3 个实验种子、10 个 inference seeds；未做显著性检验；BC 无独立测试集。
- ACD 为作者源码版本；论文公式与源码存在差异，不能把结果推广为“论文 ACD”。
- 最高 ARI 为 round 0，仅探索性；正式比较固定 round 10，未延长到 300 轮。

## 9. 产物

- `results/runs/{A..E}_seed{0,1,2}.json`：15 组逐轮结果。
- `results/per_seed.csv`、`summary.csv`、`paired.csv`、`curves.csv`、
  `coverage_summary.csv`、`highest.csv`、`summary.json`。
- `results/rounds.jsonl`、`coverage.jsonl`、`initial_check.json`、`smoke.json`。
- `results/group_gmms/`：各组 Local GMM 参数（JSON，仅服务器/本地保留，作为权重不上传）；
  `results/group_gmms_manifest.json`：文件哈希与各 client K；`results/final_states/`：
  服务器最终状态/优化器/调度器（仅服务器保留，不上传）。
- `figures/global_ari_curve.png`、`final_ari.png`、`paired_ari.png`、
  `coverage.png`、`global_nmi_curve.png`、`global_purity_curve.png`、
  `global_nll_curve.png`、`global_mmd_curve.png`、`global_swd_curve.png`、
  `server_loss_curve.png`。
- `audit/SOURCE_AUDIT.md`、`PARAMETER_REUSE.md`、`INITIAL_CHECK.md`、
  `ROUND101_RECOVERY.md`、`SMOKE_TEST.md`。
- 监控：`/Users/cheeseowl/Desktop/ACD_F01_全局实验监控.command`。
