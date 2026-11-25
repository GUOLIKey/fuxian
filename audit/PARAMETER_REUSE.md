# Local GMM 参数复用核查

本次五组直接复用上一轮 `experiments/acd_f01_frozen_v1/results/units/` 的
**完整本地数据拟合**参数；没有使用 holdout 子集拟合的 GMM。

| 本次组 | 上一轮方法 | 来源 |
|---|---|---|
| A | A0 | Checkpoint 原始 Local GMM（10 clients） |
| B | A1 | `unit.methods.A1.full.params`（原始 K + ordinary GMM） |
| C | A2 | `unit.methods.A2.full.params`（BIC 原始范围） |
| D | A4 | `unit.methods.A4.full.params`（作者 ACD + 矩估计） |
| E | A5 | `unit.methods.A5.full.params`（与 A4 同 K 的 ordinary GMM） |

核查结果：

- 30 个 `(client, seed)` unit 全部存在；
- A1/A2/A4/A5 的 full 参数均存在；
- D/E 的 **逐客户端、逐种子** K 完全相同（30/30）；
- 上一轮 full 参数是完整本地数据拟合，不是 80% holdout 拟合；
- 本轮没有重新拟合 ACD/GMM，只加载 full 参数并冻结客户端；
- 参数转换沿用上一轮：`pi=n/N`、population variance、`var=max(var,1e-6)`、
  `rho=log(pi+1e-10)`、`log_sigma2=log(var)`；A 组使用 Checkpoint 原始 `rho_`。
- 本轮只保存各组 Local GMM 参数、最终 **server** 状态/优化器/调度器和结果；
  不保存客户端 encoder/decoder，也不生成新的完整联邦 Checkpoint。
