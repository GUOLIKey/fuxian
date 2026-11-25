# FedVaDE Local GMM 接口与参数转换核查

## 原始 Local GMM 的特征定义

`FedVaDE_source/model.py::pre_train`：

1. 预训练 encoder/decoder 后，执行
   `c['vade'].encoder.log_sigma2_l.load_state_dict(c['vade'].encoder.mu_l.state_dict())`；
2. 对客户端数据执行 `mu, sigma = c['vade'].encoder(x)`，并断言
   `F.mse_loss(mu, sigma) == 0`；
3. `local_mu.append(mu)`；
4. `GaussianMixture(n_components=K_l, covariance_type='diag', random_state=42).fit(local_mu)`。

因此原始 Local GMM 的输入特征是 **encoder 的均值 `mu`**（10 维），不是后验采样值，
也不是 `log_sigma2_l`。本次 A0–A5 全部使用冻结客户端 encoder 的 `mu`，
`z_i = encoder(x_i).mu`；没有用采样 z 替代，也没有反向替换。

## 参数形状与表示

| 参数 | FedVaDE 形状/含义 |
|---|---|
| `rho_` | `(K,)`，log 混合权重；`pi = softmax(rho)` |
| `mu_c` | `(K, 10)`，分量均值 |
| `log_sigma2_c` | `(K, 10)`，对角协方差的 log 方差 |
| 方差 | `var = exp(log_sigma2_c)`；对角 GMM |
| 正则 | sklearn `GaussianMixture` 默认 `reg_covar=1e-6` |
| 权重写入 | 作者用 `rho = log(pi + 1e-10)` |
| 方差写入 | 作者用 `log_sigma2 = log(covariances)`；covariances 已由 `reg_covar` 保证正 |

## A4 的矩估计转换（无 EM）

对 ACD 给出的第 k 个子簇 `C_k`，`n_k=|C_k|`，`N` 为客户端拟合样本数：

- `pi_k = n_k / N`
- `mu_k = mean(z_i for i in C_k)`
- `var_k = mean((z_i - mu_k)^2 for i in C_k)`（population，`ddof=0`）
- `var_k = max(var_k, 1e-6)`（与原始 GMM `reg_covar` 一致；记录）
- `rho_k = log(pi_k + 1e-10)`
- `log_sigma2_k = log(var_k)`

不进行 EM、不追加优化、不合并/删除低支持分量；单样本或低支持分量保留并标记。
ACD 的 `belta` 竞争权重没有作为 π；π 只来自最终样本计数。

## 接口验证

对每个 A4 unit（10 clients × 3 seeds = 30 个 full 拟合，另有 30 个 holdout 拟合）：

- `rho_`、`mu_c`、`log_sigma2_c` 形状正确、全部有限；
- `pi` 和为 1，方差 > 0；
- 将参数装回该客户端 `VaDE`（冻结 encoder/decoder），`load_state_dict(strict=True)`；
- 调用 `predict` 与 `decoder`，输出有限且标签在 `[0,K)`；
- 结果：**full 30/30 OK，holdout 30/30 OK**；无 NaN/Inf；无失败。

A1/A2/A3/A5 的 sklearn GMM 参数按同一规则转换为 FedVaDE 形状；A5 使用与 A4
逐客户端、逐种子、逐 split 相同的 K，不使用全局平均 K。

## BIC 搜索范围（预先固定）

- A2（原始范围）：`2 .. min(14, n_fit // 5)`。
- A3（扩展范围）：`2 .. min(30, n_fit // 5)`。
- `n_fit // 5` 是预先固定的样本量约束（平均每分量至少约 5 个拟合样本），
  不根据评价结果调整。
- 原始上限 14 来自此前项目的 BIC 审计
  （`experiments/multica_pretrained_2x2_v1/scripts/build_init.py::bic_select`
  与 `configs/bic_*.json` 的 2..14）。
- 实际所有客户端 A2/A3 选中的 K 都在 2..12；A3 的扩展上限没有改变任何 unit 的
  选择（A2 与 A3 在 90/90 次检查中 K 相同）。
- BIC 只使用拟合数据与无标签 GMM；真实标签只在事后评价使用。

## 留出验证

- 每个客户端固定 20% 本地留出，split 种子 `20261125 + client_id`，不改变原始联邦
  划分，不使用标签分层。
- A1–A5 在 80% 拟合集上重新拟合/BIC/ACD，评价 20% 留出 NLL。
- A0 原始 Local GMM 已使用完整本地数据，因此留出 NLL 仅作历史参考，
  不与 A1–A5 作严格泛化比较。
- 冻结 encoder 已在原始训练数据上训练；留出只评价 GMM 拟合泛化，
  不能宣称整个 FedVaDE 的独立测试性能。
