# FedVaDE 服务器机制与源码一致性审计

## 使用的源码

| 文件 | SHA256 |
|---|---|
| `fuxian-round2/FedVaDE_source/main.py` | `33bd9927437c444f3f5355813dba8de1896fd59e2eb3fbd9f16960ff3041b87b` |
| `fuxian-round2/FedVaDE_source/model.py` | `6cdef32b831d090ee79fa952a6f55386bba2e37a6c0c38d6ebc865441c1379d9` |
| `fuxian-round2/FedVaDE_source/dataloader.py` | `28fc737a243ea550dc4c18ec98928a51af2701132452c68d1e7ef0d1db9c9cad` |
| `fuxian-round2/FedVaDE_source/experiment.py` | `979739dacb8950e074d8cbc775722aaa5ff8c304a1dba13efc64d97d9da4d758` |
| F01 full checkpoint | `3be341c451a186d1f36673080164af208993d215f0c2cb03709407892f7d6f3e` |

以上哈希与 F01 复现配置中记录的作者源码/数据哈希一致。

## 原始服务器更新阶段（`main.py`）

1. 每个客户端有冻结的 local GMM：`rho_`、`mu_c`、`log_sigma2_c`。
2. 服务器调用
   `gmm_sampling(clients_gmm_param, weight, args.sample)`，其中
   `weight = client_size / sum(client_size)`。
   - `sample` 默认 **100**；源码是**总共** 100 个样本，
     `np.random.choice(K, size=100, p=global_pi)`，不是每客户端/每分量 100。
   - 全局分量权重是各客户端 local GMM 的 π 按 client weight 扩展后拼接。
3. `Z_hat` 先由 **server decoder** 解码为 `X_hat`。
4. 为了构造 `decoded_X_hat`，按 `lengths`/`y_hat` 的全局分量归属，把每个样本交给
   其所属客户端的 **client decoder** 解码。**训练中不存在用 server decoder 解码
   client latent 的路径**；server decoder 只解码 server 侧 `Z_hat`。
5. `loss = server.ELBO_Loss_server(X_hat, decoded_X_hat)`。
6. `server.optimizer.zero_grad(); loss.backward(); server.optimizer.step();
   server.lr_s.step()`。
   - 原始 `server.optimizer = Adam(server.parameters(), lr=0.002)`，包含 server
     encoder/decoder **和 server GMM**（`rho_`、`mu_c`、`log_sigma2_c`）。
   - 原始 `server.lr_s = StepLR(step_size=10, gamma=0.85)`，每个正式轮 step 一次。
   - Client optimizer/scheduler 只用于客户端本地训练；本实验冻结客户端，因此不使用。

## 本实验的严格保留与唯一替换

- 保留：初始 server encoder/decoder/GMM；client weight；sample=100；server 更新
  loss；Adam；StepLR；server GMM 由 server optimizer 更新；评价协议。
- 唯一替换：每组每个客户端的 **local GMM 参数**。
- 跳过：客户端本地训练、client 参数更新、服务器向客户端的广播覆盖。
  这是任务规定的“冻结客户端、只执行原始服务器更新阶段”。
- 服务器 optimizer/scheduler 从 F01 @100 完整 Checkpoint 的
  `server_optimizer` / `server_scheduler` 恢复。客户端优化器/scheduler 不使用。
- 服务器 GMM 不是由 Client GMM 直接替换：它从 Checkpoint 的 server GMM 初始化，
  随后由原始 server ELBO 更新；Client GMM 只用于采样和 client decoder 解码。
- 正式每轮训练前设置同一实验种子的 NumPy/Torch RNG（`seed*100000 + round`），
  五组同种子共享随机流；不恢复 Checkpoint 的 RNG 状态，因为这是新的配对干预实验。
  这一选择在报告中说明。

## 评价协议

- Global ARI/NMI/Purity 使用历史 F01 的 Round 6/F01 评价器：
  `evaluate_saved_round6.py::evaluate_one`，在 2129 条 author client records 上，
  每个 inference seed 0–9 重新设 `torch.manual_seed(seed)`，按 author client
  DataLoader 逐 batch 调用 `server.predict`。
- 初始 Checkpoint 重载评价均值 **0.605326980742813**，与历史 F01 @100
  0.6053 精确一致（10 inference seeds）。
- 全局生成使用当前 **server GMM + server decoder**；不使用 client GMM 配
  server decoder。生成预算 5000，生成种子 0/1/2；与 10 个 inference seeds 分开统计。
- Global GMM NLL 在 569 条唯一标准化真实数据的 server-encoder 均值 latent 上计算。
- 真实标签只用于事后 ARI/NMI/Purity、分量多数类和 minority 诊断，不参与采样、
  聚合、优化器或选择。

## 第 101 轮完整流程确定性恢复检查

使用 `round7_f01_full_repro/scripts/resume_verify.py`：从 F01 @100 完整 Checkpoint
恢复 server、10 clients、optimizer、scheduler、RNG，运行原始第 101 轮完整流程，
再与 F01 `rounds.jsonl` 第 101 轮比较。

结果：`exact_1e-6 = true`；server_loss、purity、ARI delta = 0；client_loss 仅
4.77e-8（float32 累加）。该检查只用于接口/训练路径核查，不计入正式五组结果。
