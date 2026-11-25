# ACD 原始实现核查

## 论文与源码

| 项 | 值 |
|---|---|
| 论文 | *Stitch the Fragments: One-Shot Hierarchical Federated Clustering* |
| arXiv | `2601.06404v2` |
| 本地论文 | `/Users/cheeseowl/Desktop/Cai 等 - 2026 - Stitch the Fragments One-Shot Hierarchical Federated Clustering(2).pdf` |
| 作者源码 | `hongdream/Fed-HIRE` |
| Pin commit | `c116c8354f988d4c3080bbd9b1ab685117e78787` |
| 本地路径 | `reproduction_runs/upstream/Fed-HIRE/opensource/MultiClustering.py` |
| SHA256 | `2d49665775bdc0af91874a48c2533276dcf5a23086c1542f5db3b3578d0bfc49` |
| 本次使用 | `FCPL` / `_batch_cpl`，原样复制到 `vendor/FedHIRE/MultiClustering.py`，未修改 |

`fedClustering.py` 在客户端调用 `FCPL(l["data"], rate, init_k)`；`main.py` 默认
`learning_rate=0.05`、`init_k=0.5`。本次 A4 使用 `rate=0.05`、`k0=0.5`（`k0` 是
样本比例，`round(0.5*n)` 个候选簇），`MAX_ITER=100`，并将 `random_state` 设为实验
种子 0/1/2。把 `random_state` 暴露为种子是接口适配，不改变竞争机制。

## 源码行为（实际采用）

- 初始化：随机选 `k` 个样本作为种子中心，种子样本赋 0..k-1，其余样本随机赋标签；
  计算簇和、簇大小。
- 相似度：`s = 1 / (1 + ||x-c||^2)`。
- 每轮：
  - 中心为当前成员均值（空簇中心为 0）。
  - 得分 `score = s * w * gamma`，其中
    `w = 1/(1+exp(-10*belta+5))`，`gamma = 1 - cnt/sum(cnt)`。
  - 每个样本取 argmax 为新标签。
  - `belta += rate * winner_count`。
  - runner-up 惩罚：`belta -= rate * sum(score[runners])`。
  - `cnt += winner_count`。
  - 标签不变则停止，最多 100 轮。
- 结束：只保留非空簇，重新编号为 0..K-1，返回 `theta`（非空簇均值）、`label`、`K`。
- 空簇不产生参数；最终 K 是非空簇数量。

## 论文公式与源码的明确差异

本次优先使用作者源码，下列差异逐条记录；**A4 结果反映的是源码版本，不是按论文公式
重写的版本**：

1. **权重映射的位移不同。** 论文 Eq. (7)：`w = 1/(1+exp(-10(τ+5)))`；源码为
   `1/(1+exp(-10*belta+5))`，即 `sigmoid(10*belta-5)`，不是 `sigmoid(10(belta+5))`。
2. **奖励/惩罚粒度不同。** 论文按单个对象：`τ_v = τ_v + η`，runner-up
   `τ_r = τ_r - η*s(x,c_r)`；源码按轮：`belta += η*winner_count`，runner-up 惩罚累加
   的是完整 `score`（含 `gamma` 与 `w`），不是仅 `s`。
3. **淘汰规则不同。** 论文描述当 `w_j→0` 时淘汰并重新分配对象；源码没有显式权重阈值
   淘汰，只在最终把空簇丢掉并重编号。
4. **初始化细节不同。** 论文给出 k0 个候选 clusterlet 的竞争框架；源码的具体初始化是
   随机种子点 + 随机初始标签（见上）。
5. **最终归属：** 源码用最后一轮 argmax 标签并只保留非空簇；论文也是按最大相似度
   归属，但公式细节不同。

**结论：** A4 是作者 `FCPL` 源码版本；不能把 A4 的负面结果直接表述为“论文公式的 ACD
失败”。按论文公式重写会改变 ACD 内部机制，本次按任务要求未做，也没有调参寻找正面结果。

## 运行环境

- 服务器 venv-cpu：Python 3.11.18，numpy 1.25.2，scikit-learn 1.7.2。
- 作者 README 记录：Python 3.11，numpy 1.26.4，progressbar2 4.5.0，munkres 1.1.4，
  scikit-learn 1.5.2。ACD 模块只依赖 numpy；本次使用服务器 numpy 1.25.2，未改源码。
- 未引入 Fed-HIRE 的服务器层次聚类（MHR）。
