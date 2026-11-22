# 实测结果及论文对照

ACC 以百分比显示。Purity 取值为 0–1，ARI 经随机一致性校正，可以为负。均值用有效运行计算，次数一列为有效／尝试。全文复现状态见 AGENTS.md。

## 论文方案或发布程序的复跑

| 模型 | 数据集 | 指标 | 实测 | 论文 | 次数 | 说明 |
|---|---|---|---:|---:|---:|---|
| VaDE | Reuters-10K | ACC | 81.17% | 79.83% | 1/1 | 15 epoch，原脚本共完成两次，ACC 来自其中一次结束后的评估 |
| VaDE | MNIST | ACC | 94.46% | 94.46% | 1/1 | 作者权重复评 |
| FedHCA | Ecoli | Purity | 0.6852 | 0.7800 | 10/10 | 作者公开客户端分区 |
| FedHCA | Ecoli | ARI | 0.4401 | 0.6880 | 10/10 | 作者公开客户端分区 |
| FedHCA | User Knowledge Modeling | Purity | 0.4680 | 0.5290 | 10/10 | 作者公开客户端分区 |
| FedHCA | User Knowledge Modeling | ARI | 0.1282 | 0.2270 | 10/10 | 作者公开客户端分区 |
| FedHCA | Vehicle Silhouettes | Purity | 0.3750 | 0.4060 | 10/10 | 作者公开客户端分区 |
| FedHCA | Vehicle Silhouettes | ARI | 0.0638 | 0.0880 | 10/10 | 作者公开客户端分区 |
| FedHCA | HCV Egyptian Patients | Purity | 0.3431 | 0.3240 | 8/10 | 作者公开客户端分区 |
| FedHCA | HCV Egyptian Patients | ARI | 0.0098 | 0.0170 | 8/10 | 作者公开客户端分区 |
| FedHCA | Yeast | Purity | 0.4430 | 0.5270 | 10/10 | 作者公开客户端分区 |
| FedHCA | Yeast | ARI | 0.1231 | 0.2450 | 10/10 | 作者公开客户端分区 |
| FedHCA | Cardiotocography | Purity | 0.5273 | 0.6000 | 10/10 | 作者公开客户端分区 |
| FedHCA | Cardiotocography | ARI | 0.2136 | 0.2800 | 10/10 | 作者公开客户端分区 |
| FedHCA | Landsat Satellite | Purity | 0.5996 | 0.7520 | 10/10 | 作者公开客户端分区 |
| FedHCA | Landsat Satellite | ARI | 0.3875 | 0.5390 | 10/10 | 作者公开客户端分区 |
| FedHCA | Wine Quality | Purity | 0.5095 | 0.5140 | 10/10 | 作者公开客户端分区 |
| FedHCA | Wine Quality | ARI | 0.0463 | 0.0480 | 10/10 | 作者公开客户端分区 |
| FedHCA | Pen-Based Digits | Purity | 0.6327 | 0.6620 | 10/10 | 作者公开客户端分区 |
| FedHCA | Pen-Based Digits | ARI | 0.4735 | 0.5160 | 10/10 | 作者公开客户端分区 |
| FedHCA | Letter Recognition | Purity | 0.2260 | 0.3460 | 7/10 | 固定种子；LE_*.jsonl 为补充证据；原入口另有 5 次记录 |
| FedHCA | Letter Recognition | ARI | 0.0637 | 0.1740 | 7/10 | 固定种子；LE_*.jsonl 为补充证据；原入口另有 5 次记录 |
| Fed-HIRE | Iris | Purity | 0.5760 | 0.7867 | 10/10 | 作者公开 Iris 分区 |
| Fed-HIRE | Accent | Purity | 0.5362 | 0.5805 | 10/10 | 作者生成器重建分区，seed 42 |
| Fed-HIRE | Yeast | Purity | 0.5036 | 0.4650 | 10/10 | 作者生成器重建分区，seed 42 |
| Fed-HIRE | Wine Quality | Purity | 0.4695 | 0.4734 | 10/10 | 作者生成器重建分区，seed 42 |

## 诊断和独立实现

| 模型 | 数据集 | 指标 | 实测 | 论文 | 次数 | 说明 |
|---|---|---|---:|---:|---:|---|
| Fed-HIRE | Iris | Purity | 0.8033 | — | 10/10 | 仓库默认学习率 0.05 |
| Fed-HIRE | Iris | Purity | 0.5767 | — | 10/10 | 同一组种子，学习率 .0001 |
| Fed-HIRE | Iris | Purity | 0.7633 | — | 10/10 | 同一组种子，学习率 .05 |
| FedVaDE | Iris | Purity | 0.4333 | 0.8840 | 3/3 | 自建分区，150 通信轮，独立实现 |
| FedVaDE | Iris | ARI | 0.1048 | 0.7190 | 3/3 | 自建分区，150 通信轮，独立实现 |
| FedVaDE | Breast Cancer | Purity | 0.8125 | 0.8990 | 3/3 | 自建分区，150 通信轮，独立实现 |
| FedVaDE | Breast Cancer | ARI | 0.3947 | 0.6350 | 3/3 | 自建分区，150 通信轮，独立实现 |
| FedVaDE | Banknote | Purity | 0.6256 | 0.7440 | 3/3 | 自建分区，150 通信轮，独立实现 |
| FedVaDE | Banknote | ARI | 0.0904 | 0.1710 | 3/3 | 自建分区，150 通信轮，独立实现 |
| VaDE | MNIST | ACC | 79.68% | 94.46% | 1/1 | 同一训练过程第 5 epoch 的检查点 |
| VaDE | MNIST | ACC | 79.99% | 94.46% | 1/1 | 同一训练过程第 10 epoch 的检查点 |

summary.csv 也记录标准差与证据位置。FedHCA 论文对照来自正文 Table I。Fed-HIRE 对照来自 Table 2，主指标为 Purity。VaDE 对照来自 Table 2。FedVaDE 对照来自 Table II。
