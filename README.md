# 联邦聚类论文复现实验

本仓库保存 FedVaDE、FedHCA、Fed-HIRE 和 VaDE 复现过程中自行编写的实验脚本、配置、审查记录、指标与原始文本日志。作者交付的 FedVaDE 源码、论文、数据和模型权重，以及完整训练 checkpoint 保留在科研服务器，不上传到此仓库。以下是**实际观测结果**，不代表严格论文复现成功。

## 最新结果：FedVaDE Round 7–8

数据集为 Breast Cancer；训练种子固定为 0。Round 7 使用 16 组粗搜索与 4 组精细搜索配置，其中 19 次为新增训练、1 次复用既有基线。预训练 50 轮；粗搜索正式训练 150 轮，精细搜索 300 轮。Round 8 从 Round 7 的 F01 第 100 轮服务器权重出发，重新初始化缺失的客户端、优化器、调度器和随机状态，进行 4 组 20 轮的权重初始化续训；这**不是精确恢复训练**。

| 对象 | LR | KL 权重 | 调度 | 正式轮次/选择点 | ARI（10 次后验推断） | Purity（10 次后验推断） |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| 作者权重复评参考 | 未知 | 未知 | 未知 | 未知 | 约 0.6295 | 见 [Round 6](experiments/round6_checkpoint_diagnosis/round6_report.md) |
| F01 固定终点 | 0.002 | 0.05 | 每 epoch | 300 | 0.5138 | 0.8605 |
| F01 探索性最佳 | 0.002 | 0.05 | 每 epoch | 100 | **0.6140 ± 0.0084** | **0.8925 ± 0.0026** |
| F02 固定终点 | 0.0005 | 0.01 | 每 epoch | 300 | 0.5760 | 0.8801 |
| FT01 最佳续训点 | 0.000394 | 0.05 | 每 epoch | F01-100 + 5 | 0.6026 ± 0.0063 | 0.8891 ± 0.0020 |
| FT02 最佳续训点 | 0.000197 | 0.05 | 每 epoch | F01-100 + 5 | 0.5860 ± 0.0073 | 0.8838 ± 0.0023 |
| FT03 最佳续训点 | 0.0000787 | 0.05 | 每 epoch | F01-100 + 10 | 0.5595 ± 0.0069 | 0.8753 ± 0.0023 |
| FT04 最后续训点 | 0.0000394 | 0.05 | 每 epoch | F01-100 + 20 | 0.5548 ± 0.0103 | 0.8738 ± 0.0034 |

Round 8 的 FT01–FT04 全部正常结束，没有一组超过起始权重 F01-100 的 0.6140 ARI。预先设置的改善阈值没有达到，因此 FT05、FT06 按规则跳过。论文 Table II 报告的 Breast Cancer ARI 为 0.6350；探索性 F01-100 与此相差约 0.0210，与作者权重复评相差约 0.0155。**这些差距不能直接解读为独立测试集上的差距。**

F01-100 是根据同一评价集合的标签挑选的 checkpoint；粗搜索和精细搜索的配置也参考了该集合的 ARI。后验推断的 10 个种子仅重复评价**同一个训练权重**，并非 10 个独立训练种子。作者协议包含全数据拟合 StandardScaler、样本 ID 输入、重复的客户端记录，以及与标签有关的分区或 GMM 分量设定。本仓库把这些结果标为探索性结果，不能把 0.6140 当作无偏泛化成绩。

## 日志与结果

Round 8 的完成记录为 `exit_code=0`，可在 [退出状态](experiments/round8_checkpoint_finetune/task_exit_status.json) 和 [完整报告](experiments/round8_checkpoint_finetune/ROUND8_REPORT.md) 核对。任务日志的原始末尾如下：

```text
FT01 last 0.5630529010348287 best 0.6025720060665141
FT02 last 0.5580038928228589 best 0.5860204343269355
FT03 last 0.5558043723067703 best 0.5595042367793731
FT04 last 0.5547577195152371 best 0.5547577195152371
REPORT_READY /root/autodl-tmp/fedvade_project/reproduction_runs/round8_checkpoint_finetune/ROUND8_REPORT.md
```

| 内容 | 文件 |
| --- | --- |
| Round 8 总任务日志 | [task.log](experiments/round8_checkpoint_finetune/task.log) |
| Round 8 各配置日志 | [FT01](experiments/round8_checkpoint_finetune/FT01.log) · [FT02](experiments/round8_checkpoint_finetune/FT02.log) · [FT03](experiments/round8_checkpoint_finetune/FT03.log) · [FT04](experiments/round8_checkpoint_finetune/FT04.log) |
| Round 8 逐轮记录与完整指标 | [FT01 rounds.jsonl](experiments/round8_checkpoint_finetune/FT01/rounds.jsonl) · [finetune_metrics.csv](experiments/round8_checkpoint_finetune/finetune_metrics.csv) · [inference_repeats.csv](experiments/round8_checkpoint_finetune/inference_repeats.csv) |
| Round 7 搜索方案与排名 | [SEARCH_PROTOCOL.md](experiments/round7_hyperparameter_search/SEARCH_PROTOCOL.md) · [coarse_rankings.json](experiments/round7_hyperparameter_search/coarse_rankings.json) · [fine_rankings.json](experiments/round7_hyperparameter_search/fine_rankings.json) |
| Round 7 训练日志 | [F01](experiments/round7_hyperparameter_search/fine_F01.log) · [F02](experiments/round7_hyperparameter_search/fine_F02.log) · [driver.log](experiments/round7_hyperparameter_search/driver.log) |
| 早期实验与审查 | [第一轮摘要](experiments/round1/HANDOFF_SUMMARY.md) · [第三轮统计](experiments/round3/statistics.json) · [第四轮源码审查](experiments/round4_source_audit/audit_report.md) · [第五轮 A/B 日志](experiments/round5_broadcast/formal_author_seed0.log) · [第六轮同协议复评](experiments/round6_checkpoint_diagnosis/unified_evaluation.csv) |

实验脚本、单次配置与更完整的日志保存在各 `experiments/round*/` 目录。服务器原始目录为 `/root/autodl-tmp/fedvade_project/reproduction_runs/`，其中保留未上传的 checkpoint、数据和作者材料。复现或核查这些结果时应使用作者交付的相同源码与数据，并以各轮配置和源码哈希确认版本。

## 当前判断

Round 7 的标签选择结果接近作者权重，但其第 300 轮固定终点下降明显；Round 8 的短程权重初始化续训进一步下降。这支持“训练终点与模型选择会显著影响记录的 ARI”，但**尚未确认**作者权重较高 ARI 的根本原因，也没有验证生成质量是否随聚类指标改善。下一步若要建立可用于算法改进的基线，应预先划分验证与测试集合，并以多个独立训练种子评价固定配置和固定选择规则。

## Round 9：初始化与训练稳定性搜索

本轮在服务器运行 9 组初始化/预训练配置（其中 1 组复用旧结果）和 8 组超参数配置；已完整记录的新增训练 16 次。最佳 seed=0 探索性 checkpoint：ARI 0.6013 ± 0.0058，Purity 0.8889 ± 0.0018，H03 第 80 轮。

Round 7 F01 第 100 轮参考 ARI 为 0.6140；作者权重参考约 0.6295。所有按评价标签挑选的配置或 checkpoint 均为探索性结果，10 个推断种子不是 10 个独立训练种子。完整配置、逐轮记录、原始日志和失败情况见 [Round 9 报告](experiments/round9/ROUND9_REPORT.md)、[实验计划](experiments/round9/experiment_plan.json)、[Checkpoint 排名](experiments/round9/best_checkpoint_ranking.csv) 和 [任务日志](experiments/round9/task.log)。权重、数据、作者源码和密钥仅留在服务器。
