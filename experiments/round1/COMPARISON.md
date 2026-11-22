# 四种方法复现结果与论文对比

2026-11-20 10:39:28（北京时间） · 运行快照。保留失败记录，不从中途结果挑选最佳值。

样本数为有效/已结束/计划；达到计划数才公布该组均值。标准差为总体标准差；续跑拼接的 FedHCA 结果用日志中四舍五入后的单次结果计算。ACC 差值用百分点，其他用原始分数差。

## 对比分析

- **FedHCA Purity：**已出结果 10/10 组，其中 8 组低于论文；组间平均差 -0.0629，最大绝对差出现在 EC（-0.1257）。
- **FedHCA ARI：**已出结果 10/10 组，其中 9 组低于论文；组间平均差 -0.0802，最大绝对差出现在 EC（-0.2838）。
- **Fed-HIRE Purity：**已出结果 4/4 组，其中 3 组低于论文；组间平均差 -0.0637，最大绝对差出现在 iris（-0.2294）。
- **FedVaDE Purity：**已出结果 3/10 组，其中 2 组低于论文；组间平均差 -0.0500，最大绝对差出现在 iris（-0.1535）。
- **FedVaDE ARI：**已出结果 3/10 组，其中 2 组低于论文；组间平均差 -0.0314，最大绝对差出现在 banknote_authentication（+0.3527）。
- **VaDE Reuters-10K：**相同种子及作者初始化下，发布衰减配置比论文衰减配置高 1.74 个百分点。每个配置只跑 1 次，不能据此判断哪种配置平均更好。
- Fed-HIRE 只有 Iris 使用作者公开分区，其余是重建分区。交接包固定种子的 Iris 对照中，学习率 0.0001/0.05 的 Purity 分别为 0.5767/0.7633，说明学习率足以影响结果，但不能反推论文实际使用了 0.05。
- FedVaDE 使用作者提供的客户端划分和每个 batch 步进的调度器；论文终表的训练终点未确认。第 150 轮仅作为训练曲线读数，不作论文终表差值；乳腺癌作者权重复评不等同于 10 次独立训练。
- VaDE Reuters 发布脚本使用 K-means 初始化，而论文描述 GMM。MNIST 检查点与重训分开。若设定 120 轮目标，阶段分数对照论文图 3 的 90.76%，不等同于 3,000 轮终值；暂停不生成检查点。本批 VaDE 只覆盖 MNIST、Reuters-10K，不宣称覆盖论文全部数据集。
- 各论文的数据、分区和评估对象不同，这不是统一协议下的四方法排名。参数和代码版本见 [复现记录](REPORT.md)。

## 结果表

| 方法 | 数据 | 模式 / 轮次 | 指标 | 有效/结束/计划 | 实测 | 论文 | 实测－论文 |
|---|---|---|---|---:|---:|---:|---:|
| FedHCA | EC | 原入口 / one-shot | Purity | 10/10/10 | 0.6543 ± 0.0397 | 0.7800 | -0.1257 |
| FedHCA | EC | 原入口 / one-shot | ARI | 10/10/10 | 0.4042 ± 0.1021 | 0.6880 | -0.2838 |
| FedHCA | US | 原入口 / one-shot | Purity | 10/10/10 | 0.4784 ± 0.0358 | 0.5290 | -0.0506 |
| FedHCA | US | 原入口 / one-shot | ARI | 10/10/10 | 0.1460 ± 0.0602 | 0.2270 | -0.0810 |
| FedHCA | VE | 原入口 / one-shot | Purity | 10/10/10 | 0.3585 ± 0.0130 | 0.4060 | -0.0475 |
| FedHCA | VE | 原入口 / one-shot | ARI | 10/10/10 | 0.0611 ± 0.0101 | 0.0880 | -0.0269 |
| FedHCA | EP | 原入口 / one-shot | Purity | 6/10/10 | 0.3431 ± 0.0027 | 0.3240 | +0.0191 |
| FedHCA | EP | 原入口 / one-shot | ARI | 6/10/10 | 0.0116 ± 0.0070 | 0.0170 | -0.0054 |
| FedHCA | YE | 原入口 / one-shot | Purity | 10/10/10 | 0.4620 ± 0.0445 | 0.5270 | -0.0650 |
| FedHCA | YE | 原入口 / one-shot | ARI | 10/10/10 | 0.1764 ± 0.0362 | 0.2450 | -0.0686 |
| FedHCA | CA | 原入口 / one-shot | Purity | 9/10/10 | 0.5384 ± 0.0229 | 0.6000 | -0.0616 |
| FedHCA | CA | 原入口 / one-shot | ARI | 9/10/10 | 0.2120 ± 0.0304 | 0.2800 | -0.0680 |
| FedHCA | LA | 原入口 / one-shot | Purity | 10/10/10 | 0.6372 ± 0.0784 | 0.7520 | -0.1148 |
| FedHCA | LA | 原入口 / one-shot | ARI | 10/10/10 | 0.4515 ± 0.0935 | 0.5390 | -0.0875 |
| FedHCA | WI | 原入口 / one-shot | Purity | 10/10/10 | 0.5170 ± 0.0191 | 0.5140 | +0.0030 |
| FedHCA | WI | 原入口 / one-shot | ARI | 10/10/10 | 0.0621 ± 0.0319 | 0.0480 | +0.0141 |
| FedHCA | PE | 原入口 / one-shot | Purity | 10/10/10 | 0.5753 ± 0.0755 | 0.6620 | -0.0867 |
| FedHCA | PE | 原入口 / one-shot | ARI | 10/10/10 | 0.4202 ± 0.0724 | 0.5160 | -0.0958 |
| FedHCA | LE | 原入口 / one-shot | Purity | 8/10/10 | 0.2471 ± 0.0202 | 0.3460 | -0.0989 |
| FedHCA | LE | 原入口 / one-shot | ARI | 8/10/10 | 0.0749 ± 0.0179 | 0.1740 | -0.0991 |
| Fed-HIRE | iris | 原入口·论文学习率 / one-shot | Purity | 10/10/10 | 0.5573 ± 0.1030 | 0.7867 | -0.2294 |
| Fed-HIRE | iris | 原入口·论文学习率 / one-shot | ARI | 10/10/10 | 0.1769 ± 0.1610 | — | — |
| Fed-HIRE | iris | 原入口·论文学习率 / one-shot | NMI | 10/10/10 | 0.3093 ± 0.1520 | — | — |
| Fed-HIRE | iris | 原入口·论文学习率 / one-shot | ACC | 10/10/10 | 54.67% ± 9.60% | — | — |
| Fed-HIRE | accent | 原入口·论文学习率 / one-shot | Purity | 10/10/10 | 0.5255 ± 0.0140 | 0.5805 | -0.0550 |
| Fed-HIRE | accent | 原入口·论文学习率 / one-shot | ARI | 10/10/10 | 0.0592 ± 0.0470 | — | — |
| Fed-HIRE | accent | 原入口·论文学习率 / one-shot | NMI | 10/10/10 | 0.1924 ± 0.0500 | — | — |
| Fed-HIRE | accent | 原入口·论文学习率 / one-shot | ACC | 10/10/10 | 36.69% ± 3.60% | — | — |
| Fed-HIRE | yeast | 原入口·论文学习率 / one-shot | Purity | 10/10/10 | 0.4958 ± 0.0250 | 0.4650 | +0.0308 |
| Fed-HIRE | yeast | 原入口·论文学习率 / one-shot | ARI | 10/10/10 | 0.1083 ± 0.0280 | — | — |
| Fed-HIRE | yeast | 原入口·论文学习率 / one-shot | NMI | 10/10/10 | 0.2289 ± 0.0270 | — | — |
| Fed-HIRE | yeast | 原入口·论文学习率 / one-shot | ACC | 10/10/10 | 32.07% ± 4.30% | — | — |
| Fed-HIRE | wine | 原入口·论文学习率 / one-shot | Purity | 10/10/10 | 0.4723 ± 0.0060 | 0.4734 | -0.0011 |
| Fed-HIRE | wine | 原入口·论文学习率 / one-shot | ARI | 10/10/10 | 0.0338 ± 0.0030 | — | — |
| Fed-HIRE | wine | 原入口·论文学习率 / one-shot | NMI | 10/10/10 | 0.0694 ± 0.0040 | — | — |
| Fed-HIRE | wine | 原入口·论文学习率 / one-shot | ACC | 10/10/10 | 27.02% ± 2.60% | — | — |
| FedVaDE | iris | 从头训练 / 50 + 300 轮 | Purity | 10/10/10 | 0.7305 ± 0.0889 | 0.8840 | -0.1535 |
| FedVaDE | iris | 从头训练 / 50 + 300 轮 | ARI | 10/10/10 | 0.5673 ± 0.1089 | 0.7190 | -0.1517 |
| FedVaDE | iris | 从头训练 / 50 + 150 轮读数 | Purity | 10/10/10 | 0.7276 ± 0.0865 | — | — |
| FedVaDE | iris | 从头训练 / 50 + 150 轮读数 | ARI | 10/10/10 | 0.5603 ± 0.1099 | — | — |
| FedVaDE | parkinson | 从头训练 / 50 + 300 轮 | Purity | 2/2/10 | — | 0.7990 | — |
| FedVaDE | parkinson | 从头训练 / 50 + 300 轮 | ARI | 2/2/10 | — | 0.2730 | — |
| FedVaDE | parkinson | 从头训练 / 50 + 150 轮读数 | Purity | 2/2/10 | — | — | — |
| FedVaDE | parkinson | 从头训练 / 50 + 150 轮读数 | ARI | 2/2/10 | — | — | — |
| FedVaDE | glass | 从头训练 / 50 + 300 轮 | Purity | 0/0/10 | — | 0.5380 | — |
| FedVaDE | glass | 从头训练 / 50 + 300 轮 | ARI | 0/0/10 | — | 0.1910 | — |
| FedVaDE | glass | 从头训练 / 50 + 150 轮读数 | Purity | 0/0/10 | — | — | — |
| FedVaDE | glass | 从头训练 / 50 + 150 轮读数 | ARI | 0/0/10 | — | — | — |
| FedVaDE | breast_cancer | 从头训练 / 50 + 300 轮 | Purity | 10/10/10 | 0.7818 ± 0.0792 | 0.8990 | -0.1172 |
| FedVaDE | breast_cancer | 从头训练 / 50 + 300 轮 | ARI | 10/10/10 | 0.3399 ± 0.1671 | 0.6350 | -0.2951 |
| FedVaDE | breast_cancer | 从头训练 / 50 + 150 轮读数 | Purity | 10/10/10 | 0.7872 ± 0.0832 | — | — |
| FedVaDE | breast_cancer | 从头训练 / 50 + 150 轮读数 | ARI | 10/10/10 | 0.3542 ± 0.1752 | — | — |
| FedVaDE | audit | 从头训练 / 50 + 300 轮 | Purity | 0/0/10 | — | 0.8350 | — |
| FedVaDE | audit | 从头训练 / 50 + 300 轮 | ARI | 0/0/10 | — | 0.4480 | — |
| FedVaDE | audit | 从头训练 / 50 + 150 轮读数 | Purity | 0/0/10 | — | — | — |
| FedVaDE | audit | 从头训练 / 50 + 150 轮读数 | ARI | 0/0/10 | — | — | — |
| FedVaDE | banknote_authentication | 从头训练 / 50 + 300 轮 | Purity | 10/10/10 | 0.8647 ± 0.0589 | 0.7440 | +0.1207 |
| FedVaDE | banknote_authentication | 从头训练 / 50 + 300 轮 | ARI | 10/10/10 | 0.5237 ± 0.1828 | 0.1710 | +0.3527 |
| FedVaDE | banknote_authentication | 从头训练 / 50 + 150 轮读数 | Purity | 10/10/10 | 0.8631 ± 0.0591 | — | — |
| FedVaDE | banknote_authentication | 从头训练 / 50 + 150 轮读数 | ARI | 10/10/10 | 0.5208 ± 0.1815 | — | — |
| FedVaDE | yeast | 从头训练 / 50 + 300 轮 | Purity | 0/0/10 | — | 0.4820 | — |
| FedVaDE | yeast | 从头训练 / 50 + 300 轮 | ARI | 0/0/10 | — | 0.1900 | — |
| FedVaDE | yeast | 从头训练 / 50 + 150 轮读数 | Purity | 0/0/10 | — | — | — |
| FedVaDE | yeast | 从头训练 / 50 + 150 轮读数 | ARI | 0/0/10 | — | — | — |
| FedVaDE | iranian_churn | 从头训练 / 50 + 300 轮 | Purity | 0/0/10 | — | 0.9230 | — |
| FedVaDE | iranian_churn | 从头训练 / 50 + 300 轮 | ARI | 0/0/10 | — | 0.1220 | — |
| FedVaDE | iranian_churn | 从头训练 / 50 + 150 轮读数 | Purity | 0/0/10 | — | — | — |
| FedVaDE | iranian_churn | 从头训练 / 50 + 150 轮读数 | ARI | 0/0/10 | — | — | — |
| FedVaDE | wine_quality | 从头训练 / 50 + 300 轮 | Purity | 0/0/10 | — | 0.4470 | — |
| FedVaDE | wine_quality | 从头训练 / 50 + 300 轮 | ARI | 0/0/10 | — | 0.0360 | — |
| FedVaDE | wine_quality | 从头训练 / 50 + 150 轮读数 | Purity | 0/0/10 | — | — | — |
| FedVaDE | wine_quality | 从头训练 / 50 + 150 轮读数 | ARI | 0/0/10 | — | — | — |
| FedVaDE | phishing_websites | 从头训练 / 50 + 300 轮 | Purity | 0/0/10 | — | 0.6590 | — |
| FedVaDE | phishing_websites | 从头训练 / 50 + 300 轮 | ARI | 0/0/10 | — | 0.0990 | — |
| FedVaDE | phishing_websites | 从头训练 / 50 + 150 轮读数 | Purity | 0/0/10 | — | — | — |
| FedVaDE | phishing_websites | 从头训练 / 50 + 150 轮读数 | ARI | 0/0/10 | — | — | — |
| FedVaDE | breast_cancer | 权重复评 / 0 轮 | Purity | 10/10/10 | 0.8979 ± 0.0026 | 0.8990 | -0.0011 |
| FedVaDE | breast_cancer | 权重复评 / 0 轮 | ARI | 10/10/10 | 0.6315 ± 0.0083 | 0.6350 | -0.0035 |
| VaDE | MNIST | 权重复评 / 0 epoch | ACC | 1/1/1 | 94.46% | 94.46% | +0.00 pp |
| VaDE | MNIST | 从头训练 / 21/3000 epoch | ACC | 0/0/1 | — | 94.46% | — |
| VaDE | Reuters-10K | 论文学习率 / 15 epoch | ACC | 1/1/1 | 79.77% | 79.83% | -0.06 pp |
| VaDE | Reuters-10K | 发布学习率 / 15 epoch | ACC | 1/1/1 | 81.51% | 79.83% | +1.68 pp |
