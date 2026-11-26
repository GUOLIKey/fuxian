# C13 客户端 DPGMM 三数据集实验：正式结果

本目录独立于历史实验。新方法代码已复制到服务器隔离目录，短程联调和三数据集正式配对实验均已完成。正式结果见 `docs/FORMAL_RESULTS_20261127.md`。本目录是去除作者源码、数据和权重后的 GitHub 发布副本。

## 实验范围

Iris、Breast Cancer、Banknote各3组×seed0/1/2，共27条正式训练分支。

- A：客户端真实类别数定K，普通GMM拟合。
- B：显式配置的DPGMM拟合局部潜在表示。
- C：读取同一预训练起点的B组每客户端实际K，普通GMM重新拟合；不独立重估K。

三组沿用C13训练流程；Iris alpha=.01（2026-11-26三种子对照后确定），BC/Banknote alpha=.05；lr=.002、50轮预训练、300轮联邦训练、10客户端、batch200、latent10、sample100、作者按batch客户端调度、CPU单线程。全局K仍由数据集类别数指定。服务器原采样/训练/广播不变，无complete-linkage、Multi-CA或重加权改动。

## 优先检查的文件

1. `scripts/protocol.py`：精确定位作者源码的插入点，任何多重/缺失匹配立即失败。
2. `scripts/paired_runtime.py`：共享预训练状态、随机状态恢复、A/B/C混合模型拟合与证据输出。
3. `scripts/run_experiment.py`：独立运行入口，检查冻结源码哈希及参数；每次只运行一个准备步骤或一个组。
4. `SOURCE_MANIFEST.json`：冻结作者源码与已审核工具的校验值。
5. `verification/tests_final.log`：本地数值/协议单元检查结果。

原始实验中的 `vendor/author` 是作者源码快照，`vendor/audited_scripts` 是已审核共享工具快照。二者保持冻结，运行只修改新输出目录中的 source 副本。GitHub 发布副本按发布边界省略了这两个目录，只保留哈希清单、实验代码、日志和结果；因此不能把本目录称为脱离作者材料即可重跑的自包含包。

## 共享起点

作者pre_train内部含有GMM拟合，因此保存位置专门截在AE预训练结束、局部GMM拟合之前。保存服务器/客户端模型和Python、NumPy、Torch随机状态。各组恢复同一状态、跳过AE训练，然后按原流程初始化混合模型和新的正式训练Adam。

恢复时核对checkpoint哈希、数据/客户端张量哈希、源码、训练参数和依赖版本。C还核对B组完成状态、预训练checkpoint及每客户端latent数组哈希。预训练状态是实验分叉用途，不冒充任意轮次完整续训checkpoint。

## DPGMM参数选择前的记录（历史）

`configs/dp_example_NOT_SELECTED.json`仅展示所需字段，不是正式推荐配置。prune=0代表示例不剪枝，不能把所有正权重分量自动解释为真实簇。上限、先验、阈值等应先预检并冻结。非收敛、超出样本数、全部分量被剪掉均停止并保留失败。报告原始权重、保留K、移除质量和收敛信息。

## 命令模板

从本目录，使用具有torch/numpy/sklearn/scipy/pandas/munkres/threadpoolctl的Python。DATA应为仅含CSV的数据目录，不是包含model.py的源码根目录。

```bash
python scripts/run_experiment.py --group prepare --dataset iris --seed 0 --alpha 0.01 --data-dir "$DATA" --out "$NEW/prepared"
python scripts/run_experiment.py --group A --dataset iris --seed 0 --alpha 0.01 --data-dir "$DATA" --prepared "$NEW/prepared/pre_gmm.pt" --out "$NEW/A"
python scripts/run_experiment.py --group B --dataset iris --seed 0 --alpha 0.01 --data-dir "$DATA" --prepared "$NEW/prepared/pre_gmm.pt" --dp-config "$APPROVED_DP_CONFIG" --out "$NEW/B"
python scripts/run_experiment.py --group C --dataset iris --seed 0 --alpha 0.01 --data-dir "$DATA" --prepared "$NEW/prepared/pre_gmm.pt" --b-run "$NEW/B" --out "$NEW/C"
python vendor/audited_scripts/common/evaluate.py --run "$NEW/A" --data-dir "$DATA" --rounds 300 --population unique_full --output "$NEW/A/evaluation_unique_full.json"
```

每组分别做unique_full与legacy_clients评价，默认10推理种子。样本内指标不等于独立测试表现。正式比较按训练种子配对，报告所有结果，不挑最佳轮次。

## 运行记录的解读

`config.json`里的local_gmm=author描述复用的底层入口，真实实验组看`experiment.json`与`paired_model.diff`；后者包含新方法代码哈希，底层repair_package只涵盖冻结工具。运行后还需查result.json是否ok，不能以文件存在代替成功。

底层最终checkpoint当前仍是评价状态，不含完整优化器/调度器/RNG，不能精确恢复正式训练。若正式实验要求可恢复训练，应在用户检查后补充并单独验证，不在本轮宣称已经具备。

## 已授权服务器基线

`server_baselines/run_authorized.sh`只调用此前审核过的trainer，依次运行Iris和Banknote的C13原流程，各seed0/1/2；不调用本目录新方法。

服务器输出：`/root/autodl-tmp/fedvade_project/reproduction_runs/c13_iris_banknote_baselines_20261126_attempt2/`。
首次尝试在源码导入检查处停止，失败目录保留；第二次将数据路径设为作者dataset子目录，并只评价已保存的round300。

这六条是先行原流程基线，尚未导出新版共享预训练起点。正式B/C的配对比较，应从新版prepare状态重跑A，并核验A与先行基线是否一致；在核验前不把它们当成已完成的27组配对结果。

## 用户要求重启基线（2026-11-26）

旧队列已明确停止，Iris seed0/1完成记录保留，seed2中断记录保留。新队列位于`/root/autodl-tmp/fedvade_project/reproduction_runs/c13_paired_baselines_20261126/`。仅部署`server_baselines/paired_baseline_code/`中的prepare/A专用版本，无DPGMM拟合功能。队列先验证共享状态恢复后的A与原流程短程参数一致，门槛通过才开始六条完整基线。每条先保存pre_gmm.pt，再从该状态运行A；旧基线不混入新配对汇总。

## 正式运行前修正

入口现要求显式`--alpha`且按数据集验证：Iris=.01、BC/Banknote=.05。三组共享同一数据/种子/alpha。正式DP配置须通过表示预检固定，不能沿用示例文件作为未经检查的研究设置。

## 2026-11-26 服务器预检

服务器隔离目录 `reproduction_runs/c13_dpgmm_3datasets_v1` 中，Iris seed0 的 prepare→A/B/C 短程联调均成功，未观察到非有限参数。Breast Cancer seed0 的 50 轮 AE 预训练准备成功。以上不是正式结果。

作者划分在客户端间重复使用样本：Iris 150 条原始记录生成 532 条客户端记录，Breast Cancer 569→2129，Banknote 1372→4318。三组实验共用同一划分，因此组间可配对；结果仅适用于这个有重叠划分。Breast Cancer CSV 包含 ID 特征，沿用原基线，没有删除。

Mac 的 `FedVaDE_source/dataset` 与服务器实际输入目录中，这三个 CSV 分别逐文件 SHA-256 一致：Iris `818e8052…b179`，Breast Cancer `67db1943…9ffc`，Banknote `3fa17912…003f`。文件无缺失值。作者划分没有覆盖全部原始记录；按标准化后的“特征+标签”值去重，至少各有 1、1、35 个不同记录未进入任何客户端；Iris、Breast Cancer 的原始 CSV 还存在相同值的重复行，因此这里不用去重计数冒充精确的未覆盖行数。

候选 `configs/dp_precheck_candidate.json` 的 `upper_bound=5, prior=0.1, prune=0.01` 不适合声称“自适应选择局部 K”。在 50 轮预训练后的 seed0 潜在表示上，Breast Cancer、Iris、Banknote 的 10 个客户端均保留 5 个分量。把上限改为 10 后，Breast Cancer 保留 9–10，Iris 保留 7–10，Banknote 全部保留 10；把先验改为 0.001 基本不改变这些计数。把剪枝阈值改为 0.05 会降低部分计数，但会以事先未论证的阈值删除非小权重分量，且仍明显受上限控制。此项尚未通过研究目标预检，正式 B/C 训练暂未启动。Iris 的该次表示来自 alpha=0.05 先行准备；AE 预训练阶段不使用 alpha，但正式配对仍须重新生成 alpha=0.01 的 prepare 状态。

补充阈值预检（同一批 seed0 表示、prior=0.1）：上限5、阈值0.05时，Iris/BC/Banknote 分别有 2/6/1 个客户端的保留K低于5；平均移除权重为 0.8%/2.5%/0.4%。上限10、阈值0.05时，保留K分别为6–9/5–9/8–10，平均移除权重为8.3%/8.5%/4.5%。上限10、阈值0.1时，虽然保留K降到3–5左右，平均移除权重却分别达到27.2%/28.2%/40.6%。因此，单纯提高阈值不能同时得到低质量损失和对上限稳定的K；不把0.05或0.1当成已选定正式参数。

按用户提出的上限5/6/7、阈值0.05/0.06/0.07，在相同 seed0 表示上进一步预检。上限5、阈值0.05时，Iris/BC/Banknote 保留K分别为4–5/3–5/4–5，三数据集有8/4/9个客户端用满上限，平均删权重0.8%/2.5%/0.4%。上限6、阈值0.05时，保留K为4–6/3–5/全6，用满上限7/0/10个，平均删权重0.8%/4.1%/0%。上限7、阈值0.05时，保留K为4–7/3–6/6–7，用满上限3/0/7个，平均删权重2.2%/5.0%/1.3%，单客户端最大删权重7.4%/11.6%/4.9%。上限7、阈值0.06时，保留K为4–7/3–6/5–7，用满上限2/0/4个，平均删权重3.3%/6.1%/3.6%，最大删权重9.0%/11.6%/11.8%。阈值0.07继续增加删权重，不能据此认定更好。此预检仅用 seed0，仍有上限饱和，不把这些 K 当作真实簇数。

解释修正：不同分量上限会重新拟合整个混合模型，各配置的“剪掉权重比例”分母不是同一个拟合分布，不能仅凭上限5/6的删权重更少就说它们更好地保留了数据结构。上限5和6在Banknote分别有9/10和10/10个客户端顶到上限。此前口头建议把上限7、阈值0.05作为优先配置，证据不足；当前只把它保留为待比较候选，没有选定正式配置。

固定剪枝阈值0.05后，用相同 seed0 预训练潜在表示，对每个客户端做相同三折划分：每折仅以训练部分拟合上限5/6/7的DPGMM，剪枝并归一化后对该折未参与GMM拟合的表示计算平均对数密度。按客户端记录数加权，Iris 的上限5/6/7分数为 -3.775/-3.518/-3.686，Breast Cancer 为 -16.601/-16.635/-16.798，Banknote 为 -5.446/-4.917/-4.775（越高越好，不能跨数据集比较）。10个客户端中最优上限计数分别是 Iris 3/4/3、Breast Cancer 4/4/2、Banknote 0/2/8；所有拟合收敛。此验证只隔离了局部GMM拟合，AE预训练已见过这些样本；它不是独立外部测试，也不是最终ARI。按该预检，下一步试验的每数据集候选上限为 Iris 6、Breast Cancer 5、Banknote 7；Iris和Breast Cancer差距较小，结论不稳。该预检结束时正式训练尚未启动。

## 正式配对实验（2026-11-26 启动）

用户授权后，以固定阈值0.05、先验0.1、Iris上限6、Breast Cancer上限5、Banknote上限7完成正式实验。每数据集每seed共享自己的pre-GMM准备状态，依次运行A/B/C，最后分别评价unique_full和legacy_clients的round300（10个推理种子）。三个seed互不共用输出，彼此可并行；只在各seed内部按prepare→A→B→C执行。服务器输出目录为`reproduction_runs/c13_dpgmm_3datasets_v1/formal_cv_selected_20261126`，其`campaign.exit=complete`，9份准备状态、27条300轮训练、54份评价均已核对。服务器`server_sha256.json`记录965个文件的SHA-256；Mac的`server_formal/`目前只备份了902个非权重文件并逐个SHA-256核对，63个`.pt`权重文件（约4.1 GB）仍只有服务器原件，本次未完成Mac备份。A/B/C结果不得单凭局部K解释成真实簇数恢复。
