# FedVaDE 第四轮源码与协议审查

## 审查边界
新服务器 autodl-container-6jq3ca2nm5-49bab716，根目录 /root/autodl-tmp/fedvade_project。作者原始 ZIP 与 FedVaDE PDF 在该迁移范围中均不存在。迁移清单明确记录旧机源端无 PDF。原 ZIP 的内容哈希无法验证，因此“当前工作源码等于作者 ZIP 原件”标记 UNVERIFIABLE。现存 FedVaDE_source 与历史 source_hashes 匹配，迁移清单本轮核对 165/165 个相关文件的旧 SHA 成功。检查仅覆盖当前可见迁移目录。

## 第三轮实际源码
第三轮 runner.py 从 fuxian-round2/FedVaDE_source 拷贝 main.py、model.py、dataloader.py、experiment.py 与 breast_cancer.csv 到各次 formal/B_*/source。command.json 记录了 source 和 runner 路径，source_hashes 五项与当前工作源码一致。runner.py 对副本 main.py 调用 fedvade_source，再 compile/exec，sys.path 插入副本目录。原版调度组只加逐轮记录和计时，按论文调度组还移动了客户端 StepLR.step。source.diff 与本轮生成的 source_diff 保留改动。运行时导入模块的完整文件系统跟踪未保存，故 import 实际解析路径在静态证据层面标为 MANUAL_REVIEW_REQUIRED。

## 论文、源码、执行矩阵
逐项状态见 paper_code_execution_matrix.csv。论文 PDF 缺失，论文列依赖既有研究摘录和上轮记录，需要用原 PDF 复核页码及公式。已可确认的代码关系包括客户端 StepLR 位于 batch 循环内部，服务端每轮调用一次；服务器更新后向客户端广播的是服务器优化前形成的加权 VAE 参数。代码的重构项是 MSE。Table II 的 checkpoint 轮次与模型选择规则仍未知。

## 学习率
作者代码 main.py:123-128 创建两套 StepLR，step_size=10、gamma=.85；main.py:147 在客户端 batch 中调用，main.py:210 在服务端每轮调用。10 个客户端单轮 batch 数为 1、2、2、1、2、1、2、2、1、2。双 batch 客户端原版第 5 轮末降至 0.0017，修改版第 10 轮末才衰减。第三轮两组从第 6 轮出现损失差异，每对分区签名相同。5 种子第 300 轮 ARI 均值原版 0.364514、修改版 0.390347，总体标准差另见 audit_findings.json。调度差异真实存在，但不能解释全部论文差距。

## 预训练、数据和权重
main.py:120 调用 50 轮 pre_train。model.py:125-159 对编码器和解码器按本地 batch 做 MSE 训练，按客户端记录数加权聚合；model.py:176-200 使用编码均值拟合局部对角 GMM，局部分量数由真实标签种类决定，random_state=42，服务器从局部分量混合抽样后拟合全局 GMM。当前 569×32 CSV 含 A1 至 A31 和标签，A1 为样本标识符。StandardScaler 在全部 569 行拟合后划分，10 个客户端总计 2129 条记录，存在大量跨客户端重复，标签列没有进入输入张量。源 CSV 哈希及各客户端大小已记录在 data_audit.json。这里的标签辅助和 A1 输入影响无监督研究的科学性；当前本轮未更改作者协议。

作者权重和两个从头复训最终权重都能作为 PyTorch state_dict 加载。所有张量有限，均含 2 个混合分量、31 维输入及相同的模型参数键。作者权重的分量质量约 0.300/0.700，复训 seed0 为 0.722/0.278，seed1 为 0.612/0.388。参数距离仅说明状态不同。作者权重文件不含训练种子、训练轮数、预处理或模型选择元信息。详细形状、哈希及参数距离见 checkpoint_audit.json。

## 结果与根因分级
第三轮五个种子作者调度 300 轮 ARI 0.3645±0.1524，论文调度 0.3903±0.1640，总体标准差。作者权重复评历史记录约 0.6315、论文表 0.6350；这两个数字的统计协议不同。已证实的差异包括调度频率、公开代码与论文描述的参数广播与重构目标；这些差异对最终 ARI 的单项因果贡献尚未实验隔离。多个种子都低于作者权重，因此训练稳定性是 LIKELY 研究因素，但权重生成流程、Table II 轮次、训练时的软件环境和是否选择最佳模型是 UNVERIFIED。原始 ZIP 与论文 PDF 缺失，无法宣称严格复现或证明代码与作者原件一致。最终状态为 UNRESOLVED_TRAINING_GAP。

## 最小后续验证
优先取得作者原 ZIP 和论文 PDF，做只读哈希与逐行比对。如果可取得作者训练权重的生成日志或参数，先对齐初始化、数据分区和 Table II 模型选择。若仍缺失，可做同一固定种子、同一环境的两个 mini-batch 推理与学习率轨迹审查；本轮未启动正式训练。
