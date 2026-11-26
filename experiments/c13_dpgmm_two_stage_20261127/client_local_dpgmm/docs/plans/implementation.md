# C13 DPGMM implementation plan

> 状态说明：这是开始实现时的预授权计划。第 4 项中的“不得部署新方法到服务器”后来被用户对正式服务器实验的明确授权取代。已执行协议和结果以 `../FORMAL_RESULTS_20261127.md` 为准；本文保留用于说明设计演变，不能当作最终运行状态。

Goal: isolate client mixture initialization while preserving C13 server sampling and training.
Architecture: immutable author/audited snapshots; new local-only runner captures shared state immediately before local mixture fitting, branches A/B/C, and records provenance. Existing audited trainer alone runs six authorized server baselines.

1. Freeze source and C13 parameters; launch independent Iris/Banknote baseline runs with seeds 0/1/2 on existing CPU environment.
2. Write failing tests for exact-once source hooks, protocol restrictions, initialization checkpoint boundary and DP/sameK behavior.
3. Implement preparation and branch entrypoints with exclusive outputs and explicit DPGMM configuration; no silently selected research settings.
4. Verify local unit tests, static source differences and small local smoke if dependencies work. Do not deploy or run new method on server.
5. Collect authorized baseline outputs and evaluation results; report review files and limitations.

Pretraining checkpoint is before log-sigma initialization/local GMM and includes model states and Python/NumPy/Torch RNG. Branch training starts with fresh Adam as in author main. Full resume is a separate interface and is not claimed.
