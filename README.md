# FedVaDE + Multi-CA 2×2 Warm-Start Experiment

12 runs = 4 groups × 3 seeds, 300 formal rounds each. No pretraining and no
zero-start baseline. The experiment loads the author and F01 server weights,
uses a deterministic client warm start, and compares original FedVaDE
aggregation with a true Macro–Micro Multi-CA server GMM (`K_macro=2`,
`K_micro=10`).

**Start with [`REPORT.md`](REPORT.md)** and
[`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md). The mandatory limitation is
in [`audit/FEASIBILITY.md`](audit/FEASIBILITY.md): both source checkpoints are
server-only, so exact historical client-state continuation is impossible.

Key results (mean ± SD, 3 seeds):

| Group | Final ARI | NLL | Gen MMD | K_micro | Supported micros (≥20) |
|---|---:|---:|---:|---:|---:|
| A: author, original | 0.446 ± 0.046 | 17.284 | 0.190 | 2 | 2.0 |
| B: author, Multi-CA | 0.369 ± 0.106 | 17.136 | 0.171 | 10 | 3.7 |
| C: F01, original | 0.399 ± 0.003 | 16.115 | 0.149 | 2 | 2.0 |
| D: F01, Multi-CA | 0.430 ± 0.006 | 15.826 | 0.125 | 10 | 4.3 |

Multi-CA ON improves NLL/MMD/SWD in both models, but the ARI effect is
model-dependent (B−A negative; D−C positive). Micro retention is partial:
roughly half of the ON mass sits on <20-support micros. The warm start drops
all groups below their initial ARI before partial recovery.

Artifacts: `results/`, `figures/`, `configs/`, `scripts/`, `tests/`,
`monitoring/`. Data, weights, and checkpoints are not in this repository.
