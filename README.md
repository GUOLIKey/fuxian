# FedVaDE corrected 2×2 baselines

A and C are the delivered models evaluated **directly**, with no training,
continuation, warm start, client reconstruction, or weight modification.
B and D (Multi-CA) are **blocked**: both payloads are server-only and the
original client states do not exist. See `audit/BLOCKER_B_D.md`.

| Group | Model | ARI 0–2 | ARI 0–9 | Status |
|---|---|---:|---:|---|
| A | Author `.pk` | 0.6284 ± 0.0024 | 0.6295 ± 0.0088 | DONE |
| C | F01 epoch 100 | 0.6058 ± 0.0082 | 0.6140 ± 0.0088 | DONE |
| B | Author + Multi-CA | — | — | BLOCKED |
| D | F01 + Multi-CA | — | — | BLOCKED |

Start with [`REPORT.md`](REPORT.md). The protocol reproduces F01's recorded
per-seed ARIs bit for bit (`audit/PROTOCOL.md`). The previous warm-start
experiment is preserved at `../multica_pretrained_2x2_v1` and is not used as a
baseline.
