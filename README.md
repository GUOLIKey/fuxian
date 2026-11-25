# FedVaDE + ACD global performance verification

Five frozen-client local-GMM interventions (A–E), each starting from the same
F01 @100 full checkpoint, with 10 server-update rounds and 3 experiment seeds.
Primary metric: **Global ARI at round 10** (pre-determined). No 300-round
training, no client reconstruction, no checkpoint modification.

## Headline

| Group | Local K | Global ARI @10 |
|---|---:|---:|
| Frozen reference | — | **0.6053** |
| A original local GMM | 1.4 | 0.5572 ± 0.0114 |
| B original K + GMM refit | 1.4 | 0.5282 ± 0.0156 |
| C BIC + GMM | 6.7 | 0.5451 ± 0.0062 |
| D ACD + moments | 91.2 | 0.5580 ± 0.0112 |
| E same-K GMM | 91.2 | 0.5596 ± 0.0289 |

All groups fall below the frozen 0.6053. D−A is **+0.0008 ± 0.0184** with mixed
signs; D−E is **−0.0016 ± 0.0328**. D's NLL/MMD/SWD are better than A/B/C, but
the same-K E is nearly as good, so those density/generation gains cannot be
attributed to ACD. This is a negative/neutral global result.

A is a matched 10-round server-update control, not the frozen reference.
`sample=100` is the total sampling budget; D samples only ~9.3 of its 91.2
local components per client, and ~86% of the sampled unique components are
support<5. See [`REPORT.md`](REPORT.md) and the audit files.

No model weights, full checkpoints, raw datasets or keys are uploaded.
