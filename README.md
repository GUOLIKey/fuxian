# Round 7 F01 replication with full federated checkpoints

This reproduces the Round 7 FedVaDE hyperparameter search and stores the
**complete federated state** at every 25-round server checkpoint. No Multi-CA
is added; the model and loss code are unchanged.

## F01 priority (C14)

| Checkpoint | Historical 10-seed ARI | Reproduced |
|---|---:|---:|
| epoch 100 | 0.6140 | **0.6053 ± 0.0045** |
| epoch 300 | 0.5138 | **0.5208 ± 0.0062** |

The instrumented source and config are identical to the historical run. The
trajectories diverge after pretraining, consistent with CPU/BLAS floating-point
differences on the new server; the evaluation protocol is validated exact on
the historical epoch-100 server weight. Actual values are reported without
tuning seeds or changing the metric.

## Search highlights

Coarse top four by epoch-150 10-seed ARI: **C04 0.5897, C13 0.5733,
C14/F01 0.5683, C06 0.5608**. Fine runs (300 formal rounds):

| Config | lr | alpha | schedule | ARI@100 | ARI@300 |
|---|---:|---:|---|---:|---:|
| C04 | 0.0005 | 0.01 | epoch | 0.5954 | **0.5864** |
| C13 | 0.002 | 0.05 | author | 0.5967 | 0.5774 |
| C14 / F01 | 0.002 | 0.05 | epoch | **0.6053** | 0.5208 |
| C06 | 0.001 | 0.01 | epoch | 0.5688 | 0.5504 |

C04 is the best reproduced 300-round model; F01 has the highest epoch-100 ARI.
All four have complete client states at every 25-round checkpoint. Full tables
with Purity, NMI, losses and checkpoint paths are in [`REPORT.md`](REPORT.md)
and `results/all_runs.csv`.

## Full checkpoints

- F01 epoch 100 (raw):
  `/root/autodl-tmp/fedvade_project/reproduction_runs/round7_f01_full_repro/fine/F01/epoch_100_full.pt`
  — sha256 `3be341c451a186d1f36673080164af208993d215f0c2cb03709407892f7d6f3e`.
- Each full checkpoint contains the server encoder/decoder/GMM, all 10 client
  encoder/decoder/local-GMM state dicts, all optimizer and scheduler states,
  all RNG states, the data partition, the config and the round.
- F01 epoch 100 was reloaded and resumed for round 101: server loss, purity and
  ARI matched exactly; client loss differed by 4.8e-8 (float32 rounding).
  See `audit/CHECKPOINT_VERIFICATION.md` and `fine/F01/resume_verify.json`.
- Unkept full checkpoints are verified then gzip-compressed. Coarse archives
  live under `/root/round7_full_checkpoints/coarse/<config>/` with symlinks at
  the original run-dir paths; manifests record sizes and sha256 values.

## Search

Sixteen coarse configurations (50 pretrain + 150 formal) and the top four fine
configurations (50 + 300) follow the Round 7 grid, seeds, partition, optimizer,
scheduler and evaluation protocol. See [`REPORT.md`](REPORT.md) for every
config's ARI/NMI/Purity/Loss and checkpoint paths, and
[`audit/PROTOCOL.md`](audit/PROTOCOL.md) for the protocol and the two forced
deviations caused by artifacts missing on this server.

Monitor: `/Users/cheeseowl/Desktop/F01_复现实验_实时监控.command` (5-second
refresh). Checkpoints and datasets are not in this repository.
