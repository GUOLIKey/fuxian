# F01 epoch-100 full checkpoint verification

## Artifact

```
/root/autodl-tmp/fedvade_project/reproduction_runs/round7_f01_full_repro/fine/F01/epoch_100_full.pt
size    343,372,804 bytes
sha256  3be341c451a186d1f36673080164af208993d215f0c2cb03709407892f7d6f3e
round   100
```

## Contents (loaded from disk)

- server: 2-component GMM (`rho_` 2, `mu_c` 2x10, `log_sigma2_c` 2x10) and the
  encoder/decoder state dict; loads strictly into `VaDE`.
- clients: 10 state dicts, local K `[1,1,2,1,2,1,2,1,1,2]`; each loads strictly.
- `server_optimizer` and 10 `client_optimizers` (Adam state and param groups).
- `server_scheduler` and 10 `client_schedulers` (StepLR state).
- `python_rng`, `numpy_rng`, `torch_rng`.
- `partition`: 10 tensors (x and y), records
  `[115,209,326,56,218,155,350,239,138,323]`.
- `config` (lr 0.002, alpha 0.05, schedule epoch, sample 100, rounds 300,
  pretrain 50, seed 0, partition/GMM seed 42), plus `losses` for round 100
  (`client_loss 4.860340428352356`, `server_loss 1.9251958131790161`).

## Reload and exact one-round resume

`scripts/resume_verify.py` rebuilds the server, all 10 clients, both optimizer
families, both scheduler families, all 10 DataLoaders and the three RNG states
from the checkpoint, then runs round 101 with the author math and the epoch
scheduler placement. It compares the result with `rounds.jsonl` row 101:

| Quantity | Resumed | Original | Delta |
|---|---:|---:|---:|
| client_loss | 4.866437602043152 | 4.866437554359436 | 4.77e-08 |
| server_loss | 2.112144947052002 | 2.112144947052002 | 0.0 |
| purity | 0.8830436824800376 | 0.8830436824800376 | 0.0 |
| ARI | 0.5831924944280963 | 0.5831924944280963 | 0.0 |

`exact_1e-6 = true`. The 4.8e-8 client-loss difference is float32 summation
rounding; server loss, purity and ARI are bit-identical. The saved state is
therefore genuinely resumable, not just a server state dict.

## Note on the historical model

This reproduced F01 is **not** bit-identical to the historical one:

| Checkpoint | Historical 10-seed ARI | Reproduced |
|---|---:|---:|
| epoch 100 | 0.6140 | 0.6053 ± 0.0045 |
| epoch 300 | 0.5138 | 0.5208 ± 0.0062 |

The instrumented `main.py` and `source.diff` are byte-identical to the
historical run, and the config matches except for paths/platform. Divergence
appears at formal round 1, i.e. after pretraining, which points to CPU/BLAS
floating-point differences on the new machine, not a protocol change. The
evaluation protocol is validated exact on the historical epoch-100 server
weight (0.6140076504228672), so the difference is in the model, not the metric.
