# Round 7 F01 replication with full federated checkpoints

**Status:** full 16-config coarse grid + top-4 fine search completed.
Every 25-round server checkpoint also saved the complete federated state.
No Multi-CA was added; model and loss code are unchanged.

## 1. F01 priority run (C14)

| Metric | Historical | Reproduced | Δ |
|---|---:|---:|---:|
| epoch-100 10-seed ARI | 0.6140 | 0.6053 ± 0.0045 | -0.0087 |
| epoch-300 10-seed ARI | 0.5138 | 0.5208 ± 0.0062 | 0.0071 |

Config: lr 0.002, alpha 0.05, schedule epoch, sample 100, pretrain 50, rounds 300, seed 0.

Epoch-100 full checkpoint: `/root/autodl-tmp/fedvade_project/reproduction_runs/round7_f01_full_repro/fine/F01/epoch_100_full.pt`  
Epoch-300 full checkpoint: `/root/autodl-tmp/fedvade_project/reproduction_runs/round7_f01_full_repro/fine/F01/epoch_300_full.pt`  
Both are kept raw. The 10 intermediate full checkpoints are verified and gzip-compressed, with a manifest in the run directory.

### Why the reproduced F01 is not bit-identical

The instrumented `main.py` and `source.diff` are byte-identical to the historical run, and the config matches except paths/platform. The trajectories diverge at formal round 1, i.e. after pretraining. The evaluation protocol is validated exact on the historical epoch-100 server weight (0.6140076504228672), so the difference is in the model, not the metric. It is consistent with CPU/BLAS floating-point differences on the new machine. Actual values are reported without tuning seeds or changing the evaluation.

## 2. Coarse grid (16 configs, 50 pretrain + 150 formal)

Sorted by epoch-150 10-seed ARI. Top four (marked ★) entered the fine stage.

| Rank | Config | lr | alpha | schedule | ARI@100 | ARI@150 | Purity@150 | NMI@150 | client loss | server loss | epoch-100 full |
|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | ★ C04 | 0.0005 | 0.01 | epoch | 0.5954 | 0.5897 | 0.8846 | 0.4680 | 2.7658 | 0.7720 | `/root/autodl-tmp/fedvade_project/reproduction_runs/round7_f01_full_repro/coarse/C04/epoch_100_full.pt.gz` |
| 2 | ★ C13 | 0.002 | 0.05 | author | 0.5967 | 0.5733 | 0.8797 | 0.4444 | 5.5350 | 1.9870 | `/root/autodl-tmp/fedvade_project/reproduction_runs/round7_f01_full_repro/coarse/C13/epoch_100_full.pt.gz` |
| 3 | ★ C14 | 0.002 | 0.05 | epoch | 0.6053 | 0.5683 | 0.8783 | 0.4401 | 4.3034 | 1.9600 | `/root/autodl-tmp/fedvade_project/reproduction_runs/round7_f01_full_repro/coarse/C14/epoch_100_full.pt.gz` |
| 4 | ★ C06 | 0.001 | 0.01 | epoch | 0.5688 | 0.5608 | 0.8751 | 0.4413 | 2.5346 | 0.9498 | `/root/autodl-tmp/fedvade_project/reproduction_runs/round7_f01_full_repro/coarse/C06/epoch_100_full.pt.gz` |
| 5 | C03 | 0.0005 | 0.01 | author | 0.5375 | 0.5553 | 0.8731 | 0.4444 | 3.2729 | 0.9107 | `/root/autodl-tmp/fedvade_project/reproduction_runs/round7_f01_full_repro/coarse/C03/epoch_100_full.pt.gz` |
| 6 | C05 | 0.001 | 0.01 | author | 0.5442 | 0.5469 | 0.8703 | 0.4362 | 3.3406 | 1.0481 | `/root/autodl-tmp/fedvade_project/reproduction_runs/round7_f01_full_repro/coarse/C05/epoch_100_full.pt.gz` |
| 7 | C01 | 0.002 | 0.01 | author | 0.4826 | 0.5273 | 0.8634 | 0.4306 | 3.9616 | 1.4275 | `/root/autodl-tmp/fedvade_project/reproduction_runs/round7_f01_full_repro/coarse/C01/epoch_100_full.pt.gz` |
| 8 | C02 | 0.002 | 0.01 | epoch | 0.4557 | 0.5030 | 0.8549 | 0.4180 | 2.7429 | 1.2774 | `/root/autodl-tmp/fedvade_project/reproduction_runs/round7_f01_full_repro/coarse/C02/epoch_100_full.pt.gz` |
| 9 | C16 | 0.001 | 0.005 | epoch | 0.4825 | 0.4989 | 0.8535 | 0.4067 | 2.2716 | 0.8935 | `/root/autodl-tmp/fedvade_project/reproduction_runs/round7_f01_full_repro/coarse/C16/epoch_100_full.pt.gz` |
| 10 | C12 | 0.002 | 0.005 | epoch | 0.2655 | 0.3243 | 0.7849 | 0.3163 | 2.4409 | 1.2406 | `/root/autodl-tmp/fedvade_project/reproduction_runs/round7_f01_full_repro/coarse/C12/epoch_100_full.pt.gz` |
| 11 | C15 | 0.001 | 0.005 | author | 0.3254 | 0.3217 | 0.7837 | 0.2892 | 3.0846 | 0.9925 | `/root/autodl-tmp/fedvade_project/reproduction_runs/round7_f01_full_repro/coarse/C15/epoch_100_full.pt.gz` |
| 12 | C11 | 0.002 | 0.005 | author | 0.3075 | 0.3103 | 0.7787 | 0.2942 | 3.6764 | 1.3400 | `/root/autodl-tmp/fedvade_project/reproduction_runs/round7_f01_full_repro/coarse/C11/epoch_100_full.pt.gz` |
| 13 | C08 | 0.005 | 0.01 | epoch | 0.0995 | 0.1300 | 0.6810 | 0.1330 | 4.3829 | 1.4501 | `/root/autodl-tmp/fedvade_project/reproduction_runs/round7_f01_full_repro/coarse/C08/epoch_100_full.pt.gz` |
| 14 | C10 | 0.002 | 0.001 | epoch | 0.0451 | 0.0607 | 0.6407 | 0.1712 | 2.1953 | 1.1137 | `/root/autodl-tmp/fedvade_project/reproduction_runs/round7_f01_full_repro/coarse/C10/epoch_100_full.pt.gz` |
| 15 | C07 | 0.005 | 0.01 | author | 0.1064 | 0.0441 | 0.6407 | 0.0627 | 7.0145 | 1.4706 | `/root/autodl-tmp/fedvade_project/reproduction_runs/round7_f01_full_repro/coarse/C07/epoch_100_full.pt.gz` |
| 16 | C09 | 0.002 | 0.001 | author | 0.0535 | 0.0067 | 0.6407 | 0.1211 | 3.3739 | 0.9451 | `/root/autodl-tmp/fedvade_project/reproduction_runs/round7_f01_full_repro/coarse/C09/epoch_100_full.pt.gz` |

Coarse full checkpoints are structurally verified, then gzip-compressed and moved to `/root/round7_full_checkpoints/coarse/<config>/`; a symlink stays at the run-dir path recorded above. `full_checkpoint_manifest.jsonl` in each run lists raw and compressed sizes and sha256 values.

## 3. Fine stage (top four, 50 pretrain + 300 formal)

| Rank | Config | lr | alpha | schedule | ARI@100 | ARI@150 | ARI@300 | Purity@300 | NMI@300 | client loss | server loss | epoch-100 full |
|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | C04 | 0.0005 | 0.01 | epoch | 0.5954 | 0.5897 | 0.5864 | 0.8835 | 0.4664 | 2.6680 | 0.9363 | `/root/autodl-tmp/fedvade_project/reproduction_runs/round7_f01_full_repro/fine/C04/epoch_100_full.pt` |
| 2 | C13 | 0.002 | 0.05 | author | 0.5967 | 0.5733 | 0.5774 | 0.8812 | 0.4488 | 5.6917 | 2.1529 | `/root/autodl-tmp/fedvade_project/reproduction_runs/round7_f01_full_repro/fine/C13/epoch_100_full.pt` |
| 3 | C14 (F01 priority) | 0.002 | 0.05 | epoch | 0.6053 | 0.5683 | 0.5208 | 0.8629 | 0.4005 | 4.0714 | 1.9358 | `/root/autodl-tmp/fedvade_project/reproduction_runs/round7_f01_full_repro/fine/F01/epoch_100_full.pt` |
| 4 | C06 | 0.001 | 0.01 | epoch | 0.5688 | 0.5608 | 0.5504 | 0.8716 | 0.4298 | 2.3652 | 0.9400 | `/root/autodl-tmp/fedvade_project/reproduction_runs/round7_f01_full_repro/fine/C06/epoch_100_full.pt` |

Fine epoch-100 full checkpoints are kept raw. Other full checkpoints are verified and gzip-compressed; F01 epoch-300 is kept raw.

## 4. Checkpoint verification

- Every `epoch_XXX_full.pt` is loaded before compression: server and all 10 client state dicts load strictly, optimizer/scheduler counts are 10, RNG fields and partition tensors are present, and the round matches the filename.
- F01 epoch 100: reloaded into server + 10 clients + optimizers + schedulers + RNG, then resumed for round 101. server_loss, purity and ARI matched the recorded round exactly; client_loss differed by 4.8e-8 (float32 rounding). See `fine/F01/resume_verify.json`.
- C01 epoch 125: recovered from a deterministic rerun after a compression move interrupted that file; rounds 1-97 of the rerun are bit-identical to C01, and resuming from the recovered 125 matches round 126 exactly. See `coarse/C01/resume_verify_125.json`.
- All compressed files pass `gzip -t`; manifests record sha256.

## 5. Protocol and forced deviations

Grid, seeds, partition, pretraining, formal rounds, optimizer, scheduler, sample count and evaluation follow Round 7 exactly. The evaluator reproduces the historical F01 epoch-100 per-seed values bit for bit. Two artifacts were missing on this server: `round_150_full.pt` (C01 reuse) and the original fine-reuse checkpoints; those runs were retrained under the identical configs to obtain full client states. See `audit/PROTOCOL.md`.

## 6. Artifacts

- `results/all_runs.csv`, `results/all_runs.json` — every config with metrics, hyperparameters, seed, rounds and checkpoint paths.
- `audit/PROTOCOL.md`, `audit/CHECKPOINT_VERIFICATION.md`.
- `scripts/` — runner, evaluator, compressor, resumer, collectors.
- Full checkpoints stay on the server: raw paths above; coarse archives under `/root/round7_full_checkpoints/coarse/`.
