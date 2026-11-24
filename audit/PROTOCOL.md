# Round 7 replication protocol (with full client checkpoints)

## Scope

This experiment reproduces the Round 7 FedVaDE hyperparameter search and adds
full federated-state checkpoints. It does **not** add Multi-CA and does not
change the model or loss code.

- Author source: `/root/autodl-tmp/fedvade_project/fuxian-round2/FedVaDE_source`
  (the same four Python files and Breast Cancer CSV used by Round 7).
- Training seed 0 for Python, NumPy and PyTorch CPU RNG.
- Author partition: `make_noniid_clients(..., seed=42)`, 10 clients, 2129
  records, sizes `[115, 209, 326, 56, 218, 155, 350, 239, 138, 323]`.
- 50 pretraining rounds; 150 formal rounds for coarse, 300 for fine.
- Batch 200, latent 10, 2 clusters, sample count 100, Adam, StepLR(10, 0.85).
- The author client scheduler steps per mini-batch for `schedule=author`; the
  already-established Round 7 `schedule=epoch` arm moves that step to once per
  round. This is the only source instrumentation, unchanged from Round 7.
- CPU PyTorch 2.0.1, one thread per run, matching the historical protocol.

## Grid (identical to `search_driver.configs()`)

| ID | lr | alpha | schedule |
|---|---:|---:|---|
| C01 | 0.002 | 0.01 | author |
| C02 | 0.002 | 0.01 | epoch |
| C03 | 0.0005 | 0.01 | author |
| C04 | 0.0005 | 0.01 | epoch |
| C05 | 0.001 | 0.01 | author |
| C06 | 0.001 | 0.01 | epoch |
| C07 | 0.005 | 0.01 | author |
| C08 | 0.005 | 0.01 | epoch |
| C09 | 0.002 | 0.001 | author |
| C10 | 0.002 | 0.001 | epoch |
| C11 | 0.002 | 0.005 | author |
| C12 | 0.002 | 0.005 | epoch |
| C13 | 0.002 | 0.05 | author |
| C14 | 0.002 | 0.05 | epoch |
| C15 | 0.001 | 0.005 | author |
| C16 | 0.001 | 0.005 | epoch |

Sample count stays 100. Fine selection is the top four coarse configurations
by fixed epoch-150 ARI mean over 10 inference seeds; those are run for 300
rounds. C14 is the historical F01 configuration.

## Evaluation

The original Round 6 evaluator package was not migrated to this server, so its
protocol was reimplemented in `scripts/evaluate_saved_round6.py`:

- Model built and loaded once per checkpoint, before the seed loop.
- For each inference seed 0-9: `torch.manual_seed(seed)`, then iterate the
  author's client DataLoaders batch by batch and call `model.predict(batch)`.
- ARI, Purity, NMI, ACC on the 2129 client records; ARI-std over seeds.
- `client_loss_last5_mean` from `rounds.jsonl`.
- `best_checkpoint.pt` by minimum trailing-five client loss;
  `best_exploratory_ari_checkpoint.pt` by ARI (exploratory only).

Validation: on the historical F01 `epoch_100_server.pt` this evaluator returns
the exact per-seed values in `inference.csv` and the exact mean
`0.6140076504228672`. The same evaluator is used for every new run.

## Full checkpoint requirement

At **every** 25-round server checkpoint (25, 50, ..., final) the runner saves
`epoch_XXX_full.pt` containing:

- server encoder/decoder and global GMM (`rho_`, `mu_c`, `log_sigma2_c`);
- all 10 client state dicts (encoder, decoder, local GMM);
- server and 10 client Adam optimizer state dicts;
- server and 10 client StepLR scheduler state dicts;
- Python, NumPy and PyTorch RNG states;
- the data partition tensors and sizes;
- the full training config, alpha/lr/schedule/sample/rounds, and the round.

Saving does not consume RNG, so the training trajectory is unchanged. Each
checkpoint is loaded and structurally verified before optional compression.

## Forced deviations (missing artifacts on this server)

1. The original coarse C01 reused
   `round5_broadcast/formal_author_seed0/round_150_full.pt`. That file is not
   present here (only `round_300_full.pt`), so C01 is retrained for 50+150
   rounds under the identical config and protocol.
2. The original fine reuse for the author/epoch baselines used checkpoints
   that are absent or server-only here. Selected fine configurations are
   therefore trained for 50+300 rounds to obtain full client states. The F01
   priority run is reused when C14 is selected.
3. The historical run was produced on a different machine. Code, source diff,
   config and grid are identical, but pretraining diverges at formal round 1,
   consistent with CPU/BLAS-level floating-point differences. Actual results
   are reported; the evaluation method was not changed.

## Disk

Full checkpoints are ~343 MB each. To fit the 50 GB data disk, unkept full
checkpoints are verified then gzip-compressed (`gzip -1`, ~61%) and the raw
file removed. `epoch_100_full.pt` stays raw for every fine run;
`epoch_300_full.pt` stays raw for F01. Manifests record raw/compressed sizes
and sha256 values. Checkpoints are never uploaded.
