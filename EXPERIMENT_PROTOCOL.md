# FedVaDE + Multi-CA 2×2 Warm-Start Protocol

## Scope

Four groups × three seeds = 12 runs, 300 formal rounds each:

| Group | Initial server | Multi-CA |
|---|---|---|
| A | Author-delivered Breast Cancer server weight | OFF |
| B | Same author snapshot | ON |
| C | F01 epoch-100 server weight | OFF |
| D | Same F01 snapshot | ON |

No pretraining is run. No zero-start baseline is substituted. Training is on
the new server (RTX 4090 D / 256 vCPU), CPU PyTorch 2.0.1; Mac is monitor/SSH only.

## Mandatory source limitation

The author `server_model_breast_cancer.pk` and F01 `epoch_100_server.pt` are
**server-only**. They contain no client model states, local GMMs, optimizers,
schedulers, or RNG states. Exact continuation of the historical federated
trajectories is impossible.

The experiment therefore uses a deterministic warm start:

1. Load the delivered server encoder/decoder/GMM.
2. Copy the server encoder/decoder into every client.
3. Set each client's local K to the number of unique local labels (1 or 2, as
   in the author pretrain).
4. Initialize each local GMM by selecting the top local-K components of the
   *same* server GMM (by responsibility mass), with no local GMM fitting.
5. Save identical client states and encoder/decoder hashes for OFF and ON.

This is a **new warm-start experiment**, not a recovery of the historical
author/F01 client trajectories. Historical ARI values are references only.

## Multi-CA ON

`K_micro=10` is selected label-free by BIC on the initial real latents
(`configs/bic_*.json`); `K_macro=2`. Every round, Multi-CA:

- flattens the current client local GMMs with client-size weights;
- groups components by complete-linkage diagonal-Gaussian 2-Wasserstein;
- allocates ten micros across the two macros (minimum two per macro);
- merges within each macro with weighted moment matching and the normalized
  Bures guard;
- samples the `K_micro` global mixture for server training;
- replaces the server GMM with this aggregation.

The ON server optimizer trains encoder/decoder only; the GMM is supplied by
Multi-CA each round, so it cannot collapse to a K=2 gradient solution. OFF is
the author loop with `gmm_sampling` and K=2.

## Selection and pairing

- Partition: author `make_noniid_clients(..., seed=42)`, 10 clients.
- Client local K: 1 or 2 by local labels; 14 total components.
- OFF/ON share client states, encoder/decoder hashes, data-loader seeds, and
  RNG state. All 300 per-round client losses are verified identical for every
  OFF/ON seed pair (the author loop broadcasts the client average, not the
  server-trained model).
- Training seed 0/1/2 changes data-loader/RNG seed; it is paired across OFF/ON.
- A/B alpha=0.01, author per-minibatch client scheduler.
- C/D alpha=0.05, F01 per-epoch client scheduler.
- Adam lr=0.002, StepLR(10, 0.85), batch=200, 10 clients, sample M=100.
- 300 rounds; server checkpoints every 25; full client/optimizer checkpoints
  at 100 and 300.

## Evaluation

- Author protocol on 2129 client records, 10 inference seeds: ARI, Purity,
  NMI, ACC.
- Latent GMM NLL on 569 unique rows.
- Generation: 5000 samples from the final server GMM, server decoder;
  MMD/SWD against 569 real standardized rows, 5 generation seeds.
- Per-micro support, ghost components, low-mass/bottom-quartile micros,
  per-micro MMD/SWD.
- BC has no independent test split; all metrics are in-sample.
- Labels are used for evaluation and for local K setting (as in author
  pretraining), never for Multi-CA merging or capacity selection.

## Integrity

`results/integrity.json` checks 12/12 DONE, 300 rounds, final checkpoint
presence, finite metrics, paired client/encoder hashes, and ON K_micro>2.
