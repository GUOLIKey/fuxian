# Review status

## Verified

- The implementation runs from the C13/FedVaDE training path while preserving
  the original client Local GMM block.
- Only the initialization and per-round server latent-sampling calls are
  replaced in private source copies.
- Responsibility-derived routing can assign a server component to multiple
  clients; it is not a dominant-client-only shortcut.
- Three server DPGMM levels fit the identical candidate pool within each call.
- Unit, integration, checkpoint evaluation and structural precheck paths passed
  in the server CPU environment.

## Failed attempts retained on the server

- `server_smoke_iris_s0_20261127`: training completed, but the wrapper failed
  after training because a relative server-config path was read after `chdir`.
- `server_smoke_iris_s0_retry1_20261127`: training completed, but standalone
  evaluation could not import a training-only top-level runtime dependency.
- `server_smoke_iris_s0_retry2_20261127`: corrected run and evaluation passed.

The fixes resolve configuration paths before training and make the model-side
runtime import lazy inside pretraining.  Failed directories were not deleted or
presented as successful results.

## Formal result review

- The three-round smoke ARIs do not estimate method performance.
- A DPGMM concentration prior does not force a unique active K on every call.
- The seed-1 method has been compared with A and B at round 300 on all three
  datasets.  Iris is lower than A, Breast Cancer is evaluation-population
  dependent, and Banknote is higher than A at seed 1.
- The Banknote improvement does not survive the baseline-selected seed-4
  sensitivity check; seed 4 is substantially below A under both populations.
- Across the five completed C runs, mean active K increases with concentration,
  but only 98 to 136 of 301 calls per run satisfy the intended nondecreasing K
  order.  The output is therefore three concentration-conditioned partitions,
  not an established ordered hierarchy.
- The implementation is not the original Multi-CA competitive-learning method.
- A complete pre-specified multi-training-seed estimate has not been run.
  Inference draws from one checkpoint are not independent training repeats, so
  statistical significance and generalization remain unestablished.
- Full tables and the exact conclusion boundary are in
  `docs/C_SERVER_DPGMM_RESULTS_20261127.md`.
