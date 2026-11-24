# Verification record (2026-11-24)

Checks run after the 12 runs completed. No training was repeated for the
experiment; the only new run was a 3-round determinism check.

## Run completeness

- 12/12 runs `status.json = DONE`; 300 rounds each; `full_round_300.pt`
  present for all 12; no NaN/Inf ARI or NLL.
- `results/integrity.json`: `same_client_hashes=true`,
  `same_encoder_hashes=true`, `all_done=true`, `all_rounds_300=true`.

## Paired initial state

- For every model × seed, `init/*_base.pt` and `init/*_on.pt` have identical
  per-client state hashes and identical encoder/decoder hashes. Only the
  server GMM differs: K=2 (OFF) versus K_micro=10 (ON).
- ON macro allocation sums to 10 with at least two micros per macro; zero
  unsafe merges at initialization.

## ON/OFF pairing during training

- All 300 per-round `client_loss` values are bit-identical for all six
  OFF/ON seed pairs; `server_loss` differs in all 300 rounds.
- This is expected: the author loop broadcasts the client average
  (`server_vade_param`), not the server-trained model. The experiment
  therefore isolates **server-side** Multi-CA effects. See `REPORT.md` §9.

## Code-version check

- The OFF batch was launched before two config-only keys
  (`server_gmm_update`, `server_optimizer`) were added to the runner. A fresh
  3-round run with the final runner from `init/AUTHOR_seed0_base.pt`
  reproduced rounds 1–3 exactly:
  - client_loss: 126.91980614662171, 29.593462657928466, 29.333389949798583
  - server_loss: 9.526224136352539, 7.805105209350586, 6.040116786956787
  - ARI: -0.015690069030280696, 0.18185820508611894, 0.43147003580845883
  This matches `runs/AUTHOR_seed0_off/rounds.jsonl` bit for bit.

## Tests

- 14/14 unit tests pass with `python -m unittest discover -s tests`
  (Multi-CA moment matching, allocation, hierarchy, sampling, guard; paired
  initialization hashes and capacities).

## Source provenance

- AUTHOR: `FedVaDE_source/model/server_model_breast_cancer.pk`
  (sha256 `168ad5a4cd619ccee527a4a6c4a6816d35166e3a6bb6bc70b4bbbaa838c2ea85`),
  historical ARI 0.6395.
- F01: `round7_hyperparameter_search/fine/F01/epoch_100_server.pt`
  (sha256 `41f425203990644c6321b68c83b7b5a7343cafe50a56215c7f505775e9912573`),
  the checkpoint with historical ARI 0.6140 — not epoch 300 (ARI 0.5138).
- Both source files are server-only; client states are reconstructed as
  documented in `FEASIBILITY.md`. No baseline was retrained or replaced.
