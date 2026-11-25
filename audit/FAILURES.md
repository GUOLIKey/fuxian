# Failure and debugging record

## Smoke-test iterations (no formal result affected)

1. **Generation/NLL `.detach()` error.** Reading `server.rho_`/`mu_c`/
   `log_sigma2_c` while they still required grad raised
   `Can't call numpy() on Tensor that requires grad`. Fixed by detaching before
   converting to NumPy in `diag_log_prob`, `sample_gmm`, and generation.
2. **Coverage index mapping error.** The first coverage implementation compared
   a length-10 `component_client` vector with a length-100 sample label vector.
   Fixed by mapping `y_hat` through the component-to-client vector first.

Both errors occurred only during smoke-test development, before any formal run;
the final smoke test passed and the formal 15/15 runs completed with 0 failures.

## Formal runs

- 15/15 runs reached round 10; all records finite.
- D/E local K matched in 30/30 `(client, seed)` units.
- A4/FedVaDE interface checks (full and holdout) were already validated in
  `acd_f01_frozen_v1` (30/30 + 30/30); this experiment reuses those parameters.
- Initial reload check: exact 0.605326980742813.
- Round-101 full-flow recovery: exact at 1e-6 (server loss, purity, ARI delta 0).
- No formal-run failures, no NaN/Inf, no incomplete final states.

## Known warnings

`gmm_sampling` emits a PyTorch warning when converting the client weights tensor
to `torch.tensor`; it is from the author code and does not affect the values.
