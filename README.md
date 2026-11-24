# F01 + Multi-CA frozen aggregation (v3)

Loads the real F01 epoch-100 full federated checkpoint and compares, without
any training or client reconstruction:

- **A**: original F01 server GMM, K=2.
- **B**: flat GMM built from the same 10 client local GMMs, K=7.
- **C**: adaptive Macro–Micro Multi-CA GMM, K=7.
- **Union**: raw union of the 14 client components (diagnostic).

## Headline result

A reproduces the historical F01 per-seed ARI exactly (0.6053 over seeds 0–9).
**B and C are the same model**: identical leaf partitions, weights, means,
log-variances, macro map and all metrics. Multi-CA does not beat a
same-capacity flat GMM here; the macro layer is a relabeling of the same
complete-linkage Wasserstein tree. B alone accounts for the difference from A,
and because A is in the server space while B/C are in the client space, the
A-vs-B/C magnitude is descriptive, not a clean capacity estimate.

## Latent-space check

All 10 client encoders/decoders are identical to each other (the author's
broadcast average) but differ from the server encoder/decoder: client-vs-server
latent mean L2 0.6175, covariance Frobenius 1.4253, RBF-MMD 0.0715. B/C are
native in the shared client space; A is native in the server space.

## Key files

- `REPORT.md` — full tables, merge tree, minority/rare diagnostics, caveats.
- `results/metrics.json`, `results/metrics.csv`
- `results/aggregation.json`, `results/merge_tree.json`
- `results/latent_compatibility.json`
- `results/per_micro.csv`
- `scripts/` — frozen aggregation code.

No weights, datasets or checkpoints are included. The F01 full checkpoint is
read from the server at the path recorded in `REPORT.md`.
