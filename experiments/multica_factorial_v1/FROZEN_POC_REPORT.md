# Frozen-Model Proof of Concept

**Status:** completed before formal training. No neural training was performed.  
**Checkpoints:** existing full checkpoints with real client GMM states:

- Author config: `reproduction_runs/round5_broadcast/formal_author_seed0/round_300_full.pt`
- F01 config: `reproduction_runs/round7_hyperparameter_search/fine/F01/last_checkpoint.pt`

Both contain server, 10 client VAEs/GMMs, and decoders. BC has no independent
test split; this is an in-sample diagnostic.

## Protocol

Four distributions per checkpoint:

1. `server_gmm` — original K=2 server GMM.
2. `flat_gmm_K2` — K=2 diagonal GMM fitted on 10,000 union samples.
3. `plain_wasserstein_K2` — average-linkage Wasserstein agglomeration + moment
   matching to K=2.
4. `multi_ca_K2` — deterministic complete-linkage Multi-CA + moment matching to
   K=2, normalized-Bures merge order.

Same server encoder/decoder, same real reference, same 2,000 generated samples ×
5 inference seeds, same sampling RNG seeds. A random forest trained on all 569
rows is used only as an **in-sample minority-class probe**; it is not a held-out
evaluator.

## Results (unique 569 rows)

| Checkpoint | Method | NLL | ARI | NMI | Purity | Overall MMD | Overall SWD | Minority MMD | Minority SWD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Author | server GMM | 22.374 | 0.471 | 0.383 | 0.844 | 0.1387 | 0.4057 | 0.2125 | 0.5686 |
| Author | flat K=2 | 21.247 | 0.510 | 0.405 | 0.858 | 0.1186 | 0.3738 | 0.1960 | 0.5506 |
| Author | plain W2 K=2 | 21.222 | 0.510 | 0.402 | 0.858 | 0.1183 | 0.3730 | 0.1969 | 0.5521 |
| Author | Multi-CA K=2 | 21.222 | 0.510 | 0.402 | 0.858 | 0.1183 | 0.3730 | 0.1969 | 0.5521 |
| F01 | server GMM | 16.955 | 0.486 | 0.375 | 0.851 | 0.0487 | 0.2439 | 0.0865 | 0.3854 |
| F01 | flat K=2 | 16.603 | 0.476 | 0.367 | 0.847 | 0.0277 | 0.1963 | 0.0722 | 0.3541 |
| F01 | plain W2 K=2 | 16.624 | 0.476 | 0.367 | 0.847 | 0.0274 | 0.1958 | 0.0722 | 0.3558 |
| F01 | Multi-CA K=2 | 16.624 | 0.476 | 0.367 | 0.847 | 0.0274 | 0.1958 | 0.0722 | 0.3558 |

## What the PoC shows

1. **Aggregation method matters versus the original server GMM** on these frozen
   checkpoints: flat/moment-matched K=2 improves NLL and MMD/SWD for Author, and
   improves NLL/MMD/SWD for F01 (with a small ARI decrease for F01).
2. **Multi-CA and plain average-linkage Wasserstein are identical at K=2** on both
   checkpoints: same components, same weights, same NLL/ARI/generation metrics.
   The adaptive/complete-linkage/normalized-order part adds no detectable
   difference in this capacity-matched frozen setting.
3. This is consistent with the pre-registered H2: at K=2, any gain over original
   aggregation is from moment-matched aggregation, not adaptive stopping.

## Decision

Proceed to Stage A 12-run formal training because:

- the PoC shows the aggregation path genuinely changes the generated/training
  distribution (not a no-op);
- the formal matrix is needed to test end-to-end training effects;
- the negative Multi-CA-vs-plain result is preserved and will not be claimed as a
  Multi-CA advantage.

No capacity-increasing Multi-CA is used in Stage A. Stage B must provide
same-capacity Flat-GMM controls before any fine-grained-mode claim.

## Caveats

- In-sample BC-only; no independent test.
- Minority-class classifier is in-sample; minority MMD/SWD are exploratory.
- The checkpoints are author/F01 historical runs, not the new factorial runs.
- Multi-CA=plain equality is specific to these checkpoints and K=2; it is not a
  universal theorem.
