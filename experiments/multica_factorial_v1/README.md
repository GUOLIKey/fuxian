# Multi-CA Factorial v1 — Stage A (Breast Cancer)

**Status:** Stage A 12/12 runs COMPLETED; final analysis done. Multi-CA v1 shows no measurable benefit at matched K=2.  
**Capacity-matched design:** Multi-CA target K=2, same as the author global GMM.  
**No ACD, no client distillation, no new broadcast, no CTG/Stage B/C in this run.**

## Current scope

Stage A only:

| Group | Training | Aggregation |
|---|---|---|
| A | Author config | original FedVaDE |
| B | Author config | Multi-CA |
| C | F01 config | original FedVaDE |
| D | F01 config | Multi-CA |

- 4 groups × seeds 0/1/2 = 12 runs.
- Shared 50-epoch pretrain per seed for A/B/C/D.
- 300 formal rounds; checkpoints every 25; full checkpoints at 100/200/300.
- Main result is round 300; no best-checkpoint selection.
- BC has **no independent test split**; reported metrics are author-protocol
  in-sample metrics or explicitly labeled exploratory proxies.

## Documents

- `METHOD_DEFINITION.md` — Multi-CA v1 math, pseudocode, complexity.
- `ORIGINAL_VS_MULTICA.md` — exact integration with the author code.
- `EXPERIMENT_PROTOCOL.md` — pre-registered 2×2 protocol and metrics.
- `FROZEN_POC_REPORT.md` — frozen-checkpoint PoC and negative result.
- `experiment_manifest.json` — live status of all 12 runs.
- `FINAL_REPORT.md` — final Stage A report and honest negative result.
- `results/` — summary CSVs after analysis.
- `figures/` — training-curve figures after analysis.

## Code

- `scripts/multica_core.py` — W2, moment matching, agglomerative merge, sampling.
- `scripts/pretrain_shared.py` — shared author-protocol pretrain.
- `scripts/run_factorial.py` — one A/B/C/D run.
- `scripts/run_stage_a.sh` — sequential 12-run launcher.
- `scripts/make_manifest.py` — read-only manifest builder.
- `scripts/frozen_poc.py` — frozen-checkpoint comparison.
- `scripts/analysis_utils.py` — metrics.
- `tests/` — core unit tests.

## Monitoring

Server-side read-only:

`experiments/multica_factorial_v1/monitoring/remote_status.sh`

Mac auto-refresh (default 5 seconds):

`/Users/cheeseowl/Desktop/MultiCA_实时监控.command`

(or the copy at `experiments/multica_factorial_v1/monitoring/check_multica.command`)

Run manually:

```bash
/Users/cheeseowl/Desktop/MultiCA_实时监控.command
```

It SSHes, prints status, clears the screen, and refreshes every 5 seconds until Ctrl-C. Use `check_multica.command --once` for a single query. It never starts/stops training.

## Critical interpretation rule

The frozen PoC showed Multi-CA and plain average-linkage Wasserstein merging
produce **identical** K=2 groupings on the two available full BC checkpoints.
Stage A tests the implemented capacity-matched aggregation end-to-end; it does
not establish an advantage of Multi-CA's adaptive stopping over plain merging.
That question, and any component-count capacity effect, belongs to Stage B with
same-capacity Flat-GMM controls.
