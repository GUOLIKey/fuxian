# Multi-CA Factorial v1 — Stage A Final Report (Breast Cancer)

**Status:** 12/12 Stage A runs COMPLETED; no FAILED/INTERRUPTED.  
**Capacity-matched design:** Multi-CA K=2, same as author global GMM.  
**Bottom line:** Multi-CA v1 does **not** improve clustering, GMM density, or
minority-mode generation over the original aggregation in this Stage A test.
F01 training improves density/generation but not unique-row ARI. These are
honest negative results, not tuned positives.

## 1. What was run

| Group | Training | Aggregation | Seeds |
|---|---|---|---|
| A | Author: alpha=0.01, client scheduler per minibatch | original | 0,1,2 |
| B | Author | Multi-CA K=2 | 0,1,2 |
| C | F01: alpha=0.05, client scheduler per epoch | original | 0,1,2 |
| D | F01 | Multi-CA K=2 | 0,1,2 |

- 50-epoch shared author pretrain per seed; A/B/C/D for a seed load the same
  pretrain checkpoint and RNG state.
- 300 formal rounds; server checkpoint every 25; full checkpoint at 100/200/300.
- Main result: round 300; no best-checkpoint selection.
- BC has **no independent test split**. All BC metrics are author-protocol
  in-sample; generation minority metrics are in-sample classifier proxies.

## 2. Four-group results (mean ± SD over seeds)

| Group | Unique ARI | Unique NMI | Unique Purity | Server GMM NLL | Gen MMD | Gen SWD | Minority MMD | Minority SWD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.3753 ± 0.1451 | 0.2872 ± 0.1210 | 0.8032 ± 0.0616 | 21.592 ± 0.869 | 0.14695 | 0.46123 | 0.26865 | 0.60291 |
| B | 0.3775 ± 0.1416 | 0.2886 ± 0.1186 | 0.8043 ± 0.0597 | 21.601 ± 0.869 | 0.14688 | 0.46112 | 0.27130 | 0.60590 |
| C | 0.3660 ± 0.1114 | 0.2707 ± 0.0950 | 0.8037 ± 0.0462 | 16.588 ± 0.515 | 0.04064 | 0.25538 | 0.07248 | 0.34701 |
| D | 0.3673 ± 0.1092 | 0.2717 ± 0.0933 | 0.8043 ± 0.0452 | 16.591 ± 0.515 | 0.04073 | 0.25566 | 0.07295 | 0.34689 |

Purity is similar across groups (~0.80, with large seed SD) and is not discriminative here. ARI/NMI
and NLL are the informative metrics. Full precision in
`results/four_group_summary.csv`.

### Per-seed raw results

| Group | Seed | Unique ARI | Unique NMI | Server GMM NLL |
|---|---:|---:|---:|---:|
| A | 0 | 0.5084 | 0.3872 | 21.693 |
| A | 1 | 0.3968 | 0.3217 | 20.677 |
| A | 2 | 0.2205 | 0.1527 | 22.407 |
| B | 0 | 0.5084 | 0.3872 | 21.651 |
| B | 1 | 0.3968 | 0.3217 | 20.708 |
| B | 2 | 0.2271 | 0.1570 | 22.444 |
| C | 0 | 0.4259 | 0.3308 | 16.852 |
| C | 1 | 0.4346 | 0.3201 | 15.994 |
| C | 2 | 0.2375 | 0.1612 | 16.918 |
| D | 0 | 0.4259 | 0.3308 | 16.842 |
| D | 1 | 0.4346 | 0.3201 | 15.999 |
| D | 2 | 0.2413 | 0.1641 | 16.932 |


## 3. Paired effects (same seed; mean ± SD)

| Contrast | Δ Unique ARI | Δ NMI | Δ Server NLL | Δ Gen MMD | Δ Gen SWD |
|---|---:|---:|---:|---:|---:|
| B − A | +0.0022 ± 0.0038 | +0.0015 ± 0.0025 | +0.0085 ± 0.0439 | −0.00007 | −0.00011 |
| D − C | +0.0013 ± 0.0022 | +0.0010 ± 0.0017 | +0.0031 ± 0.0124 | +0.00008 | +0.00029 |
| C − A | −0.0093 | −0.0165 | **−5.0045** | −0.1063 | −0.2059 |
| D − B | −0.0102 | −0.0170 | −5.0099 | −0.1062 | −0.2055 |
| Interaction | −0.00094 | −0.00048 | −0.0054 | +0.00015 | +0.00039 |

- Multi-CA effects (B−A, D−C) are negligible and inconsistent in sign; the
  seedwise ARI differences are essentially zero or +0.006.
- F01 effects (C−A) are large and favorable for NLL/generation, but **not** for
  unique ARI.
- Interaction is negligible: Multi-CA does not interact meaningfully with the
  F01 training configuration.

Per-seed values are in `results/per_seed_results.csv`; paired raw rows in
`results/paired_effects.csv`.

## 4. Training stability

- Author groups A/B: unique ARI SD **0.145/0.142**; NLL SD **0.869/0.869**.
- F01 groups C/D: ARI SD **0.111/0.109**; NLL SD **0.515/0.515**.
- Seed 2 is the low-ARI seed in all groups. F01 reduces NLL and generation
  variance, but does not remove the seed-2 clustering drop.
- Multi-CA ON/OFF within the same training config is almost identical; there is
  no stability gain from Multi-CA.

## 5. Merge diagnostics

- B/D: 14 input local components → 2 macro components; **12 merges/run**.
- Unsafe merges: **0/12 seeds** under normalized-Bures guard 1.0.
- Max normalized merge distance: **0.655–0.797** (below 1.0).
- No capacity increase: K_macro = K_global = 2.
- Final server GMM weights differ only at ~1e-4 between A vs B and C vs D; the
  macro sampling distribution did not materially change the converged server GMM.

This explains the null result: at K=2, complete-linkage Multi-CA grouping and
plain moment-matched aggregation coincide on these component sets, and the server
GMM converges to nearly the same solution. The adaptive/guard part never had to
prevent an unsafe merge.

## 6. Frozen PoC (pre-training)

On existing Author and F01 full checkpoints, Multi-CA K=2 and plain
average-linkage Wasserstein K=2 produced **identical groups, weights, NLL and
generation metrics**. Moment-matched/flat K=2 aggregation improved over the
stored server GMM on Author (NLL 22.37→21.22, ARI 0.471→0.510), and improved
NLL/MMD/SWD on F01 while slightly lowering ARI. This shows the aggregation **path**
matters versus the old server GMM, but **not** that Multi-CA's adaptive part is
better than plain merging. See `FROZEN_POC_REPORT.md`.

## 7. Generation and minority mode (exploratory, in-sample)

C/D (F01) have much lower overall MMD/SWD than A/B:
overall MMD 0.0406–0.0407 vs 0.1469, SWD 0.2554–0.2557 vs 0.4611.
Minority-class MMD/SWD are also lower for C/D (MMD ~0.0725 vs ~0.269; SWD
~0.347 vs ~0.603). However:

- D−C changes are effectively zero; Multi-CA does not add minority coverage.
- The minority classifier is trained on all 569 rows; these are in-sample
  proxies, not independent rare-mode precision/recall.
- BC has two clinical classes, not a controlled rare-client submode; this cannot
  establish the rare-mode hypothesis.

## 8. Answers to Q1–Q10

| Question | Answer |
|---|---|
| Q1. Preserve more fine-grained modes? | **No evidence in Stage A.** K=2 capacity; no component-count increase. |
| Q2. Closer rare-mode generation? | **No Multi-CA-specific improvement.** F01 improves minority MMD/SWD, but D−C ≈ 0. |
| Q3. Better than plain hierarchical merge? | **No evidence.** Frozen PoC: identical K=2 groups/metrics; formal has no plain-merge arm. |
| Q4. Gain only from more components? | **No component increase in Stage A**, so this is not a capacity win. Adaptive K>2 was not tested. |
| Q5. Improve ARI/NMI/overall generation? | **No.** B−A ARI +0.002; D−C ARI +0.001; NLL slightly worse; generation unchanged. |
| Q6. Effect in both Author and F01? | **No**; B−A and D−C are both negligible. |
| Q7. Which config more stable? | **F01** (lower NLL/generation variance), but not for unique ARI; seed 2 remains weak. |
| Q8. Consistent across seeds? | **No meaningful consistency**; effects are zero or tiny and mixed. |
| Q9. Truly changes server Global GMM? | **Barely.** Final GMM weights/params nearly identical A/B and C/D. |
| Q10. Worth broadcasting global generation knowledge? | **Not justified by Stage A.** First need a capacity-controlled Multi-CA/Flat-GMM test and Stage B rare-mode design. |

## 9. Why this is a negative result

1. Stage A forced K=2 to avoid a capacity confound. At this capacity, the local
   component geometry has two clear macro groups; Multi-CA and plain moment
   matching find the same partition.
2. The macro mixture is close enough to the raw client-union mixture that the
   server GMM/VAE converge to nearly identical parameters.
3. No unsafe merge is encountered (max normalized distance ~0.8), so the adaptive
   guard has no opportunity to help.
4. The F01 improvement is an alpha/scheduler effect, not a Multi-CA effect.

This does **not** prove Multi-CA is useless. It proves that a capacity-matched
K=2 Multi-CA v1 on BC has no detectable benefit. A fair test of the original
fine-grained hypothesis requires Stage B: adaptive K>2 with a same-capacity
Flat-GMM control and a controlled rare-client partition.

## 10. Integrity and artifacts

- `diagnostics/INTEGRITY_CHECK.json`: 12/12 status ok, 300 rounds, full
  checkpoints at 100/200/300, 10 clients each, pretrain hashes paired per seed.
- `results/`: per-seed, group summary, paired effects, training curves.
- `figures/`: SVG training ARI/NLL and paired-effect figures.
- `monitoring/remote_status.sh`: read-only server monitor.
- Mac auto-refresh monitor: `/Users/cheeseowl/Desktop/MultiCA_实时监控.command`
  (5-second refresh, Ctrl-C to stop; `--once` for one query).
- Checkpoints are kept on the server under
  `reproduction_runs/multica_factorial_v1/stage_a/`; not included in GitHub.

## 11. Caveats

- No independent BC test split; metrics are author-protocol in-sample.
- Three seeds only; no significance claim.
- Multi-CA K=2 is capacity-matched; fine-grained mode retention is untested.
- Frozen PoC and formal generation minority metrics use in-sample classifiers.
- Stage B/C were not started; no CTG result is mixed into this report.

## 12. GitHub

Pushed to:

- Branch: `research/multica-factorial-v1`
- Branch URL: https://github.com/GUOLIKey/fuxian/tree/research/multica-factorial-v1
- Artifact commit: `ff1d1e837495e957f3a5f6567aa07ff2ac7406fc`
- Commit URL: https://github.com/GUOLIKey/fuxian/commit/ff1d1e837495e957f3a5f6567aa07ff2ac7406fc

Only documentation, source, tests, configs, summary CSVs, figures, frozen-PoC
CSV/JSON, and monitor scripts were pushed. No raw data, model weights,
checkpoints, keys, environment variables, or large logs were included.
`github_publish_status.json` records the same status.
