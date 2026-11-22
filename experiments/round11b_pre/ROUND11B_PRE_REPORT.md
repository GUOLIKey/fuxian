# FedVaDE Round 11B-pre: Federated GMM factor diagnosis

**No VaDE/FedVaDE neural training was performed.** All results are frozen-model GMM/statistics diagnostics. The CTG test split was not used for selection.

## Protocol

- Selected M: **10000**
- Selected covariance: **shrink_percomp** (base=full, lambda=, kappa=50.0)
- Selected K by BIC: **6 diag**
- Evaluator: fixed Round 11A calibrated_isotonic, tau=0.50, radius fixed.

## Stage A: M sweep (K=3, diag, union samples)

| M | mean val LL | rare proxy | class-2 MMD | class-2 SWD | overall MMD | overall SWD |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | -18.1567 | 0.0203 | 0.1844 | 0.7504 | 0.0674 | 0.3187 |
| 1000 | -17.7579 | 0.0116 | 0.1687 | 0.7214 | 0.0767 | 0.3373 |
| 10000 | -17.6763 | 0.0111 | 0.1753 | 0.7350 | 0.0767 | 0.3393 |

- Validation-LL gain M=1000 vs M=100: **0.3989 nats/sample**.
- Validation-LL gain M=10000 vs M=1000: **0.0816 nats/sample**.
- Selection rule: select 10000 only if mean val LL exceeds 1000 by > 0.05; otherwise 1000; 100 is baseline only; selected M=10000.
- BIC is on union fit samples and is not comparable across different M.

## Stage B: covariance ablation (fixed M, K=3, union samples)

| Config | lambda | kappa | mean val LL | max cond | stable | rare proxy | class-2 MMD | class-2 SWD |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| diag | — | — | -17.6763 | 19.0 | True | 0.0111 | 0.1753 | 0.7455 |
| full | 0.0 | — | -17.4919 | 35.3 | True | 0.0108 | 0.1382 | 0.6752 |
| shrink_fixed | 0.25 | — | -17.5202 | 28.6 | True | 0.0111 | 0.1464 | 0.6921 |
| shrink_fixed | 0.5 | — | -17.5603 | 24.3 | True | 0.0112 | 0.1509 | 0.6993 |
| shrink_fixed | 0.75 | — | -17.6139 | 21.4 | True | 0.0109 | 0.1530 | 0.7058 |
| shrink_percomp | — | 20.0 | -17.4908 | 29.9 | True | 0.0108 | 0.1388 | 0.6752 |
| shrink_percomp | — | 50.0 | -17.4902 | 26.3 | True | 0.0109 | 0.1395 | 0.6725 |
| shrink_percomp | — | 100.0 | -17.4902 | 23.7 | True | 0.0110 | 0.1396 | 0.6747 |

- Best stable covariance: **shrink_percomp**; val-LL gain over diag: **0.1861 nats/sample**.
- Shrinkage is post-fit, not a newly trained MAP-EM model; BIC for shrinkage is approximate.

## Stage C: K capacity and component audit

| K | covariance | mean val LL | mean BIC | stable | rare proxy | class-2 MMD | class-2 SWD | PSEUDO_RARE components | LOW_SUPPORT components |
| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 3 | diag | -17.6763 | 334644.1 | True | 0.0111 | 0.1753 | 0.7478 | 0 | 0 |
| 3 | full | -17.4919 | 334956.2 | True | 0.0108 | 0.1382 | 0.6778 | 0 | 0 |
| 6 | diag | -17.5683 | 333085.6 | True | 0.0117 | 0.1779 | 0.7520 | 0 | 1 |
| 6 | full | -17.5447 | 335227.6 | True | 0.0116 | 0.1527 | 0.7114 | 0 | 1 |
| 10 | diag | -17.5439 | 333391.5 | True | 0.0119 | 0.1658 | 0.7256 | 0 | 2 |
| 10 | full | -17.5613 | 337152.5 | True | 0.0119 | 0.1784 | 0.7375 | 0 | 1 |

- BIC-selected K: **6 diag** (rule: lowest mean BIC on fit samples among stable configs).
- PSEUDO_RARE_RISK components: **0**; LOW_SUPPORT components: **5**.
- Flags are diagnostic only; they were never used to filter or reweight components.

## Stage D: federated statistics aggregation and oracle gap

| Method | mean val LL | rare proxy | class-2 MMD | class-2 SWD | gap val-LL vs D4 full K3 | gap rare vs D4 full K3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D0_original | -17.5114 | 0.0122 | 0.1686 | 0.7283 | -3.8863 | -0.0406 |
| D1_union_em_diag | -17.6763 | 0.0111 | 0.1753 | 0.7477 | -4.0512 | -0.0417 |
| D1_union_em_full | -17.4919 | 0.0108 | 0.1382 | 0.6723 | -3.8668 | -0.0420 |
| D2_moment_full | -18.0663 | 0.0336 | 0.2169 | 0.8792 | -4.4412 | -0.0192 |
| D2_moment_shrunk | -18.0675 | 0.0336 | 0.2175 | 0.8881 | -4.4424 | -0.0193 |
| D2_safe_full | -17.8500 | 0.0253 | 0.1918 | 0.8005 | -4.2249 | -0.0275 |
| D3_moment_init_em_full | -17.5526 | 0.0130 | 0.1382 | 0.6966 | -3.9275 | -0.0398 |
| D4_oracle_full_K3 | -13.6251 | 0.0529 | 0.0552 | 0.4516 | 0.0000 | 0.0000 |
| D4_oracle_full_K6 | -13.4358 | 0.0656 | 0.0474 | 0.4181 | 0.1893 | 0.0127 |

- Moment matching vs union EM full: rare-proxy change **0.0228**; moment-init EM vs union EM full: **0.0022**.
- D2 oracle rare gap: **-0.0192**; D3 oracle rare gap: **-0.0398**.
- Unsafe K=3 moment merges: **5 group rows** across models.
- D4 is a centralized real-latent oracle, not a federated method.

## Answers

1. **Is M=100 too small?** See the Stage A val-LL/rare table. M=1000 vs 100 val-LL gain is 0.3989 nats/sample; M=10000 vs 1000 is 0.0816. The report selects M based on the locked validation rule, not rare proxy.

2. **Does Full Covariance still help under controlled M/K/source?** See Stage B. Best stable covariance is shrink_percomp with val-LL gain 0.1861 over diag. If the gain is small or unstable, the answer is NO/INCONCLUSIVE.

3. **Is shrinkage more stable?** See Stage B max condition numbers and stability column. Shrinkage is useful if it improves val LL or reduces condition numbers without losing rare/class-2 quality.

4. **Is K=6 better than K=3?** See Stage C BIC/val LL and rare/class-2 metrics. BIC selected K=6 diag. Do not select K by rare proxy.

5. **Does K=10 create untrustworthy small components?** PSEUDO_RARE_RISK components=0, LOW_SUPPORT components=5. See component_quality_audit.csv for occupancy, n_eff, class-2 MMD/SWD and flags.

6. **Can client statistics approach centralized full GMM?** Compare D2/D3 with D4 in Stage D. D2 rare gap to D4 full K3 is -0.0192; D3 is -0.0398. A small negative gap (federated below oracle) is expected; a large gap means the statistical aggregation is insufficient.

7. **Is moment matching better than ordinary Union Sampling + EM?** D2 rare-proxy change vs D1 full is 0.0228; D3 is 0.0022. Positive is better; near-zero means no advantage.

8. **Which components should not be merged?** See moment_matching_diagnostics.csv and moment_matching_pairwise.csv. Unsafe groups are those with maximum normalized Bures-Wasserstein dispersion above the locked threshold; their members are listed.

9. **Is label-free rare-mode protection still needed?** Yes if PSEUDO_RARE_RISK or credible rare-rich LOW_SUPPORT components appear, or if D2 K=3 has unsafe merges. The flags are diagnostic; they were not used to merge or filter.

10. **Next priority: simple server Full-Covariance GMM or Macro/Micro?** See the provisional decision below. Do not start ACD or Multi-CA before a controlled same-capacity baseline.

## Provisional decision (pre-registered rules)

**B. 优先开发 Full-Covariance / Shrinkage 服务器 GMM：在固定 M、K、来源后仍有稳定增益。**

This is a rule-based reading of the controlled validation diagnostics, not a claim of a new independent test result. If the evidence is contradictory, the report says E/INCONCLUSIVE.

## Evidence limits

- No neural training; frozen Round 10 checkpoints.
- Union samples come from local GMMs; they are not raw patient data.
- Full covariance is post-fit in Stage B; it is not yet integrated into the VaDE ELBO.
- Shrinkage BIC is approximate; covariance selection uses validation LL and stability.
- Moment matching is a first-two-moment Gaussian approximation and can collapse multimodal groups; the diagnostics mark unsafe merges.
- Rare proxy is a fixed classifier+distance proxy, not a true generated-label metric.
- Test split was not used.
