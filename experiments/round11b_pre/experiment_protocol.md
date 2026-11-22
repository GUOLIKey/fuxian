# Round 11B-pre experiment protocol (locked)

This protocol is written before any Round 11B-pre experiment is run. No VaDE/FedVaDE
neural training or gradient update is permitted. All selection rules use training
and validation data only; the CTG test split is not used for model selection.

## Frozen resources

- Models: fedvade_severe_seed0/1/2 and centralized reference, all Round 10 selected checkpoints.
- Client statistics: Round 10 last_full.pt client GMM parameters (pi, mu, log_sigma2).
- Encoder/Decoder: frozen; never updated.
- Evaluator: Round 11A calibrated_isotonic, tau=0.50,
  radius=5.185265. Fixed; not retuned from generation results.
- Data: frozen Round 10 train/validation split; train for GMM fitting,
  validation for label-free model selection and diagnostic generation metrics.
  Test is not used in this round.

## Randomness

- GMM initialization seeds: (0, 1, 2).
- Union-sample seeds: (0, 1, 2); seed 0 is the main grid, seeds 1/2 are sampling-variability checks.
- Generation inference seeds: (0, 1, 2, 3, 4).
- Neural training seeds are the existing Round 10 seeds (0,1,2 for severe).

## Stage A: M sweep

Fixed: K=3, diag, sklearn EM, source=union MoG.
M values: (100, 1000, 10000).
Main grid: sample_seed=0, init_seeds=(0, 1, 2).
Sampling variability: sample_seeds 1/2, init_seed=0.
Generation: 2000 samples per configuration per inference seed; inference seeds=(0, 1, 2, 3, 4).
Selection rule (label-free): average validation latent log-likelihood over models and
init seeds. If mean LL(M=10000) > mean LL(M=1000) + 0.05, select M=10000;
otherwise select M=1000. M=100 is the baseline; it is not used for Stage B unless
M=1000 is unavailable. BIC is computed on the union fit samples and is not compared
across different M because the sample size changes.

## Stage B: covariance ablation

Fixed: M=selected from Stage A, K=3, source=union MoG, sample_seed=0,
init_seeds=(0, 1, 2).
Configurations: diag; full; fixed shrinkage lambda=0.25/0.50/0.75;
per-component shrinkage kappa=20/50/100 where
lambda_j = kappa / (n_j_eff + kappa), n_j_eff = (sum_i r_ij)^2 / sum_i r_ij^2.
Shrinkage is applied post-fit to the full-covariance GMM; it is a diagnostic
post-fit shrinkage, not a newly trained MAP-EM model.
Selection rule (label-free): among configurations whose covariances are numerically
stable (Cholesky succeeds, min eigenvalue > 0, condition number < 1e6), choose the
highest mean validation latent log-likelihood across models and init seeds;
tie-break by lower maximum condition number. If full is best but unstable, choose the
best stable shrinkage configuration.

## Stage C: K capacity

Fixed: M=selected, source=union MoG, sample_seed=0, init_seeds=(0, 1, 2).
K values: (3, 6, 10); covariance: diag and full.
Component-conditioned audit uses init_seed=0 and 1000 samples per component.
Flags:
- LOW_SUPPORT: n_eff < 30.0 or weight < 0.01.
- PSEUDO_RARE_RISK: component rare proxy >= 0.5 and
  (class-2 MMD >= 0.5 or class-2 SWD >= 1.5) with
  at least 20 class-2-assigned samples; otherwise
  INSUFFICIENT_CLASS2.
Flags are diagnostic only and are never used to filter, weight, or merge components.
Selection rule: report BIC on fit samples and validation LL; do not select K by
rare proxy. Stage D uses K=3 as the primary fair comparison with D0; K=6 oracle is
reported in D4.

## Stage D: federated statistics aggregation

Fixed: M=selected, K=3 primary, source=union MoG unless marked oracle.
Methods:
- D0: original FedVaDE global diag K=3.
- D1: union sampling + sklearn EM, diag and full, init seeds=(0, 1, 2).
- D2: moment matching of local Gaussian components, K=3, full covariance.
  Matching: average-linkage agglomerative clustering on diagonal Bures-Wasserstein
  distances; no labels. It is recorded whether K=3 requires unsafe merges.
- D2-safe: increase K until every group's maximum normalized within-group
  Bures-Wasserstein distance <= 1.0, or K=9.
- D2-shrunk: apply the Stage B selected shrinkage (if not diag/full) to D2 group covariances.
- D3: moment-matched parameters as initialization for sklearn EM on union samples,
  K=3 full.
- D4: centralized oracle, real training latents, diag/full K=3/6, init seeds=(0, 1, 2).
  It is a diagnostic upper bound, never a federated result.

No method may use raw patient features, centralized latents, true class labels, or
test data except D4. The rare class labels are used only for the fixed evaluator's
diagnostic component audit; they never select or merge components.

## Metrics

Primary: validation latent log-likelihood (label-free model selection).
Diagnostic: rare-mode proxy (Round 11A fixed evaluator), class-2 MMD/SWD,
overall MMD/SWD, assigned rare fraction, component occupancy, n_eff, covariance
condition number, Cholesky status, minimum eigenvalue.

Class-2 MMD/SWD are computed only if at least two generated samples are assigned
class 2 and at least two real validation samples exist; otherwise they are recorded
as missing (never filled with zero). Component-conditioned results are reported
separately from unconditional generation.

## Decision rule (provisional, pre-registered)

A. Simple GMM initialization improvement is sufficient: Stage A shows a clear
   M gain and Stage B full/shrinkage does not improve over diag at fixed M/K.
B. Full-covariance / shrinkage server GMM is the priority: Stage B shows a stable
   improvement over diag at fixed M/K.
C. Local-statistics probabilistic aggregation is the priority: D2/D3 approaches
   D4 within a small validation-log-likelihood/generation gap and beats D1.
D. Safe component merging + rare-mode protection is needed: moment-matching
   diagnostics show unsafe merges or component audits show credible rare-rich
   low-support components, and D2/D3 do not approach D4.
E. INCONCLUSIVE: results are unstable or contradictory.

The report must state INCONCLUSIVE rather than forcing a direction.
