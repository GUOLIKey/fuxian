"""Lock the Round 11B-pre experimental protocol before running any experiment."""
import hashlib
import json
import time

from common import BASE, R11A, atomic_json

M_VALUES = (100, 1000, 10000)
INIT_SEEDS = (0, 1, 2)
SAMPLE_SEEDS = (0, 1, 2)
INFERENCE_SEEDS = (0, 1, 2, 3, 4)
K_VALUES = (3, 6, 10)

COV_CONFIGS = (
    ('diag', 'diag', None, None),
    ('full', 'full', 0.0, None),
    ('shrink_fixed', 'full', 0.25, None),
    ('shrink_fixed', 'full', 0.50, None),
    ('shrink_fixed', 'full', 0.75, None),
    ('shrink_percomp', 'full', None, 20.0),
    ('shrink_percomp', 'full', None, 50.0),
    ('shrink_percomp', 'full', None, 100.0),
)

LOW_SUPPORT_N_EFF = 30.0
LOW_SUPPORT_WEIGHT = 0.01
PSEUDO_RARE_PROXY = 0.5
PSEUDO_RARE_MMD = 0.5
PSEUDO_RARE_SWD = 1.5
PSEUDO_RARE_MIN_CLASS2 = 20
UNSAFE_MERGE_NORMALIZED_BW = 1.0
M_SELECTION_DELTA = 0.05

TOTAL_CONFIGS = 255


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    eval_cfg = json.loads((R11A / 'evaluator_config.json').read_text())
    text = f"""# Round 11B-pre experiment protocol (locked)

This protocol is written before any Round 11B-pre experiment is run. No VaDE/FedVaDE
neural training or gradient update is permitted. All selection rules use training
and validation data only; the CTG test split is not used for model selection.

## Frozen resources

- Models: fedvade_severe_seed0/1/2 and centralized reference, all Round 10 selected checkpoints.
- Client statistics: Round 10 last_full.pt client GMM parameters (pi, mu, log_sigma2).
- Encoder/Decoder: frozen; never updated.
- Evaluator: Round 11A calibrated_isotonic, tau={eval_cfg['tau']:.2f},
  radius={eval_cfg['radius']:.6f}. Fixed; not retuned from generation results.
- Data: frozen Round 10 train/validation split; train for GMM fitting,
  validation for label-free model selection and diagnostic generation metrics.
  Test is not used in this round.

## Randomness

- GMM initialization seeds: {INIT_SEEDS}.
- Union-sample seeds: {SAMPLE_SEEDS}; seed 0 is the main grid, seeds 1/2 are sampling-variability checks.
- Generation inference seeds: {INFERENCE_SEEDS}.
- Neural training seeds are the existing Round 10 seeds (0,1,2 for severe).

## Stage A: M sweep

Fixed: K=3, diag, sklearn EM, source=union MoG.
M values: {M_VALUES}.
Main grid: sample_seed=0, init_seeds={INIT_SEEDS}.
Sampling variability: sample_seeds 1/2, init_seed=0.
Generation: 2000 samples per configuration per inference seed; inference seeds={INFERENCE_SEEDS}.
Selection rule (label-free): average validation latent log-likelihood over models and
init seeds. If mean LL(M=10000) > mean LL(M=1000) + {M_SELECTION_DELTA}, select M=10000;
otherwise select M=1000. M=100 is the baseline; it is not used for Stage B unless
M=1000 is unavailable. BIC is computed on the union fit samples and is not compared
across different M because the sample size changes.

## Stage B: covariance ablation

Fixed: M=selected from Stage A, K=3, source=union MoG, sample_seed=0,
init_seeds={INIT_SEEDS}.
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

Fixed: M=selected, source=union MoG, sample_seed=0, init_seeds={INIT_SEEDS}.
K values: {K_VALUES}; covariance: diag and full.
Component-conditioned audit uses init_seed=0 and 1000 samples per component.
Flags:
- LOW_SUPPORT: n_eff < {LOW_SUPPORT_N_EFF} or weight < {LOW_SUPPORT_WEIGHT}.
- PSEUDO_RARE_RISK: component rare proxy >= {PSEUDO_RARE_PROXY} and
  (class-2 MMD >= {PSEUDO_RARE_MMD} or class-2 SWD >= {PSEUDO_RARE_SWD}) with
  at least {PSEUDO_RARE_MIN_CLASS2} class-2-assigned samples; otherwise
  INSUFFICIENT_CLASS2.
Flags are diagnostic only and are never used to filter, weight, or merge components.
Selection rule: report BIC on fit samples and validation LL; do not select K by
rare proxy. Stage D uses K=3 as the primary fair comparison with D0; K=6 oracle is
reported in D4.

## Stage D: federated statistics aggregation

Fixed: M=selected, K=3 primary, source=union MoG unless marked oracle.
Methods:
- D0: original FedVaDE global diag K=3.
- D1: union sampling + sklearn EM, diag and full, init seeds={INIT_SEEDS}.
- D2: moment matching of local Gaussian components, K=3, full covariance.
  Matching: average-linkage agglomerative clustering on diagonal Bures-Wasserstein
  distances; no labels. It is recorded whether K=3 requires unsafe merges.
- D2-safe: increase K until every group's maximum normalized within-group
  Bures-Wasserstein distance <= {UNSAFE_MERGE_NORMALIZED_BW}, or K=9.
- D2-shrunk: apply the Stage B selected shrinkage (if not diag/full) to D2 group covariances.
- D3: moment-matched parameters as initialization for sklearn EM on union samples,
  K=3 full.
- D4: centralized oracle, real training latents, diag/full K=3/6, init seeds={INIT_SEEDS}.
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
"""
    (BASE / 'experiment_protocol.md').write_text(text)

    evaluation = f"""# Round 11B-pre evaluation protocol

## Fixed evaluator (inherited from Round 11A)

- Primary classifier: `{eval_cfg['primary_classifier']}`
- Rare threshold tau: {eval_cfg['tau']}
- Rare distance radius: {eval_cfg['radius']}
- Radius source: train-rare self nearest-neighbor q95
- The evaluator is fixed before Round 11B-pre experiments and is not retuned.

## Rare-mode proxy

`rare_proxy = mean(calibrated p(class 2) >= tau AND distance_to_train_rare <= radius)`.

This is a proxy, not a true synthetic NSP label. It is reported with class-2 MMD/SWD,
assigned rare fraction, and class-2 sample count. If class-2 count is <2, class-2
MMD/SWD are missing, not zero.

## Selection versus evaluation

- GMM fitting: training data or union samples only.
- Model selection: validation latent log-likelihood / BIC; never rare proxy or test.
- Rare labels: used only in the fixed evaluator and diagnostic component audit;
  never used to fit, initialize, merge, filter, or select GMMs.
- Test split: not used in Round 11B-pre.

## Generation budget

- Unconditional: 2000 samples per configuration per inference seed, seeds 0-4.
- Component-conditioned: 1000 samples per component, inference seed 0.
- Same frozen decoder and same evaluator for every method.

## Reported metrics

- Validation latent log-likelihood.
- BIC/AIC on fit samples (sample source stated).
- Rare proxy, assigned rare fraction, class-2 count.
- Class-2 MMD/SWD and overall MMD/SWD.
- Component weight, occupancy, effective support n_eff, min eigenvalue,
  condition number, Cholesky status.
"""
    (BASE / 'evaluation_protocol.md').write_text(evaluation)

    lock = {
        'created_at': time.time(),
        'experiment_protocol_sha256': sha256(BASE / 'experiment_protocol.md'),
        'evaluation_protocol_sha256': sha256(BASE / 'evaluation_protocol.md'),
        'm_values': list(M_VALUES),
        'init_seeds': list(INIT_SEEDS),
        'sample_seeds': list(SAMPLE_SEEDS),
        'inference_seeds': list(INFERENCE_SEEDS),
        'k_values': list(K_VALUES),
        'covariance_configs': [list(c) for c in COV_CONFIGS],
        'selection_rules': {
            'M': 'choose 10000 only if mean val LL exceeds 1000 by > 0.05 nats; else 1000',
            'covariance': 'highest mean val LL among numerically stable configs; tie-break lower max condition number',
            'K': 'report BIC/val LL; do not select by rare proxy; Stage D uses K=3 primary',
        },
        'flags': {
            'LOW_SUPPORT_N_EFF': LOW_SUPPORT_N_EFF,
            'LOW_SUPPORT_WEIGHT': LOW_SUPPORT_WEIGHT,
            'PSEUDO_RARE_PROXY': PSEUDO_RARE_PROXY,
            'PSEUDO_RARE_MMD': PSEUDO_RARE_MMD,
            'PSEUDO_RARE_SWD': PSEUDO_RARE_SWD,
            'PSEUDO_RARE_MIN_CLASS2': PSEUDO_RARE_MIN_CLASS2,
            'UNSAFE_MERGE_NORMALIZED_BW': UNSAFE_MERGE_NORMALIZED_BW,
        },
        'total_configs': TOTAL_CONFIGS,
        'training': 'none',
    }
    atomic_json(BASE / 'protocol_lock.json', lock)
    print('PROTOCOL_LOCKED', lock['experiment_protocol_sha256'], flush=True)


if __name__ == '__main__':
    main()
