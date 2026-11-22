# Round 11B-pre evaluation protocol

## Fixed evaluator (inherited from Round 11A)

- Primary classifier: `calibrated_isotonic`
- Rare threshold tau: 0.5
- Rare distance radius: 5.1852652072906436
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
