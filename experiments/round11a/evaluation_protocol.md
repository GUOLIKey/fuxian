# Round 11A evaluation protocol

## Data
- Reuse the frozen Round 10 UCI CTG split: train 1360, validation 339, test 427.
- No re-splitting, no re-shuffling, no test-based model selection.
- Test is used only for exploratory diagnostics; it is not used to choose tau, radius, K, or checkpoints.

## Classifier
- Base: RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample', random_state=42, n_jobs=1).
- Trained on the training split only.
- Primary classifier: calibrated_isotonic; calibration: CalibratedClassifierCV(method='isotonic', cv=5) on train only.
- Primary classifier/threshold chosen by validation Youden only; test is not used.
- The validation split is reserved for positive/negative control and threshold/radius rules.
- Rare class = NSP class 2.

## Rare-distance reference
- Reference set: real training samples of class 2.
- Distance: Euclidean in the standardized 21-feature space.
- Self-NN distance excludes the sample itself.
- Primary candidates: q90/q95/q99 of training-rare self-NN distance.
- Selected radius rule: fallback train-rare q95 (no quantile met val positive>=0.90, negative<=0.05).
- Selected radius: 5.185265.
- The Round 10 test-rare q95 radius (4.618951) is reproduced for comparison only.

## Rare threshold
- Candidate tau: 0.50, 0.60, 0.70, 0.80, 0.90.
- Primary rule: validation Youden-best tau=0.50 for calibrated_isotonic; no test labels used.
- Selected tau: 0.50.
- Validation controls: real rare samples (positive), real non-rare samples (negative).
- Bootstrap: 1000 resamples, percentile 95% interval.

## Evaluation rules
- `assigned_label` = argmax primary class probability.
- `rare_flag` = primary classifier p(class 2) >= tau.
- `trusted_rare` = rare_flag AND distance-to-training-rare <= radius.
- `rare_mode_recovery_proxy` = mean(trusted_rare) over generated samples.
- Generated samples have no true NSP labels; classifier predictions are proxies, not ground truth.
- Same classifier, threshold, radius, generation budget and random seeds are used for every model.

## Generation budget
- Unconditional: 2000 samples per model per inference seed; inference seeds 0-4.
- Component-conditioned: 1000 samples per component, inference seed 0 (diagnostic only).
- No test-based selection of generation hyperparameters.

## Status labels
- `OK`, `LIMITED`, `INCOMPATIBLE`, `N/A` are used for the client-union vs server-decoder compatibility audit.
- An `INCOMPATIBLE` union comparison is not interpreted as evidence of aggregation failure.
