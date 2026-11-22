"""E1: audit and calibrate the rare-mode evaluator using train/validation only."""
import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

from common import BASE, RARE, split_data, update_progress, bootstrap_mean
from evaluator import (THRESHOLDS, fit_calibrated, fit_uncalibrated, class_report,
                       prob_rare, distance_reference, distance_to_rare,
                       youden_threshold, threshold_table)


def write_rows(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    start = time.time()
    Xtr, ytr, Xval, yval, Xte, yte = split_data()
    update_progress(stage='EVALUATOR_CALIBRATION', status='RUNNING',
                    current_model='reference classifier', current_experiment='E1',
                    progress_current=2, progress_total=12, rare_mode_proxy=None)

    print('fitting uncalibrated RF...', flush=True)
    uncal = fit_uncalibrated(Xtr, ytr, n_estimators=300, seed=42)
    print('fitting calibrated sigmoid...', flush=True)
    cal_sig = fit_calibrated(Xtr, ytr, n_estimators=300, seed=42, cv=5, method='sigmoid')
    print('fitting calibrated isotonic...', flush=True)
    cal_iso = fit_calibrated(Xtr, ytr, n_estimators=300, seed=42, cv=5, method='isotonic')

    classifiers = {
        'uncalibrated': uncal,
        'calibrated_sigmoid': cal_sig,
        'calibrated_isotonic': cal_iso,
    }

    nn_train_rare, radius_q95, self_train = distance_reference(Xtr, ytr, quantile=0.95)
    _, radius_q99, _ = distance_reference(Xtr, ytr, quantile=0.99)
    _, radius_train_q90, _ = distance_reference(Xtr, ytr, quantile=0.90)
    # Test-based radius is only for reproducing the Round 10 protocol, never for selection.
    rare_test = np.asarray(Xte)[np.asarray(yte) == RARE]
    test_self = NearestNeighbors(n_neighbors=2).fit(rare_test).kneighbors(rare_test)[0][:, 1]
    radius_test_q95 = float(np.quantile(test_self, 0.95))

    rows = []
    for name, clf in classifiers.items():
        for split, X, y in (('val', Xval, yval), ('test', Xte, yte)):
            report, pred, p_rare, conf = class_report(clf, X, y)
            for cls in (0, 1, 2):
                r = report[str(cls)]
                rows.append({
                    'section': 'classifier_metrics', 'classifier': name, 'split': split,
                    'class': cls, 'threshold': '', 'group': '', 'metric': 'precision_recall_f1',
                    'value': '', 'precision': r['precision'], 'recall': r['recall'],
                    'f1': r['f1-score'], 'support': r['support'], 'n': len(y),
                    'bootstrap_low': '', 'bootstrap_high': '', 'note': '',
                })
        # Threshold and control tables on validation and exploratory test.
        for split, X, y in (('val', Xval, yval), ('test', Xte, yte)):
            p_rare = prob_rare(clf, X)
            dist = distance_to_rare(nn_train_rare, X)
            for group, mask in (('positive', y == RARE), ('negative', y != RARE)):
                for tau in THRESHOLDS:
                    rows.append({
                        'section': 'threshold_sensitivity', 'classifier': name, 'split': split,
                        'class': '', 'threshold': tau, 'group': group,
                        'metric': 'classifier_pass', 'value': float(np.mean(p_rare[mask] >= tau)),
                        'precision': '', 'recall': '', 'f1': '', 'support': int(mask.sum()),
                        'n': int(mask.sum()), 'bootstrap_low': '', 'bootstrap_high': '',
                        'note': 'distance independent of tau',
                    })
                    rows.append({
                        'section': 'threshold_sensitivity', 'classifier': name, 'split': split,
                        'class': '', 'threshold': tau, 'group': group,
                        'metric': 'joint_pass',
                        'value': float(np.mean((p_rare[mask] >= tau) & (dist[mask] <= radius_q95))),
                        'precision': '', 'recall': '', 'f1': '', 'support': int(mask.sum()),
                        'n': int(mask.sum()), 'bootstrap_low': '', 'bootstrap_high': '',
                        'note': f'radius_q95={radius_q95:.6f}',
                    })

    # Distance-only sensitivity, independent of classifier.
    for split, X, y in (('val', Xval, yval), ('test', Xte, yte)):
        dist = distance_to_rare(nn_train_rare, X)
        for quantile, radius in (('q90', radius_train_q90), ('q95', radius_q95), ('q99', radius_q99)):
            for group, mask in (('positive', y == RARE), ('negative', y != RARE)):
                rows.append({
                    'section': 'distance_sensitivity', 'classifier': 'none', 'split': split,
                    'class': '', 'threshold': quantile, 'group': group,
                    'metric': 'distance_pass', 'value': float(np.mean(dist[mask] <= radius)),
                    'precision': '', 'recall': '', 'f1': '', 'support': int(mask.sum()),
                    'n': int(mask.sum()), 'bootstrap_low': '', 'bootstrap_high': '',
                    'note': f'radius={radius:.6f}',
                })

    # Choose the primary classifier and threshold by validation Youden only.
    candidates = []
    for name, clf in classifiers.items():
        p_val = prob_rare(clf, Xval)
        best_row, table = youden_threshold(p_val, yval)
        candidates.append({'name': name, 'clf': clf, 'best': best_row, 'table': table})
    primary = max(candidates, key=lambda c: (c['best']['youden'],
                                             c['best']['positive_pass'],
                                             -c['best']['negative_pass']))
    primary_name = primary['name']
    primary_clf = primary['clf']
    tau = float(primary['best']['threshold'])
    tau_table = primary['table']
    best_j = primary['best']
    tau_rule = f'validation Youden-best tau={tau:.2f} for {primary_name}; no test labels used'

    # Select distance quantile on validation controls.
    dist_val = distance_to_rare(nn_train_rare, Xval)
    radius_candidates = []
    for quantile, radius in (('q90', radius_train_q90), ('q95', radius_q95), ('q99', radius_q99)):
        pos = float(np.mean(dist_val[yval == RARE] <= radius))
        neg = float(np.mean(dist_val[yval != RARE] <= radius))
        radius_candidates.append({'quantile': quantile, 'radius': radius,
                                  'positive_pass': pos, 'negative_pass': neg})
    eligible = [r for r in radius_candidates
                if r['positive_pass'] >= 0.90 and r['negative_pass'] <= 0.05]
    if eligible:
        chosen_radius = min(eligible, key=lambda r: r['radius'])
        radius_rule = 'smallest train-rare quantile with val positive>=0.90, negative<=0.05'
    else:
        chosen_radius = next(r for r in radius_candidates if r['quantile'] == 'q95')
        radius_rule = 'fallback train-rare q95 (no quantile met val positive>=0.90, negative<=0.05)'
    radius = float(chosen_radius['radius'])

    # Primary controls with bootstrap uncertainty.
    primary = {}
    for split, X, y in (('val', Xval, yval), ('test', Xte, yte)):
        p_rare = prob_rare(primary_clf, X)
        dist = distance_to_rare(nn_train_rare, X)
        for group, mask in (('positive', y == RARE), ('negative', y != RARE)):
            clf_pass = p_rare[mask] >= tau
            dist_pass = dist[mask] <= radius
            joint = clf_pass & dist_pass
            mean_j, lo_j, hi_j = bootstrap_mean(joint.astype(float), n_boot=1000, seed=7100)
            mean_c, lo_c, hi_c = bootstrap_mean(clf_pass.astype(float), n_boot=1000, seed=7101)
            mean_d, lo_d, hi_d = bootstrap_mean(dist_pass.astype(float), n_boot=1000, seed=7102)
            primary[(split, group)] = {
                'n': int(mask.sum()), 'classifier_pass': float(mean_c),
                'classifier_low': lo_c, 'classifier_high': hi_c,
                'distance_pass': float(mean_d), 'distance_low': lo_d, 'distance_high': hi_d,
                'joint_pass': float(mean_j), 'joint_low': lo_j, 'joint_high': hi_j,
            }
            for metric, mean, lo, hi in (('classifier_pass', mean_c, lo_c, hi_c),
                                         ('distance_pass', mean_d, lo_d, hi_d),
                                         ('joint_pass', mean_j, lo_j, hi_j)):
                rows.append({
                    'section': 'primary_control', 'classifier': primary_name,
                    'split': split, 'class': '', 'threshold': tau, 'group': group,
                    'metric': metric, 'value': mean, 'precision': '', 'recall': '',
                    'f1': '', 'support': int(mask.sum()), 'n': int(mask.sum()),
                    'bootstrap_low': lo, 'bootstrap_high': hi,
                    'note': f'radius={radius:.6f}; tau_rule={tau_rule}; radius_rule={radius_rule}',
                })

    # Round 10 reproduction: uncalibrated, tau=0.8, test-rare q95 radius.
    p_test = prob_rare(uncal, Xte)
    dist_test = distance_to_rare(nn_train_rare, Xte)
    # Round 10 compared generated samples to test rare samples; reproduce that distance reference.
    nn_test_rare = NearestNeighbors(n_neighbors=1).fit(rare_test)
    dist_test_r10 = nn_test_rare.kneighbors(Xte)[0][:, 0]
    for group, mask in (('positive', yte == RARE), ('negative', yte != RARE)):
        joint = (p_test[mask] >= 0.80) & (dist_test_r10[mask] <= radius_test_q95)
        rows.append({
            'section': 'round10_reproduction', 'classifier': 'uncalibrated', 'split': 'test',
            'class': '', 'threshold': 0.80, 'group': group, 'metric': 'joint_pass',
            'value': float(np.mean(joint)), 'precision': '', 'recall': '', 'f1': '',
            'support': int(mask.sum()), 'n': int(mask.sum()), 'bootstrap_low': '',
            'bootstrap_high': '', 'note': f'Round10 radius test_rare_q95={radius_test_q95:.6f}',
        })

    write_rows(BASE / 'evaluator_calibration.csv', rows)

    config = {
        'primary_classifier': primary_name,
        'base_classifier': 'RandomForestClassifier(n_estimators=300, class_weight=balanced_subsample, random_state=42, n_jobs=1)',
        'calibration': ('uncalibrated' if primary_name == 'uncalibrated'
                        else f"CalibratedClassifierCV(method='{primary_name.replace('calibrated_', '')}', cv=5) on train only"),
        'classifier_youden': {c['name']: c['best'] for c in candidates},
        'trained_on': 'Round 10 train split only',
        'thresholds_considered': list(THRESHOLDS),
        'tau': tau,
        'tau_rule': tau_rule,
        'tau_table': tau_table,
        'tau_best_youden': best_j,
        'radius': radius,
        'radius_rule': radius_rule,
        'radius_candidates': radius_candidates,
        'radius_test_q95_for_reproduction_only': radius_test_q95,
        'rare_class': RARE,
        'positive_control': 'real validation rare samples',
        'negative_control': 'real validation non-rare samples',
        'test_status': 'exploratory; not used for threshold or model selection',
        'primary_controls': {f'{k[0]}:{k[1]}': v for k, v in primary.items()},
        'classifier_metrics': {
            name: {split: class_report(clf, X, y)[0]
                   for split, X, y in (('val', Xval, yval), ('test', Xte, yte))}
            for name, clf in classifiers.items()
        },
    }
    (BASE / 'evaluator_config.json').write_text(json.dumps(config, indent=2) + '\n')

    protocol = f"""# Round 11A evaluation protocol

## Data
- Reuse the frozen Round 10 UCI CTG split: train {len(Xtr)}, validation {len(Xval)}, test {len(Xte)}.
- No re-splitting, no re-shuffling, no test-based model selection.
- Test is used only for exploratory diagnostics; it is not used to choose tau, radius, K, or checkpoints.

## Classifier
- Base: RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample', random_state=42, n_jobs=1).
- Trained on the training split only.
- Primary classifier: {primary_name}; calibration: {config['calibration']}.
- Primary classifier/threshold chosen by validation Youden only; test is not used.
- The validation split is reserved for positive/negative control and threshold/radius rules.
- Rare class = NSP class 2.

## Rare-distance reference
- Reference set: real training samples of class 2.
- Distance: Euclidean in the standardized 21-feature space.
- Self-NN distance excludes the sample itself.
- Primary candidates: q90/q95/q99 of training-rare self-NN distance.
- Selected radius rule: {radius_rule}.
- Selected radius: {radius:.6f}.
- The Round 10 test-rare q95 radius ({radius_test_q95:.6f}) is reproduced for comparison only.

## Rare threshold
- Candidate tau: {', '.join(f'{t:.2f}' for t in THRESHOLDS)}.
- Primary rule: {tau_rule}.
- Selected tau: {tau:.2f}.
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
"""
    (BASE / 'evaluation_protocol.md').write_text(protocol)
    update_progress(progress_current=3, rare_mode_proxy=None,
                    best_result=f'E1 tau={tau:.2f}, radius={radius:.4f}')
    print('E1_OK', 'tau', tau, 'radius', radius, 'elapsed', time.time() - start, flush=True)


if __name__ == '__main__':
    main()
