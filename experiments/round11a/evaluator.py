"""Evaluator construction, calibration and deployment for Round 11A.

Primary rule:
  - RandomForest trained on the Round 10 training split only.
  - Sigmoid (Platt) calibration via 5-fold out-of-fold predictions on train.
  - Rare class = NSP class 2.
  - Rare distance reference = 95th percentile of train-rare self NN distance.
  - Rare threshold tau is selected on validation controls, never on test.
"""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report

from common import bootstrap_mean, RARE

THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)


def _rf(n_estimators=300, seed=42):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight='balanced_subsample',
        random_state=seed,
        n_jobs=1,
    )


def fit_uncalibrated(X, y, n_estimators=300, seed=42):
    model = _rf(n_estimators=n_estimators, seed=seed)
    model.fit(X, y)
    return model


def fit_calibrated(X, y, n_estimators=300, seed=42, cv=5, method='sigmoid'):
    base = _rf(n_estimators=n_estimators, seed=seed)
    try:
        model = CalibratedClassifierCV(estimator=base, method=method, cv=cv)
    except TypeError:
        model = CalibratedClassifierCV(base_estimator=base, method=method, cv=cv)
    model.fit(X, y)
    return model


def fit_primary_evaluator(X, y, n_estimators=300, seed=42, cv=5):
    return fit_calibrated(X, y, n_estimators=n_estimators, seed=seed, cv=cv, method='sigmoid')


def fit_named_evaluator(name, X, y, n_estimators=300, seed=42, cv=5):
    if name == 'uncalibrated':
        return fit_uncalibrated(X, y, n_estimators=n_estimators, seed=seed)
    if name == 'calibrated_sigmoid':
        return fit_calibrated(X, y, n_estimators=n_estimators, seed=seed, cv=cv, method='sigmoid')
    if name == 'calibrated_isotonic':
        return fit_calibrated(X, y, n_estimators=n_estimators, seed=seed, cv=cv, method='isotonic')
    raise ValueError(f'Unknown evaluator: {name}')


def prob_rare(model, X):
    prob = model.predict_proba(X)
    idx = list(model.classes_).index(RARE)
    return prob[:, idx]


def predict_all(model, X):
    prob = model.predict_proba(X)
    idx = list(model.classes_).index(RARE)
    pred = np.asarray(model.classes_)[prob.argmax(axis=1)]
    return pred, prob[:, idx], prob.max(axis=1)


def class_report(model, X, y):
    pred, p_rare, conf = predict_all(model, X)
    report = classification_report(y, pred, labels=[0, 1, 2], output_dict=True, zero_division=0)
    return report, pred, p_rare, conf


def distance_reference(X_train, y_train, quantile=0.95):
    rare = np.asarray(X_train)[np.asarray(y_train) == RARE]
    self_dist = NearestNeighbors(n_neighbors=2).fit(rare).kneighbors(rare)[0][:, 1]
    radius = float(np.quantile(self_dist, quantile))
    nn = NearestNeighbors(n_neighbors=1).fit(rare)
    return nn, radius, self_dist


def distance_to_rare(nn, X):
    return nn.kneighbors(np.asarray(X))[0][:, 0]


def pass_rates(p_rare, distance, radius, tau):
    pos = np.asarray(p_rare) >= tau
    dist = np.asarray(distance) <= radius
    return {
        'classifier_pass': float(np.mean(pos)),
        'distance_pass': float(np.mean(dist)),
        'joint_pass': float(np.mean(pos & dist)),
    }


def threshold_table(p_rare, distance, radius, thresholds=THRESHOLDS):
    rows = []
    for tau in thresholds:
        r = pass_rates(p_rare, distance, radius, tau)
        rows.append({'threshold': float(tau), **r})
    return rows


def threshold_rows(p_rare_val, y_val, thresholds=THRESHOLDS):
    y_val = np.asarray(y_val)
    pos_mask = y_val == RARE
    neg_mask = y_val != RARE
    table = []
    for tau in thresholds:
        pos = float(np.mean(p_rare_val[pos_mask] >= tau))
        neg = float(np.mean(p_rare_val[neg_mask] >= tau))
        table.append({'threshold': float(tau), 'positive_pass': pos, 'negative_pass': neg,
                      'youden': pos - neg})
    return table


def youden_threshold(p_rare_val, y_val, thresholds=THRESHOLDS):
    table = threshold_rows(p_rare_val, y_val, thresholds)
    best = max(table, key=lambda r: (r['youden'], r['positive_pass'], -r['negative_pass']))
    return best, table


def select_threshold(p_rare_val, y_val, thresholds=THRESHOLDS,
                     min_positive=0.90, max_negative=0.05):
    """Select tau on validation controls.

    Primary: validation Youden-best tau (positive - negative), a standard
    validation-only rule. The stricter 90%/5% rule is reported as sensitivity.
    """
    table = threshold_rows(p_rare_val, y_val, thresholds)
    best_j = max(table, key=lambda r: (r['youden'], r['positive_pass'], -r['negative_pass']))
    eligible = [r for r in table if r['positive_pass'] >= min_positive
                and r['negative_pass'] <= max_negative]
    if eligible:
        rule = f'validation Youden-best tau={best_j["threshold"]:.2f}; strict 90/5 also feasible'
    else:
        rule = (f'validation Youden-best tau={best_j["threshold"]:.2f}; '
                f'no tau met positive>={min_positive:.2f}, negative<={max_negative:.2f}')
    return float(best_j['threshold']), rule, table, best_j


def evaluate_generated(generated, model, nn_rare, radius, tau):
    generated = np.asarray(generated)
    pred, p_rare, conf = predict_all(model, generated)
    dist = distance_to_rare(nn_rare, generated)
    trusted = (p_rare >= tau) & (dist <= radius)
    return {
        'n': int(len(generated)),
        'rare_proxy': float(np.mean(trusted)),
        'assigned_rare_fraction': float(np.mean(pred == RARE)),
        'high_conf_rare_fraction': float(np.mean(p_rare >= tau)),
        'classifier_rare_prob_mean': float(np.mean(p_rare)),
        'distance_to_rare_mean': float(np.mean(dist)),
        'distance_pass_fraction': float(np.mean(dist <= radius)),
        'trusted_rare_count': int(np.sum(trusted)),
        'assigned_label_counts': {str(c): int(np.sum(pred == c)) for c in (0, 1, 2)},
    }


def bootstrap_pass(values, n_boot=1000, seed=12345):
    values = np.asarray(values, dtype=np.float64)
    return bootstrap_mean(values, n_boot=n_boot, seed=seed)
