"""Fixed Round 11A evaluator reused for Round 11B-pre.

The calibration, threshold and distance radius are loaded from Round 11A and
must not be retuned from Round 11B-pre generation results.
"""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors

from common import RARE


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


def fit_named_evaluator(name, X, y, n_estimators=300, seed=42, cv=5):
    if name == 'uncalibrated':
        return fit_uncalibrated(X, y, n_estimators=n_estimators, seed=seed)
    if name == 'calibrated_sigmoid':
        return fit_calibrated(X, y, n_estimators=n_estimators, seed=seed, cv=cv,
                              method='sigmoid')
    if name == 'calibrated_isotonic':
        return fit_calibrated(X, y, n_estimators=n_estimators, seed=seed, cv=cv,
                              method='isotonic')
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


def distance_reference(X_train, y_train, quantile=0.95):
    rare = np.asarray(X_train)[np.asarray(y_train) == RARE]
    self_dist = NearestNeighbors(n_neighbors=2).fit(rare).kneighbors(rare)[0][:, 1]
    radius = float(np.quantile(self_dist, quantile))
    nn = NearestNeighbors(n_neighbors=1).fit(rare)
    return nn, radius
