"""Supervised pairwise match model.

A gradient-boosted decision tree (scikit-learn HistGradientBoostingClassifier,
BSD-3 licensed — well within the MIT/Apache <=8B-param rule). GBDTs handle the
mixed-scale, non-linear similarity features here better than logistic
regression and need no feature scaling. Persisted with joblib.
"""
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


def build_model(seed=42):
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.1,
        max_iter=400,
        max_leaf_nodes=63,
        min_samples_leaf=100,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=seed,
    )


def train(X, y, seed=42):
    model = build_model(seed)
    model.fit(X, y)
    return model


def predict_proba(model, X):
    if len(X) == 0:
        return np.zeros(0, dtype=np.float64)
    return model.predict_proba(X)[:, 1]


def save(model, path):
    joblib.dump(model, path)


def load(path):
    return joblib.load(path)


def feature_importance(model, feature_names, X_sample, y_sample=None, n_repeats=3, seed=42):
    """Permutation importance on a sample (works for any sklearn estimator)."""
    from sklearn.inspection import permutation_importance
    if y_sample is None:
        return None
    r = permutation_importance(
        model, X_sample, y_sample, n_repeats=n_repeats,
        random_state=seed, scoring="average_precision",
    )
    order = np.argsort(r.importances_mean)[::-1]
    return [(feature_names[i], float(r.importances_mean[i])) for i in order]
