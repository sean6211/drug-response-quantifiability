"""Estimators named in the methods.

The principal model is a class-balanced random forest fit on the whole
training fold. Isolated blocks, the scaffold comparison, mechanism holdout,
and each stage of the operational ranking use the same forest with the tree
count stated for that analysis. Logistic regression is the linear comparator
and is fit on training-fold standardized features.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def positive_probability(model, features) -> np.ndarray:
    """P(class = 1). A one-class training fold returns zeros."""
    probabilities = model.predict_proba(features)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(features), dtype=np.float64)
    return probabilities[:, classes.index(1)]


def forest(n_estimators: int, seed: int, n_jobs: int = -1) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        class_weight="balanced",
        n_jobs=n_jobs,
        random_state=seed,
    )


def fit_forest(x_train, y_train, x_test, n_estimators: int, seed: int, n_jobs: int = -1):
    model = forest(n_estimators, seed, n_jobs=n_jobs)
    model.fit(x_train, y_train)
    return positive_probability(model, x_test), model


def fit_logistic(x_train, y_train, x_test, seed: int):
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=seed,
    )
    model.fit(x_train_s, y_train)
    return positive_probability(model, x_test_s), model


def subgroup_booster(seed: int) -> GradientBoostingClassifier:
    """Held-out subgroup table only. The main comparison is the random forest."""
    return GradientBoostingClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=seed,
    )


def two_stage_scores(
    x_train,
    y_active_train,
    y_quant_train,
    x_test,
    n_estimators: int = 300,
    seed: int = 42,
    n_jobs: int = -1,
) -> np.ndarray:
    """P(active) times P(quantifiable | active).

    A reportable potency occurs only among actives, so the product ranks
    pairs by the chance that profiling returns a potency. Stage 2 is fit
    only on training pairs that were active in the 11-point experiment.
    """
    stage_one = forest(n_estimators, seed, n_jobs=n_jobs)
    stage_one.fit(x_train, y_active_train)
    p_active = positive_probability(stage_one, x_test)
    active = np.asarray(y_active_train) == 1
    quant = np.asarray(y_quant_train)
    if active.sum() < 2 or len(np.unique(quant[active])) < 2:
        return p_active
    stage_two = forest(n_estimators, seed, n_jobs=n_jobs)
    stage_two.fit(x_train[active], quant[active])
    return p_active * positive_probability(stage_two, x_test)
