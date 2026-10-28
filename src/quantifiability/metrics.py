"""Ranking metrics for a profiling budget.

Recall at fraction ``f`` uses the top ``floor(f * n)`` pairs. The budget
required to retain a fraction ``r`` of reportable potencies is the shortest
prefix of the ranking whose captured potencies reach that fraction.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _as_arrays(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    if y_true.shape[0] != scores.shape[0]:
        raise ValueError("y_true and scores must have the same length")
    if y_true.size == 0:
        raise ValueError("y_true is empty")
    return y_true, scores


def _order(scores: np.ndarray) -> np.ndarray:
    # Default argsort matches the analysis notebook, including tie order.
    return np.argsort(-scores)


def _prefix_length(n: int, fraction: float) -> int:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    return min(n, max(1, int(fraction * n)))


def recall_at_budget(y_true, scores, fraction: float) -> float:
    """Fraction of positives captured in the top ``fraction`` of the ranking."""
    y_true, scores = _as_arrays(y_true, scores)
    total = float(y_true.sum())
    if total <= 0:
        return float("nan")
    k = _prefix_length(len(y_true), fraction)
    picked = y_true[_order(scores)][:k]
    return float(picked.sum() / total)


def yield_at_budget(y_true, scores, fraction: float) -> float:
    """Precision of the top ``fraction`` of the ranking."""
    y_true, scores = _as_arrays(y_true, scores)
    k = _prefix_length(len(y_true), fraction)
    picked = y_true[_order(scores)][:k]
    return float(picked.sum() / k)


def retention_prefix_length(y_true, scores, retention: float) -> int:
    """Smallest prefix that keeps ``retention`` of all positives."""
    y_true, scores = _as_arrays(y_true, scores)
    if not 0 < retention <= 1:
        raise ValueError("retention must be in (0, 1]")
    total = float(y_true.sum())
    if total <= 0:
        return len(y_true)
    cumulative = np.cumsum(y_true[_order(scores)].astype(float))
    index = int(np.searchsorted(cumulative, retention * total, side="left"))
    return min(len(y_true), index + 1)


def budget_at_retention(y_true, scores, retention: float) -> float:
    """Prefix length as a fraction of the test set."""
    y_true, scores = _as_arrays(y_true, scores)
    return retention_prefix_length(y_true, scores, retention) / len(y_true)


def ranking_metrics(y_true, scores) -> dict[str, float]:
    """AUROC, AUPRC, recall at 50%, and yield at 25%."""
    y_true, scores = _as_arrays(y_true, scores)
    if len(np.unique(y_true)) < 2:
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = float(roc_auc_score(y_true, scores))
        auprc = float(average_precision_score(y_true, scores))
    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "Recall@50%": recall_at_budget(y_true, scores, 0.50),
        "Yield@25%": yield_at_budget(y_true, scores, 0.25),
    }


def brier_score(y_true, probabilities) -> float:
    y_true, probabilities = _as_arrays(y_true, probabilities)
    return float(np.mean((probabilities - y_true.astype(float)) ** 2))


def expected_calibration_error(y_true, probabilities, n_bins: int = 10) -> float:
    """Absolute-gap ECE on ``n_bins`` equal-width probability bins."""
    y_true, probabilities = _as_arrays(y_true, probabilities)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)
    error = 0.0
    for bin_index in range(n_bins):
        lower, upper = edges[bin_index], edges[bin_index + 1]
        if bin_index == n_bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        count = int(mask.sum())
        if count == 0:
            continue
        gap = abs(float(y_true[mask].mean()) - float(probabilities[mask].mean()))
        error += (count / total) * gap
    return float(error)


def summarize_folds(fold_rows: list[dict], metric_keys: list[str]) -> dict[str, float]:
    """Mean and NumPy standard deviation (ddof=0) across folds."""
    summary = {}
    for key in metric_keys:
        values = np.array([row[key] for row in fold_rows], dtype=float)
        summary[f"{key}_mean"] = float(np.nanmean(values))
        summary[f"{key}_std"] = float(np.nanstd(values))
    summary["n_folds"] = len(fold_rows)
    return summary
