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
    return np.argsort(-scores)


def _prefix_length(n: int, fraction: float) -> int:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    return min(n, max(1, int(fraction * n)))


def recall_at_budget(y_true, scores, fraction: float) -> float:
    y_true, scores = _as_arrays(y_true, scores)
    total = float(y_true.sum())
    if total <= 0:
        return float("nan")
    k = _prefix_length(len(y_true), fraction)
    picked = y_true[_order(scores)][:k]
    return float(picked.sum() / total)


def yield_at_budget(y_true, scores, fraction: float) -> float:
    y_true, scores = _as_arrays(y_true, scores)
    k = _prefix_length(len(y_true), fraction)
    picked = y_true[_order(scores)][:k]
    return float(picked.sum() / k)


def retention_prefix_length(y_true, scores, retention: float) -> int:
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
    y_true, scores = _as_arrays(y_true, scores)
    return retention_prefix_length(y_true, scores, retention) / len(y_true)


def ranking_metrics(y_true, scores) -> dict[str, float]:
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
    summary = {}
    for key in metric_keys:
        values = np.array([row[key] for row in fold_rows], dtype=float)
        summary[f"{key}_mean"] = float(np.nanmean(values))
        summary[f"{key}_std"] = float(np.nanstd(values))
    summary["n_folds"] = len(fold_rows)
    return summary


def cohens_d(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return float("nan")
    pooled = ((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (
        a.size + b.size - 2
    )
    if pooled <= 0:
        return float("nan")
    return float((a.mean() - b.mean()) / np.sqrt(pooled))


def cluster_bootstrap(clusters, statistic, n_resamples: int = 2000, seed: int = 42) -> np.ndarray:
    """Resample whole clusters with replacement; `statistic` maps row indices to values.

    Resamples for which `statistic` returns None are dropped.
    """
    members: dict = {}
    for row, cluster in enumerate(clusters):
        members.setdefault(cluster, []).append(row)
    blocks = [np.asarray(rows, dtype=np.int64) for rows in members.values()]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_resamples):
        picked = rng.integers(0, len(blocks), size=len(blocks))
        value = statistic(np.concatenate([blocks[i] for i in picked]))
        if value is not None:
            draws.append(np.atleast_1d(np.asarray(value, dtype=float)))
    if not draws:
        return np.empty((0, 0))
    return np.vstack(draws)
