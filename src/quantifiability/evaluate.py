"""Shared splitters for the main-text comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

from quantifiability.metrics import ranking_metrics, summarize_folds

METRIC_KEYS = ["AUROC", "AUPRC", "Recall@50%", "Yield@25%"]


def iter_splits(x, y, n_splits: int, seed: int, groups=None):
    if groups is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        yield from splitter.split(x, y)
        return
    splitter = GroupKFold(n_splits=n_splits)
    yield from splitter.split(x, y, groups=groups)


def fold_table(records: list[dict]) -> pd.DataFrame:
    """Mean and standard deviation of each policy, one row per policy."""
    frame = pd.DataFrame(records)
    rows = []
    for policy, group in frame.groupby("policy", sort=False):
        summary = summarize_folds(group.to_dict("records"), METRIC_KEYS)
        summary["policy"] = policy
        rows.append(summary)
    columns = ["policy"]
    for key in METRIC_KEYS:
        columns.extend([f"{key}_mean", f"{key}_std"])
    columns.append("n_folds")
    return pd.DataFrame(rows)[columns]


def release_label(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.1f}"


def metric_row(y_true, scores) -> dict:
    return ranking_metrics(y_true, scores)
