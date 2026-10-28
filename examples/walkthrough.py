#!/usr/bin/env python3
"""Run the screening representation and a small forest on synthetic pairs.

This is a tutorial, not the paper estimate. The paper comparisons read the
prepared EvE Bio matrix and use the tree counts in the methods.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from quantifiability.evaluate import iter_splits  # noqa: E402
from quantifiability.metrics import ranking_metrics  # noqa: E402
from quantifiability.models import fit_forest  # noqa: E402
from quantifiability.synthetic import synthetic_cohort  # noqa: E402


def main() -> None:
    cohort = synthetic_cohort()
    print(
        f"Synthetic promoted pairs: {len(cohort.y)}  "
        f"features: {cohort.X.shape[1]}  "
        f"quantifiable: {int(cohort.y.sum())}"
    )
    columns = cohort.groups["three_point"]
    folds = []
    for train_idx, test_idx in iter_splits(cohort.X, cohort.y, n_splits=3, seed=0):
        scores = fit_forest(
            cohort.X[train_idx][:, columns],
            cohort.y[train_idx],
            cohort.X[test_idx][:, columns],
            n_estimators=25,
            seed=0,
            n_jobs=1,
        )[0]
        folds.append(ranking_metrics(cohort.y[test_idx], scores))
    auroc = sum(row["AUROC"] for row in folds) / len(folds)
    recall = sum(row["Recall@50%"] for row in folds) / len(folds)
    print(f"Three-point forest, 3-fold demo: AUROC {auroc:.3f}  Recall@50% {recall:.3f}")
    print("Feature groups:", ", ".join(f"{name}={len(idx)}" for name, idx in cohort.groups.items()))


if __name__ == "__main__":
    main()
