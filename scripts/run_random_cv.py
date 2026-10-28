#!/usr/bin/env python3
"""Five-fold comparison of the screening policies in the main performance table.

The principal forest has 500 trees and is fit on the full training fold.
Fingerprint-only and three-point-only forests have 300 trees. Logistic
regression is fit on features standardized within the training fold.
The two-stage score (300 trees per stage) is the operational ranking; on
this split it changes AUROC only in the third decimal relative to the
direct forest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]

from scripts._common import load_cohort, prepared_parser, write_table  # noqa: E402
from quantifiability.evaluate import fold_table, iter_splits  # noqa: E402
from quantifiability.features import REPORTED_MAX  # noqa: E402
from quantifiability.metrics import ranking_metrics  # noqa: E402
from quantifiability.models import fit_forest, fit_logistic, two_stage_scores  # noqa: E402


def main() -> None:
    parser = prepared_parser(__doc__)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--full-trees", type=int, default=500)
    parser.add_argument("--block-trees", type=int, default=300)
    args = parser.parse_args()
    cohort = load_cohort(args.data)
    names = cohort.feature_names
    mean_col = names.index("meas_mean_act")
    max_col = names.index(REPORTED_MAX)
    fingerprint_cols = cohort.groups["fingerprints"]
    three_point_cols = cohort.groups["three_point"]
    records = []
    splits = iter_splits(cohort.X, cohort.y, args.splits, args.seed)
    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"Fold {fold + 1}/{args.splits}", flush=True)
        x_train, x_test = cohort.X[train_idx], cohort.X[test_idx]
        y_train, y_test = cohort.y[train_idx], cohort.y[test_idx]
        rng = np.random.RandomState(args.seed + fold)
        scores = {
            "Random": rng.rand(len(y_test)),
            "Maximum three-point activity": x_test[:, max_col],
            "Mean three-point activity": x_test[:, mean_col],
        }
        scores["Logistic regression"] = fit_logistic(
            x_train, y_train, x_test, args.seed
        )[0]
        scores["Random forest (fingerprints)"] = fit_forest(
            x_train[:, fingerprint_cols],
            y_train,
            x_test[:, fingerprint_cols],
            args.block_trees,
            args.seed,
            args.n_jobs,
        )[0]
        scores["Random forest (three-point)"] = fit_forest(
            x_train[:, three_point_cols],
            y_train,
            x_test[:, three_point_cols],
            args.block_trees,
            args.seed,
            args.n_jobs,
        )[0]
        scores["Random forest (full)"] = fit_forest(
            x_train, y_train, x_test, args.full_trees, args.seed, args.n_jobs
        )[0]
        scores["Two-stage ranking"] = two_stage_scores(
            x_train,
            cohort.y_active[train_idx],
            y_train,
            x_test,
            n_estimators=args.block_trees,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )
        for policy, values in scores.items():
            row = ranking_metrics(y_test, values)
            row["policy"] = policy
            records.append(row)
    write_table(args.out, "random_cv.csv", fold_table(records))


if __name__ == "__main__":
    main()
