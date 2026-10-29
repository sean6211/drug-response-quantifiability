#!/usr/bin/env python3
"""Five-fold CV. Also writes out-of-fold scores for run_bootstrap.py and run_calibration.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]

from scripts._common import load_cohort, prepared_parser, write_table  # noqa: E402
from quantifiability.evaluate import fold_table, iter_splits  # noqa: E402
from quantifiability.features import REPORTED_MAX  # noqa: E402
from quantifiability.metrics import ranking_metrics  # noqa: E402
from quantifiability.models import fit_forest, fit_logistic, two_stage_scores  # noqa: E402

FULL = "Random forest (full)"
NO_ANNOTATIONS = "Random forest (full, no screening category or promotion reason)"
OOF_FILE = "random_cv_oof.csv.gz"


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
    kept_cols = np.setdiff1d(np.arange(cohort.X.shape[1]), cohort.groups["annotations"])
    records = []
    out_of_fold: dict[str, np.ndarray] = {}
    fold_ids = np.full(len(cohort.y), -1, dtype=int)
    splits = iter_splits(cohort.X, cohort.y, args.splits, args.seed)
    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"Fold {fold + 1}/{args.splits}", flush=True)
        fold_ids[test_idx] = fold
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
        scores[FULL] = fit_forest(
            x_train, y_train, x_test, args.full_trees, args.seed, args.n_jobs
        )[0]
        scores[NO_ANNOTATIONS] = fit_forest(
            x_train[:, kept_cols],
            y_train,
            x_test[:, kept_cols],
            args.full_trees,
            args.seed,
            args.n_jobs,
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
            out_of_fold.setdefault(policy, np.full(len(cohort.y), np.nan))[test_idx] = values
    write_table(args.out, "random_cv.csv", fold_table(records))
    frame = pd.DataFrame(
        {
            "Compound_ID": cohort.meta["Compound_ID"].astype(str),
            "Assay_ID": cohort.meta["Assay_ID"].astype(str),
            "fold": fold_ids,
            "y": cohort.y,
        }
    )
    for policy, values in out_of_fold.items():
        frame[policy] = values
    path = Path(args.out) / OOF_FILE
    frame.to_csv(path, index=False)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
