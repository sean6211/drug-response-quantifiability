#!/usr/bin/env python3
"""ECE and Brier score for two-stage out-of-fold probabilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]

from scripts._common import load_cohort, prepared_parser, write_table  # noqa: E402
from quantifiability.evaluate import iter_splits  # noqa: E402
from quantifiability.metrics import brier_score, expected_calibration_error  # noqa: E402
from quantifiability.models import two_stage_scores  # noqa: E402


def main() -> None:
    parser = prepared_parser(__doc__)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--trees", type=int, default=300)
    args = parser.parse_args()
    cohort = load_cohort(args.data)
    out_of_fold = np.full(len(cohort.y), np.nan)
    for fold, (train_idx, test_idx) in enumerate(
        iter_splits(cohort.X, cohort.y, args.splits, args.seed)
    ):
        print(f"Fold {fold + 1}/{args.splits}", flush=True)
        out_of_fold[test_idx] = two_stage_scores(
            cohort.X[train_idx],
            cohort.y_active[train_idx],
            cohort.y[train_idx],
            cohort.X[test_idx],
            n_estimators=args.trees,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )
    table = pd.DataFrame(
        [
            {
                "ece": expected_calibration_error(cohort.y, out_of_fold),
                "brier": brier_score(cohort.y, out_of_fold),
                "n": int(len(cohort.y)),
                "trees_per_stage": args.trees,
            }
        ]
    )
    write_table(args.out, "calibration.csv", table)


if __name__ == "__main__":
    main()
