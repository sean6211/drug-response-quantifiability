#!/usr/bin/env python3
"""Leakage-safe rolling evaluation across EvE Bio releases.

For each test release, training pairs are promoted pairs from earlier
releases, and the nine compound and assay profile columns are recomputed
from screening results strictly before that release. AUROC is the direct
500-tree forest. Recall at 50% and the budget that retains 90% of potencies
use the two-stage ranking with 300 trees per stage, compared with ranking
by the reported screening maximum.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]

from scripts._common import load_cohort, prepared_parser, write_table  # noqa: E402
from quantifiability.evaluate import release_label  # noqa: E402
from quantifiability.features import REPORTED_MAX  # noqa: E402
from quantifiability.metrics import (  # noqa: E402
    ranking_metrics,
    recall_at_budget,
    retention_prefix_length,
)
from quantifiability.models import fit_forest, two_stage_scores  # noqa: E402


def main() -> None:
    parser = prepared_parser(__doc__)
    parser.add_argument("--direct-trees", type=int, default=500)
    parser.add_argument("--stage-trees", type=int, default=300)
    parser.add_argument("--min-train", type=int, default=500)
    parser.add_argument("--min-test", type=int, default=200)
    args = parser.parse_args()
    cohort = load_cohort(args.data)
    release = cohort.meta["release"].to_numpy(dtype=float)
    max_col = cohort.feature_names.index(REPORTED_MAX)
    rows = []
    releases = []
    for value in sorted(float(item) for item in release if np.isfinite(item)):
        if any(np.isclose(value, seen) for seen in releases):
            continue
        releases.append(value)
    for value in releases:
        train = release < value - 1e-9
        test = np.isclose(release, value)
        if train.sum() < args.min_train or test.sum() < args.min_test:
            continue
        if cohort.y[test].sum() < 10 or cohort.y[train].sum() < 10:
            continue
        print(f"Release {release_label(value)}", flush=True)
        features = cohort.with_historical_profiles(value)
        direct = fit_forest(
            features[train],
            cohort.y[train],
            features[test],
            args.direct_trees,
            args.seed,
            args.n_jobs,
        )[0]
        staged = two_stage_scores(
            features[train],
            cohort.y_active[train],
            cohort.y[train],
            features[test],
            n_estimators=args.stage_trees,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )
        screening = features[test, max_col]
        y_test = cohort.y[test]
        direct_metrics = ranking_metrics(y_test, direct)
        kept = retention_prefix_length(y_test, staged, 0.90)
        screening_kept = retention_prefix_length(y_test, screening, 0.90)
        n_test = int(test.sum())
        rows.append(
            {
                "release": release_label(value),
                "release_value": float(value),
                "train_n": int(train.sum()),
                "test_n": n_test,
                "quantifiable": int(y_test.sum()),
                "positive_rate": float(y_test.mean()),
                "auroc": direct_metrics["AUROC"],
                "auprc": direct_metrics["AUPRC"],
                "recall50_model": recall_at_budget(y_test, staged, 0.50),
                "recall50_screening": recall_at_budget(y_test, screening, 0.50),
                "budget90_model": kept / n_test,
                "budget90_screening": screening_kept / n_test,
                "experiments_withheld_at_90": n_test - kept,
            }
        )
    table = pd.DataFrame(rows).drop(columns=["release_value"])
    withheld = int(table["experiments_withheld_at_90"].sum()) if len(table) else 0
    tested = int(table["test_n"].sum()) if len(table) else 0
    write_table(args.out, "rolling_policy.csv", table)
    if tested:
        print(
            f"Across {len(table)} releases, retaining 90% of potencies withholds "
            f"{withheld:,} of {tested:,} profiling experiments "
            f"({withheld / tested:.1%})."
        )


if __name__ == "__main__":
    main()
