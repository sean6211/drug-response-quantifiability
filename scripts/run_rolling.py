#!/usr/bin/env python3
"""Rolling release evaluation of the full forest. Profiles are recomputed from earlier releases only."""

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
from quantifiability.models import fit_forest  # noqa: E402


def main() -> None:
    parser = prepared_parser(__doc__)
    parser.add_argument("--trees", type=int, default=500)
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
        model = fit_forest(
            features[train],
            cohort.y[train],
            features[test],
            args.trees,
            args.seed,
            args.n_jobs,
        )[0]
        screening = features[test, max_col]
        y_test = cohort.y[test]
        metrics = ranking_metrics(y_test, model)
        model_kept = retention_prefix_length(y_test, model, 0.90)
        screening_kept = retention_prefix_length(y_test, screening, 0.90)
        n_test = int(test.sum())
        rows.append(
            {
                "release": release_label(value),
                "train_n": int(train.sum()),
                "test_n": n_test,
                "quantifiable": int(y_test.sum()),
                "positive_rate": float(y_test.mean()),
                "auroc": metrics["AUROC"],
                "auprc": metrics["AUPRC"],
                "recall50_model": recall_at_budget(y_test, model, 0.50),
                "recall50_screening": recall_at_budget(y_test, screening, 0.50),
                "budget90_model": model_kept / n_test,
                "budget90_screening": screening_kept / n_test,
                "kept90_model": int(model_kept),
                "kept90_screening": int(screening_kept),
            }
        )
    table = pd.DataFrame(rows)
    write_table(args.out, "rolling_policy.csv", table)
    if not len(table):
        return
    tested = int(table["test_n"].sum())
    model_kept = int(table["kept90_model"].sum())
    screening_kept = int(table["kept90_screening"].sum())
    summary = pd.DataFrame(
        [
            {
                "releases": int(len(table)),
                "tested": tested,
                "kept90_model": model_kept,
                "kept90_screening": screening_kept,
                "budget90_model": model_kept / tested,
                "budget90_screening": screening_kept / tested,
                "fewer_than_screening": screening_kept - model_kept,
                "reduction_vs_screening": (screening_kept - model_kept) / screening_kept,
                "fewer_than_profiling_all": tested - model_kept,
                "releases_smaller_budget": int(
                    (table["kept90_model"] < table["kept90_screening"]).sum()
                ),
            }
        ]
    )
    write_table(args.out, "rolling_summary.csv", summary)


if __name__ == "__main__":
    main()
