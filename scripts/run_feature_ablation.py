#!/usr/bin/env python3
"""Feature-block ablation. Isolated blocks are 5-fold; nested models are 3-fold."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]

from scripts._common import load_cohort, prepared_parser, write_table  # noqa: E402
from quantifiability.evaluate import iter_splits  # noqa: E402
from quantifiability.metrics import ranking_metrics, summarize_folds  # noqa: E402
from quantifiability.models import fit_forest  # noqa: E402


def _evaluate(x, y, columns, n_splits, trees, seed, n_jobs) -> dict:
    rows = []
    for train_idx, test_idx in iter_splits(x, y, n_splits, seed):
        scores = fit_forest(
            x[train_idx][:, columns],
            y[train_idx],
            x[test_idx][:, columns],
            trees,
            seed,
            n_jobs,
        )[0]
        rows.append(ranking_metrics(y[test_idx], scores))
    summary = summarize_folds(rows, ["AUROC", "AUPRC", "Recall@50%"])
    summary["n_features"] = int(len(columns))
    return summary


def main() -> None:
    parser = prepared_parser(__doc__)
    parser.add_argument("--trees", type=int, default=300)
    args = parser.parse_args()
    cohort = load_cohort(args.data)
    y = cohort.y
    fingerprints = cohort.groups["fingerprints"]
    three_point = cohort.groups["three_point"]
    profiles = cohort.groups["profiles"]
    isolated = {
        "Three-point only": three_point,
        "Fingerprints only": fingerprints,
        "Full model": np.arange(cohort.X.shape[1]),
    }
    isolated_rows = []
    for name, columns in isolated.items():
        print(f"Isolated: {name}", flush=True)
        row = _evaluate(cohort.X, y, columns, 5, args.trees, args.seed, args.n_jobs)
        row["feature_set"] = name
        row["design"] = "isolated_five_fold"
        isolated_rows.append(row)

    nested_sets = [
        ("Three-point only", three_point),
        ("+ fingerprints", np.concatenate([three_point, fingerprints])),
        ("+ profiles", np.concatenate([three_point, fingerprints, profiles])),
        ("+ metadata (full)", np.arange(cohort.X.shape[1])),
    ]
    nested_rows = []
    previous = None
    for name, columns in nested_sets:
        print(f"Nested: {name}", flush=True)
        row = _evaluate(cohort.X, y, columns, 3, args.trees, args.seed, args.n_jobs)
        row["feature_set"] = name
        row["design"] = "nested_three_fold"
        row["delta_auroc"] = (
            float("nan") if previous is None else row["AUROC_mean"] - previous
        )
        previous = row["AUROC_mean"]
        nested_rows.append(row)
    table = pd.DataFrame(isolated_rows + nested_rows)
    front = ["design", "feature_set", "n_features", "AUROC_mean", "AUROC_std", "delta_auroc"]
    remaining = [column for column in table.columns if column not in front]
    write_table(args.out, "feature_ablation.csv", table[front + remaining])


if __name__ == "__main__":
    main()
