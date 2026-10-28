#!/usr/bin/env python3
"""Leave-one-mechanism-family-out."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]

from scripts._common import load_cohort, prepared_parser, write_table  # noqa: E402
from quantifiability.features import MECHANISMS  # noqa: E402
from quantifiability.metrics import ranking_metrics  # noqa: E402
from quantifiability.models import fit_forest  # noqa: E402


def main() -> None:
    parser = prepared_parser(__doc__)
    parser.add_argument("--trees", type=int, default=300)
    args = parser.parse_args()
    cohort = load_cohort(args.data)
    feature_sets = {
        "Fingerprints": cohort.groups["fingerprints"],
        "Fingerprints + three-point": np.concatenate(
            [cohort.groups["fingerprints"], cohort.groups["three_point"]]
        ),
        "Fingerprints + three-point + mechanism": np.concatenate(
            [
                cohort.groups["fingerprints"],
                cohort.groups["three_point"],
                cohort.groups["mechanism"],
            ]
        ),
    }
    mechanisms = cohort.meta["mechanism"].to_numpy()
    rows = []
    for held_out in MECHANISMS:
        test = mechanisms == held_out
        train = ~test
        print(
            f"{held_out}: train {int(train.sum()):,}, test {int(test.sum()):,}",
            flush=True,
        )
        if test.sum() < 20 or len(np.unique(cohort.y[train])) < 2:
            continue
        for name, columns in feature_sets.items():
            scores = fit_forest(
                cohort.X[train][:, columns],
                cohort.y[train],
                cohort.X[test][:, columns],
                args.trees,
                args.seed,
                args.n_jobs,
            )[0]
            row = ranking_metrics(cohort.y[test], scores)
            row["held_out_mechanism"] = held_out
            row["feature_set"] = name
            row["n_test"] = int(test.sum())
            row["positive_rate"] = float(cohort.y[test].mean())
            rows.append(row)
    detail = pd.DataFrame(rows)
    means = (
        detail.groupby("feature_set", sort=False)[["AUROC", "AUPRC"]]
        .mean()
        .reset_index()
    )
    write_table(args.out, "mechanism_holdout.csv", detail)
    write_table(args.out, "mechanism_holdout_mean.csv", means)


if __name__ == "__main__":
    main()
