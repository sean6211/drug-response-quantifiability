#!/usr/bin/env python3
"""Gini importance summed by feature group."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]

from scripts._common import load_cohort, prepared_parser, write_table  # noqa: E402
from quantifiability.models import forest  # noqa: E402

GROUP_ORDER = ["three_point", "fingerprints", "profiles", "metadata"]


def main() -> None:
    parser = prepared_parser(__doc__)
    parser.add_argument("--trees", type=int, default=500)
    args = parser.parse_args()
    cohort = load_cohort(args.data)
    model = forest(args.trees, args.seed, n_jobs=args.n_jobs)
    print(f"Fitting a {args.trees}-tree forest on {len(cohort.y):,} pairs", flush=True)
    model.fit(cohort.X, cohort.y)
    importance = model.feature_importances_
    rows = []
    for name in GROUP_ORDER:
        indices = cohort.groups[name]
        rows.append(
            {
                "feature_group": name,
                "n_features": int(len(indices)),
                "importance_sum": float(importance[indices].sum()),
                "importance_mean": float(importance[indices].mean()),
            }
        )
    write_table(args.out, "group_importance.csv", pd.DataFrame(rows))


if __name__ == "__main__":
    main()
