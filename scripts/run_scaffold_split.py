#!/usr/bin/env python3
"""Random versus Bemis-Murcko scaffold-disjoint evaluation.

Each feature set is a 300-tree class-balanced forest. Scaffold groups keep
every test compound off the scaffolds seen in training. Compounds with no
ring scaffold each receive their own group, so they are not pooled together.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]

from scripts._common import load_cohort, prepared_parser, write_table  # noqa: E402
from quantifiability.evaluate import iter_splits  # noqa: E402
from quantifiability.metrics import ranking_metrics, summarize_folds  # noqa: E402
from quantifiability.models import fit_forest  # noqa: E402


def _evaluate(x, y, columns, n_splits, trees, seed, n_jobs, groups=None) -> dict:
    rows = []
    for train_idx, test_idx in iter_splits(x, y, n_splits, seed, groups=groups):
        scores = fit_forest(
            x[train_idx][:, columns],
            y[train_idx],
            x[test_idx][:, columns],
            trees,
            seed,
            n_jobs,
        )[0]
        rows.append(ranking_metrics(y[test_idx], scores))
    return summarize_folds(rows, ["AUROC", "AUPRC", "Recall@50%"])


def main() -> None:
    parser = prepared_parser(__doc__)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--trees", type=int, default=300)
    args = parser.parse_args()
    cohort = load_cohort(args.data)
    feature_sets = {
        "Full": range(cohort.X.shape[1]),
        "Three-point only": cohort.groups["three_point"],
        "Fingerprints only": cohort.groups["fingerprints"],
    }
    scaffolds = cohort.meta["scaffold"].to_numpy()
    print(f"Bemis-Murcko scaffolds: {len(set(scaffolds)):,}")
    rows = []
    for name, columns in feature_sets.items():
        columns = list(columns)
        for split_name, groups in [("random", None), ("scaffold", scaffolds)]:
            print(f"{name} / {split_name}", flush=True)
            summary = _evaluate(
                cohort.X,
                cohort.y,
                columns,
                args.splits,
                args.trees,
                args.seed,
                args.n_jobs,
                groups=groups,
            )
            summary["feature_set"] = name
            summary["split"] = split_name
            rows.append(summary)
    table = pd.DataFrame(rows)
    write_table(args.out, "scaffold_split.csv", table)


if __name__ == "__main__":
    main()
