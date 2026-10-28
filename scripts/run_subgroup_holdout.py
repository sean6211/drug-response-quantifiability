#!/usr/bin/env python3
"""Subgroup metrics on the first held-out fold."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]

from scripts._common import load_cohort, prepared_parser, write_table  # noqa: E402
from quantifiability.evaluate import iter_splits  # noqa: E402
from quantifiability.metrics import ranking_metrics, recall_at_budget  # noqa: E402
from quantifiability.models import fit_forest, positive_probability, subgroup_booster  # noqa: E402


def _subgroup_rows(y_true, scores, labels, family: str) -> list[dict]:
    rows = []
    for label in pd.unique(labels):
        mask = labels == label
        if mask.sum() < 15 or len(np.unique(y_true[mask])) < 2:
            continue
        metrics = ranking_metrics(y_true[mask], scores[mask])
        rows.append(
            {
                "family": family,
                "subgroup": label,
                "n": int(mask.sum()),
                "positive_rate": float(y_true[mask].mean()),
                "AUROC": metrics["AUROC"],
                "Recall@50%": recall_at_budget(y_true[mask], scores[mask], 0.50),
            }
        )
    return rows


def main() -> None:
    parser = prepared_parser(__doc__)
    parser.add_argument("--trees", type=int, default=500)
    args = parser.parse_args()
    cohort = load_cohort(args.data)
    train_idx, test_idx = next(iter_splits(cohort.X, cohort.y, 5, args.seed))
    print(
        f"Held-out fold: train {len(train_idx):,}, test {len(test_idx):,}",
        flush=True,
    )
    booster = subgroup_booster(args.seed)
    booster.fit(cohort.X[train_idx], cohort.y[train_idx])
    boosted = positive_probability(booster, cohort.X[test_idx])
    y_test = cohort.y[test_idx]
    meta = cohort.meta.iloc[test_idx]
    rows = _subgroup_rows(y_test, boosted, meta["target_class"].to_numpy(), "target_class")
    rows.extend(
        _subgroup_rows(
            y_test,
            boosted,
            meta["screening_category"].to_numpy(),
            "screening_category",
        )
    )
    write_table(args.out, "subgroup_holdout.csv", pd.DataFrame(rows))

    forest_scores = fit_forest(
        cohort.X[train_idx],
        cohort.y[train_idx],
        cohort.X[test_idx],
        args.trees,
        args.seed,
        args.n_jobs,
    )[0]
    k = len(y_test) // 2
    in_budget = np.zeros(len(y_test), dtype=bool)
    in_budget[np.argsort(-forest_scores)[:k]] = True
    quantified = y_test == 1
    missed = meta.loc[quantified & ~in_budget].copy()
    recovered = meta.loc[quantified & in_budget]
    summary = pd.DataFrame(
        [
            {
                "group": "recovered",
                "n": int(len(recovered)),
                "mean_screen_max": float(recovered["scr_observed_max"].mean()),
            },
            {
                "group": "missed",
                "n": int(len(missed)),
                "mean_screen_max": float(missed["scr_observed_max"].mean()) if len(missed) else float("nan"),
            },
        ]
    )
    by_mechanism = (
        missed["mechanism"].value_counts().rename_axis("mechanism").reset_index(name="n_missed")
    )
    write_table(args.out, "missed_quantifiable.csv", summary)
    write_table(args.out, "missed_by_mechanism.csv", by_mechanism)


if __name__ == "__main__":
    main()
