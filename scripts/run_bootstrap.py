#!/usr/bin/env python3
"""Compound-cluster bootstrap intervals for the five-fold out-of-fold scores."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]

from scripts._common import prepared_parser, write_table  # noqa: E402
from scripts.run_random_cv import FULL, NO_ANNOTATIONS, OOF_FILE  # noqa: E402
from quantifiability.metrics import cluster_bootstrap  # noqa: E402

MEAN_ACTIVITY = "Mean three-point activity"


def main() -> None:
    parser = prepared_parser(__doc__)
    parser.add_argument("--oof", type=Path, default=None, help=f"default: <out>/{OOF_FILE}")
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    oof = args.oof or Path(args.out) / OOF_FILE
    if not oof.exists():
        raise SystemExit(f"No out-of-fold scores at {oof}. Run: python scripts/run_random_cv.py")
    frame = pd.read_csv(oof)
    y = frame["y"].to_numpy()
    full = frame[FULL].to_numpy()
    mean_activity = frame[MEAN_ACTIVITY].to_numpy()
    no_annotations = frame[NO_ANNOTATIONS].to_numpy()

    def statistic(rows: np.ndarray):
        labels = y[rows]
        if labels.min() == labels.max():
            return None
        model = roc_auc_score(labels, full[rows])
        return (
            model,
            model - roc_auc_score(labels, mean_activity[rows]),
            model - roc_auc_score(labels, no_annotations[rows]),
        )

    point = statistic(np.arange(len(y)))
    print(f"Bootstrapping {frame['Compound_ID'].nunique():,} compounds", flush=True)
    draws = cluster_bootstrap(
        frame["Compound_ID"].astype(str).to_numpy(), statistic, args.resamples, args.seed
    )
    labels = [
        "Full-model AUROC",
        "Full model minus mean three-point activity",
        "Full model minus full model without annotations",
    ]
    rows = []
    for column, (label, estimate) in enumerate(zip(labels, point)):
        low, high = np.percentile(draws[:, column], [2.5, 97.5])
        rows.append(
            {
                "quantity": label,
                "pooled_out_of_fold": float(estimate),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "resamples": int(len(draws)),
                "clusters": int(frame["Compound_ID"].nunique()),
            }
        )
    write_table(args.out, "bootstrap.csv", pd.DataFrame(rows))


if __name__ == "__main__":
    main()
