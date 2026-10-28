#!/usr/bin/env python3
"""Rank a matched table by screen_max. Columns: endpoint, screen_max, quantified."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]

from scripts._common import DEFAULT_OUT, write_table  # noqa: E402
from quantifiability.evaluate import iter_splits  # noqa: E402
from quantifiability.metrics import ranking_metrics, summarize_folds  # noqa: E402


def _score_splits(frame: pd.DataFrame, groups, n_splits: int, seed: int) -> dict:
    rows = []
    y = frame["quantified"].to_numpy()
    scores = frame["screen_max"].to_numpy(dtype=float)
    x = scores.reshape(-1, 1)
    for _train_idx, test_idx in iter_splits(x, y, n_splits, seed, groups=groups):
        rows.append(ranking_metrics(y[test_idx], scores[test_idx]))
    return summarize_folds(rows, ["AUROC", "AUPRC", "Recall@50%"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = pd.read_csv(args.table)
    required = {"screen_max", "quantified", "endpoint"}
    missing = required.difference(frame.columns)
    if missing:
        raise SystemExit(f"table is missing columns: {sorted(missing)}")
    random_split = _score_splits(frame, None, args.splits, args.seed)
    random_split["design"] = "random_five_fold"
    grouped = _score_splits(
        frame, frame["endpoint"].to_numpy(), args.splits, args.seed
    )
    grouped["design"] = "endpoint_held_out"
    write_table(args.out, "external_amplitude.csv", pd.DataFrame([random_split, grouped]))


if __name__ == "__main__":
    main()
