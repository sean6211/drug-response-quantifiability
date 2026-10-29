#!/usr/bin/env python3
"""ECE and Brier score for the full forest's five-fold out-of-fold scores."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]

from scripts._common import prepared_parser, write_table  # noqa: E402
from scripts.run_random_cv import FULL, OOF_FILE  # noqa: E402
from quantifiability.metrics import brier_score, expected_calibration_error  # noqa: E402


def main() -> None:
    parser = prepared_parser(__doc__)
    parser.add_argument("--oof", type=Path, default=None, help=f"default: <out>/{OOF_FILE}")
    args = parser.parse_args()
    oof = args.oof or Path(args.out) / OOF_FILE
    if not oof.exists():
        raise SystemExit(f"No out-of-fold scores at {oof}. Run: python scripts/run_random_cv.py")
    frame = pd.read_csv(oof)
    y = frame["y"].to_numpy()
    scores = frame[FULL].to_numpy()
    table = pd.DataFrame(
        [
            {
                "ece": expected_calibration_error(y, scores),
                "brier": brier_score(y, scores),
                "n": int(len(y)),
                "model": FULL,
            }
        ]
    )
    write_table(args.out, "calibration.csv", table)


if __name__ == "__main__":
    main()
