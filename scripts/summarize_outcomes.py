#!/usr/bin/env python3
"""Outcome counts, amplitude, and mechanism failure for promoted pairs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]

from scripts._common import DEFAULT_EVE, DEFAULT_OUT, write_table  # noqa: E402
from quantifiability.outcomes import (  # noqa: E402
    active_poor_count,
    amplitude_by_outcome,
    mechanism_failure,
    outcome_counts,
    promoted_frame,
)

COLUMNS = [
    "progressed",
    "Final_Category",
    "Mechanism",
    "Scr_Observed_Max",
    "Prof_Max",
    "Prof_Category_Detailed",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detailed",
        type=Path,
        default=DEFAULT_EVE / "detailed_result_table.csv",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    detailed = pd.read_csv(args.detailed, usecols=COLUMNS, low_memory=False)
    promoted = promoted_frame(detailed)
    counts = outcome_counts(promoted)
    amplitude = amplitude_by_outcome(promoted)
    failure = mechanism_failure(promoted)
    poor = active_poor_count(promoted)
    write_table(args.out, "outcome_counts.csv", counts)
    write_table(args.out, "amplitude_by_outcome.csv", amplitude)
    write_table(args.out, "mechanism_failure.csv", failure)
    print(
        "Active-Poor among active-unquantifiable pairs: "
        f"{poor['active_poor']:,} / {poor['active_unquantifiable']:,} "
        f"({poor['active_poor_fraction']:.1%})"
    )


if __name__ == "__main__":
    main()
