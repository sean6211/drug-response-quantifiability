#!/usr/bin/env python3
"""Build the promoted-pair matrix from a local EvE Bio release.

The release files stay outside this repository. Pass the directory that
contains summary_result_table.csv, detailed_result_table.csv,
observed_points_table.csv, compound_table.csv, and assay_table.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]

from scripts._common import DEFAULT_DATA, DEFAULT_EVE  # noqa: E402
from quantifiability.dataset import read_evebio  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evebio", type=Path, default=DEFAULT_EVE)
    parser.add_argument("--out", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    cohort = read_evebio(args.evebio)
    cohort.save(args.out)
    print(
        f"Prepared {cohort.X.shape[0]:,} pairs x {cohort.X.shape[1]:,} features "
        f"at {args.out}"
    )
    print(f"Quantifiable pairs: {int(cohort.y.sum()):,} ({cohort.y.mean():.1%})")


if __name__ == "__main__":
    main()
