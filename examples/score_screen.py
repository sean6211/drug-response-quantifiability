#!/usr/bin/env python3
"""Rank a new three-point screen by the 16 screening features.

The default ranking is the mean of the measured activities, which is the
principal screening heuristic. ``--rank-by scr_observed_max`` ranks by the
maximum of the per-concentration means instead. Both values are computed
from the point rows in the file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from quantifiability.features import screen_feature_frame  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path)
    parser.add_argument(
        "--rank-by",
        choices=["meas_mean_act", "scr_observed_max"],
        default="meas_mean_act",
    )
    args = parser.parse_args()
    points = pd.read_csv(args.measurements)
    features = screen_feature_frame(points)
    ranked = features.sort_values(args.rank_by, ascending=False)
    columns = ["compound_id", "assay_id", "meas_mean_act", "scr_observed_max", "meas_slope_proxy"]
    print(ranked[columns].to_string(index=False))
    print(
        "\nHigher rows are the pairs a mean-activity or screening-maximum "
        "rule would profile first. A fitted forest consumes the same 16 columns "
        "plus, when available, fingerprints and assay context."
    )


if __name__ == "__main__":
    main()
