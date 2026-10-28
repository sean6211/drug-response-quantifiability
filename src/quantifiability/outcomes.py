"""Quantifiability label and the amplitude summaries in the main text."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantifiability.dataset import as_flag
from quantifiability.features import QUANTIFIED_LABEL

OUTCOME_ORDER = [
    "Active - Quantified (11pt)",
    "Active - Unquantified (11pt)",
    "Inactive (11pt)",
]


def promoted_frame(detailed: pd.DataFrame) -> pd.DataFrame:
    frame = detailed.loc[as_flag(detailed["progressed"])].copy()
    frame["Final_Category"] = frame["Final_Category"].astype(str)
    frame["quantified"] = (frame["Final_Category"] == QUANTIFIED_LABEL).astype(int)
    frame["active"] = frame["Final_Category"].str.startswith("Active").astype(int)
    frame["Scr_Observed_Max"] = pd.to_numeric(frame["Scr_Observed_Max"], errors="coerce")
    frame["Prof_Max"] = pd.to_numeric(frame["Prof_Max"], errors="coerce")
    return frame


def outcome_counts(promoted: pd.DataFrame) -> pd.DataFrame:
    counts = promoted["Final_Category"].value_counts()
    rows = []
    for label in OUTCOME_ORDER:
        rows.append({"outcome": label, "n": int(counts.get(label, 0))})
    known = set(OUTCOME_ORDER)
    other = int(sum(count for label, count in counts.items() if label not in known))
    if other:
        rows.append({"outcome": "other", "n": other})
    return pd.DataFrame(rows)


def amplitude_by_outcome(promoted: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in OUTCOME_ORDER:
        group = promoted.loc[promoted["Final_Category"] == label]
        rows.append(
            {
                "outcome": label,
                "n": int(len(group)),
                "mean_screen_max": float(group["Scr_Observed_Max"].mean()),
                "mean_profile_max": float(group["Prof_Max"].mean()),
                "profile_max_below_40": float((group["Prof_Max"] < 40).mean()) if len(group) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def mechanism_failure(promoted: pd.DataFrame) -> pd.DataFrame:
    """Hill-fit failure among pairs that were active at 11 concentrations."""
    active = promoted.loc[promoted["active"] == 1]
    rows = []
    for mechanism, group in active.groupby("Mechanism", sort=True):
        quantified = int(group["quantified"].sum())
        n_active = int(len(group))
        rows.append(
            {
                "mechanism": mechanism,
                "n_active": n_active,
                "quantifiable": quantified,
                "unquantifiable": n_active - quantified,
                "failure": (n_active - quantified) / n_active if n_active else float("nan"),
            }
        )
    total_unquant = int((active["quantified"] == 0).sum())
    total = int(len(active))
    rows.append(
        {
            "mechanism": "All active pairs",
            "n_active": total,
            "quantifiable": total - total_unquant,
            "unquantifiable": total_unquant,
            "failure": total_unquant / total if total else float("nan"),
        }
    )
    return pd.DataFrame(rows)


def active_poor_count(promoted: pd.DataFrame) -> dict:
    """EvE Bio's Active-Poor curation label among active-unquantifiable pairs."""
    unquantified = promoted.loc[
        promoted["Final_Category"] == "Active - Unquantified (11pt)"
    ]
    category = unquantified["Prof_Category_Detailed"].astype(str)
    poor = category.str.contains("Poor", case=False, na=False)
    return {
        "active_unquantifiable": int(len(unquantified)),
        "active_poor": int(poor.sum()),
        "active_poor_fraction": float(poor.mean()) if len(unquantified) else float("nan"),
    }
