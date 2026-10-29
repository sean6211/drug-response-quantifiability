#!/usr/bin/env python3
"""Welch tests and Cohen's d for RDKit descriptors, quantifiable vs active-unquantifiable.

Compound rows use each compound once per group it appears in. Pair rows carry a
compound-cluster bootstrap interval for d, because pairs repeat compounds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from scipy import stats

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]

from scripts._common import prepared_parser, write_table  # noqa: E402
from quantifiability.features import QUANTIFIED_LABEL  # noqa: E402
from quantifiability.metrics import cluster_bootstrap, cohens_d  # noqa: E402

DESCRIPTORS = {
    "Molecular weight": Descriptors.MolWt,
    "LogP": Descriptors.MolLogP,
    "H-bond donors": Descriptors.NumHDonors,
    "H-bond acceptors": Descriptors.NumHAcceptors,
    "Rotatable bonds": Descriptors.NumRotatableBonds,
    "TPSA": Descriptors.TPSA,
}


def compound_descriptors(meta: pd.DataFrame) -> pd.DataFrame:
    RDLogger.DisableLog("rdApp.*")
    rows = []
    for compound_id, smiles in meta.groupby("Compound_ID")["smiles"].first().items():
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles.strip() else None
        if mol is None:
            continue
        row = {"Compound_ID": compound_id}
        row.update({name: float(function(mol)) for name, function in DESCRIPTORS.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def compare(unit: str, quantified: pd.DataFrame, unquantified: pd.DataFrame) -> list[dict]:
    rows = []
    for name in DESCRIPTORS:
        a = quantified[name].to_numpy(dtype=float)
        b = unquantified[name].to_numpy(dtype=float)
        test = stats.ttest_ind(a, b, equal_var=False)
        rows.append(
            {
                "unit": unit,
                "descriptor": name,
                "n_quantifiable": int(a.size),
                "mean_quantifiable": float(a.mean()),
                "sd_quantifiable": float(a.std(ddof=1)),
                "n_unquantifiable": int(b.size),
                "mean_unquantifiable": float(b.mean()),
                "sd_unquantifiable": float(b.std(ddof=1)),
                "welch_t": float(test.statistic),
                "welch_df": float(test.df),
                "p": float(test.pvalue),
                "cohens_d": cohens_d(a, b),
                "d_ci95_low": float("nan"),
                "d_ci95_high": float("nan"),
            }
        )
    return rows


def pair_d_intervals(pairs: pd.DataFrame, resamples: int, seed: int) -> dict[str, tuple]:
    values = pairs[list(DESCRIPTORS)].to_numpy(dtype=float)
    is_quantified = (pairs["final_category"] == QUANTIFIED_LABEL).to_numpy()

    def statistic(rows: np.ndarray):
        group = is_quantified[rows]
        if group.all() or not group.any():
            return None
        block = values[rows]
        return [cohens_d(block[group, j], block[~group, j]) for j in range(block.shape[1])]

    draws = cluster_bootstrap(pairs["Compound_ID"].astype(str).to_numpy(), statistic, resamples, seed)
    return {
        name: tuple(np.percentile(draws[:, j], [2.5, 97.5]))
        for j, name in enumerate(DESCRIPTORS)
    }


def main() -> None:
    parser = prepared_parser(__doc__)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    meta = pd.read_csv(args.data / "meta.csv", usecols=["Compound_ID", "final_category", "smiles"])
    quantified = meta["final_category"] == QUANTIFIED_LABEL
    unquantified = meta["final_category"].astype(str).str.startswith("Active") & ~quantified
    descriptors = compound_descriptors(meta)
    pairs = meta.merge(descriptors, on="Compound_ID", how="inner")
    pair_q = pairs.loc[pairs["final_category"] == QUANTIFIED_LABEL]
    pair_u = pairs.loc[
        pairs["final_category"].astype(str).str.startswith("Active")
        & (pairs["final_category"] != QUANTIFIED_LABEL)
    ]
    compounds_q = set(meta.loc[quantified, "Compound_ID"])
    compounds_u = set(meta.loc[unquantified, "Compound_ID"])
    by_compound = descriptors.set_index("Compound_ID")
    rows = compare(
        "compound",
        by_compound.loc[by_compound.index.isin(compounds_q)],
        by_compound.loc[by_compound.index.isin(compounds_u)],
    )
    pair_rows = compare("pair", pair_q, pair_u)
    intervals = pair_d_intervals(pd.concat([pair_q, pair_u]), args.resamples, args.seed)
    for row in pair_rows:
        row["d_ci95_low"], row["d_ci95_high"] = intervals[row["descriptor"]]
    write_table(args.out, "descriptor_tests.csv", pd.DataFrame(rows + pair_rows))
    both = compounds_q & compounds_u & set(by_compound.index)
    print(f"Compounds with descriptors in both groups: {len(both):,}")


if __name__ == "__main__":
    main()
