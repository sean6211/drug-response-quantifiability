from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quantifiability.features import (
    FP_BITS,
    MEASUREMENT_COLUMNS,
    METADATA_BLOCKS,
    PROFILE_COLUMNS,
    QUANTIFIED_LABEL,
    REPORTED_MAX,
    THREE_POINT_COLUMNS,
    measurement_features,
    morgan_bits,
    murcko_scaffold,
    one_hot,
)

SCREENING_PHASE = "Screening (3pt)"


def parse_release(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def as_flag(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.upper().isin(["TRUE", "1", "T", "YES"])


def _column(frame: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name]
    raise KeyError(f"none of {names} are columns of the table")


@dataclass
class Cohort:
    X: np.ndarray
    y: np.ndarray
    y_active: np.ndarray
    feature_names: list[str]
    groups: dict[str, np.ndarray]
    meta: pd.DataFrame
    summary: pd.DataFrame

    def block(self, *group_names: str) -> np.ndarray:
        indices = np.concatenate([self.groups[name] for name in group_names])
        return self.X[:, indices]

    def with_historical_profiles(self, test_release: float) -> np.ndarray:
        features = self.X.copy()
        features[:, self.groups["profiles"]] = historical_profile_block(
            self.summary, self.meta, test_release
        )
        return features

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "X.npy", self.X)
        np.save(directory / "y.npy", self.y)
        np.save(directory / "y_active.npy", self.y_active)
        (directory / "feature_names.json").write_text(json.dumps(self.feature_names))
        groups = {name: indices.tolist() for name, indices in self.groups.items()}
        (directory / "groups.json").write_text(json.dumps(groups))
        self.meta.to_csv(directory / "meta.csv", index=False)
        self.summary.to_csv(directory / "summary.csv.gz", index=False)

    @classmethod
    def load(cls, directory: Path) -> "Cohort":
        directory = Path(directory)
        feature_names = json.loads((directory / "feature_names.json").read_text())
        raw_groups = json.loads((directory / "groups.json").read_text())
        groups = {name: np.asarray(indices, dtype=int) for name, indices in raw_groups.items()}
        return cls(
            X=np.load(directory / "X.npy"),
            y=np.load(directory / "y.npy"),
            y_active=np.load(directory / "y_active.npy"),
            feature_names=feature_names,
            groups=groups,
            meta=pd.read_csv(directory / "meta.csv"),
            summary=pd.read_csv(directory / "summary.csv.gz"),
        )


def _indices(feature_names: list[str], wanted: list[str]) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(feature_names)}
    missing = [name for name in wanted if name not in lookup]
    if missing:
        raise KeyError(f"feature matrix is missing {missing}")
    return np.asarray([lookup[name] for name in wanted], dtype=int)


def column_groups(feature_names: list[str]) -> dict[str, np.ndarray]:
    metadata = ["is_mutant"]
    for prefix, _source, levels in METADATA_BLOCKS:
        metadata.extend(f"{prefix}_{level}" for level in levels)
    groups = {
        "fingerprints": _indices(feature_names, [f"fp_{i}" for i in range(FP_BITS)]),
        "three_point": _indices(feature_names, THREE_POINT_COLUMNS),
        "profiles": _indices(feature_names, PROFILE_COLUMNS),
        "metadata": _indices(feature_names, metadata),
    }
    groups["mechanism"] = _indices(
        feature_names, [name for name in feature_names if name.startswith("mech_")]
    )
    return groups


def historical_profile_block(
    summary: pd.DataFrame,
    meta: pd.DataFrame,
    test_release: float,
) -> np.ndarray:
    history = summary.loc[summary["rel"] < float(test_release) - 1e-9].copy()
    n_rows = len(meta)
    if history.empty:
        return np.zeros((n_rows, len(PROFILE_COLUMNS)), dtype=np.float32)

    active = history["Result"].astype(str).str.contains("Active", na=False)
    history = history.assign(_active=active)
    compound = history.groupby("Compound_ID", sort=False).agg(
        n_active=("_active", "sum"),
        n_assays=("Assay_ID", "count"),
        mean_omax=("OMax", "mean"),
        max_omax=("OMax", "max"),
        n_classes=("Class", "nunique"),
        n_mechanisms=("Mechanism", "nunique"),
    )
    compound["active_fraction"] = compound["n_active"] / compound["n_assays"]
    assay = history.groupby("Assay_ID", sort=False).agg(
        assay_n_active=("_active", "sum"),
        assay_n_total=("Compound_ID", "count"),
        assay_mean_omax=("OMax", "mean"),
    )
    assay["assay_active_frac"] = assay["assay_n_active"] / assay["assay_n_total"]

    history_omax = float(history["OMax"].mean())
    prior = {
        "active_fraction": float(compound["active_fraction"].mean()),
        "n_active": float(compound["n_active"].mean()),
        "mean_omax": history_omax,
        "max_omax": float(compound["max_omax"].mean()),
        "n_classes": float(compound["n_classes"].mean()),
        "n_mechanisms": float(compound["n_mechanisms"].mean()),
        "assay_active_frac": float(assay["assay_active_frac"].mean()),
        "assay_n_active": float(assay["assay_n_active"].mean()),
        "assay_mean_omax": history_omax,
    }
    compound = compound.reindex(meta["Compound_ID"].to_numpy())
    assay = assay.reindex(meta["Assay_ID"].to_numpy())
    columns = [
        compound["active_fraction"].fillna(prior["active_fraction"]).to_numpy(),
        compound["n_active"].fillna(prior["n_active"]).to_numpy(),
        compound["mean_omax"].fillna(prior["mean_omax"]).to_numpy(),
        compound["max_omax"].fillna(prior["max_omax"]).to_numpy(),
        compound["n_classes"].fillna(prior["n_classes"]).to_numpy(),
        compound["n_mechanisms"].fillna(prior["n_mechanisms"]).to_numpy(),
        assay["assay_active_frac"].fillna(prior["assay_active_frac"]).to_numpy(),
        assay["assay_n_active"].fillna(prior["assay_n_active"]).to_numpy(),
        assay["assay_mean_omax"].fillna(prior["assay_mean_omax"]).to_numpy(),
    ]
    return np.column_stack(columns).astype(np.float32)


def _measurement_matrix(promoted: pd.DataFrame, measurements: dict) -> np.ndarray:
    features = np.zeros((len(promoted), len(MEASUREMENT_COLUMNS)), dtype=np.float32)
    for index, (compound_id, assay_id) in enumerate(
        zip(promoted["Compound_ID"], promoted["Assay_ID"])
    ):
        points = measurements.get((compound_id, assay_id))
        if points is None:
            continue
        features[index] = measurement_features(points[:, 0], points[:, 1])
    return features


def _prepare_summary(summary: pd.DataFrame, detailed: pd.DataFrame) -> pd.DataFrame:
    release_lookup = pd.DataFrame(
        {
            "Compound_ID": _column(detailed, "Compound_ID"),
            "Assay_ID": _column(detailed, "Assay_ID"),
            "Release_detailed": _column(detailed, "Release"),
        }
    ).drop_duplicates(["Compound_ID", "Assay_ID"])
    merged = summary.merge(release_lookup, on=["Compound_ID", "Assay_ID"], how="left")
    if "Release" in merged.columns:
        release = merged["Release_detailed"].fillna(merged["Release"])
    else:
        release = merged["Release_detailed"]
    prepared = pd.DataFrame(
        {
            "Compound_ID": merged["Compound_ID"],
            "Assay_ID": merged["Assay_ID"],
            "Result": merged["Result"],
            "OMax": pd.to_numeric(merged["OMax"], errors="coerce"),
            "Class": merged["Class"],
            "Mechanism": merged["Mechanism"],
            "rel": release.map(parse_release),
        }
    )
    return prepared


def build_cohort(
    detailed: pd.DataFrame,
    summary: pd.DataFrame,
    compounds: pd.DataFrame,
    assays: pd.DataFrame,
    measurements: dict,
) -> Cohort:
    promoted = detailed.loc[as_flag(_column(detailed, "progressed", "Progressed"))].copy()
    promoted = promoted.reset_index(drop=True)
    if promoted.empty:
        raise ValueError("no promoted pairs")

    final_category = promoted["Final_Category"].astype(str)
    y = (final_category == QUANTIFIED_LABEL).to_numpy(dtype=np.int8)
    y_active = final_category.str.startswith("Active").to_numpy(dtype=np.int8)

    assay_meta = assays[["Assay_ID", "Class", "Mode", "Mechanism", "Technology"]].drop_duplicates(
        "Assay_ID"
    )
    promoted = promoted.merge(assay_meta, on="Assay_ID", how="left", suffixes=("", "_assay"))
    if promoted["Technology"].isna().any():
        missing = promoted.loc[promoted["Technology"].isna(), "Assay_ID"].unique()[:5]
        raise ValueError(f"promoted assays missing technology, for example {list(missing)}")

    measured = _measurement_matrix(promoted, measurements)
    reported_max = (
        pd.to_numeric(promoted["Scr_Observed_Max"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    )
    smiles = compounds.set_index("Compound_ID")["SMILES"].to_dict()
    fingerprints = np.vstack(
        [morgan_bits(smiles.get(compound_id, "")) for compound_id in promoted["Compound_ID"]]
    ).astype(np.float32)

    history = _prepare_summary(summary, detailed)
    meta_keys = pd.DataFrame(
        {"Compound_ID": promoted["Compound_ID"], "Assay_ID": promoted["Assay_ID"]}
    )
    profiles = historical_profile_block(history, meta_keys, test_release=float("inf"))

    blocks = [one_hot(promoted[source], levels, prefix) for prefix, source, levels in METADATA_BLOCKS]
    mutant = as_flag(promoted["Is_Mutant_Flag"]).to_numpy(dtype=np.float32).reshape(-1, 1)

    pieces = [
        fingerprints,
        measured,
        reported_max.reshape(-1, 1),
        mutant,
        *[block.to_numpy(dtype=np.float32) for block in blocks],
        profiles,
    ]
    matrix = np.hstack(pieces).astype(np.float32)
    feature_names = (
        [f"fp_{i}" for i in range(FP_BITS)]
        + MEASUREMENT_COLUMNS
        + [REPORTED_MAX, "is_mutant"]
        + [column for block in blocks for column in block.columns]
        + PROFILE_COLUMNS
    )
    if matrix.shape[1] != len(feature_names):
        raise RuntimeError(f"matrix width {matrix.shape[1]} != {len(feature_names)} names")

    meta = pd.DataFrame(
        {
            "Compound_ID": promoted["Compound_ID"].to_numpy(),
            "Assay_ID": promoted["Assay_ID"].to_numpy(),
            "release": promoted["Release"].map(parse_release).to_numpy(),
            "mechanism": promoted["Mechanism"].astype(str).to_numpy(),
            "target_class": promoted["Class"].astype(str).to_numpy(),
            "screening_category": promoted["Scr_Category_Detailed"].astype(str).to_numpy(),
            "final_category": final_category.to_numpy(),
            "prof_category": promoted["Prof_Category_Detailed"].astype(str).to_numpy()
            if "Prof_Category_Detailed" in promoted.columns
            else "",
            "scr_observed_max": reported_max,
            "prof_max": pd.to_numeric(promoted["Prof_Max"], errors="coerce").to_numpy()
            if "Prof_Max" in promoted.columns
            else np.nan,
            "smiles": [smiles.get(compound_id, "") for compound_id in promoted["Compound_ID"]],
        }
    )
    meta["scaffold"] = [
        murcko_scaffold(smiles_value, compound_id)
        for compound_id, smiles_value in zip(meta["Compound_ID"], meta["smiles"])
    ]
    return Cohort(
        X=matrix,
        y=y,
        y_active=y_active,
        feature_names=feature_names,
        groups=column_groups(feature_names),
        meta=meta,
        summary=history,
    )


def load_measurements(observed_points: Path, promoted_keys: pd.DataFrame) -> dict:
    from collections import defaultdict

    buckets: dict[tuple, list] = defaultdict(list)
    keys = promoted_keys[["Compound_ID", "Assay_ID"]].drop_duplicates()
    usecols = ["Compound_ID", "Assay_ID", "Phase", "Concentration", "Activity"]
    reader = pd.read_csv(observed_points, usecols=usecols, chunksize=500_000)
    for chunk in reader:
        screening = chunk.loc[chunk["Phase"] == SCREENING_PHASE]
        if screening.empty:
            continue
        matched = screening.merge(keys, on=["Compound_ID", "Assay_ID"], how="inner")
        if matched.empty:
            continue
        matched["Concentration"] = pd.to_numeric(matched["Concentration"], errors="coerce")
        matched["Activity"] = pd.to_numeric(matched["Activity"], errors="coerce")
        matched = matched.dropna(subset=["Concentration", "Activity"])
        for (compound_id, assay_id), group in matched.groupby(
            ["Compound_ID", "Assay_ID"], sort=False
        ):
            buckets[(compound_id, assay_id)].append(
                group[["Concentration", "Activity"]].to_numpy(dtype=np.float64)
            )
    measurements = {}
    for key, parts in buckets.items():
        measurements[key] = parts[0] if len(parts) == 1 else np.vstack(parts)
    return measurements


def read_evebio(data_dir: Path) -> Cohort:
    data_dir = Path(data_dir)
    detailed_cols = [
        "Assay_ID",
        "Class",
        "Mode",
        "Mechanism",
        "Compound_ID",
        "Scr_Observed_Max",
        "Scr_Category_Detailed",
        "progressed",
        "progression_reason",
        "Prof_Max",
        "Prof_Category_Detailed",
        "Final_Category",
        "Is_Mutant_Flag",
        "Release",
    ]
    summary_cols = [
        "Assay_ID",
        "Class",
        "Mode",
        "Mechanism",
        "Compound_ID",
        "Result",
        "OMax",
        "Release",
    ]
    print(f"Reading EvE Bio tables from {data_dir}")
    detailed = pd.read_csv(
        data_dir / "detailed_result_table.csv", usecols=detailed_cols, low_memory=False
    )
    summary = pd.read_csv(
        data_dir / "summary_result_table.csv", usecols=summary_cols, low_memory=False
    )
    compounds = pd.read_csv(data_dir / "compound_table.csv", usecols=["Compound_ID", "SMILES"])
    assays = pd.read_csv(
        data_dir / "assay_table.csv",
        usecols=["Assay_ID", "Class", "Mode", "Mechanism", "Technology"],
    )
    promoted = detailed.loc[as_flag(detailed["progressed"]), ["Compound_ID", "Assay_ID"]]
    print(f"Loading three-point measurements for {len(promoted):,} promoted pairs")
    measurements = load_measurements(data_dir / "observed_points_table.csv", promoted)
    print(f"Measurements found for {len(measurements):,} pairs")
    return build_cohort(detailed, summary, compounds, assays, measurements)
