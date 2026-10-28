"""Screening-stage features used to predict quantifiability.

The 16 three-point features are the representation the main text isolates.
Morgan fingerprints (radius 2, 1,024 bits) and the assay-metadata levels below
complete the 1,074-column matrix. Levels are the ones present among promoted
EvE Bio pairs, so a refit stays aligned with that matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import ConvertToNumpyArray

RDLogger.DisableLog("rdApp.*")

FP_BITS = 1024
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)

QUANTIFIED_LABEL = "Active - Quantified (11pt)"

MEASUREMENT_COLUMNS = [
    "meas_mean_act",
    "meas_std_act",
    "meas_max_act",
    "meas_min_act",
    "meas_median_act",
    "meas_range_act",
    "meas_act_highconc",
    "meas_act_lowconc",
    "meas_slope_proxy",
    "meas_mean_rep_std",
    "meas_max_rep_std",
    "meas_n_points",
    "meas_n_concs",
    "meas_strong_flag",
    "meas_vstrong_flag",
]
REPORTED_MAX = "scr_observed_max"
THREE_POINT_COLUMNS = MEASUREMENT_COLUMNS + [REPORTED_MAX]

PROFILE_COLUMNS = [
    "active_fraction",
    "n_active",
    "mean_omax",
    "max_omax",
    "n_classes",
    "n_mechanisms",
    "assay_active_frac",
    "assay_n_active",
    "assay_mean_omax",
]

SCREENING_CATEGORIES = [
    "Compound of Interest",
    "Inactive",
    "Inconsistent - High",
    "Inconsistent - Low",
    "Inconsistent - Random",
    "Low Activity",
    "Unmeasurable pXC50",
]
PROGRESSION_REASONS = [
    "Bundled Target Result",
    "Convenience",
    "Screening Result",
]
TARGET_CLASSES = ["7TM", "Kinase", "NR"]
ASSAY_MODES = ["Agonist", "Antagonist", "Binding"]
MECHANISMS = [
    "Barr2 Recruitment",
    "Co-factor Recruitment",
    "Competition Binding",
    "G Protein Activation",
    "Heterodimer",
]
TECHNOLOGIES = ["FRET", "Luminescence", "TR-FRET"]

# prefix, source column, fixed levels. Order matches the published matrix.
METADATA_BLOCKS = [
    ("scr_cat", "Scr_Category_Detailed", SCREENING_CATEGORIES),
    ("prog", "progression_reason", PROGRESSION_REASONS),
    ("class", "Class", TARGET_CLASSES),
    ("mode", "Mode", ASSAY_MODES),
    ("mech", "Mechanism", MECHANISMS),
    ("tech", "Technology", TECHNOLOGIES),
]


def measurement_features(concentrations, activities) -> np.ndarray:
    """Fifteen summaries of one pair's three-point screen.

    The reported screening maximum is separate: EvE Bio stores it as
    ``Scr_Observed_Max``, the maximum of the per-concentration means.
    """
    concentrations = np.asarray(concentrations, dtype=np.float64)
    activities = np.asarray(activities, dtype=np.float64)
    if concentrations.shape != activities.shape:
        raise ValueError("concentrations and activities must have the same shape")
    ok = np.isfinite(concentrations) & np.isfinite(activities)
    concentrations = concentrations[ok]
    activities = activities[ok]
    out = np.zeros(len(MEASUREMENT_COLUMNS), dtype=np.float32)
    if activities.size == 0:
        return out

    unique = np.unique(concentrations)
    means = []
    rep_stds = []
    for concentration in unique:
        values = activities[concentrations == concentration]
        means.append(float(values.mean()))
        rep_stds.append(float(values.std()) if values.size > 1 else 0.0)

    out[0] = activities.mean()
    out[1] = activities.std() if activities.size > 1 else 0.0
    out[2] = activities.max()
    out[3] = activities.min()
    out[4] = np.median(activities)
    out[5] = activities.max() - activities.min()
    out[6] = means[-1]
    out[7] = means[0]
    out[8] = (means[-1] - means[0]) if len(means) >= 2 else 0.0
    out[9] = float(np.mean(rep_stds))
    out[10] = float(np.max(rep_stds)) if rep_stds else 0.0
    out[11] = activities.size
    out[12] = unique.size
    out[13] = 1.0 if activities.max() > 30 else 0.0
    out[14] = 1.0 if activities.max() > 50 else 0.0
    return out


def reported_screening_max(concentrations, activities) -> float:
    """Maximum of the per-concentration mean activities."""
    concentrations = np.asarray(concentrations, dtype=np.float64)
    activities = np.asarray(activities, dtype=np.float64)
    ok = np.isfinite(concentrations) & np.isfinite(activities)
    concentrations = concentrations[ok]
    activities = activities[ok]
    if activities.size == 0:
        return 0.0
    means = [
        float(activities[concentrations == concentration].mean())
        for concentration in np.unique(concentrations)
    ]
    return float(max(means))


def screen_feature_frame(measurements: pd.DataFrame) -> pd.DataFrame:
    """One row of 16 three-point features per compound-assay pair.

    ``measurements`` is long-form, with columns ``compound_id``, ``assay_id``,
    ``concentration``, and ``activity``. Pairs with no finite points get zeros.
    """
    required = {"compound_id", "assay_id", "concentration", "activity"}
    missing = required.difference(measurements.columns)
    if missing:
        raise ValueError(f"measurements is missing columns: {sorted(missing)}")
    rows = []
    keys = []
    grouped = measurements.groupby(["compound_id", "assay_id"], sort=False)
    for (compound_id, assay_id), group in grouped:
        feats = measurement_features(group["concentration"], group["activity"])
        reported = reported_screening_max(group["concentration"], group["activity"])
        rows.append(np.concatenate([feats, np.array([reported], dtype=np.float32)]))
        keys.append((compound_id, assay_id))
    frame = pd.DataFrame(rows, columns=THREE_POINT_COLUMNS)
    frame.insert(0, "assay_id", [key[1] for key in keys])
    frame.insert(0, "compound_id", [key[0] for key in keys])
    return frame


def morgan_bits(smiles: str) -> np.ndarray:
    """Radius-2 Morgan fingerprint. Unparseable SMILES return the zero vector."""
    out = np.zeros(FP_BITS, dtype=np.float32)
    if not isinstance(smiles, str) or not smiles.strip():
        return out
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return out
    ConvertToNumpyArray(_MORGAN.GetFingerprint(molecule), out)
    return out


def murcko_scaffold(smiles: str, compound_id: str) -> str:
    """Bemis-Murcko scaffold, or a compound-specific label when none exists."""
    fallback = f"NOSCAF_{compound_id}"
    if not isinstance(smiles, str) or not smiles.strip():
        return fallback
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return fallback
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
    except Exception:
        return fallback
    return scaffold if scaffold else fallback


def one_hot(series: pd.Series, categories: list[str], prefix: str) -> pd.DataFrame:
    """Dummy columns for a fixed level set, including levels absent from ``series``."""
    observed = set(series.dropna().astype(str))
    unexpected = sorted(observed.difference(categories))
    if unexpected:
        raise ValueError(f"Unexpected {prefix} levels: {unexpected}")
    categorical = pd.Categorical(series.astype(str), categories=categories)
    dummies = pd.get_dummies(categorical, prefix=prefix, dtype=np.float32)
    expected = [f"{prefix}_{level}" for level in categories]
    return dummies.reindex(columns=expected, fill_value=np.float32(0.0))
