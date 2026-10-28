"""A tiny cohort so the tutorial runs without the EvE Bio release."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantifiability.dataset import build_cohort

_SMILES = [
    "c1ccccc1",
    "CCO",
    "c1ccncc1",
    "CC(=O)O",
    "c1ccc2ccccc2c1",
    "CCN",
    "c1ccc(O)cc1",
    "CC(C)O",
]
_ASSAYS = [
    ("A0", "7TM", "Agonist", "Barr2 Recruitment", "TR-FRET"),
    ("A1", "Kinase", "Binding", "Competition Binding", "FRET"),
    ("A2", "NR", "Antagonist", "Co-factor Recruitment", "Luminescence"),
    ("A3", "7TM", "Antagonist", "G Protein Activation", "TR-FRET"),
]


def synthetic_tables(seed: int = 0):
    """Return the frames and measurement dict accepted by ``build_cohort``."""
    rng = np.random.default_rng(seed)
    compounds = [{"Compound_ID": f"C{i}", "SMILES": smiles} for i, smiles in enumerate(_SMILES)]
    assays = [
        {
            "Assay_ID": assay_id,
            "Class": target_class,
            "Mode": mode,
            "Mechanism": mechanism,
            "Technology": technology,
        }
        for assay_id, target_class, mode, mechanism, technology in _ASSAYS
    ]
    rows = []
    measurements = {}
    index = 0
    for compound in compounds:
        for assay in assays:
            strong = index % 2 == 0
            base = 75.0 if strong else 12.0
            concentrations = np.array([1e-6, 1e-6, 1e-5, 1e-5, 1e-4, 1e-4])
            activities = np.clip(
                base + rng.normal(0, 1.5, size=concentrations.size),
                0,
                100,
            )
            measurements[(compound["Compound_ID"], assay["Assay_ID"])] = np.column_stack(
                [concentrations, activities]
            )
            screen_max = float(
                max(
                    activities[concentrations == concentration].mean()
                    for concentration in np.unique(concentrations)
                )
            )
            if strong:
                final = "Active - Quantified (11pt)"
                profile_category = "Active"
            elif index % 4 == 1:
                final = "Inactive (11pt)"
                profile_category = "Inactive"
            else:
                final = "Active - Unquantified (11pt)"
                profile_category = "Active - Poor"
            rows.append(
                {
                    "Compound_ID": compound["Compound_ID"],
                    "Assay_ID": assay["Assay_ID"],
                    "Class": assay["Class"],
                    "Mode": assay["Mode"],
                    "Mechanism": assay["Mechanism"],
                    "Scr_Observed_Max": screen_max,
                    "Scr_Category_Detailed": "Compound of Interest" if strong else "Low Activity",
                    "progressed": True,
                    "progression_reason": "Screening Result",
                    "Prof_Max": screen_max + (8 if strong else 0),
                    "Prof_Category_Detailed": profile_category,
                    "Final_Category": final,
                    "Is_Mutant_Flag": False,
                    "Release": "3.2" if index < 12 else "4",
                    "Result": final,
                    "OMax": screen_max,
                }
            )
            index += 1
    detailed = pd.DataFrame(rows)
    summary = detailed[
        ["Compound_ID", "Assay_ID", "Class", "Mode", "Mechanism", "Result", "OMax", "Release"]
    ].copy()
    return detailed, summary, pd.DataFrame(compounds), pd.DataFrame(assays), measurements


def synthetic_cohort(seed: int = 0):
    return build_cohort(*synthetic_tables(seed))
