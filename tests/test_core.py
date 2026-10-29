from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantifiability.dataset import build_cohort, historical_profile_block  # noqa: E402
from quantifiability.features import (  # noqa: E402
    FP_BITS,
    measurement_features,
    morgan_bits,
    murcko_scaffold,
    reported_screening_max,
)
from quantifiability.metrics import (  # noqa: E402
    budget_at_retention,
    cluster_bootstrap,
    cohens_d,
    recall_at_budget,
    retention_prefix_length,
    yield_at_budget,
)
from quantifiability.models import two_stage_scores  # noqa: E402


class FeatureTests(unittest.TestCase):
    def test_three_point_summaries(self):
        concentrations = [1e-6, 1e-6, 1e-5, 1e-5, 1e-4, 1e-4]
        activities = [10, 20, 30, 50, 80, 100]
        features = measurement_features(concentrations, activities)
        self.assertEqual(len(features), 15)
        self.assertAlmostEqual(float(features[0]), float(np.mean(activities)), places=5)
        self.assertAlmostEqual(float(features[2]), 100, places=5)
        self.assertAlmostEqual(float(features[3]), 10, places=5)
        self.assertAlmostEqual(float(features[4]), 40, places=5)
        self.assertAlmostEqual(float(features[5]), 90, places=5)
        self.assertAlmostEqual(float(features[6]), 90, places=5)
        self.assertAlmostEqual(float(features[7]), 15, places=5)
        self.assertAlmostEqual(float(features[8]), 75, places=5)
        self.assertAlmostEqual(float(features[9]), (5 + 10 + 10) / 3, places=5)
        self.assertAlmostEqual(features[10], 10)
        self.assertEqual(features[11], 6)
        self.assertEqual(features[12], 3)
        self.assertEqual(features[13], 1)
        self.assertEqual(features[14], 1)
        self.assertAlmostEqual(reported_screening_max(concentrations, activities), 90)

    def test_empty_screen_is_zero(self):
        self.assertTrue(np.all(measurement_features([], []) == 0))

    def test_fingerprint_and_scaffold(self):
        bits = morgan_bits("c1ccccc1")
        self.assertEqual(bits.shape, (FP_BITS,))
        self.assertGreater(bits.sum(), 0)
        self.assertEqual(morgan_bits("not a smiles").sum(), 0)
        self.assertEqual(murcko_scaffold("c1ccccc1", "a"), murcko_scaffold("Cc1ccccc1", "b"))
        self.assertNotEqual(murcko_scaffold("CCO", "ethanol"), murcko_scaffold("CCO", "other"))


class MetricTests(unittest.TestCase):
    def test_budget_definitions(self):
        labels = np.array([1, 0, 1, 0])
        scores = np.array([0.1, 0.9, 0.8, 0.2])
        self.assertAlmostEqual(recall_at_budget(labels, scores, 0.5), 0.5)
        self.assertAlmostEqual(yield_at_budget(labels, scores, 0.5), 0.5)
        self.assertEqual(retention_prefix_length(labels, scores, 0.5), 2)
        self.assertAlmostEqual(budget_at_retention(labels, scores, 1.0), 1.0)

    def test_recall_budget(self):
        labels = np.array([1, 1, 0, 0])
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        self.assertAlmostEqual(recall_at_budget(labels, scores, 0.5), 1.0)
        self.assertAlmostEqual(budget_at_retention(labels, scores, 1.0), 0.5)

    def test_cohens_d(self):
        self.assertAlmostEqual(cohens_d([1, 2, 3], [2, 3, 4]), -1.0)
        self.assertTrue(np.isnan(cohens_d([1], [2, 3])))

    def test_cluster_bootstrap_keeps_clusters_whole(self):
        clusters = np.array(["a", "a", "b", "b", "c"])
        sizes = cluster_bootstrap(clusters, lambda rows: len(rows), n_resamples=50, seed=0)
        self.assertEqual(sizes.shape, (50, 1))
        draws = cluster_bootstrap(
            clusters, lambda rows: None if len(rows) > 5 else len(rows), n_resamples=50, seed=0
        )
        self.assertTrue(np.all(draws <= 5))


class CohortTests(unittest.TestCase):
    def test_matrix_width(self):
        cohort = _small_cohort()
        self.assertEqual(cohort.X.shape, (4, 1074))
        self.assertEqual(len(cohort.groups["three_point"]), 16)
        self.assertEqual(len(cohort.groups["fingerprints"]), 1024)
        self.assertEqual(len(cohort.groups["profiles"]), 9)
        self.assertEqual(len(cohort.groups["screening_category"]), 7)
        self.assertEqual(len(cohort.groups["promotion_reason"]), 3)
        self.assertEqual(len(cohort.groups["annotations"]), 10)
        names = [cohort.feature_names[i] for i in cohort.groups["annotations"]]
        self.assertTrue(all(name.startswith(("scr_cat_", "prog_")) for name in names))

    def test_historical_profiles(self):
        summary = pd.DataFrame(
            {
                "Compound_ID": ["c1", "c1", "c2"],
                "Assay_ID": ["a1", "a2", "a1"],
                "Result": [
                    "Active - Quantified (11pt)",
                    "Inactive (11pt)",
                    "Active - Quantified (11pt)",
                ],
                "OMax": [80.0, 10.0, 50.0],
                "Class": ["7TM", "7TM", "Kinase"],
                "Mechanism": ["Barr2 Recruitment", "Barr2 Recruitment", "Heterodimer"],
                "rel": [1.0, 2.0, 2.0],
            }
        )
        meta = pd.DataFrame({"Compound_ID": ["c1", "c2"], "Assay_ID": ["a1", "a1"]})
        before = historical_profile_block(summary, meta, test_release=2.0)
        self.assertAlmostEqual(before[1, 2], 80.0)
        self.assertAlmostEqual(before[0, 0], 1.0)
        self.assertAlmostEqual(before[0, 7], 1.0)
        after = historical_profile_block(summary, meta, test_release=3.0)
        self.assertAlmostEqual(after[1, 2], 50.0)

    def test_two_stage(self):
        rng = np.random.default_rng(0)
        labels = np.zeros(80, dtype=int)
        labels[::2] = 1
        features = rng.normal(size=(80, 4))
        features[labels == 1, 0] += 3
        scores = two_stage_scores(
            features[:60],
            labels[:60],
            labels[:60],
            features[60:],
            n_estimators=25,
            seed=0,
            n_jobs=1,
        )
        held_out = labels[60:]
        self.assertGreater(scores[held_out == 1].mean(), scores[held_out == 0].mean())


def _small_cohort():
    compounds = pd.DataFrame(
        {
            "Compound_ID": ["C0", "C1"],
            "SMILES": ["c1ccccc1", "CCO"],
        }
    )
    assays = pd.DataFrame(
        [
            {
                "Assay_ID": "A0",
                "Class": "7TM",
                "Mode": "Agonist",
                "Mechanism": "Barr2 Recruitment",
                "Technology": "TR-FRET",
            },
            {
                "Assay_ID": "A1",
                "Class": "Kinase",
                "Mode": "Binding",
                "Mechanism": "Competition Binding",
                "Technology": "FRET",
            },
        ]
    )
    rows = []
    measurements = {}
    conc = np.array([1e-6, 1e-5, 1e-4])
    for i, (compound_id, assay_id) in enumerate(
        [("C0", "A0"), ("C0", "A1"), ("C1", "A0"), ("C1", "A1")]
    ):
        activity = np.array([70, 80, 90]) if i % 2 == 0 else np.array([5, 8, 10])
        measurements[(compound_id, assay_id)] = np.column_stack([conc, activity])
        final = "Active - Quantified (11pt)" if i % 2 == 0 else "Inactive (11pt)"
        assay = assays.loc[assays["Assay_ID"] == assay_id].iloc[0]
        rows.append(
            {
                "Compound_ID": compound_id,
                "Assay_ID": assay_id,
                "Class": assay["Class"],
                "Mode": assay["Mode"],
                "Mechanism": assay["Mechanism"],
                "Scr_Observed_Max": float(activity.mean()),
                "Scr_Category_Detailed": "Compound of Interest",
                "progressed": True,
                "progression_reason": "Screening Result",
                "Prof_Max": float(activity.max()),
                "Prof_Category_Detailed": "Active",
                "Final_Category": final,
                "Is_Mutant_Flag": False,
                "Release": "4",
                "Result": final,
                "OMax": float(activity.mean()),
            }
        )
    detailed = pd.DataFrame(rows)
    summary = detailed[
        ["Compound_ID", "Assay_ID", "Class", "Mode", "Mechanism", "Result", "OMax", "Release"]
    ]
    return build_cohort(detailed, summary, compounds, assays, measurements)


if __name__ == "__main__":
    unittest.main()
