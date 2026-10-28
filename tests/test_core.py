"""Tests for the feature definitions, metrics, and leakage rule."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantifiability.dataset import historical_profile_block  # noqa: E402
from quantifiability.features import (  # noqa: E402
    FP_BITS,
    measurement_features,
    morgan_bits,
    murcko_scaffold,
    reported_screening_max,
    screen_feature_frame,
)
from quantifiability.metrics import (  # noqa: E402
    budget_at_retention,
    recall_at_budget,
    retention_prefix_length,
    yield_at_budget,
)
from quantifiability.models import two_stage_scores  # noqa: E402
from quantifiability.synthetic import synthetic_cohort  # noqa: E402


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

    def test_screen_table_ranks_the_stronger_pair_higher(self):
        points = pd.read_csv(Path(__file__).resolve().parents[1] / "examples" / "example_screen.csv")
        table = screen_feature_frame(points)
        ordered = table.sort_values("meas_mean_act", ascending=False)["compound_id"].tolist()
        self.assertEqual(ordered[0], "C1")
        self.assertEqual(ordered[-1], "C2")

    def test_fingerprint_width_and_scaffold_groups(self):
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

    def test_perfect_ranking_keeps_half_the_budget_at_full_recall_when_balanced(self):
        labels = np.array([1, 1, 0, 0])
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        self.assertAlmostEqual(recall_at_budget(labels, scores, 0.5), 1.0)
        self.assertAlmostEqual(budget_at_retention(labels, scores, 1.0), 0.5)


class CohortTests(unittest.TestCase):
    def test_matrix_width_matches_the_paper(self):
        cohort = synthetic_cohort()
        self.assertEqual(cohort.X.shape, (32, 1074))
        self.assertEqual(len(cohort.groups["three_point"]), 16)
        self.assertEqual(len(cohort.groups["fingerprints"]), 1024)
        self.assertEqual(len(cohort.groups["profiles"]), 9)
        self.assertEqual(cohort.y.sum() + (cohort.y == 0).sum(), 32)

    def test_historical_profiles_ignore_the_test_release(self):
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
        # c2 is absent before release 2, so it receives c1's release-1 profile (OMax 80),
        # not its own release-2 OMax of 50.
        self.assertAlmostEqual(before[1, 2], 80.0)  # mean_omax
        self.assertAlmostEqual(before[0, 0], 1.0)  # active_fraction from the one earlier row
        # The assay's earlier history is a single active call, not the later c2 row.
        self.assertAlmostEqual(before[0, 7], 1.0)  # assay_n_active
        after = historical_profile_block(summary, meta, test_release=3.0)
        self.assertAlmostEqual(after[1, 2], 50.0)

    def test_two_stage_separates_an_easy_label(self):
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


if __name__ == "__main__":
    unittest.main()
