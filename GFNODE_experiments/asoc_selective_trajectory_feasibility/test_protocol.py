"""Focused regression tests for the C1-S0R calibration/bootstrap correction."""
from __future__ import annotations

import inspect
import unittest

import numpy as np
import pandas as pd

import run_selective_feasibility as screen


class ScopeMatchedCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scores = np.asarray([0.05, 0.10, 0.20, 0.30, 10.0, 20.0, 30.0, 40.0])
        self.calibration = np.asarray([False, False, False, False, True, True, True, True])
        self.full = np.ones(8, dtype=bool)
        self.daylight = np.asarray([True, False, True, False, True, False, True, False])

    def test_daylight_threshold_uses_daylight_calibration_subset(self) -> None:
        threshold, mask, _ = screen.calibrated_threshold(
            self.scores, self.calibration, self.daylight, 0.5
        )
        expected_mask = self.calibration & self.daylight
        self.assertTrue(np.array_equal(mask, expected_mask))
        self.assertEqual(threshold, float(np.quantile(self.scores[expected_mask], 0.5)))

    def test_full_threshold_uses_full_calibration_subset(self) -> None:
        threshold, mask, _ = screen.calibrated_threshold(
            self.scores, self.calibration, self.full, 0.5
        )
        self.assertTrue(np.array_equal(mask, self.calibration))
        self.assertEqual(threshold, float(np.quantile(self.scores[self.calibration], 0.5)))

    def test_full_and_daylight_calibration_masks_are_not_reused(self) -> None:
        full_mask = screen.scope_calibration_mask(self.calibration, self.full)
        daylight_mask = screen.scope_calibration_mask(self.calibration, self.daylight)
        self.assertFalse(np.array_equal(full_mask, daylight_mask))
        self.assertLess(daylight_mask.sum(), full_mask.sum())

    def test_calibration_coverage_is_recomputed_in_matching_scope(self) -> None:
        threshold, mask, realized = screen.calibrated_threshold(
            self.scores, self.calibration, self.daylight, 0.5
        )
        self.assertEqual(realized, float(np.mean(self.scores[mask] <= threshold)))

    def test_threshold_api_cannot_receive_test_scores(self) -> None:
        parameters = inspect.signature(screen.calibrated_threshold).parameters
        self.assertNotIn("test_scores", parameters)
        threshold, _, _ = screen.calibrated_threshold(
            self.scores, self.calibration, self.daylight, 0.5
        )
        sentinel_test_scores = np.asarray([-1e9, 1e9])
        self.assertNotIn(threshold, sentinel_test_scores.tolist())


class NaturalDayClusterBootstrapTests(unittest.TestCase):
    def _arrays(self):
        labels = np.zeros((6, 3), dtype=float)
        prediction = np.asarray([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]) * np.ones((1, 3))
        persistence = prediction + 1.0
        accepted = np.asarray([True, False, True, False, False, False])
        daylight = np.ones(6, dtype=bool)
        times = pd.to_datetime([
            "2022-11-01 09:00", "2022-11-01 10:00",
            "2022-11-02 09:00", "2022-11-02 10:00",
            "2022-11-03 09:00", "2022-11-03 10:00",
        ]).to_numpy(dtype="datetime64[ns]").astype(np.int64)
        return labels, prediction, persistence, accepted, daylight, times

    def test_bootstrap_recomputes_unselected_and_accepted_rmse(self) -> None:
        labels, prediction, persistence, accepted, _, _ = self._arrays()
        chosen = np.asarray([0, 1, 2, 3])
        values = screen.bootstrap_replicate_metrics(
            labels, prediction, persistence, accepted, chosen, 3
        )
        self.assertIsNotNone(values)
        expected_unselected = float(np.sqrt(np.mean(prediction[chosen] ** 2)))
        expected_accepted = float(np.sqrt(np.mean(prediction[[0, 2]] ** 2)))
        self.assertAlmostEqual(values["unselected_daylight_rmse"], expected_unselected)
        self.assertAlmostEqual(values["accepted_rmse"], expected_accepted)

    def test_bootstrap_sampling_frame_contains_all_daylight_days(self) -> None:
        labels, prediction, persistence, accepted, daylight, times = self._arrays()
        summaries, skipped, day_count = screen.natural_day_cluster_bootstrap(
            labels, prediction, persistence, accepted, daylight, times, 3, 25, 123
        )
        self.assertEqual(day_count, 3)
        self.assertGreaterEqual(skipped, 0)
        self.assertIsNone(screen.bootstrap_replicate_metrics(
            labels, prediction, persistence, accepted, np.asarray([4, 5]), 3
        ))
        self.assertIn("unselected_daylight_rmse", summaries)


if __name__ == "__main__":
    unittest.main(verbosity=2)
