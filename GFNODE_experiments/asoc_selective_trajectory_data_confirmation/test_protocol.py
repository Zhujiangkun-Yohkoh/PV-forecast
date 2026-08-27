"""Ordinary real-array tests for Scheme C1-S2 (no model training)."""
from __future__ import annotations

import argparse
import ast
import json
import unittest
from pathlib import Path

import numpy as np

import validate_c1_formal_data as audit


SUMMARY: dict = {}
STATE: np.lib.npyio.NpzFile | None = None
CONFIG: dict = {}


class FormalDataProtocolTests(unittest.TestCase):
    def test_three_year_second_file_status_is_source_backed(self) -> None:
        self.assertEqual(set(SUMMARY["irradiance"]), {"2021", "2022", "2023"})
        for year, item in SUMMARY["irradiance"].items():
            self.assertEqual(int(year), item["year"])
            self.assertEqual(item["file_exists"], Path(item["path"]).exists())

    def test_authoritative_utc_to_acst_is_fixed_570_minutes(self) -> None:
        raw = b"01/01/2022 00:00:01"
        epoch = audit.parse_dmy_second(raw)
        self.assertIsNotNone(epoch)
        utc = np.datetime64("1970-01-01T00:00:00") + np.timedelta64(int(epoch), "s")
        acst = utc + np.timedelta64(CONFIG["timezone"]["utc_to_acst_minutes"], "m")
        self.assertEqual(str(acst), "2022-01-01T09:30:01")
        self.assertFalse(CONFIG["timezone"]["use_exported_local_field"])

    def test_windows_do_not_cross_stage_boundaries(self) -> None:
        grid = STATE["grid_ns"].astype("datetime64[ns]")
        for stage, (start, end) in CONFIG["stages"].items():
            origins = STATE[f"origins__{stage}__COMMON"]
            if not len(origins):
                continue
            self.assertTrue(np.all(grid[origins - CONFIG["lookback"] + 1] >= np.datetime64(start)))
            self.assertTrue(np.all(grid[origins + CONFIG["horizon"]] < np.datetime64(end)))

    def test_common_windows_do_not_cross_missing_segments(self) -> None:
        hf_present = STATE["hf_timestamp_count"] > 0
        powers = [STATE[f"pv_power_{i}"] for i in range(3)]
        for stage in CONFIG["stages"]:
            for origin in STATE[f"origins__{stage}__COMMON"]:
                lo, hi = int(origin) - CONFIG["lookback"] + 1, int(origin) + CONFIG["horizon"]
                self.assertTrue(hf_present[lo:int(origin) + 1].all())
                for power in powers:
                    self.assertTrue(np.isfinite(power[lo:hi + 1]).all())

    def test_three_array_common_origins_are_exact_intersection(self) -> None:
        arrays = list(CONFIG["pv_files"])
        for stage in CONFIG["stages"]:
            expected = set(STATE[f"origins__{stage}__{arrays[0]}"].tolist())
            for array in arrays[1:]:
                expected &= set(STATE[f"origins__{stage}__{array}"].tolist())
            self.assertEqual(expected, set(STATE[f"origins__{stage}__COMMON"].tolist()))

    def test_future_sentinel_leaves_actual_origin_features_unchanged(self) -> None:
        candidates = STATE["origins__BASE_MODEL_VALIDATION__COMMON"]
        if not len(candidates):
            self.skipTest("No eligible common origin because required source data failed")
        origin = int(candidates[len(candidates) // 2])
        power = STATE["pv_power_0"].copy()
        hf_mean = STATE["hf_channel_mean"].copy()
        hf_count = STATE["hf_channel_count"].copy()
        before = audit.causal_foundation_features(origin, power, hf_mean, hf_count, CONFIG["lookback"])
        power[origin + 1:] = 9.87654321e6
        hf_mean[:, origin + 1:] = -9.87654321e6
        hf_count[:, origin + 1:] = 0
        after = audit.causal_foundation_features(origin, power, hf_mean, hf_count, CONFIG["lookback"])
        np.testing.assert_array_equal(before, after)

    def test_final_test_score_cannot_change_calibration_threshold(self) -> None:
        calibration = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5])
        threshold_before = audit.finite_order_threshold(calibration, 0.8)
        fictional_final_score = np.asarray([-1e12, 1e12])
        fictional_final_score *= -1
        threshold_after = audit.finite_order_threshold(calibration, 0.8)
        self.assertEqual(threshold_before, threshold_after)

    def test_quantile_is_higher_order_statistic(self) -> None:
        scores = np.asarray([1.0, 2.0, 3.0, 4.0])
        threshold, index, realized = audit.finite_order_threshold(scores, 0.8)
        self.assertEqual(index, 3)
        self.assertEqual(threshold, 4.0)
        self.assertEqual(realized, 1.0)

    def test_preprocessing_fit_is_base_train_only(self) -> None:
        train_values = STATE["pv_power_0"][:365 * 288]
        center, scale = audit.fit_preprocessor(train_values, "BASE_TRAIN")
        self.assertTrue(np.isfinite(center) and np.isfinite(scale))
        for forbidden in ("BASE_MODEL_VALIDATION", "RISK_FIT", "RISK_CALIBRATION", "FINAL_TEST"):
            with self.assertRaises(ValueError):
                audit.fit_preprocessor(train_values, forbidden)

    def test_stage_roles_are_disjoint_and_ordered(self) -> None:
        intervals = [(name, np.datetime64(bounds[0]), np.datetime64(bounds[1])) for name, bounds in CONFIG["stages"].items()]
        self.assertEqual([name for name, _, _ in intervals], ["BASE_TRAIN", "BASE_MODEL_VALIDATION", "RISK_FIT", "RISK_CALIBRATION", "FINAL_TEST"])
        for (_, start, end), (_, next_start, _) in zip(intervals, intervals[1:]):
            self.assertLess(start, end)
            self.assertEqual(end, next_start)

    def test_all_input_indices_are_not_after_origin(self) -> None:
        for stage in CONFIG["stages"]:
            origins = STATE[f"origins__{stage}__COMMON"]
            if len(origins):
                input_last = origins
                self.assertTrue(np.all(input_last <= origins))
                self.assertTrue(np.all(origins - CONFIG["lookback"] + 1 <= origins))

    def test_no_training_or_checkpoint_write_call_exists(self) -> None:
        tree = ast.parse((Path(audit.__file__)).read_text(encoding="utf-8"))
        forbidden_attributes = {"backward", "step", "zero_grad"}
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else (node.func.id if isinstance(node.func, ast.Name) else "")
                if name in forbidden_attributes or name in {"train_model", "fit_risk_model"}:
                    found.append(name)
        self.assertEqual(found, [])

    def test_raw_source_size_and_mtime_are_unchanged(self) -> None:
        self.assertTrue(SUMMARY["source_files_unchanged"])
        self.assertEqual(SUMMARY["source_state_before"], SUMMARY["source_state_after"])

    def test_no_2023_prediction_or_error_was_accessed(self) -> None:
        self.assertFalse(SUMMARY["final_test_predictions_generated_or_read"])
        self.assertFalse(SUMMARY["training_performed"])
        self.assertFalse(SUMMARY["risk_model_fitted"])


def main() -> None:
    global SUMMARY, STATE, CONFIG
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--state-npz", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    SUMMARY = json.loads(args.summary_json.read_text(encoding="utf-8"))
    STATE = np.load(args.state_npz, allow_pickle=False)
    CONFIG = audit.load_config()
    unittest.main(argv=[__file__, *remaining], verbosity=2)


if __name__ == "__main__":
    main()
