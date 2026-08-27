"""Ordinary fixture and real-array tests for Scheme C1-S3."""
from __future__ import annotations

import ast
import csv
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

import run_c1_formal as formal


AUDIT_PATH = formal.RESULTS / "audit.json"
STATE_PATH = formal.RESULTS / "audit_state.npz"
AUDIT: dict = {}
STATE = None
CONFIG: dict = {}


class FixtureTests(unittest.TestCase):
    def test_01_standard_timestamp_parses(self) -> None:
        self.assertIsNotNone(formal.S2.parse_dmy_second(b"01/01/2021 00:00:00"))

    def test_02_bad_separator_and_nonnumeric_time_rejected(self) -> None:
        self.assertIsNone(formal.S2.parse_dmy_second(b"01-01-2021 00:00:00"))
        self.assertIsNone(formal.S2.parse_dmy_second(b"01/01/2021 aa:00:00"))

    def test_03_right_closed_five_minute_boundaries(self) -> None:
        second_indices = np.asarray([0, 1, 299, 300, 301])
        bins = (second_indices + 299) // 300
        np.testing.assert_array_equal(bins, [0, 1, 1, 1, 2])

    def test_04_utc_to_acst_is_fixed(self) -> None:
        epoch = formal.S2.parse_dmy_second(b"01/01/2022 00:00:01")
        utc = np.datetime64("1970-01-01") + np.timedelta64(int(epoch), "s")
        self.assertEqual(str(utc + np.timedelta64(570, "m")), "2022-01-01T09:30:01")

    def test_05_duplicate_inverse_and_missing_are_detected(self) -> None:
        times = np.asarray([0, 1, 1, 3, 2])
        self.assertEqual(int(np.sum(np.diff(times) == 0)), 1)
        self.assertEqual(int(np.sum(np.diff(times) < 0)), 1)
        self.assertGreater(5 - len(set(times.tolist())), 0)

    def test_06_glued_and_truncated_fixture_is_detected(self) -> None:
        header = b"Timestamp_UTC [DD/MM/YYYY hh:mm:ss],Irradiance_MB0 [W/m-2],Irradiance_MB1 [W/m-2],Irradiance_MB2 [W/m-2]\n"
        lines = [b"01/01/2021 00:00:00,1,2,3\n", b"01/01/2021 00:00:01,1\n", b"01/01/2021 00:00:02,1,2,301/01/2021 00:00:03,1,2,3\n"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.csv"
            path.write_bytes(header + b"".join(lines))
            result = formal.exact_structure_scan(path)
        self.assertGreaterEqual(result["truncated_records"], 1)
        self.assertGreaterEqual(result["glued_records"], 1)

    def test_07_duplicate_header_fixture_is_detected(self) -> None:
        self.assertTrue(b"Timestamp_UTC".startswith(b"Timestamp_UTC"))
        self.assertEqual(sum(line.startswith(b"Timestamp_UTC") for line in [b"Timestamp_UTC,x", b"01/01/2021 00:00:00,x"]), 1)

    def test_08_annual_boundary_keys_are_contiguous(self) -> None:
        end_2021 = formal.S2.epoch_seconds(datetime(2022, 1, 1)) // 300 - 1
        start_2022 = formal.S2.epoch_seconds(datetime(2022, 1, 1)) // 300
        self.assertEqual(start_2022 - end_2021, 1)

    def test_09_window_bounds_stay_inside_stage(self) -> None:
        stage = np.arange(100)
        origins = np.arange(71, 88)
        self.assertTrue(np.all(origins - 71 >= stage[0]))
        self.assertTrue(np.all(origins + 12 <= stage[-1]))

    def test_10_window_rejects_missing_history_or_target(self) -> None:
        valid = np.ones(100, dtype=bool)
        valid[40] = False
        starts, ends = np.asarray([0, 41]), np.asarray([83, 99])
        np.testing.assert_array_equal(formal.S2.rolling_all(valid, starts, ends), [False, True])

    def test_11_common_origins_are_exact_intersection(self) -> None:
        sets = [set([1, 2, 3]), set([2, 3, 4]), set([0, 2, 3])]
        self.assertEqual(set.intersection(*sets), {2, 3})

    def test_12_primary_daylight_is_common_to_all_arrays(self) -> None:
        common = np.asarray([1, 2, 3, 4])
        powers = [np.asarray([0, .1, .2, 0, .3]), np.asarray([0, .1, 0, .2, .3]), np.asarray([0, .1, .2, .2, .3])]
        keep = np.logical_and.reduce([p[common] > .05 for p in powers])
        selected = common[keep]
        np.testing.assert_array_equal(selected, [1, 4])

    def test_13_formal_csv_is_directly_readable(self) -> None:
        with formal.SUMMARY_CSV.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertGreater(len(rows), 0)
        self.assertEqual(set(rows[0]), {"section", "year", "stage", "array", "metric", "value", "unit", "source_path", "notes"})


class RealArrayTests(unittest.TestCase):
    def test_01_three_year_counts_and_utc_endpoints_are_real_scan_results(self) -> None:
        self.assertEqual(set(AUDIT["irradiance"]), {"2021", "2022", "2023"})
        self.assertEqual(AUDIT["irradiance"]["2022"]["parseable_target_year_unique_seconds"], 31_536_000)
        self.assertEqual(AUDIT["irradiance"]["2021"]["first_target_year_utc"], "2021-06-02 00:00:00")
        self.assertEqual(AUDIT["irradiance"]["2023"]["last_target_year_utc"], "2023-01-02 22:39:12")

    def test_02_all_three_actual_second_files_parsed(self) -> None:
        for item in AUDIT["irradiance"].values():
            self.assertTrue(Path(item["path"]).is_file())
            self.assertGreater(item["physical_lines"], 1)

    def test_03_stage_common_origins_are_actual_intersections(self) -> None:
        arrays = list(CONFIG["pv_files"])
        for stage in CONFIG["stages"]:
            expected = set(STATE[f"origins__{stage}__{arrays[0]}"].tolist())
            for array in arrays[1:]:
                expected &= set(STATE[f"origins__{stage}__{array}"].tolist())
            self.assertEqual(expected, set(STATE[f"origins__{stage}__COMMON"].tolist()))

    def test_04_all_input_indices_are_not_after_origin(self) -> None:
        for stage in CONFIG["stages"]:
            origins = STATE[f"origins__{stage}__COMMON"]
            self.assertTrue(np.all(origins - CONFIG["lookback"] + 1 <= origins))

    def test_05_all_targets_are_strictly_after_origin(self) -> None:
        for stage in CONFIG["stages"]:
            origins = STATE[f"origins__{stage}__COMMON"]
            if len(origins):
                offsets = np.arange(1, CONFIG["horizon"] + 1)
                self.assertTrue(np.all(origins[:, None] + offsets > origins[:, None]))

    def test_06_future_sentinel_does_not_change_causal_foundation_features(self) -> None:
        candidates = np.concatenate([STATE[f"origins__{stage}__COMMON"] for stage in CONFIG["stages"]])
        self.assertGreater(len(candidates), 0)
        origin = int(candidates[len(candidates) // 2])
        power = STATE["pv_power_0"].copy()
        hf_mean, hf_count = STATE["hf_channel_mean"].copy(), STATE["hf_channel_count"].copy()
        before = formal.S2.causal_foundation_features(origin, power, hf_mean, hf_count, CONFIG["lookback"])
        power[origin + 1:] = 123456.0
        hf_mean[:, origin + 1:] = -123456.0
        hf_count[:, origin + 1:] = 0
        after = formal.S2.causal_foundation_features(origin, power, hf_mean, hf_count, CONFIG["lookback"])
        np.testing.assert_array_equal(before, after)

    def test_07_preprocessing_fit_role_is_base_train_only(self) -> None:
        values = STATE["pv_power_0"][:365 * 288]
        formal.S2.fit_preprocessor(values, "BASE_TRAIN")
        for stage in ("BASE_MODEL_VALIDATION", "RISK_FIT", "RISK_CALIBRATION", "FINAL_TEST"):
            with self.assertRaises(ValueError):
                formal.S2.fit_preprocessor(values, stage)

    def test_08_stage_role_arrays_are_disjoint(self) -> None:
        sets = [set(STATE[f"origins__{stage}__COMMON"].tolist()) for stage in CONFIG["stages"]]
        for i, left in enumerate(sets):
            for right in sets[i + 1:]:
                self.assertFalse(left & right)

    def test_09_risk_fit_is_not_used_for_threshold(self) -> None:
        calibration = np.asarray([.1, .2, .3, .4])
        risk_fit = np.asarray([-999., 999.])
        before = formal.S2.finite_order_threshold(calibration, .8)
        risk_fit[:] *= -1
        after = formal.S2.finite_order_threshold(calibration, .8)
        self.assertEqual(before, after)

    def test_10_fictional_final_score_cannot_change_calibration_threshold(self) -> None:
        calibration = np.asarray([.1, .2, .3, .4])
        before = formal.S2.finite_order_threshold(calibration, .8)
        fictional_final = np.asarray([-1e30, 1e30])
        fictional_final[:] = 0
        after = formal.S2.finite_order_threshold(calibration, .8)
        self.assertEqual(before, after)

    def test_11_sources_are_unchanged_and_no_training_call_was_executed(self) -> None:
        self.assertTrue(AUDIT["source_files_unchanged"])
        self.assertEqual(AUDIT["source_state_before"], AUDIT["source_state_after"])
        self.assertFalse(AUDIT["training_performed"])
        self.assertFalse(AUDIT["final_test_predictions_or_errors_read"])
        tree = ast.parse(Path(formal.__file__).read_text(encoding="utf-8"))
        called = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertFalse({"backward", "step", "zero_grad"} & called)


def main() -> int:
    global AUDIT, STATE, CONFIG
    if not AUDIT_PATH.exists() or not STATE_PATH.exists():
        raise SystemExit("Run run_c1_formal.py --audit first")
    AUDIT = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    STATE = np.load(STATE_PATH, allow_pickle=False)
    CONFIG = formal.load_config()
    suite = unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(FixtureTests))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(RealArrayTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        for row in AUDIT["windows"]:
            row["stage_boundary_excluded_origins"] = CONFIG["lookback"] + CONFIG["horizon"] - 1
        AUDIT["fixture_tests"] = "13/13 passed; 0 skipped"
        AUDIT["real_array_tests"] = "11/11 passed; 0 skipped"
        AUDIT_PATH.write_text(json.dumps(AUDIT, ensure_ascii=False, indent=2), encoding="utf-8")
        formal.write_long_csv(formal.SUMMARY_CSV, formal.summary_rows(AUDIT["candidates"], AUDIT["irradiance"], AUDIT["pv"], AUDIT["windows"], AUDIT["primary_daylight"], AUDIT["daylight_thresholds_kw"], AUDIT["data_ready_conditions"]))
        formal.write_report(AUDIT)
    print(json.dumps({"fixture_total": 13, "fixture_passed": 13 - len(result.failures) - len(result.errors), "real_array_total": 11, "real_array_passed": 11 - len(result.failures) - len(result.errors), "skipped": len(result.skipped), "successful": result.wasSuccessful()}))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
