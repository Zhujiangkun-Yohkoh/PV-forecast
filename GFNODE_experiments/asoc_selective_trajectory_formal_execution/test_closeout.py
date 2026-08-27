"""Side-effect-free tests for the terminal Scheme C1 archive state."""
from __future__ import annotations

import contextlib
import csv
import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import c1_formal_pipeline as pipeline
import run_c1_formal as formal


ROOT = Path(__file__).resolve().parent
DECISION = ROOT / "decision.json"
REPORT = ROOT / "REPORT.md"
PER_SEED = ROOT / "metrics_per_seed.csv"
SUMMARY = ROOT / "metrics_summary_mean_sd.csv"
TERMINAL_FILES = (DECISION, REPORT, PER_SEED, SUMMARY)
EXPECTED_STATUS = "C1_ROUTE_CLOSED_DATA_UNAVAILABLE"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def invoke_cli(*args: str) -> dict[str, object]:
    output = io.StringIO()
    with mock.patch.object(sys, "argv", ["run_c1_formal.py", *args]), contextlib.redirect_stdout(output):
        formal.main()
    return json.loads(output.getvalue())


class CloseoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.before = {path: path.read_bytes() for path in TERMINAL_FILES}
        cls.decision = json.loads(DECISION.read_text(encoding="utf-8"))
        cls.runs = read_csv(PER_SEED)
        cls.summary = read_csv(SUMMARY)
        cls.report = REPORT.read_text(encoding="utf-8")

    def test_01_terminal_decision(self):
        self.assertEqual(self.decision["route_status"], EXPECTED_STATUS)
        self.assertEqual(self.decision["decision"], EXPECTED_STATUS)
        self.assertFalse(self.decision["data_ready"])

    def test_02_method_not_evaluated(self):
        self.assertEqual(self.decision["scientific_method_outcome"], "NOT_EVALUATED")
        self.assertFalse(self.decision["closure_is_method_failure"])
        self.assertFalse(self.decision["closure_is_forecaster_failure"])

    def test_03_no_future_execution(self):
        self.assertFalse(self.decision["future_gpu_execution_authorized"])
        self.assertEqual((self.decision["completed_runs"], self.decision["expected_runs"]), (0, 9))

    def test_04_nine_unique_runs(self):
        combos = {(row["array"], int(row["seed"])) for row in self.runs}
        self.assertEqual(len(self.runs), 9)
        self.assertEqual(len(combos), 9)
        self.assertEqual({int(row["seed"]) for row in self.runs}, {42, 43, 44})

    def test_05_all_runs_not_run(self):
        self.assertEqual({row["run_status"] for row in self.runs}, {"NOT_RUN"})
        self.assertEqual({row["reason"] for row in self.runs}, {EXPECTED_STATUS})

    def test_06_final_test_never_accessed(self):
        self.assertEqual({row["final_test_accessed"].lower() for row in self.runs}, {"false"})
        self.assertFalse(self.decision["final_test_model_predictions_generated"])
        self.assertFalse(self.decision["final_test_prediction_errors_accessed"])

    def test_07_status_records_only(self):
        self.assertEqual({row["record_type"] for row in self.runs}, {"EXECUTION_STATUS_ONLY"})
        forbidden = {"rmse", "mae", "coverage", "aurc"}
        self.assertFalse(forbidden & {name.lower() for name in self.runs[0]})

    def test_08_summary_is_non_scientific(self):
        self.assertEqual(len(self.summary), 1)
        row = self.summary[0]
        self.assertEqual(row["record_type"], "EXECUTION_STATUS_ONLY")
        self.assertEqual((row["completed_runs"], row["expected_runs"]), ("0", "9"))
        self.assertEqual((row["scientific_metrics_available"], row["method_evaluated"]), ("false", "false"))

    def test_09_report_decision_csv_consistency(self):
        self.assertIn(EXPECTED_STATUS, self.report)
        self.assertIn("not a method-performance failure", self.report)
        self.assertIn("Completed runs: 0", self.report)
        self.assertIn("Expected runs: 9", self.report)

    def test_10_saved_audit_arithmetic(self):
        self.assertEqual(18_403_200 + 13_132_800, 31_536_000)
        self.assertEqual(167_952 + 31_368_048, 31_536_000)
        self.assertEqual(58_711 + 2_633 + 43_776, 105_120)
        self.assertEqual(99_211 + 5_909, 105_120)
        self.assertEqual(464 + 96 + 104_560, 105_120)
        self.assertIn("FINAL_TEST | 105,037 | 604 | **0**", self.report)

    def test_11_audit_cli_is_closed_before_audit(self):
        with mock.patch.object(formal, "audit_data", side_effect=AssertionError("audit must not run")):
            state = invoke_cli("--audit")
        self.assertEqual(state["route_status"], EXPECTED_STATUS)
        self.assertFalse(state["training_started"])

    def test_12_formal_cli_is_closed_before_preparation(self):
        with mock.patch.object(formal, "audit_data", side_effect=AssertionError("audit must not run")), \
             mock.patch.object(formal, "prepare_from_audit_state", side_effect=AssertionError("payload must not materialize")), \
             mock.patch.object(formal, "execute_formal", side_effect=AssertionError("execution must not run")):
            state = invoke_cli("--execute-formal", "--authorize-real-execution")
        self.assertEqual(state["route_status"], EXPECTED_STATUS)
        self.assertFalse(state["final_test_accessed"])

    def test_13_pipeline_guard_precedes_scientific_calls(self):
        config = formal.load_config()
        forbidden = ("make_model", "train_base_model", "fit_risk_estimator", "predict_stage")
        patches = [mock.patch.object(pipeline, name, side_effect=AssertionError(f"{name} must not run")) for name in forbidden]
        started = [item.start() for item in patches]
        self.addCleanup(lambda: [item.stop() for item in reversed(patches)])
        self.assertEqual(len(started), len(forbidden))
        sentinel = {"FINAL_TEST": object()}
        state = pipeline.execute_formal(config, sentinel, ROOT / "never-created", True, True)
        self.assertEqual(state["status"], EXPECTED_STATUS)
        self.assertFalse(state["training_started"])
        self.assertFalse((ROOT / "never-created").exists())

    def test_14_config_and_failure_conditions(self):
        self.assertTrue(formal.load_config()["route_closed"])
        expected = {
            "2021_target_year_seconds_incomplete",
            "2023_target_year_seconds_incomplete",
            "2023_structural_anomalies_present",
            "base_train_full_year_coverage_failed",
            "final_test_full_year_coverage_failed",
            "final_test_strict_common_origins_zero",
        }
        self.assertEqual(set(self.decision["failed_readiness_conditions"]), expected)

    def test_15_terminal_files_unchanged_after_real_calls(self):
        after = {path: path.read_bytes() for path in TERMINAL_FILES}
        self.assertEqual(self.before, after)

    def test_16_obsolete_approval_label_absent(self):
        obsolete = "C1_FORMAL_IMPLEMENTATION_" + "READY_FOR_GPU_REVIEW"
        for path in ROOT.glob("*"):
            if path.suffix.lower() in {".py", ".json", ".md", ".csv"}:
                self.assertNotIn(obsolete, path.read_text(encoding="utf-8-sig"), path.name)


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False).result
    raise SystemExit(0 if result.wasSuccessful() else 1)
