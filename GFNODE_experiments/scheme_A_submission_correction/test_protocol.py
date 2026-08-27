"""Ordinary and completed-artifact tests for the final Scheme A evidence audit."""
from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_corrected_benchmark as bench

_PROTOCOLS = None


def protocols():
    global _PROTOCOLS
    if _PROTOCOLS is None:
        _PROTOCOLS = bench.preflight(bench.load_config())
    return _PROTOCOLS


class ConstantModel(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.as_tensor(values, dtype=torch.float32))

    def forward(self, x):
        return self.values[:x.shape[0]]


class OrdinaryUnitTests(unittest.TestCase):
    """Tests that do not require the local results directory."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = bench.load_config()
        cls.protocols = protocols()

    def test_01_audited_feature_order_and_dimensions(self):
        expected = [
            "Performance_Ratio", "Weather_Temperature_Celsius", "Weather_Relative_Humidity",
            "Global_Horizontal_Radiation", "Diffuse_Horizontal_Radiation",
            "Radiation_Global_Tilted", "Radiation_Diffuse_Tilted",
        ]
        self.assertEqual(self.cfg["original_feature_columns"], expected)
        for protocol in self.protocols.values():
            self.assertEqual(protocol.feature_columns, expected + ["Active_Power"])
            self.assertEqual(protocol.train_windows.x.shape[-1], 17)

    def test_02_preprocessors_fit_train_only(self):
        for protocol in self.protocols.values():
            self.assertEqual({entry["split"] for entry in protocol.fit_log}, {"train"})
            self.assertEqual(len(protocol.fit_log), 4)

    def test_03_splits_do_not_overlap(self):
        for protocol in self.protocols.values():
            indexes = {name: set(frame.index) for name, frame in protocol.transformed.items()}
            self.assertFalse(indexes["train"] & indexes["validation"])
            self.assertFalse(indexes["train"] & indexes["test"])
            self.assertFalse(indexes["validation"] & indexes["test"])

    def test_04_strict_five_minute_chronology(self):
        for protocol in self.protocols.values():
            for bundle in (protocol.train_windows, protocol.validation_windows,
                           protocol.evaluation_windows["test"]):
                self.assertTrue(np.all(bundle.target_start - bundle.forecast_origin == np.timedelta64(5, "m")))
                self.assertTrue(np.all(bundle.forecast_origin - bundle.input_start == np.timedelta64(355, "m")))

    def test_05_future_active_power_is_not_an_input(self):
        for protocol in self.protocols.values():
            bundle = protocol.evaluation_windows["test"]
            self.assertTrue(np.all(bundle.forecast_origin < bundle.target_start))
            self.assertEqual(bundle.x.shape[1], self.cfg["lookback"])
            self.assertNotIn("future", "|".join(protocol.feature_columns).lower())

    def test_06_validation_is_global_sse_over_target_count(self):
        x = torch.zeros((3, 1, 1))
        y = torch.tensor([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]])
        mask = torch.ones_like(y, dtype=torch.bool)
        loader = DataLoader(TensorDataset(x, y, mask), batch_size=2, shuffle=False)
        measured = bench.global_masked_mse(ConstantModel(torch.zeros_like(y)), loader, torch.device("cpu"))
        expected = float(torch.mean(y.square()))
        batch_mean_average = float((torch.mean(y[:2].square()) + torch.mean(y[2:].square())) / 2)
        self.assertAlmostEqual(measured, expected, places=6)
        self.assertNotAlmostEqual(measured, batch_mean_average, places=4)

    def test_07_training_api_has_no_test_loader(self):
        parameters = inspect.signature(bench.train_model).parameters
        self.assertNotIn("test_loader", parameters)
        self.assertNotIn("test", parameters)

    def test_08_all_models_output_h144_without_gradient_execution(self):
        sample = torch.randn(2, 72, 17)
        with torch.inference_mode():
            for name in bench.MODEL_NAMES:
                model = bench.make_model(name, 17, self.cfg).eval()
                self.assertEqual(tuple(model(sample).shape), (2, 144))

    def test_09_horizon_specific_model_persistence_counts_match(self):
        for protocol in self.protocols.values():
            bundle = protocol.evaluation_windows["test"]
            last, _ = bench.persistence_arrays(protocol, bundle)
            for horizon in self.cfg["evaluation_horizons"]:
                for scope in ("regular_full_timeline", "daylight"):
                    _, mask = bench.evaluation_mask(bundle, horizon, scope,
                                                    protocol.daylight_threshold, False)
                    self.assertEqual(int((mask & np.isfinite(bundle.y_raw[:, :horizon])).sum()),
                                     int((mask & np.isfinite(last[:, :horizon])).sum()))

    def test_10_origin_equals_target_start_minus_five_minutes(self):
        for protocol in self.protocols.values():
            for bundle in protocol.evaluation_windows.values():
                self.assertTrue(np.all(bundle.forecast_origin == bundle.target_start - np.timedelta64(5, "m")))

    def test_11_horizon_specific_targets_stay_inside_test(self):
        for protocol in self.protocols.values():
            bundle = protocol.evaluation_windows["test"]
            split_start = np.datetime64(self.cfg["splits"]["test"][0])
            split_end = np.datetime64(self.cfg["splits"]["test"][1])
            for horizon in self.cfg["evaluation_horizons"]:
                eligible = bundle.target_valid[:, :horizon].all(axis=1)
                end = bundle.target_start[eligible] + np.timedelta64((horizon - 1) * 5, "m")
                self.assertTrue(np.all(bundle.target_start[eligible] >= split_start))
                self.assertTrue(np.all(end <= split_end))

    def test_12_validation_checkpoint_uses_complete_h144_windows(self):
        for protocol in self.protocols.values():
            self.assertEqual(protocol.validation_windows.y_raw.shape[1], 144)
            self.assertTrue(protocol.validation_windows.target_valid.all())

    def test_13_inputs_for_preprocessing_are_finite(self):
        for protocol in self.protocols.values():
            self.assertTrue(np.isfinite(protocol.train_windows.x).all())
            self.assertTrue(np.isfinite(protocol.validation_windows.x).all())
            self.assertTrue(np.isfinite(protocol.evaluation_windows["test"].x).all())

    def test_14_daylight_is_evaluation_only(self):
        for protocol in self.protocols.values():
            self.assertAlmostEqual(protocol.daylight_threshold, 0.01 * protocol.train_target_max)
            self.assertNotIn("daylight", "|".join(protocol.feature_columns).lower())

    def test_15_daily_matched_mask_removes_daily_missing_points_for_every_method(self):
        for protocol in self.protocols.values():
            bundle = protocol.evaluation_windows["test"]
            _, daily = bench.persistence_arrays(protocol, bundle)
            for horizon in self.cfg["evaluation_horizons"]:
                for scope in ("regular_full_timeline", "daylight"):
                    mask = bench.daily_matched_point_mask(
                        bundle, daily, horizon, scope, protocol.daylight_threshold
                    )
                    self.assertFalse(np.any(mask & ~np.isfinite(daily[:, :horizon])))

    def test_16_stale_array_is_rejected(self):
        with self.assertRaises(bench.StaleArtifactError):
            bench._array_equal("labels", np.array([1.0]), np.array([2.0]))


class ArtifactIntegrityTests(unittest.TestCase):
    """Tests that require all 36 local completed runs; absence is a failure, not a skip."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = bench.load_config()
        cls.protocols = protocols()
        cls.root = bench.RESULTS
        cls.markers = sorted(cls.root.glob("*/completed.json"))
        if len(cls.markers) != 36:
            raise AssertionError(f"Expected 36 completed artifacts, found {len(cls.markers)} at {cls.root}")
        cls.signatures = {
            p: (p.stat().st_size, p.stat().st_mtime_ns)
            for marker in cls.markers
            for p in (marker, marker.parent / "test_H144.npz", marker.parent / "best_validation.pt")
        }

    def test_01_all_run_identities_and_checkpoint_inputs_match(self):
        for marker in self.markers:
            info = json.loads(marker.read_text(encoding="utf-8"))
            _, _, model = bench.validate_completed_artifact(
                self.protocols[info["dataset"]], info["model"], int(info["seed"]),
                self.cfg, torch.device("cpu"), marker.parent, reforward=False,
            )
            self.assertEqual(bench.parameter_count(model), int(info["parameter_count"]))

    def test_02_saved_arrays_equal_current_protocol(self):
        for marker in self.markers:
            info = json.loads(marker.read_text(encoding="utf-8"))
            bundle = self.protocols[info["dataset"]].evaluation_windows["test"]
            with np.load(marker.parent / "test_H144.npz") as saved:
                self.assertTrue(np.array_equal(saved["labels"], bundle.y_raw, equal_nan=True))
                self.assertTrue(np.array_equal(saved["target_valid"], bundle.target_valid))
                self.assertTrue(np.array_equal(saved["forecast_origin"].astype("datetime64[ns]"),
                                               bundle.forecast_origin.astype("datetime64[ns]")))
                self.assertTrue(np.array_equal(saved["target_start"].astype("datetime64[ns]"),
                                               bundle.target_start.astype("datetime64[ns]")))

    def test_03_predictions_are_finite_h144(self):
        for marker in self.markers:
            with np.load(marker.parent / "test_H144.npz") as saved:
                self.assertEqual(saved["predictions"].shape[1], 144)
                self.assertTrue(np.isfinite(saved["predictions"]).all())

    def test_04_checkpoint_is_validation_selected(self):
        for marker in self.markers:
            info = json.loads(marker.read_text(encoding="utf-8"))
            state = torch.load(marker.parent / "best_validation.pt", map_location="cpu", weights_only=False)
            self.assertEqual(int(state["epoch"]), int(info["best_epoch"]))
            self.assertAlmostEqual(float(state["validation_global_mse"]),
                                   float(info["best_validation_mse"]), places=10)

    def test_05_all_models_share_labels_origins_and_masks_by_dataset(self):
        references = {}
        for marker in self.markers:
            info = json.loads(marker.read_text(encoding="utf-8"))
            with np.load(marker.parent / "test_H144.npz") as saved:
                arrays = (saved["labels"].copy(), saved["target_valid"].copy(),
                          saved["forecast_origin"].copy())
            if info["dataset"] not in references:
                references[info["dataset"]] = arrays
            else:
                ref = references[info["dataset"]]
                self.assertTrue(np.array_equal(arrays[0], ref[0], equal_nan=True))
                self.assertTrue(np.array_equal(arrays[1], ref[1]))
                self.assertTrue(np.array_equal(arrays[2], ref[2]))

    def test_06_daily_matched_counts_are_identical_for_every_method(self):
        metrics = bench.pd.read_csv(bench.METRICS_PATH)
        q = metrics[
            metrics.analysis.eq("supplementary_daily_matched") &
            metrics.metric.eq("RMSE") &
            metrics.statistic.isin(["per_seed", "deterministic"])
        ]
        for _, group in q.groupby(["dataset", "horizon_steps", "scope"]):
            self.assertEqual(group.forecast_origin_count.nunique(), 1)
            self.assertEqual(group.valid_target_count.nunique(), 1)
            self.assertEqual(len(group), 14)

    def test_07_recomputed_metrics_are_finite(self):
        metrics = bench.pd.read_csv(bench.METRICS_PATH)
        self.assertTrue(np.isfinite(metrics["value"].astype(float)).all())

    def test_08_no_protected_artifact_was_modified(self):
        for path, signature in self.signatures.items():
            self.assertEqual((path.stat().st_size, path.stat().st_mtime_ns), signature)

    def test_09_completed_artifact_reuse_never_calls_training(self):
        marker = self.markers[0]
        info = json.loads(marker.read_text(encoding="utf-8"))
        original = bench.train_model
        bench.train_model = lambda *args, **kwargs: self.fail("train_model was called")
        try:
            bench.run_one(self.protocols[info["dataset"]], info["model"], int(info["seed"]),
                          self.cfg, torch.device("cpu"), allow_training=False)
        finally:
            bench.train_model = original


class IndependentEvidenceTests(unittest.TestCase):
    """Checks for the separate NumPy/Pandas evidence implementation and final PDF."""

    @classmethod
    def setUpClass(cls):
        cls.script_path = HERE / "independent_verify_evidence.py"
        cls.audit_path = HERE / "INDEPENDENT_EVIDENCE_AUDIT.json"
        if not cls.audit_path.is_file():
            raise AssertionError("Independent audit JSON is missing")
        cls.source = cls.script_path.read_text(encoding="utf-8")
        cls.audit = json.loads(cls.audit_path.read_text(encoding="utf-8"))
        cls.manuscript = HERE.parents[1] / "manuscript/clean_pv_benchmark"

    def test_01_independent_script_does_not_import_production_metrics(self):
        self.assertNotIn("import run_corrected_benchmark", self.source)
        self.assertNotIn("from run_corrected_benchmark", self.source)
        for name in ("metric_values", "evaluation_mask", "daily_matched_point_mask",
                     "evaluate_prediction", "evaluate_daily_matched", "aggregate_neural_rows",
                     "add_rank_rows", "write_metrics", "evidence_only"):
            self.assertNotIn(f"bench.{name}", self.source)

    def test_02_independent_script_has_no_training_calls(self):
        for token in (".backward(", "optimizer.step(", "train_model(", "torch.optim"):
            self.assertNotIn(token, self.source)
        self.assertFalse(self.audit["training_executed"])

    def test_03_primary_combinations_complete(self):
        self.assertEqual(self.audit["primary_combination_count"], 24)

    def test_04_daily_matched_combinations_complete(self):
        self.assertEqual(self.audit["daily_matched_combination_count"], 24)

    def test_05_daily_matched_counts_are_common(self):
        self.assertEqual(self.audit["failed_comparisons"], 0)
        count_checks = [item for item in self.audit["comparisons"]
                        if item["key"].endswith("|origins") or item["key"].endswith("|points")]
        self.assertGreater(len(count_checks), 0)
        self.assertTrue(all(item["status"] == "PASS" for item in count_checks))

    def test_06_daily_win_count_is_22_of_24(self):
        self.assertEqual(self.audit["daily_matched_daily_wins"], 22)
        self.assertEqual(self.audit["daily_matched_neural_wins"], 2)

    def test_07_neural_daily_wins_are_hanwha_h12(self):
        identities = {(item["dataset"], item["horizon"], item["scope"])
                      for item in self.audit["daily_matched_neural_win_details"]}
        self.assertEqual(identities, {
            ("Hanwha", 12, "regular_full_timeline"), ("Hanwha", 12, "daylight")})

    def test_08_qcells_h12_counts(self):
        q = self.audit["qcells_h12"]
        self.assertEqual((q["origins"], q["full_target_points"],
                          q["daylight_target_points"]), (6463, 77556, 36504))

    def test_09_all_csv_comparisons_pass(self):
        self.assertEqual(self.audit["verdict"], "INDEPENDENT_EVIDENCE_PASS")
        self.assertEqual(self.audit["comparison_count"], self.audit["passed_comparisons"])
        self.assertEqual(self.audit["failed_comparisons"], 0)

    def test_10_protected_sources_are_unchanged(self):
        self.assertFalse(self.audit["checkpoint_modified"])
        self.assertFalse(self.audit["prediction_artifact_modified"])
        self.assertFalse(self.audit["raw_data_modified"])

    def test_11_final_pdf_fonts_are_embedded(self):
        pdf = self.manuscript / "main.pdf"
        result = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True,
                                encoding="utf-8", errors="replace", check=True)
        rows = [line.split() for line in result.stdout.splitlines()[2:] if line.strip()]
        self.assertGreater(len(rows), 0)
        self.assertTrue(all("yes" in [cell.lower() for cell in row] for row in rows))

    def test_12_efficiency_memory_unit_is_mib(self):
        tex = (self.manuscript / "main.tex").read_text(encoding="utf-8")
        self.assertIn("Peak GPU memory (MiB)", tex)
        self.assertNotIn("Peak GPU memory (MB)", tex)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-only", action="store_true")
    parser.add_argument("--artifact-only", action="store_true")
    parser.add_argument("--independent-only", action="store_true")
    args = parser.parse_args()
    if sum((args.unit_only, args.artifact_only, args.independent_only)) > 1:
        raise SystemExit("Choose only one test subset")
    loader = unittest.TestLoader()
    if args.unit_only:
        suite = loader.loadTestsFromTestCase(OrdinaryUnitTests)
    elif args.artifact_only:
        suite = loader.loadTestsFromTestCase(ArtifactIntegrityTests)
    elif args.independent_only:
        suite = loader.loadTestsFromTestCase(IndependentEvidenceTests)
    else:
        suite = unittest.TestSuite([
            loader.loadTestsFromTestCase(OrdinaryUnitTests),
            loader.loadTestsFromTestCase(ArtifactIntegrityTests),
            loader.loadTestsFromTestCase(IndependentEvidenceTests),
        ])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
