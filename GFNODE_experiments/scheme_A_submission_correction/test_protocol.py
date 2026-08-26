"""Ordinary implementation tests for the Scheme A submission correction."""
from __future__ import annotations

import inspect
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_corrected_benchmark as bench


class ConstantModel(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.as_tensor(values, dtype=torch.float32))

    def forward(self, x):
        return self.values[:x.shape[0]]


class CorrectedProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = bench.load_config()
        cls.protocols = bench.preflight(cls.cfg)

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
        model = ConstantModel(torch.zeros_like(y))
        measured = bench.global_masked_mse(model, loader, torch.device("cpu"))
        expected = float(torch.mean(y.square()))
        batch_mean_average = float((torch.mean(y[:2].square()) + torch.mean(y[2:].square())) / 2)
        self.assertAlmostEqual(measured, expected, places=6)
        self.assertNotAlmostEqual(measured, batch_mean_average, places=4)

    def test_07_training_api_has_no_test_loader(self):
        parameters = inspect.signature(bench.train_model).parameters
        self.assertNotIn("test_loader", parameters)
        self.assertNotIn("test", parameters)

    def test_08_all_models_output_h144_and_backward(self):
        sample = torch.randn(2, 72, 17)
        for name in bench.MODEL_NAMES:
            bench.set_seed(42)
            model = bench.make_model(name, 17, self.cfg)
            output = model(sample)
            self.assertEqual(tuple(output.shape), (2, 144))
            output.square().mean().backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_09_horizon_specific_model_persistence_counts_match(self):
        for protocol in self.protocols.values():
            bundle = protocol.evaluation_windows["test"]
            last, _ = bench.persistence_arrays(protocol, bundle)
            for horizon in self.cfg["evaluation_horizons"]:
                for scope in ("regular_full_timeline", "daylight"):
                    _, mask = bench.evaluation_mask(bundle, horizon, scope,
                                                    protocol.daylight_threshold, False)
                    model_count = int((mask & np.isfinite(bundle.y_raw[:, :horizon])).sum())
                    persistence_count = int((mask & np.isfinite(last[:, :horizon])).sum())
                    self.assertEqual(model_count, persistence_count)

    def test_10_origin_equals_target_start_minus_five_minutes(self):
        for protocol in self.protocols.values():
            for bundle in protocol.evaluation_windows.values():
                self.assertTrue(np.all(bundle.forecast_origin == bundle.target_start - np.timedelta64(5, "m")))

    def test_11_horizon_specific_windows_stay_inside_split(self):
        for protocol in self.protocols.values():
            for split, bundle in protocol.evaluation_windows.items():
                split_end = np.datetime64(self.cfg["splits"][split][1])
                for horizon in self.cfg["evaluation_horizons"]:
                    eligible = bundle.target_valid[:, :horizon].all(axis=1)
                    end = bundle.target_start[eligible] + np.timedelta64((horizon - 1) * 5, "m")
                    self.assertTrue(np.all(end <= split_end))

    def test_12_inputs_and_fitted_arrays_are_finite(self):
        for protocol in self.protocols.values():
            self.assertTrue(np.isfinite(protocol.train_windows.x).all())
            self.assertTrue(np.isfinite(protocol.validation_windows.x).all())
            self.assertTrue(np.isfinite(protocol.evaluation_windows["test"].x).all())

    def test_13_daylight_is_evaluation_only(self):
        for protocol in self.protocols.values():
            expected = 0.01 * protocol.train_target_max
            self.assertAlmostEqual(protocol.daylight_threshold, expected)
            self.assertNotIn("daylight", "|".join(protocol.feature_columns).lower())

    def test_14_source_files_are_not_written(self):
        before = {p.data_path: (p.data_path.stat().st_size, p.data_path.stat().st_mtime_ns)
                  for p in self.protocols.values()}
        for path, signature in before.items():
            self.assertEqual((path.stat().st_size, path.stat().st_mtime_ns), signature)

    def test_15_completed_artifacts_are_finite_and_fair(self):
        completed = list(bench.RESULTS.glob("*/completed.json"))
        if not completed:
            self.skipTest("full experiment has not completed yet")
        self.assertEqual(len(completed), 36)
        references = {}
        for marker in completed:
            artifact = marker.parent / "test_H144.npz"
            self.assertTrue(artifact.exists())
            with np.load(artifact) as saved:
                prediction = saved["predictions"]
                labels = saved["labels"]
                valid = saved["target_valid"]
                origins = saved["forecast_origin"]
                starts = saved["target_start"]
            self.assertEqual(prediction.shape[1], 144)
            self.assertTrue(np.isfinite(prediction).all())
            self.assertTrue(np.all(starts - origins == np.timedelta64(5, "m")))
            dataset = marker.parent.name.rsplit("_", 2)[-2]
            if dataset not in references:
                references[dataset] = (labels, valid, origins)
            else:
                ref = references[dataset]
                self.assertTrue(np.array_equal(labels, ref[0], equal_nan=True))
                self.assertTrue(np.array_equal(valid, ref[1]))
                self.assertTrue(np.array_equal(origins, ref[2]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
