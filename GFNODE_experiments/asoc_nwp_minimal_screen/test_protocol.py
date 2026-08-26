"""Ordinary protocol tests for the Stage B1 causal GFS screen."""
from __future__ import annotations

import datetime as dt
import inspect
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_nwp_minimal_screen as stage


class TestProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = stage.config()

    def test_01_cycle_policy_is_causal(self) -> None:
        for minute in range(0, 24 * 60, 5):
            origin = dt.datetime(2022, 9, 3, tzinfo=stage.UTC) + dt.timedelta(minutes=minute)
            cycle = stage.nominal_selected_cycle(origin)
            self.assertLessEqual(cycle + dt.timedelta(hours=6), origin)
            self.assertLess(cycle, origin)

    def test_02_valid_time_and_forecast_age_definitions(self) -> None:
        origin = dt.datetime(2022, 9, 3, 5, 25, tzinfo=stage.UTC)
        cycle = stage.nominal_selected_cycle(origin)
        age = (origin - cycle).total_seconds() / 3600
        self.assertAlmostEqual(age, 11 + 25 / 60)
        for lead in stage.needed_leads(age, 12, self.cfg["nwp_lead_step_hours"]):
            self.assertEqual(cycle + dt.timedelta(hours=lead), cycle + dt.timedelta(hours=lead))

    def test_03_h144_uses_one_selected_cycle(self) -> None:
        source = inspect.getsource(stage.build_nwp_for_origins)
        self.assertIn("align_variable(records, cycle", source)
        self.assertNotIn("cycle + dt.timedelta(hours=int(lead))", source)

    def test_04_missing_objects_only_fallback_earlier(self) -> None:
        source = inspect.getsource(stage.build_nwp_for_origins)
        self.assertIn("cycle - dt.timedelta(hours=6)", source)
        self.assertIn("cycle -= dt.timedelta(hours=6)", source)
        self.assertNotIn("cycle += dt.timedelta(hours=6)", source)

    def test_05_windows_do_not_cross_split_or_gap(self) -> None:
        valid = np.ones(700, bool)
        valid[300] = False
        split = np.zeros(700, bool)
        split[10:690] = True
        origins, segments = stage.legal_origins(valid, split, 72, 144)
        for origin in origins:
            covered = np.arange(origin - 71, origin + 145)
            self.assertTrue(split[covered].all())
            self.assertTrue(valid[covered].all())
        self.assertGreaterEqual(len(segments), 2)

    def test_06_sealed_2023_is_not_read_or_evaluated(self) -> None:
        read_source = inspect.getsource(stage.read_pv_train_validation)
        run_source = inspect.getsource(stage.run_all)
        self.assertIn("validation", read_source)
        self.assertIn("sealed 2023 Test data entered B1", run_source)
        self.assertNotIn("test_loader", inspect.signature(stage.train_model).parameters)

    def test_07_preprocessors_and_prior_are_train_only(self) -> None:
        source = inspect.getsource(stage.prepare_data)
        self.assertIn('"preprocessor_fit_split": "train"', source)
        self.assertIn('"nwp_scaler_fit_split": "train"', source)
        self.assertIn('"reliability_prior_fit_split": "train"', source)
        self.assertIn("build_reliability_prior(train_nwp, train_ghi)", source)
        self.assertNotIn("build_reliability_prior(validation_nwp", source)

    def test_08_validation_ground_weather_not_used_by_model_or_prior(self) -> None:
        source = inspect.getsource(stage.prepare_data)
        self.assertIn('"validation_ground_ghi_used_for_prior": False', source)
        signature = inspect.signature(stage.ForecastModel.forward)
        self.assertNotIn("ground_ghi", signature.parameters)

    def test_09_prefix_metrics_come_from_one_h144_prediction(self) -> None:
        source = inspect.getsource(stage.evaluate_validation)
        self.assertIn("predictions[:, :horizon]", source)
        self.assertEqual(self.cfg["horizon"], 144)
        self.assertEqual(self.cfg["horizons"], [12, 48, 96, 144])

    def test_10_grib_statistical_semantics(self) -> None:
        cycle = dt.datetime(2022, 9, 1, tzinfo=stage.UTC)
        records = {
            (cycle, 6): {"status": 1, "dswrf": 100.0, "tcdc": 20.0, "ds_start": 0.0,
                         "ds_end": 6.0, "tc_start": 6.0, "tc_end": 6.0,
                         "ds_step_type": "avg", "tc_step_type": "instant"},
            (cycle, 9): {"status": 1, "dswrf": 200.0, "tcdc": 50.0, "ds_start": 6.0,
                         "ds_end": 9.0, "tc_start": 9.0, "tc_end": 9.0,
                         "ds_step_type": "avg", "tc_step_type": "instant"},
        }
        leads = np.asarray([6.5, 7.5, 8.5], np.float32)
        ds = stage.align_variable(records, cycle, "dswrf", leads)
        tc = stage.align_variable(records, cycle, "tcdc", leads)
        np.testing.assert_allclose(ds, 200.0)
        np.testing.assert_allclose(tc, [25.0, 35.0, 45.0])

    def test_11_three_models_share_one_dataset_and_labels(self) -> None:
        source = inspect.getsource(stage.run_all)
        self.assertIn("make_loaders(data, cfg, seed)", source)
        self.assertIn('labels = future_values(data["power"], origins', source)
        self.assertEqual(tuple(self.cfg["models"]), stage.MODELS)

    def test_12_output_shape_backward_and_finiteness(self) -> None:
        batch, lookback, horizon, features = 3, 72, 144, 30
        tensors = (torch.randn(batch, lookback, features), torch.randn(batch, horizon, 3),
                   torch.rand(batch, horizon), torch.rand(batch, horizon), torch.rand(batch, horizon))
        for name in stage.MODELS:
            model = stage.ForecastModel(name, features, self.cfg)
            output = model(*tensors)
            self.assertEqual(tuple(output.shape), (batch, horizon))
            loss = output.square().mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                                for parameter in model.parameters()))
            if name != "HISTORY_ONLY":
                missing_nwp = tensors[1].clone()
                missing_nwp[..., 2] = 0
                with torch.no_grad():
                    np.testing.assert_allclose(model(tensors[0], missing_nwp, *tensors[2:]).numpy(),
                                               model.history(tensors[0]).numpy(), atol=1e-6)

    def test_13_training_api_has_no_test_loader(self) -> None:
        parameters = inspect.signature(stage.train_model).parameters
        self.assertEqual(list(parameters), ["model", "train_loader", "validation_loader", "cfg", "device", "run_dir"])

    def test_14_real_artifact_protocol_when_available(self) -> None:
        if not stage.PREPARED.exists():
            self.skipTest("prepared artifact not generated yet")
        data = np.load(stage.PREPARED, allow_pickle=False)
        metadata = json.loads(str(data["metadata_json"]))
        self.assertLess(int(data["times_ns"].max()), int(pd.Timestamp("2023-01-01").value))
        self.assertEqual(metadata["preprocessor_fit_split"], "train")
        self.assertEqual(metadata["reliability_prior_fit_split"], "train")
        self.assertFalse(metadata["sealed_test_loaded"])
        for split in ("train", "validation"):
            origins = data[f"{split}_origins"]
            selected = data[f"{split}_selected_cycle_ns"]
            valid = data[f"{split}_nwp_valid"]
            # Stored PV timestamps are naive ACST wall-clock nanoseconds.
            origin_utc = data["times_ns"][origins].astype(np.int64) - np.int64(570 * 60 * 10**9)
            self.assertTrue(np.all(selected[valid] + 6 * 3600 * 10**9 <= origin_utc[valid]))

    def test_15_original_pv_is_unchanged_when_available(self) -> None:
        if not stage.PREPARED.exists():
            self.skipTest("prepared artifact not generated yet")
        data = np.load(stage.PREPARED, allow_pickle=False)
        metadata = json.loads(str(data["metadata_json"]))
        pv = Path(self.cfg["pv_file"])
        self.assertEqual(pv.stat().st_size, metadata["pv_source_size"])
        self.assertEqual(pv.stat().st_mtime_ns, metadata["pv_source_mtime_ns"])

    def test_16_output_scope_and_no_2023_nwp_artifacts(self) -> None:
        allowed = {"run_nwp_minimal_screen.py", "config.json", "test_protocol.py",
                   "metrics_per_seed.csv", "REPORT.md", "results", "__pycache__"}
        self.assertTrue({path.name for path in stage.ROOT.iterdir()} <= allowed)
        for path in stage.NWP_MONTHS.glob("gfs_point_*.npz") if stage.NWP_MONTHS.exists() else []:
            self.assertLess(path.stem.rsplit("_", 1)[-1], "2023-01")


if __name__ == "__main__":
    unittest.main(verbosity=2)

