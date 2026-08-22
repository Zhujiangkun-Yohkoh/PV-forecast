"""Ordinary tests for the specific historical failure modes addressed here."""
import inspect
import unittest

import numpy as np
import torch

from .asoc_clean_decision import (
    CleanDataProtocol, DiscreteTrajectoryDecoder, TimeConditionedGFNODE,
    evaluate_prefixes, load_config, make_loaders, train_one,
)


class CleanProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.protocol = CleanDataProtocol(cls.config, "Sanyo")
        cls.protocol.load_regularized_raw()
        cls.protocol.fit_transform()

    def test_01_split_timestamps_do_not_intersect(self):
        stamps = {name: set(frame.index) for name, frame in self.protocol.transformed.items()}
        self.assertFalse(stamps["train"] & stamps["validation"])
        self.assertFalse(stamps["train"] & stamps["test"])
        self.assertFalse(stamps["validation"] & stamps["test"])

    def test_02_windows_are_inside_their_split(self):
        for split, windows in self.protocol.windows.items():
            start, end = map(np.datetime64, self.config["splits"][split])
            self.assertTrue(np.all(windows.input_start >= start))
            self.assertTrue(np.all(windows.target_end <= end))

    def test_03_windows_preserve_five_minute_steps(self):
        expected_transition = np.timedelta64(self.config["lookback"] * 5, "m")
        expected_horizon = np.timedelta64((self.config["horizon"] - 1) * 5, "m")
        for windows in self.protocol.windows.values():
            self.assertTrue(np.all(windows.target_start - windows.input_start == expected_transition))
            self.assertTrue(np.all(windows.target_end - windows.target_start == expected_horizon))

    def test_04_all_preprocessors_fit_only_train(self):
        self.assertEqual({entry["split"] for entry in self.protocol.fit_log}, {"train"})
        self.assertEqual({entry["preprocessor"] for entry in self.protocol.fit_log},
                         {"KNNImputer", "IsolationForest", "feature_MinMaxScaler", "target_MinMaxScaler"})

    def test_05_training_api_cannot_receive_test_loader(self):
        self.assertNotIn("test", inspect.signature(train_one).parameters)
        train, validation, test = make_loaders(self.protocol.windows, self.config["training"]["batch_size"])
        self.assertIsNot(train, test)
        self.assertIsNot(validation, test)

    def test_06_ode_vector_field_depends_on_time(self):
        model = TimeConditionedGFNODE(self.protocol.windows["train"].x.shape[-1], self.config["model"])
        z = torch.ones(2, self.config["model"]["embedding_dim"])
        context = torch.full_like(z, 0.5)
        self.assertFalse(torch.allclose(model.ode_func(torch.tensor(0.0), z, context),
                                        model.ode_func(torch.tensor(6.0), z, context)))

    def test_07_time_grid_ends_at_twelve_hours(self):
        model = TimeConditionedGFNODE(self.protocol.windows["train"].x.shape[-1], self.config["model"])
        self.assertEqual(len(model.time_grid), 145)
        self.assertTrue(torch.isclose(model.time_grid[-1], torch.tensor(12.0)))

    def test_08_prefix_metrics_come_from_one_h144_array(self):
        y = np.ones((2, 144), dtype=np.float32)
        prediction = y + 0.1
        rows = evaluate_prefixes(y, prediction, np.ones_like(y, dtype=bool), 1.0)
        self.assertEqual({row["horizon"] for row in rows}, {12, 48, 96, 144})
        self.assertEqual({row["prefix_source"] for row in rows}, {"same_H144_prediction"})

    def test_parameter_match_is_within_five_percent(self):
        input_dim = self.protocol.windows["train"].x.shape[-1]
        ode = TimeConditionedGFNODE(input_dim, self.config["model"])
        discrete = DiscreteTrajectoryDecoder(input_dim, self.config["model"])
        ode_n = sum(p.numel() for p in ode.parameters())
        discrete_n = sum(p.numel() for p in discrete.parameters())
        self.assertLessEqual(abs(ode_n - discrete_n) / max(ode_n, discrete_n), 0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
