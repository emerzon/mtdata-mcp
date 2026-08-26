"""Tests for train/predict split on concrete forecast method implementations.

Verifies that real registered methods correctly implement the
``supports_training`` / ``train()`` / ``predict_with_model()`` lifecycle
without requiring live MT5 or GPU hardware.
"""

import unittest
import warnings

import numpy as np
import pandas as pd

from mtdata.forecast.forecast_registry import ForecastRegistry
from mtdata.forecast.interface import TrainResult


def _make_series(n: int = 200) -> pd.Series:
    """Deterministic synthetic series for reproducible tests."""
    rng = np.random.RandomState(42)
    trend = np.linspace(100, 120, n)
    noise = rng.normal(0, 0.5, n)
    return pd.Series(trend + noise, dtype=float)


# ------------------------------------------------------------------
# Registry metadata tests
# ------------------------------------------------------------------


class TestRegistryTrainableDiscovery(unittest.TestCase):
    """get_method_info() should surface training metadata for registered methods."""

    @classmethod
    def setUpClass(cls):
        # Force method registration by importing
        from mtdata.forecast.methods import (  # noqa: F401
            ets_arima,
            mlforecast,
            neural,
            sktime,
            statsforecast,
        )

    def test_list_trainable_non_empty(self):
        trainable = [
            name
            for name in ForecastRegistry.get_all_method_names()
            if ForecastRegistry.get_method_info(name)["supports_training"]
        ]
        self.assertGreater(len(trainable), 0)

    def test_neural_methods_are_trainable(self):
        for name in ("nhits", "nbeatsx", "tft", "patchtst"):
            info = ForecastRegistry.get_method_info(name)
            self.assertTrue(info["supports_training"], f"{name} should be trainable")
            self.assertEqual(info["training_category"], "heavy")

    def test_ml_methods_are_trainable(self):
        for name in ("mlf_rf", "mlf_lightgbm"):
            info = ForecastRegistry.get_method_info(name)
            self.assertTrue(info["supports_training"], f"{name} should be trainable")
            self.assertEqual(info["training_category"], "fast")

    def test_statsforecast_methods_are_trainable(self):
        info = ForecastRegistry.get_method_info("statsforecast")
        self.assertTrue(info["supports_training"])
        self.assertEqual(info["training_category"], "moderate")

    def test_sktime_methods_are_trainable(self):
        info = ForecastRegistry.get_method_info("skt_theta")
        self.assertTrue(info["supports_training"])
        self.assertEqual(info["training_category"], "fast")

    def test_ets_arima_not_trainable_yet(self):
        info = ForecastRegistry.get_method_info("ses")
        self.assertFalse(info["supports_training"])
        self.assertEqual(info["training_category"], "fast")


# ------------------------------------------------------------------
# MLForecast train/predict
# ------------------------------------------------------------------


class TestMLForecastTrainPredict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from mtdata.forecast.methods import mlforecast  # noqa: F401

    def test_mlf_rf_train_and_predict(self):
        method = ForecastRegistry.get("mlf_rf")
        series = _make_series(100)
        result = method.train(series, horizon=5, seasonality=1, params={})
        self.assertIsInstance(result, TrainResult)
        self.assertGreater(len(result.artifact_bytes), 0)

        artifact = method.deserialize_artifact(result.artifact_bytes)
        pred = method.predict_with_model(artifact, series, 5, 1, {})
        self.assertEqual(len(pred.forecast), 5)
        self.assertTrue(np.all(np.isfinite(pred.forecast)))

    def test_mlf_lightgbm_train_and_predict(self):
        method = ForecastRegistry.get("mlf_lightgbm")
        series = _make_series(100)
        result = method.train(series, horizon=5, seasonality=1, params={})
        self.assertIsInstance(result, TrainResult)

        artifact = method.deserialize_artifact(result.artifact_bytes)
        pred = method.predict_with_model(artifact, series, 5, 1, {})
        self.assertEqual(len(pred.forecast), 5)
        self.assertTrue(np.all(np.isfinite(pred.forecast)))


# ------------------------------------------------------------------
# StatsForecast train/predict
# ------------------------------------------------------------------


class TestStatsForecastTrainPredict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from mtdata.forecast.methods import statsforecast  # noqa: F401

    def test_sf_theta_train_and_predict(self):
        method = ForecastRegistry.get("sf_theta")
        series = _make_series(100)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = method.train(series, horizon=5, seasonality=12, params={})
        self.assertIsInstance(result, TrainResult)

        artifact = method.deserialize_artifact(result.artifact_bytes)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = method.predict_with_model(artifact, series, 5, 12, {})
        self.assertEqual(len(pred.forecast), 5)

    def test_statsforecast_generic_autoarima(self):
        method = ForecastRegistry.get("statsforecast")
        series = _make_series(100)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = method.train(
                series, horizon=5, seasonality=1,
                params={"model_name": "AutoARIMA"},
            )
        self.assertIsInstance(result, TrainResult)

        artifact = method.deserialize_artifact(result.artifact_bytes)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = method.predict_with_model(artifact, series, 5, 1, {})
        self.assertEqual(len(pred.forecast), 5)


# ------------------------------------------------------------------
# Sktime train/predict
# ------------------------------------------------------------------


class TestSktimeTrainPredict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from mtdata.forecast.methods import sktime  # noqa: F401
        try:
            import sktime as _skt  # noqa: F401
            cls._has_sktime = True
        except ImportError:
            cls._has_sktime = False

    def test_skt_theta_train_and_predict(self):
        if not self._has_sktime:
            self.skipTest("sktime not installed")
        method = ForecastRegistry.get("skt_theta")
        series = _make_series(100)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = method.train(series, horizon=5, seasonality=12, params={})
        self.assertIsInstance(result, TrainResult)

        artifact = method.deserialize_artifact(result.artifact_bytes)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = method.predict_with_model(artifact, series, 5, 12, {})
        self.assertEqual(len(pred.forecast), 5)


# ------------------------------------------------------------------
# Neural — only test metadata (no GPU / heavy deps in unit tests)
# ------------------------------------------------------------------


class TestNeuralMetadata(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from mtdata.forecast.methods import neural  # noqa: F401

    def test_neural_supports_training_and_fingerprint(self):
        method = ForecastRegistry.get("nhits")
        self.assertTrue(method.supports_training)
        self.assertEqual(method.training_category, "heavy")

        fp = method.training_fingerprint(
            horizon=24, seasonality=12,
            params={"max_steps": 100, "batch_size": 64, "ci_alpha": 0.05},
            timeframe="H1",
        )
        self.assertEqual(fp["method"], "nhits")
        self.assertEqual(fp["horizon"], 24)
        self.assertEqual(fp["batch_size"], 64)
        self.assertNotIn("ci_alpha", fp["params"])
        self.assertIn("input_size", fp)


if __name__ == "__main__":
    unittest.main()
