"""Tests for the train/predict interface extensions in ForecastMethod."""

import pickle
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import numpy as np
import pandas as pd

from mtdata.forecast.interface import (
    ArtifactCompatibilityError,
    ForecastMethod,
    ForecastResult,
    TrainedModelHandle,
    TrainingProgress,
    TrainResult,
)

# ── Concrete test doubles ─────────────────────────────────────────────

class _TrainableDummy(ForecastMethod):
    """Trainable method that returns a simple numpy model."""

    @property
    def name(self) -> str:
        return "trainable_dummy"

    @property
    def supports_training(self) -> bool:
        return True

    @property
    def training_category(self):
        return "fast"

    def forecast(self, series, horizon, seasonality, params, **kw):
        return ForecastResult(forecast=np.ones(horizon))

    def train(self, series, horizon, seasonality, params, *, progress_callback=None, exog=None, **kw):
        artifact = {"weights": np.zeros(3), "horizon": horizon}
        artifact_bytes = self.serialize_artifact(artifact)
        if progress_callback:
            progress_callback(TrainingProgress(step=1, total_steps=1, message="done"))
        return TrainResult(
            artifact_bytes=artifact_bytes,
            params_used=params,
            metadata={"steps": 1},
        )

    def predict_with_model(self, model, series, horizon, seasonality, params, **kw):
        return ForecastResult(
            forecast=np.full(horizon, 42.0),
            metadata={"from_model": True},
        )


class _NonTrainable(ForecastMethod):
    """Classical method — no training support."""

    @property
    def name(self) -> str:
        return "non_trainable"

    def forecast(self, series, horizon, seasonality, params, **kw):
        return ForecastResult(forecast=np.ones(horizon))


class _CustomSerializationDummy(ForecastMethod):
    """Method with custom (non-pickle) serialization."""

    @property
    def name(self) -> str:
        return "custom_serial"

    @property
    def supports_training(self) -> bool:
        return True

    @property
    def training_category(self):
        return "moderate"

    def forecast(self, series, horizon, seasonality, params, **kw):
        return ForecastResult(forecast=np.ones(horizon))

    def serialize_artifact(self, artifact):
        import json
        return json.dumps(artifact).encode()

    def deserialize_artifact(self, data):
        import json
        return json.loads(data.decode())

    def train(self, series, horizon, seasonality, params, **kw):
        artifact = {"method": "custom_serial", "v": 1}
        return TrainResult(
            artifact_bytes=self.serialize_artifact(artifact),
            params_used=params,
        )

    def predict_with_model(self, model, series, horizon, seasonality, params, **kw):
        assert model == {"method": "custom_serial", "v": 1}
        return ForecastResult(forecast=np.ones(horizon) * model["v"])


# ── Test cases ─────────────────────────────────────────────────────────

class TestTrainingProgress(unittest.TestCase):

    def test_fraction_basic(self):
        p = TrainingProgress(step=50, total_steps=100)
        self.assertAlmostEqual(p.fraction, 0.5)

    def test_fraction_zero_total(self):
        p = TrainingProgress(step=10, total_steps=0)
        self.assertEqual(p.fraction, 0.0)

    def test_fraction_clamps_over_1(self):
        p = TrainingProgress(step=200, total_steps=100)
        self.assertEqual(p.fraction, 1.0)

    def test_optional_fields_default_none(self):
        p = TrainingProgress(step=0, total_steps=10)
        self.assertIsNone(p.loss)
        self.assertIsNone(p.metrics)
        self.assertIsNone(p.eta_seconds)
        self.assertIsNone(p.message)


class TestTrainResult(unittest.TestCase):

    def test_defaults(self):
        r = TrainResult(artifact_bytes=b"data")
        self.assertEqual(r.artifact_bytes, b"data")
        self.assertEqual(r.params_used, {})
        self.assertEqual(r.metadata, {})

    def test_with_fields(self):
        r = TrainResult(
            artifact_bytes=b"data",
            params_used={"lr": 0.01},
            metadata={"loss": 0.5},
        )
        self.assertEqual(r.params_used["lr"], 0.01)
        self.assertEqual(r.metadata["loss"], 0.5)


class TestTrainPredictLifecycle(unittest.TestCase):

    def test_trainable_method_supports_training(self):
        m = _TrainableDummy()
        self.assertTrue(m.supports_training)
        self.assertEqual(m.training_category, "fast")

    def test_non_trainable_method_defaults(self):
        m = _NonTrainable()
        self.assertFalse(m.supports_training)
        self.assertEqual(m.training_category, "instant")

    def test_non_trainable_train_raises(self):
        m = _NonTrainable()
        s = pd.Series(range(100), dtype=float)
        with self.assertRaises(NotImplementedError):
            m.train(s, 10, 1, {})

    def test_non_trainable_predict_with_model_falls_back(self):
        m = _NonTrainable()
        s = pd.Series(range(100), dtype=float)
        result = m.predict_with_model("ignored", s, 5, 1, {})
        self.assertEqual(len(result.forecast), 5)

    def test_train_returns_train_result(self):
        m = _TrainableDummy()
        s = pd.Series(range(50), dtype=float)
        result = m.train(s, 10, 1, {"lr": 0.01})
        self.assertIsInstance(result, TrainResult)
        self.assertIsInstance(result.artifact_bytes, bytes)
        self.assertGreater(len(result.artifact_bytes), 0)

    def test_train_with_progress_callback(self):
        m = _TrainableDummy()
        s = pd.Series(range(50), dtype=float)
        updates: list = []
        result = m.train(s, 10, 1, {}, progress_callback=updates.append)
        self.assertEqual(len(updates), 1)
        self.assertIsInstance(updates[0], TrainingProgress)
        self.assertEqual(updates[0].message, "done")

    def test_roundtrip_serialize_deserialize(self):
        m = _TrainableDummy()
        s = pd.Series(range(50), dtype=float)
        result = m.train(s, 10, 1, {})
        artifact = m.deserialize_artifact(result.artifact_bytes)
        self.assertIn("weights", artifact)
        np.testing.assert_array_equal(artifact["weights"], np.zeros(3))

    def test_default_artifact_is_versioned_not_raw_pickle(self):
        method = _TrainableDummy()
        artifact = {"weights": np.zeros(2)}

        serialized = method.serialize_artifact(artifact)

        self.assertTrue(serialized.startswith(b"MTDATA-ARTIFACT\x00"))
        self.assertNotEqual(serialized, pickle.dumps(artifact))

    def test_legacy_unversioned_pickle_is_rejected_before_unpickling(self):
        method = _TrainableDummy()
        legacy = pickle.dumps({"weights": np.zeros(2)})

        with (
            patch.object(pickle, "loads") as pickle_loads,
            self.assertRaisesRegex(
                ArtifactCompatibilityError,
                "unversioned legacy artifact",
            ),
        ):
            method.deserialize_artifact(legacy)

        pickle_loads.assert_not_called()

    def test_runtime_version_mismatch_is_rejected_before_unpickling(self):
        method = _TrainableDummy()
        with patch(
            "mtdata.forecast.interface._distribution_version",
            return_value="saved-version",
        ):
            serialized = method.serialize_artifact({"weights": np.zeros(2)})

        with (
            patch(
                "mtdata.forecast.interface._distribution_version",
                return_value="current-version",
            ),
            patch.object(pickle, "loads") as pickle_loads,
            self.assertRaisesRegex(
                ArtifactCompatibilityError,
                "runtime is incompatible",
            ),
        ):
            method.deserialize_artifact(serialized)

        pickle_loads.assert_not_called()

    def test_predict_with_model_uses_model(self):
        m = _TrainableDummy()
        s = pd.Series(range(50), dtype=float)
        result = m.train(s, 5, 1, {})
        artifact = m.deserialize_artifact(result.artifact_bytes)
        pred = m.predict_with_model(artifact, s, 5, 1, {})
        np.testing.assert_array_equal(pred.forecast, np.full(5, 42.0))
        self.assertTrue(pred.metadata["from_model"])


class TestCustomSerialization(unittest.TestCase):
    """Methods can override serialize/deserialize for non-pickle formats."""

    def test_json_roundtrip(self):
        m = _CustomSerializationDummy()
        s = pd.Series(range(20), dtype=float)
        result = m.train(s, 5, 1, {})
        # Should be valid JSON, not pickle
        import json
        obj = json.loads(result.artifact_bytes.decode())
        self.assertEqual(obj["method"], "custom_serial")

    def test_predict_with_custom_model(self):
        m = _CustomSerializationDummy()
        s = pd.Series(range(20), dtype=float)
        result = m.train(s, 5, 1, {})
        artifact = m.deserialize_artifact(result.artifact_bytes)
        pred = m.predict_with_model(artifact, s, 5, 1, {})
        self.assertEqual(len(pred.forecast), 5)


class TestTrainingFingerprint(unittest.TestCase):

    def test_basic_fingerprint(self):
        m = _TrainableDummy()
        fp = m.training_fingerprint(
            horizon=24, seasonality=12,
            params={"lr": 0.01, "epochs": 100},
            timeframe="H1",
        )
        self.assertEqual(fp["method"], "trainable_dummy")
        self.assertEqual(fp["horizon"], 24)
        self.assertEqual(fp["seasonality"], 12)
        self.assertEqual(fp["timeframe"], "H1")
        self.assertFalse(fp["has_exog"])
        self.assertIn("lr", fp["params"])

    def test_engine_level_params_are_excluded(self):
        m = _TrainableDummy()
        fp = m.training_fingerprint(
            horizon=10, seasonality=1,
            params={
                "lr": 0.01,
                "ci_alpha": 0.05,
                "quantity": "price",
                "as_of": "2024-01-01",
                "seasonality": 1,
            },
        )
        self.assertNotIn("ci_alpha", fp["params"])
        self.assertNotIn("as_of", fp["params"])
        self.assertNotIn("seasonality", fp["params"])
        self.assertEqual(fp["params"]["quantity"], "price")
        self.assertIn("lr", fp["params"])

        without_duplicate = m.training_fingerprint(
            horizon=10,
            seasonality=1,
            params={"lr": 0.01, "quantity": "price"},
        )
        self.assertEqual(fp, without_duplicate)

    def test_fingerprint_deterministic(self):
        m = _TrainableDummy()
        fp1 = m.training_fingerprint(10, 1, {"b": 2, "a": 1})
        fp2 = m.training_fingerprint(10, 1, {"a": 1, "b": 2})
        self.assertEqual(fp1, fp2)

    def test_hash_fingerprint_stable(self):
        m = _TrainableDummy()
        fp = m.training_fingerprint(10, 1, {"a": 1})
        h1 = ForecastMethod.hash_fingerprint(fp)
        h2 = ForecastMethod.hash_fingerprint(fp)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 16)

    def test_different_horizons_different_hash(self):
        m = _TrainableDummy()
        fp1 = m.training_fingerprint(10, 1, {"a": 1})
        fp2 = m.training_fingerprint(20, 1, {"a": 1})
        h1 = ForecastMethod.hash_fingerprint(fp1)
        h2 = ForecastMethod.hash_fingerprint(fp2)
        self.assertNotEqual(h1, h2)

    def test_exog_changes_hash(self):
        m = _TrainableDummy()
        fp_no = m.training_fingerprint(10, 1, {}, has_exog=False)
        fp_yes = m.training_fingerprint(10, 1, {}, has_exog=True)
        h_no = ForecastMethod.hash_fingerprint(fp_no)
        h_yes = ForecastMethod.hash_fingerprint(fp_yes)
        self.assertNotEqual(h_no, h_yes)


class TestTrainedModelHandle(unittest.TestCase):

    def test_fields(self):
        h = TrainedModelHandle(
            model_id="nhits/EURUSD_H1/abc",
            method="nhits",
            data_scope="EURUSD_H1",
            params_hash="abc",
            created_at=1000.0,
            metadata={"loss": 0.01},
        )
        self.assertEqual(h.model_id, "nhits/EURUSD_H1/abc")
        self.assertEqual(h.metadata["loss"], 0.01)

    def test_default_metadata_empty(self):
        h = TrainedModelHandle(
            model_id="m/d/p", method="m", data_scope="d",
            params_hash="p", created_at=0,
        )
        self.assertEqual(h.metadata, {})


if __name__ == "__main__":
    unittest.main()
