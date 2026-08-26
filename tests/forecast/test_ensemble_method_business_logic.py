from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mtdata.forecast.common import _normalize_weights
from mtdata.forecast.ensemble_dispatch import dispatch_registered_forecast
from mtdata.forecast.interface import ForecastMethod
from mtdata.forecast.methods import ensemble as em


def test_ensemble_rmse_weights_remain_non_degenerate():
    series = pd.Series(np.linspace(1.0, 20.0, 20))
    method = em.EnsembleMethod()

    def dispatch(method_name, series_in, horizon, seasonality, params):
        return np.array([1.0, 2.0], dtype=float) if method_name == "naive" else np.array([1.1, 2.1], dtype=float)

    def prepare_cv(*args, **kwargs):
        X_cv = np.array(
            [
                [1.0, 1.000001],
                [2.0, 2.000002],
                [3.0, 3.000003],
                [4.0, 4.000004],
            ],
            dtype=float,
        )
        y_cv = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        return X_cv, y_cv

    out = method.forecast(
        series,
        horizon=2,
        seasonality=1,
        params={"methods": ["naive", "theta"], "mode": "rmse_weighted"},
        ensemble_dispatch_method=dispatch,
        prepare_ensemble_cv=prepare_cv,
        get_available_methods=lambda: ("naive", "theta"),
    )

    weights = out.metadata["weights"]
    assert weights[0] > 0.5
    assert weights[1] > 0.05


def test_ensemble_rmse_weighting_requires_cv_for_every_member():
    series = pd.Series(np.linspace(1.0, 20.0, 20))
    method = em.EnsembleMethod()

    def dispatch(method_name, series_in, horizon, seasonality, params):
        if method_name == "naive":
            return np.array([10.0, 20.0], dtype=float)
        return np.array([1.0, 2.0], dtype=float)

    def prepare_cv(*args, **kwargs):
        X_cv = np.array(
            [
                [1.0, np.nan],
                [2.0, np.nan],
                [3.0, np.nan],
            ],
            dtype=float,
        )
        y_cv = np.array([1.0, 2.0, 3.0], dtype=float)
        return X_cv, y_cv

    with pytest.raises(ValueError, match="theta"):
        method.forecast(
            series,
            horizon=2,
            seasonality=1,
            params={"methods": ["naive", "theta"], "mode": "rmse_weighted"},
            ensemble_dispatch_method=dispatch,
            prepare_ensemble_cv=prepare_cv,
            get_available_methods=lambda: ("naive", "theta"),
        )


def test_ensemble_rejects_component_failures():
    series = pd.Series(np.linspace(1.0, 20.0, 20))
    method = em.EnsembleMethod()

    class Dispatch:
        def __call__(self, method_name, series_in, horizon, seasonality, params):
            if method_name == "bad":
                self._last_error = {"method": "bad", "error": "boom", "error_type": "RuntimeError"}
                return None
            self._last_error = None
            return np.array([5.0, 6.0], dtype=float)

    dispatch = Dispatch()

    with pytest.raises(ValueError, match="bad: boom"):
        method.forecast(
            series,
            horizon=2,
            seasonality=1,
            params={"methods": ["naive", "bad"], "mode": "average"},
            ensemble_dispatch_method=dispatch,
            get_available_methods=lambda: ("naive", "bad"),
        )


def test_ensemble_stacking_reports_normalized_weights_and_raw_coefficients():
    series = pd.Series(np.linspace(1.0, 20.0, 20))
    method = em.EnsembleMethod()

    def dispatch(method_name, series_in, horizon, seasonality, params):
        if method_name == "naive":
            return np.array([10.0, 20.0], dtype=float)
        return np.array([1.0, 2.0], dtype=float)

    def prepare_cv(*args, **kwargs):
        return np.ones((3, 2), dtype=float), np.ones(3, dtype=float)

    with patch.object(
        em.np.linalg,
        "lstsq",
        return_value=(np.array([0.5, 2.0, 1.0], dtype=float), np.array([]), 2, np.array([])),
    ):
        out = method.forecast(
            series,
            horizon=2,
            seasonality=1,
            params={"methods": ["naive", "theta"], "mode": "stacking"},
            ensemble_dispatch_method=dispatch,
            prepare_ensemble_cv=prepare_cv,
            get_available_methods=lambda: ("naive", "theta"),
        )

    assert np.allclose(out.forecast, [21.5, 42.5])
    assert out.metadata["weights"] == pytest.approx([2.0 / 3.0, 1.0 / 3.0])
    assert out.metadata["coefficients"] == [2.0, 1.0]
    assert out.metadata["weight_semantics"] == "normalized_coefficients"
    assert out.metadata["intercept"] == 0.5


def test_ensemble_stacking_falls_back_to_raw_coefficients_for_non_positive_sum():
    series = pd.Series(np.linspace(1.0, 20.0, 20))
    method = em.EnsembleMethod()

    def dispatch(method_name, series_in, horizon, seasonality, params):
        if method_name == "naive":
            return np.array([10.0, 20.0], dtype=float)
        return np.array([1.0, 2.0], dtype=float)

    def prepare_cv(*args, **kwargs):
        return np.ones((3, 2), dtype=float), np.ones(3, dtype=float)

    with patch.object(
        em.np.linalg,
        "lstsq",
        return_value=(np.array([0.5, 1.0, -1.0], dtype=float), np.array([]), 2, np.array([])),
    ):
        out = method.forecast(
            series,
            horizon=2,
            seasonality=1,
            params={"methods": ["naive", "theta"], "mode": "stacking"},
            ensemble_dispatch_method=dispatch,
            prepare_ensemble_cv=prepare_cv,
            get_available_methods=lambda: ("naive", "theta"),
        )

    assert np.allclose(out.forecast, [9.5, 18.5])
    assert out.metadata["weights"] == [1.0, -1.0]
    assert out.metadata["coefficients"] == [1.0, -1.0]
    assert out.metadata["weight_semantics"] == "raw_coefficients"
    assert out.metadata["intercept"] == 0.5


def test_ensemble_rejects_component_dispatch_exceptions():
    series = pd.Series(np.linspace(1.0, 20.0, 20))
    method = em.EnsembleMethod()

    def dispatch(method_name, series_in, horizon, seasonality, params):
        if method_name == "bad":
            raise RuntimeError("boom")
        return np.array([5.0, 6.0], dtype=float)

    with pytest.raises(ValueError, match="bad: boom"):
        method.forecast(
            series,
            horizon=2,
            seasonality=1,
            params={"methods": ["naive", "bad"], "mode": "average"},
            ensemble_dispatch_method=dispatch,
            get_available_methods=lambda: ("naive", "bad"),
        )


def test_ensemble_uses_local_helpers_without_engine_injection():
    assert "prepare_forecast_call" not in em.EnsembleMethod.__dict__
    assert em.EnsembleMethod.prepare_forecast_call is ForecastMethod.prepare_forecast_call
    assert em._normalize_weights_default is _normalize_weights
    assert em.dispatch_registered_forecast is dispatch_registered_forecast


def test_ensemble_forecast_uses_local_dispatch_when_kwargs_omitted(monkeypatch):
    calls: list[str] = []

    def fake_dispatch(method_name, series, horizon, seasonality, params):
        calls.append(method_name)
        return np.array([1.0, 2.0], dtype=float), None

    monkeypatch.setattr(em, "dispatch_registered_forecast", fake_dispatch)
    monkeypatch.setattr(
        em,
        "get_forecast_method_availability_snapshot",
        lambda: {"naive": True, "theta": True},
    )
    monkeypatch.setattr(
        em.ForecastRegistry,
        "get_all_method_names",
        lambda: ("naive", "theta"),
    )

    out = em.EnsembleMethod().forecast(
        pd.Series(np.linspace(1.0, 20.0, 20)),
        horizon=2,
        seasonality=1,
        params={"methods": ["naive", "theta"], "mode": "average"},
    )

    assert calls == ["naive", "theta"]
    assert out.forecast.tolist() == [1.0, 2.0]


def test_ensemble_forecast_kwargs_seam_overrides_local_dispatch():
    calls: list[str] = []

    def dispatch(method_name, series_in, horizon, seasonality, params):
        calls.append(method_name)
        return np.array([3.0, 4.0], dtype=float)

    out = em.EnsembleMethod().forecast(
        pd.Series(np.linspace(1.0, 20.0, 20)),
        horizon=2,
        seasonality=1,
        params={"methods": ["naive", "theta"], "mode": "average"},
        ensemble_dispatch_method=dispatch,
        get_available_methods=lambda: ("naive", "theta"),
    )

    assert calls == ["naive", "theta"]
    assert out.forecast.tolist() == [3.0, 4.0]


@pytest.mark.parametrize("weights", ([0.25], [0.2, 0.3, 0.5]))
def test_ensemble_rejects_weight_count_mismatch(weights):
    method = em.EnsembleMethod()

    with pytest.raises(ValueError, match="exactly 2"):
        method.forecast(
            pd.Series(np.linspace(1.0, 20.0, 20)),
            horizon=2,
            seasonality=1,
            params={
                "methods": ["naive", "theta"],
                "mode": "average",
                "weights": weights,
            },
            ensemble_dispatch_method=lambda *_args: np.array([1.0, 2.0]),
            get_available_methods=lambda: ("naive", "theta"),
        )


def test_ensemble_rejects_unavailable_requested_components():
    method = em.EnsembleMethod()

    with pytest.raises(ValueError, match="sf_autoarima"):
        method.forecast(
            pd.Series(np.linspace(1.0, 20.0, 20)),
            horizon=2,
            seasonality=1,
            params={"methods": ["naive", "sf_autoarima"]},
            ensemble_dispatch_method=lambda *_args: np.array([1.0, 2.0]),
            get_available_methods=lambda: ("naive",),
        )


def test_ensemble_rejects_insufficient_cv_instead_of_downgrading():
    method = em.EnsembleMethod()
    with pytest.raises(ValueError, match="requires at least 3"):
        method.forecast(
            pd.Series(np.linspace(1.0, 20.0, 20)),
            horizon=2,
            seasonality=1,
            params={"methods": ["naive", "theta"], "mode": "rmse_weighted"},
            ensemble_dispatch_method=lambda *_args: np.array([1.0, 2.0]),
            prepare_ensemble_cv=lambda *_args: (np.empty((0, 2)), np.empty((0,))),
            get_available_methods=lambda: ("naive", "theta"),
        )


def test_ensemble_rejects_removed_bma_mode():
    method = em.EnsembleMethod()
    with pytest.raises(ValueError, match="rmse_weighted"):
        method.forecast(
            pd.Series(np.linspace(1.0, 20.0, 20)),
            horizon=2,
            seasonality=1,
            params={"methods": ["naive", "theta"], "mode": "bma"},
            get_available_methods=lambda: ("naive", "theta"),
        )


def test_ensemble_params_applied_metadata():
    """Verify component-specific params are tracked in metadata."""
    series = pd.Series(np.linspace(1.0, 20.0, 20))
    method = em.EnsembleMethod()

    captured_params = {}

    def dispatch(method_name, series_in, horizon, seasonality, params):
        captured_params[method_name] = params
        return np.array([5.0, 6.0], dtype=float)

    def dispatch_with_error(method_name, series_in, horizon, seasonality, params):
        captured_params[method_name] = params
        return np.array([5.0, 6.0], dtype=float), None

    out = method.forecast(
        series,
        horizon=2,
        seasonality=1,
        params={
            "methods": ["naive", "theta"],
            "mode": "average",
            "method_params": {"naive": {"alpha": 0.5}, "theta": {}},
        },
        ensemble_dispatch_method=dispatch,
        ensemble_dispatch_with_error=dispatch_with_error,
        get_available_methods=lambda: ("naive", "theta"),
    )

    # naive had non-empty params → should appear in params_applied
    assert "params_applied" in out.metadata
    assert out.metadata["params_applied"]["naive"] == {"alpha": 0.5}
    # theta had empty dict → should NOT appear
    assert "theta" not in out.metadata["params_applied"]
    # verify dispatch actually received the params
    assert captured_params["naive"] == {"alpha": 0.5}
    assert captured_params["theta"] == {}


def test_component_dispatch_fn_type_alias_exists():
    """Verify the canonical dispatch type alias is importable."""
    assert hasattr(em, "ComponentDispatchFn")
    # Should be a typing alias (callable type)
    assert em.ComponentDispatchFn is not None
