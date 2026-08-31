from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mtdata.forecast import forecast_engine as fe
from mtdata.forecast import forecast_preprocessing as fp
from mtdata.forecast.requests import ForecastGenerateRequest
from mtdata.utils.time import _format_time_minimal, bar_close_epoch


def _history_frame(rows: int = 100) -> pd.DataFrame:
    close = np.linspace(100.0, 105.0, rows)
    return pd.DataFrame(
        {
            "time": np.arange(rows, dtype=float) * 3600.0 + 1_700_000_000.0,
            "open": close - 0.2,
            "high": close + 0.4,
            "low": close - 0.5,
            "close": close,
            "volume": np.linspace(1_000.0, 2_000.0, rows),
        }
    )


def test_feature_forecast_rejects_incompatible_method_before_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fe, "_get_available_methods", lambda: ("naive",))

    def unexpected_fetch(*args, **kwargs):
        raise AssertionError("history must not be fetched for unsupported features")

    monkeypatch.setattr(fe, "_fetch_history", unexpected_fetch)

    result = fe.forecast_engine(
        symbol="EURUSD",
        timeframe="H1",
        method="naive",
        features={
            "include": ["volume"],
            "observed_future_policy": "carry_forward",
        },
    )

    assert result["error_code"] == "feature_consumption_unsupported"
    assert result["incompatible_methods"] == [
        {
            "method": "naive",
            "supports_historical_exog": False,
            "supports_future_exog": False,
        }
    ]


@pytest.mark.parametrize("attested", [False, True])
def test_feature_forecast_requires_matching_runtime_attestation(
    monkeypatch: pytest.MonkeyPatch,
    attested: bool,
) -> None:
    class FeatureAdapter:
        supports_historical_exog = True
        supports_future_exog = True

    monkeypatch.setattr(fe, "_get_available_methods", lambda: ("feature_model",))
    monkeypatch.setattr(
        fe.ForecastRegistry,
        "get",
        staticmethod(lambda _name: FeatureAdapter()),
    )

    def fake_run(**kwargs):
        metadata = {}
        if attested:
            feature_count = int(kwargs["X"].shape[1])
            metadata = {
                "diagnostics": {
                    "feature_consumption": {
                        "status": "consumed",
                        "historical_consumed": True,
                        "future_consumed": True,
                        "historical_rows": int(len(kwargs["target_series"])),
                        "future_rows": int(len(kwargs["future_exog"])),
                        "n_features": feature_count,
                        "adapter_columns": [
                            f"x{index}" for index in range(feature_count)
                        ],
                    }
                }
            }
        return np.array([105.1, 105.2]), None, metadata

    monkeypatch.setattr(fe, "_run_registered_forecast_method", fake_run)
    result = fe.forecast_engine(
        symbol="EURUSD",
        timeframe="H1",
        method="feature_model",
        horizon=2,
        ci_alpha=None,
        features={
            "include": ["volume"],
            "observed_future_policy": "carry_forward",
        },
        prefetched_df=_history_frame(),
    )

    if not attested:
        assert result["error_code"] == "feature_consumption_unverified"
        assert "attestation is missing" in result["error"]
    else:
        assert result["success"] is True
        assert result["diagnostics"]["feature_consumption"]["status"] == "consumed"


def test_selectkbest_applies_fitted_columns_to_prediction_rows() -> None:
    frame = _history_frame(20)
    future_times = [float(frame["time"].iloc[-1]) + 3600.0 * step for step in (1, 2)]

    training, future, info = fp.prepare_features(
        frame,
        {
            "include": ["open", "high", "low", "volume"],
            "observed_future_policy": "carry_forward",
        },
        future_times,
        2,
        dimred_method="selectkbest",
        dimred_params={"k": 2},
    )

    assert training is not None and training.shape == (20, 2)
    assert future is not None and future.shape == (2, 2)
    assert info["dimred_method"] == "selectkbest"
    assert info["dimred_n_features"] == 2
    assert info["selected_columns"] == ["selectkbest_0", "selectkbest_1"]
    assert future[0].tolist() == future[1].tolist()


def test_tsne_is_rejected_as_forecast_preprocessing() -> None:
    with pytest.raises(ValidationError, match="cannot transform out-of-sample"):
        ForecastGenerateRequest(
            symbol="EURUSD",
            features={"include": ["volume"]},
            dimred={"method": "tsne"},
        )

    frame = _history_frame(20)
    with pytest.raises(ValueError, match="cannot transform out-of-sample"):
        fp.prepare_features(
            frame,
            {
                "include": ["volume"],
                "observed_future_policy": "carry_forward",
            },
            [float(frame["time"].iloc[-1]) + 3600.0],
            1,
            dimred_method="tsne",
        )


def test_explicit_lookback_shortfall_degrades_forecast_reliability() -> None:
    result = fe.forecast_engine(
        symbol="EURUSD",
        timeframe="H1",
        method="naive",
        horizon=3,
        lookback=100,
        start="2026-08-20",
        end="2026-08-21",
        ci_alpha=None,
        prefetched_df=_history_frame(45),
    )

    assert result["success"] is True
    assert result["history_sample_ok"] is False
    assert result["forecast_reliability"] == "low"
    assert result["forecast_reliability_reason"] == "requested_lookback_shortfall"
    assert result["lookback_satisfied"] is False
    assert result["lookback_shortfall_bars"] == 55
    assert any("45 of 100 bars" in warning for warning in result["warnings"])


def test_historical_range_uses_information_cutoff_for_target_states() -> None:
    times = [
        datetime(2026, 8, 21, hour, tzinfo=timezone.utc).timestamp()
        for hour in (19, 20, 21)
    ]
    frame = pd.DataFrame(
        {
            "time": times,
            "open": [1.0, 1.1, 1.2],
            "high": [1.1, 1.2, 1.3],
            "low": [0.9, 1.0, 1.1],
            "close": [1.05, 1.15, 1.25],
        }
    )

    result = fe.forecast_engine(
        symbol="EURUSD",
        timeframe="H1",
        method="naive",
        horizon=3,
        start="2026-08-21",
        end="2026-08-21",
        ci_alpha=None,
        prefetched_df=frame,
    )

    reference_epoch = bar_close_epoch(float(times[-1]), "H1")
    assert result["forecast_bar_states"] == ["future", "future", "future"]
    assert result["bar_state_reference"] == "historical_cutoff"
    assert result["bar_state_reference_time"] == _format_time_minimal(reference_epoch)
