import importlib.util
import sys
import warnings
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


INDICATORS_PATH = SRC / "mtdata" / "utils" / "indicators.py"
# Load from source under a private module name so this file cannot pollute the
# real `mtdata.utils.indicators` entry used by other tests.
_MODULE_NAME = "mtdata.utils.indicators_column_names_under_test"
spec = importlib.util.spec_from_file_location(_MODULE_NAME, INDICATORS_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load indicators module from {INDICATORS_PATH}")
indicators = importlib.util.module_from_spec(spec)
sys.modules[_MODULE_NAME] = indicators
spec.loader.exec_module(indicators)
_apply_ta_indicators = indicators._apply_ta_indicators


def _sample_df(rows: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=rows, freq="D")
    return pd.DataFrame({"close": np.linspace(1.0, 2.0, len(idx))}, index=idx)


@pytest.mark.parametrize(
    ("ti_spec", "expected_cols"),
    [
        (
            "ema(20),rsi(14),macd(12,26,9)",
            ["EMA_20", "RSI_14", "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9"],
        ),
        (
            "ema(20.0),rsi(14.0),macd(12.0,26.0,9.0)",
            ["EMA_20", "RSI_14", "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9"],
        ),
        ("ema21", ["EMA_21"]),
    ],
)
def test_ti_column_names_use_ints_for_integer_like_params(
    ti_spec: str, expected_cols: list[str]
) -> None:
    df = _sample_df()
    _apply_ta_indicators(df, ti_spec)

    created = [c for c in df.columns if c != "close"]
    for col in expected_cols:
        assert col in created

    assert all(".0" not in c for c in created)


def test_apply_ta_indicators_raises_for_missing_required_columns() -> None:
    df = _sample_df()

    with pytest.raises(ValueError, match=r"Indicator 'atr' requires columns: high, low, close"):
        _apply_ta_indicators(df, "atr(14)")


def test_apply_ta_indicators_rejects_unknown_indicators_before_partial_apply() -> None:
    df = _sample_df()

    with pytest.raises(ValueError, match=r"Unknown indicator\(s\): fake_indicator_xyz"):
        _apply_ta_indicators(df, "ema(20),fake_indicator_xyz(14)")

    assert list(df.columns) == ["close"]


def test_apply_ta_indicators_rejects_reversed_macd_periods() -> None:
    df = _sample_df()

    with pytest.raises(ValueError, match=r"fast < slow"):
        _apply_ta_indicators(df, "macd(26,12,9)")


@pytest.mark.parametrize("ti_spec", ["rsi(0)", "sma(-1)"])
def test_apply_ta_indicators_rejects_nonpositive_periods(ti_spec: str) -> None:
    df = _sample_df()

    with pytest.raises(ValueError, match="must be greater than 0"):
        _apply_ta_indicators(df, ti_spec)

    assert list(df.columns) == ["close"]


def test_apply_ta_indicators_rejects_period_beyond_available_history() -> None:
    df = _sample_df(rows=20)

    with pytest.raises(ValueError, match="only 20 input rows are available"):
        _apply_ta_indicators(df, "rsi(100000)")

    assert list(df.columns) == ["close"]


@pytest.mark.parametrize(
    ("ti_spec", "message"),
    [
        ("ema(twenty)", "must be a finite number"),
        ("ema(length=abc)", "must be a finite number"),
        ("ema(lenght=5)", "does not accept parameter.*lenght"),
        ("ema(14.5)", "must be a whole number of bars"),
    ],
)
def test_apply_ta_indicators_rejects_malformed_explicit_parameters(
    ti_spec: str,
    message: str,
) -> None:
    df = _sample_df()

    with pytest.raises(ValueError, match=message):
        _apply_ta_indicators(df, ti_spec)

    assert list(df.columns) == ["close"]


def test_apply_ta_indicators_validates_all_specs_before_mutation() -> None:
    df = _sample_df()

    with pytest.raises(ValueError, match="does not accept parameter.*lenght"):
        _apply_ta_indicators(df, "ema(20),rsi(lenght=14)")

    assert list(df.columns) == ["close"]


def test_apply_ta_indicators_restores_original_index_on_value_error() -> None:
    df = pd.DataFrame(
        {
            "time": np.arange(1_700_000_000, 1_700_000_010),
            "close": np.linspace(1.0, 2.0, 10),
        }
    )
    original_index = df.index.copy()

    with pytest.raises(ValueError, match=r"Indicator 'atr' requires columns: high, low, close"):
        _apply_ta_indicators(df, "atr(14)")

    assert df.index.equals(original_index)


@pytest.mark.parametrize("volume_col", ["tick_volume", "real_volume"])
def test_apply_ta_indicators_accepts_volume_alias_columns(volume_col: str) -> None:
    df = _sample_df()
    df[volume_col] = np.arange(1, len(df) + 1, dtype=float)

    added = _apply_ta_indicators(df, "obv")

    assert any(str(col).upper().startswith("OBV") for col in added)


def test_apply_ta_indicators_prefers_nonzero_tick_volume_over_zero_volume(monkeypatch) -> None:
    df = _sample_df()
    df["volume"] = 0.0
    df["real_volume"] = 0.0
    df["tick_volume"] = np.arange(1, len(df) + 1, dtype=float)
    observed = {}

    def _fake_obv(close, volume):
        observed["volume"] = volume.copy()
        return pd.Series(np.asarray(volume, dtype=float), index=close.index, name="OBV")

    monkeypatch.setattr(indicators.pta, "obv", _fake_obv, raising=False)

    _apply_ta_indicators(df, "obv")

    assert observed["volume"].reset_index(drop=True).equals(df["tick_volume"].reset_index(drop=True))


def test_apply_ta_indicators_rejects_all_zero_volume() -> None:
    df = _sample_df()
    df["high"] = df["close"] + 0.1
    df["low"] = df["close"] - 0.1
    df["volume"] = 0.0
    df["tick_volume"] = 0.0
    df["real_volume"] = 0.0

    with pytest.raises(ValueError, match="volume"):
        _apply_ta_indicators(df, "obv")


def test_apply_ta_indicators_prefers_real_volume_over_tick_alias(monkeypatch) -> None:
    df = _sample_df()
    df["volume"] = np.arange(1, len(df) + 1, dtype=float)
    df["real_volume"] = np.arange(1001, 1001 + len(df), dtype=float)
    df["tick_volume"] = np.arange(1, len(df) + 1, dtype=float)
    observed = {}

    def _fake_obv(close, volume):
        observed["volume"] = volume.copy()
        return pd.Series(np.asarray(volume, dtype=float), index=close.index, name="OBV")

    monkeypatch.setattr(indicators.pta, "obv", _fake_obv, raising=False)

    _apply_ta_indicators(df, "obv")

    assert observed["volume"].reset_index(drop=True).equals(
        df["real_volume"].reset_index(drop=True)
    )
    assert df.attrs["indicator_volume_source"] == "real_volume"


def test_apply_ta_indicators_maps_open_column_to_open_Parameter(monkeypatch) -> None:
    idx = pd.date_range("2024-01-01", periods=20, freq="h")
    close = np.linspace(100.0, 101.0, len(idx))
    df = pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
        },
        index=idx,
    )
    observed = {}

    def _fake_bop(open_, high, low, close):
        observed["open"] = open_.copy()
        return pd.Series(close - open_, index=close.index, name="BOP")

    monkeypatch.setattr(indicators.pta, "bop", _fake_bop, raising=False)

    added = _apply_ta_indicators(df, "bop")

    assert "BOP" in added
    assert observed["open"].equals(df["open"])


def test_vwap_uses_utc_epoch_and_resets_on_broker_day(monkeypatch) -> None:
    epochs = np.array([1_700_000_000 + 3600 * i for i in range(20)], dtype=float)
    close = np.linspace(100.0, 101.0, len(epochs))
    df = pd.DataFrame(
        {
            "__epoch": epochs,
            "time": ["2099-01-01 00:00"] * len(epochs),
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": np.arange(1, len(epochs) + 1, dtype=float),
        }
    )
    monkeypatch.setattr(
        "mtdata.bootstrap.settings.mt5_config.get_server_tz",
        lambda: ZoneInfo("Europe/Nicosia"),
    )

    _apply_ta_indicators(df, "vwap")

    assert "VWAP_D" in df
    assert df.attrs["vwap_reset_calendar"] == "broker_server_day"


def test_vwap_resets_at_broker_midnight_not_utc_midnight(monkeypatch) -> None:
    idx = pd.to_datetime(
        ["2026-08-05T20:00:00Z", "2026-08-05T21:00:00Z", "2026-08-05T22:00:00Z"]
    )
    df = pd.DataFrame(
        {
            "high": [1.0, 3.0, 5.0],
            "low": [1.0, 3.0, 5.0],
            "close": [1.0, 3.0, 5.0],
            "volume": [1.0, 1.0, 1.0],
        },
        index=idx,
    )
    monkeypatch.setattr(
        "mtdata.bootstrap.settings.mt5_config.get_server_tz",
        lambda: ZoneInfo("Europe/Nicosia"),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _apply_ta_indicators(df, "vwap")

    assert not caught
    assert df["VWAP_D"].tolist() == pytest.approx([1.0, 3.0, 4.0])


def test_apply_ta_indicators_raises_actionable_error_without_retries(monkeypatch) -> None:
    df = _sample_df()

    def _broken_indicator(close, length=None, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(indicators.pta, "ema", _broken_indicator, raising=False)

    with pytest.raises(
        ValueError,
        match="Indicator 'ema' failed with parameters close, length: boom",
    ):
        _apply_ta_indicators(df, "ema(20)")


def test_apply_ta_indicators_supports_supertrend_multi_series_signature() -> None:
    idx = pd.date_range("2024-01-01", periods=80, freq="h")
    close = np.linspace(100.0, 108.0, len(idx))
    df = pd.DataFrame(
        {
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
        },
        index=idx,
    )

    added = _apply_ta_indicators(df, "supertrend(7,3)")

    assert added
    assert any(str(col).upper().startswith("SUPERT") for col in added)
