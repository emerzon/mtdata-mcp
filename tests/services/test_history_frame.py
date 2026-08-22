from contextlib import contextmanager
from types import SimpleNamespace

import pandas as pd
import pytest

from mtdata.services.data_service import candles


def _install_ready_provider(monkeypatch, rates):
    captured = {}
    info = SimpleNamespace(visible=True)

    monkeypatch.setattr(candles, "resolve_broker_symbol_name", lambda symbol: symbol)
    monkeypatch.setattr(candles, "get_symbol_info_cached", lambda _symbol: info)

    @contextmanager
    def ready(symbol, *, info_before):
        captured["ready"] = (symbol, info_before)
        yield None, info

    def fetch(*args, **kwargs):
        captured["fetch_args"] = args
        captured["fetch_kwargs"] = kwargs
        return rates, None

    monkeypatch.setattr(candles, "_symbol_ready_guard", ready)
    monkeypatch.setattr(candles, "_fetch_rates_with_warmup", fetch)
    return captured


def test_fetch_history_frame_uses_canonical_fetch_and_closed_bar_policy(monkeypatch):
    rates = [
        {"time": float(index * 3600), "close": 1.0 + index}
        for index in range(4)
    ]
    captured = _install_ready_provider(monkeypatch, rates)
    monkeypatch.setattr(
        candles,
        "_drop_incomplete_tail_df",
        lambda frame, _timeframe: (frame.iloc[:-1], True),
    )

    frame = candles.fetch_history_frame("EURUSD", "H1", 3, retry=False)

    assert frame["time"].tolist() == [0.0, 3600.0, 7200.0]
    assert frame["volume"].isna().all() if "volume" in frame else True
    assert captured["ready"][0] == "EURUSD"
    assert captured["fetch_args"][3:5] == (3, 0)
    assert captured["fetch_kwargs"] == {
        "include_incomplete": False,
        "retry": False,
        "sanity_check": False,
    }


def test_fetch_history_frame_preserves_incomplete_tail_when_requested(monkeypatch):
    rates = [
        {"time": float(index * 3600), "close": 1.0 + index}
        for index in range(3)
    ]
    captured = _install_ready_provider(monkeypatch, rates)

    frame = candles.fetch_history_frame(
        "EURUSD", "H1", 3, include_incomplete=True
    )

    assert frame["time"].tolist() == [0.0, 3600.0, 7200.0]
    assert captured["fetch_kwargs"]["include_incomplete"] is True


def test_fetch_history_frame_preserves_full_explicit_range(monkeypatch):
    start = "2026-08-01T00:00:00Z"
    end = "2026-08-01T05:00:00Z"
    base = pd.Timestamp(start).timestamp()
    rates = [
        {"time": base + index * 3600, "close": 1.0 + index}
        for index in range(5)
    ]
    captured = _install_ready_provider(monkeypatch, rates)

    frame = candles.fetch_history_frame(
        "EURUSD",
        "H1",
        2,
        start=start,
        end=end,
    )

    assert len(frame) == 5
    assert captured["fetch_args"][3] >= 7
    assert captured["fetch_args"][5:7] == (start, end)


def test_fetch_history_frame_rejects_coarser_provider_cadence(monkeypatch):
    rates = [
        {"time": float(index * 86400), "close": 1.0 + index}
        for index in range(8)
    ]
    _install_ready_provider(monkeypatch, rates)

    with pytest.raises(RuntimeError, match="observed_median_bar_seconds=86400.0"):
        candles.fetch_history_frame(
            "EURUSD", "H1", 8, include_incomplete=True
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"as_of": "bad"}, "Invalid as_of time"),
        (
            {"as_of": "2026-08-01", "start": "2026-07-01"},
            "as_of cannot be combined",
        ),
    ],
)
def test_fetch_history_frame_validates_time_bounds(kwargs, message):
    with pytest.raises(RuntimeError, match=message):
        candles.fetch_history_frame("EURUSD", "H1", 5, **kwargs)


def test_fetch_history_frame_reports_readiness_and_empty_data(monkeypatch):
    monkeypatch.setattr(candles, "resolve_broker_symbol_name", lambda symbol: symbol)
    monkeypatch.setattr(candles, "get_symbol_info_cached", lambda _symbol: None)

    @contextmanager
    def unavailable(*_args, **_kwargs):
        yield "symbol unavailable", None

    monkeypatch.setattr(candles, "_symbol_ready_guard", unavailable)
    with pytest.raises(RuntimeError, match="symbol unavailable"):
        candles.fetch_history_frame("EURUSD", "H1", 5)

    @contextmanager
    def ready(*_args, **_kwargs):
        yield None, SimpleNamespace(visible=True)

    monkeypatch.setattr(candles, "_symbol_ready_guard", ready)
    monkeypatch.setattr(
        candles, "_fetch_rates_with_warmup", lambda *_args, **_kwargs: ([], None)
    )
    with pytest.raises(ValueError, match="No data is available"):
        candles.fetch_history_frame("EURUSD", "H1", 5)
