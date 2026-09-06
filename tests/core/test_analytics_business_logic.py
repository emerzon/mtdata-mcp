from __future__ import annotations

import math
import warnings
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mtdata.analytics.engine_common import _tick_frame, _window
from mtdata.analytics.engines import (
    analyze_execution_quality,
    analyze_microstructure,
    decompose_portfolio_risk,
    rank_relative_strength,
    validate_strategies,
)
from mtdata.analytics.execution_quality import (
    _execution_duration_display,
    _execution_percentiles,
)
from mtdata.analytics.microstructure import _classify_trade_sides
from mtdata.analytics.portfolio_risk import (
    _bootstrap_window_sums,
    _filtered_historical_returns,
    _portfolio_mark_context,
)
from mtdata.analytics.relative_strength import (
    _relative_strength_quote_status,
    _robust_z,
)
from mtdata.analytics.strategy_validate import (
    _barrier_returns,
    _builtin_signal,
    _observed_spread_bps,
    _position_reversal_returns,
)
from mtdata.core.analytics_requests import (
    MarketMicrostructureRequest,
    MarketRelativeStrengthRequest,
    PortfolioRiskDecomposeRequest,
    StrategyCandidate,
    StrategyValidateRequest,
    TradeExecutionQualityRequest,
)


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        ("sma_cross_event", [0.0, 0.0, 1.0, 0.0, -1.0]),
        ("ema_cross_event", [0.0, 0.0, 1.0, -1.0, 0.0]),
    ],
)
def test_builtin_ma_cross_event_signals_only_on_cross_events(
    strategy: str,
    expected: list[float],
) -> None:
    close = pd.Series([3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0])
    candidate = StrategyCandidate(
        id="cross",
        type="builtin_strategy",
        strategy=strategy,
        params={"fast_period": 2, "slow_period": 3},
    )

    signal = _builtin_signal(close, candidate)

    assert signal.iloc[:2].isna().all()
    assert signal.iloc[2:].tolist() == expected


@pytest.mark.parametrize("strategy", ["sma_cross", "ema_cross"])
def test_builtin_ma_cross_is_always_in_state(strategy: str) -> None:
    close = pd.Series([3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0])
    candidate = StrategyCandidate(
        id="cross",
        type="builtin_strategy",
        strategy=strategy,
        params={"fast_period": 2, "slow_period": 3},
    )

    signal = _builtin_signal(close, candidate)

    assert signal.iloc[:2].isna().all()
    finite = signal.iloc[2:]
    assert set(finite.dropna().unique()).issubset({-1.0, 0.0, 1.0})
    assert (finite != 0.0).any()
    assert finite.abs().sum() > 1.0


def test_builtin_rsi_reversion_signals_only_on_zone_entry() -> None:
    close = pd.Series([100.0, 90.0, 80.0, 90.0, 100.0, 90.0, 80.0])
    candidate = StrategyCandidate(
        id="rsi",
        type="builtin_strategy",
        strategy="rsi_reversion",
        params={"rsi_length": 2, "oversold": 40, "overbought": 60},
    )

    signal = _builtin_signal(close, candidate)

    assert signal.iloc[:2].isna().all()
    assert signal.iloc[2:].tolist() == [0.0, 0.0, -1.0, 1.0, 0.0]


def test_strategy_candidate_rejects_blank_id() -> None:
    with pytest.raises(ValueError, match="candidate id must not be blank"):
        StrategyCandidate(
            id="   ",
            type="builtin_strategy",
            strategy="sma_cross",
        )


def test_strategy_validation_rejects_normalized_duplicate_ids() -> None:
    with pytest.raises(
        ValueError,
        match=r"'Alpha' at positions \[0, 1\]",
    ):
        StrategyValidateRequest(
            symbol="EURUSD",
            candidates=[
                {
                    "id": "Alpha",
                    "type": "builtin_strategy",
                    "strategy": "sma_cross",
                },
                {
                    "id": " alpha ",
                    "type": "builtin_strategy",
                    "strategy": "ema_cross",
                },
            ],
        )


def test_strategy_validation_trims_unique_candidate_ids() -> None:
    request = StrategyValidateRequest(
        symbol="EURUSD",
        candidates=[
            {
                "id": " first ",
                "type": "builtin_strategy",
                "strategy": "sma_cross",
            },
            {
                "id": "SECOND",
                "type": "builtin_strategy",
                "strategy": "ema_cross",
            },
        ],
    )

    assert [candidate.id for candidate in request.candidates] == ["first", "SECOND"]


def test_strategy_validation_single_strategy_shortcut_builds_candidate() -> None:
    request = StrategyValidateRequest(symbol="EURUSD", strategy="ema_cross")

    assert len(request.candidates) == 1
    assert request.candidates[0] == StrategyCandidate(
        id="ema_cross",
        type="builtin_strategy",
        strategy="ema_cross",
    )


def test_strategy_validation_requires_exactly_one_candidate_input() -> None:
    with pytest.raises(ValueError, match="Example: --strategy ema_cross"):
        StrategyValidateRequest(symbol="EURUSD")
    with pytest.raises(ValueError, match="strategy and candidates cannot be combined"):
        StrategyValidateRequest(
            symbol="EURUSD",
            strategy="ema_cross",
            candidates=[
                {
                    "id": "cross",
                    "type": "builtin_strategy",
                    "strategy": "sma_cross",
                }
            ],
        )
from mtdata.utils.sessions import market_session_label


@pytest.fixture(autouse=True)
def _open_market_session(monkeypatch):
    monkeypatch.setattr(
        "mtdata.utils.freshness.closed_session_context",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "mtdata.utils.market_metadata.closed_session_context",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "mtdata.analytics.microstructure.closed_session_context",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "mtdata.analytics.relative_strength.closed_session_context",
        lambda *_args, **_kwargs: None,
    )


def _now() -> int:
    import time

    return int(time.time())


def _ticks(count: int = 200, *, start: int | None = None, real_volume: bool = False):
    start = start or (_now() - count)
    rows = []
    for idx in range(count):
        mid = 1.1 + idx * 0.000001
        rows.append(
            {
                "time": start + idx,
                "time_msc": (start + idx) * 1000,
                "bid": mid - 0.00005,
                "ask": mid + 0.00005,
                "last": mid if real_volume else 0.0,
                "volume": 1,
                "volume_real": 2.0 if real_volume else 0.0,
                "flags": 1054 if real_volume else 6,
            }
        )
    return rows


def _bars(count: int = 500, *, drift: float = 0.0002):
    end = _now() - 7200
    start = end - count * 3600
    rows = []
    price = 1.0
    for idx in range(count):
        change = drift + np.sin(idx / 8.0) * 0.0004
        opened = price
        price = max(0.1, price * (1.0 + change))
        rows.append(
            {
                "time": start + idx * 3600,
                "open": opened,
                "high": max(opened, price) * 1.002,
                "low": min(opened, price) * 0.998,
                "close": price,
                "tick_volume": 1000 + idx,
                "real_volume": 0,
                "spread": 10,
            }
        )
    return rows


class FakeGateway:
    COPY_TICKS_ALL = 0
    DEAL_TYPE_BUY = 0
    DEAL_TYPE_SELL = 1
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1

    def __init__(self):
        self.tick_rows = _ticks()
        self.bar_rows = {"EURUSD": _bars(), "GBPUSD": _bars(drift=0.0001), "USDJPY": _bars(drift=-0.00005)}
        self.deals = []
        self.orders = []
        self.positions = []

    def copy_ticks_range(self, symbol, start, end, flags):
        lo = start.timestamp()
        hi = end.timestamp()
        return [row for row in self.tick_rows if lo <= row["time"] <= hi]

    def copy_rates_from_pos(self, symbol, timeframe, start, count):
        return self.bar_rows[symbol][-count:]

    def copy_rates_range(self, symbol, timeframe, start, end):
        return [row for row in self.bar_rows[symbol] if start.timestamp() <= row["time"] <= end.timestamp()]

    def history_deals_get(self, start, end, **kwargs):
        return self.deals

    def history_orders_get(self, start, end, **kwargs):
        return self.orders

    def positions_get(self):
        return self.positions

    def symbols_get(self):
        return [SimpleNamespace(name=name, path="Forex\\Majors", visible=True) for name in self.bar_rows]

    def symbol_info_tick(self, symbol):
        return SimpleNamespace(bid=1.0999, ask=1.1001, time=_now())

    def symbol_info(self, symbol):
        return SimpleNamespace(
            point=0.00001,
            digits=5,
            volume_min=0.01,
            volume_max=200.0,
            volume_step=0.01,
        )

    def order_calc_profit(self, action, symbol, volume, opened, closed):
        sign = 1.0 if action == self.ORDER_TYPE_BUY else -1.0
        return sign * (closed - opened) * 100_000 * volume

    def order_calc_margin(self, action, symbol, volume, price):
        return volume * 1000.0


def test_microstructure_resolves_public_symbol_aliases() -> None:
    gateway = FakeGateway()
    original_info = gateway.symbol_info

    def symbol_info(name):
        if name != "EURUSD":
            return None
        return original_info(name)

    gateway.symbol_info = symbol_info

    result = analyze_microstructure(
        MarketMicrostructureRequest(symbol="EUR/USD", minutes_back=60, detail="compact"),
        gateway,
    )

    assert result["success"] is True
    assert result["symbol"] == "EURUSD"
    assert result["symbol_input"] == "EUR/USD"


def test_microstructure_distinguishes_trade_volume_from_quote_proxy() -> None:
    gateway = FakeGateway()
    gateway.tick_rows = _ticks(real_volume=True)
    result = analyze_microstructure(
        MarketMicrostructureRequest(symbol="EURUSD", minutes_back=60, detail="standard"),
        gateway,
    )
    assert result["success"] is True
    assert result["summary"]["feed_tier"] == "trade_volume"
    assert result["summary"]["tick_rate_basis"] == "broker_tick_updates_per_second"
    assert result["units"]["ticks_per_second"] == "broker_tick_updates_per_second"
    assert result["method_applicability"]["volume_impact_metrics"] is True
    assert "signed_volume_imbalance" in result["summary"]
    assert "kyle_lambda" not in result["summary"]
    assert "amihud_impact" not in result["summary"]
    assert result["estimator_scope"]["market_scope"] == "connected_broker_tick_feed"
    assert result["timezone"] == "UTC"
    assert result["summary"]["spread_points"]["median"] == pytest.approx(10.0)
    assert result["summary"]["spread_pips"]["median"] == pytest.approx(1.0)
    assert result["units"]["spread_points"] == "broker_points"
    mids = np.asarray(
        [(row["bid"] + row["ask"]) / 2.0 for row in gateway.tick_rows],
        dtype=float,
    )
    expected_realized = np.sqrt(np.sum(np.square(np.diff(np.log(mids)))))
    assert result["summary"][
        "mid_log_return_realized_volatility_observed_window"
    ] == pytest.approx(expected_realized)
    assert result["summary"]["mid_return_observations"] == len(mids) - 1
    assert "mid_realized_volatility" not in result["summary"]
    assert result["units"][
        "mid_log_return_realized_volatility_observed_window"
    ] == "decimal_log_return_realized_over_observed_window"
    definitions = result["estimator_scope"]["volatility_metrics"]
    assert definitions["cross_metric_comparable"] is False
    assert definitions["cross_window_comparable"] is False
    assert definitions[
        "mid_log_return_realized_volatility_observed_window"
    ]["annualized"] is False
    assert all("start" in item and "end" in item for item in result["liquidity_events"])
    assert all("start_epoch" not in item for item in result["liquidity_events"])


def test_microstructure_full_windows_are_chronological_but_events_are_ranked() -> None:
    gateway = FakeGateway()
    gateway.tick_rows = _ticks(120, real_volume=True)
    for index, row in enumerate(gateway.tick_rows):
        if index >= 90:
            row["ask"] = row["bid"] + 0.0005

    result = analyze_microstructure(
        MarketMicrostructureRequest(
            symbol="EURUSD",
            minutes_back=60,
            bucket_seconds=10,
            detail="full",
        ),
        gateway,
    )

    starts = [row["start_epoch"] for row in result["windows"]]
    assert starts == sorted(starts)
    assert result["windows_order"] == "chronological"
    assert result["liquidity_events_order"] == "spread_p95_desc_then_ticks_desc"
    event_spreads = [row["spread_p95"] for row in result["liquidity_events"]]
    assert event_spreads == sorted(event_spreads, reverse=True)
    first_window = result["windows"][0]
    first_mids = np.asarray(
        [
            (row["bid"] + row["ask"]) / 2.0
            for row in gateway.tick_rows[: first_window["ticks"]]
        ],
        dtype=float,
    )
    expected_std = np.nanstd(np.diff(np.log(first_mids)))
    assert first_window[
        "mid_log_return_std_per_quote_update"
    ] == pytest.approx(expected_std)
    assert first_window["mid_return_observations"] == len(first_mids) - 1
    assert "mid_volatility" not in first_window
    assert result["units"]["mid_log_return_std_per_quote_update"] == (
        "decimal_log_return_stddev_per_quote_update"
    )


def test_microstructure_rejects_unknown_symbol_before_tick_fetch() -> None:
    gateway = FakeGateway()
    gateway.symbol_info = lambda _symbol: None
    gateway.copy_ticks_range = MagicMock()

    result = analyze_microstructure(
        MarketMicrostructureRequest(symbol="NOTREAL", minutes_back=5), gateway
    )

    assert result["error_code"] == "symbol_not_found"
    assert result["related_tools"] == ["symbols_list"]
    gateway.copy_ticks_range.assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minutes_back": 0},
        {
            "start": "2026-08-12T10:00:00Z",
            "end": "2026-08-12T10:00:00Z",
        },
        {
            "start": "2026-08-12T11:00:00Z",
            "end": "2026-08-12T10:00:00Z",
        },
    ],
)
def test_microstructure_request_rejects_invalid_windows(kwargs) -> None:
    with pytest.raises(ValueError):
        MarketMicrostructureRequest(symbol="EURUSD", **kwargs)


def test_microstructure_request_rejects_overflowing_minutes_back() -> None:
    with pytest.raises(ValidationError, match="less than or equal"):
        MarketMicrostructureRequest(symbol="EURUSD", minutes_back=999_999_999_999)


def test_execution_quality_request_rejects_unordered_window() -> None:
    with pytest.raises(ValueError, match="start must be earlier"):
        TradeExecutionQualityRequest(
            start="2026-08-12T10:00:00Z",
            end="2026-08-12T10:00:00Z",
        )


def test_microstructure_compact_output_omits_research_events() -> None:
    gateway = FakeGateway()
    gateway.tick_rows = _ticks(real_volume=True)

    result = analyze_microstructure(
        MarketMicrostructureRequest(symbol="EURUSD", minutes_back=60),
        gateway,
    )

    assert result["summary"]["feed_tier"] == "trade_volume"
    assert result["summary"]["spread"]["unit"] == "fx_pips"
    assert result["summary"]["spread"]["latest"] == pytest.approx(1.0)
    assert result["summary"]["spread"]["recent_5m_median"] == pytest.approx(1.0)
    assert result["summary"]["spread"]["window_median"] == pytest.approx(1.0)
    assert result["summary"]["spread"]["window_p95"] == pytest.approx(1.0)
    assert result["summary"]["spread"]["regime"] == "near_window_median"
    assert result["summary"]["spread"]["basis"] == "historical_tick_window_distribution"
    assert result["summary"]["spread"]["source"] == "mt5.copy_ticks_range"
    assert result["window"]["source"] == "minutes_back"
    assert result["window"]["requested"] == {"minutes_back": 60}
    assert result["observed_window"]["start"].endswith("Z")
    assert result["observed_window"]["end"].endswith("Z")
    assert "liquidity_events" not in result
    assert "method_applicability" not in result
    assert set(result["data_quality"]) == {
        "quote_coverage",
        "invalid_partial_quote_ticks",
        "locked_quote_ticks",
        "executable_spread_ticks",
        "spread_ticks_excluded",
        "executable_spread_coverage",
        "latest_raw_update_quality",
        "truncated",
        "retained",
        "requested_start",
        "requested_end",
        "requested_duration_seconds",
        "observed_duration_seconds",
        "temporal_coverage_pct",
    }
    assert any("broker's tick feed" in warning for warning in result["warnings"])


def test_microstructure_spread_distribution_excludes_non_executable_snapshots() -> None:
    gateway = FakeGateway()
    gateway.tick_rows = _ticks(count=60)
    for index, row in enumerate(gateway.tick_rows):
        mid = (row["bid"] + row["ask"]) / 2.0
        if index < 30:
            row["bid"] = mid
            row["ask"] = mid
            row["flags"] = 6
        elif index < 40:
            row["bid"] = mid - 0.000005
            row["ask"] = mid + 0.000005
            row["flags"] = 2
        else:
            row["bid"] = mid - 0.00004
            row["ask"] = mid + 0.00004
            row["flags"] = 6

    result = analyze_microstructure(
        MarketMicrostructureRequest(symbol="EURUSD", minutes_back=5),
        gateway,
    )

    assert result["summary"]["spread"]["window_median"] == pytest.approx(0.8)
    assert result["data_quality"]["executable_spread_ticks"] == 30
    assert result["data_quality"]["spread_ticks_excluded"] == 30


def test_microstructure_compact_discloses_latest_tail_coverage() -> None:
    gateway = FakeGateway()
    gateway.tick_rows = _ticks(count=60, real_volume=True)

    result = analyze_microstructure(
        MarketMicrostructureRequest(
            symbol="EURUSD", minutes_back=5, max_ticks=20
        ),
        gateway,
    )

    quality = result["data_quality"]
    assert quality["truncated"] is True
    assert quality["retained"] == "latest"
    assert quality["requested_duration_seconds"] == 300.0
    assert quality["observed_duration_seconds"] == pytest.approx(19.0)
    assert quality["temporal_coverage_pct"] == pytest.approx(19.0 / 3.0)
    assert any("retained latest-tick tail" in item for item in result["warnings"])


def test_microstructure_untruncated_sparse_window_reports_elapsed_coverage() -> None:
    gateway = FakeGateway()
    gateway.tick_rows = _ticks(count=60, real_volume=True)

    result = analyze_microstructure(
        MarketMicrostructureRequest(symbol="EURUSD", minutes_back=5, max_ticks=100),
        gateway,
    )

    quality = result["data_quality"]
    assert quality["truncated"] is False
    assert quality["observed_duration_seconds"] == pytest.approx(59.0)
    assert quality["temporal_coverage_pct"] == pytest.approx(59.0 / 3.0)
    assert any("less than 90%" in item for item in result["warnings"])


def test_microstructure_compact_distinguishes_latest_from_window_spread() -> None:
    gateway = FakeGateway()
    gateway.tick_rows = _ticks(count=900)
    for row in gateway.tick_rows[-300:]:
        mid = (row["bid"] + row["ask"]) / 2.0
        row["bid"] = mid - 0.00025
        row["ask"] = mid + 0.00025

    result = analyze_microstructure(
        MarketMicrostructureRequest(symbol="EURUSD", minutes_back=60),
        gateway,
    )

    spread = result["summary"]["spread"]
    assert spread["window_median"] == pytest.approx(1.0)
    assert spread["latest"] == pytest.approx(5.0)
    assert spread["recent_5m_median"] == pytest.approx(5.0)
    assert spread["latest_as_of"].endswith("Z")
    assert spread["regime"] == "wider_than_window"
    assert spread["latest_to_window_median_ratio"] == pytest.approx(5.0)
    assert any("differs materially" in warning for warning in result["warnings"])


def test_microstructure_does_not_recount_last_trade_snapshots() -> None:
    gateway = FakeGateway()
    gateway.tick_rows = _ticks(real_volume=True)
    for row in gateway.tick_rows:
        row["last"] = 1.101
        row["volume_real"] = 5.0
        row["flags"] = 6
    gateway.tick_rows[0]["flags"] = 1032

    result = analyze_microstructure(
        MarketMicrostructureRequest(symbol="EURUSD", minutes_back=60, detail="standard"),
        gateway,
    )

    assert result["success"] is True
    assert result["summary"]["trade_count"] == 1
    assert result["data_quality"]["trade_tick_coverage"] == pytest.approx(1 / 200)


def test_microstructure_tick_rule_uses_immediately_preceding_trade() -> None:
    trades = pd.DataFrame({"last": [100.0, 101.0, 100.5]}, index=[0, 1, 2])
    prevailing_mid = pd.Series([100.0, 100.5, 100.5], index=[0, 1, 2])

    sides = _classify_trade_sides(trades, prevailing_mid)

    assert list(sides) == [0.0, 1.0, -1.0]


def test_microstructure_trade_sign_uses_preceding_quote() -> None:
    trades = pd.DataFrame({"last": [100.0, 101.0]}, index=[0, 1])
    contemporaneous_mid = pd.Series([100.0, 101.5], index=[0, 1])
    prevailing_mid = contemporaneous_mid.ffill().shift(1)

    sides = _classify_trade_sides(trades, prevailing_mid)

    assert list(sides) == [0.0, 1.0]


def test_relative_strength_robust_z_preserves_tail_order() -> None:
    values = pd.Series(range(100), dtype=float)

    scores = _robust_z(values)

    assert scores.iloc[-1] > scores.iloc[-2] > scores.iloc[-3]
    assert scores.iloc[0] < scores.iloc[1] < scores.iloc[2]


def test_microstructure_keeps_single_sided_quote_updates() -> None:
    gateway = FakeGateway()
    gateway.tick_rows = _ticks(real_volume=False)
    for index, row in enumerate(gateway.tick_rows):
        row["flags"] = 2 if index % 2 == 0 else 4

    result = analyze_microstructure(
        MarketMicrostructureRequest(symbol="EURUSD", minutes_back=60),
        gateway,
    )

    assert result["success"] is True
    assert result["data_quality"]["quote_coverage"] == pytest.approx(1.0)
    assert result["data_quality"]["invalid_partial_quote_ticks"] == 0


def test_tick_frame_keeps_distinct_same_timestamp_events() -> None:
    gateway = FakeGateway()
    epoch = _now() - 10
    gateway.tick_rows = [
        {
            "time": epoch,
            "time_msc": epoch * 1000,
            "bid": 1.1,
            "ask": 1.2,
            "last": 1.15,
            "volume": 5,
            "volume_real": 5.0,
            "flags": 1032,
        },
        {
            "time": epoch,
            "time_msc": epoch * 1000,
            "bid": 1.1,
            "ask": 1.2,
            "last": 1.15,
            "volume": 5,
            "volume_real": 5.0,
            "flags": 6,
        },
    ]

    frame, truncated = _tick_frame(
        gateway,
        "EURUSD",
        datetime.fromtimestamp(epoch - 1, tz=timezone.utc),
        datetime.fromtimestamp(epoch + 1, tz=timezone.utc),
        10,
    )

    assert truncated is False
    assert len(frame) == 2


def test_tick_frame_empty_result_retains_analysis_schema() -> None:
    gateway = FakeGateway()
    gateway.tick_rows = []
    now = datetime.now(timezone.utc)

    frame, truncated = _tick_frame(gateway, "EURUSD", now, now, 10)

    assert truncated is False
    assert frame.empty
    assert {"epoch", "bid", "ask", "mid", "spread"}.issubset(frame.columns)


def test_microstructure_reports_closed_session_for_short_tick_stream(monkeypatch) -> None:
    gateway = FakeGateway()
    gateway.tick_rows = _ticks(3)
    monkeypatch.setattr(
        "mtdata.analytics.microstructure.closed_session_context",
        lambda *args, **kwargs: {
            "market_status": "closed",
            "market_status_reason": "weekend",
        },
    )

    result = analyze_microstructure(
        MarketMicrostructureRequest(symbol="EURUSD", minutes_back=60), gateway
    )

    assert result["error_code"] == "market_closed"
    assert result["ticks_available"] == 3
    assert result["minimum_ticks_required"] == 20
    assert result["window_mode"] == "relative"
    assert result["requested_start"].endswith("Z")
    assert result["requested_end"].endswith("Z")
    assert result["market_status_reason"] == "weekend"
    assert "--minutes-back" in result["remediation"]
    assert result["related_tools"] == ["market_status"]


def test_microstructure_short_explicit_window_has_supported_remediation(
    monkeypatch,
) -> None:
    gateway = FakeGateway()
    gateway.tick_rows = []
    monkeypatch.setattr(
        "mtdata.analytics.microstructure.closed_session_context",
        lambda *args, **kwargs: None,
    )

    result = analyze_microstructure(
        MarketMicrostructureRequest(
            symbol="EURUSD",
            start="2026-08-20T00:00:00Z",
            end="2026-08-20T01:00:00Z",
        ),
        gateway,
    )

    assert result["error_code"] == "insufficient_data"
    assert result["window_mode"] == "explicit"
    assert result["requested_start"] == "2026-08-20T00:00:00Z"
    assert result["requested_end"] == "2026-08-20T01:00:00Z"
    assert result["ticks_available"] == 0
    assert result["minimum_ticks_required"] == 20
    assert "--start" in result["remediation"]
    assert "--end" in result["remediation"]
    assert "bars" not in result["remediation"]
    assert "timeframe" not in result["remediation"]


def test_microstructure_uses_completed_session_window_when_weekend_is_closed(
    monkeypatch,
) -> None:
    gateway = FakeGateway()
    completed_end = datetime(2026, 7, 31, 21, tzinfo=timezone.utc)
    completed_start = completed_end - timedelta(minutes=60)
    rows = _ticks(30, start=int(completed_start.timestamp()))
    gateway.copy_ticks_range = lambda _symbol, start, _end, _flags: (
        rows if start == completed_start else []
    )
    monkeypatch.setattr(
        "mtdata.analytics.microstructure.closed_session_context",
        lambda *args, **kwargs: {
            "market_status": "closed",
            "market_status_reason": "weekend",
            "note": "Market is closed; showing the latest completed session tick stream.",
        },
    )
    monkeypatch.setattr(
        "mtdata.analytics.microstructure.standard_weekend_window",
        lambda _now: (
            completed_end,
            datetime(2026, 8, 2, 21, tzinfo=timezone.utc),
        ),
    )

    result = analyze_microstructure(
        MarketMicrostructureRequest(symbol="EURUSD", minutes_back=60), gateway
    )

    assert result["success"] is True
    assert result["summary"]["ticks"] == 30
    assert result["market_status"] == "closed"
    assert any("completed-session" in warning for warning in result["warnings"])


def test_tick_frame_marks_locked_quotes_as_unusable() -> None:
    gateway = FakeGateway()
    gateway.tick_rows = [
        {"time": 1, "bid": 1.1, "ask": 1.1, "flags": 2},
        {"time": 2, "bid": 1.1, "ask": 1.1002, "flags": 2},
        {"time": 3, "bid": 1.2, "ask": 1.2, "flags": 6},
    ]

    frame, _ = _tick_frame(
        gateway,
        "EURUSD",
        datetime.fromtimestamp(0, tz=timezone.utc),
        datetime.fromtimestamp(3, tz=timezone.utc),
        100,
    )

    assert bool(frame.iloc[0]["spread_valid"]) is False
    assert frame.iloc[0]["spread_quality"] == "one_sided_update"
    assert math.isnan(frame.iloc[0]["mid"])
    assert math.isnan(frame.iloc[0]["spread"])
    assert bool(frame.iloc[1]["spread_valid"]) is True
    assert bool(frame.iloc[1]["spread_sample_eligible"]) is True
    assert frame.iloc[1]["spread_quality"] == "two_sided"
    assert frame.iloc[1]["spread"] == pytest.approx(0.0002)
    assert frame.iloc[2]["spread_quality"] == "locked"


def test_microstructure_reconciles_latest_locked_update_with_live_quote() -> None:
    gateway = FakeGateway()
    gateway.tick_rows = _ticks()
    latest = gateway.tick_rows[-1]
    latest["ask"] = latest["bid"]

    result = analyze_microstructure(
        MarketMicrostructureRequest(symbol="EURUSD", minutes_back=60),
        gateway,
    )

    spread = result["summary"]["spread"]
    assert spread["latest"] == pytest.approx(2.0)
    assert spread["spread_valid"] is True
    assert spread["spread_quality"] == "two_sided"
    assert spread["raw_update_quality"] == "locked"
    assert spread["source"] == "mt5.symbol_info_tick"
    assert spread["regime"] == "wider_than_window"
    assert spread["latest_to_window_median_ratio"] == pytest.approx(2.0)
    assert result["data_quality"]["locked_quote_ticks"] == 1
    assert result["data_quality"]["latest_raw_update_quality"] == "locked"
    assert any("canonical reconciled" in warning for warning in result["warnings"])


def test_microstructure_uses_recent_executable_quote_when_live_quote_is_locked() -> None:
    gateway = FakeGateway()
    gateway.tick_rows = _ticks()
    latest = gateway.tick_rows[-1]
    latest["ask"] = latest["bid"]
    gateway.symbol_info_tick = lambda _symbol: SimpleNamespace(
        bid=latest["bid"],
        ask=latest["bid"],
        time=latest["time"],
    )

    result = analyze_microstructure(
        MarketMicrostructureRequest(symbol="EURUSD", minutes_back=60),
        gateway,
    )

    spread = result["summary"]["spread"]
    assert spread["latest"] == pytest.approx(1.0)
    assert spread["spread_valid"] is True
    assert spread["spread_quality"] == "two_sided"
    assert spread["raw_update_quality"] == "locked"
    assert spread["source"] == "mt5.copy_ticks_range"
    assert spread["source_state"] == "reconciled_recent_two_sided_stream"
    assert spread["regime"] == "near_window_median"
    assert spread["latest_to_window_median_ratio"] == pytest.approx(1.0)
    assert any("canonical reconciled" in warning for warning in result["warnings"])


def test_execution_quality_matches_order_and_computes_markout() -> None:
    gateway = FakeGateway()
    gateway.account_info = lambda: SimpleNamespace(currency="USD")
    start = _now() - 100
    gateway.tick_rows = _ticks(100, start=start)
    gateway.orders = [
        {"ticket": 10, "type": 0, "price_open": 1.10005, "volume_initial": 1.0, "time_setup_msc": (start + 9) * 1000}
    ]
    gateway.deals = [
        {"ticket": 20, "order": 10, "position_id": 30, "symbol": "EURUSD", "type": 0, "volume": 1.0, "price": 1.10008, "time": start + 10, "time_msc": (start + 10) * 1000, "commission": -0.25, "fee": -0.05}
    ]
    result = analyze_execution_quality(
        TradeExecutionQualityRequest(minutes_back=60, markout_seconds=[1, 5], detail="full"),
        gateway,
    )
    assert result["summary"]["fills"] == 1
    assert result["items"][0]["commission_fee_per_lot"] == pytest.approx(0.30)
    assert result["summary"]["commission_fee_per_lot"]["mean"] == pytest.approx(
        0.30
    )
    assert result["currency"] == "USD"
    assert result["units"]["commission_fee_per_lot"] == (
        "account_currency_per_broker_lot"
    )
    assert result["items"][0]["benchmark_source"] == "arrival_quote"
    assert result["items"][0]["benchmark_price"] == pytest.approx(1.100059)
    assert result["items"][0]["fill_time_quote"] == pytest.approx(1.10006)
    assert result["items"][0]["benchmark_epoch"] == start + 9
    assert result["items"][0]["execution_shortfall_currency_estimate"] > 0
    assert result["units"]["execution_shortfall_currency_estimate"] == (
        "account_currency_positive_is_worse"
    )
    assert result["items"][0]["order_to_fill_duration_ms"] == 1000.0
    assert result["items"][0]["fill_timing_basis"] == "market_fill_latency"
    assert result["summary"]["market_fill_latency_ms"]["mean"] == 1000.0
    assert result["summary"]["market_order_fills"] == 1
    assert result["summary"]["non_market_order_fills"] == 0
    assert result["summary"]["pending_time_to_fill_ms"]["mean"] is None
    assert result["timing_definition"]["order_to_fill_duration_ms"].endswith(
        "not_execution_latency"
    )
    assert result["items"][0]["markout_bps"]["5"] is not None
    for horizon in ("1", "5"):
        markout = result["summary"]["markout_bps"][horizon]
        assert markout["observations"] == 1
        assert markout["missing"] == 0
        assert markout["coverage_pct"] == 100.0
        assert markout["sample_status"] == "insufficient"
    assert result["fill_sample_quality"]["observed"] == 1
    assert result["fill_sample_quality"]["scope"] == (
        "matched_fills_for_fill_level_metrics"
    )
    assert result["items"][0]["order_type"] == "BUY"
    assert result["items"][0]["order_type_code"] == 0
    assert result["breakdowns"]["by_order_type"][0]["order_type"] == "BUY"
    assert result["breakdowns"]["by_order_type"][0]["order_type_code"] == 0
    assert result["data_quality"]["session_definition"]["basis"] == (
        "dst_aware_market_sessions"
    )
    assert result["window"]["source"] == "minutes_back"
    assert result["window"]["timezone"] == "UTC"
    assert result["window"]["minutes_back_requested"] == 60
    assert result["window"]["minutes_back_effective"] == pytest.approx(60.0)

    compact = analyze_execution_quality(
        TradeExecutionQualityRequest(
            minutes_back=60, markout_seconds=[1, 5], detail="compact"
        ),
        gateway,
    )
    assert "requested_window" not in compact
    assert "window" in compact
    assert "pending_time_to_fill_ms" not in compact["summary"]
    assert "pending_time_to_fill_ms" in compact.get("omitted_metrics", [])
    assert "eligible_symbols" not in compact["data_quality"]
    assert "analyzed_symbols" not in compact["data_quality"]
    assert "quote_reads" not in compact["data_quality"]
    assert compact["data_quality"]["eligible_symbol_count"] == 1
    assert compact["data_quality"]["analyzed_symbol_count"] == 1
    if compact["summary"].get("price_improvement_pct") is not None:
        assert 0.0 <= compact["summary"]["price_improvement_pct"] <= 100.0


def test_execution_quality_does_not_average_unlike_lot_fees() -> None:
    gateway = FakeGateway()
    fill_epoch = _now() - 10
    gateway.orders = [
        {"ticket": 10, "price_open": 65000.0, "volume_initial": 1.0},
        {"ticket": 11, "price_open": 250.0, "volume_initial": 50.0},
    ]
    gateway.deals = [
        {
            "ticket": 20,
            "order": 10,
            "symbol": "BTCUSD",
            "type": 0,
            "volume": 1.0,
            "price": 65000.0,
            "time_msc": fill_epoch * 1000,
            "commission": -0.02,
            "fee": 0.0,
        },
        {
            "ticket": 21,
            "order": 11,
            "symbol": "TSLA.NAS-24",
            "type": 0,
            "volume": 50.0,
            "price": 250.0,
            "time_msc": (fill_epoch + 1) * 1000,
            "commission": -5.0,
            "fee": 0.0,
        },
    ]

    def _symbol_info(symbol):
        sizes = {"BTCUSD": 1.0, "TSLA.NAS-24": 1.0}
        return SimpleNamespace(
            point=0.01,
            digits=2,
            volume_min=0.01,
            volume_max=200.0,
            volume_step=0.01,
            trade_contract_size=sizes[symbol],
            path="CFD",
        )

    gateway.symbol_info = _symbol_info
    gateway.symbols_get = lambda: [
        SimpleNamespace(name="BTCUSD", path="Crypto"),
        SimpleNamespace(name="TSLA.NAS-24", path="CFD"),
    ]

    result = analyze_execution_quality(
        TradeExecutionQualityRequest(
            minutes_back=60,
            benchmark="order_price",
            detail="full",
        ),
        gateway,
    )

    assert "commission_fee_per_lot" not in result["summary"]
    assert result["summary"]["total_commission_fee"] == pytest.approx(5.02)
    assert result["summary"]["commission_fee"]["mean"] == pytest.approx(2.51)
    assert result["summary"]["commission_fee_bps"]["mean"] is not None
    by_symbol = {row["symbol"]: row for row in result["breakdowns"]["by_symbol"]}
    assert by_symbol["BTCUSD"]["commission_fee_per_lot"]["mean"] == pytest.approx(0.02)
    assert by_symbol["TSLA.NAS-24"]["commission_fee_per_lot"]["mean"] == pytest.approx(
        0.10
    )
    assert any("not comparable across symbols" in warning for warning in result["warnings"])


def test_execution_quality_empty_explicit_range_retains_analysis_window() -> None:
    result = analyze_execution_quality(
        TradeExecutionQualityRequest(
            start="2020-01-01",
            end="2020-01-02",
            detail="full",
        ),
        FakeGateway(),
    )

    assert result["success"] is True
    assert result["summary"]["fills"] == 0
    assert result["window"]["source"] == "explicit_range"
    assert result["window"]["start"] == "2020-01-01T00:00:00Z"
    assert result["window"]["end"] == "2020-01-02T23:59:59.999999Z"
    assert result["window"]["requested"] == {
        "start": "2020-01-01",
        "end": "2020-01-02",
    }
    assert "minutes_back_requested" not in result["window"]
    assert "defaulted" not in result["window"]
    assert result["sample"]["sample_start"] is None
    assert result["sample"]["sample_end"] is None


def test_execution_quality_fee_percentiles_use_positive_cost_magnitudes() -> None:
    gateway = FakeGateway()
    fill_epoch = _now() - 10
    gateway.orders = [
        {"ticket": 10, "price_open": 1.1, "volume_initial": 1.0},
        {"ticket": 11, "price_open": 1.1, "volume_initial": 0.5},
        {"ticket": 12, "price_open": 1.1, "volume_initial": 1.0},
    ]
    gateway.deals = [
        {
            "ticket": 20,
            "order": 10,
            "symbol": "EURUSD",
            "type": 0,
            "volume": 1.0,
            "price": 1.1001,
            "time_msc": fill_epoch * 1000,
            "commission": -3.5,
            "fee": 0.0,
        },
        {
            "ticket": 21,
            "order": 11,
            "symbol": "EURUSD",
            "type": 0,
            "volume": 0.5,
            "price": 1.1001,
            "time_msc": (fill_epoch + 1) * 1000,
            "commission": -0.7,
            "fee": 0.0,
        },
        {
            "ticket": 22,
            "order": 12,
            "symbol": "EURUSD",
            "type": 0,
            "volume": 1.0,
            "price": 1.1001,
            "time_msc": (fill_epoch + 2) * 1000,
            "commission": 0.5,
            "fee": 0.0,
        },
    ]

    result = analyze_execution_quality(
        TradeExecutionQualityRequest(
            minutes_back=60,
            benchmark="order_price",
            detail="full",
        ),
        gateway,
    )

    costs = [item["commission_fee_per_lot"] for item in result["items"]]
    percentiles = result["summary"]["commission_fee_per_lot"]
    assert costs == pytest.approx([3.5, 1.4, 0.0])
    assert percentiles["max"] == 3.5
    assert percentiles["max"] >= percentiles["p99"] >= percentiles["p95"]
    assert percentiles["median"] >= 0.0


def test_execution_quality_rejects_substring_as_missing_exact_symbol() -> None:
    result = analyze_execution_quality(
        TradeExecutionQualityRequest(
            symbol="USD",
            minutes_back=60,
            benchmark="order_price",
        ),
        FakeGateway(),
    )

    assert result["error_code"] == "symbol_not_found"
    assert result["symbol"] == "USD"


def test_execution_quality_exact_symbol_postfilters_gateway_results() -> None:
    gateway = FakeGateway()
    fill_epoch = _now() - 10
    gateway.orders = [
        {"ticket": 10, "price_open": 1.1, "volume_initial": 1.0},
        {"ticket": 11, "price_open": 1.2, "volume_initial": 1.0},
    ]
    gateway.deals = [
        {
            "ticket": 20,
            "order": 10,
            "symbol": "EURUSD",
            "type": 0,
            "volume": 1.0,
            "price": 1.1001,
            "time_msc": fill_epoch * 1000,
        },
        {
            "ticket": 21,
            "order": 11,
            "symbol": "GBPUSD",
            "type": 0,
            "volume": 1.0,
            "price": 1.2001,
            "time_msc": fill_epoch * 1000,
        },
    ]
    query_groups = []

    def _deals(start, end, **kwargs):
        query_groups.append(kwargs.get("group"))
        return gateway.deals

    def _orders(start, end, **kwargs):
        query_groups.append(kwargs.get("group"))
        return gateway.orders

    gateway.history_deals_get = _deals
    gateway.history_orders_get = _orders

    result = analyze_execution_quality(
        TradeExecutionQualityRequest(
            symbol="EURUSD",
            minutes_back=60,
            benchmark="order_price",
            detail="full",
        ),
        gateway,
    )

    assert query_groups == ["EURUSD", "EURUSD"]
    assert result["symbol_filter"] == {
        "requested": "EURUSD",
        "resolved": "EURUSD",
        "match_mode": "exact",
    }
    assert {item["symbol"] for item in result["items"]} == {"EURUSD"}
    assert result["data_quality"]["eligible_symbols"] == ["EURUSD"]
    assert result["data_quality"]["analyzed_symbols"] == ["EURUSD"]


def test_execution_quality_separates_eligible_and_analyzed_symbols() -> None:
    gateway = FakeGateway()
    fill_epoch = _now() - 10
    gateway.orders = [
        {"ticket": 10, "price_open": 1.1, "volume_initial": 1.0},
    ]
    gateway.deals = [
        {
            "ticket": 20,
            "order": 10,
            "symbol": "EURUSD",
            "type": 0,
            "volume": 1.0,
            "price": 1.1001,
            "time_msc": fill_epoch * 1000,
        },
        {
            "ticket": 21,
            "order": 11,
            "symbol": "GBPUSD",
            "type": 0,
            "volume": 1.0,
            "price": 1.2001,
            "time_msc": fill_epoch * 1000,
        },
    ]

    result = analyze_execution_quality(
        TradeExecutionQualityRequest(
            minutes_back=60,
            benchmark="order_price",
            detail="full",
        ),
        gateway,
    )

    assert result["data_quality"]["eligible_symbols"] == ["EURUSD", "GBPUSD"]
    assert result["data_quality"]["analyzed_symbols"] == ["EURUSD"]
    assert result["data_quality"]["skipped"]["unbenchmarked"] == 1


def test_execution_quality_reports_magic_filter_without_precision_loss() -> None:
    gateway = FakeGateway()
    fill_epoch = _now() - 10
    magic = (1 << 63) + 17
    gateway.orders = [
        {"ticket": 10, "price_open": 1.1, "volume_initial": 1.0},
    ]
    gateway.deals = [
        {
            "ticket": 20,
            "order": 10,
            "symbol": "EURUSD",
            "type": 0,
            "volume": 1.0,
            "price": 1.1001,
            "time_msc": fill_epoch * 1000,
            "magic": magic,
        },
    ]

    result = analyze_execution_quality(
        TradeExecutionQualityRequest(
            magic=str(magic),
            minutes_back=60,
            benchmark="order_price",
            detail="full",
        ),
        gateway,
    )

    assert result["filters_applied"] == {
        "magic": magic,
        "magic_exact": str(magic),
    }
    assert result["summary"]["fills"] == 1


def test_execution_quality_uses_continuous_calendar_for_crypto() -> None:
    gateway = FakeGateway()
    gateway.symbols_get = lambda: [
        SimpleNamespace(name="BTCUSD", path="Cryptocurrencies", visible=True)
    ]
    fill_epoch = _now() - 10
    gateway.orders = [
        {"ticket": 10, "price_open": 60_000.0, "volume_initial": 0.1}
    ]
    gateway.deals = [
        {
            "ticket": 20,
            "order": 10,
            "symbol": "BTCUSD",
            "type": 0,
            "volume": 0.1,
            "price": 60_010.0,
            "time_msc": fill_epoch * 1000,
        }
    ]

    result = analyze_execution_quality(
        TradeExecutionQualityRequest(
            symbol="BTCUSD",
            minutes_back=60,
            benchmark="order_price",
            detail="full",
        ),
        gateway,
    )

    assert result["items"][0]["session"] == "continuous"
    assert result["items"][0]["session_calendar"] == "continuous_24_7"
    assert result["data_quality"]["session_definition"]["calendar"] == (
        "continuous_24_7"
    )
    assert result["breakdowns"]["by_session"] == [
        {
            "session_calendar": "continuous_24_7",
            "session": "continuous",
            "fills": 1,
            "slippage_bps": result["breakdowns"]["by_session"][0][
                "slippage_bps"
            ],
        }
    ]


def test_execution_quality_uses_utc_hours_when_calendar_is_unknown() -> None:
    gateway = FakeGateway()
    gateway.symbols_get = lambda: [
        SimpleNamespace(
            name="AAPL.NAS",
            path="Stock CFD's\\Nasdaq",
            visible=True,
        )
    ]
    fill_epoch = _now() - 10
    gateway.orders = [
        {"ticket": 10, "price_open": 210.0, "volume_initial": 1.0}
    ]
    gateway.deals = [
        {
            "ticket": 20,
            "order": 10,
            "symbol": "AAPL.NAS",
            "type": 0,
            "volume": 1.0,
            "price": 210.1,
            "time_msc": fill_epoch * 1000,
        }
    ]

    result = analyze_execution_quality(
        TradeExecutionQualityRequest(
            symbol="AAPL.NAS",
            minutes_back=60,
            benchmark="order_price",
            detail="full",
        ),
        gateway,
    )

    assert result["items"][0]["session_calendar"] == "utc_hour_only"
    assert result["items"][0]["session"] is None
    assert result["breakdowns"]["by_session"] == []
    assert result["data_quality"]["session_definition"]["calendar"] == (
        "utc_hour_only"
    )
    assert any("by_hour_utc" in warning for warning in result["warnings"])


def test_execution_quality_qualifies_mixed_session_breakdowns_by_calendar() -> None:
    gateway = FakeGateway()
    gateway.symbols_get = lambda: [
        SimpleNamespace(name="EURUSD", path="Forex\\Majors", visible=True),
        SimpleNamespace(name="BTCUSD", path="Cryptocurrencies", visible=True),
    ]
    fill_epoch = _now() - 10
    gateway.orders = [
        {"ticket": 10, "price_open": 1.1, "volume_initial": 1.0},
        {"ticket": 11, "price_open": 60_000.0, "volume_initial": 0.1},
    ]
    gateway.deals = [
        {
            "ticket": 20,
            "order": 10,
            "symbol": "EURUSD",
            "type": 0,
            "volume": 1.0,
            "price": 1.1001,
            "time_msc": fill_epoch * 1000,
        },
        {
            "ticket": 21,
            "order": 11,
            "symbol": "BTCUSD",
            "type": 0,
            "volume": 0.1,
            "price": 60_010.0,
            "time_msc": fill_epoch * 1000,
        },
    ]

    result = analyze_execution_quality(
        TradeExecutionQualityRequest(
            minutes_back=60,
            benchmark="order_price",
            detail="standard",
        ),
        gateway,
    )

    assert set(result["data_quality"]["session_definitions"]) == {
        "continuous_24_7",
        "fx",
    }
    assert {
        (row["session_calendar"], row["session"])
        for row in result["breakdowns"]["by_session"]
    } == {
        ("continuous_24_7", "continuous"),
        (
            "fx",
            market_session_label(
                datetime.fromtimestamp(fill_epoch, tz=timezone.utc),
                session_calendar="fx",
            ),
        ),
    }


def test_execution_quality_compact_omits_expanded_breakdowns() -> None:
    gateway = FakeGateway()
    start = _now() - 100
    gateway.tick_rows = _ticks(100, start=start)
    gateway.orders = [
        {
            "ticket": 10,
            "type": 0,
            "price_open": 1.10005,
            "volume_initial": 1.0,
            "time_setup_msc": (start + 9) * 1000,
        }
    ]
    gateway.deals = [
        {
            "ticket": 20,
            "order": 10,
            "symbol": "EURUSD",
            "type": 0,
            "volume": 1.0,
            "price": 1.10008,
            "time_msc": (start + 10) * 1000,
        }
    ]

    result = analyze_execution_quality(
        TradeExecutionQualityRequest(minutes_back=60, markout_seconds=[1]),
        gateway,
    )

    assert result["summary"]["fills"] == 1
    assert result["summary_scope"] == "requested_window"
    assert result["sample"]["selection_order"] == "latest_first"
    assert result["sample"]["truncated"] is False
    assert "breakdowns" not in result
    assert "session_calendars" not in result["data_quality"]
    assert "session_definition" not in result["data_quality"]
    assert "timing_definition" not in result
    assert result["price_quality_definition"]["slippage_bps"] == (
        "market_arrival_quote"
    )
    assert result["price_quality_definition"]["markout_bps"] == (
        "post_fill_midpoint_markout_positive_is_favorable"
    )
    assert result["units"]["slippage_bps"] == "basis_points_positive_is_worse"
    assert result["units"]["markout_bps"] == (
        "basis_points_positive_is_favorable"
    )


def test_execution_quality_summary_matches_compact_headlines() -> None:
    gateway = FakeGateway()
    start = _now() - 100
    gateway.tick_rows = _ticks(100, start=start)
    gateway.orders = [
        {
            "ticket": 10,
            "type": 0,
            "price_open": 1.10005,
            "volume_initial": 1.0,
            "time_setup_msc": (start + 9) * 1000,
        }
    ]
    gateway.deals = [
        {
            "ticket": 20,
            "order": 10,
            "symbol": "EURUSD",
            "type": 0,
            "volume": 1.0,
            "price": 1.10008,
            "time_msc": (start + 10) * 1000,
        }
    ]

    result = analyze_execution_quality(
        TradeExecutionQualityRequest(
            minutes_back=60,
            markout_seconds=[1],
            detail="summary",
        ),
        gateway,
    )

    assert result["summary"]["fills"] == 1
    assert "breakdowns" not in result
    assert "items" not in result
    assert "requested_window" not in result
    assert "timing_definition" not in result
    assert result["price_quality_definition"]["slippage_bps"] == (
        "market_arrival_quote"
    )
    assert result["units"]["slippage_bps"] == "basis_points_positive_is_worse"


def test_execution_quality_labels_truncated_latest_fill_sample() -> None:
    gateway = FakeGateway()
    start = _now() - 400
    gateway.tick_rows = _ticks(400, start=start)
    gateway.orders = []
    gateway.deals = []
    for index in range(3):
        fill_epoch = start + 100 + (index * 80)
        gateway.orders.append(
            {
                "ticket": 10 + index,
                "type": 0,
                "price_open": 1.10005,
                "volume_initial": 1.0,
                "time_setup_msc": (fill_epoch - 1) * 1000,
            }
        )
        gateway.deals.append(
            {
                "ticket": 20 + index,
                "order": 10 + index,
                "symbol": "EURUSD",
                "type": 0,
                "volume": 1.0,
                "price": 1.10008,
                "time_msc": fill_epoch * 1000,
            }
        )

    result = analyze_execution_quality(
        TradeExecutionQualityRequest(minutes_back=60, limit=2, markout_seconds=[1]),
        gateway,
    )

    assert result["summary"]["fills"] == 2
    assert result["data_quality"]["eligible_trade_deals"] == 3
    assert result["summary_scope"] == "latest_2_of_3"
    assert result["sample"]["selection_order"] == "latest_first"
    assert result["sample"]["truncated"] is True
    assert result["effective_analysis_window"]["scope"] == "latest_2_of_3"
    assert result["window"]["minutes_back_requested"] == 60
    assert any("latest 2 matched fill" in warning for warning in result["warnings"])


def test_execution_quality_reuses_symbol_time_quote_chunks() -> None:
    gateway = FakeGateway()
    chunk = math.floor((_now() - 600) / 300) * 300
    start = chunk + 80
    gateway.tick_rows = _ticks(100, start=start)
    gateway.orders = [
        {
            "ticket": 10,
            "type": 0,
            "price_open": 1.10005,
            "volume_initial": 1.0,
            "time_setup_msc": (start + 9) * 1000,
        },
        {
            "ticket": 11,
            "type": 0,
            "price_open": 1.10005,
            "volume_initial": 1.0,
            "time_setup_msc": (start + 19) * 1000,
        },
    ]
    gateway.deals = [
        {
            "ticket": 20,
            "order": 10,
            "symbol": "EURUSD",
            "type": 0,
            "volume": 1.0,
            "price": 1.10008,
            "time_msc": (start + 10) * 1000,
        },
        {
            "ticket": 21,
            "order": 11,
            "symbol": "EURUSD",
            "type": 0,
            "volume": 1.0,
            "price": 1.10009,
            "time_msc": (start + 20) * 1000,
        },
    ]

    result = analyze_execution_quality(
        TradeExecutionQualityRequest(
            minutes_back=60,
            markout_seconds=[1, 5],
            detail="full",
        ),
        gateway,
    )

    quote_reads = result["data_quality"]["quote_reads"]
    assert result["summary"]["fills"] == 2
    assert quote_reads["broker_queries"] == 1
    assert quote_reads["cache_hits"] == 3
    assert quote_reads["strategy"] == "symbol_5_minute_chunk_cache"


def test_execution_quality_excludes_future_dated_fills() -> None:
    gateway = FakeGateway()
    future = _now() + 1_000
    gateway.orders = []
    gateway.deals = [
        {
            "ticket": 20,
            "order": 10,
            "symbol": "EURUSD",
            "type": 0,
            "volume": 1.0,
            "price": 1.10008,
            "time_msc": future * 1000,
        }
    ]

    result = analyze_execution_quality(
        TradeExecutionQualityRequest(minutes_back=60),
        gateway,
    )

    assert result["summary"]["fills"] == 0
    assert result["data_quality"]["skipped"]["future_timestamp"] == 1
    assert "ahead of the observation clock" in result["warnings"][0]


def test_execution_quality_aggregates_partial_fills_by_order() -> None:
    gateway = FakeGateway()
    start = _now() - 100
    gateway.tick_rows = _ticks(100, start=start)
    gateway.orders = [
        {
            "ticket": 10,
            "type": 0,
            "price_open": 1.10005,
            "volume_initial": 1.0,
            "time_setup_msc": (start + 9) * 1000,
        }
    ]
    gateway.deals = [
        {
            "ticket": 20,
            "order": 10,
            "symbol": "EURUSD",
            "type": 0,
            "volume": 0.4,
            "price": 1.10008,
            "time_msc": (start + 10) * 1000,
        },
        {
            "ticket": 21,
            "order": 10,
            "symbol": "EURUSD",
            "type": 0,
            "volume": 0.6,
            "price": 1.10009,
            "time_msc": (start + 11) * 1000,
        },
    ]

    result = analyze_execution_quality(
        TradeExecutionQualityRequest(
            minutes_back=60,
            benchmark="order_price",
            markout_seconds=[1],
            detail="full",
        ),
        gateway,
    )

    assert result["summary"]["fills"] == 2
    assert result["summary"]["orders"] == 1
    assert result["summary"]["partial_orders"] == 0
    assert result["summary"]["partial_fill_pct"] == 0.0
    assert result["summary"]["partial_fill_rate_basis"] == (
        "all_eligible_deals_in_requested_window"
    )
    assert [row["deal_fill_ratio"] for row in result["items"]] == [0.4, 0.6]


def test_execution_session_clock_tracks_london_new_york_dst_mismatch() -> None:
    winter = datetime(2026, 1, 12, 12, 30, tzinfo=timezone.utc)
    summer = datetime(2026, 7, 13, 12, 30, tzinfo=timezone.utc)

    assert market_session_label(winter, session_calendar="fx") == "london"
    assert market_session_label(summer, session_calendar="fx") == (
        "london_ny_overlap"
    )


def test_execution_quality_separates_pending_wait_from_market_latency() -> None:
    gateway = FakeGateway()
    start = _now() - 100
    gateway.tick_rows = _ticks(100, start=start)
    gateway.orders = [
        {
            "ticket": 10,
            "type": 0,
            "price_open": 1.10005,
            "volume_initial": 1.0,
            "time_setup_msc": (start + 9) * 1000,
        },
        {
            "ticket": 11,
            "type": 2,
            "price_open": 1.10005,
            "volume_initial": 1.0,
            "time_setup_msc": start * 1000,
        },
    ]
    gateway.deals = [
        {
            "ticket": 20,
            "order": 10,
            "symbol": "EURUSD",
            "type": 0,
            "volume": 1.0,
            "price": 1.10008,
            "time_msc": (start + 10) * 1000,
        },
        {
            "ticket": 21,
            "order": 11,
            "symbol": "EURUSD",
            "type": 0,
            "volume": 1.0,
            "price": 1.10008,
            "time_msc": (start + 10) * 1000,
        },
    ]

    result = analyze_execution_quality(
        TradeExecutionQualityRequest(
            minutes_back=60,
            benchmark="order_price",
            markout_seconds=[1],
            detail="full",
        ),
        gateway,
    )

    assert result["summary"]["market_fill_latency_ms"]["mean"] == 1000.0
    assert result["summary"]["pending_time_to_fill_ms"]["mean"] == 10000.0
    assert result["summary"]["order_to_fill_duration_ms"]["mean"] == 5500.0
    duration_display = result["summary"]["duration_display"]
    assert duration_display["pending_time_to_fill"]["mean"] == "10s"
    assert duration_display["pending_time_to_fill"]["p95"] == "10s"
    assert duration_display["order_to_fill_duration"]["mean"] == "5.5s"
    assert duration_display["order_to_fill_duration"]["p95"] == "9.55s"
    assert result["items"][1]["fill_timing_basis"] == "pending_time_to_fill"
    assert any("not broker execution latency" in item for item in result["warnings"])
    assert result["summary"]["slippage_basis"] == "explicit_order_price_all_fills"
    assert (
        result["price_quality_definition"]["market_fill_slippage_bps"]
        == "market_fill_vs_submitted_order_price"
    )
    assert (
        result["price_quality_definition"]["market_fill_vs_order_price_bps"]
        == "market_fill_vs_submitted_order_price"
    )
    assert result["summary"]["market_fill_vs_order_price_bps"]["mean"] is not None
    arrival_stats = result["summary"]["market_fill_vs_arrival_quote_bps"]
    assert all(value is None for value in arrival_stats.values())


def test_execution_quality_separates_pending_opportunity_cost_from_slippage() -> None:
    gateway = FakeGateway()
    start = _now() - 120
    gateway.tick_rows = _ticks(100, start=start)
    gateway.orders = [
        {
            "ticket": 10,
            "type": 0,
            "price_open": 1.10005,
            "volume_initial": 1.0,
            "time_setup_msc": (start + 9) * 1000,
        },
        {
            "ticket": 11,
            "type": 2,
            "price_open": 1.10100,
            "volume_initial": 1.0,
            "time_setup_msc": start * 1000,
        },
    ]
    gateway.deals = [
        {
            "ticket": 20,
            "order": 10,
            "symbol": "EURUSD",
            "type": 0,
            "volume": 1.0,
            "price": 1.10008,
            "time_msc": (start + 10) * 1000,
        },
        {
            "ticket": 21,
            "order": 11,
            "symbol": "EURUSD",
            "type": 0,
            "volume": 1.0,
            "price": 1.10110,
            "time_msc": (start + 50) * 1000,
        },
    ]

    result = analyze_execution_quality(
        TradeExecutionQualityRequest(
            minutes_back=60,
            benchmark="arrival_quote",
            markout_seconds=[1],
            detail="full",
        ),
        gateway,
    )

    market_fill, pending_fill = result["items"]
    assert market_fill["benchmark_source"] == "arrival_quote"
    assert pending_fill["benchmark_source"] == "pending_order_price"
    assert pending_fill["benchmark_price"] == pytest.approx(1.101)
    assert pending_fill["arrival_quote_price"] == pytest.approx(1.10005)
    assert pending_fill["arrival_implementation_shortfall_bps"] > 9.0
    assert pending_fill["slippage_bps"] < 1.0

    summary = result["summary"]
    assert summary["slippage_basis"] == "market_arrival_quote"
    assert summary["slippage_bps"] == summary["market_fill_slippage_bps"]
    assert summary["slippage_bps"]["mean"] == pytest.approx(
        market_fill["slippage_bps"]
    )
    assert summary["pending_fill_vs_order_bps"]["mean"] == pytest.approx(
        pending_fill["slippage_bps"]
    )
    assert summary["pending_arrival_implementation_shortfall_bps"][
        "mean"
    ] == pytest.approx(pending_fill["arrival_implementation_shortfall_bps"])
    assert any("market-order fills only" in item for item in result["warnings"])


def test_execution_quality_statistics_remove_binary_float_tails() -> None:
    stats = _execution_percentiles(
        [-0.47683719545640957, 0.2623938317292096, 1.375197583110125]
    )

    assert stats["mean"] == pytest.approx(0.386918)
    assert stats["median"] == pytest.approx(0.262394)
    assert stats["p95"] == pytest.approx(1.26392)


def test_execution_quality_formats_long_pending_durations() -> None:
    display = _execution_duration_display(
        {"mean": 7_063_990.0, "p95": 41_793_600.0, "max": 48_936_100.0}
    )

    assert display == {"mean": "1h 57m", "p95": "11h 36m", "max": "13h 35m"}


def test_execution_quality_zero_fills_are_an_empty_state() -> None:
    gateway = FakeGateway()
    gateway.deals = []
    gateway.orders = []

    result = analyze_execution_quality(
        TradeExecutionQualityRequest(
            minutes_back=60,
            min_sample=1,
            markout_seconds=[1, 5, 30],
            detail="full",
        ),
        gateway,
    )

    assert result["success"] is True
    assert result["empty"] is True
    assert result["status"] == "no_matching_fills"
    assert "No matching fills" in result["message"]
    assert result["summary"]["fills"] == 0
    assert result["summary"]["slippage_basis"] == "not_applicable"
    assert result["fill_sample_quality"]["status"] == "not_applicable"
    assert result["price_quality_definition"]["slippage_bps"] == "not_applicable"
    assert result["warnings"] == [] or not any(
        "Markout evidence" in str(item) for item in result.get("warnings") or []
    )


def test_execution_quality_rejects_overflowing_minutes_back() -> None:
    with pytest.raises(ValidationError, match="less than or equal"):
        TradeExecutionQualityRequest(minutes_back=999_999_999_999)


def test_execution_quality_duration_display_preserves_subsecond_resolution() -> None:
    display = _execution_duration_display(
        {"mean": 571.6, "median": 252.0, "p90": 1211.4, "max": 1851.0}
    )

    assert display == {
        "mean": "572ms",
        "median": "252ms",
        "p90": "1.21s",
        "max": "1.85s",
    }


def test_execution_quality_handles_empty_tick_history() -> None:
    gateway = FakeGateway()
    gateway.tick_rows = []
    fill_epoch = _now() - 10
    gateway.orders = [{"ticket": 10, "price_open": 1.1, "volume_initial": 1.0}]
    gateway.deals = [
        {
            "ticket": 20,
            "order": 10,
            "position_id": 30,
            "symbol": "EURUSD",
            "type": 0,
            "volume": 1.0,
            "price": 1.1001,
            "time": fill_epoch,
            "time_msc": fill_epoch * 1000,
        }
    ]

    result = analyze_execution_quality(
        TradeExecutionQualityRequest(minutes_back=60, markout_seconds=[1, 5]),
        gateway,
    )

    assert result["success"] is True
    assert result["summary"]["fills"] == 0
    assert result["data_quality"]["skipped"]["unbenchmarked"] == 1
    assert result["data_quality"]["benchmark"]["fallback_count"] == 0
    assert result["data_quality"]["benchmark"]["arrival_quote_coverage"] == 0.0

    fallback_result = analyze_execution_quality(
        TradeExecutionQualityRequest(
            minutes_back=60,
            markout_seconds=[1],
            benchmark_fallback="order_price",
            detail="full",
        ),
        gateway,
    )
    assert fallback_result["summary"]["fills"] == 1
    assert fallback_result["items"][0]["benchmark_source"] == "order_price_fallback"
    assert fallback_result["data_quality"]["benchmark"]["fallback_count"] == 1
    markout = fallback_result["summary"]["markout_bps"]["1"]
    assert markout["observations"] == 0
    assert markout["missing"] == 1
    assert markout["coverage_pct"] == 0.0
    assert markout["sample_status"] == "insufficient"
    assert any("used order price" in warning for warning in fallback_result["warnings"])
    assert any("1s" in warning for warning in fallback_result["warnings"])


def test_execution_quality_limit_selects_latest_eligible_fill() -> None:
    gateway = FakeGateway()
    start = _now() - 100
    gateway.tick_rows = []
    gateway.orders = [
        {"ticket": 10, "type": 0, "price_open": 1.1, "volume_initial": 1.0},
        {"ticket": 11, "type": 0, "price_open": 1.2, "volume_initial": 1.0},
    ]
    gateway.deals = [
        {
            "ticket": 20,
            "order": 10,
            "symbol": "EURUSD",
            "type": 0,
            "volume": 1.0,
            "price": 1.1001,
            "time_msc": start * 1000,
        },
        {
            "ticket": 21,
            "order": 11,
            "symbol": "EURUSD",
            "type": 0,
            "volume": 1.0,
            "price": 1.2001,
            "time_msc": (start + 50) * 1000,
        },
    ]

    result = analyze_execution_quality(
        TradeExecutionQualityRequest(
            minutes_back=60,
            limit=1,
            min_sample=1,
            benchmark="order_price",
            markout_seconds=[1],
            detail="full",
        ),
        gateway,
    )

    assert [item["deal_ticket"] for item in result["items"]] == [21]
    assert result["sample"]["selection_order"] == "latest_first"
    assert result["sample"]["total_eligible"] == 2
    assert result["sample"]["truncated"] is True


def test_execution_quality_limit_does_not_manufacture_partial_fill() -> None:
    gateway = FakeGateway()
    start = _now() - 100
    gateway.tick_rows = []
    gateway.orders = [
        {
            "ticket": 10,
            "type": 0,
            "price_open": 1.10005,
            "volume_initial": 1.0,
            "time_setup_msc": (start + 9) * 1000,
        }
    ]
    gateway.deals = [
        {
            "ticket": 20,
            "order": 10,
            "symbol": "EURUSD",
            "type": 0,
            "volume": 0.4,
            "price": 1.10008,
            "time_msc": (start + 10) * 1000,
        },
        {
            "ticket": 21,
            "order": 10,
            "symbol": "EURUSD",
            "type": 0,
            "volume": 0.6,
            "price": 1.10009,
            "time_msc": (start + 11) * 1000,
        },
    ]

    result = analyze_execution_quality(
        TradeExecutionQualityRequest(
            minutes_back=60,
            limit=1,
            min_sample=1,
            benchmark="order_price",
            markout_seconds=[1],
            detail="full",
        ),
        gateway,
    )

    assert [item["deal_ticket"] for item in result["items"]] == [21]
    assert result["summary"]["partial_orders"] == 0
    assert result["summary"]["partial_fill_pct"] == 0.0
    assert result["summary"]["orders_evaluated_for_partial_fills"] == 1
    assert result["summary"]["partial_fill_deals_evaluated"] == 2


def test_strategy_validation_returns_walk_forward_oos_metrics() -> None:
    gateway = FakeGateway()
    request = StrategyValidateRequest(
        symbol="EURUSD",
        lookback=400,
        candidates=[
            {"id": "cross", "type": "builtin_strategy", "strategy": "sma_cross", "params": {"fast_period": 5, "slow_period": 20}}
        ],
        barrier={"horizon": 5},
        n_splits=3,
        cost_model="fixed",
        spread_bps=1.0,
        commission_bps_per_side=0.25,
        bootstrap_samples=100,
        seed=7,
        detail="full",
    )
    result = validate_strategies(request, gateway)
    assert result["success"] is True
    selection = result["data_quality"]["history_selection"]
    assert selection["lookback_bars_requested"] == 400
    assert selection["fetch_bars_requested"] == 410
    assert selection["outcome_tail_bars"] == 5
    assert selection["warmup_bars"] == 5
    assert selection["evaluation_bars"] == selection["fetch_bars"] - 10
    assert result["validation"]["outcome_horizon_bars"] == 5
    assert result["validation"]["extra_purge_bars"] == 0
    assert result["validation"]["protocol"] == "anchored_expanding_fixed_candidate_oos"
    assert result["validation"]["execution_timing"] == "next_bar_open"
    assert result["cost_model"]["commission_bps_per_side"] == 0.25
    assert result["cost_model"]["slippage_bps_per_side"] == 1.0
    assert result["cost_model"]["round_trip_bps"] == 3.5
    assert result["rankings"][0]["id"] == "cross"
    assert result["rankings"][0]["strategy"] == "sma_cross"
    assert result["rankings"][0]["effective_parameters"] == {
        "fast_period": 5,
        "slow_period": 20,
    }
    assert result["rankings"][0]["trades"] > 0
    assert result["rankings"][0]["evaluation_status"] == "partial"
    candidate = result["rankings"][0]
    assert candidate["folds_evaluated"] == 2
    assert candidate["folds_requested"] == 3
    assert candidate["skipped_folds"][0]["reason"] == "insufficient_training_trades"
    assert candidate["signal_definition"] == "state_reversal"
    assert candidate["outcome_model"] == "position_reversal"
    assert "barrier_window" not in result["validation"]
    assert result["validation"]["outcome_model"] == "position_reversal"
    assert candidate["evidence"]["criteria"]["cost_model_complete"] is True
    assert candidate["evidence"]["provisional_positive_before_complete_costs"] is False
    assert "calibration" not in candidate
    assert "direction_base_rate_stability" in candidate
    if candidate["sharpe"] is not None:
        assert candidate["mean_return_t_stat"] == pytest.approx(
            candidate["sharpe"] * math.sqrt(candidate["trades"])
        )

    assert result["units"]["trades"] == "non_overlapping_positions"
    assert result["units"]["max_drawdown"] == "nonnegative_return_fraction"
    assert result["rankings"][0]["max_drawdown"] >= 0.0
    assert result["rankings"][0]["evidence"]["classification"] in {
        "positive", "negative", "inconclusive"
    }
    for fold in result["rankings"][0]["folds"]:
        assert fold["test_end_bar"] + request.barrier.horizon <= fold["test_window_end_bar"]


@pytest.mark.parametrize(
    ("request_type", "kwargs"),
    [
        (
            MarketMicrostructureRequest,
            {
                "symbol": "EURUSD",
                "start": "2026-08-12T10:00:00Z",
                "end": "2026-08-12T11:00:00Z",
                "minutes_back": 30,
            },
        ),
        (
            TradeExecutionQualityRequest,
            {
                "start": "2026-08-12T10:00:00Z",
                "end": "2026-08-12T11:00:00Z",
                "minutes_back": 30,
            },
        ),
    ],
)
def test_analytics_requests_reject_conflicting_time_controls(
    request_type,
    kwargs,
) -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        request_type(**kwargs)


@pytest.mark.parametrize("magic", [-1, 1 << 64])
def test_execution_quality_request_rejects_magic_outside_mt5_domain(magic) -> None:
    with pytest.raises(ValueError):
        TradeExecutionQualityRequest(magic=magic)


def test_execution_quality_request_accepts_mt5_magic_boundaries() -> None:
    assert TradeExecutionQualityRequest(magic=0).magic == 0
    assert TradeExecutionQualityRequest(magic=(1 << 64) - 1).magic == (1 << 64) - 1


def test_strategy_validation_explicit_range_is_not_tailed_to_lookback() -> None:
    gateway = FakeGateway()
    rows = gateway.bar_rows["EURUSD"]
    start = datetime.fromtimestamp(rows[0]["time"], timezone.utc).isoformat()
    end = datetime.fromtimestamp(rows[-1]["time"], timezone.utc).isoformat()
    request = StrategyValidateRequest(
        symbol="EURUSD",
        lookback=200,
        start=start,
        end=end,
        candidates=[
            {
                "id": "cross",
                "type": "builtin_strategy",
                "strategy": "sma_cross",
                "params": {"fast_period": 5, "slow_period": 20},
            }
        ],
        barrier={"horizon": 1},
        n_splits=2,
        cost_model="fixed",
        spread_bps=1.0,
        bootstrap_samples=100,
    )

    result = validate_strategies(request, gateway)

    selection = result["data_quality"]["history_selection"]
    assert result["success"] is True
    assert result["data_quality"]["bars"] == len(rows) - 1
    assert selection["mode"] == "explicit_range"
    assert selection["lookback_bars_requested"] == 200
    assert selection["lookback_applied"] is False
    assert selection["fetch_bars"] == len(rows) - 1
    assert selection["evaluation_bars"] == len(rows) - 1
    assert selection["outcome_tail_bars"] == 1
    assert selection["warmup_bars"] == 0
    assert selection["bars_used"] == len(rows) - 1
    assert selection["requested_start"] == start
    assert selection["requested_end"] == end


@pytest.mark.parametrize(
    ("fold_windows", "indices", "skipped_reason"),
    [
        (
            [(101, 110), (120, 140)],
            [10, 20, 30, 40, 50, 121, 122, 123, 124, 125],
            "no_test_trades",
        ),
        (
            [(30, 40), (100, 110)],
            [10, 20, 31, 32, 33, 34, 35, 101, 102, 103, 104, 105],
            "insufficient_training_trades",
        ),
    ],
)
def test_strategy_validation_marks_skipped_requested_folds_partial(
    monkeypatch,
    fold_windows,
    indices,
    skipped_reason,
) -> None:
    gateway = FakeGateway()
    monkeypatch.setattr(
        "mtdata.analytics.strategy_validate._walk_forward_windows",
        lambda *args, **kwargs: (fold_windows, []),
    )
    monkeypatch.setattr(
        "mtdata.analytics.strategy_validate._builtin_signal",
        lambda close, candidate: pd.Series(1.0, index=close.index),
    )
    monkeypatch.setattr(
        "mtdata.analytics.strategy_validate._barrier_returns",
        lambda *args, **kwargs: (
            np.asarray(indices, dtype=int),
            np.full(len(indices), 0.01, dtype=float),
        ),
    )
    request = StrategyValidateRequest(
        symbol="EURUSD",
        lookback=400,
        candidates=[
            {
                "id": "partial-cross",
                "type": "builtin_strategy",
                "strategy": "sma_cross_event",
                "params": {"fast_period": 5, "slow_period": 20},
            }
        ],
        barrier={"horizon": 1, "tp_pct": 0.15, "sl_pct": 0.15},
        n_splits=2,
        cost_model="fixed",
        spread_bps=1.0,
        bootstrap_samples=100,
        seed=7,
        detail="full",
    )

    result = validate_strategies(request, gateway)
    candidate = result["rankings"][0]

    assert result["bootstrap_samples"] == 100
    assert result["bootstrap_seed"] == 7
    assert result["seed_source"] == "request"

    assert candidate["evaluation_status"] == "partial"
    assert candidate["folds_requested"] == 2
    assert candidate["folds_evaluated"] == 1
    assert candidate["fold_coverage"] == 0.5
    assert candidate["skipped_folds"][0]["reason"] == skipped_reason
    assert candidate["evidence"]["classification"] != "positive"
    assert candidate["evidence"]["criteria"]["all_requested_folds_evaluated"] is False
    assert any("evaluated 1 of 2 requested folds" in item for item in result["warnings"])


def test_strategy_validation_historical_spread_can_receive_positive_classification(
    monkeypatch,
) -> None:
    gateway = FakeGateway()
    monkeypatch.setattr(
        "mtdata.analytics.strategy_validate._bootstrap_mean_ci",
        lambda *_args, **_kwargs: (0.001, 0.002),
    )
    monkeypatch.setattr(
        "mtdata.analytics.strategy_validate._block_bootstrap_positive_mean_p_value",
        lambda *_args, **_kwargs: 0.001,
    )
    monkeypatch.setattr(
        "mtdata.analytics.strategy_validate._builtin_signal",
        lambda close, _candidate: pd.Series(
            np.where(np.arange(len(close)) % 2 == 0, 1.0, -1.0),
            index=close.index,
        ),
    )
    request = StrategyValidateRequest(
        symbol="EURUSD",
        lookback=400,
        candidates=[
            {
                "id": "cross",
                "type": "builtin_strategy",
                "strategy": "sma_cross",
                "params": {"fast_period": 5, "slow_period": 20},
            }
        ],
        barrier={"horizon": 5},
        n_splits=3,
        bootstrap_samples=100,
        min_positive_fold_share=0.0,
        cost_model="historical_bar_spread",
    )

    result = validate_strategies(request, gateway)
    evidence = result["rankings"][0]["evidence"]

    assert result["cost_model"]["complete"] is True
    assert result["cost_model"]["source"] == (
        "mt5_historical_bar_spread_median"
    )
    assert result["cost_model"]["window"]["basis"] == "historical_bar_spread"
    assert result["cost_model"]["window"]["coverage_pct"] == 100.0
    assert evidence["criteria"]["cost_model_complete"] is True
    assert evidence["provisional_positive_before_complete_costs"] is False
    assert evidence["classification"] == "positive"


def test_strategy_validation_fixed_model_requires_explicit_spread() -> None:
    with pytest.raises(ValueError, match="--spread-bps is required"):
        StrategyValidateRequest(
            symbol="EURUSD",
            candidates=[
                {
                    "id": "cross",
                    "type": "builtin_strategy",
                    "strategy": "sma_cross",
                }
            ],
            cost_model="fixed",
        )


def test_analytics_window_expands_date_only_end_through_utc_day() -> None:
    start, end = _window("2026-08-13", "2026-08-13", 60)

    assert start == datetime(2026, 8, 13, tzinfo=timezone.utc)
    assert end == datetime(2026, 8, 13, 23, 59, 59, 999999, tzinfo=timezone.utc)


def test_analytics_window_keeps_explicit_equal_instants_invalid() -> None:
    with pytest.raises(ValueError, match="start must be earlier than end"):
        _window(
            "2026-08-13T10:00:00Z",
            "2026-08-13T10:00:00Z",
            60,
        )


def test_microstructure_rejects_future_only_history_window() -> None:
    gateway = FakeGateway()
    gateway.copy_ticks_range = MagicMock()
    result = analyze_microstructure(
        MarketMicrostructureRequest(
            symbol="EURUSD",
            start="2028-01-01T10:00:00Z",
            end="2028-01-01T11:00:00Z",
        ),
        gateway,
    )

    assert result["success"] is False
    assert result["error_code"] == "future_date_range"
    assert "at or before" in result["remediation"]
    gateway.copy_ticks_range.assert_not_called()


def test_execution_quality_rejects_future_only_history_window() -> None:
    result = analyze_execution_quality(
        TradeExecutionQualityRequest(
            start="2030-01-01",
            end="2030-01-02",
        ),
        FakeGateway(),
    )

    assert result["success"] is False
    assert result["error_code"] == "future_date_range"
    assert result["details"]["resolved_start"].startswith("2030-01-01")
    assert result["details"]["resolved_end"].startswith("2030-01-02")
    assert result["details"]["current_time"]
    assert "at or before" in result["remediation"]


def test_execution_quality_request_accepts_one_sided_explicit_window() -> None:
    start_only = TradeExecutionQualityRequest(start="2026-08-24T10:00:00Z")
    end_only = TradeExecutionQualityRequest(end="2026-08-24T11:00:00Z")

    assert start_only.start == "2026-08-24T10:00:00Z"
    assert start_only.end is None
    assert end_only.start is None
    assert end_only.end == "2026-08-24T11:00:00Z"


def test_strategy_validation_historical_model_rejects_explicit_spread() -> None:
    with pytest.raises(ValueError, match="--spread-bps is only valid"):
        StrategyValidateRequest(
            symbol="EURUSD",
            candidates=[
                {
                    "id": "cross",
                    "type": "builtin_strategy",
                    "strategy": "sma_cross",
                }
            ],
            spread_bps=1.25,
        )


def test_strategy_validation_rejects_removed_commission_name() -> None:
    with pytest.raises(ValueError, match="commission_bps_per_side"):
        StrategyValidateRequest(
            symbol="EURUSD",
            strategy="sma_cross",
            commission_bps=0.25,
        )


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_strategy_validation_rejects_invalid_per_side_costs(value) -> None:
    with pytest.raises(ValueError):
        StrategyValidateRequest(
            symbol="EURUSD",
            strategy="sma_cross",
            commission_bps_per_side=value,
        )
    with pytest.raises(ValueError):
        StrategyValidateRequest(
            symbol="EURUSD",
            strategy="sma_cross",
            slippage_bps=value,
        )


def test_strategy_validation_explicit_fixed_spread_is_complete() -> None:
    request = StrategyValidateRequest(
        symbol="EURUSD",
        candidates=[
            {
                "id": "cross",
                "type": "builtin_strategy",
                "strategy": "sma_cross",
            }
        ],
        cost_model="fixed",
        spread_bps=1.25,
    )

    spread, source, complete, window = _observed_spread_bps(
        request,
        FakeGateway(),
        pd.DataFrame(),
    )

    assert spread == 1.25
    assert source == "explicit"
    assert complete is True
    assert window == {"basis": "request"}


def test_strategy_validation_marks_sparse_historical_spreads_incomplete() -> None:
    gateway = FakeGateway()
    frame = pd.DataFrame(_bars(20))
    frame.loc[frame.index[:3], "spread"] = 0.0
    request = StrategyValidateRequest(
        symbol="EURUSD",
        candidates=[
            {
                "id": "cross",
                "type": "builtin_strategy",
                "strategy": "sma_cross",
            }
        ],
        cost_model="historical_bar_spread",
    )

    spread, source, complete, window = _observed_spread_bps(
        request,
        gateway,
        frame,
    )

    assert spread is not None
    assert source == "mt5_historical_bar_spread_median"
    assert complete is False
    assert window["observations"] == 17
    assert window["coverage_pct"] == 85.0


def test_strategy_validation_auto_uses_conservative_estimate_when_coverage_is_sparse() -> None:
    gateway = FakeGateway()
    frame = pd.DataFrame(_bars(20))
    frame.loc[frame.index[:3], "spread"] = 0.0
    request = StrategyValidateRequest(
        symbol="EURUSD",
        candidates=[
            {
                "id": "cross",
                "type": "builtin_strategy",
                "strategy": "sma_cross",
            }
        ],
    )

    spread, source, complete, window = _observed_spread_bps(
        request,
        gateway,
        frame,
    )

    assert request.cost_model == "auto"
    assert spread is not None
    assert complete is True
    assert source in {
        "mt5_historical_bar_spread_p75",
        "current_bid_ask_snapshot",
        "mt5_symbol_info_spread",
    }
    assert window["basis"] == "auto_conservative_estimate"
    assert window["selection_reason"] == "incomplete_historical_spread_coverage"


def test_strategy_validation_rejects_barrier_tp_sl_for_state_reversal() -> None:
    result = validate_strategies(
        StrategyValidateRequest(
            symbol="EURUSD",
            lookback=400,
            candidates=[
                {
                    "id": "cross",
                    "type": "builtin_strategy",
                    "strategy": "sma_cross",
                    "params": {"fast_period": 5, "slow_period": 20},
                }
            ],
            barrier={"horizon": 5, "tp_pct": 0.15, "sl_pct": 0.15},
            cost_model="fixed",
            spread_bps=1.0,
        ),
        FakeGateway(),
    )

    assert result["success"] is False
    assert result["error_code"] == "incompatible_barrier_for_state_reversal"
    assert result["outcome_model"] == "position_reversal"
    assert "cross" in result["incompatible_candidates"]


def test_strategy_validation_compact_keeps_effective_parameters() -> None:
    result = validate_strategies(
        StrategyValidateRequest(
            symbol="EURUSD",
            lookback=400,
            candidates=[
                {
                    "id": "cross",
                    "type": "builtin_strategy",
                    "strategy": "sma_cross",
                    "params": {"fast_period": 5, "slow_period": 20},
                }
            ],
            barrier={"horizon": 5},
            n_splits=3,
            cost_model="fixed",
            spread_bps=1.0,
            bootstrap_samples=100,
            detail="compact",
        ),
        FakeGateway(),
    )

    assert result["success"] is True
    assert result["rankings"][0]["effective_parameters"] == {
        "fast_period": 5,
        "slow_period": 20,
    }
    assert result["rankings"][0]["outcome_model"] == "position_reversal"
    assert "folds" not in result["rankings"][0]


def test_strategy_barrier_entry_uses_next_bar_open_after_gap() -> None:
    frame = pd.DataFrame(
        {
            "open": [100.0, 110.0, 110.0],
            "high": [101.0, 111.0, 111.0],
            "low": [99.0, 109.0, 109.0],
            "close": [100.0, 110.0, 110.0],
        }
    )
    signal = pd.Series([1.0, np.nan, np.nan])

    indices, outcomes = _barrier_returns(
        frame,
        signal,
        horizon=1,
        tp_pct=5.0,
        sl_pct=5.0,
    )

    assert indices.tolist() == [0]
    assert outcomes.tolist() == pytest.approx([0.0])


def test_strategy_barrier_stop_gap_uses_opening_fill_loss() -> None:
    frame = pd.DataFrame(
        {
            "open": [100.0, 100.0, 90.0],
            "high": [100.0, 101.0, 91.0],
            "low": [100.0, 99.0, 89.0],
            "close": [100.0, 100.0, 90.0],
        }
    )
    signal = pd.Series([1.0, np.nan, np.nan])

    indices, outcomes = _barrier_returns(
        frame,
        signal,
        horizon=2,
        tp_pct=20.0,
        sl_pct=5.0,
    )

    assert indices.tolist() == [0]
    assert outcomes.tolist() == pytest.approx([-0.10])


def test_strategy_barrier_stop_gap_precedes_same_bar_tp_policy() -> None:
    frame = pd.DataFrame(
        {
            "open": [100.0, 100.0, 90.0],
            "high": [100.0, 101.0, 125.0],
            "low": [100.0, 99.0, 89.0],
            "close": [100.0, 100.0, 110.0],
        }
    )
    signal = pd.Series([1.0, np.nan, np.nan])

    _, outcomes = _barrier_returns(
        frame,
        signal,
        horizon=2,
        tp_pct=20.0,
        sl_pct=5.0,
        same_bar_policy="tp_first",
    )

    assert outcomes.tolist() == pytest.approx([-0.10])


def test_strategy_barrier_returns_do_not_overlap_persistent_signals() -> None:
    frame = pd.DataFrame(
        {
            "open": np.full(12, 100.0),
            "high": np.full(12, 100.1),
            "low": np.full(12, 99.9),
            "close": np.full(12, 100.0),
        }
    )
    signal = pd.Series(np.ones(12))

    indices, outcomes = _barrier_returns(
        frame,
        signal,
        horizon=3,
        tp_pct=5.0,
        sl_pct=5.0,
    )

    assert indices.tolist() == [0, 3, 6]
    assert outcomes.tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_strategy_validation_fails_when_cost_spread_is_unavailable() -> None:
    gateway = FakeGateway()
    gateway.bar_rows["EURUSD"] = [
        {**row, "spread": 0.0} for row in gateway.bar_rows["EURUSD"]
    ]
    request = StrategyValidateRequest(
        symbol="EURUSD",
        lookback=400,
        candidates=[
            {
                "id": "cross",
                "type": "builtin_strategy",
                "strategy": "sma_cross",
                "params": {"fast_period": 5, "slow_period": 20},
            }
        ],
        cost_model="historical_bar_spread",
    )

    result = validate_strategies(request, gateway)

    assert result["success"] is False
    assert result["error_code"] == "cost_model_unavailable"
    assert result["cost_model"]["spread_bps"] is None


def test_strategy_validation_rejects_unknown_symbol_before_history_fetch() -> None:
    gateway = FakeGateway()
    gateway.symbol_info = lambda _symbol: None
    gateway.copy_rates_from_pos = MagicMock()
    request = StrategyValidateRequest(
        symbol="NO_SUCH_SYMBOL",
        candidates=[
            {
                "id": "cross",
                "type": "builtin_strategy",
                "strategy": "sma_cross",
            }
        ],
    )

    result = validate_strategies(request, gateway)

    assert result["error_code"] == "symbol_not_found"
    assert result["symbol"] == "NO_SUCH_SYMBOL"
    assert result["related_tools"] == ["symbols_list"]
    gateway.copy_rates_from_pos.assert_not_called()


def test_forecast_strategy_folds_cover_computed_signal_window(monkeypatch) -> None:
    gateway = FakeGateway()

    def fake_execute_forecast(**kwargs):
        history_length = len(kwargs["prefetched_df"])
        return {"expected_return": 0.01 if history_length % 2 else -0.01}

    monkeypatch.setattr(
        "mtdata.forecast.forecast.execute_forecast",
        fake_execute_forecast,
    )
    request = StrategyValidateRequest(
        symbol="EURUSD",
        lookback=400,
        candidates=[
            {
                "id": "forecast",
                "type": "forecast_threshold",
                "method": "naive",
                "params": {"lookback": 20},
                "horizon": 1,
                "long_above": 0.0,
                "short_below": 0.0,
            }
        ],
        barrier={"horizon": 1, "tp_pct": 0.15, "sl_pct": 0.15},
        n_splits=3,
        cost_model="fixed",
        spread_bps=1.0,
        bootstrap_samples=100,
        detail="full",
    )

    result = validate_strategies(request, gateway)
    candidate = result["rankings"][0]

    assert candidate["evaluation_status"] == "complete"
    assert candidate["signal_coverage"]["anchors_computed"] == 200
    assert candidate["signal_coverage"]["anchor_limit"] == 200
    assert candidate["folds_requested"] == 3
    assert candidate["folds_evaluated"] == 3
    assert candidate["fold_coverage"] == 1.0
    assert candidate["evidence"]["criteria"]["all_requested_folds_evaluated"] is True
    assert result["validation"]["forecast_signal_anchor_limit"] == 200


def test_forecast_strategy_consumes_canonical_price_forecast(monkeypatch) -> None:
    gateway = FakeGateway()

    def fake_execute_forecast(**kwargs):
        anchor = float(kwargs["prefetched_df"]["close"].iloc[-1])
        direction = 1.0 if len(kwargs["prefetched_df"]) % 2 else -1.0
        return {
            "success": True,
            "forecast_price": [anchor * (1.0 + direction * 0.01)],
        }

    monkeypatch.setattr(
        "mtdata.forecast.forecast.execute_forecast",
        fake_execute_forecast,
    )
    request = StrategyValidateRequest(
        symbol="EURUSD",
        lookback=400,
        candidates=[
            {
                "id": "forecast",
                "type": "forecast_threshold",
                "method": "naive",
                "params": {"lookback": 20},
                "horizon": 1,
                "long_above": 0.0,
                "short_below": 0.0,
            }
        ],
        barrier={"horizon": 1, "tp_pct": 0.15, "sl_pct": 0.15},
        n_splits=2,
        cost_model="fixed",
        spread_bps=1.0,
        bootstrap_samples=100,
        detail="full",
    )

    result = validate_strategies(request, gateway)
    candidate = result["rankings"][0]

    assert candidate["evaluation_status"] == "complete"
    assert candidate["signal_coverage"]["anchors_computed"] == 200
    assert candidate["signal_counts"]["long"] == 100
    assert candidate["signal_counts"]["short"] == 100
    assert candidate["trades"] > 0


def test_forecast_strategy_insufficient_data_explains_threshold_coverage(
    monkeypatch,
) -> None:
    gateway = FakeGateway()
    monkeypatch.setattr(
        "mtdata.forecast.forecast.execute_forecast",
        lambda **_kwargs: {"expected_return": 0.0},
    )
    request = StrategyValidateRequest(
        symbol="EURUSD",
        lookback=400,
        candidates=[
            {
                "id": "forecast",
                "type": "forecast_threshold",
                "method": "naive",
                "params": {"lookback": 20},
                "horizon": 1,
                "long_above": 0.01,
                "short_below": -0.01,
            }
        ],
        barrier={"horizon": 1, "tp_pct": 0.15, "sl_pct": 0.15},
        n_splits=2,
        cost_model="fixed",
        spread_bps=1.0,
        bootstrap_samples=100,
        detail="full",
    )

    result = validate_strategies(request, gateway)
    candidate = result["rankings"][0]

    assert candidate["evaluation_status"] == "insufficient_data"
    assert candidate["insufficient_data_reason"] == "threshold_not_crossed"
    assert candidate["minimum_trades_required"] == 10
    assert candidate["signal_coverage"]["anchors_computed"] == 200
    assert candidate["signal_counts"]["long"] == 0
    assert candidate["signal_counts"]["short"] == 0
    assert candidate["signal_counts"]["neutral"] == 200
    assert candidate["signal_counts"]["non_finite_or_unavailable"] > 0
    assert result["success"] is False
    assert result["error_code"] == "strategy_validation_no_evaluable_candidates"
    assert result["candidate_counts"]["insufficient_data"] == 1
    assert result["candidate_counts"]["complete"] == 0


def test_portfolio_risk_marks_empty_position_book() -> None:
    gateway = FakeGateway()
    gateway.positions = []
    gateway.account_info = lambda: SimpleNamespace(equity=12_345.67, currency="USD")

    result = decompose_portfolio_risk(
        PortfolioRiskDecomposeRequest(),
        gateway,
    )

    assert result["success"] is True
    assert result["empty"] is True
    assert result["status"] == "no_open_positions"
    assert result["portfolio_status"] == "no_open_positions"
    assert result["actionability"] == "informational_no_exposure"
    assert result["positions"] == 0
    assert result["equity"] == 12_345.67
    assert result["currency"] == "USD"
    assert result["valuation_time"].endswith("Z")
    assert result["model_context"]["valuation_time"] == result["valuation_time"]
    assert result["model_context"]["valuation_basis"] == (
        "account_snapshot_no_position_marks"
    )
    assert result["risk"] == []
    assert result["timeframe"] == "H1"
    assert result["holding_periods"] == ["1 H1 bar", "5 H1 bars"]
    assert result["model_context"]["random_seed"] == 42


@pytest.mark.parametrize(
    ("volume", "nearest"),
    [
        (0.0001, 0.01),
        (0.015, 0.01),
        (201.0, 200.0),
    ],
)
def test_portfolio_risk_rejects_invalid_proposed_broker_volume_before_history(
    volume: float,
    nearest: float,
) -> None:
    gateway = FakeGateway()
    gateway.copy_rates_from_pos = MagicMock()

    result = decompose_portfolio_risk(
        PortfolioRiskDecomposeRequest(
            proposed_trade={
                "symbol": "EURUSD",
                "side": "buy",
                "volume": volume,
            },
        ),
        gateway,
    )

    assert result["success"] is False
    assert result["error_code"] == "invalid_proposed_trade_volume"
    assert result["field"] == "proposed_trade.volume"
    assert result["requested_volume"] == volume
    assert result["constraints"] == {
        "volume_min": 0.01,
        "volume_max": 200.0,
        "volume_step": 0.01,
    }
    assert result["nearest_valid_volume"] == nearest
    gateway.copy_rates_from_pos.assert_not_called()


def test_portfolio_risk_resolves_and_accepts_valid_proposed_broker_volume() -> None:
    gateway = FakeGateway()

    result = decompose_portfolio_risk(
        PortfolioRiskDecomposeRequest(
            lookback=300,
            horizon_bars=[1],
            confidence=[0.95],
            simulations=500,
            proposed_trade={
                "symbol": "eur/usd",
                "side": "buy",
                "volume": 0.01,
            },
        ),
        gateway,
    )

    assert result["success"] is True
    assert result["summary"]["positions_after_proposed"] == 1
    proposed = result["proposed_trade"]
    assert proposed["symbol"] == "EURUSD"
    assert proposed["side"] == "buy"
    assert proposed["volume"] == 0.01
    assert proposed["margin_required"] == 10.0
    assert proposed["symbol_input"] == "EUR/USD"
    assert proposed["mark_price"] == pytest.approx(1.1001)
    assert proposed["mark_price_basis"] == "ask"
    assert proposed["quote_time"]


def test_portfolio_risk_same_symbol_proposal_keeps_base_exposure() -> None:
    gateway = FakeGateway()
    gateway.positions = [
        {
            "ticket": 1,
            "symbol": "EURUSD",
            "type": 0,
            "volume": 1.0,
            "price_current": 1.1,
        }
    ]
    common = {
        "lookback": 300,
        "horizon_bars": [1],
        "confidence": [0.95],
        "method": "bootstrap_historical",
        "simulations": 500,
        "seed": 42,
    }

    baseline = decompose_portfolio_risk(
        PortfolioRiskDecomposeRequest(**common),
        gateway,
    )["risk"][0]
    proposed = decompose_portfolio_risk(
        PortfolioRiskDecomposeRequest(
            **common,
            proposed_trade={
                "symbol": "EURUSD",
                "side": "buy",
                "volume": 0.01,
            },
        ),
        gateway,
    )["risk"][0]

    assert proposed["before_cvar"] == pytest.approx(baseline["cvar"])
    assert proposed["incremental_cvar"] == pytest.approx(
        proposed["cvar"] - baseline["cvar"]
    )


def test_portfolio_risk_rejects_unknown_position_side() -> None:
    gateway = FakeGateway()
    gateway.positions = [
        {
            "ticket": 1,
            "symbol": "EURUSD",
            "type": 99,
            "volume": 1.0,
            "price_current": 1.1,
        }
    ]

    result = decompose_portfolio_risk(
        PortfolioRiskDecomposeRequest(
            lookback=300,
            horizon_bars=[1],
            confidence=[0.95],
            simulations=500,
        ),
        gateway,
    )

    assert result["success"] is False
    assert result["error_code"] == "portfolio_pricing_incomplete"
    assert result["failures"] == [
        {
            "symbol": "EURUSD",
            "ticket": 1,
            "reason": "unknown position side: 99",
        }
    ]


def test_bootstrap_window_sums_match_contiguous_slice_sums() -> None:
    values = np.arange(36, dtype=float).reshape(12, 3) / 10.0
    starts = np.asarray([0, 4, 4, 7], dtype=int)

    actual = _bootstrap_window_sums(values, starts, 3)
    expected = np.stack(
        [values[start : start + 3].sum(axis=0) for start in starts]
    )

    assert actual == pytest.approx(expected)


def test_portfolio_mark_freshness_is_aggregated_by_symbol() -> None:
    gateway = FakeGateway()
    calls = []

    def _tick(symbol):
        calls.append(symbol)
        return SimpleNamespace(bid=1.0999, ask=1.1001, time=_now())

    gateway.symbol_info_tick = _tick
    context = _portfolio_mark_context(
        gateway,
        [
            {"ticket": 1, "symbol": "EURUSD"},
            {"ticket": 2, "symbol": "EURUSD"},
            {"ticket": 3, "symbol": "GBPUSD"},
        ],
    )

    assert calls == ["EURUSD", "GBPUSD"]
    assert [
        (row["symbol"], row["positions"])
        for row in context["mark_freshness"]
    ] == [("EURUSD", 2), ("GBPUSD", 1)]


def test_empty_portfolio_marks_are_not_subject_to_live_quote_gate() -> None:
    context = _portfolio_mark_context(FakeGateway(), [])

    assert context["mark_freshness_status"] == "not_applicable"
    assert context["valuation_basis"] == "no_position_marks"
    assert "usable_for_live_trading" not in context


def test_portfolio_mark_conflict_uses_canonical_execution_gate() -> None:
    gateway = FakeGateway()
    now = _now()
    gateway.symbol_info_tick = lambda _symbol: SimpleNamespace(
        bid=1.1000,
        ask=1.1002,
        time=now,
        time_msc=now * 1000,
    )
    gateway.tick_rows = [
        {
            "bid": 1.1001,
            "ask": 1.1003,
            "time": now,
            "time_msc": now * 1000,
        }
    ]

    context = _portfolio_mark_context(
        gateway,
        [{"ticket": 1, "symbol": "EURUSD"}],
    )

    assert context["usable_for_live_trading"] is True
    assert context["mark_freshness"][0]["quote_source_conflict"]
    assert context["unusable_marks"] == []


def test_portfolio_risk_compact_keeps_quote_conflict_warning() -> None:
    gateway = FakeGateway()
    now = _now()
    gateway.symbol_info_tick = lambda _symbol: SimpleNamespace(
        bid=1.1000,
        ask=1.1002,
        time=now,
        time_msc=now * 1000,
    )
    gateway.tick_rows = [
        {
            "bid": 1.1001,
            "ask": 1.1003,
            "time": now,
            "time_msc": now * 1000,
        }
    ]
    gateway.positions = [
        {"ticket": 1, "symbol": "EURUSD", "type": 0, "volume": 1.0, "price_current": 1.1},
    ]

    result = decompose_portfolio_risk(
        PortfolioRiskDecomposeRequest(
            lookback=300,
            horizon_bars=[1],
            confidence=[0.95],
            simulations=500,
            detail="compact",
        ),
        gateway,
    )

    assert result["success"] is True
    assert result["warnings"]
    assert any(str(warning).startswith("EURUSD:") for warning in result["warnings"])
    marks = result["model_context"]["mark_freshness"]
    assert marks[0]["symbol"] == "EURUSD"
    assert marks[0]["warning"]
    assert "quote_source_state" in marks[0]
    assert "quote_source_conflict" not in marks[0]


def test_portfolio_risk_accepts_long_short_proposed_side() -> None:
    gateway = FakeGateway()

    result = decompose_portfolio_risk(
        PortfolioRiskDecomposeRequest(
            lookback=300,
            horizon_bars=[1],
            confidence=[0.95],
            simulations=500,
            proposed_trade={
                "symbol": "EURUSD",
                "side": "long",
                "volume": 0.01,
            },
        ),
        gateway,
    )

    assert result["success"] is True
    assert result["proposed_trade"]["side"] == "buy"
    assert result["proposed_trade"]["mark_price_basis"] == "ask"


def test_portfolio_risk_bootstrap_historical_omits_unused_ewma_half_life() -> None:
    gateway = FakeGateway()
    gateway.positions = [
        {"ticket": 1, "symbol": "EURUSD", "type": 0, "volume": 1.0, "price_current": 1.1},
    ]

    result = decompose_portfolio_risk(
        PortfolioRiskDecomposeRequest(
            lookback=300,
            horizon_bars=[1],
            confidence=[0.95],
            simulations=500,
            method="bootstrap_historical",
        ),
        gateway,
    )

    assert result["success"] is True
    assert result["method"] == "bootstrap_historical"
    assert result["scenario_generation"] == "bootstrap_historical_windows"
    assert "ewma_half_life" not in result["model_context"]


def test_portfolio_risk_bootstrap_historical_rejects_non_default_ewma_half_life() -> None:
    with pytest.raises(ValidationError, match="filtered_historical"):
        PortfolioRiskDecomposeRequest(
            method="bootstrap_historical",
            ewma_half_life=90.0,
        )


def test_portfolio_risk_reconciles_component_expected_shortfall() -> None:
    gateway = FakeGateway()
    gateway.account_info = lambda: SimpleNamespace(currency="USD", equity=25000.0)
    gateway.positions = [
        {"ticket": 1, "symbol": "EURUSD", "type": 0, "volume": 1.0, "price_current": 1.1},
        {"ticket": 2, "symbol": "GBPUSD", "type": 1, "volume": 0.5, "price_current": 1.1},
    ]
    result = decompose_portfolio_risk(
        PortfolioRiskDecomposeRequest(lookback=300, horizon_bars=[1], confidence=[0.95], simulations=500),
        gateway,
    )
    assert result["success"] is True
    assert result["currency"] == "USD"
    assert result["equity"] == 25000.0
    row = result["risk"][0]
    component_total = sum(item["value"] for item in row["component_cvar"])
    assert component_total == pytest.approx(row["cvar"])
    assert row["var_pct_of_equity"] == pytest.approx(row["var"] / 25000.0 * 100.0)
    assert row["cvar_pct_of_equity"] == pytest.approx(row["cvar"] / 25000.0 * 100.0)
    assert sum(item["pct_of_equity"] for item in row["component_cvar"]) == pytest.approx(
        row["cvar_pct_of_equity"]
    )
    assert "correlation_to_one_loss_proxy" not in result["stresses"]
    assert result["stresses"]["perfect_positive_correlation_1sigma"][0]["horizon_bars"] == 1
    stress = result["stresses"]["two_times_worst_simulated_loss"][0]
    assert stress["horizon_bars"] == 1
    assert stress["holding_period"] == "1 H1 bar"
    assert stress["basis"] == "2 * worst_simulated_pnl"
    assert result["stresses"]["two_times_worst_simulated_loss_worst_across_horizons"] == (
        stress
    )
    assert "volatility_double_worst_pnl" not in result["stresses"]
    assert result["timeframe"] == "H1"
    assert result["holding_periods"] == ["1 H1 bar"]
    assert row["holding_period"] == "1 H1 bar"
    assert result["model_context"] == {
        "timeframe": "H1",
        "horizon_bars": [1],
        "holding_periods": ["1 H1 bar"],
        "lookback_requested": 300,
        "confidence_levels": [0.95],
        "simulations": 500,
        "ewma_half_life": 60.0,
        "scenario_generation": "ewma_filtered_bootstrap_windows",
        "random_seed": 42,
        "completion_policy": "fail_closed",
        "valuation_time": result["model_context"]["valuation_time"],
        "valuation_basis": "live_position_marks_with_completed_bar_return_history",
        "data_stale": False,
        "usable_for_live_trading": True,
        "marks_evaluated": 2,
        "unusable_marks": [],
        "aligned_returns": result["summary"]["aligned_rows"],
        "aligned_returns_available": result["model_context"]["aligned_returns_available"],
        "warmup_returns_discarded": result["model_context"]["warmup_returns_discarded"],
        "data_start": result["model_context"]["data_start"],
        "data_end": result["model_context"]["data_end"],
    }
    assert datetime.fromisoformat(
        result["model_context"]["valuation_time"].replace("Z", "+00:00")
    ).tzinfo == timezone.utc
    assert "mark_freshness" not in result["model_context"]
    assert "as_of" not in result["model_context"]


def test_portfolio_risk_fails_closed_on_unusable_material_mark() -> None:
    gateway = FakeGateway()
    gateway.positions = [
        {
            "ticket": 1,
            "symbol": "EURUSD",
            "type": 0,
            "volume": 1.0,
            "price_current": 1.1,
        }
    ]
    stale_mark_time = _now() - 3_600
    gateway.symbol_info_tick = lambda symbol: SimpleNamespace(
        bid=1.0999,
        ask=1.1001,
        time=stale_mark_time,
    )
    gateway.tick_rows = _ticks(10, start=stale_mark_time - 10)

    result = decompose_portfolio_risk(
        PortfolioRiskDecomposeRequest(
            lookback=300,
            horizon_bars=[1],
            confidence=[0.95],
            simulations=500,
        ),
        gateway,
    )

    assert result["success"] is False
    assert result["error_code"] == "portfolio_mark_unusable"
    assert result["failures"][0]["stage"] == "mark_freshness"
    assert result["model_context"]["usable_for_live_trading"] is False
    assert "risk" not in result


def test_portfolio_risk_partial_mode_omits_unusable_mark() -> None:
    gateway = FakeGateway()
    gateway.positions = [
        {
            "ticket": 1,
            "symbol": "EURUSD",
            "type": 0,
            "volume": 1.0,
            "price_current": 1.1,
        },
        {
            "ticket": 2,
            "symbol": "GBPUSD",
            "type": 1,
            "volume": 0.5,
            "price_current": 1.1,
        },
    ]
    now = _now()
    stale = now - 3_600

    def _tick(symbol: str):
        return SimpleNamespace(
            bid=1.0999,
            ask=1.1001,
            time=now if symbol == "EURUSD" else stale,
        )

    def _copy_ticks(symbol, start, end, flags):
        timestamp = now if symbol == "EURUSD" else stale
        return [
            {
                "time": timestamp,
                "time_msc": timestamp * 1000,
                "bid": 1.0999,
                "ask": 1.1001,
            }
        ]

    gateway.symbol_info_tick = _tick
    gateway.copy_ticks_range = _copy_ticks

    result = decompose_portfolio_risk(
        PortfolioRiskDecomposeRequest(
            lookback=300,
            horizon_bars=[1],
            confidence=[0.95],
            simulations=500,
            allow_partial=True,
        ),
        gateway,
    )

    assert result["success"] is True
    assert result["data_quality"]["symbols_modeled"] == ["EURUSD"]
    assert result["data_quality"]["symbols_omitted"] == ["GBPUSD"]
    assert result["data_quality"]["mark_omissions"][0]["symbol"] == "GBPUSD"
    assert any("non-live marks" in warning for warning in result["warnings"])


def test_portfolio_risk_horizons_share_one_stable_calibration_window() -> None:
    gateway = FakeGateway()
    gateway.positions = [
        {"ticket": 1, "symbol": "EURUSD", "type": 0, "volume": 1.0, "price_current": 1.1},
        {"ticket": 2, "symbol": "GBPUSD", "type": 1, "volume": 0.5, "price_current": 1.1},
    ]

    common = dict(
        lookback=300,
        confidence=[0.95],
        method="bootstrap_historical",
        simulations=500,
        seed=7,
    )
    short = decompose_portfolio_risk(
        PortfolioRiskDecomposeRequest(horizon_bars=[1, 5], **common),
        gateway,
    )
    long = decompose_portfolio_risk(
        PortfolioRiskDecomposeRequest(horizon_bars=[1, 50], **common),
        gateway,
    )

    assert short["model_context"]["aligned_returns"] == 300
    assert long["model_context"]["aligned_returns"] == 300
    short_one = next(row for row in short["risk"] if row["horizon_bars"] == 1)
    long_one = next(row for row in long["risk"] if row["horizon_bars"] == 1)
    assert short_one == long_one
    assert short_one["calibration_observations"] == 300
    assert short_one["horizon_windows_available"] == 300


def test_portfolio_risk_converts_log_scenarios_to_simple_return_pnl() -> None:
    gateway = FakeGateway()
    gateway.positions = [
        {
            "ticket": 1,
            "symbol": "EURUSD",
            "type": 0,
            "volume": 1.0,
            "price_current": 1.1,
        }
    ]
    bars = _bars(130)
    price = 2.0
    for row in bars:
        row["open"] = price
        price *= 0.99
        row["close"] = price
        row["high"] = row["open"]
        row["low"] = price
    gateway.bar_rows["EURUSD"] = bars

    result = decompose_portfolio_risk(
        PortfolioRiskDecomposeRequest(
            lookback=120,
            horizon_bars=[2],
            confidence=[0.95],
            method="bootstrap_historical",
            simulations=500,
        ),
        gateway,
    )

    sensitivity = 110_010.0
    expected_two_bar_pnl = sensitivity * ((0.99**2) - 1.0)
    assert result["risk"][0]["worst_simulated_pnl"] == pytest.approx(
        expected_two_bar_pnl
    )
    assert result["stresses"]["worst_historical_bar_pnl"] == pytest.approx(
        sensitivity * -0.01
    )


def test_filtered_historical_shock_uses_pre_shock_volatility() -> None:
    baseline = np.tile(np.array([-0.01, 0.01]), 60)
    values = np.concatenate([baseline, np.array([0.20])])
    returns = pd.DataFrame({"EURUSD": values})
    alpha = 0.1

    standardized, _ = _filtered_historical_returns(returns, alpha=alpha)
    concurrent_vol = returns.ewm(alpha=alpha, adjust=False).std().iloc[-1, 0]

    assert standardized.iloc[-1, 0] == pytest.approx(
        values[-1] / returns.ewm(alpha=alpha, adjust=False).std().shift(1).iloc[-1, 0]
    )
    assert standardized.iloc[-1, 0] > values[-1] / concurrent_vol * 2.0


def test_portfolio_risk_fails_closed_when_symbol_history_is_missing() -> None:
    gateway = FakeGateway()
    gateway.positions = [
        {"ticket": 1, "symbol": "EURUSD", "type": 0, "volume": 1.0, "price_current": 1.1},
        {"ticket": 2, "symbol": "GBPUSD", "type": 1, "volume": 0.5, "price_current": 1.1},
    ]
    gateway.bar_rows["GBPUSD"] = _bars(50)

    result = decompose_portfolio_risk(
        PortfolioRiskDecomposeRequest(
            lookback=300,
            horizon_bars=[1],
            confidence=[0.95],
            simulations=500,
        ),
        gateway,
    )

    assert result["success"] is False
    assert result["error_code"] == "portfolio_pricing_incomplete"
    assert result["failures"] == [
        {
            "symbol": "GBPUSD",
            "stage": "return_history",
            "bars_available": 50,
            "bars_required": 100,
            "reason": "insufficient completed return history",
        }
    ]


def test_portfolio_risk_rejects_unknown_proposed_symbol() -> None:
    gateway = FakeGateway()
    gateway.symbol_info = lambda symbol: None

    result = decompose_portfolio_risk(
        PortfolioRiskDecomposeRequest(
            proposed_trade={
                "symbol": "NOPE",
                "side": "buy",
                "volume": 0.01,
            },
        ),
        gateway,
    )

    assert result["success"] is False
    assert result["error_code"] == "symbol_not_found"
    assert result["symbol"] == "NOPE"


def test_portfolio_risk_fails_closed_when_pricing_unavailable() -> None:
    gateway = FakeGateway()
    gateway.positions = [
        {
            "ticket": 1,
            "symbol": "EURUSD",
            "type": 0,
            "volume": 1.0,
            "price_current": 1.1,
        }
    ]
    gateway.order_calc_profit = lambda *args, **kwargs: None

    result = decompose_portfolio_risk(
        PortfolioRiskDecomposeRequest(
            lookback=300,
            horizon_bars=[1],
            confidence=[0.95],
            simulations=500,
        ),
        gateway,
    )

    assert result["success"] is False
    assert result["error_code"] == "portfolio_pricing_incomplete"
    assert result["failures"][0]["symbol"] == "EURUSD"


def test_portfolio_risk_fails_closed_when_aligned_history_is_short() -> None:
    gateway = FakeGateway()
    gateway.positions = [
        {
            "ticket": 1,
            "symbol": "EURUSD",
            "type": 0,
            "volume": 1.0,
            "price_current": 1.1,
        },
        {
            "ticket": 2,
            "symbol": "GBPUSD",
            "type": 1,
            "volume": 0.5,
            "price_current": 1.1,
        },
    ]
    eurusd = _bars(200)
    gbpusd = _bars(200)
    for row in gbpusd:
        row["time"] -= 10_000_000
    gateway.bar_rows["EURUSD"] = eurusd
    gateway.bar_rows["GBPUSD"] = gbpusd

    result = decompose_portfolio_risk(
        PortfolioRiskDecomposeRequest(
            lookback=300,
            horizon_bars=[1],
            confidence=[0.95],
            simulations=500,
        ),
        gateway,
    )

    assert result["success"] is False
    assert result["error_code"] == "insufficient_data"
    assert result["aligned_rows"] < 100


def test_portfolio_risk_fails_closed_when_partial_history_leaves_no_series() -> None:
    gateway = FakeGateway()
    gateway.positions = [
        {
            "ticket": 1,
            "symbol": "EURUSD",
            "type": 0,
            "volume": 1.0,
            "price_current": 1.1,
        }
    ]
    gateway.bar_rows["EURUSD"] = _bars(50)

    result = decompose_portfolio_risk(
        PortfolioRiskDecomposeRequest(
            lookback=300,
            horizon_bars=[1],
            confidence=[0.95],
            simulations=500,
            allow_partial=True,
        ),
        gateway,
    )

    assert result["success"] is False
    assert result["error_code"] == "insufficient_data"
    assert result["failures"][0]["symbol"] == "EURUSD"


def test_portfolio_risk_discloses_history_omissions_in_partial_mode() -> None:
    gateway = FakeGateway()
    gateway.positions = [
        {"ticket": 1, "symbol": "EURUSD", "type": 0, "volume": 1.0, "price_current": 1.1},
        {"ticket": 2, "symbol": "GBPUSD", "type": 1, "volume": 0.5, "price_current": 1.1},
    ]
    gateway.bar_rows["GBPUSD"] = _bars(50)

    result = decompose_portfolio_risk(
        PortfolioRiskDecomposeRequest(
            lookback=300,
            horizon_bars=[1],
            confidence=[0.95],
            simulations=500,
            allow_partial=True,
        ),
        gateway,
    )

    assert result["success"] is True
    assert result["summary"]["symbols"] == 1
    assert result["summary"]["symbols_requested"] == 2
    assert result["data_quality"]["symbols_modeled"] == ["EURUSD"]
    assert result["data_quality"]["symbols_omitted"] == ["GBPUSD"]
    assert result["data_quality"]["history_failures"][0]["symbol"] == "GBPUSD"
    assert any("allow_partial=true" in warning for warning in result["warnings"])


def test_relative_strength_ranks_and_reports_breadth() -> None:
    gateway = FakeGateway()
    request = MarketRelativeStrengthRequest(
        symbols="EURUSD,GBPUSD,USDJPY",
        horizons=[5, 20],
        weights=[0.4, 0.6],
        volatility_lookback=30,
        limit=3,
    )
    result = rank_relative_strength(request, gateway)
    assert result["success"] is True
    assert len(result["leaders"]) == 2
    assert len(result["laggards"]) == 1
    assert {
        row["symbol"] for row in result["leaders"]
    }.isdisjoint(row["symbol"] for row in result["laggards"])
    assert result["leaders"][0]["rank"] <= result["leaders"][-1]["rank"]
    assert all("score" not in row for row in result["leaders"])
    assert set(result["breadth"]["positive_by_horizon"]) == {"5", "20"}
    assert result["universe_size"] == 3
    assert result["rank_quality"] == "illustrative_small_universe"
    assert result["universe_sensitivity"]["status"] == "small_universe"
    assert result["universe_sensitivity"]["standardized_scores"] == "withheld"
    assert result["score_definition"]["method"] == "rank_only_small_universe"
    assert result["score_definition"]["weights"] == [0.4, 0.6]
    assert all("rank_percentile" not in row for row in result["leaders"])
    assert any("standardized z-scores were withheld" in warning for warning in result["warnings"])
    first_row = [*result["leaders"], *result["laggards"]][0]
    expected_spread_pct = (1.1001 - 1.0999) / 1.1 * 100.0
    assert first_row["spread_pct"] == pytest.approx(expected_spread_pct)
    assert result["units"]["spread_pct"] == "percent (1.0 = 1%)"
    assert result["units"]["breadth.positive_by_horizon"] == (
        "fraction_0_to_1"
    )
    assert result["units"]["breadth.advance_decline_balance"] == (
        "signed_fraction_-1_to_1"
    )
    assert result["units"]["breadth.dispersion"] == "composite_score_stddev"
    assert result["units"]["breadth.above_sma20"] == "fraction_0_to_1"
    assert result["units"]["breadth.above_sma50"] == "fraction_0_to_1"


def test_relative_strength_uses_reconciled_quote_for_spread_filter() -> None:
    gateway = FakeGateway()
    now = _now()
    gateway.symbol_info_tick = lambda _symbol: SimpleNamespace(
        bid=1.10000,
        ask=1.10009,
        time=now + 12,
    )
    gateway.copy_ticks_range = lambda *_args: [
        {
            "time": now - 1,
            "time_msc": (now - 1) * 1000,
            "bid": 1.10004,
            "ask": 1.10005,
        }
    ]

    result = rank_relative_strength(
        MarketRelativeStrengthRequest(
            symbols="EURUSD,GBPUSD,USDJPY",
            horizons=[5],
            weights=[1.0],
            volatility_lookback=30,
            max_spread_pct=0.005,
            limit=3,
            detail="full",
        ),
        gateway,
    )

    assert result["universe_size"] == 3
    assert all(
        row["quote_quality"]["quote_source"] == "mt5.copy_ticks_range"
        for row in result["rankings"]
    )
    assert all(row["spread_pct"] < 0.005 for row in result["rankings"])


def test_relative_strength_keeps_locked_quotes_in_historical_rankings() -> None:
    gateway = FakeGateway()
    now = _now()
    original_tick = gateway.symbol_info_tick
    original_copy_ticks = gateway.copy_ticks_range

    def symbol_info_tick(symbol: str):
        if symbol == "USDJPY":
            return SimpleNamespace(bid=1.1, ask=1.1, time=now)
        return original_tick(symbol)

    def copy_ticks_range(symbol, start, end, flags):
        if symbol == "USDJPY":
            return []
        return original_copy_ticks(symbol, start, end, flags)

    gateway.symbol_info_tick = symbol_info_tick
    gateway.copy_ticks_range = copy_ticks_range

    result = rank_relative_strength(
        MarketRelativeStrengthRequest(
            symbols="EURUSD,GBPUSD,USDJPY",
            horizons=[5],
            weights=[1.0],
            volatility_lookback=30,
            limit=3,
        ),
        gateway,
    )
    assert result["status"] == "ranked"
    assert result["universe_size"] == 3
    assert result["returned_count"] == 3
    ranked_symbols = {
        row["symbol"] for row in [*result["leaders"], *result["laggards"]]
    }
    assert ranked_symbols == {"EURUSD", "GBPUSD", "USDJPY"}
    assert result["data_quality"]["quote_not_live_ready_symbols"] == ["USDJPY"]
    locked = next(
        row for row in [*result["leaders"], *result["laggards"]]
        if row["symbol"] == "USDJPY"
    )
    assert locked["quote_status"] == "locked_quote"
    assert any("not live-ready" in warning for warning in result["warnings"])


def test_relative_strength_keeps_stale_quote_with_fresh_history() -> None:
    gateway = FakeGateway()
    gateway.bar_rows["XAUUSD"] = [
        {**row, "time": row["time"] + 3_600}
        for row in _bars(drift=0.0003)
    ]
    stale_epoch = _now() - 3_600
    original_tick = gateway.symbol_info_tick
    original_copy_ticks = gateway.copy_ticks_range

    def symbol_info_tick(symbol: str):
        if symbol == "XAUUSD":
            return SimpleNamespace(bid=2400.0, ask=2400.2, time=stale_epoch)
        return original_tick(symbol)

    def copy_ticks_range(symbol, start, end, flags):
        if symbol == "XAUUSD":
            return []
        return original_copy_ticks(symbol, start, end, flags)

    gateway.symbol_info_tick = symbol_info_tick
    gateway.copy_ticks_range = copy_ticks_range

    result = rank_relative_strength(
        MarketRelativeStrengthRequest(
            symbols="EURUSD,GBPUSD,USDJPY,XAUUSD",
            horizons=[5],
            weights=[1.0],
            volatility_lookback=30,
            limit=4,
            detail="full",
        ),
        gateway,
    )

    assert result["universe_size"] == 4
    assert {row["symbol"] for row in result["rankings"]} == {
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "XAUUSD",
    }
    xau = next(row for row in result["rankings"] if row["symbol"] == "XAUUSD")
    assert xau["data_window"]["freshness"] == "fresh"
    assert xau["quote_quality"]["usable_for_live_trading"] is False
    assert _relative_strength_quote_status(xau["quote_quality"]) == "stale"
    assert result["data_quality"]["quote_not_live_ready_symbols"] == ["XAUUSD"]


def test_relative_strength_withholds_directional_ranking_for_tied_scores() -> None:
    gateway = FakeGateway()
    gateway.bar_rows["GBPUSD"] = [dict(row) for row in gateway.bar_rows["EURUSD"]]

    result = rank_relative_strength(
        MarketRelativeStrengthRequest(
            symbols="EURUSD,GBPUSD",
            horizons=[5],
            weights=[1.0],
            volatility_lookback=30,
            limit=2,
            detail="full",
        ),
        gateway,
    )

    assert result["status"] == "tied"
    assert result["rank_quality"] == "tied_scores"
    assert result["leaders"] == []
    assert result["laggards"] == []
    assert result["rankings"] == []
    assert result["score_definition"]["score_tie_tolerance"] == 1e-12


def test_relative_strength_projects_rows_by_detail() -> None:
    gateway = FakeGateway()
    base = dict(
        symbols="EURUSD,GBPUSD,USDJPY",
        horizons=[5],
        weights=[1.0],
        volatility_lookback=30,
        limit=3,
    )

    compact = rank_relative_strength(
        MarketRelativeStrengthRequest(**base, detail="compact"), gateway
    )
    summary = rank_relative_strength(
        MarketRelativeStrengthRequest(**base, detail="summary"), gateway
    )
    full = rank_relative_strength(
        MarketRelativeStrengthRequest(**base, detail="full"), gateway
    )

    compact_row = compact["leaders"][0]
    summary_row = summary["leaders"][0]
    full_row = full["leaders"][0]
    assert "quote_quality" not in compact_row
    assert "data_window" not in compact_row
    assert "quote_status" in compact_row
    assert "history_status" in compact_row
    assert "raw_momentum" in compact_row
    assert set(summary_row) <= {
        "symbol",
        "rank",
        "score",
        "rank_percentile",
        "quote_status",
        "history_status",
    }
    assert "raw_momentum" not in summary_row
    assert "quote_quality" in full_row
    assert "data_window" in full_row


def test_relative_strength_limit_caps_total_returned_rankings() -> None:
    gateway = FakeGateway()
    result = rank_relative_strength(
        MarketRelativeStrengthRequest(
            symbols="EURUSD,GBPUSD,USDJPY",
            horizons=[5],
            weights=[1.0],
            volatility_lookback=30,
            limit=1,
        ),
        gateway,
    )

    assert result["success"] is True
    assert result["returned_count"] == 1
    assert result["applied_limit"] == 1
    assert len(result["leaders"]) == 1
    assert result["laggards"] == []


def test_relative_strength_full_detail_keeps_every_ranking_collection_bounded() -> None:
    gateway = FakeGateway()
    gateway.bar_rows = {
        f"SYM{index:02d}": _bars(drift=(index - 8) * 0.00002)
        for index in range(17)
    }
    symbols = ",".join(gateway.bar_rows)

    result = rank_relative_strength(
        MarketRelativeStrengthRequest(
            symbols=symbols,
            horizons=[5],
            weights=[1.0],
            volatility_lookback=30,
            limit=10,
            max_symbols=20,
            detail="full",
        ),
        gateway,
    )

    assert result["success"] is True
    assert result["universe_size"] == 17
    assert result["returned_count"] == 10
    assert len(result["leaders"]) == 5
    assert len(result["laggards"]) == 5
    assert len(result["rankings"]) == 10
    assert "all_rankings" not in result
    assert len({row["symbol"] for row in result["rankings"]}) == 10
    assert [row["rank"] for row in result["laggards"]] == sorted(
        row["rank"] for row in result["laggards"]
    )
    assert result["ranking_selection"]["method"] == "strongest_and_weakest_tails"


def test_relative_strength_default_uses_dominant_endpoint_cohort() -> None:
    gateway = FakeGateway()
    gateway.bar_rows["WTI_U6"] = [
        {**row, "time": row["time"] - 4 * 3600}
        for row in _bars(drift=0.0003)
    ]

    result = rank_relative_strength(
        MarketRelativeStrengthRequest(
            horizons=[5],
            weights=[1.0],
            volatility_lookback=30,
            limit=4,
            detail="full",
        ),
        gateway,
    )

    alignment = result["data_window"]["endpoint_alignment"]
    assert result["success"] is True
    assert result["status"] == "ranked"
    assert result["returned_count"] == 3
    assert alignment["comparable"] is True
    assert alignment["cohort_policy"] == "dominant_latest_endpoint_aligned_cohort"
    assert alignment["excluded_symbols"] == [
        {
            "symbol": "WTI_U6",
            "bar_close": alignment["excluded_symbols"][0]["bar_close"],
            "reason": "outside dominant endpoint-aligned cohort",
        }
    ]
    assert "WTI_U6" not in {
        row["symbol"] for row in [*result["leaders"], *result["laggards"]]
    }


def test_relative_strength_reports_mixed_bar_endpoints_and_alignment_windows() -> None:
    gateway = FakeGateway()
    gateway.bar_rows["GBPUSD"] = [
        {**row, "time": row["time"] - 7200}
        for row in gateway.bar_rows["GBPUSD"]
    ]

    result = rank_relative_strength(
        MarketRelativeStrengthRequest(
            symbols="EURUSD,GBPUSD,USDJPY",
            horizons=[5],
            weights=[1.0],
            volatility_lookback=30,
            limit=3,
            detail="full",
        ),
        gateway,
    )

    assert result["success"] is True
    assert result["analysis_as_of"].endswith("Z")
    assert result["data_window"]["effective_common"]["start"]
    assert result["data_window"]["effective_common"]["end"]
    assert result["data_window"]["effective_common"]["timestamp_basis"] == "bar_open"
    alignment = result["data_window"]["endpoint_alignment"]
    assert alignment["timestamp_basis"] == "bar_close"
    assert alignment["earliest_bar_close"]
    assert alignment["latest_bar_close"]
    assert "earliest" not in alignment
    assert "latest" not in alignment
    assert alignment["cohort_policy"] == "dominant_latest_endpoint_aligned_cohort"
    assert alignment["comparable"] is True
    assert alignment["excluded_symbols"] == [
        {
            "symbol": "GBPUSD",
            "bar_close": alignment["excluded_symbols"][0]["bar_close"],
            "reason": "outside dominant endpoint-aligned cohort",
        }
    ]
    assert result["status"] == "ranked"
    assert result["returned_count"] == 2
    assert "GBPUSD" not in {
        row["symbol"] for row in [*result["leaders"], *result["laggards"]]
    }
    assert result["data_quality"]["scored_symbols"] == 2


def test_relative_strength_rejects_one_symbol_before_fetching_history() -> None:
    with pytest.raises(ValueError, match="requires at least two comma-separated symbols"):
        MarketRelativeStrengthRequest(symbols="EURUSD")


def test_relative_strength_fetches_external_benchmark_without_ranking_it() -> None:
    gateway = FakeGateway()
    request = MarketRelativeStrengthRequest(
        symbols="EURUSD,GBPUSD",
        benchmark="USDJPY",
        horizons=[5],
        weights=[1.0],
        volatility_lookback=30,
        limit=2,
        detail="full",
    )

    result = rank_relative_strength(request, gateway)

    assert result["success"] is True
    assert result["universe_size"] == 2
    assert result["data_quality"]["selected_symbols"] == 2
    assert result["data_quality"]["data_symbols_fetched"] == 3
    assert "USDJPY" not in {row["symbol"] for row in result["rankings"]}


@pytest.mark.parametrize(
    ("candidate", "benchmark", "path"),
    [
        ("GBPUSD", "EURUSD", "Forex\\Majors"),
        ("ETHUSD", "BTCUSD", "Crypto\\Majors"),
    ],
)
def test_relative_strength_supports_pairwise_benchmark_comparison(
    candidate: str,
    benchmark: str,
    path: str,
) -> None:
    gateway = FakeGateway()
    gateway.bar_rows[candidate] = _bars(drift=0.0002)
    gateway.bar_rows[benchmark] = _bars(drift=0.00005)
    gateway.symbols_get = lambda: [
        SimpleNamespace(name=candidate, path=path, visible=True),
        SimpleNamespace(name=benchmark, path=path, visible=True),
    ]
    request = MarketRelativeStrengthRequest(
        symbols=candidate,
        benchmark=benchmark,
        horizons=[5],
        weights=[1.0],
        volatility_lookback=30,
        limit=1,
        detail="full",
    )

    result = rank_relative_strength(request, gateway)

    assert result["success"] is True
    assert result["status"] == "compared"
    assert result["rank_quality"] == "pairwise_benchmark"
    assert result["ranking_selection"]["method"] == (
        "pairwise_benchmark_comparison"
    )
    assert result["score_definition"]["method"] == (
        "weighted_volatility_scaled_benchmark_residual_momentum"
    )
    assert result["breadth"]["status"] == "not_applicable_pairwise"
    assert result["leaders"][0]["symbol"] == candidate
    assert result["laggards"] == []
    assert result["rankings"][0]["symbol"] == candidate
    assert result["factor"]["source"] == benchmark


def test_relative_strength_rejects_unknown_group_before_history_fetch() -> None:
    gateway = FakeGateway()
    gateway.copy_rates_from_pos = MagicMock()

    result = rank_relative_strength(
        MarketRelativeStrengthRequest(group="NoSuchGroup"),
        gateway,
    )

    assert result["error_code"] == "symbol_group_error"
    assert result["requested_group"] == "NoSuchGroup"
    assert result["available_groups"] == ["Forex\\Majors"]
    assert result["related_tools"] == ["symbols_list"]
    gateway.copy_rates_from_pos.assert_not_called()


def test_relative_strength_normalizes_repeated_group_separators() -> None:
    result = rank_relative_strength(
        MarketRelativeStrengthRequest(group="Forex\\\\Majors"),
        FakeGateway(),
    )

    assert result.get("error_code") != "symbol_group_error"


def test_relative_strength_does_not_rank_benchmark_from_requested_symbols() -> None:
    gateway = FakeGateway()
    request = MarketRelativeStrengthRequest(
        symbols="EURUSD,GBPUSD,USDJPY",
        benchmark="USDJPY",
        horizons=[5],
        weights=[1.0],
        volatility_lookback=30,
        limit=3,
        detail="full",
    )

    result = rank_relative_strength(request, gateway)

    assert result["success"] is True
    assert result["universe_size"] == 2
    assert "USDJPY" not in {row["symbol"] for row in result["rankings"]}
    assert result["data_quality"]["benchmark_excluded_from_ranking"] == "USDJPY"


def test_relative_strength_rejects_unavailable_explicit_benchmark() -> None:
    result = rank_relative_strength(
        MarketRelativeStrengthRequest(
            symbols="EURUSD,GBPUSD",
            benchmark="NOTREAL",
            horizons=[5],
            weights=[1.0],
        ),
        FakeGateway(),
    )

    assert result["error_code"] == "benchmark_not_found"
    assert result["benchmark"] == "NOTREAL"


def test_relative_strength_requires_two_available_explicit_candidates() -> None:
    result = rank_relative_strength(
        MarketRelativeStrengthRequest(
            symbols="EURUSD,NOTREAL",
            horizons=[5],
            weights=[1.0],
        ),
        FakeGateway(),
    )

    assert result["error_code"] == "insufficient_data"
    assert result["missing_symbols"] == ["NOTREAL"]


def test_relative_strength_reports_all_candidates_with_insufficient_history() -> None:
    gateway = FakeGateway()
    gateway.bar_rows["EURUSD"] = _bars(20)
    gateway.bar_rows["GBPUSD"] = _bars(20, drift=0.0001)

    result = rank_relative_strength(
        MarketRelativeStrengthRequest(
            symbols="EURUSD,GBPUSD",
            horizons=[5],
            weights=[1.0],
        ),
        gateway,
    )

    assert result["error_code"] == "insufficient_data"
    assert result["empty_reason"] == "insufficient_history"
    assert result["empty_reason_counts"] == {"insufficient_history": 2}
    assert "quote/volume" not in result["message"]


def test_relative_strength_omits_unavailable_candidate_from_valid_basket() -> None:
    result = rank_relative_strength(
        MarketRelativeStrengthRequest(
            symbols="EURUSD,GBPUSD,USDJPY,NOTREAL",
            horizons=[5],
            weights=[1.0],
            limit=3,
        ),
        FakeGateway(),
    )

    assert result["success"] is True
    assert result["universe_size"] == 3
    assert result["data_quality"]["missing_symbols"] == ["NOTREAL"]
    assert result["data_quality"]["unavailable_symbols"] == ["NOTREAL"]
    assert result["data_quality"]["history_unavailable_symbols"] == []
    assert any("NOTREAL" in warning for warning in result["warnings"])


def test_relative_strength_omits_candidate_with_insufficient_history() -> None:
    gateway = FakeGateway()
    gateway.bar_rows["XAUUSD"] = _bars(drift=0.0003)
    gateway.bar_rows["USDJPY"] = _bars(20, drift=-0.00005)

    result = rank_relative_strength(
        MarketRelativeStrengthRequest(
            symbols="EURUSD,GBPUSD,USDJPY,XAUUSD",
            horizons=[5],
            weights=[1.0],
            limit=4,
        ),
        gateway,
    )

    assert result["success"] is True
    assert result["universe_size"] == 3
    assert result["data_quality"]["missing_symbols"] == ["USDJPY"]
    assert result["data_quality"]["unavailable_symbols"] == []
    assert result["data_quality"]["history_unavailable_symbols"] == ["USDJPY"]
    assert any("insufficient history" in warning for warning in result["warnings"])


def test_relative_strength_reports_empty_filtered_result_without_warnings() -> None:
    request = MarketRelativeStrengthRequest(
        horizons=[5],
        weights=[1.0],
        min_tick_volume=1_000_000_000,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = rank_relative_strength(request, FakeGateway())

    assert result["success"] is True
    assert result["status"] == "no_matches"
    assert result["returned_count"] == 0
    assert result["breadth"]["positive_by_horizon"] == {"5": None}
    assert result["empty_reason"] == "tick_volume_filter"
    assert result["empty_reason_counts"] == {"tick_volume_filter": 3}
    assert result["message"] == "No symbols passed --min-tick-volume."


def test_relative_strength_reports_empty_spread_filter_result() -> None:
    result = rank_relative_strength(
        MarketRelativeStrengthRequest(
            horizons=[5],
            weights=[1.0],
            max_spread_pct=0.0,
        ),
        FakeGateway(),
    )

    assert result["status"] == "no_matches"
    assert result["empty_reason"] == "spread_filter"
    assert result["empty_reason_counts"] == {"spread_filter": 3}
    assert result["message"] == "No symbols passed --max-spread-pct."


def test_relative_strength_reports_factor_alignment_empty_reason() -> None:
    gateway = FakeGateway()
    eurusd = _bars(100)
    gbpusd = [dict(row) for row in eurusd]
    for index, row in enumerate(gbpusd):
        row["close"] *= 1.0 + index * 0.00001
        if index < 72:
            row["time"] -= 7 * 86_400
    gateway.bar_rows["EURUSD"] = eurusd
    gateway.bar_rows["GBPUSD"] = gbpusd

    result = rank_relative_strength(
        MarketRelativeStrengthRequest(
            symbols="EURUSD,GBPUSD",
            horizons=[5, 20],
            weights=[0.5, 0.5],
            volatility_lookback=60,
            limit=2,
        ),
        gateway,
    )

    assert result["success"] is True
    assert result["status"] == "no_matches"
    assert result["empty_reason"] == "insufficient_factor_alignment"
    assert result["empty_reason_counts"] == {
        "insufficient_factor_alignment": 2
    }
    assert result["empty_reason_details"] == {
        "maximum_aligned_observations": 28,
        "required_aligned_observations": 60,
    }
    assert "28 aligned observations" in result["message"]
    assert "60 are required" in result["message"]
    assert "quote/volume" not in result["message"]
    assert "--volatility-lookback" in result["remediation"]


def test_position_reversal_waits_for_fresh_signal_after_max_hold() -> None:
    n = 15
    prices = [1.0 + value / 100.0 for value in range(n)]
    frame = pd.DataFrame({"open": prices, "close": prices})
    signal = pd.Series([1.0] * n)
    entries, exits, _ = _position_reversal_returns(frame, signal, max_hold_bars=3)
    assert len(entries) == 1
    assert int(exits[0] - (entries[0] + 1)) == 3


def test_strategy_validate_ema_cross_shortcut_uses_horizon_only_barrier() -> None:
    request = StrategyValidateRequest(
        symbol="EURUSD",
        lookback=400,
        strategy="ema_cross",
        n_splits=2,
        cost_model="fixed",
        spread_bps=1.0,
        bootstrap_samples=100,
    )
    assert request.barrier.tp_pct is None
    assert request.barrier.sl_pct is None
    result = validate_strategies(request, FakeGateway())
    assert result.get("error_code") != "incompatible_barrier_for_state_reversal"
