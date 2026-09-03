from __future__ import annotations

from typing import Any

import pytest
from jsonschema import Draft202012Validator

from mtdata.bootstrap.tools import bootstrap_tools
from mtdata.core.schema_attach import get_public_tool_schemas

_PAIR_BARRIER = {
    "kind": "tp_sl",
    "unit": "pct",
    "take_profit": 1,
    "stop_loss": 1,
}


SCHEMA_CONTRACT_CASES = [
    (
        "wait-requires-timeframe",
        "wait_event",
        {"timeframe": "H1"},
        {"symbol": "EURUSD"},
    ),
    (
        "asset-performance-order-needs-rank",
        "asset_performance",
        {"rank_by": "day", "order": "desc"},
        {"order": "desc"},
    ),
    (
        "asset-performance-insider-controls",
        "asset_performance",
        {"universe": "insider"},
        {"universe": "insider", "rank_by": "day"},
    ),
    (
        "calendar-period-kind",
        "calendar",
        {"view": "period", "kind": "earnings"},
        {"view": "period"},
    ),
    (
        "causal-selector",
        "causal_discover_signals",
        {"symbols": "EURUSD,GBPUSD"},
        {"symbols": "EURUSD,GBPUSD", "group": "Forex"},
    ),
    (
        "causal-selector-required",
        "causal_discover_signals",
        {"group": "Forex"},
        {},
    ),
    (
        "causal-significance",
        "causal_discover_signals",
        {"group": "Forex", "significance": 0.05},
        {"group": "Forex", "significance": 1.0},
    ),
    (
        "cointegration-selector",
        "cointegration_test",
        {"group": "Forex"},
        {"symbols": "EURUSD,GBPUSD", "group": "Forex"},
    ),
    (
        "cointegration-engle-minimum",
        "cointegration_test",
        {"group": "Forex", "window_bars": 20, "min_overlap": 20},
        {"group": "Forex", "window_bars": 19, "min_overlap": 20},
    ),
    (
        "cointegration-johansen-minimum",
        "cointegration_test",
        {
            "group": "Forex",
            "method": "johansen",
            "window_bars": 2,
            "min_overlap": 2,
        },
        {
            "group": "Forex",
            "method": "johansen",
            "window_bars": 1,
            "min_overlap": 2,
        },
    ),
    (
        "cointegration-johansen-significance",
        "cointegration_test",
        {"group": "Forex", "method": "johansen", "significance": 0.05},
        {"group": "Forex", "method": "johansen", "significance": 0.03},
    ),
    (
        "correlation-selector",
        "correlation_matrix",
        {"symbols": "EURUSD,GBPUSD"},
        {"symbols": "EURUSD,GBPUSD", "group": "Forex"},
    ),
    (
        "cross-correlation-pair",
        "cross_correlation",
        {"symbols": "EURUSD,GBPUSD", "max_lag": 0},
        {"symbols": "EURUSD,GBPUSD,USDJPY"},
    ),
    (
        "cross-correlation-minimums",
        "cross_correlation",
        {"symbols": "EURUSD,GBPUSD", "window_bars": 10, "min_overlap": 5},
        {"symbols": "EURUSD,GBPUSD", "window_bars": 9, "min_overlap": 5},
    ),
    (
        "forecast-as-of-range",
        "forecast_generate",
        {"symbol": "EURUSD", "as_of": "2026-01-01"},
        {
            "symbol": "EURUSD",
            "as_of": "2026-01-01",
            "start": "2025-01-01",
        },
    ),
    (
        "forecast-model-cache",
        "forecast_generate",
        {"symbol": "EURUSD", "model_cache": "ephemeral"},
        {
            "symbol": "EURUSD",
            "model_cache": "ephemeral",
            "model_id": "m/s/h",
        },
    ),
    (
        "forecast-async-cache",
        "forecast_generate",
        {"symbol": "EURUSD", "async_mode": True},
        {
            "symbol": "EURUSD",
            "async_mode": True,
            "model_cache": "ephemeral",
        },
    ),
    (
        "conformal-as-of-range",
        "forecast_conformal_intervals",
        {"symbol": "EURUSD", "as_of": "2026-01-01"},
        {
            "symbol": "EURUSD",
            "as_of": "2026-01-01",
            "end": "2026-01-01",
        },
    ),
    (
        "forecast-train-as-of-range",
        "forecast_train",
        {"symbol": "EURUSD", "method": "theta", "as_of": "2026-01-01"},
        {
            "symbol": "EURUSD",
            "method": "theta",
            "as_of": "2026-01-01",
            "start": "2025-01-01",
        },
    ),
    (
        "volatility-as-of-range",
        "forecast_volatility_estimate",
        {"symbol": "EURUSD", "as_of": "2026-01-01"},
        {
            "symbol": "EURUSD",
            "as_of": "2026-01-01",
            "end": "2026-01-01",
        },
    ),
    (
        "tune-as-of-range",
        "forecast_tune_genetic",
        {"symbol": "EURUSD", "as_of": "2026-01-01"},
        {
            "symbol": "EURUSD",
            "as_of": "2026-01-01",
            "end": "2026-01-01",
        },
    ),
    (
        "tune-trading-costs",
        "forecast_tune_genetic",
        {
            "symbol": "EURUSD",
            "metric": "win_rate",
            "spread_bps": 1,
            "commission_bps_per_side": 0,
        },
        {"symbol": "EURUSD", "metric": "win_rate"},
    ),
    (
        "tune-annualized-sample",
        "forecast_tune_optuna",
        {
            "symbol": "EURUSD",
            "metric": "sharpe_ratio",
            "steps": 30,
            "spread_bps": 1,
            "commission_bps_per_side": 0,
        },
        {
            "symbol": "EURUSD",
            "metric": "sharpe_ratio",
            "steps": 29,
            "spread_bps": 1,
            "commission_bps_per_side": 0,
        },
    ),
    (
        "optimize-trading-costs",
        "forecast_optimize_hints",
        {
            "symbol": "EURUSD",
            "fitness_metric": "composite",
            "steps": 30,
            "spread_bps": 1,
            "commission_bps_per_side": 0,
        },
        {
            "symbol": "EURUSD",
            "fitness_metric": "composite",
            "steps": 30,
        },
    ),
    (
        "optimize-fitness-weights",
        "forecast_optimize_hints",
        {
            "symbol": "EURUSD",
            "fitness_metric": "composite",
            "steps": 30,
            "spread_bps": 1,
            "commission_bps_per_side": 0,
            "fitness_weights": {"avg_rmse": 1},
        },
        {
            "symbol": "EURUSD",
            "fitness_metric": "avg_rmse",
            "fitness_weights": {"avg_rmse": 1},
        },
    ),
    (
        "strategy-backtest-cost-model",
        "strategy_backtest",
        {"symbol": "EURUSD", "cost_model": "fixed", "spread_bps": 1},
        {"symbol": "EURUSD", "cost_model": "auto", "spread_bps": 1},
    ),
    (
        "strategy-validate-source",
        "strategy_validate",
        {"symbol": "EURUSD", "strategy": "sma_cross"},
        {"symbol": "EURUSD"},
    ),
    (
        "strategy-validate-barrier",
        "strategy_validate",
        {
            "symbol": "EURUSD",
            "strategy": "sma_cross",
            "barrier": {"horizon": 12},
        },
        {
            "symbol": "EURUSD",
            "strategy": "sma_cross",
            "barrier": {"horizon": 12, "tp_pct": 0.5},
        },
    ),
    (
        "candles-cursor-start",
        "data_fetch_candles",
        {"symbol": "EURUSD", "start": "2026-01-01", "cursor": "x"},
        {"symbol": "EURUSD", "cursor": "x"},
    ),
    (
        "candles-end-only-selection",
        "data_fetch_candles",
        {"symbol": "EURUSD", "end": "2026-01-01", "selection": "last_n"},
        {"symbol": "EURUSD", "end": "2026-01-01", "selection": "first_n"},
    ),
    (
        "ticks-cursor-bounds",
        "data_fetch_ticks",
        {
            "symbol": "EURUSD",
            "start": "2026-01-01",
            "end": "2026-01-02",
            "cursor": "x",
        },
        {"symbol": "EURUSD", "start": "2026-01-01", "cursor": "x"},
    ),
    (
        "history-time-controls",
        "trade_history",
        {"end": "2026-01-02", "minutes_back": 60},
        {"start": "2026-01-01", "minutes_back": 60},
    ),
    (
        "history-order-side",
        "trade_history",
        {"history_kind": "orders", "side": "buy"},
        {"history_kind": "orders", "side": "long"},
    ),
    (
        "trade-ticket-bounds",
        "trade_modify",
        {"ticket": 1, "price": 1.1},
        {"ticket": 0, "price": 1.1},
    ),
    (
        "trade-open-side",
        "trade_get_open",
        {"side": "long"},
        {"side": "sideways"},
    ),
    (
        "trade-pending-order-type",
        "trade_get_pending",
        {"order_type": "sell_stop"},
        {"order_type": "BUY"},
    ),
    (
        "trade-stress-nonempty",
        "trade_stress_test",
        {"shocks": {"EURUSD": -2.0}},
        {"shocks": {}},
    ),
    (
        "journal-time-controls",
        "trade_journal_analyze",
        {"end": "2026-01-02", "minutes_back": 60},
        {"start": "2026-01-01", "minutes_back": 60},
    ),
    (
        "barrier-method-kind",
        "forecast_barrier_prob",
        {
            "symbol": "EURUSD",
            "method": "closed_form",
            "barrier": {"kind": "single_price", "level": 2},
        },
        {
            "symbol": "EURUSD",
            "method": "closed_form",
            "barrier": _PAIR_BARRIER,
        },
    ),
    (
        "barrier-grid-preset",
        "forecast_barrier_optimize",
        {"symbol": "EURUSD", "grid_style": "preset", "preset": "intraday"},
        {"symbol": "EURUSD", "grid_style": "preset"},
    ),
    (
        "barrier-grid-preset-implied-style",
        "forecast_barrier_optimize",
        {"symbol": "EURUSD", "preset": "intraday"},
        {"symbol": "EURUSD", "grid_style": "fixed", "preset": "intraday"},
    ),
    (
        "trade-place-market-price",
        "trade_place",
        {"symbol": "EURUSD", "volume": 0.1, "order_type": "BUY"},
        {
            "symbol": "EURUSD",
            "volume": 0.1,
            "order_type": "BUY",
            "price": 1.1,
        },
    ),
    (
        "trade-place-pending-price",
        "trade_place",
        {
            "symbol": "EURUSD",
            "volume": 0.1,
            "order_type": "BUY_LIMIT",
            "price": 1.1,
        },
        {"symbol": "EURUSD", "volume": 0.1, "order_type": "BUY_LIMIT"},
    ),
    (
        "trade-modify-change",
        "trade_modify",
        {"ticket": 1, "price": 1.1},
        {"ticket": 1},
    ),
    (
        "trade-modify-clear-conflict",
        "trade_modify",
        {"ticket": 1, "clear_stop_loss": True, "stop_loss": 0},
        {"ticket": 1, "clear_stop_loss": True, "stop_loss": 1},
    ),
    (
        "trade-close-scope",
        "trade_close",
        {"ticket": 1},
        {},
    ),
    (
        "trade-close-volume",
        "trade_close",
        {"ticket": 1, "volume": 0.1},
        {"symbol": "EURUSD", "volume": 0.1},
    ),
    (
        "trade-close-live-confirmation",
        "trade_close",
        {"symbol": "EURUSD", "dry_run": False, "confirm_close_all": True},
        {"symbol": "EURUSD", "dry_run": False},
    ),
    (
        "labels-as-of-range",
        "labels_triple_barrier",
        {"symbol": "EURUSD", "barrier": _PAIR_BARRIER, "as_of": "2026-01-01"},
        {
            "symbol": "EURUSD",
            "barrier": _PAIR_BARRIER,
            "as_of": "2026-01-01",
            "end": "2026-01-01",
        },
    ),
    (
        "microstructure-window",
        "market_microstructure_analyze",
        {"symbol": "EURUSD", "start": "2026-01-01", "end": "2026-01-02"},
        {"symbol": "EURUSD", "start": "2026-01-01"},
    ),
    (
        "execution-quality-window",
        "trade_execution_quality",
        {"start": "2026-01-01"},
        {"start": "2026-01-01", "minutes_back": 60},
    ),
    (
        "execution-quality-markouts",
        "trade_execution_quality",
        {"markout_seconds": [1, 3600]},
        {"markout_seconds": [0]},
    ),
    (
        "portfolio-risk-method-controls",
        "portfolio_risk_decompose",
        {"method": "bootstrap_historical"},
        {"method": "bootstrap_historical", "ewma_half_life": 20},
    ),
    (
        "portfolio-risk-horizons",
        "portfolio_risk_decompose",
        {"horizon_bars": [1, 50]},
        {"horizon_bars": [0]},
    ),
    (
        "portfolio-risk-confidence",
        "portfolio_risk_decompose",
        {"confidence": [0.95]},
        {"confidence": [0.5]},
    ),
    (
        "relative-strength-selector",
        "market_relative_strength",
        {"symbols": "EURUSD,GBPUSD"},
        {"symbols": "EURUSD,GBPUSD", "group": "Forex"},
    ),
    (
        "relative-strength-nonblank-symbols",
        "market_relative_strength",
        {"symbols": "EURUSD,GBPUSD"},
        {"symbols": "   "},
    ),
    (
        "market-radar-nonblank-symbols",
        "market_radar",
        {},
        {"symbols": "   "},
    ),
    (
        "relative-strength-universe",
        "market_relative_strength",
        {"universe": "all", "group": "Forex"},
        {"universe": "all"},
    ),
    (
        "market-scan-selector",
        "market_scan",
        {"symbols": "EURUSD"},
        {"symbols": "EURUSD", "group": "Forex"},
    ),
    (
        "market-scan-indicator-period",
        "market_scan",
        {"rsi_length": 1, "sma_period": 1},
        {"rsi_length": 0, "sma_period": 1},
    ),
    (
        "market-status-selector",
        "market_status",
        {"symbol": "EURUSD"},
        {"symbol": "EURUSD", "venue": "NYSE"},
    ),
    (
        "news-ticker-symbol",
        "news",
        {"view": "ticker", "symbol": "AAPL"},
        {"view": "ticker"},
    ),
    (
        "news-unified-pagination",
        "news",
        {"view": "unified"},
        {"view": "unified", "page": 2},
    ),
    (
        "screener-mode-controls",
        "screener",
        {"list_filters": True, "filter_name": "Market Cap."},
        {"filter_name": "Market Cap."},
    ),
    (
        "volume-profile-bucket-control",
        "volume_profile_levels",
        {"symbol": "EURUSD", "bucket_count": 10},
        {"symbol": "EURUSD", "bucket_count": 10, "bucket_size": 0.1},
    ),
    (
        "volume-profile-window-mode",
        "volume_profile_levels",
        {"symbol": "EURUSD", "timeframe": "H1", "lookback": 20},
        {"symbol": "EURUSD", "start": "2026-01-01", "timeframe": "H1"},
    ),
    (
        "patterns-engine-mode",
        "patterns_detect",
        {"symbol": "EURUSD", "mode": "classic", "engine": "native"},
        {"symbol": "EURUSD", "engine": "native"},
    ),
    (
        "patterns-last-bars-mode",
        "patterns_detect",
        {"symbol": "EURUSD", "mode": "all", "last_n_bars": 3},
        {"symbol": "EURUSD", "mode": "harmonic", "last_n_bars": 3},
    ),
    (
        "patterns-ensemble-weights",
        "patterns_detect",
        {
            "symbol": "EURUSD",
            "mode": "classic",
            "ensemble": True,
            "ensemble_weights": {"native": 1},
        },
        {
            "symbol": "EURUSD",
            "mode": "classic",
            "ensemble_weights": {"native": 1},
        },
    ),
    (
        "regime-threshold-method",
        "regime_detect",
        {"symbol": "EURUSD", "method": "bocpd", "threshold": 0.5},
        {"symbol": "EURUSD", "threshold": 0.5},
    ),
    (
        "regime-threshold-bounds",
        "regime_detect",
        {"symbol": "EURUSD", "method": "bocpd", "threshold": 1.0},
        {"symbol": "EURUSD", "method": "bocpd", "threshold": 1.1},
    ),
    (
        "regime-default-history-minimum",
        "regime_detect",
        {"symbol": "EURUSD", "fetch_limit": 20},
        {"symbol": "EURUSD", "fetch_limit": 19},
    ),
    (
        "regime-non-rule-history-minimum",
        "regime_detect",
        {"symbol": "EURUSD", "method": "bocpd", "fetch_limit": 10},
        {"symbol": "EURUSD", "method": "bocpd", "fetch_limit": 9},
    ),
    (
        "regime-garch-target",
        "regime_detect",
        {"symbol": "EURUSD", "method": "garch", "target": "return"},
        {"symbol": "EURUSD", "method": "garch", "target": "price"},
    ),
    (
        "temporal-timeframe-group",
        "temporal_analyze",
        {"symbol": "EURUSD", "timeframe": "D1", "group_by": "month"},
        {"symbol": "EURUSD", "timeframe": "D1", "group_by": "hour"},
    ),
    (
        "report-range",
        "report_generate",
        {"symbol": "EURUSD", "end": "2026-01-01"},
        {"symbol": "EURUSD", "start": "2025-01-01"},
    ),
    (
        "report-minimal-method-count",
        "report_generate",
        {"symbol": "EURUSD", "template": "minimal", "methods": ["theta"]},
        {
            "symbol": "EURUSD",
            "template": "minimal",
            "methods": ["theta", "drift"],
        },
    ),
    (
        "report-template-timeframe",
        "report_generate",
        {"symbol": "EURUSD", "template": "scalping", "timeframe": "M5"},
        {"symbol": "EURUSD", "template": "scalping", "timeframe": "D1"},
    ),
    (
        "model-delete-confirmation",
        "forecast_models_delete",
        {
            "model_id": "m/s/h",
            "dry_run": False,
            "confirm_model_id": "m/s/h",
        },
        {"model_id": "m/s/h", "dry_run": False},
    ),
    (
        "options-heston-parameter-bundle",
        "options_barrier_price",
        {
            "spot": 100,
            "strike": 100,
            "barrier": 120,
            "maturity_days": 30,
            "model": "heston",
            "heston_v0": 0.04,
            "heston_kappa": 1,
            "heston_theta": 0.04,
            "heston_sigma": 0.3,
            "heston_rho": -0.5,
        },
        {
            "spot": 100,
            "strike": 100,
            "barrier": 120,
            "maturity_days": 30,
            "model": "heston",
        },
    ),
    (
        "options-bsm-excludes-heston",
        "options_barrier_price",
        {"spot": 100, "strike": 100, "barrier": 120, "maturity_days": 30},
        {
            "spot": 100,
            "strike": 100,
            "barrier": 120,
            "maturity_days": 30,
            "heston_v0": 0.04,
        },
    ),
    (
        "options-bsm-volatility",
        "options_barrier_price",
        {
            "spot": 100,
            "strike": 100,
            "barrier": 120,
            "maturity_days": 30,
            "volatility": 0.2,
        },
        {
            "spot": 100,
            "strike": 100,
            "barrier": 120,
            "maturity_days": 30,
            "volatility": 0,
        },
    ),
    (
        "stationarity-target-lookback",
        "stationarity_test",
        {"symbol": "EURUSD", "target": "close", "lookback": 20},
        {"symbol": "EURUSD", "lookback": 20},
    ),
    (
        "seasonality-period-minimum",
        "seasonality_detect",
        {"symbol": "EURUSD", "min_period": 2},
        {"symbol": "EURUSD", "min_period": 1},
    ),
    (
        "outlier-threshold",
        "outliers_detect",
        {"symbol": "EURUSD", "threshold": 0.1},
        {"symbol": "EURUSD", "threshold": 0},
    ),
    (
        "support-resistance-lookback",
        "support_resistance_levels",
        {"symbol": "EURUSD", "lookback": 3},
        {"symbol": "EURUSD", "lookback": 2},
    ),
    (
        "confluence-lookback",
        "confluence_levels",
        {"symbol": "EURUSD", "lookback": 3},
        {"symbol": "EURUSD", "lookback": 2},
    ),
    (
        "forecast-task-list-window",
        "forecast_task_list",
        {"since_minutes": 0},
        {"since_minutes": -1},
    ),
    (
        "forecast-tuning-seed",
        "forecast_tune_optuna",
        {"symbol": "EURUSD", "seed": 0},
        {"symbol": "EURUSD", "seed": -1},
    ),
    (
        "symbols-list-mode-search",
        "symbols_list",
        {"list_mode": "groups", "search_mode": "group"},
        {"list_mode": "groups", "search_mode": "name"},
    ),
    (
        "pivot-cutoff-aliases",
        "pivot_compute_points",
        {"symbol": "EURUSD", "end": "2026-01-01"},
        {"symbol": "EURUSD", "end": "2026-01-01", "as_of": "2026-01-01"},
    ),
]


@pytest.fixture(scope="module")
def public_schemas() -> dict[str, dict[str, Any]]:
    bootstrap_tools()
    return get_public_tool_schemas()


def test_all_public_tool_schemas_are_valid_draft_2020_12(
    public_schemas: dict[str, dict[str, Any]],
) -> None:
    for tool_name, schema in public_schemas.items():
        try:
            Draft202012Validator.check_schema(schema)
        except Exception as exc:  # pragma: no cover - assertion context
            pytest.fail(f"{tool_name} has an invalid JSON Schema: {exc}")


@pytest.mark.parametrize(
    ("case_name", "tool_name", "valid_payload", "invalid_payload"),
    SCHEMA_CONTRACT_CASES,
    ids=[case[0] for case in SCHEMA_CONTRACT_CASES],
)
def test_public_schema_matches_cross_field_contract(
    public_schemas: dict[str, dict[str, Any]],
    case_name: str,
    tool_name: str,
    valid_payload: dict[str, Any],
    invalid_payload: dict[str, Any],
) -> None:
    validator = Draft202012Validator(public_schemas[tool_name])
    valid_errors = list(validator.iter_errors(valid_payload))
    invalid_errors = list(validator.iter_errors(invalid_payload))

    assert not valid_errors, (
        case_name,
        [error.message for error in valid_errors],
    )
    assert invalid_errors, case_name
