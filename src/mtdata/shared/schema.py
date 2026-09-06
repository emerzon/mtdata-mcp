"""
Shared JSON schema helpers for CLI/server tool inputs.

Provides reusable $defs such as TimeframeSpec and helpers to apply them
to per-tool parameter schemas.
"""
import inspect
import types
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    get_args,
    get_origin,
    is_typeddict,
)

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from typing_extensions import TypedDict

from .annotations import get_runtime_annotations, get_runtime_signature
from .constants import TIMEFRAME_MAP
from .parameter_contracts import PARAMETER_HELP


class DimensionalityReductionSpec(BaseModel):
    """Dimensionality-reduction method and its method-specific parameters."""

    model_config = ConfigDict(extra="forbid")

    method: str = Field(description="Registered dimensionality-reduction method name.")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Method-specific dimensionality-reduction parameters.",
    )


class BarrierPairSpec(BaseModel):
    """Take-profit and stop-loss pair expressed in one unit family."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["tp_sl"] = Field(
        "tp_sl",
        description=(
            "Optional barrier-family discriminator. Accepted so the same TP/SL "
            "object can be reused by forecast_barrier_prob and labels_triple_barrier."
        ),
    )
    unit: Literal["price", "pct", "ticks", "pips"] = Field(
        description=(
            "Barrier unit: price means absolute instrument price levels; pct, "
            "ticks, and pips mean positive distances from the entry price. "
            "ticks uses the broker trade tick/point (0.1 pip on typical 5-digit "
            "FX), not conventional FX pips. pips uses forex_pip_size."
        )
    )
    take_profit: float = Field(
        gt=0.0,
        description=(
            "Positive absolute take-profit level when unit=price, otherwise a "
            "positive distance from entry in percent, trade ticks, or FX pips."
        ),
    )
    stop_loss: float = Field(
        gt=0.0,
        description=(
            "Positive absolute stop-loss level when unit=price, otherwise a "
            "positive distance from entry in percent, trade ticks, or FX pips."
        ),
    )

    def as_legacy_kwargs(self) -> Dict[str, float]:
        suffix = {
            "price": "abs",
            "pct": "pct",
            "ticks": "ticks",
            "pips": "pips",
        }[self.unit]
        return {
            f"tp_{suffix}": float(self.take_profit),
            f"sl_{suffix}": float(self.stop_loss),
        }


def normalize_required_symbol(value: Any, *, error_message: str = "symbol is required") -> str:
    """Normalize a required symbol while allowing callers to preserve wording."""
    normalized = str(value or "").strip().upper()
    if not normalized:
        raise ValueError(error_message)
    return normalized


def normalize_optional_symbol(value: Any) -> Optional[str]:
    """Normalize an optional symbol to uppercase, preserving missing values."""
    if not value:
        return None
    return str(value).strip().upper()


def validate_complete_time_window(
    start: Any,
    end: Any,
    *,
    error_message: str = "start and end must be supplied together",
) -> None:
    """Require start and end to be supplied together."""
    if bool(start) != bool(end):
        raise ValueError(error_message)


def validate_as_of_time_window(
    as_of: Any,
    start: Any,
    end: Any,
    *,
    error_message: str = "as_of cannot be combined with start/end",
) -> None:
    """Reject an as-of anchor combined with an explicit start/end range."""
    if as_of and (start or end):
        raise ValueError(error_message)


def reject_removed_field(values: Any, *, field_name: str, replacement: str) -> Any:
    if isinstance(values, dict) and field_name in values:
        raise ValueError(f"{field_name} was removed; use {replacement}")
    return values


PARAM_HINTS = {
    **PARAMETER_HELP,
    "direction": "Trade direction (long/short).",
    "limit": "Maximum count; see command help for what is counted.",
    "offset": "Rows to skip before returning paginated results.",
    "start": (
        "Inclusive start time. A date-only value starts at 00:00 UTC, except "
        "session-based D1/W1/MN1 market-data windows use the broker calendar."
    ),
    "end": (
        "Inclusive end time. A date-only value includes the full UTC day, except "
        "session-based D1/W1/MN1 market-data windows use the broker calendar."
    ),
    "search": "Case-insensitive search text used to filter returned rows.",
    "search_term": "Case-insensitive search text used to filter returned rows.",
    "category": "Category filter for catalog/listing tools.",
    "group_by": "Temporal grouping: dow/day_of_week, hour, month, or all.",
    "day_of_week": "Weekday filter (0-6 or Mon..Sun).",
    "month": "Month filter (1-12 or Jan..Dec).",
    "time_range": "Time-of-day filter 'HH:MM-HH:MM' (start inclusive, end exclusive; wraps midnight).",
    "rank_by": (
        "Ranking metric for radar and scan tools. Default abs_price_change_pct uses "
        "completed-bar closes, not the live bid/ask; use abs_live_price_change_pct "
        "to rank by executable quotes."
    ),
    "return_mode": "Return calculation mode: pct or log.",
    "ohlcv": (
        "Returned OHLCV column selector (e.g. 'close', 'high,low'). Projection "
        "runs after denoise and indicators, which receive full source OHLCV."
    ),
    "indicators": "Indicators as compact specs like 'rsi_14', 'rsi(length=14)', 'macd(12,26,9)', or 'macd(fast=12,slow=26,signal=9)', or JSON like '[{\"name\":\"rsi\",\"params\":{\"length\":14}}]'. Bare names such as 'rsi' are also accepted.",
    "denoise": "Denoise preset name or JSON spec. Examples: --denoise kalman or --denoise '{\"method\":\"kalman\",\"params\":{\"lookback\":100}}'.",
    "simplify": "Simplify preset name or JSON spec. Examples: --simplify select, --simplify '{\"mode\":\"select\",\"method\":\"lttb\",\"ratio\":0.2}', or --simplify select --simplify-params \"ratio=0.2\".",
    "include_spread": (
        "Request MT5 historical per-bar spread values; when unavailable, the result "
        "reports a single non-historical reference or spread_mode=unavailable."
    ),
    "include_incomplete": "Include the latest forming candle; defaults to false. Compact candle responses expose forming_candle_status=skipped and an inclusion hint when a forming bar is omitted; full detail also includes forming-candle counts and booleans.",
    "allow_stale": (
        "Allow unrecognized stale closed bars for unbounded latest-N queries; "
        "defaults to false. Recognized weekend/session closures still return the "
        "last session bar and set freshness_policy_relaxed."
    ),
    "explain_indicators": "Add compact latest-value interpretation notes for requested indicators; defaults to false.",
    "method": "Method/algorithm for this tool.",
    "adapter": "Optional forecast adapter-family filter, such as statsforecast or sktime.",
    "mode": "Mode for this tool.",
    "source": "Input data source selector; volume profile uses auto, ticks, or m1_bars.",
    "engine": "Detection engine or comma-separated engines (for ensemble mode).",
    "transform": "Preprocessing transform applied before analysis, such as log_return, pct, diff, level, or log_level depending on the tool.",
    "min_overlap": "Minimum overlapping transformed samples required for each pair before a pairwise statistic is calculated.",
    "min_regime_bars": "Minimum bars a detected regime must span; shorter runs are merged to reduce noisy state flicker.",
    "ensemble": "Enable multi-engine consensus merge when true.",
    "ensemble_weights": "Optional JSON weight map used when ensemble aggregation is weighted.",
    "library": "Forecast library/group (e.g. native, statsforecast, sktime).",
    "model": "Model identifier for this tool.",
    "template": (
        "Report template: minimal fast context+forecast (default), basic balanced research, "
        "advanced regimes/HAR/conformal, scalping M5, intraday H1, swing H4/D1, "
        "or position D1/W1."
    ),
    "horizon": (
        "Forecast horizon in bars, counted from the open of the current forming "
        "bar unless --as-of or a historical range is set."
    ),
    "steps": "Number of backtest anchors or steps to run.",
    "spacing": "Spacing between backtest anchors (in bars).",
    "methods": "One or more method names (comma-separated or space-separated).",
    "timeframes": "One or more MT5 timeframes to evaluate.",
    "alpha": "Alpha parameter for the selected method.",
    "params": "Method parameters as JSON or k=v pairs. Examples: --params alpha=0.3,beta=0.1 or --params '{\"alpha\":0.3,\"beta\":0.1}'.",
    "params_per_method": "Per-method params map (e.g. {method: {k: v}}).",
    "as_of": "Inclusive reference cutoff; a date-only value includes that full trading day.",
    "ci_alpha": "Interval alpha = 1 - nominal coverage; 0.05 requests 95% bands.",
    "features": (
        "Feature spec as JSON or k=v pairs. Examples: "
        "--features future_covariates=hour or --features "
        "'{\"indicators\":\"rsi(14),roc(12)\","
        "\"future_covariates\":[\"hour\",\"dow\"],"
        "\"observed_future_policy\":\"carry_forward\"}'."
    ),
    "dimred": "Dimensionality-reduction method and its method-specific parameters.",
    "target_spec": "Target spec (JSON or k=v).",
    "quantity": "Quantity to model (price/return/volatility).",
    "target": "Target series (price/return).",
    "points": "Target point count.",
    "ratio": "Target compression ratio.",
    "epsilon": "Tolerance value (e.g. RDP).",
    "max_error": "Max approximation error.",
    "segments": "Segment count.",
    "bucket_seconds": "Resample bucket size in seconds.",
    "buffer_seconds": "Extra seconds to wait after the candle close before returning.",
    "schema": "Encoding schema (e.g. delta).",
    "bits": "Bits per symbol for encoding schemas.",
    "paa": "PAA segments for symbolic representation.",
    "znorm": "Apply z-normalization before processing.",
    "threshold_pct": "Segmentation change threshold (percent).",
    "threshold": "Change-point probability threshold (0-1).",
    "value_col": "Column name to use for value-based operations.",
    "lookback": "Historical bars to use.",
    "horizon_bars": (
        "Holding period in bars of the requested timeframe. Default 1 is one-bar "
        "VaR; pass 5 to match portfolio_risk_decompose."
    ),
    "min_bars": "Exclude grouped rows with fewer than this many bars.",
    "last_n_bars": "Restrict pattern checks to the most recent N bars.",
    "spacing_pct": "Spacing as percent of duration.",
    "history_kind": "History type (deals or orders).",
    "list_mode": "List mode (symbols or groups).",
    "search_mode": "Symbol search mode: auto, name, description, group, exact, or all.",
    "sections": "Comma-separated output sections to include.",
    "include_sections": (
        "Only execute these template sections; comma-separated or repeated values. "
        "Names include context, pivot, contexts_multi, pivot_multi, volatility, "
        "backtest, forecast, barriers, patterns, confluence, market, "
        "execution_gates, session, news, temporal, volume_profile, regime, "
        "volatility_har_rv, and forecast_conformal; availability varies by template."
    ),
    "max_sections": "Maximum report sections to include after filtering.",
    "include_account": "Include account snapshot fields in the session context.",
    "region": "Market region filter or status region.",
    "timezone_display": "Display timezone for market-hours timestamps.",
    "price_field": "Quote field to use as the ticker price, such as bid, ask, last, or mid.",
    "volume": "Order volume (lots).",
    "risk_pct": (
        "Account risk in percent (0.5 means 0.5% of equity)."
    ),
    "comment": "Order comment tag.",
    "deviation": "Max slippage (points).",
    "order_type": "Required order type: BUY/SELL for market orders, or BUY_LIMIT/BUY_STOP/SELL_LIMIT/SELL_STOP for pending orders.",
    "price": "Entry price (required for pending).",
    "stop_loss": (
        "Stop-loss price. Required for trade_place unless --require-sl-tp false."
    ),
    "take_profit": (
        "Take-profit price. Required for trade_place unless --require-sl-tp false."
    ),
    "stop_limit_price": "Limit price activated after a stop-limit order's trigger price.",
    "require_sl_tp": "Require both stop_loss and take_profit for market and pending orders and fail if protection cannot be attached. Defaults to true.",
    "auto_close_on_sl_tp_fail": "Always-on policy: if a filled market order cannot attach TP/SL, immediately try to close the unprotected position. Not configurable.",
    "idempotency_key": "Optional durable SQLite dedupe key for retrying the same live request. Dry-run previews are not stored; completed live outcomes persist across processes and restarts for the configured TTL (24 hours by default).",
    "ticket": "Ticket/order ID.",
    "expiration": (
        "Expiration time/date. Trade-order YYYY-MM-DD values last through 23:59:59 "
        "in the client calendar; also accepts a dateparser string, UTC epoch seconds, "
        "or GTC. Option tools accept YYYY-MM-DD."
    ),
    "dry_run": "Preview the action without applying changes.",
    "check_only": "Return sample sufficiency/status checks without running the full analysis.",
    "pnl_filter": "Filter positions by profit state: all, profit, or loss.",
    "confirm_close_all": "Required confirmation flag when closing all matching positions.",
    "column_style": (
        "Display label style for trade-history tables: snake_case (canonical JSON) "
        "or humanized (TOON/display labels only)."
    ),
    "breakdown_limit": "Maximum rows per journal breakdown table.",
    "min_sample": "Recommended minimum realized exit deals for journal statistics.",
    "sizing": (
        "Position-sizing specification for fixed-fraction or Kelly sizing. "
        "Kelly avg_win and avg_loss are stake-normalized R-multiples, not "
        "account-currency averages from trade_journal_analyze."
    ),
    "strict_risk": "Block positive suggested volume when broker minimum volume would exceed requested risk.",
    "include_pending": "Include pending orders in exposure/risk calculations.",
    "entry": "Proposed entry price; when omitted, live quote is used where supported.",
    "position_ticket": "Filter history rows by the linked position ticket.",
    "deal_ticket": "Filter deal history by deal ticket.",
    "order_ticket": "Filter history by order ticket.",
    "minutes_back": "Look back this many minutes from end/now instead of using start.",
    "cursor": (
        "Opaque continuation token from pagination.next_cursor. Reuse it with "
        "the same filters."
    ),
    "spread_bps": (
        "Round-trip spread in basis points deducted from every simulated trade. "
        "Required with commission_bps_per_side for trading-metric searches."
    ),
    "commission_bps_per_side": (
        "Commission in basis points per side, deducted twice per simulated "
        "round-trip. Required with spread_bps for trading-metric searches."
    ),
    "min_strike": "Minimum option strike to include before pagination.",
    "max_strike": "Maximum option strike to include before pagination.",
    "min_moneyness_pct": (
        "Minimum moneyness percent: (strike / underlying_price - 1) * 100."
    ),
    "max_moneyness_pct": (
        "Maximum moneyness percent: (strike / underlying_price - 1) * 100."
    ),
    "quote_usable_only": (
        "Keep only contracts with a two-sided quote and a provider quote "
        "timestamp within the live age threshold. Yahoo and Tradier do not "
        "supply option-quote timestamps, so this filter is rejected as "
        "capability_unavailable."
    ),
    "max_quote_age_seconds": (
        "Maximum age in seconds for a provider quote timestamp. Unknown quote "
        "timestamps are excluded. Yahoo and Tradier do not supply option-quote "
        "timestamps, so this filter is rejected as capability_unavailable."
    ),
    "sort_by": (
        "Option-chain sort: nearest_strike, strike, open_interest, volume, or "
        "moneyness_pct."
    ),
    "min_strength": (
        "Candlestick strength threshold 0.0-1.0, default 0.70. "
        "Strength uses the detected candle's OHLC geometry plus pattern "
        "reliability and span. "
        "Use 0.30-0.50 for exploratory scans, 0.50-0.70 for broader "
        "trading context, and 0.70+ for stricter high-conviction detections."
    ),
    "min_gap": "Minimum bars between detected patterns.",
    "robust_only": (
        "Restrict candlestick detection to a curated subset of established "
        "multi-bar pattern types; does not change min_strength."
    ),
    "whitelist": "Comma-separated pattern names to include.",
    "universe": "Symbol scan universe: visible (fast default) or all (includes hidden tradable symbols and may be slower).",
    "series_time": "Series timestamp format (string or epoch).",
    "include_completed": (
        "Include completed lifecycle structures alongside forming results. "
        "Candlestick mode always scans closed-bar detections; use last_n_bars "
        "to restrict their recency. Harmonic mode always returns both states."
    ),
    "include_series": "Include raw series in output.",
    "config": "Pattern-specific config overrides (JSON or k=v).",
    "top_k": "Return top-K results.",
    "top_n": "Return the top N rows or candidates.",
    "tp_abs": "Take-profit absolute price level.",
    "sl_abs": "Stop-loss absolute price level.",
    "tp_pct": "Take-profit distance in percent (e.g. 0.5 means 0.5%, not 50%).",
    "sl_pct": "Stop-loss distance in percent (e.g. 0.5 means 0.5%, not 50%).",
    "tp_ticks": "Take-profit barrier distance in ticks.",
    "sl_ticks": "Stop-loss barrier distance in ticks.",
    "label_on": "Barrier evaluation basis: close or high_low.",
    "barrier": "Single-price or take-profit/stop-loss barrier specification.",
    "barriers": "Take-profit and stop-loss values expressed in one unit family.",
    "mu": "Drift override for closed-form barrier method.",
    "sigma": "Volatility override for closed-form barrier method.",
    "search_space": "Genetic search space (JSON or k=v).",
    "metric": "Optimization metric.",
    "fitness_metric": "Forecast tuning fitness metric, such as composite or a single score name.",
    "fitness_weights": "JSON metric-weight map used when fitness_metric=composite.",
    "max_search_time_seconds": "Maximum tuning search time budget in seconds.",
    "tradable_only": "Only keep barrier candidates that pass tradability/cost viability checks.",
    "min_ev": "Minimum expected-value threshold for barrier candidates.",
    "min_edge": "Minimum edge over breakeven required for barrier candidates.",
    "min_kelly": "Minimum Kelly fraction required for barrier candidates.",
    "population": "Population size.",
    "generations": "Generation count.",
    "n_trials": "Optuna trial count.",
    "timeout": "Optimization timeout in seconds.",
    "n_jobs": "Parallel worker count for optimization.",
    "sampler": "Optuna sampler strategy.",
    "pruner": "Optuna pruner strategy.",
    "study_name": "Optuna study name (for persistence/resume).",
    "storage": (
        "Optuna storage URL (for persistence/resume); URL credentials are redacted "
        "from public output."
    ),
    "model_id": "Trained model-store identifier.",
    "task_id": "Forecast background task identifier.",
    "data_scope": "Task data-scope filter.",
    "since_minutes": "Only include task/activity records newer than this many minutes.",
    "timeout_seconds": "Maximum seconds to wait before returning.",
    "older_than_days": "Model age threshold in days for cleanup.",
    "available_only": "Only return methods currently available in this environment.",
    "show_unavailable": "Include unavailable methods/models and their requirement notes.",
    "supports_ci": "Filter to methods that support confidence intervals.",
    "supports_training": "Filter to methods that support background training.",
    "crossover_rate": "Genetic crossover probability (0-1).",
    "mutation_rate": "Genetic mutation probability (0-1).",
    "seed": "Random seed for reproducibility.",
    "trade_threshold": "Trade threshold for backtests.",
    "slippage_bps": "Backtest slippage per fill side in basis points; see the command-specific default.",
    "objective": "Optimization objective.",
    "return_grid": "Include full grid results in output.",
    "candidate_filter": "Barrier candidates to return: all or mathematically viable.",
    "concise": "Return a shorter barrier-optimization payload when true.",
    "grid_style": "TP/SL grid style.",
    "preset": "TP/SL grid preset: scalp, intraday, swing, or position.",
    "tp_min": "Minimum TP grid distance: percent in pct mode (0.5 means 0.5%) or ticks in ticks mode.",
    "tp_max": "Maximum TP grid distance: percent in pct mode (0.5 means 0.5%) or ticks in ticks mode.",
    "tp_steps": "Number of TP grid steps.",
    "sl_min": "Minimum SL grid distance: percent in pct mode (0.5 means 0.5%) or ticks in ticks mode.",
    "sl_max": "Maximum SL grid distance: percent in pct mode (0.5 means 0.5%) or ticks in ticks mode.",
    "sl_steps": "Number of SL grid steps.",
    "vol_window": "Lookback window for volatility-based grid.",
    "vol_min_mult": "Minimum volatility multiple for grid.",
    "vol_max_mult": "Maximum volatility multiple for grid.",
    "vol_steps": "Number of volatility grid steps.",
    "vol_floor_pct": "Minimum TP/SL percent floor when using volatility grid.",
    "vol_floor_ticks": "Minimum volatility-derived barrier distance in ticks.",
    "ratio_min": "Minimum TP/SL ratio for ratio grid.",
    "ratio_max": "Maximum TP/SL ratio for ratio grid.",
    "ratio_steps": "Number of ratio grid steps.",
    "refine": "Run a second-stage refinement around best grid point.",
    "refine_radius": "Refinement radius around best grid point.",
    "refine_steps": "Number of refinement steps per axis.",
    "optimizer": "Barrier optimizer backend: grid or optuna.",
    "fast_defaults": "Use a faster low-cost optimization profile (fewer sims/steps/trials). Example: '--fast-defaults true'.",
    "search_profile": "Search intensity profile: fast, medium, or long.",
    "statistical_robustness": "Enable statistical robustness diagnostics for the selected barrier pair.",
    "target_ci_width": "Requested probability CI width used for minimum-simulation guidance.",
    "n_seeds_stability": "Number of alternate seed re-runs for cross-seed stability checks.",
    "enable_bootstrap": "Enable bootstrap uncertainty estimation for selected metrics.",
    "n_bootstrap": "Number of bootstrap resamples when bootstrap uncertainty is enabled.",
    "enable_convergence_check": "Enable convergence diagnostics for the selected objective metric.",
    "convergence_window": "Rolling window size used by the convergence diagnostic.",
    "convergence_threshold": "Tolerance threshold used by the convergence diagnostic.",
    "enable_power_analysis": "Enable statistical power analysis for the selected candidate.",
    "power_effect_size": "Minimum detectable effect size assumed by power analysis.",
    "enable_sensitivity_analysis": "Enable local TP/SL sensitivity analysis around the selected candidate.",
    "sensitivity_params": "List of barrier parameters to vary during sensitivity analysis.",
    "gap_aware_stops": "Use adverse first-crossing prices for stop-loss gap payoffs.",
    "enable_drift_stress": "Evaluate the selected barriers under scaled historical-drift scenarios.",
    "drift_stress_multipliers": "Historical-drift multipliers used for barrier stress scenarios.",
    "enable_oos_validation": "Run held-out walk-forward barrier selection and realized validation.",
    "oos_folds": "Number of held-out walk-forward barrier validation folds.",
    "oos_n_sims": "Simulation paths per held-out barrier validation fold.",
    "oos_holdout_bars": "Trailing bars reserved for walk-forward barrier validation.",
    "ensemble_methods": "Comma-list or array of member simulators for method=ensemble.",
    "ensemble_agg": "Ensemble aggregation: median or weighted_mean (equal weights by default).",
    "optuna_pareto": "Enable Optuna multi-objective Pareto optimization.",
    "optuna_pareto_objectives": "JSON metric->direction map for Pareto optimization.",
    "pareto_limit": "Maximum number of Pareto-front rows returned.",
    "option_type": "Option side filter: call, put, or both.",
    "min_open_interest": "Minimum option open interest filter.",
    "min_volume": "Minimum option volume filter.",
    "risk_free_rate": "Annual risk-free rate as a decimal fraction; 0.05 = 5% (default 0.02).",
    "dividend_yield": "Annual dividend yield for option pricing/calibration.",
    "maturity_days": "Option time-to-maturity in days.",
    "spot": "Current underlying spot price.",
    "strike": "Option strike price.",
    "max_contracts": "Maximum option contracts to use during calibration.",
    "barrier_type": "Barrier style: up_in, up_out, down_in, down_out.",
    "rebate": (
        "Cash rebate: for knock-out options, paid when the barrier is hit; for "
        "knock-in options, paid at expiry if the barrier is never hit."
    ),
    "volatility": "Annualized volatility as a decimal fraction; 0.20 = 20% (default 0.2).",
    # Finviz parameters
    "news_type": "News type: 'news' or 'blogs'.",
    "filters": "JSON filter dict for stock screener.",
    "view": "Screener view: overview, valuation, financial, ownership, performance, technical.",
    "option": "Insider activity type: latest, top week, top owner trade, insider buy, insider sale.",
    "period": "Finviz earnings period: This Week, Next Week, Previous Week, This Month.",
    "impact": "Economic calendar impact filter: low, medium, high, or comma-separated (high,medium).",
    "page": PARAMETER_HELP["page"],
    "name": "Name of the item to describe.",
    "tolerance_pct": (
        "Level-clustering tolerance in percent; 0.15 means 0.15%."
    ),
    "tolerance_points": "Tolerance as a broker-point count; for five-digit EURUSD, 10 points at point=0.00001 gives a 0.0001 price width.",
    "min_touches": "Minimum historical tests/touches required for a level.",
    "max_levels": "Maximum support/resistance levels to return per side or method.",
    "max_distance_pct": "Maximum distance from current price as percent; pass none/null where supported for all levels.",
    "min_source_families": "Minimum independent level-source families required for confluence. Default 2 requires agreement from at least two families.",
    "pivot_timeframe": "MT5 timeframe used for formula pivot levels; D1 gives conventional daily pivots.",
    "sr_timeframe": "Support/resistance timeframe; auto merges M15, H1, H4, and D1 where supported.",
    "pivot_method": "Pivot formula to use: classic, fibonacci, camarilla, woodie, or demark.",
    "volume_weighting": "Volume weighting mode for level scoring: off or auto.",
    "reaction_bars": "Bars after each level touch used to measure bounce/reaction strength.",
    "adx_period": "ADX lookback period in bars for trend-strength scoring.",
    "decay_half_life_bars": "Optional half-life in bars for recency decay in level scoring.",
    "volume_profile_source": "Volume-profile input source: auto, ticks, or m1_bars.",
    "volume_profile_max_tick_window_days": "Maximum raw-tick lookback window in days for confluence volume profile.",
    "volume_profile_max_ticks": "Maximum raw ticks fetched for confluence volume profile.",
    "spread": "Compute and include bid/ask spread metrics from returned market data.",
    "venue": "Static major-equity venue calendar identifier.",
    "price_source": "Volume-profile price source: mid, last, bid, or ask.",
    "volume_source": (
        "Volume-profile volume source: auto, real_volume, tick_volume, "
        "volume_real, volume, or tick_count. tick_count is rejected with "
        "source=m1_bars; use tick_volume or source=ticks."
    ),
    "bucket_size": "Volume-profile bucket width in symbol price units.",
    "bucket_points": "Volume-profile bucket width in MT5 points.",
    "bucket_count": "Target number of volume-profile price buckets.",
    "max_buckets": "Maximum volume-profile buckets returned after auto sizing.",
    "value_area_pct": (
        "Value area in percent for VAH/VAL; 70 means 70%."
    ),
    "reference_price": "Reference price used for nearest-level context; defaults to current quote when available.",
    "max_tick_window_days": "Maximum raw-tick lookback window in days for volume profile.",
    "max_ticks": "Maximum raw ticks fetched for volume profile.",
    "max_m1_bars": "Maximum M1 bars fetched when using M1-bar volume-profile approximation.",
    "max_regimes": "Maximum regime summary rows to return in compact output.",
    "limit_per_bucket": "Maximum news items returned per bucket.",
    "core_only": "Return only methods implemented by the core package.",
}


_TIMEFRAME_CHOICES = tuple(TIMEFRAME_MAP.keys())
TimeframeLiteral = Literal[_TIMEFRAME_CHOICES]  # type: ignore
AutoTimeframeLiteral = Union[TimeframeLiteral, Literal["auto"]]
CANONICAL_OUTPUT_SHAPE_DETAILS = ("compact", "standard", "summary", "full")
DetailLiteral = Literal["compact", "standard", "summary", "full"]

# pandas-ta-classic categories are part of this public request contract. Keep
# the small, stable vocabulary local so importing shared schemas does not import
# pandas and the full indicator registry on every CLI/server process start.
_CATEGORY_CHOICES = [
    "candles",
    "cycles",
    "momentum",
    "overlap",
    "performance",
    "statistics",
    "trend",
    "volatility",
    "volume",
]

if _CATEGORY_CHOICES:
    # Create a Literal type alias dynamically
    CategoryLiteral = Literal[tuple(_CATEGORY_CHOICES)]  # type: ignore
else:
    CategoryLiteral = str  # fallback

IndicatorNameLiteral = str

class IndicatorSpec(TypedDict, total=False):
    """Structured TI spec: name with optional numeric or boolean params.

    Note: 'name' accepts any string to allow compact forms like "rsi(20)".
    The optional 'params' field accepts either positional values or a named parameter map.
    Numeric and boolean parameters retain their types.
    """
    name: str
    params: Union[List[Any], Dict[str, Any]]

# ---- Denoising (spec + application) ----
# Keep in sync with `@register_filter` names plus schema-only `"none"`.
_DENOISE_METHODS = (
    "none",        # no-op
    "ema",         # exponential moving average
    "sma",         # simple moving average
    "median",      # rolling median
    "lowpass_fft", # zero-phase FFT low-pass
    "butterworth", # Butterworth IIR filter
    "hp",          # Hodrick-Prescott trend filter
    "savgol",      # Savitzky-Golay smoothing
    "supersmoother",  # Ehlers 2-pole SuperSmoother
    "kama",        # Kaufman adaptive moving average
    "tv",          # total variation denoising
    "kalman",      # 1D Kalman filter smoothing
    "kalman_robust",  # Student-t robust Kalman
    "preaverage",  # Jacod-style pre-averaging
    "hampel",      # Hampel outlier filter
    "bilateral",   # bilateral smoothing
    "wavelet_packet", # wavelet packet denoise
    "ssa",         # singular spectrum analysis
    "l1_trend",    # L1 trend filtering
    "lms",         # adaptive LMS filter
    "rls",         # adaptive RLS filter
    "beta",        # beta-IRLS smoothing
    "vmd",         # variational mode decomposition
    "loess",       # LOESS/LOWESS smoothing
    "stl",         # seasonal-trend decomposition
    "whittaker",   # Whittaker smoothing
    "gaussian",    # Gaussian kernel smoothing
    "wavelet",     # wavelet shrinkage (PyWavelets optional)
    "emd",         # empirical mode decomposition (PyEMD optional)
    "eemd",        # ensemble EMD (PyEMD optional)
    "ceemdan",     # complementary EEMD with adaptive noise (PyEMD optional)    
)

DenoiseMethodLiteral = Literal[_DENOISE_METHODS]  # type: ignore

class DenoiseSpec(TypedDict, total=False):
    method: DenoiseMethodLiteral  # type: ignore
    params: Dict[str, Any]
    columns: Union[str, List[str]]
    when: Literal['pre_ti', 'post_ti']  # type: ignore
    causality: Literal['causal', 'zero_phase']  # type: ignore
    keep_original: bool
    suffix: str


def normalize_denoise_input(value: Any) -> Any:
    """Accept a preset name or a DenoiseSpec object at public request boundaries."""
    if value is None or isinstance(value, dict):
        return value
    if isinstance(value, str):
        normalized = value.strip()
        return {"method": normalized} if normalized else None
    return value


DenoiseSpecInput = Annotated[
    Optional[DenoiseSpec],
    BeforeValidator(
        normalize_denoise_input,
        json_schema_input_type=Optional[Union[str, DenoiseSpec]],
    ),
]

# ---- Simplify (schema for MCP) ----
_SIMPLIFY_MODES = (
    'select',        # pick representative existing rows
    'approximate',   # aggregate between selected rows
    'resample',      # time-bucket aggregation
    'encode',        # compact encodings (envelope, delta)
    'segment',       # swing points (e.g., ZigZag)
    'symbolic',      # SAX symbolic representation
)
_SIMPLIFY_METHODS = (
    'lttb', 'rdp', 'pla', 'apca'
)
SimplifyModeLiteral = Literal[_SIMPLIFY_MODES]  # type: ignore
SimplifyMethodLiteral = Literal[_SIMPLIFY_METHODS]  # type: ignore
EncodeSchemaLiteral = Literal['envelope', 'delta', 'sax']  # type: ignore

class SimplifySpec(TypedDict, total=False):
    # Common
    mode: SimplifyModeLiteral  # type: ignore
    method: SimplifyMethodLiteral  # type: ignore
    points: int
    ratio: float
    # RDP/PLA/APCA specifics
    epsilon: float
    max_error: float
    segments: int
    # Resample
    bucket_seconds: int
    # Encode specifics
    schema: EncodeSchemaLiteral  # 'envelope' | 'delta' (or 'sax' when mode='symbolic')
    bits: int
    as_chars: bool
    alphabet: str
    scale: float
    zero_char: str
    # Segment specifics
    algo: Literal['zigzag']  # type: ignore
    threshold_pct: float
    value_col: str
    # Symbolic specifics
    paa: int
    znorm: bool

# ---- Pivot Point methods (enums) ----
_PIVOT_METHODS = (
    "classic",
    "fibonacci",
    "camarilla",
    "woodie",
    "demark",
)

PivotMethodLiteral = Literal[_PIVOT_METHODS]  # type: ignore

# ---- Fast Forecast methods (enums) ----
#
# Use a conservative static list here. Importing the forecast registry during
# shared schema module import pulls in optional forecast method stacks and their
# heavy dependencies (for example torch), which makes unrelated tools noisy and
# slow at startup.
_FALLBACK_FORECAST_METHODS: Tuple[str, ...] = (
    "naive",
    "seasonal_naive",
    "drift",
    "theta",
    "fourier_ols",
    "ses",
    "holt",
    "holt_winters_add",
    "holt_winters_mul",
    "ets",
    "arima",
    "sarima",
    "mc_gbm",
    "hmm_mc",
    "mlforecast",
    "mlf_rf",
    "mlf_lightgbm",
    "statsforecast",
    "sktime",
    "chronos2",
    "chronos_bolt",
    "timesfm",
    "timesfm3",
    "ensemble",
    "analog",
)

_FORECAST_METHODS: Tuple[str, ...] = _FALLBACK_FORECAST_METHODS

ForecastLibraryLiteral = Literal[
    "native",
    "statsforecast",
    "sktime",
    "mlforecast",
    "pretrained",
]

ForecastMethodLiteral = Literal[_FORECAST_METHODS]  # type: ignore



def shared_defs() -> Dict[str, Any]:
    """Return shared $defs for input schemas (e.g., TimeframeSpec).

    Note: Additional shared enums (SimplifyMode, etc.) are injected by the server.
    """
    return {
        "TimeframeSpec": {
            "type": "string",
            "enum": sorted(TIMEFRAME_MAP.keys()),
            "description": "MT5 timeframe code (e.g. H1/M30/D1)",
        }
    }


def complex_defs() -> Dict[str, Any]:
    """Return complex reusable definitions for nested params.

    These use $ref to shared enums that the server injects (e.g., SimplifyMode).
    """
    return {
        "IndicatorSpec": {
            "type": "object",
            "properties": {
                "name": {"$ref": "#/$defs/IndicatorName"},
                "params": {
                    "oneOf": [
                        {"type": "array", "items": {"type": "number"}},
                        {"type": "object", "additionalProperties": {"type": "number"}},
                    ]
                },
            },
            "required": ["name"],
            "additionalProperties": False,
            "description": "Indicator name plus optional numeric params.",
        },
        "DenoiseSpec": {
            "type": "object",
            "properties": {
                "method": {"$ref": "#/$defs/DenoiseMethod"},
                "params": {"type": "object", "description": "Method-specific overrides", "additionalProperties": True},
                "columns": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ]
                },
                "when": {"$ref": "#/$defs/WhenSpec"},
                "causality": {"$ref": "#/$defs/CausalitySpec"},
                "keep_original": {"type": "boolean"},
                "suffix": {"type": "string"},
            },
            "required": ["method"],
            "additionalProperties": False,
            "description": "Denoise spec: method plus optional columns/params.",
        },
        "SimplifySpec": {
            "type": "object",
            "properties": {
                "mode": {"$ref": "#/$defs/SimplifyMode"},
                "method": {"$ref": "#/$defs/SimplifyMethod"},
                "points": {"type": "integer"},
                "ratio": {"type": "number"},
                "epsilon": {"type": "number"},
                "max_error": {"type": "number"},
                "segments": {"type": "integer"},
                "bucket_seconds": {"type": "integer"},
                "schema": {"oneOf": [
                    {"$ref": "#/$defs/EncodeSchema"},
                    {"$ref": "#/$defs/SymbolicSchema"}
                ]},
                "bits": {"type": "integer"},
                "as_chars": {"type": "boolean"},
                "alphabet": {"type": "string"},
                "scale": {"type": "number"},
                "zero_char": {"type": "string"},
                "algo": {"type": "string", "enum": ["zigzag"]},
                "threshold_pct": {"type": "number"},
                "value_col": {"type": "string"},
                "paa": {"type": "integer"},
                "znorm": {"type": "boolean"},
            },
            "additionalProperties": False,
            "description": "Simplify/segment/encode spec for outputs.",
        },
    }


def _ensure_defs(schema: Dict[str, Any]) -> Dict[str, Any]:
    if "$defs" not in schema or not isinstance(schema.get("$defs"), dict):
        schema["$defs"] = {}
    # Merge shared defs without overwriting existing keys
    defs = schema["$defs"]
    for k, v in shared_defs().items():
        defs.setdefault(k, v)
    return schema





def apply_param_hints(schema: Dict[str, Any]) -> Dict[str, Any]:
    params_obj = (
        schema
        if isinstance(schema.get("properties"), dict)
        else _parameters_obj(schema)
    )
    props = params_obj.get("properties", {}) if isinstance(params_obj, dict) else {}
    for name, prop in list(props.items()):
        if not isinstance(prop, dict):
            continue
        hint = PARAM_HINTS.get(name)
        if hint and not prop.get("description"):
            prop["description"] = hint
    return schema

def _parameters_obj(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Get or create the OpenAI/MCP-style parameters object inside a schema."""
    if not isinstance(schema.get("parameters"), dict):
        schema["parameters"] = {"type": "object", "properties": {}}
    params = schema["parameters"]
    if not isinstance(params.get("properties"), dict):
        params["properties"] = {}
    return params


def apply_timeframe_ref(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Replace simple timeframe property shapes with a $ref to TimeframeSpec.

    Looks for common parameter names and applies a $ref if present.
    """
    _ensure_defs(schema)
    params = _parameters_obj(schema)
    props = params["properties"]
    for key in ("timeframe", "target_timeframe", "source_timeframe"):
        if key in props and isinstance(props.get(key), dict):
            props[key] = {"$ref": "#/$defs/TimeframeSpec"}
    return schema



def _allow_null(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of schema that also accepts null."""
    updated = dict(schema)
    schema_type = updated.get("type")
    if schema_type is None:
        # Avoid explicit {"type": "null"} in anyOf/oneOf constructs as it triggers
        # "Cannot apply filter 'string' to type: NullValue" in some MCP clients (Jinja).
        # For optional parameters, relying on "required": False is sufficient.
        return updated

    if isinstance(schema_type, list):
        if "null" not in schema_type:
            updated["type"] = schema_type + ["null"]
    else:
        if schema_type != "null":
            updated["type"] = [schema_type, "null"]
    return updated


_TYPED_DICT_REFS = {
    "IndicatorSpec": "#/$defs/IndicatorSpec",
    "DenoiseSpec": "#/$defs/DenoiseSpec",
    "SimplifySpec": "#/$defs/SimplifySpec",
}

def _is_typed_dict_type(type_hint: Any) -> bool:
    try:
        if is_typeddict(type_hint):
            return True
    except Exception:
        pass
    annotations = getattr(type_hint, "__annotations__", None)
    return isinstance(annotations, dict) and (
        getattr(type_hint, "__required_keys__", None) is not None
        or getattr(type_hint, "__optional_keys__", None) is not None
    )


def _type_hint_to_schema(type_hint: Any) -> Dict[str, Any]:  # noqa: C901
    """Convert a Python type hint to a minimal JSON Schema fragment."""
    if type_hint is None:
        return {"type": "string"}
    if type_hint is Any:  # allow arbitrary content
        return {}
    origin = get_origin(type_hint)
    if origin is Literal:
        literals = [lit for lit in get_args(type_hint) if lit is not None]
        if not literals:
            return {"type": "string"}
        literal_types = {type(lit) for lit in literals}
        if literal_types == {bool}:
            return {"type": "boolean"}
        if literal_types == {int}:
            return {"type": "integer", "enum": literals}
        if literal_types == {float}:
            return {"type": "number", "enum": literals}
        return {"type": "string", "enum": [str(lit) for lit in literals]}
    if origin in (Union, types.UnionType):
        args = list(get_args(type_hint))
        allow_null = False
        non_null_args = []
        for arg in args:
            if arg is type(None):
                allow_null = True
            else:
                non_null_args.append(arg)
        if not non_null_args:
            return {"type": "null"}
        if len(non_null_args) == 1:
            schema = _type_hint_to_schema(non_null_args[0])
        else:
            schema = {"oneOf": [_type_hint_to_schema(arg) for arg in non_null_args]}
        if allow_null:
            schema = _allow_null(schema)
        return schema
    if origin in (list, List, tuple, Tuple, set, frozenset):
        args = get_args(type_hint)
        item_type = args[0] if args else Any
        item_schema = _type_hint_to_schema(item_type)
        # Ensure items schema defaults to accepting any value if empty
        if not item_schema:
            item_schema = {}
        return {"type": "array", "items": item_schema}
    if origin in (dict, Dict):
        args = get_args(type_hint)
        value_type = args[1] if len(args) > 1 else Any
        value_schema = _type_hint_to_schema(value_type)
        if not value_schema:
            value_schema = {}
        return {"type": "object", "additionalProperties": value_schema or True}
    # Handle direct builtins and aliases
    if type_hint in (str, bytes):
        return {"type": "string"}
    if type_hint is int:
        return {"type": "integer"}
    if type_hint is float:
        return {"type": "number"}
    if type_hint is bool:
        return {"type": "boolean"}
    if type_hint is dict:
        return {"type": "object", "additionalProperties": True}
    if type_hint is list or type_hint is tuple:
        return {"type": "array"}
    ref_name = getattr(type_hint, "__name__", "")
    if ref_name in _TYPED_DICT_REFS:
        return {"$ref": _TYPED_DICT_REFS[ref_name]}
    if _is_typed_dict_type(type_hint) and ref_name:
        ref = _TYPED_DICT_REFS.get(ref_name)
        if ref:
            return {"$ref": ref}
    return {"type": "string"}

def build_minimal_schema(func_info: Dict[str, Any]) -> Dict[str, Any]:
    """Build a minimal parameters schema from a discovered function description.

    - Only includes parameter names and required flags.
    - Applies TimeframeSpec $ref to known timeframe param names.
    """
    schema: Dict[str, Any] = {"parameters": {"type": "object", "properties": {}, "required": []}}
    props = schema["parameters"]["properties"]
    req = schema["parameters"]["required"]
    for p in func_info.get("params", []):
        name = p.get("name")
        if not name:
            continue
        prop_schema = _type_hint_to_schema(p.get("type"))
        if not prop_schema:
            prop_schema = {"type": "string"}
        props[name] = prop_schema
        default_val = p.get("default")
        if default_val is not None:
            if isinstance(default_val, (str, int, float, bool, list, dict)):
                try:
                    props[name]["default"] = default_val
                except Exception:
                    pass
        if p.get("required"):
            req.append(name)
    _ensure_defs(schema)
    apply_timeframe_ref(schema)
    apply_param_hints(schema)
    return schema


def enrich_schema_with_shared_defs(schema: Dict[str, Any], func_info: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure schema has $defs and timeframe refs. If empty, build minimal one."""
    if not isinstance(schema, dict) or not schema:
        schema = build_minimal_schema(func_info)
        return schema
    _ensure_defs(schema)
    apply_timeframe_ref(schema)
    apply_param_hints(schema)
    return schema




def get_shared_enum_lists() -> Dict[str, List[str]]:
    """Return enum lists used to enrich schemas when attaching to tools."""
    enums: Dict[str, List[str]] = {
        "DENOISE_METHODS": list(_DENOISE_METHODS),
        "SIMPLIFY_MODES": list(_SIMPLIFY_MODES),
        "SIMPLIFY_METHODS": list(_SIMPLIFY_METHODS),
        "PIVOT_METHODS": list(_PIVOT_METHODS),
        "FORECAST_METHODS": list(_FORECAST_METHODS),
    }
    if _CATEGORY_CHOICES:
        enums["CATEGORY_CHOICES"] = list(_CATEGORY_CHOICES)
    return enums


def get_function_info(func: Any) -> Dict[str, Any]:
    """Extract minimal parameter info from a function for schema building."""
    # Introspect original function if wrapped
    try:
        target = inspect.unwrap(func)
    except Exception:
        target = func
    sig = get_runtime_signature(target)
    type_hints = get_runtime_annotations(target)

    params = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        annotation = type_hints.get(name, param.annotation)
        if annotation is inspect._empty:
            annotation = None
        params.append({
            "name": name,
            "required": param.default == inspect._empty,  # type: ignore[attr-defined]
            "default": None if param.default == inspect._empty else param.default,  # type: ignore[attr-defined]
            "type": annotation,
        })

    return {
        "name": getattr(target, "__name__", ""),
        "doc": inspect.getdoc(target) or "",
        "params": params,
    }
