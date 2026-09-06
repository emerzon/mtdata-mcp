import math
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from ...shared.constants import TIMEFRAME_SECONDS
from ...shared.market_units import forex_pip_size
from ...utils.barriers import get_tick_size as _get_tick_size
from ...utils.mt5 import get_symbol_info_cached
from ...utils.quote import compute_spread_metrics
from ...utils.time import format_datetime_utc
from ..tool_calling import call_tool_sync_structured
from .shared import (
    _get_indicator_value,
    _indicator_key_variants,
    format_number,
)
from .trend import _compute_compact_trend

_REPORT_PROGRESS_CALLBACK: ContextVar[
    Optional[Callable[[str, str], None]]
] = ContextVar("mtdata_report_progress_callback", default=None)
_REPORT_RUNTIME_DEADLINE: ContextVar[Optional[float]] = ContextVar(
    "mtdata_report_runtime_deadline",
    default=None,
)


@contextmanager
def report_execution_scope(
    *,
    progress_callback: Optional[Callable[[str, str], None]] = None,
    deadline: Optional[float] = None,
) -> Iterator[None]:
    """Bind report progress and cooperative deadline state to one request."""
    progress_token = _REPORT_PROGRESS_CALLBACK.set(progress_callback)
    deadline_token = _REPORT_RUNTIME_DEADLINE.set(deadline)
    try:
        yield
    finally:
        _REPORT_RUNTIME_DEADLINE.reset(deadline_token)
        _REPORT_PROGRESS_CALLBACK.reset(progress_token)


def emit_report_progress(operation: str, state: str) -> None:
    callback = _REPORT_PROGRESS_CALLBACK.get()
    if callback is None:
        return
    try:
        callback(str(operation), str(state))
    except Exception:
        pass


def report_runtime_expired() -> bool:
    deadline = _REPORT_RUNTIME_DEADLINE.get()
    return deadline is not None and time.perf_counter() >= deadline


def report_runtime_error(operation: str) -> Dict[str, Any]:
    from ..error_envelope import build_error_payload

    payload = build_error_payload(
        (
            "Report max_runtime budget was exhausted before "
            f"{operation} could start."
        ),
        code="report_runtime_budget_exhausted",
        operation=str(operation),
    )
    payload["runtime_budget_exhausted"] = True
    return payload


def now_utc_iso() -> str:
    return format_datetime_utc(datetime.now(timezone.utc), timespec="minutes")


def resolve_report_context_end(end: Any, timeframe: str) -> Any:
    """Resolve date-only intraday cutoffs to an exact completed-bar boundary."""
    text = str(end or "").strip()
    seconds = TIMEFRAME_SECONDS.get(str(timeframe or "").strip().upper())
    if (
        text
        and len(text) == 10
        and text[4:5] == "-"
        and text[7:8] == "-"
        and seconds is not None
        and seconds < 86_400
    ):
        try:
            datetime.strptime(text, "%Y-%m-%d")
        except ValueError:
            pass
        else:
            return f"{text}T23:59:59.999999Z"
    return end


def _normalize_source_bar_time(value: Any) -> Optional[str]:
    if isinstance(value, bool) or value in (None, ""):
        return None
    if isinstance(value, datetime):
        return format_datetime_utc(value)
    if isinstance(value, (int, float)):
        try:
            return format_datetime_utc(datetime.fromtimestamp(float(value), timezone.utc))
        except (OSError, OverflowError, TypeError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(
            text[:-1] + "+00:00" if text.endswith("Z") else text
        )
    except ValueError:
        try:
            return format_datetime_utc(
                datetime.fromtimestamp(float(text), timezone.utc)
            )
        except (OSError, OverflowError, TypeError, ValueError):
            return None
    return format_datetime_utc(parsed)


_CURRENT_ONLY_OMISSION_REASON = "current_only_section_omitted"


def is_bounded_report_window(start: Any, end: Any) -> bool:
    return start not in (None, "") or end not in (None, "")


def current_only_section_omission(
    section: str,
    *,
    start: Any,
    end: Any,
) -> Dict[str, Any]:
    return {
        "status": "omitted",
        "reason": _CURRENT_ONLY_OMISSION_REASON,
        "section": section,
        "requested_window": {"start": start, "end": end},
        "message": (
            f"{section} was omitted because it cannot currently honor the "
            "report's bounded market window."
        ),
    }


_REPORT_FORECAST_FIELDS = (
    "forecast",
    "forecast_summary",
    "forecast_price",
    "forecast_return",
    "forecast_series",
    "lower_price",
    "upper_price",
    "trend",
    "forecast_vs_last_price",
    "uncertainty",
    "ci_status",
    "ci_alpha",
    "forecast_mode",
    "quantity",
    "quantity_note",
    "timezone",
    "last_observation_time",
    "last_observation_epoch",
    "forecast_start_time",
    "forecast_start_epoch",
    "forecast_anchor",
    "forecast_start_gap_bars",
    "forecast_step_seconds",
    "data_window",
    "freshness",
    "data_stale",
    "last_observation_stale",
    "last_price_stale",
    "market_status",
    "market_status_reason",
    "last_price",
    "last_price_source",
    "path_flat",
    "path_range",
    "volatility_per_bar",
    "volatility_annualized",
    "volatility_horizon",
    "volatility_horizon_annualized",
    "volatility_unit",
    "warnings",
)


def extract_report_forecast_values(payload: Any) -> List[float]:
    """Extract finite forecast values from legacy or canonical forecast payloads."""
    if not isinstance(payload, dict):
        return []

    values: List[float] = []

    def _append(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                _append(item)
            return
        if isinstance(value, dict):
            for key in (
                "value",
                "forecast_price",
                "forecast_return",
                "volatility",
                "volatility_per_bar",
            ):
                if key in value:
                    _append(value.get(key))
                    return
            return
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return
        if math.isfinite(numeric):
            values.append(numeric)

    for key in (
        "forecast_price",
        "forecast_return",
        "forecast_series",
        "forecast",
        "volatility_per_bar",
        "volatility_annualized",
        "volatility_horizon",
        "volatility_horizon_annualized",
    ):
        if key in payload:
            _append(payload.get(key))
        if values:
            return values

    summary = payload.get("forecast_summary")
    if isinstance(summary, dict):
        _append(summary.get("first"))
        _append(summary.get("last"))
    return values


def adapt_forecast_payload_for_report(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Keep canonical forecast decisions and timing while dropping engine noise."""
    out = {
        key: payload.get(key)
        for key in _REPORT_FORECAST_FIELDS
        if payload.get(key) not in (None, "", [], {})
    }
    if not extract_report_forecast_values(payload):
        out["error"] = "Forecast result did not contain any finite forecast values."
    return out


def report_section_enabled(params: Optional[Dict[str, Any]], section: str) -> bool:
    """Return whether a template section belongs to the request execution plan."""
    if not isinstance(params, dict) or "_report_execution_sections" not in params:
        return True
    requested = params.get("_report_execution_sections")
    if not isinstance(requested, (list, tuple, set, frozenset)):
        return True
    section_key = str(section).strip().casefold()
    return any(str(item).strip().casefold() == section_key for item in requested)


def parse_table_tail(data: Any, tail: int = 1) -> List[Dict[str, Any]]:
    """Return the last N rows from a tabular payload (list[dict] or {data|bars: ...})."""
    try:
        def _table_dict_to_rows(table: Any) -> List[Dict[str, Any]]:
            if not isinstance(table, dict):
                return []
            columns = table.get('columns')
            rows = table.get('rows')
            if not isinstance(columns, list) or not isinstance(rows, list):
                return []
            out_rows: List[Dict[str, Any]] = []
            for row in rows:
                if not isinstance(row, list):
                    continue
                out_rows.append({
                    str(col): row[idx] if idx < len(row) else None
                    for idx, col in enumerate(columns)
                })
            return out_rows

        if isinstance(data, dict):
            rows_obj = data.get('data')
            if not isinstance(rows_obj, list):
                rows_obj = data.get('bars')
        else:
            rows_obj = data
        if isinstance(rows_obj, dict):
            rows_obj = _table_dict_to_rows(rows_obj)
        if not isinstance(rows_obj, list):
            return []
        tail_i = int(tail or 0)
        rows_in = [r for r in rows_obj if isinstance(r, dict)]
        if tail_i > 0:
            rows_in = rows_in[-tail_i:]

        def _coerce(v: Any) -> Any:
            if v is None or isinstance(v, (int, float, bool)):
                return v
            if not isinstance(v, str):
                return v
            s = v.strip()
            if s == "":
                return ""
            if s.lower() in ("nan", "inf", "+inf", "-inf"):
                try:
                    return float(s)
                except Exception:
                    return v
            if '.' in s or 'e' in s.lower():
                try:
                    return float(s)
                except Exception:
                    return v
            if s.lstrip('-').isdigit():
                try:
                    return int(s)
                except Exception:
                    return v
            return v

        out: List[Dict[str, Any]] = []
        for row in rows_in:
            out.append({str(k): _coerce(val) for k, val in row.items()})
        return out
    except Exception:
        return []


_PUBLIC_CANDLE_FRESHNESS_KEYS = (
    "data_stale",
    "last_observation_stale",
    "last_price_stale",
    "market_status",
    "market_status_reason",
    "history_policy_ok",
    "freshness_policy_relaxed",
    "stale_warning",
    "usable_for_live_trading",
)


def _public_candle_freshness_fields(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    return {
        key: data[key]
        for key in _PUBLIC_CANDLE_FRESHNESS_KEYS
        if data.get(key) not in (None, "", [], {})
    }


def extract_candle_freshness_diagnostics(data: Any) -> Optional[Dict[str, Any]]:
    try:
        if not isinstance(data, dict):
            return None
        extracted: Optional[Dict[str, Any]] = None
        for container_key in ('meta', 'details'):
            container = data.get(container_key)
            if not isinstance(container, dict):
                continue
            diagnostics = container.get('diagnostics')
            if not isinstance(diagnostics, dict):
                continue
            freshness = diagnostics.get('freshness')
            if isinstance(freshness, dict) and freshness:
                extracted = dict(freshness)
                break
        public = _public_candle_freshness_fields(data)
        if extracted:
            for key, value in public.items():
                extracted.setdefault(key, value)
            return extracted
        return public or None
    except Exception:
        return None


def attach_candle_freshness_diagnostics(payload: Dict[str, Any], data: Any) -> Dict[str, Any]:
    try:
        out = dict(payload) if isinstance(payload, dict) else {}
        freshness = out.get('freshness')
        extracted = (
            dict(freshness)
            if isinstance(freshness, dict) and freshness
            else extract_candle_freshness_diagnostics(data)
        )
        if not extracted:
            public = _public_candle_freshness_fields(data)
            if public:
                extracted = public
        if extracted:
            out['freshness'] = extracted
            if extracted.get('data_stale') is not None:
                out.setdefault('data_stale', extracted['data_stale'])
            market_status = extracted.get('market_status') or extracted.get(
                'market_session_status'
            )
            if market_status not in (None, "", [], {}):
                out.setdefault('market_status', market_status)
            if extracted.get('market_status_reason') not in (None, "", [], {}):
                out.setdefault(
                    'market_status_reason',
                    extracted['market_status_reason'],
                )
        return out
    except Exception:
        return dict(payload) if isinstance(payload, dict) else {}


def pick_best_forecast_method(
    bt: Dict[str, Any],
    rmse_tolerance: float = 0.05,
    min_directional_accuracy: Optional[float] = None,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    try:
        results = bt.get('results') if isinstance(bt, dict) else None
        if not isinstance(results, dict) or not results:
            return None
        min_da: Optional[float] = None
        if min_directional_accuracy is not None:
            try:
                candidate = float(min_directional_accuracy)
            except Exception:
                candidate = float("nan")
            if math.isfinite(candidate):
                min_da = max(0.0, min(1.0, candidate))
        entries: List[Tuple[str, Dict[str, Any], float, Optional[float], int]] = []
        for m, res in results.items():
            if not isinstance(res, dict):
                continue
            if not res.get('success'):
                continue
            if not has_complete_forecast_backtest_coverage(res):
                continue
            try:
                rmse = float(res.get('avg_rmse', float('inf')))
            except Exception:
                rmse = float('inf')
            try:
                da_val = res.get('avg_directional_accuracy')
                da = float(da_val) if da_val is not None else float('nan')
            except Exception:
                da = float('nan')
            ok = int(res.get('successful_tests', 0))
            if not (rmse == rmse):
                continue
            entries.append((m, res, rmse, da if da == da else None, ok))
        if min_da is not None:
            entries = [e for e in entries if e[3] is not None and float(e[3]) >= float(min_da)]
        if not entries:
            return None
        entries.sort(key=lambda x: x[2])
        best_rmse = entries[0][2]
        tol = max(0.0, float(rmse_tolerance))
        limit = best_rmse * (1.0 + tol)
        candidates = [e for e in entries if e[2] <= limit]
        with_dir = [e for e in candidates if e[3] is not None]
        if with_dir:
            with_dir.sort(key=lambda x: (-(x[3] or 0.0), x[2], -x[4]))
            chosen = with_dir[0]
        else:
            entries.sort(key=lambda x: (x[2], -x[4]))
            chosen = entries[0]
        return (chosen[0], chosen[1])
    except Exception:
        return None


def has_complete_forecast_backtest_coverage(result: Any) -> bool:
    """Return whether a method result is safe for cross-method selection.

    Current backtests publish ``status``, ``complete_success``, and anchor-test
    counts.  Older payloads may omit one or more of those fields, so negative
    evidence always wins while a legacy payload with no coverage evidence keeps
    its historical behavior.  There is no public common-anchor score contract
    yet; ad-hoc fields must not make a partial aggregate selectable.
    """
    if not isinstance(result, dict) or result.get("success") is False:
        return False

    status = str(result.get("status") or "").strip().lower()
    if status in {"partial", "failed"}:
        return False

    if "complete_success" in result and result.get("complete_success") is not True:
        return False

    def _finite_count(value: Any) -> Optional[float]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if math.isfinite(numeric) else None

    failed_tests = _finite_count(result.get("failed_tests"))
    if failed_tests is not None and failed_tests != 0.0:
        return False

    successful_tests = _finite_count(result.get("successful_tests"))
    num_tests = _finite_count(result.get("num_tests"))
    if successful_tests is not None and num_tests is not None:
        if (
            successful_tests < 0.0
            or num_tests <= 0.0
            or successful_tests != num_tests
        ):
            return False

    return True


def normalize_report_methods(value: Any) -> List[str]:
    """Normalize documented comma/whitespace method inputs in stable order."""
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, (list, tuple)):
        raw_items = list(value)
    else:
        raw_items = []
    normalized: List[str] = []
    for raw in raw_items:
        for token in str(raw or "").replace(",", " ").split():
            method = token.strip()
            if method and method not in normalized:
                normalized.append(method)
    return normalized


_BARRIER_DECISION_FIELDS = (
    'status',
    'status_reason',
    'recommendation',
    'recommendation_reason',
    'mathematically_viable',
    'viable',
    'tradable',
    'usable_for_live_trading',
    'candidates_evaluated',
    'candidates_viable',
    'candidates_returned',
    'execution_blockers',
    'actionability',
    'actionability_reason',
    'remediation',
)


def _barrier_decision_summary(grid: Any) -> Dict[str, Any]:
    if not isinstance(grid, dict):
        return {}
    return {
        key: grid[key]
        for key in _BARRIER_DECISION_FIELDS
        if key in grid and grid[key] not in (None, [], {})
    }


def _barrier_lineage_summary(grid: Any) -> Dict[str, Any]:
    if not isinstance(grid, dict):
        return {}
    lineage = {
        key: grid[key]
        for key in (
            'data_as_of',
            'data_as_of_epoch',
            'data_stale',
            'data_freshness_seconds',
            'timezone',
            'timeframe',
            'horizon',
            'history_window',
            'history_bars_used',
            'last_observation_close_time',
            'reference_price_time',
            'reference_quote_source',
            'last_price_source',
            'method',
            'simulation_seed',
            'simulation_seed_source',
            'trading_costs',
        )
        if grid.get(key) not in (None, [], {})
    }
    compute_profile = grid.get('compute_profile')
    if isinstance(compute_profile, dict):
        simulation = {
            key: compute_profile[key]
            for key in ('profile', 'n_sims', 'n_seeds', 'paths_evaluated', 'seed', 'seed_source')
            if compute_profile.get(key) is not None
        }
        if simulation:
            lineage['simulation'] = simulation
    return lineage


def summarize_barrier_grid(grid: Dict[str, Any], top_k: int = 3) -> Dict[str, Any]:
    try:
        best = grid.get('best') if isinstance(grid, dict) else None
        top = grid.get('top') if isinstance(grid, dict) else None
        if not top and isinstance(grid.get('results'), list):
            items = grid['results']
            try:
                items = sorted(items, key=lambda x: float(x.get('score', x.get('edge', -1e9))), reverse=True)
            except Exception:
                pass
            top = items[:top_k]
        lineage = _barrier_lineage_summary(grid)
        out = {
            **_barrier_decision_summary(grid),
            **({'lineage': lineage} if lineage else {}),
        }
        direction = grid.get('direction') if isinstance(grid, dict) else None
        if direction:
            out['direction'] = direction
        if isinstance(best, dict):
            best_out = {
                'tp': best.get('tp'),
                'sl': best.get('sl'),
                'objective': best.get('objective') or grid.get('objective'),
                'edge': best.get('edge'),
                'edge_vs_breakeven': best.get('edge_vs_breakeven'),
                'kelly': best.get('kelly'),
                'ev': best.get('ev'),
                'prob_tp_first': best.get('prob_tp_first'),
                'prob_sl_first': best.get('prob_sl_first'),
                'prob_no_hit': best.get('prob_no_hit'),
                'median_time_to_tp': best.get('median_time_to_tp'),
                'tp_price': best.get('tp_price'),
                'sl_price': best.get('sl_price'),
            }
            try:
                ev_val = best_out.get('ev')
                edge_metric = "edge_vs_breakeven"
                edge_val = best_out.get(edge_metric)
                if edge_val is None:
                    edge_metric = "edge"
                    edge_val = best_out.get(edge_metric)
                if ev_val is not None and edge_val is not None:
                    ev_f = float(ev_val)
                    edge_f = float(edge_val)
                    if (ev_f > 0.0 and edge_f < 0.0) or (ev_f < 0.0 and edge_f > 0.0):
                        best_out['ev_edge_conflict'] = True
                        best_out['ev_edge_conflict_reason'] = f"ev and {edge_metric} have opposite signs"
            except Exception:
                pass
            out['best'] = best_out
            if bool(best_out.get('ev_edge_conflict')):
                out['ev_edge_conflict'] = True
                out['ev_edge_conflict_reason'] = best_out.get(
                    'ev_edge_conflict_reason',
                    "ev and edge have opposite signs",
                )
                out['caution'] = (
                    "EV and edge signs conflict for the selected candidate; inspect "
                    "win probability and break-even threshold before trading."
                )
        if isinstance(top, list):
            def _round_metric(value: Any, decimals: int) -> Any:
                try:
                    if value is None:
                        return None
                    num = float(value)
                    if not math.isfinite(num):
                        return str(value)
                    return round(num, decimals)
                except Exception:
                    return value

            def _row_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
                return (
                    _round_metric(row.get('tp'), 4),
                    _round_metric(row.get('sl'), 4),
                    _round_metric(row.get('tp_price'), 6),
                    _round_metric(row.get('sl_price'), 6),
                    _round_metric(row.get('edge'), 4),
                    _round_metric(row.get('edge_vs_breakeven'), 4),
                    _round_metric(row.get('kelly'), 4),
                    _round_metric(row.get('ev'), 4),
                    _round_metric(row.get('prob_tp_first'), 4),
                    _round_metric(row.get('prob_sl_first'), 4),
                    _round_metric(row.get('prob_no_hit'), 4),
                )

            trimmed = []
            seen_keys: set = set()
            for it in top:
                if not isinstance(it, dict):
                    continue
                key = _row_key(it)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                trimmed.append({
                    'tp': it.get('tp'), 'sl': it.get('sl'),
                    'tp_price': it.get('tp_price'), 'sl_price': it.get('sl_price'),
                    'edge': it.get('edge'), 'edge_vs_breakeven': it.get('edge_vs_breakeven'),
                    'kelly': it.get('kelly'), 'ev': it.get('ev'),
                    'prob_tp_first': it.get('prob_tp_first'), 'prob_sl_first': it.get('prob_sl_first'), 'prob_no_hit': it.get('prob_no_hit'),
                })
                if len(trimmed) >= int(top_k):
                    break
            if trimmed:
                out['top'] = trimmed
        for key in ("caution", "ev_edge_conflict", "ev_edge_conflict_reason", "selection_warnings"):
            value = grid.get(key)
            if value in (None, [], {}):
                continue
            out[key] = value
        if out and not isinstance(best, dict) and 'top' not in out:
            out['note'] = (
                'no viable barrier candidates'
                if str(out.get('status') or '').strip().lower() == 'non_viable'
                else 'no grid summary'
            )
        return out or {"note": "no grid summary"}
    except Exception:
        return {"note": "no grid summary"}


def merge_params(base: Optional[Dict[str, Any]], extra: Dict[str, Any], override: bool = False) -> Dict[str, Any]:
    p = dict(base or {})
    for k, v in extra.items():
        if override or k not in p:
            p[k] = v
    return p


def report_market_quote(  # noqa: C901
    symbol: str, timezone: str = 'UTC'
) -> Dict[str, Any]:
    operation = "report_market_quote"
    if report_runtime_expired():
        emit_report_progress(operation, "skipped_runtime_budget")
        return report_runtime_error(operation)
    emit_report_progress(operation, "started")
    try:
        from ..market_depth import (
            market_depth_fetch as _fetch_market_depth,
        )
        from ..market_depth import (
            market_ticker as _market_ticker,
        )

        ticker = call_tool_sync_structured(
            _market_ticker,
            symbol=symbol,
            detail="full",
        )
        if not isinstance(ticker, dict) or ticker.get("success") is not True:
            error = (
                ticker.get("error")
                if isinstance(ticker, dict)
                else "Market ticker returned an unexpected payload."
            )
            return {
                "status": "error",
                "error": str(error or "Live Level 1 quote is unavailable."),
                "error_code": (
                    ticker.get("error_code")
                    if isinstance(ticker, dict)
                    else "report_market_quote_unavailable"
                ),
                "quote_source": "market_ticker",
                "depth_status": "not_fetched",
            }

        bid = ticker.get("bid")
        ask = ticker.get("ask")
        spread = ticker.get("spread")
        if spread is None and bid is not None and ask is not None:
            try:
                spread = float(ask) - float(bid)
            except (TypeError, ValueError):
                spread = None

        depth_status = "unavailable"
        depth_reason = None
        top_buy_vol = None
        top_sell_vol = None
        total_buy_vol = None
        total_sell_vol = None
        dom = call_tool_sync_structured(
            _fetch_market_depth,
            symbol=symbol,
        )
        if isinstance(dom, dict) and dom.get('success'):
            t = dom.get('type')
            data = dom.get('data') or {}
            if t == 'quote_fallback':
                depth_status = "quote_only"
                depth_reason = "Broker returned Level 1 data without an order book."
            elif t == 'full_depth':
                depth_status = "available"
                buys = data.get('buy_orders') or []
                sells = data.get('sell_orders') or []
                if buys:
                    try:
                        top_buy_vol = float(buys[0].get('volume') or 0.0)
                    except Exception:
                        top_buy_vol = None
                if sells:
                    try:
                        top_sell_vol = float(sells[0].get('volume') or 0.0)
                    except Exception:
                        top_sell_vol = None
                try:
                    total_buy_vol = float(sum(float(b.get('volume') or 0.0) for b in buys)) if buys else None
                except Exception:
                    total_buy_vol = None
                try:
                    total_sell_vol = float(sum(float(s.get('volume') or 0.0) for s in sells)) if sells else None
                except Exception:
                    total_sell_vol = None
            else:
                depth_reason = "Market depth returned no recognized DOM payload."
        elif isinstance(dom, dict) and dom.get("error_code") == "feature_disabled":
            depth_status = "disabled"
            depth_reason = str(
                dom.get("why_disabled")
                or "Market depth is disabled; Level 1 quote data remains available."
            )
        elif isinstance(dom, dict):
            depth_reason = str(dom.get("error") or "Market depth is unavailable.")
        info = get_symbol_info_cached(symbol)
        tick_size = _get_tick_size(symbol, symbol_info=info)
        point_size = None
        digits = None
        if info is not None:
            try:
                point_size = float(getattr(info, "point", 0.0) or 0.0)
            except Exception:
                point_size = None
            try:
                digits = int(getattr(info, "digits", 0) or 0)
            except Exception:
                digits = None
        if tick_size is not None:
            try:
                tick_size = float(tick_size)
            except Exception:
                tick_size = None
        if point_size is not None:
            try:
                point_size = float(point_size)
            except Exception:
                point_size = None
            if point_size is not None and point_size <= 0:
                point_size = None
        pip_size = (
            forex_pip_size(
                symbol,
                path=str(getattr(info, "path", "") or ""),
                point=point_size,
                digits=digits,
            )
            if point_size is not None and digits is not None
            else None
        )
        spread_ticks = None
        if tick_size and spread is not None:
            try:
                spread_ticks = float(spread) / float(tick_size) if tick_size > 0 else None
            except Exception:
                spread_ticks = None
        spread_points = ticker.get("spread_points")
        if spread_points is None and point_size and spread is not None:
            try:
                spread_points = float(spread) / float(point_size) if point_size > 0 else None
            except Exception:
                spread_points = None
        spread_pips = ticker.get("spread_pips")
        if spread_pips is None and pip_size and spread is not None:
            try:
                spread_pips = float(spread) / float(pip_size) if pip_size > 0 else None
            except Exception:
                spread_pips = None
        snapshot = {
            'bid': bid,
            'ask': ask,
            'spread': spread,
            'tick_size': tick_size,
            'point_size': point_size,
            'pip_size': pip_size,
            'spread_ticks': spread_ticks,
            'spread_points': spread_points,
            'spread_pips': spread_pips,
            'dom_top_buy_vol': top_buy_vol,
            'dom_top_sell_vol': top_sell_vol,
            'dom_total_buy_vol': total_buy_vol,
            'dom_total_sell_vol': total_sell_vol,
            'quote_source': ticker.get('quote_source') or 'market_ticker',
            'quote_time': ticker.get('time'),
            'quote_time_epoch': ticker.get('time_epoch'),
            'freshness_state': ticker.get('freshness_state'),
            'data_age_seconds': ticker.get('data_age_seconds'),
            'spread_valid': ticker.get('spread_valid'),
            'spread_quality': ticker.get('spread_quality'),
            'usable_for_live_trading': ticker.get('usable_for_live_trading'),
            'usable_for_live_trading_basis': ticker.get(
                'usable_for_live_trading_basis'
            ),
            'depth_status': depth_status,
        }
        if depth_reason:
            snapshot['depth_reason'] = depth_reason

        if not isinstance(snapshot.get('spread_valid'), bool):
            spread_metrics = compute_spread_metrics(
                bid,
                ask,
                point=point_size,
                tick_size=tick_size,
            )
            snapshot['spread_valid'] = spread_metrics['spread_valid']
            snapshot['spread_quality'] = spread_metrics['spread_quality']
        valid_pair = snapshot.get('spread_valid') is True
        if not valid_pair:
            snapshot.update(
                {
                    'status': 'error',
                    'error': 'Market ticker did not provide a valid two-sided Level 1 quote.',
                    'error_code': 'report_market_quote_unavailable',
                }
            )
        elif ticker.get('usable_for_live_trading') is False:
            snapshot['quote_health'] = {
                'status': 'degraded',
                'error': str(
                    ticker.get('warning')
                    or 'The Level 1 quote is not currently execution-ready.'
                ),
            }
        return snapshot
    except Exception as e:
        return {'error': str(e)}
    finally:
        emit_report_progress(operation, "finished")


def apply_market_gates(section: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(section, dict) or section.get('error'):
        return {
            'status': 'error',
            'execution_ready': False,
            'error': str(
                section.get('error')
                if isinstance(section, dict)
                else 'Market quote is unavailable.'
            ),
            'error_code': 'report_execution_gate_quote_unavailable',
        }

    gate: Dict[str, Any] = {
        'quote_usable_for_live_trading': section.get('usable_for_live_trading'),
        'quote_usability_basis': section.get('usable_for_live_trading_basis'),
        'spread_valid': section.get('spread_valid'),
    }
    if gate['spread_valid'] is None:
        spread_metrics = compute_spread_metrics(
            section.get('bid'),
            section.get('ask'),
            point=section.get('point_size'),
            tick_size=section.get('tick_size'),
        )
        gate['spread_valid'] = spread_metrics['spread_valid']
    if gate['quote_usable_for_live_trading'] is None:
        gate['quote_usable_for_live_trading'] = bool(gate['spread_valid'])

    spread_limit_configured = False
    spread_ok: Optional[bool] = None
    try:
        max_ticks = params.get('spread_max_ticks')
        max_pips = params.get('spread_max_pips')
        if max_ticks is not None and isinstance(section, dict):
            spread_limit_configured = True
            sp = section.get('spread_ticks')
            if sp is not None:
                spread_ok = bool(float(sp) <= float(max_ticks))
                gate['spread_ticks'] = float(sp)
                gate['spread_max_ticks'] = float(max_ticks)
        elif max_pips is not None and isinstance(section, dict):
            spread_limit_configured = True
            sp = section.get('spread_pips')
            if sp is not None:
                spread_ok = bool(float(sp) <= float(max_pips))
                gate['spread_pips'] = float(sp)
                gate['spread_max_pips'] = float(max_pips)
    except (TypeError, ValueError):
        return {
            **gate,
            'status': 'error',
            'execution_ready': False,
            'error': 'Configured spread threshold must be a finite number.',
            'error_code': 'report_execution_gate_invalid_threshold',
        }

    gate['spread_limit_status'] = (
        'not_configured'
        if not spread_limit_configured
        else 'unavailable'
        if spread_ok is None
        else 'pass'
        if spread_ok
        else 'fail'
    )
    if spread_ok is not None:
        gate['spread_ok'] = spread_ok
    execution_ready = bool(
        gate['quote_usable_for_live_trading'] is True
        and gate['spread_valid'] is True
        and spread_ok is not False
        and (not spread_limit_configured or spread_ok is not None)
    )
    gate['execution_ready'] = execution_ready
    gate['status'] = 'pass' if execution_ready else 'fail'
    if not spread_limit_configured:
        gate['note'] = (
            'No maximum spread threshold is configured; this gate verifies only '
            'quote execution readiness and a positive spread.'
        )
    elif spread_ok is None:
        gate['reason'] = 'Configured spread threshold could not be evaluated.'
    return gate


DEFAULT_REPORT_CONTEXT_INDICATORS = "ema(20),ema(50),ema(200),rsi(14),macd(12,26,9)"


def resolve_report_context_indicators(
    params: Optional[Dict[str, Any]],
    *,
    default: str = DEFAULT_REPORT_CONTEXT_INDICATORS,
) -> str:
    if isinstance(params, dict):
        indicators = str(params.get('context_indicators') or '').strip()
        if indicators:
            return indicators
    return default


def _report_context_cache_key(indicators: str) -> str:
    return ''.join(str(indicators or '').casefold().split())


def context_for_tf(
    symbol: str,
    timeframe: str,
    denoise: Optional[Dict[str, Any]],
    limit: int = 200,
    tail: int = 30,
    *,
    indicators: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    allow_stale: bool = False,
    _fetch_cache: Optional[Dict[Tuple[str, ...], Optional[Dict[str, Any]]]] = None,
) -> Optional[Dict[str, Any]]:
    indicator_spec = str(indicators or '').strip() or DEFAULT_REPORT_CONTEXT_INDICATORS
    cache_key = (
        symbol.upper(),
        timeframe.upper(),
        _report_context_cache_key(indicator_spec),
        str(start or ""),
        str(end or ""),
        str(bool(allow_stale)),
    )
    if _fetch_cache is not None and cache_key in _fetch_cache:
        return _fetch_cache[cache_key]
    operation = f"data_fetch_candles[{str(timeframe).upper()}]"
    if report_runtime_expired():
        emit_report_progress(operation, "skipped_runtime_budget")
        return report_runtime_error(operation)
    emit_report_progress(operation, "started")
    try:
        from ..data import data_fetch_candles as _fetch_candles
        res = call_tool_sync_structured(
            _fetch_candles,
            symbol=symbol,
            timeframe=timeframe,
            limit=int(limit),
            start=start,
            end=end,
            indicators=indicator_spec,
            denoise=denoise,
            allow_stale=bool(allow_stale),
        )

        if not isinstance(res, dict):
            if _fetch_cache is not None:
                _fetch_cache[cache_key] = None
            return None
        if res.get('error'):
            error_out = attach_candle_freshness_diagnostics(
                {'error': res.get('error')},
                res,
            )
            cached_error = error_out if isinstance(error_out.get('freshness'), dict) else None
            if _fetch_cache is not None:
                _fetch_cache[cache_key] = cached_error
            return cached_error
        freshness = extract_candle_freshness_diagnostics(res)
        all_rows = parse_table_tail(res, tail=int(limit))
        rows = all_rows[-int(tail):]

        if not rows:
            empty_out = {'freshness': freshness} if freshness else None
            if _fetch_cache is not None:
                _fetch_cache[cache_key] = empty_out
            return empty_out
        last = rows[-1]
        out = {
            'close': last.get('close'),
            'EMA_20': _get_indicator_value(last, 'EMA_20'),
            'EMA_50': _get_indicator_value(last, 'EMA_50'),
            'RSI_14': _get_indicator_value(last, 'RSI_14'),
            'macd': _get_indicator_value(last, 'MACD_12_26_9'),
        }
        source_bar_time = _normalize_source_bar_time(last.get('time'))
        if source_bar_time is not None:
            out['source_bar_time'] = source_bar_time
            out['source_bar_timezone'] = 'UTC'
            out['source_bar_state'] = 'completed'

        # Compute trend compact data for MTF matrix
        try:
            compact = _compute_compact_trend(all_rows)
            if compact:
                out['trend_compact'] = compact
        except Exception:
            # If trend compact calculation fails, continue without it
            pass

        # Add individual indicator values for MTF matrix
        if rows:
            last_row = rows[-1]
            out['rsi'] = _get_indicator_value(last_row, 'RSI_14')
            out['macd_signal'] = _get_indicator_value(last_row, 'MACDs_12_26_9')
            out['ema20'] = _get_indicator_value(last_row, 'EMA_20')
            out['ema50'] = _get_indicator_value(last_row, 'EMA_50')
            out['ema200'] = _get_indicator_value(last_row, 'EMA_200')
            out['price'] = last_row.get('close')
        if freshness or source_bar_time is not None:
            freshness_out = dict(freshness or {})
            freshness_state = res.get('freshness_state')
            if freshness_state in (None, ''):
                within_policy = freshness_out.get('last_bar_within_policy_window')
                freshness_state = (
                    'fresh'
                    if within_policy is True
                    else 'stale'
                    if within_policy is False
                    else 'not_evaluated'
                )
            freshness_out.setdefault('state', freshness_state)
            if source_bar_time is not None:
                freshness_out.setdefault('source_bar_time', source_bar_time)
                freshness_out.setdefault('timezone', 'UTC')
            out['freshness'] = freshness_out

        if _fetch_cache is not None:
            _fetch_cache[cache_key] = out
        return out
    except Exception:
        if _fetch_cache is not None:
            _fetch_cache[cache_key] = None
        return None
    finally:
        emit_report_progress(operation, "finished")


def _extract_base_timeframe(report: Dict[str, Any]) -> Optional[str]:
    """Try to infer the base timeframe from report metadata or context."""
    base_tf = None
    try:
        meta = report.get('meta') if isinstance(report, dict) else None
        if isinstance(meta, dict) and meta.get('timeframe'):
            base_tf = str(meta.get('timeframe')).upper()
    except Exception:
        base_tf = None
    if base_tf is None:
        try:
            sections = report.get('sections') if isinstance(report, dict) else None
            context = sections.get('context') if isinstance(sections, dict) else None
            if isinstance(context, dict) and context.get('timeframe'):
                base_tf = str(context.get('timeframe')).upper()
        except Exception:
            base_tf = None
    return base_tf


def _collect_timeframe_section_entries(section: Any) -> Dict[str, Any]:
    entries: Dict[str, Any] = {}
    if not isinstance(section, dict):
        return entries
    for key, value in section.items():
        tf_key = str(key).upper()
        if not tf_key or tf_key.startswith("__"):
            continue
        entries[tf_key] = value
    return entries


def attach_multi_timeframes(  # noqa: C901
    report: Dict[str, Any],
    symbol: str,
    denoise: Optional[Dict[str, Any]],
    extra_timeframes: List[str],
    pivot_timeframes: Optional[List[str]] = None,
    *,
    context_indicators: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    allow_stale: bool = False,
    _fetch_cache: Optional[Dict[Tuple[str, ...], Optional[Dict[str, Any]]]] = None,
) -> None:
    sections = report.setdefault('sections', {})
    contexts: Dict[str, Any] = {}
    trend_mtf: Dict[str, Any] = {}
    base_tf = _extract_base_timeframe(report)
    existing_contexts = _collect_timeframe_section_entries(sections.get('contexts_multi'))
    context_section = sections.get('context')
    existing_trend_mtf = (
        _collect_timeframe_section_entries(context_section.get('trend_mtf'))
        if isinstance(context_section, dict)
        else {}
    )

    for tf in extra_timeframes or []:
        tf_str = str(tf).upper()
        if base_tf and tf_str == base_tf:
            continue
        existing_context = existing_contexts.get(tf_str)
        existing_trend = existing_trend_mtf.get(tf_str)
        if existing_context is not None:
            snap = existing_context
            if isinstance(existing_trend, dict):
                trend_mtf[str(tf)] = existing_trend.copy()
        else:
            snap = context_for_tf(
                symbol,
                tf,
                denoise,
                limit=200,
                tail=30,
                indicators=context_indicators,
                # Multi-timeframe entries are snapshots at the shared cutoff,
                # not range calculations. A start plus a fixed limit selects
                # the beginning of a long range and silently makes the snapshot stale.
                start=None,
                end=end,
                allow_stale=allow_stale,
                _fetch_cache=_fetch_cache,
            )
        if snap:
            snap_for_contexts = snap
            if isinstance(snap, dict):
                # Keep contexts_multi focused on indicator/price snapshots; trend_compact
                # is represented in context.trend_mtf to avoid duplicating the same blob.
                snap_for_contexts = dict(snap)
                snap_for_contexts.pop('trend_compact', None)
                snap_for_contexts.pop('trend_compact_legend', None)
                snap_for_contexts.pop('trend_compact_explained', None)
            if not isinstance(snap_for_contexts, dict) or any(v is not None for v in snap_for_contexts.values()):
                contexts[str(tf)] = snap_for_contexts

            # Extract trend compact data for MTF matrix
            if existing_context is None and isinstance(snap, dict):
                trend_compact = snap.get('trend_compact')
                if isinstance(trend_compact, dict):
                    trend_mtf[str(tf)] = trend_compact.copy()

    if contexts:
        sections['contexts_multi'] = contexts

    # Attach compact trend info for the main context section
    if trend_mtf or contexts:
        if 'context' in sections and isinstance(sections['context'], dict):
            sections['context']['trend_mtf'] = trend_mtf
        else:
            sections['context'] = {'trend_mtf': trend_mtf}

    base_pivot_tf = None
    try:
        base_pivot = report.setdefault('sections', {}).get('pivot')
        if isinstance(base_pivot, dict) and base_pivot.get('timeframe'):
            base_pivot_tf = str(base_pivot.get('timeframe')).upper()
    except Exception:
        base_pivot_tf = None

    if pivot_timeframes:
        filtered_tfs: List[str] = []
        for tfp in pivot_timeframes:
            tfp_str = str(tfp).upper()
            if base_pivot_tf and tfp_str == base_pivot_tf:
                continue
            filtered_tfs.append(str(tfp))
        pivs: Dict[str, Any] = {}
        pivot_errors: Dict[str, Dict[str, str]] = {}
        existing_pivots = _collect_timeframe_section_entries(sections.get('pivot_multi'))
        if filtered_tfs:
            from ..pivot import pivot_compute_points as _compute_pivot_points
            for tfp in filtered_tfs:
                try:
                    tfp_key = str(tfp).upper()
                    existing_pivot = existing_pivots.get(tfp_key)
                    if isinstance(existing_pivot, dict):
                        pivs[str(tfp)] = dict(existing_pivot)
                        continue
                    operation = f"pivot_compute_points[{tfp_key}]"
                    if report_runtime_expired():
                        emit_report_progress(operation, "skipped_runtime_budget")
                        pivot_errors[str(tfp)] = report_runtime_error(operation)
                        continue
                    emit_report_progress(operation, "started")
                    res = call_tool_sync_structured(
                        _compute_pivot_points,
                        symbol=symbol,
                        timeframe=tfp,
                        end=end,
                    )
                    emit_report_progress(operation, "finished")
                    if isinstance(res, dict) and not res.get('error'):
                        period = res.get('period')
                        source_bar_time = _normalize_source_bar_time(
                            period.get('end') if isinstance(period, dict) else period
                        )
                        pivot_out = {
                            'levels': res.get('levels'),
                            'methods': res.get('methods'),
                            'period': period,
                            'timeframe': tfp,
                            'calculation_basis': res.get('calculation_basis'),
                            'timezone': res.get('timezone'),
                        }
                        for key in ('historical_cutoff', 'analysis_as_of'):
                            if res.get(key) is not None:
                                pivot_out[key] = res[key]
                        if source_bar_time is not None:
                            pivot_out['source_bar_time'] = source_bar_time
                            pivot_out['source_bar_timezone'] = 'UTC'
                            pivot_out['source_bar_state'] = 'completed'
                        pivs[str(tfp)] = pivot_out
                    else:
                        pivot_errors[str(tfp)] = {
                            'error': str(
                                res.get('error')
                                if isinstance(res, dict)
                                else 'pivot tool returned no structured payload'
                            )
                        }
                except Exception as exc:
                    pivot_errors[str(tfp)] = {'error': str(exc)}
        if pivs:
            if base_pivot_tf:
                pivs['__base_timeframe__'] = base_pivot_tf
            if pivot_errors:
                pivs['timeframe_errors'] = pivot_errors
            sections['pivot_multi'] = pivs
        elif pivot_errors:
            sections['pivot_multi'] = {
                'status': 'error',
                'error': 'All requested pivot timeframes failed.',
                'timeframe_errors': pivot_errors,
            }
        else:
            sections['pivot_multi'] = {
                'status': 'omitted',
                'reason': 'No requested pivot timeframe differed from the base pivot timeframe.',
            }


def attach_report_timeframes(
    report: Dict[str, Any],
    symbol: str,
    denoise: Optional[Dict[str, Any]],
    params: Optional[Dict[str, Any]],
    *,
    default_extra: List[str],
    default_pivots: Optional[List[str]] = None,
    _fetch_cache: Optional[Dict[Tuple[str, ...], Optional[Dict[str, Any]]]] = None,
) -> None:
    context_enabled = report_section_enabled(params, 'contexts_multi')
    pivot_enabled = report_section_enabled(params, 'pivot_multi')
    if not context_enabled and not pivot_enabled:
        return
    extra = ((params or {}).get('extra_timeframes') or default_extra) if context_enabled else []
    pivots = ((params or {}).get('pivot_timeframes') or default_pivots) if pivot_enabled else []
    end = (params or {}).get('end')
    base_timeframe = _extract_base_timeframe(report) or str(
        (params or {}).get('timeframe') or 'H1'
    )
    context_end = resolve_report_context_end(end, base_timeframe)
    attach_multi_timeframes(
        report,
        symbol,
        denoise,
        extra_timeframes=extra,
        pivot_timeframes=pivots,
        context_indicators=resolve_report_context_indicators(params),
        start=None,
        end=context_end,
        allow_stale=bool((params or {}).get('allow_stale', False)),
        _fetch_cache=_fetch_cache,
    )


def attach_market_and_timeframes(
    report: Dict[str, Any],
    symbol: str,
    denoise: Optional[Dict[str, Any]],
    params: Optional[Dict[str, Any]],
    *,
    default_extra: List[str],
    default_pivots: Optional[List[str]] = None,
    snapshot: Optional[Dict[str, Any]] = None,
    _fetch_cache: Optional[Dict[Tuple[str, ...], Optional[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    market_enabled = report_section_enabled(params, 'market')
    gates_enabled = report_section_enabled(params, 'execution_gates')
    start = (params or {}).get('start')
    end = (params or {}).get('end')
    snap: Dict[str, Any] = {}
    if market_enabled or gates_enabled:
        sections = report.setdefault('sections', {})
        if is_bounded_report_window(start, end):
            if market_enabled:
                sections['market'] = current_only_section_omission(
                    'market', start=start, end=end
                )
            if gates_enabled:
                sections['execution_gates'] = current_only_section_omission(
                    'execution_gates', start=start, end=end
                )
        else:
            snap = snapshot if snapshot is not None else report_market_quote(symbol)
            sections['market'] = snap
            gates = apply_market_gates(
                snap if isinstance(snap, dict) else {}, params or {}
            )
            if gates:
                sections['execution_gates'] = gates
    attach_report_timeframes(
        report,
        symbol,
        denoise,
        params,
        default_extra=default_extra,
        default_pivots=default_pivots,
        _fetch_cache=_fetch_cache,
    )
    return snap


