from __future__ import annotations

from typing import Any, Dict, Optional

from ...shared.schema import DenoiseSpec
from ..report.trend import _TREND_COMPACT_LEGEND, _compute_compact_trend
from ..report.utils import (
    adapt_forecast_payload_for_report,
    attach_candle_freshness_diagnostics,
    normalize_report_methods,
    now_utc_iso,
    parse_table_tail,
    report_section_enabled,
    resolve_report_context_end,
    resolve_report_context_indicators,
)
from .basic import _get_raw_result

_MINIMAL_SKIPPED_SECTIONS = (
    "pivot",
    "contexts_multi",
    "pivot_multi",
    "volatility",
    "backtest",
    "barriers",
    "patterns",
)


def _resolve_minimal_forecast_method(params: Dict[str, Any]) -> str:
    direct_method = str(params.get("method") or "").strip()
    if direct_method:
        return direct_method
    for method_name in normalize_report_methods(params.get("methods")):
        return method_name
    return "theta"


def template_minimal(
    symbol: str,
    horizon: int,
    denoise: Optional[DenoiseSpec],
    params: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    p = dict(params or {})
    tf = str(p.get("timeframe", "H1"))
    start = p.get("start")
    end = p.get("end")
    context_end = resolve_report_context_end(end, tf)
    forecast_method = _resolve_minimal_forecast_method(p)
    forecast_library = str(p.get("library") or "native").strip() or "native"

    report: Dict[str, Any] = {
        "meta": {
            "symbol": symbol,
            "timeframe": tf,
            "horizon": int(horizon),
            "template": "minimal",
            "generated_at": now_utc_iso(),
            "fast_path": True,
            "skipped_sections": list(_MINIMAL_SKIPPED_SECTIONS),
        },
        "sections": {},
    }

    indicators = resolve_report_context_indicators(p)
    from ..data import data_fetch_candles

    ctx = (
        _get_raw_result(
            data_fetch_candles,
            symbol=symbol,
            timeframe=tf,
            limit=int(p.get("context_limit", 200)),
            # Request validation requires an end whenever start is supplied.
            # The context snapshot anchors at that shared report cutoff.
            start=None,
            end=context_end,
            indicators=indicators,  # type: ignore[arg-type]
            denoise=denoise,
            allow_stale=bool(p.get("allow_stale", False)),
        )
        if report_section_enabled(p, "context")
        else {"error": "context section not requested"}
    )

    if "error" in ctx:
        report["sections"]["context"] = attach_candle_freshness_diagnostics(
            {"error": ctx["error"]},
            ctx,
        )
    else:
        context_limit = int(p.get("context_limit", 200))
        context_rows = parse_table_tail(ctx, tail=context_limit)
        tail_n = int(p.get("context_tail", 40))
        tail_rows = context_rows[-tail_n:]
        if not tail_rows:
            if isinstance(ctx, dict) and isinstance(ctx.get("bars"), list):
                context_rows = list(ctx.get("bars"))  # type: ignore[arg-type]
                tail_rows = context_rows[-tail_n:]
            elif isinstance(ctx, dict) and isinstance(ctx.get("data"), list):
                context_rows = list(ctx.get("data"))  # type: ignore[arg-type]
                tail_rows = context_rows[-tail_n:]
            elif isinstance(ctx, list):
                context_rows = ctx
                tail_rows = context_rows[-tail_n:]
            else:
                tail_rows = []

        if not tail_rows:
            report["sections"]["context"] = attach_candle_freshness_diagnostics(
                {"error": "No candle data available for context section."},
                ctx,
            )
        else:
            last = tail_rows[-1] if tail_rows else {}
            compact = _compute_compact_trend(context_rows)
            ctx_obj: Dict[str, Any] = {
                "symbol": symbol,
                "timeframe": tf,
                "last_snapshot": last,
                "notes": "Minimal template keeps only candle context plus a direct forecast.",
            }
            timezone_label = ctx.get("timezone") if isinstance(ctx, dict) else None
            if timezone_label not in (None, "", [], {}):
                ctx_obj["timezone"] = timezone_label
            for key in ("price_precision", "price_point"):
                value = ctx.get(key) if isinstance(ctx, dict) else None
                if value not in (None, "", [], {}):
                    ctx_obj[key] = value
            if compact:
                ctx_obj["trend_compact"] = compact
                ctx_obj["trend_compact_legend"] = dict(_TREND_COMPACT_LEGEND)
            report["sections"]["context"] = attach_candle_freshness_diagnostics(
                ctx_obj,
                ctx,
            )

    from ..forecast import forecast_generate

    forecast_kwargs: Dict[str, Any] = {
        "symbol": symbol,
        "timeframe": tf,
        "library": forecast_library,
        "method": forecast_method,
        "horizon": int(horizon),
        "start": start,
        "end": end,
        "denoise": denoise,
    }
    forecast_params = p.get("forecast_params")
    if isinstance(forecast_params, dict) and forecast_params:
        forecast_kwargs["params"] = forecast_params

    fc = (
        _get_raw_result(forecast_generate, **forecast_kwargs)
        if report_section_enabled(p, "forecast")
        else {"error": "forecast section not requested"}
    )
    if "error" in fc:
        report["sections"]["forecast"] = {
            "error": fc["error"],
            "method": forecast_method,
            "library": forecast_library,
            "selection_mode": "direct",
            "selection_note": "Minimal template skips backtest ranking and barrier optimization.",
        }
        if not report_section_enabled(p, "forecast"):
            report["sections"].pop("forecast", None)
        if not report_section_enabled(p, "context"):
            report["sections"].pop("context", None)
        return report

    forecast_section = {
        "method": forecast_method,
        "library": forecast_library,
        "selection_mode": "direct",
        "selection_note": "Minimal template skips backtest ranking and barrier optimization.",
    }
    forecast_section.update(adapt_forecast_payload_for_report(fc))
    report["sections"]["forecast"] = forecast_section
    if not report_section_enabled(p, "context"):
        report["sections"].pop("context", None)
    return report
