from __future__ import annotations

from typing import Any, Dict, Optional

from ...shared.schema import DenoiseSpec
from ..report.utils import (
    adapt_forecast_payload_for_report,
    normalize_report_methods,
    now_utc_iso,
    report_section_enabled,
    resolve_report_context_end,
)
from .basic import _build_context_section, _get_raw_result

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

    report["sections"]["context"] = _build_context_section(
        symbol=symbol,
        timeframe=tf,
        denoise=denoise,
        params=p,
        context_end=context_end,
        notes="Minimal template keeps only candle context plus a direct forecast.",
        default_context_limit=200,
        fetch_result=_get_raw_result,
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
