import logging
from typing import Any, Dict, Union

from ...utils.mt5 import ensure_mt5_connection_or_raise
from .._mcp_instance import mcp
from ..error_envelope import build_error_payload
from ..mt5_gateway import create_mt5_gateway, mt5_connection_error
from ..runtime_metadata import run_mt5_logged_operation
from .requests import ReportGenerateRequest
from .use_cases import run_report_generate
from .utils import _get_indicator_value, format_number

logger = logging.getLogger(__name__)


def _normalize_report_error_message(message: Any) -> str:
    text = str(message).strip()
    if not text:
        text = 'Unknown error.'
    return text


def _report_error_payload(message: Any) -> Dict[str, Any]:
    return build_error_payload(
        _normalize_report_error_message(message),
        code="report_generation_error",
        operation="report_generate",
    )

def _report_connection_error() -> Dict[str, Any] | None:
    return mt5_connection_error(
        create_mt5_gateway(ensure_connection_impl=ensure_mt5_connection_or_raise),
    )


def _append_diagnostic_warning(report: Dict[str, Any], message: str) -> None:
    text = str(message or "").strip()
    if not text:
        return
    diagnostics = report.get("diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    warnings_list = diagnostics.get("warnings")
    if not isinstance(warnings_list, list):
        warnings_list = []
    if text not in warnings_list:
        warnings_list.append(text)
    diagnostics["warnings"] = warnings_list
    report["diagnostics"] = diagnostics


def _attach_report_compute_hint(report: Any, request: ReportGenerateRequest) -> Any:
    if not isinstance(report, dict) or report.get("error"):
        return report
    template = str(request.template or "minimal").strip().lower()
    if template == "minimal" or request.detail != "full":
        return report
    diagnostics = report.get("diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    diagnostics.setdefault(
        "compute_intensity",
        "high" if template == "advanced" else "moderate",
    )
    diagnostics.setdefault(
        "compute_hint",
        "Non-minimal templates may run several analysis sub-tools; use template=minimal for the fast path.",
    )
    report["diagnostics"] = diagnostics
    return report


@mcp.tool()
def report_generate(
    request: ReportGenerateRequest,
) -> Union[str, Dict[str, Any]]:
    """Generate fast context+forecast; use template=basic for broader analysis.

    - template: 'minimal' (default fast path: context + direct forecast),
                'basic' (context, daily pivots, confluence, one vol estimator, optional backtest,
                forecast, fast barrier search, recent patterns),
                'advanced' (adds regimes, HAR-RV, conformal unless the forecast already has intervals),
                'scalping' (M5 quote/DOM/session gates and short-horizon barriers),
                'intraday' (H1 plus session, news, and session seasonality),
                'swing' | 'position' (higher-timeframe confluence, volume profile, and news).
    - params: optional dict for template/sub-tool overrides:
              timeframe, methods, context_limit/context_tail, backtest_steps/backtest_spacing,
              barrier_method/search_profile/grid_style/TP-SL grid keys, patterns_mode/patterns_limit,
              extra_timeframes/pivot_timeframes, and advanced regime/conformal keys
              (regime_limit, regime_lookback, cp_threshold, hmm_states, conformal_*).
    - denoise: pass-through to candles, forecast, backtest, volatility, and barriers.
    - max_runtime: cooperative wall-clock budget in seconds. Static section
      estimates are advisory and do not omit work; new sub-tools stop after the
      deadline, and an already-running native/MT5 call is allowed to finish.
    - allow_partial: defaults true so usable sections survive nested failures; set false for strict runs.
    - allow_stale: permit closed-session completed candle context while retaining stale warnings.
    - progress: emit sub-tool start/finish lines to stderr.
    - start/end: start requires end; end alone creates an as-of historical report.
    """
    def _run() -> Union[str, Dict[str, Any]]:
        connection_error = _report_connection_error()
        if connection_error is not None:
            return connection_error
        report = run_report_generate(
            request,
            format_number=format_number,
            get_indicator_value=_get_indicator_value,
            report_error_payload=_report_error_payload,
            append_diagnostic_warning=_append_diagnostic_warning,
        )
        return _attach_report_compute_hint(report, request)

    return run_mt5_logged_operation(
        logger,
        operation="report_generate",
        symbol=request.symbol,
        template=request.template,
        func=_run,
    )
