"""Tests for core/report.py — report_generate tool.

Covers lines 45-245 by mocking template functions and external data fetching.
"""
import logging
import warnings
from contextlib import redirect_stderr
from io import StringIO
from typing import Any, Dict, List, get_type_hints
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from mtdata.utils.mt5 import MT5ConnectionError


def test_barrier_best_summary_keeps_text_and_structured_fields_aligned():
    from mtdata.core.report.use_cases import _build_barrier_best_summary

    details, entry = _build_barrier_best_summary(
        {
            "tp": 1.234,
            "sl": 0.456,
            "ev": 0.25,
            "edge": 0.1,
            "edge_vs_breakeven": -0.2,
        },
        direction="long",
        include_direction_field=True,
        format_number=str,
    )

    assert details[:3] == ["dir=long", "tp=1.23%", "sl=0.46%"]
    assert entry["direction"] == "long"
    assert entry["tp_pct"] == 1.23
    assert entry["sl_pct"] == 0.46
    assert entry["probability_edge"] == 0.1
    assert entry["edge_vs_breakeven"] == -0.2
    assert entry["ev_edge_conflict"] is True
    assert "ev_edge_conflict=true" in details


def test_barrier_best_summary_keeps_nontradable_decision_fields():
    from mtdata.core.report.use_cases import _build_barrier_best_summary

    details, entry = _build_barrier_best_summary(
        {"tp": 0.4, "sl": 0.2, "ev": 0.01},
        decision={
            "status": "ok",
            "recommendation": "avoid",
            "mathematically_viable": True,
            "tradable": False,
            "usable_for_live_trading": False,
            "execution_blockers": ["trading_costs_incomplete"],
        },
        direction="short",
        format_number=str,
    )

    assert entry["tradable"] is False
    assert entry["usable_for_live_trading"] is False
    assert entry["execution_blockers"] == ["trading_costs_incomplete"]
    assert "tradable=False" in details


def test_sections_status_preserves_intentional_omissions():
    from mtdata.core.report.use_cases import _build_sections_status

    status = _build_sections_status(
        {
            "context": {"close": 1.2},
            "pivot": {
                "status": "omitted",
                "reason": "current_only_section_omitted",
            },
        }
    )

    assert status["sections"] == {"context": "ok", "pivot": "omitted"}
    assert status["summary"] == {
        "ok": 1,
        "partial": 0,
        "error": 0,
        "omitted": 1,
        "total": 2,
    }
    assert status["details"]["pivot"]["reason"] == "current_only_section_omitted"


def test_sections_status_marks_scheduled_missing_sections_as_errors():
    from mtdata.core.report.use_cases import _build_sections_status

    status = _build_sections_status(
        {"context": {"close": 1.2}},
        expected_sections=["context", "forecast"],
    )

    assert status["sections"] == {"context": "ok", "forecast": "error"}
    assert status["details"]["forecast"]["reason"] == (
        "scheduled section returned no payload"
    )
    assert status["summary"]["error"] == 1


@pytest.mark.parametrize(
    "payload",
    [
        {
            "error": "upstream failed",
            "error_code": "context_unavailable",
            "symbol": "EURUSD",
        },
        {
            "error": "All volatility estimators failed.",
            "errors": [{"method": "ewma", "error": "history unavailable"}],
            "hint": "Retry with a longer history window.",
        },
        {
            "error": "All pattern modes failed.",
            "modes": ["candlestick", "classic"],
            "errors": [{"mode": "classic", "error": "history unavailable"}],
        },
    ],
)
def test_sections_status_does_not_count_error_metadata_as_usable_data(payload):
    from mtdata.core.report.use_cases import _build_sections_status

    status = _build_sections_status({"context": payload})

    assert status["sections"]["context"] == "error"
    assert status["summary"]["error"] == 1
    assert status["summary"]["partial"] == 0


def test_sections_status_keeps_real_data_with_nested_error_partial():
    from mtdata.core.report.use_cases import _build_sections_status

    status = _build_sections_status(
        {
            "volatility": {
                "volatility_annualized": 0.18,
                "error": "A secondary estimator failed.",
                "errors": [{"method": "garch", "error": "dependency unavailable"}],
            }
        }
    )

    assert status["sections"]["volatility"] == "partial"
    assert status["summary"]["partial"] == 1


def test_sections_status_rejects_conformal_section_without_finite_bounds():
    from mtdata.core.report.use_cases import _build_sections_status

    status = _build_sections_status(
        {
            "forecast_conformal": {
                "method": "theta",
                "lower_price": None,
                "upper_price": None,
            }
        }
    )

    assert status["sections"]["forecast_conformal"] == "error"
    errors = status["details"]["forecast_conformal"]["errors"]
    assert "no complete finite interval" in errors[0]["message"]


def test_sections_status_marks_all_non_viable_barriers_partial():
    from mtdata.core.report.use_cases import _build_sections_status

    non_viable = {
        "status": "non_viable",
        "recommendation": "avoid",
        "mathematically_viable": False,
        "candidates_evaluated": 0,
        "candidates_viable": 0,
        "candidates_returned": 0,
        "execution_blockers": ["optimizer_non_viable"],
    }

    status = _build_sections_status(
        {"barriers": {"long": dict(non_viable), "short": dict(non_viable)}}
    )

    assert status["sections"]["barriers"] == "partial"
    assert status["summary"]["partial"] == 1
    assert status["details"]["barriers"]["reason_code"] == (
        "barrier_optimizer_non_viable"
    )
    assert status["details"]["barriers"]["recommendation"] == "avoid"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def _get_report_generate():
    from mtdata.core.report import report_generate
    raw = _unwrap(report_generate)

    def _call(symbol, **kwargs):
        from mtdata.core import report as report_mod
        from mtdata.core.report.requests import ReportGenerateRequest

        with patch.object(report_mod, "ensure_mt5_connection_or_raise", return_value=None):
            if kwargs.get("format") == "toon":
                kwargs.pop("format")
            kwargs.setdefault("detail", "full")
            return raw(request=ReportGenerateRequest(symbol=symbol, **kwargs))

    return _call


def _make_report(sections=None, error=None):
    """Build a minimal report dict."""
    rep: Dict[str, Any] = {}
    if error:
        rep["error"] = error
    if sections is not None:
        rep["sections"] = sections
    return rep


def test_report_generate_request_rejects_removed_output_field():
    from mtdata.core.report.requests import ReportGenerateRequest

    with pytest.raises(ValidationError, match="output was removed; use json"):
        ReportGenerateRequest(symbol="EURUSD", output="json")


def test_report_generate_request_rejects_removed_structured_format_alias():
    from mtdata.core.report.requests import ReportGenerateRequest

    with pytest.raises(ValidationError, match="format was removed; use json"):
        ReportGenerateRequest(symbol="EURUSD", format="structured")


def test_report_generate_request_defaults_to_compact_detail():
    from mtdata.core.report.requests import ReportGenerateRequest

    request = ReportGenerateRequest(symbol="EURUSD")

    assert request.detail == "compact"
    assert request.template == "minimal"
    assert request.max_runtime is None
    assert request.allow_partial is True
    assert request.allow_stale is False
    assert request.progress is False


def test_report_generate_request_validates_runtime_budget():
    from mtdata.core.report.requests import ReportGenerateRequest

    assert ReportGenerateRequest(symbol="EURUSD", max_runtime=10).max_runtime == 10
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        ReportGenerateRequest(symbol="EURUSD", max_runtime=0.5)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"start": "banana"}, "Could not parse historical datetime"),
        (
            {"start": "2026-08-15", "end": "2026-08-10"},
            "start must be before or equal to end",
        ),
        ({"horizon": 501}, "less than or equal to 500"),
        (
            {"start": "2026-08-01"},
            "end is required when start is supplied",
        ),
    ],
)
def test_report_generate_request_rejects_invalid_shared_inputs(kwargs, message):
    from mtdata.core.report.requests import ReportGenerateRequest

    with pytest.raises(ValidationError, match=message):
        ReportGenerateRequest(symbol="EURUSD", **kwargs)


def test_report_generate_request_accepts_consistent_historical_window():
    from mtdata.core.report.requests import ReportGenerateRequest

    request = ReportGenerateRequest(
        symbol="EURUSD",
        start="2026-08-01",
        end="2026-08-10",
        horizon=500,
    )

    assert request.start == "2026-08-01"
    assert request.end == "2026-08-10"


def test_minimal_report_rejects_multiple_forecast_methods():
    from mtdata.core.report.requests import ReportGenerateRequest

    with pytest.raises(ValidationError, match="accepts one forecast method"):
        ReportGenerateRequest(
            symbol="EURUSD", template="minimal", methods="theta,drift"
        )

    request = ReportGenerateRequest(
        symbol="EURUSD", template="basic", methods="theta,drift"
    )
    assert request.methods == "theta,drift"


def test_report_generate_request_rejects_removed_summary_only_field():
    from mtdata.core.report.requests import ReportGenerateRequest

    with pytest.raises(ValidationError, match="summary_only was removed; use detail='summary'"):
        ReportGenerateRequest(symbol="EURUSD", summary_only=True)


def test_report_section_control_type_hints_resolve() -> None:
    from mtdata.core.report.use_cases import (
        _apply_report_section_controls,
        _split_report_section_names,
    )

    assert get_type_hints(_split_report_section_names)
    assert get_type_hints(_apply_report_section_controls)


@pytest.mark.parametrize(
    "template",
    ["minimal", "basic", "advanced", "scalping", "intraday", "swing", "position"],
)
def test_report_section_plan_limits_every_template_to_context(template: str) -> None:
    from mtdata.core.report.use_cases import _resolve_report_section_plan

    plan = _resolve_report_section_plan(
        template,
        include_sections=["context", "forecast"],
        max_sections=1,
    )

    assert plan["selected"] == ["context"]
    assert plan["execution"] == ["context"]
    assert plan["requested"] == ["context", "forecast"]
    assert plan["capped"] == ["forecast"]
    assert plan["requested_execution"] == ["context", "forecast"]


def test_report_section_plan_only_runs_dependencies_available_to_template() -> None:
    from mtdata.core.report.use_cases import _resolve_report_section_plan

    minimal = _resolve_report_section_plan(
        "minimal",
        include_sections=["forecast"],
    )
    basic = _resolve_report_section_plan(
        "basic",
        include_sections=["forecast"],
    )

    assert minimal["execution"] == ["forecast"]
    assert basic["execution"] == ["forecast"]
    assert basic["selected_runtime_estimate_seconds"] == 5.0
    assert basic["required_dependencies"] == {}


def test_report_section_plan_runtime_estimates_are_advisory() -> None:
    from mtdata.core.report.use_cases import _resolve_report_section_plan

    insufficient = _resolve_report_section_plan(
        "basic",
        include_sections=["context", "forecast"],
        max_runtime=8.0,
    )
    sufficient = _resolve_report_section_plan(
        "basic",
        include_sections=["context", "forecast"],
        max_runtime=9.0,
    )

    assert insufficient["selected_runtime_estimate_seconds"] == 9.0
    assert insufficient["execution"] == ["context", "forecast"]
    assert insufficient["runtime_omitted"] == []
    assert insufficient["estimate_policy"] == "advisory_only"
    assert sufficient["execution"] == ["context", "forecast"]


def test_report_section_plan_schedules_all_work_before_actual_deadline() -> None:
    from mtdata.core.report.use_cases import _resolve_report_section_plan

    plan = _resolve_report_section_plan("basic", max_runtime=10.0)

    assert plan["execution"] == [
        "context",
        "pivot",
        "contexts_multi",
        "pivot_multi",
        "volatility",
        "backtest",
        "forecast",
        "barriers",
        "patterns",
        "confluence",
    ]
    assert plan["estimated_runtime_seconds"] == 78.0
    assert plan["runtime_omitted"] == []


def test_report_section_plan_adds_conformal_backtest_dependency() -> None:
    from mtdata.core.report.use_cases import _resolve_report_section_plan

    plan = _resolve_report_section_plan(
        "advanced",
        include_sections=["forecast_conformal"],
        max_runtime=1.0,
    )

    assert plan["requested"] == ["forecast_conformal"]
    assert plan["selected"] == ["forecast_conformal"]
    assert plan["execution"] == ["backtest", "forecast_conformal"]
    assert plan["required_dependencies"] == {
        "forecast_conformal": [
            {"section": "backtest", "estimated_runtime_seconds": 25.0}
        ]
    }
    assert plan["selected_runtime_estimate_seconds"] == 50.0


def test_report_generate_request_template_choices_are_validated():
    from mtdata.core.report.requests import ReportGenerateRequest

    assert ReportGenerateRequest(symbol="EURUSD", template="SCALPING").template == "scalping"
    with pytest.raises(ValidationError, match="template"):
        ReportGenerateRequest(symbol="EURUSD", template="unknown_xyz")


def test_basic_report_volatility_failure_keeps_method_errors(monkeypatch):
    from mtdata.core.report_templates import basic as template_basic_mod

    def fake_raw_result(func, *args, **kwargs):
        name = getattr(func, "__name__", "")
        if name == "forecast_volatility_estimate":
            return {"error": f"{kwargs.get('method')} unavailable"}
        return {"error": "stubbed section"}

    monkeypatch.setattr(template_basic_mod, "_get_raw_result", fake_raw_result)

    out = template_basic_mod.template_basic(
        "EURUSD",
        horizon=3,
        denoise=None,
        params={"timeframe": "H1"},
    )

    volatility = out["sections"]["volatility"]
    assert volatility["error"] == "Volatility estimation failed."
    assert volatility["hint"].startswith("Run forecast_volatility_estimate")
    assert volatility["errors"][0]["method"] == "yang_zhang"
    assert "unavailable" in volatility["errors"][0]["error"]


def test_basic_report_does_not_fallback_outside_rmse_gate(monkeypatch):
    from mtdata.core.report_templates import basic as template_basic_mod

    forecast_calls = []

    def fake_raw_result(func, *args, **kwargs):
        name = getattr(func, "__name__", "")
        if name == "forecast_backtest_run":
            return {
                "results": {
                    "theta": {
                        "success": True,
                        "avg_rmse": 0.9,
                        "avg_directional_accuracy": 0.7,
                    },
                    "naive": {
                        "success": True,
                        "avg_rmse": 0.95,
                        "avg_directional_accuracy": 0.8,
                    },
                }
            }
        if name == "forecast_generate":
            forecast_calls.append(kwargs["method"])
            if kwargs["method"] == "theta":
                return {"error": "model unavailable"}
            return {"forecast_price": [1.1, 1.2]}
        return {"error": "section not requested"}

    monkeypatch.setattr(template_basic_mod, "_get_raw_result", fake_raw_result)
    monkeypatch.setattr(
        template_basic_mod,
        "pick_best_forecast_method",
        lambda *args, **kwargs: (
            "theta",
            {
                "success": True,
                "avg_rmse": 0.9,
                "avg_directional_accuracy": 0.7,
            },
        ),
    )

    out = template_basic_mod.template_basic(
        "EURUSD",
        horizon=3,
        denoise=None,
        params={"_report_execution_sections": ["backtest", "forecast"]},
    )

    assert forecast_calls == ["theta"]
    assert out["sections"]["forecast"]["error"] == "model unavailable"
    assert out["sections"]["forecast"]["eligible_methods"] == ["theta"]


def test_run_report_generate_logs_finish_event(caplog):
    from mtdata.core.report.requests import ReportGenerateRequest
    from mtdata.core.report.use_cases import run_report_generate

    basic_template = MagicMock(
        return_value={"sections": _make_full_sections(), "diagnostics": {}}
    )
    with patch("mtdata.core.report_templates.template_basic", basic_template, create=True), \
         patch("mtdata.core.report_templates.template_minimal", basic_template, create=True), \
         patch("mtdata.core.report_templates.template_advanced", basic_template, create=True), \
         patch("mtdata.core.report_templates.template_scalping", basic_template, create=True), \
         patch("mtdata.core.report_templates.template_intraday", basic_template, create=True), \
         patch("mtdata.core.report_templates.template_swing", basic_template, create=True), \
         patch("mtdata.core.report_templates.template_position", basic_template, create=True), \
         caplog.at_level("DEBUG", logger="mtdata.core.report.use_cases"):
        result = run_report_generate(
            ReportGenerateRequest(symbol="EURUSD"),
            format_number=lambda value: str(value),
            get_indicator_value=lambda payload, key: payload.get(key),
            report_error_payload=lambda message: {"error": str(message)},
            append_diagnostic_warning=lambda report, message: None,
        )

    assert isinstance(result, dict)
    assert any(
        "event=finish operation=report_generate success=True" in record.message
        for record in caplog.records
    )


def test_run_report_generate_progress_respects_stderr_redirection():
    from mtdata.core.report.requests import ReportGenerateRequest
    from mtdata.core.report.use_cases import run_report_generate
    from mtdata.core.report.utils import emit_report_progress

    def template(*_args, **_kwargs):
        emit_report_progress("test_operation", "started")
        return {"sections": _make_full_sections(), "diagnostics": {}}

    captured = StringIO()
    with (
        patch("mtdata.core.report_templates.template_minimal", template, create=True),
        redirect_stderr(captured),
    ):
        run_report_generate(
            ReportGenerateRequest(symbol="EURUSD", progress=True),
            format_number=lambda value: str(value),
            get_indicator_value=lambda payload, key: payload.get(key),
            report_error_payload=lambda message: {"error": str(message)},
            append_diagnostic_warning=lambda report, message: None,
        )

    assert "report_generate progress" in captured.getvalue()


def test_run_report_generate_scopes_volatility_rate_cache():
    from mtdata.core.report.requests import ReportGenerateRequest
    from mtdata.core.report.use_cases import run_report_generate
    from mtdata.forecast import volatility

    cache_states = []

    def basic_template(*args, **kwargs):
        cache_states.append(volatility._RATES_CACHE.get() is not None)
        return {"sections": _make_full_sections(), "diagnostics": {}}

    with (
        patch("mtdata.core.report_templates.template_basic", basic_template, create=True),
        patch("mtdata.core.report_templates.template_minimal", basic_template, create=True),
        patch("mtdata.core.report_templates.template_advanced", basic_template, create=True),
        patch("mtdata.core.report_templates.template_scalping", basic_template, create=True),
        patch("mtdata.core.report_templates.template_intraday", basic_template, create=True),
        patch("mtdata.core.report_templates.template_swing", basic_template, create=True),
        patch("mtdata.core.report_templates.template_position", basic_template, create=True),
    ):
        run_report_generate(
            ReportGenerateRequest(symbol="EURUSD", template="basic"),
            format_number=lambda value: str(value),
            get_indicator_value=lambda payload, key: payload.get(key),
            report_error_payload=lambda message: {"error": str(message)},
            append_diagnostic_warning=lambda report, message: None,
        )

    assert cache_states == [True]
    assert volatility._RATES_CACHE.get() is None


def test_run_report_generate_uses_actual_runtime_instead_of_estimate_cap():
    from mtdata.core.report.requests import ReportGenerateRequest
    from mtdata.core.report.use_cases import run_report_generate

    captured_execution = []

    def basic_template(_symbol, _horizon, _denoise, params):
        captured_execution.extend(params["_report_execution_sections"])
        sections = {
            name: {"value": 1}
            for name in params["_report_execution_sections"]
        }
        sections["context"] = {"last_snapshot": {"close": 1.1}}
        sections["forecast"] = {"forecast": [1.1, 1.2]}
        return {"sections": sections}

    with patch(
        "mtdata.core.report_templates.template_basic",
        basic_template,
        create=True,
    ):
        result = run_report_generate(
            ReportGenerateRequest(
                symbol="EURUSD",
                template="basic",
                max_runtime=10,
                detail="full",
            ),
            format_number=lambda value: str(value),
            get_indicator_value=lambda payload, key: payload.get(key),
            report_error_payload=lambda message: {"error": str(message)},
            append_diagnostic_warning=lambda report, message: None,
        )

    assert captured_execution == [
        "context",
        "pivot",
        "contexts_multi",
        "pivot_multi",
        "volatility",
        "backtest",
        "forecast",
        "barriers",
        "patterns",
        "confluence",
    ]
    assert result["success"] is True
    assert result["section_run_status"] == "complete"
    assert result["sections_status"]["summary"] == {
        "ok": 10,
        "partial": 0,
        "error": 0,
        "omitted": 0,
        "total": 10,
    }
    assert result["runtime_plan"]["max_runtime_seconds"] == 10.0
    assert result["runtime_plan"]["estimated_runtime_seconds"] == 78.0
    assert result["runtime_plan"]["estimate_policy"] == "advisory_only"
    assert result["runtime_plan"]["runtime_budget_exhausted"] is False
    assert result["execution_progress"]["omitted_sections"] == []


def test_run_report_generate_exposes_dependency_inclusive_runtime_plan():
    from mtdata.core.report.requests import ReportGenerateRequest
    from mtdata.core.report.use_cases import run_report_generate

    def basic_template(_symbol, _horizon, _denoise, params):
        assert params["_report_execution_sections"] == ["context", "forecast"]
        return {
            "sections": {
                "context": {"last_snapshot": {"close": 1.1}},
                "forecast": {"forecast": [1.1, 1.2]},
            }
        }

    with patch(
        "mtdata.core.report_templates.template_basic",
        basic_template,
        create=True,
    ):
        result = run_report_generate(
            ReportGenerateRequest(
                symbol="EURUSD",
                template="basic",
                include_sections=["context", "forecast"],
                max_runtime=8,
                detail="full",
            ),
            format_number=lambda value: str(value),
            get_indicator_value=lambda payload, key: payload.get(key),
            report_error_payload=lambda message: {"error": str(message)},
            append_diagnostic_warning=lambda report, message: None,
        )

    plan = result["runtime_plan"]
    assert plan["selected_runtime_estimate_seconds"] == 9.0
    assert plan["scheduled_sections"] == ["context", "forecast"]
    assert plan["required_dependencies"] == {}
    assert plan["runtime_omitted_sections"] == []
    assert plan["runtime_budget_exhausted"] is False


def test_run_report_generate_preserves_capped_request_provenance():
    from mtdata.core.report.requests import ReportGenerateRequest
    from mtdata.core.report.use_cases import run_report_generate

    def minimal_template(_symbol, _horizon, _denoise, params):
        assert params["_report_execution_sections"] == ["context"]
        return {
            "meta": {"template": "minimal"},
            "sections": {"context": {"last_snapshot": {"close": 1.1}}},
        }

    with patch(
        "mtdata.core.report_templates.template_minimal",
        minimal_template,
        create=True,
    ):
        result = run_report_generate(
            ReportGenerateRequest(
                symbol="EURUSD",
                include_sections=["context", "forecast"],
                max_sections=1,
                detail="compact",
            ),
            format_number=lambda value: str(value),
            get_indicator_value=lambda payload, key: payload.get(key),
            report_error_payload=lambda message: {"error": str(message)},
            append_diagnostic_warning=lambda report, message: None,
        )

    assert result["runtime_plan"]["requested_sections"] == ["context", "forecast"]
    assert result["runtime_plan"]["selected_sections"] == ["context"]
    assert result["runtime_plan"]["requested_execution_sections"] == [
        "context",
        "forecast",
    ]
    assert result["section_controls"]["capped_requested_sections"] == ["forecast"]
    assert result["section_controls"]["exclusion_reasons"] == {
        "forecast": "max_sections_limited"
    }
    assert result["execution_progress"]["complete"] is False
    assert result["execution_progress"]["scheduled_selection_complete"] is True
    assert "Forecast was excluded by max_sections" in result["assessment"]["summary"]
    assert "Forecast was not requested" not in result["assessment"]["summary"]
    assert result["request_completion_status"] == "partial"
    assert result["section_run_status"] == "complete"


def test_conformal_only_report_executes_but_hides_backtest_dependency():
    from mtdata.core.report.requests import ReportGenerateRequest
    from mtdata.core.report.use_cases import run_report_generate

    def advanced_template(_symbol, _horizon, _denoise, params):
        assert params["_report_execution_sections"] == [
            "backtest",
            "forecast_conformal",
        ]
        return {
            "sections": {
                "backtest": {"best_method": {"method": "theta"}},
                "forecast_conformal": {
                    "method": "theta",
                    "intervals": [
                        {
                            "time": "2026-08-19T01:00:00Z",
                            "forecast": 1.1,
                            "lower_price": 1.0,
                            "upper_price": 1.2,
                        }
                    ],
                },
            }
        }

    with patch(
        "mtdata.core.report_templates.template_advanced",
        advanced_template,
        create=True,
    ):
        result = run_report_generate(
            ReportGenerateRequest(
                symbol="EURUSD",
                template="advanced",
                include_sections=["forecast_conformal"],
                detail="full",
            ),
            format_number=lambda value: str(value),
            get_indicator_value=lambda payload, key: payload.get(key),
            report_error_payload=lambda message: {"error": str(message)},
            append_diagnostic_warning=lambda report, message: None,
        )

    assert list(result["sections"]) == ["forecast_conformal"]
    assert result["sections_status"]["sections"] == {"forecast_conformal": "ok"}
    assert result["runtime_plan"]["required_dependencies"] == {
        "forecast_conformal": [
            {"section": "backtest", "estimated_runtime_seconds": 25.0}
        ]
    }


def test_run_report_generate_marks_actual_deadline_omission():
    from mtdata.core.report.requests import ReportGenerateRequest
    from mtdata.core.report.use_cases import run_report_generate

    deadline_error = {
        "error": "Report max_runtime budget was exhausted before context could start.",
        "error_code": "report_runtime_budget_exhausted",
        "runtime_budget_exhausted": True,
    }

    with patch(
        "mtdata.core.report_templates.template_minimal",
        return_value={"sections": {"context": deadline_error}},
        create=True,
    ):
        result = run_report_generate(
            ReportGenerateRequest(
                symbol="EURUSD",
                include_sections=["context"],
                max_runtime=1,
                detail="full",
            ),
            format_number=lambda value: str(value),
            get_indicator_value=lambda payload, key: payload.get(key),
            report_error_payload=lambda message: {"error": str(message)},
            append_diagnostic_warning=lambda report, message: None,
        )

    assert result["sections_status"]["sections"]["context"] == "omitted"
    assert result["sections_status"]["details"]["context"]["reason_code"] == (
        "report_runtime_deadline_exhausted"
    )
    assert result["runtime_plan"]["runtime_omitted_sections"] == ["context"]
    assert result["runtime_plan"]["estimate_limited_sections"] == []
    assert result["runtime_plan"]["runtime_budget_exhausted"] is True


def test_non_viable_barrier_decision_survives_compact_report():
    from mtdata.core.report.requests import ReportGenerateRequest
    from mtdata.core.report.use_cases import run_report_generate

    decision = {
        "status": "non_viable",
        "status_reason": "No candidate passed the viability filter.",
        "recommendation": "avoid",
        "mathematically_viable": False,
        "usable_for_live_trading": False,
        "candidates_evaluated": 0,
        "candidates_viable": 0,
        "candidates_returned": 0,
        "execution_blockers": ["optimizer_non_viable"],
    }
    with patch(
        "mtdata.core.report_templates.template_basic",
        return_value={
            "sections": {
                "barriers": {
                    "status": "non_viable",
                    "recommendation": "avoid",
                    "long": dict(decision),
                    "short": dict(decision),
                }
            }
        },
        create=True,
    ):
        result = run_report_generate(
            ReportGenerateRequest(
                symbol="EURUSD",
                template="basic",
                include_sections=["barriers"],
                detail="compact",
            ),
            format_number=lambda value: str(value),
            get_indicator_value=lambda payload, key: payload.get(key),
            report_error_payload=lambda message: {"error": str(message)},
            append_diagnostic_warning=lambda report, message: None,
        )

    assert result["section_run_status"] == "partial"
    assert result["sections_status"]["issues"]["barriers"]["reason"] == (
        "no barrier direction produced a mathematically viable setup"
    )
    barriers = result["summary_structured"]["barriers"]
    assert barriers["status"] == "non_viable"
    assert barriers["recommendation"] == "avoid"
    assert barriers["long"]["execution_blockers"] == ["optimizer_non_viable"]
    assert barriers["short"]["mathematically_viable"] is False
    assert result["assessment"]["recommended_action"] == "review_partial_sections"


def test_run_report_generate_promotes_common_symbol_failure_in_compact_output():
    from mtdata.core.report.requests import ReportGenerateRequest
    from mtdata.core.report.use_cases import run_report_generate

    message = "Symbol 'NO_SUCH_SYMBOL' was not found in MT5."

    def minimal_template(_symbol, _horizon, _denoise, _params):
        return {
            "sections": {
                "context": {"error": message},
                "forecast": {"error": message},
            }
        }

    with patch(
        "mtdata.core.report_templates.template_minimal",
        minimal_template,
        create=True,
    ):
        result = run_report_generate(
            ReportGenerateRequest(symbol="NO_SUCH_SYMBOL", detail="compact"),
            format_number=lambda value: str(value),
            get_indicator_value=lambda payload, key: payload.get(key),
            report_error_payload=lambda error: {"error": str(error)},
            append_diagnostic_warning=lambda report, warning: None,
        )

    assert result["success"] is False
    assert result["error_code"] == "symbol_not_found"
    assert result["request_id"]
    assert result["operation"] == "report_generate"
    assert result["related_tools"] == ["symbols_list"]
    assert result["sections_status"]["summary"]["error"] == 2


def test_run_report_generate_warns_once_for_degraded_sections(caplog):
    from mtdata.core.report.requests import ReportGenerateRequest
    from mtdata.core.report.use_cases import run_report_generate

    sections = _make_full_sections()
    sections["backtest"] = {"error": "model artifact disappeared"}
    basic_template = MagicMock(
        return_value={"sections": sections, "diagnostics": {}}
    )
    with (
        patch("mtdata.core.report_templates.template_basic", basic_template, create=True),
        patch("mtdata.core.report_templates.template_minimal", basic_template, create=True),
        patch("mtdata.core.report_templates.template_advanced", basic_template, create=True),
        patch("mtdata.core.report_templates.template_scalping", basic_template, create=True),
        patch("mtdata.core.report_templates.template_intraday", basic_template, create=True),
        patch("mtdata.core.report_templates.template_swing", basic_template, create=True),
        patch("mtdata.core.report_templates.template_position", basic_template, create=True),
        caplog.at_level(logging.WARNING, logger="mtdata.core.report.use_cases"),
    ):
        result = run_report_generate(
            ReportGenerateRequest(symbol="EURUSD", detail="full"),
            format_number=lambda value: str(value),
            get_indicator_value=lambda payload, key: payload.get(key),
            report_error_payload=lambda message: {"error": str(message)},
            append_diagnostic_warning=lambda report, message: None,
        )

    assert result["section_run_status"] == "partial"
    records = [
        record
        for record in caplog.records
        if "event=report_sections_degraded" in record.message
    ]
    assert len(records) == 1
    assert "error_sections=backtest" in records[0].message
    assert "backtest:model artifact disappeared" in records[0].message


def test_report_generate_returns_connection_error_payload(monkeypatch):
    from mtdata.core import report as report_mod

    raw = _unwrap(report_mod.report_generate)

    def fail_connection():
        raise MT5ConnectionError("Failed to connect to MetaTrader5. Ensure MT5 terminal is running.")

    monkeypatch.setattr(report_mod, "ensure_mt5_connection_or_raise", fail_connection)

    out = raw(request=report_mod.ReportGenerateRequest(symbol="EURUSD"))

    assert out["success"] is False
    assert out["error"] == "Failed to connect to MetaTrader5. Ensure MT5 terminal is running."
    assert out["error_code"] == "mt5_connection_error"
    assert out["operation"] == "mt5_ensure_connection"
    assert isinstance(out.get("request_id"), str)


def test_report_generate_logs_finish_event(monkeypatch, caplog):
    from mtdata.core import report as report_mod

    raw = _unwrap(report_mod.report_generate)
    monkeypatch.setattr(report_mod, "_report_connection_error", lambda: None)
    monkeypatch.setattr(report_mod, "run_report_generate", lambda *args, **kwargs: {"success": True, "sections": {}})

    with caplog.at_level(logging.DEBUG, logger=report_mod.logger.name):
        out = raw(request=report_mod.ReportGenerateRequest(symbol="EURUSD"))

    assert out["success"] is True
    assert any(
        "event=finish operation=report_generate success=True" in record.message
        for record in caplog.records
    )


def test_report_generate_uses_canonical_symbol_and_preserves_alias(monkeypatch):
    from mtdata.core import report as report_mod

    raw = _unwrap(report_mod.report_generate)
    seen = []
    monkeypatch.setattr(report_mod, "_report_connection_error", lambda: None)
    monkeypatch.setattr(
        report_mod,
        "resolve_public_symbol",
        lambda symbol: ("EURUSD", symbol),
    )

    def fake_run_report_generate(request, **kwargs):
        seen.append(request.symbol)
        return {"success": True, "sections": {}}

    monkeypatch.setattr(report_mod, "run_report_generate", fake_run_report_generate)

    out = raw(request=report_mod.ReportGenerateRequest(symbol="EUR/USD"))

    assert seen == ["EURUSD"]
    assert out["symbol"] == "EURUSD"
    assert out["symbol_input"] == "EUR/USD"


def test_report_generate_compact_structured_payload(monkeypatch):
    from mtdata.core import report as report_mod

    raw = _unwrap(report_mod.report_generate)
    monkeypatch.setattr(report_mod, "_report_connection_error", lambda: None)
    monkeypatch.setattr(
        report_mod,
        "run_report_generate",
        lambda *args, **kwargs: {
            "success": True,
            "detail": "compact",
            "summary_structured": {"market": {"close": 1.10}},
            "sections_status": {"summary": {"ok": 1, "partial": 0, "error": 0}},
        },
    )

    out = raw(request=report_mod.ReportGenerateRequest(symbol="EURUSD"))

    assert out["detail"] == "compact"
    assert out["summary_structured"] == {"market": {"close": 1.10}}
    assert "summary" not in out
    assert "sections" not in out


def test_report_generate_rejects_future_end():
    from datetime import datetime, timedelta, timezone

    from pydantic import ValidationError

    from mtdata.core.report.requests import ReportGenerateRequest

    future = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
    with pytest.raises(
        ValidationError,
        match="in the future; historical ranges must have elapsed",
    ) as caught:
        ReportGenerateRequest(symbol="EURUSD", end=future)

    payload = {
        "success": False,
        "error": str(caught.value.errors()[0]["msg"]),
        "error_code": "tool_error",
        "operation": "report_generate",
    }
    from mtdata.core.error_envelope import normalize_error_payload

    normalized = normalize_error_payload(payload, operation="report_generate")
    assert normalized["error_code"] == "report_end_in_future"


def test_compact_report_payload_summarizes_runtime_on_success():
    from mtdata.core.report.use_cases import _compact_report_payload

    out = _compact_report_payload(
        {
            "success": True,
            "section_run_status": "complete",
            "runtime_plan": {
                "actual_runtime_seconds": 4.25,
                "runtime_budget_exhausted": False,
                "estimated_runtime_seconds": 78.0,
                "required_dependencies": {"context": ["market_ticker"]},
            },
            "execution_progress": {
                "completed_sections": ["context", "forecast"],
                "selected_sections": ["context", "forecast"],
                "scheduled_sections": ["context", "forecast", "patterns"],
            },
            "summary_structured": {"market": {"close": 1.10}},
        },
        symbol="EURUSD",
        template="minimal",
    )

    assert out["sections_completed"] == 2
    assert out["runtime_seconds"] == 4.25
    assert out["runtime_budget_exhausted"] is False
    assert "runtime_plan" not in out
    assert "execution_progress" not in out


def test_compact_partial_report_keeps_runtime_internals():
    from mtdata.core.report.use_cases import _compact_report_payload

    plan = {"actual_runtime_seconds": 9.0, "runtime_budget_exhausted": True}
    out = _compact_report_payload(
        {
            "success": True,
            "section_run_status": "partial",
            "runtime_plan": plan,
            "execution_progress": {"completed_sections": ["context"]},
            "summary_structured": {"market": {"close": 1.10}},
        },
        symbol="EURUSD",
        template="minimal",
    )

    assert out["runtime_plan"] == plan
    assert out["execution_progress"]["completed_sections"] == ["context"]


def test_compact_report_payload_omits_string_summary_when_structured_exists():
    from mtdata.core.report.use_cases import _compact_report_payload

    out = _compact_report_payload(
        {
            "success": True,
            "summary": ["close=1.10"],
            "summary_structured": {"market": {"close": 1.10}},
            "sections_status": {"summary": {"ok": 1, "partial": 0, "error": 0}},
        },
        symbol="EURUSD",
        template="basic",
    )

    assert out["summary_structured"] == {"market": {"close": 1.10}}
    assert "summary" not in out


def test_compact_failed_report_preserves_error_envelope():
    from mtdata.core.report.use_cases import _compact_report_payload

    out = _compact_report_payload(
        {
            "success": False,
            "error": "Symbol 'NO_SUCH_SYMBOL' was not found in MT5.",
            "error_code": "symbol_not_found",
            "request_id": "request-1",
            "operation": "report_generate",
            "remediation": "Use symbols_list.",
            "related_tools": ["symbols_list"],
            "sections_status": {"summary": {"ok": 0, "partial": 0, "error": 2}},
        },
        symbol="NO_SUCH_SYMBOL",
        template="minimal",
    )

    assert out["error_code"] == "symbol_not_found"
    assert out["request_id"] == "request-1"
    assert out["related_tools"] == ["symbols_list"]


def test_compact_report_payload_rounds_summary_floats_to_six_significant_digits():
    from mtdata.core.report.use_cases import _compact_report_payload

    out = _compact_report_payload(
        {
            "success": True,
            "summary_structured": {
                "market": {"close": 1.14375, "rsi": 47.792770221604044},
                "backtest": {
                    "stats": {
                        "avg_rmse": 0.0018917237203165866,
                        "avg_directional_accuracy": 0.5309090909090909,
                    }
                },
                "volatility": {"sigma": 0.001660812248886065},
            },
        },
        symbol="EURUSD",
        template="basic",
    )

    structured = out["summary_structured"]
    assert structured["market"] == {"close": 1.14375, "rsi": 47.7928}
    assert structured["backtest"]["stats"] == {
        "avg_rmse": 0.00189172,
        "avg_directional_accuracy": 0.530909,
    }
    assert structured["volatility"]["sigma"] == 0.00166081


def test_compact_report_payload_omits_duplicate_assessment_blocks():
    from mtdata.core.report.use_cases import _compact_report_payload

    out = _compact_report_payload(
        {
            "success": True,
            "section_run_status": "partial",
            "content_detail": "summary_only",
            "executive_summary": {
                "recommended_action": "review_partial_sections",
                "confidence": "medium",
                "section_health": {"ok": 7, "partial": 1, "error": 0, "total": 8},
                "sections_with_issues": {"partial": ["barriers"]},
            },
            "overall_assessment": {
                "recommended_action": "review_partial_sections",
                "confidence": "medium",
                "section_health": {"ok": 7, "partial": 1, "error": 0, "total": 8},
                "partial_sections": ["barriers"],
            },
            "sections_with_issues": {"partial": ["barriers"]},
            "sections_status": {
                "summary": {"ok": 7, "partial": 1, "error": 0, "total": 8},
                "sections": {"forecast": "ok", "barriers": "partial"},
                "details": {
                    "barriers": {
                        "status": "partial",
                        "reason": "section contains usable data plus one or more nested errors",
                        "errors": [{"path": "short", "message": "optimizer failed"}],
                    }
                },
            },
            "summary_structured": {
                "template_focus": {"profile": "balanced"},
            },
        },
        symbol="EURUSD",
        template="basic",
    )

    assert out["assessment"]["partial_sections"] == ["barriers"]
    assert out["section_run_status"] == "partial"
    assert out["content_detail"] == "summary_only"
    assert "section_health" not in out["assessment"]
    assert out["summary_structured"]["template_focus"] == {"profile": "balanced"}
    assert "executive_summary" not in out
    assert "overall_assessment" not in out
    assert "sections_with_issues" not in out
    assert out["sections_status"] == {
        "summary": {"ok": 7, "partial": 1, "error": 0, "total": 8},
        "issues": {
            "barriers": {
                "status": "partial",
                "reason": "section contains usable data plus one or more nested errors",
                "errors": [{"path": "short", "message": "optimizer failed"}],
            }
        },
    }


def test_compact_report_omitted_sections_excludes_temporal_alignment():
    from mtdata.core.report.use_cases import _compact_report_payload

    out = _compact_report_payload(
        {
            "success": True,
            "sections_status": {
                "summary": {"ok": 2, "partial": 0, "error": 0, "omitted": 0}
            },
            "summary_structured": {
                "market": {"close": 1.10},
                "forecast": {"method": "arima"},
                "temporal_alignment": {"status": "aligned"},
            },
        },
        symbol="EURUSD",
        template="minimal",
    )

    structured = out["summary_structured"]
    omitted = structured.get("omitted_sections") or []
    assert "temporal_alignment" not in omitted
    assert out["health"]["omitted"] == len(omitted)
    assert out["health"]["omitted"] == 0


def test_compact_report_payload_uses_one_named_assessment_when_healthy():
    from mtdata.core.report.use_cases import _compact_report_payload

    out = _compact_report_payload(
        {
            "success": True,
            "overall_assessment": {
                "is_trade_signal": False,
                "recommended_action": "review_key_levels_and_risk",
                "assembly_confidence": "high",
                "assembly_confidence_basis": "report_section_health",
                "section_health": {"ok": 4, "partial": 0, "error": 0, "total": 4},
            },
            "executive_summary": {
                "is_trade_signal": False,
                "recommended_action": "review_key_levels_and_risk",
                "assembly_confidence": "high",
                "section_health": {"ok": 4, "partial": 0, "error": 0, "total": 4},
            },
            "sections_status": {"summary": {"ok": 4, "partial": 0, "error": 0}},
            "summary_structured": {"market": {"close": 1.1}},
        },
        symbol="EURUSD",
        template="basic",
    )

    assert out["assessment"] == {
        "is_trade_signal": False,
        "recommended_action": "review_key_levels_and_risk",
        "section_completeness": "high",
        "section_health": {"ok": 4, "partial": 0, "error": 0},
    }
    assert "overall_assessment" not in out
    assert "executive_summary" not in out


def test_compact_report_payload_elevates_barrier_conflicts():
    from mtdata.core.report.use_cases import _compact_report_payload

    out = _compact_report_payload(
        {
            "success": True,
            "summary": ["dense duplicate"],
            "summary_structured": {
                "barriers": {
                    "long": {"ev_edge_conflict": True},
                    "short": {"ev_edge_conflict": True},
                }
            },
            "diagnostics": {"warnings": ["existing warning"]},
        },
        symbol="EURUSD",
        template="basic",
    )

    assert "summary" not in out
    assert out["warnings"] == [
        "existing warning",
        "Barrier EV/edge conflict detected for long and short direction(s).",
    ]
    assert "trading_note" not in out["summary_structured"]["barriers"]["long"]


def test_report_generate_compact_keeps_actionable_section_summaries():
    fn = _get_report_generate()
    sections = _make_full_sections()
    sections["pivot"] = {
        "method": "classic",
        "levels": {"PP": 1.102, "R1": 1.106, "S1": 1.098},
        "timezone": "UTC",
    }
    sections["volatility"] = {
        "methods": ["ewma", "parkinson"],
        "aggregate_method": "ensemble_mean",
        "matrix": [
            {
                "horizon": 3,
                "ewma": 0.0041,
                "parkinson": 0.0039,
                "avg": 0.004,
                "avg_method": "ensemble_mean",
                "contributors": [
                    {"method": "ewma", "value": 0.0041, "weight": 0.5},
                    {"method": "parkinson", "value": 0.0039, "weight": 0.5},
                ],
            }
        ],
    }
    sections["forecast"] = {
        "method": "EMA",
        "forecast_price": [1.101, 1.103, 1.104],
    }
    sections["backtest"] = {
        "best_method": {
            "method": "EMA",
            "stats": {
                "avg_rmse": 0.001,
                "avg_directional_accuracy": 0.61,
            },
        }
    }
    sections["patterns"] = {
        "recent": [
            {"pattern": "hammer", "direction": "bullish", "confidence": 0.8, "debug": "drop"}
        ]
    }
    rep = _make_report(sections=sections)
    mock_basic = MagicMock(return_value=rep)

    with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
         patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
         patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
         patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
         patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
         patch("mtdata.core.report_templates.template_position", mock_basic, create=True), \
         patch(_FMT_NUM, side_effect=str):
        out = fn("EURUSD", template="basic", horizon=3, detail="compact")

    structured = out["summary_structured"]
    assert structured["pivot"] == {
        "method": "classic",
        "PP": 1.102,
        "R1": 1.106,
        "S1": 1.098,
        "display_timezone": "UTC",
    }
    assert structured["volatility"] == {
        "horizon": 3,
        "sigma": 0.004,
        "method": "ensemble_mean",
    }
    assert structured["forecast"]["first"] == 1.101
    assert structured["forecast"]["last"] == 1.104
    assert structured["backtest"]["best_method"] == "EMA"
    assert structured["patterns"]["recent"] == [
        {
            "pattern": "hammer",
            "name": "hammer",
            "direction": "bullish",
            "confidence": 0.8,
            "match_score": 0.8,
        }
    ]
    assert out["timezone"] == "UTC"
    assert "sections" not in out


def test_report_generate_compact_stamps_close_as_of_in_narrative():
    fn = _get_report_generate()
    sections = _make_full_sections()
    sections["context"] = {
        "timeframe": "H1",
        "price_precision": 5,
        "last_snapshot": {
            "close": 1.15825,
            "time": "2026-09-02T23:00Z",
        },
    }
    rep = _make_report(sections=sections)
    mock_basic = MagicMock(return_value=rep)

    with patch(
        "mtdata.core.report_templates.template_basic", mock_basic, create=True
    ):
        out = fn("EURUSD", template="basic", horizon=3, detail="compact")

    market = out["summary_structured"]["market"]
    assert market["close_as_of"] == "2026-09-03T00:00:00Z"
    assert market["bar_open"] == "2026-09-02T23:00Z"
    assert market["close_bar_state"] == "completed"
    assert (
        "Last close 1.15825 as of 2026-09-03T00:00:00Z (completed bar)"
        in out["summary_structured"]["narrative"]
    )


def test_report_generate_compact_preserves_symbol_price_precision():
    fn = _get_report_generate()
    sections = _make_full_sections()
    sections["context"] = {
        "price_precision": 5,
        "last_snapshot": {"close": 1.15825},
    }
    rep = _make_report(sections=sections)
    mock_basic = MagicMock(return_value=rep)

    with patch(
        "mtdata.core.report_templates.template_basic", mock_basic, create=True
    ):
        out = fn("EURUSD", template="basic", horizon=3, detail="compact")

    market = out["summary_structured"]["market"]
    assert market["close"] == 1.15825
    assert market["price_precision"] == 5
    assert "Last close 1.15825" in out["summary_structured"]["narrative"]


def test_report_generate_compact_exposes_template_focus():
    fn = _get_report_generate()
    sections = _make_full_sections()
    sections["contexts_multi"] = {
        "H1": {"trend_compact": "up"},
        "H4": {"trend_compact": "up"},
        "D1": {"trend_compact": "mixed"},
    }
    sections["pivot_multi"] = {
        "D1": {"levels": {}},
        "W1": {"levels": {}},
        "__base_timeframe__": "H4",
    }
    sections["volume_profile"] = {"poc": 1.101}
    sections["news"] = {"status": "no_results"}
    rep = _make_report(sections=sections)
    rep["meta"] = {"timeframe": "H4"}
    mock_swing = MagicMock(return_value=rep)

    with patch("mtdata.core.report_templates.template_basic", mock_swing, create=True), \
         patch("mtdata.core.report_templates.template_advanced", mock_swing, create=True), \
         patch("mtdata.core.report_templates.template_scalping", mock_swing, create=True), \
         patch("mtdata.core.report_templates.template_intraday", mock_swing, create=True), \
         patch("mtdata.core.report_templates.template_swing", mock_swing, create=True), \
         patch("mtdata.core.report_templates.template_position", mock_swing, create=True), \
         patch(_FMT_NUM, side_effect=str):
        out = fn("EURUSD", template="swing", horizon=24, detail="compact")

    focus = out["summary_structured"]["template_focus"]
    assert out["section_run_status"] == "complete"
    assert out["content_detail"] == "summary_only"
    assert focus["profile"] == "swing_mtf"
    assert focus["base_timeframe"] == "H4"
    assert focus["horizon"] == 24
    assert focus["context_timeframes"] == ["H1", "H4", "D1"]
    assert focus["pivot_timeframes"] == ["D1", "W1"]
    assert "__base_timeframe__" not in focus["pivot_timeframes"]
    assert focus["extra_sections"] == ["confluence", "volume_profile", "news"]


def test_report_generate_compact_keeps_style_distinctive_summaries():
    fn = _get_report_generate()
    sections = _make_full_sections()
    sections["market"] = {"bid": 1.1, "ask": 1.1002, "spread_ticks": 2.0, "depth_status": "quote_only"}
    sections["execution_gates"] = {"status": "pass", "execution_ready": True}
    sections["session"] = {"status": "open"}
    sections["news"] = {"upcoming_events": [{"title": "CPI"}]}
    sections["regime"] = {"hmm": {"summary": "range"}}
    sections["volume_profile"] = {"poc": 1.101}
    rep = _make_report(sections=sections)
    mock_intraday = MagicMock(return_value=rep)

    with patch("mtdata.core.report_templates.template_basic", mock_intraday, create=True), \
         patch("mtdata.core.report_templates.template_advanced", mock_intraday, create=True), \
         patch("mtdata.core.report_templates.template_scalping", mock_intraday, create=True), \
         patch("mtdata.core.report_templates.template_intraday", mock_intraday, create=True), \
         patch("mtdata.core.report_templates.template_swing", mock_intraday, create=True), \
         patch("mtdata.core.report_templates.template_position", mock_intraday, create=True), \
         patch(_FMT_NUM, side_effect=str):
        out = fn("EURUSD", template="intraday", horizon=12, detail="compact")

    structured = out["summary_structured"]
    assert structured["session"]["status"] == "open"
    assert structured["news"]["upcoming_events"][0]["title"] == "CPI"
    assert structured["confluence"]["levels"][0]["price"] == 1.102
    assert structured["levels"][0]["price"] == 1.102
    assert structured["market"]["bid"] == 1.1
    assert structured["execution_gates"]["execution_ready"] is True
    assert "narrative" in structured
    assert out["health"]["ok"] >= 1


def test_report_generate_standard_infers_root_timezone_from_sections():
    fn = _get_report_generate()
    rep = _make_report(
        sections={
            "context": {
                "timezone": "America/Chicago",
                "last_snapshot": {"time": "2026-03-29 10:00", "close": 1.1},
            }
        }
    )
    mock_basic = MagicMock(return_value=rep)

    with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
         patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
         patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
         patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
         patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
         patch("mtdata.core.report_templates.template_position", mock_basic, create=True):
        out = fn(
            "EURUSD",
            template="basic",
            horizon=3,
            detail="standard",
            include_sections=["context"],
        )

    assert out["timezone"] == "America/Chicago"
    assert out["sections"]["context"]["timezone"] == "America/Chicago"
    assert out["section_run_status"] == "complete"
    assert out["content_detail"] == "selected_sections"


def _make_full_sections():
    """Create a rich sections dict to exercise summary extraction."""
    return {
        "context": {
            "last_snapshot": {
                "close": 1.1020,
                "EMA_20": 1.1010,
                "EMA_50": 1.1000,
                "RSI_14": 55.5,
            },
        },
        "pivot": {
            "levels": [
                {"level": "R1", "classic": 1.1060},
                {"level": "PP", "classic": 1.1020},
                {"level": "S1", "classic": 1.0980},
            ],
            "methods": [{"method": "classic"}],
        },
        "contexts_multi": {"H1": {"trend_compact": "up"}},
        "pivot_multi": {"D1": {"levels": {}}},
        "volatility": {
            "horizon_sigma_price": 0.0045,
        },
        "backtest": {
            "best_method": {"method": "EMA", "stats": {"avg_rmse": 0.001}},
        },
        "forecast": {
            "method": "EMA",
            "forecast": [{"time": "2026-01-01T01:00Z", "value": 1.103}],
        },
        "barriers": {
            "long": {
                "best": {"tp": 1.5, "sl": 0.8, "edge": 0.3},
            },
            "short": {
                "best": {"tp": 1.2, "sl": 0.7, "edge": 0.2},
            },
        },
        "patterns": {"recent": []},
        "confluence": {"levels": [{"price": 1.102, "role": "at", "score": 12.0}]},
    }


_TEMPLATES_PATH = "mtdata.core.report"
_FMT_NUM = "mtdata.core.report.format_number"
_GET_IND = "mtdata.core.report._get_indicator_value"


# Template mock shortcuts
def _patch_templates():
    """Return a patch context for the template imports block."""
    return patch(f"{_TEMPLATES_PATH}.template_basic", create=True), \
           patch(f"{_TEMPLATES_PATH}.template_advanced", create=True), \
           patch(f"{_TEMPLATES_PATH}.template_scalping", create=True), \
           patch(f"{_TEMPLATES_PATH}.template_intraday", create=True), \
           patch(f"{_TEMPLATES_PATH}.template_swing", create=True), \
           patch(f"{_TEMPLATES_PATH}.template_position", create=True)


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------

from mtdata.core.report import _report_error_payload


class TestReportErrorHelpers:

    def test_error_payload_normal(self):
        result = _report_error_payload("oops")
        assert result["success"] is False
        assert result["error"] == "oops"
        assert result["error_code"] == "report_generation_error"
        assert result["operation"] == "report_generate"
        assert isinstance(result.get("request_id"), str)

    def test_error_payload_empty(self):
        result = _report_error_payload("")
        assert result["error"] == "Unknown error."
        assert result["error_code"] == "report_generation_error"


# ---------------------------------------------------------------------------
# report_generate — template dispatch
# ---------------------------------------------------------------------------


class TestReportTemplateDispatch:

    def _run(self, template, format="toon", horizon=None, methods=None, timeframe=None):
        fn = _get_report_generate()
        rep = _make_report(sections=_make_full_sections())

        mock_templates = {
            "basic": MagicMock(return_value=rep),
            "minimal": MagicMock(return_value=rep),
            "advanced": MagicMock(return_value=rep),
            "scalping": MagicMock(return_value=rep),
            "intraday": MagicMock(return_value=rep),
            "swing": MagicMock(return_value=rep),
            "position": MagicMock(return_value=rep),
        }

        with patch(f"{_TEMPLATES_PATH}._t_basic", mock_templates["basic"], create=True), \
             patch(f"{_TEMPLATES_PATH}._t_minimal", mock_templates["minimal"], create=True), \
             patch(f"{_TEMPLATES_PATH}._t_advanced", mock_templates["advanced"], create=True), \
             patch(f"{_TEMPLATES_PATH}._t_scalping", mock_templates["scalping"], create=True), \
             patch(f"{_TEMPLATES_PATH}._t_intraday", mock_templates["intraday"], create=True), \
             patch(f"{_TEMPLATES_PATH}._t_swing", mock_templates["swing"], create=True), \
             patch(f"{_TEMPLATES_PATH}._t_position", mock_templates["position"], create=True), \
             patch(_FMT_NUM, side_effect=str):
            # Patch the import block inside the function
            mock_mod = MagicMock()
            mock_mod.template_basic = mock_templates["basic"]
            mock_mod.template_minimal = mock_templates["minimal"]
            mock_mod.template_advanced = mock_templates["advanced"]
            mock_mod.template_scalping = mock_templates["scalping"]
            mock_mod.template_intraday = mock_templates["intraday"]
            mock_mod.template_swing = mock_templates["swing"]
            mock_mod.template_position = mock_templates["position"]

            with patch(f"{_TEMPLATES_PATH}.report_generate") as mock_rg:
                # Call the real unwrapped function
                pass

            # Actually run the function with template import patched
            with patch("mtdata.core.report_templates.template_basic", mock_templates["basic"], create=True), \
                 patch("mtdata.core.report_templates.template_minimal", mock_templates["minimal"], create=True), \
                 patch("mtdata.core.report_templates.template_advanced", mock_templates["advanced"], create=True), \
                 patch("mtdata.core.report_templates.template_scalping", mock_templates["scalping"], create=True), \
                 patch("mtdata.core.report_templates.template_intraday", mock_templates["intraday"], create=True), \
                 patch("mtdata.core.report_templates.template_swing", mock_templates["swing"], create=True), \
                 patch("mtdata.core.report_templates.template_position", mock_templates["position"], create=True):
                res = fn("EURUSD", template=template, format=format,
                         horizon=horizon, methods=methods, timeframe=timeframe)
        return res

    def test_basic_template_toon(self):
        res = self._run("basic")
        assert isinstance(res, dict)

    def test_advanced_template(self):
        res = self._run("advanced")
        assert isinstance(res, dict)

    def test_minimal_template(self):
        res = self._run("minimal")
        assert isinstance(res, dict)

    def test_scalping_template(self):
        res = self._run("scalping")
        assert isinstance(res, dict)

    def test_intraday_template(self):
        res = self._run("intraday")
        assert isinstance(res, dict)

    def test_swing_template(self):
        res = self._run("swing")
        assert isinstance(res, dict)

    def test_position_template(self):
        res = self._run("position")
        assert isinstance(res, dict)


class TestReportUnknownTemplate:

    def test_unknown_toon(self):
        fn = _get_report_generate()
        with pytest.raises(ValidationError, match="template"):
            fn("EURUSD", template="unknown_xyz", format="toon")


# ---------------------------------------------------------------------------
# Template import failure
# ---------------------------------------------------------------------------


class TestReportTemplateImportError:

    def test_import_error_toon(self):
        fn = _get_report_generate()
        with patch.dict("sys.modules", {"mtdata.core.report_templates": None}):
            res = fn("EURUSD", template="basic", format="toon")
        assert "error" in res


# ---------------------------------------------------------------------------
# Template returns non-dict / error
# ---------------------------------------------------------------------------


class TestReportBadTemplateReturn:

    def test_non_dict_toon(self):
        fn = _get_report_generate()
        mock_basic = MagicMock(return_value="not a dict")
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_minimal", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True):
            res = fn("EURUSD", template="basic", format="toon")
        assert "error" in res

    def test_error_in_report_toon(self):
        fn = _get_report_generate()
        mock_basic = MagicMock(return_value={"error": "data fetch failed"})
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True):
            res = fn("EURUSD", template="basic", format="toon")
        assert "error" in res


# ---------------------------------------------------------------------------
# Horizon selection
# ---------------------------------------------------------------------------


class TestReportHorizon:

    def _run_with_horizon(self, horizon=None, params=None, template="basic"):
        fn = _get_report_generate()
        rep = _make_report(sections={})
        mock_basic = MagicMock(return_value=rep)
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_minimal", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True), \
             patch(_FMT_NUM, side_effect=str):
            fn("EURUSD", template=template, horizon=horizon, params=params)
        return mock_basic

    def test_default_horizon_basic(self):
        mock = self._run_with_horizon(template="basic")
        args = mock.call_args
        # Second positional arg is horizon
        assert args[0][1] == 12

    def test_explicit_horizon(self):
        mock = self._run_with_horizon(horizon=20, template="basic")
        assert mock.call_args[0][1] == 20

    def test_horizon_from_params(self):
        mock = self._run_with_horizon(params={"horizon": 30}, template="basic")
        assert mock.call_args[0][1] == 30

    def test_minimal_default_horizon(self):
        mock = self._run_with_horizon(template="minimal")
        assert mock.call_args[0][1] == 12

    def test_scalping_default_horizon(self):
        mock = self._run_with_horizon(template="scalping")
        assert mock.call_args[0][1] == 8

    def test_swing_default_horizon(self):
        mock = self._run_with_horizon(template="swing")
        assert mock.call_args[0][1] == 24

    def test_position_default_horizon(self):
        mock = self._run_with_horizon(template="position")
        assert mock.call_args[0][1] == 30


# ---------------------------------------------------------------------------
# Summary extraction — context
# ---------------------------------------------------------------------------


class TestReportSummaryContext:

    def _run_report(self, sections):
        fn = _get_report_generate()
        rep = _make_report(sections=sections)
        mock_basic = MagicMock(return_value=rep)
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True), \
             patch(_FMT_NUM, side_effect=str):
            res = fn("EURUSD", template="basic", format="toon")
        return res

    def test_close_in_summary(self):
        sec = _make_full_sections()
        res = self._run_report(sec)
        assert any("close=" in s for s in res.get("summary", []))

    def test_trend_above_emas(self):
        sec = _make_full_sections()
        res = self._run_report(sec)
        assert any("above EMAs" in s for s in res.get("summary", []))

    def test_trend_mixed(self):
        sec = _make_full_sections()
        sec["context"]["last_snapshot"]["close"] = 1.0990  # below EMA_50
        res = self._run_report(sec)
        assert any("mixed" in s for s in res.get("summary", []))
        market = res.get("summary_structured", {}).get("market") or {}
        assert market.get("trend") == "mixed"
        assert market.get("trend_basis") == "last_completed_close_vs_ema20_ema50"

    def test_trend_skipped_when_close_missing(self):
        sec = _make_full_sections()
        sec["context"]["last_snapshot"]["close"] = None
        res = self._run_report(sec)
        assert not any("trend:" in s for s in res.get("summary", []))

    def test_rsi_in_summary(self):
        sec = _make_full_sections()
        res = self._run_report(sec)
        assert any("RSI=" in s for s in res.get("summary", []))

    def test_summary_key_precedes_sections_in_payload(self):
        sec = _make_full_sections()
        res = self._run_report(sec)
        keys = list(res.keys())
        assert keys.index("summary") < keys.index("sections")

    def test_no_context(self):
        res = self._run_report({})
        assert "summary" in res


# ---------------------------------------------------------------------------
# Summary extraction — pivot
# ---------------------------------------------------------------------------


class TestReportSummaryPivot:

    def _run_report(self, sections):
        fn = _get_report_generate()
        rep = _make_report(sections=sections)
        mock_basic = MagicMock(return_value=rep)
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True), \
             patch(_FMT_NUM, side_effect=str):
            return fn("EURUSD", template="basic", format="toon")

    def test_pivot_in_summary(self):
        sec = _make_full_sections()
        res = self._run_report(sec)
        assert any("pivot" in s for s in res.get("summary", []))

    def test_pivot_method_fallback(self):
        """When chosen method not in available methods, fallback to first available."""
        sec = _make_full_sections()
        sec["pivot"]["methods"] = [{"method": "fibonacci"}]
        sec["pivot"]["levels"] = [
            {"level": "R1", "woodie": 1.106},
            {"level": "PP", "woodie": 1.102},
            {"level": "S1", "woodie": 1.098},
        ]
        res = self._run_report(sec)
        # Should fallback to 'woodie' since 'fibonacci' not in level columns
        assert isinstance(res, dict)

    def test_pivot_context_in_summary(self):
        sec = _make_full_sections()
        sec["pivot"]["calculation_basis"] = {
            "session_boundary": "MT5 broker/session calendar",
            "display_timezone": "UTC",
        }
        res = self._run_report(sec)
        assert any("pivot context" in s for s in res.get("summary", []))


# ---------------------------------------------------------------------------
# Summary extraction — volatility & forecast
# ---------------------------------------------------------------------------


class TestReportSummaryVolForecast:

    def _run_report(self, sections):
        fn = _get_report_generate()
        rep = _make_report(sections=sections)
        mock_basic = MagicMock(return_value=rep)
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True), \
             patch(_FMT_NUM, side_effect=str):
            return fn("EURUSD", template="basic", format="toon")

    def test_vol_sigma(self):
        sec = _make_full_sections()
        res = self._run_report(sec)
        assert any("sigma=" in s for s in res.get("summary", []))

    def test_vol_return_sigma_uses_canonical_field(self):
        sec = _make_full_sections()
        sec["volatility"] = {"volatility_horizon": 0.003}
        res = self._run_report(sec)
        assert any("sigma=" in s for s in res.get("summary", []))

    def test_forecast_in_summary(self):
        sec = _make_full_sections()
        res = self._run_report(sec)
        assert any("forecast=" in s for s in res.get("summary", []))

    def test_forecast_timing_in_summary(self):
        sec = _make_full_sections()
        sec["forecast"].update({
            "last_observation_epoch": 1740948000.0,
            "forecast_start_epoch": 1740951600.0,
            "forecast_anchor": "next_timeframe_bar_after_last_observation",
        })
        res = self._run_report(sec)
        assert any("forecast timing:" in s for s in res.get("summary", []))

    def test_forecast_selection_criteria_in_summary(self):
        sec = _make_full_sections()
        sec["backtest"] = {
            "selection_criteria": {
                "primary_metric": "avg_rmse",
                "rmse_tolerance_pct": 5.0,
                "tie_breaker": "avg_directional_accuracy",
            },
            "best_method": {"method": "naive"},
        }
        res = self._run_report(sec)
        assert any("forecast selection:" in s for s in res.get("summary", []))

    def test_forecast_selection_criteria_includes_min_directional_accuracy(self):
        sec = _make_full_sections()
        sec["backtest"] = {
            "selection_criteria": {
                "primary_metric": "avg_rmse",
                "rmse_tolerance_pct": 5.0,
                "tie_breaker": "avg_directional_accuracy",
                "min_directional_accuracy": 0.55,
            },
            "best_method": {"method": "naive"},
        }
        res = self._run_report(sec)
        assert any("min-dir-acc>=" in s for s in res.get("summary", []))


# ---------------------------------------------------------------------------
# Summary extraction — barriers
# ---------------------------------------------------------------------------


class TestReportSummaryBarriers:

    def _run_report(self, sections):
        fn = _get_report_generate()
        rep = _make_report(sections=sections)
        mock_basic = MagicMock(return_value=rep)
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True), \
             patch(_FMT_NUM, side_effect=str):
            return fn("EURUSD", template="basic", format="toon")

    def test_long_short_barriers(self):
        sec = _make_full_sections()
        res = self._run_report(sec)
        summ = res.get("summary", [])
        assert any("dir=long" in s for s in summ)
        assert any("dir=short" in s for s in summ)

    def test_single_best_barrier(self):
        """Old-style barriers with single best/direction (lines 217-233)."""
        sec = _make_full_sections()
        sec["barriers"] = {
            "best": {"tp": 2.0, "sl": 1.0, "edge": 0.5},
            "direction": "long",
        }
        res = self._run_report(sec)
        assert any("barrier best" in s for s in res.get("summary", []))

    def test_barrier_summary_includes_ev_and_conflict_hint(self):
        sec = _make_full_sections()
        sec["barriers"]["long"]["best"]["ev"] = 0.03
        sec["barriers"]["long"]["best"]["edge"] = -0.1
        sec["barriers"]["long"]["best"]["edge_vs_breakeven"] = -0.2
        res = self._run_report(sec)
        long_line = [s for s in res.get("summary", []) if "barrier best" in s and "dir=long" in s][0]
        assert "ev=" in long_line
        assert "edge=" in long_line
        assert "edge_vs_breakeven=" in long_line
        assert "ev_edge_conflict=true" in long_line
        assert "ev_edge_conflict_reason=" in long_line
        structured = res["summary_structured"]["barriers"]["long"]
        assert structured["ev"] == 0.03
        assert structured["probability_edge"] == -0.1
        assert structured["edge_vs_breakeven"] == -0.2
        assert structured["ev_edge_conflict"] is True
        assert structured["conflict_reason"] == "ev and edge_vs_breakeven have opposite signs"
        assert "Expected value and break-even edge disagree" in structured["trading_note"]
        metric_basis = res["summary_structured"]["barriers"]["metric_basis"]
        assert metric_basis["ev"]["definition"].startswith("mean simulated barrier payoff")
        assert metric_basis["probability_edge"] == (
            "take_profit_first_probability minus stop_loss_first_probability"
        )

    def test_no_barriers_section(self):
        sec = _make_full_sections()
        del sec["barriers"]
        res = self._run_report(sec)
        assert "summary" in res


class TestReportWarnings:

    def test_template_warnings_are_captured_in_diagnostics(self):
        fn = _get_report_generate()

        def _warn_template(*args, **kwargs):
            warnings.warn("model convergence warning", RuntimeWarning)
            return _make_report(sections=_make_full_sections())

        with patch("mtdata.core.report_templates.template_basic", _warn_template, create=True), \
             patch("mtdata.core.report_templates.template_advanced", _warn_template, create=True), \
             patch("mtdata.core.report_templates.template_scalping", _warn_template, create=True), \
             patch("mtdata.core.report_templates.template_intraday", _warn_template, create=True), \
             patch("mtdata.core.report_templates.template_swing", _warn_template, create=True), \
             patch("mtdata.core.report_templates.template_position", _warn_template, create=True), \
             patch(_FMT_NUM, side_effect=str):
            res = fn("EURUSD", template="basic", format="toon")

        assert isinstance(res, dict)
        assert "diagnostics" in res
        assert "warnings" in res["diagnostics"]
        assert "model convergence warning" in res["diagnostics"]["warnings"][0]

    def test_library_deprecation_warnings_are_not_user_facing(self):
        fn = _get_report_generate()

        def _warn_template(*args, **kwargs):
            warnings.warn("Importing from torchao.dtypes is deprecated", DeprecationWarning)
            warnings.warn("model convergence warning", RuntimeWarning)
            return _make_report(sections=_make_full_sections())

        with patch("mtdata.core.report_templates.template_basic", _warn_template, create=True), \
             patch("mtdata.core.report_templates.template_advanced", _warn_template, create=True), \
             patch("mtdata.core.report_templates.template_scalping", _warn_template, create=True), \
             patch("mtdata.core.report_templates.template_intraday", _warn_template, create=True), \
             patch("mtdata.core.report_templates.template_swing", _warn_template, create=True), \
             patch("mtdata.core.report_templates.template_position", _warn_template, create=True), \
             patch(_FMT_NUM, side_effect=str):
            res = fn("EURUSD", template="basic", format="toon")

        warnings_out = res["diagnostics"]["warnings"]
        assert warnings_out == ["model convergence warning"]

    def test_flat_forecast_is_flagged_in_summary_and_diagnostics(self):
        fn = _get_report_generate()
        sec = _make_full_sections()
        sec["forecast"] = {
            "method": "sf_autoarima",
            "forecast_price": [65955.1] * 12,
        }
        rep = _make_report(sections=sec)
        mock_basic = MagicMock(return_value=rep)
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True), \
             patch(_FMT_NUM, side_effect=str):
            res = fn("EURUSD", template="basic", format="toon")

        assert any("forecast=sf_autoarima (flat)" in s for s in res.get("summary", []))
        assert "diagnostics" in res
        assert "warnings" in res["diagnostics"]
        assert any("degenerate" in str(w).lower() for w in res["diagnostics"]["warnings"])

    def test_execution_time_metric_is_added_to_diagnostics(self):
        fn = _get_report_generate()
        rep = _make_report(sections=_make_full_sections())
        mock_basic = MagicMock(return_value=rep)
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True), \
             patch(_FMT_NUM, side_effect=str):
            res = fn("EURUSD", template="basic", format="toon")

        assert isinstance(res, dict)
        assert "diagnostics" in res
        assert "execution_time_ms" in res["diagnostics"]
        assert float(res["diagnostics"]["execution_time_ms"]) >= 0.0

    def test_sections_status_summary_is_added(self):
        fn = _get_report_generate()
        rep = _make_report(sections=_make_full_sections())
        mock_basic = MagicMock(return_value=rep)
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True), \
             patch(_FMT_NUM, side_effect=str):
            res = fn("EURUSD", template="basic", format="toon")

        summary = res["sections_status"]["summary"]
        assert summary["ok"] >= 1
        assert summary["total"] == summary["ok"] + summary["partial"] + summary["error"]
        assert res["sections_status"]["sections"]["forecast"] == "ok"
        assert res["overall_assessment"]["section_health"]["total"] == summary["total"]
        assert res["overall_assessment"]["summary"] != "No report sections were available for assessment."
        assert res["success"] is True

    def test_summary_detail_uses_source_section_health(self):
        fn = _get_report_generate()
        rep = _make_report(sections=_make_full_sections())
        mock_basic = MagicMock(return_value=rep)
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True), \
             patch(_FMT_NUM, side_effect=str):
            res = fn("EURUSD", template="basic", detail="summary", format="toon")

        assert res["sections"] == {}
        assert res["section_controls"]["included_count"] == 0
        assert res["section_controls"]["summary_mode"] is True
        assert sorted(res["sections_available"]) == sorted(_make_full_sections().keys())
        assert res["sections_status"]["summary"]["total"] > 0
        assert res["overall_assessment"]["section_health"]["total"] > 0
        assert res["section_run_status"] == "complete"
        assert res["content_detail"] == "summary_only"
        assert res["detail"] == "summary"
        assert "diagnostics" not in res
        assert res["overall_assessment"]["summary"] != (
            "No report sections were available for assessment."
        )

    def test_unknown_section_selection_is_not_reported_as_success(self):
        fn = _get_report_generate()
        rep = _make_report(sections=_make_full_sections())
        mock_basic = MagicMock(return_value=rep)
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch(_FMT_NUM, side_effect=str):
            res = fn(
                "EURUSD",
                template="basic",
                include_sections=["not-a-section"],
                format="toon",
            )

        assert res["success"] is False
        assert res["error_code"] == "report_sections_not_found"
        assert res["invalid_sections"] == ["not-a-section"]
        assert "forecast" in res["valid_sections"]
        assert "overall_assessment" not in res
        mock_basic.assert_not_called()

    @pytest.mark.parametrize(
        "allow_partial",
        [False, True],
    )
    def test_mixed_known_and_unknown_sections_fail_before_execution(
        self,
        allow_partial,
    ):
        fn = _get_report_generate()
        rep = _make_report(sections=_make_full_sections())
        mock_basic = MagicMock(return_value=rep)
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch(_FMT_NUM, side_effect=str):
            res = fn(
                "EURUSD",
                template="basic",
                include_sections=["forecast", "not-a-section"],
                allow_partial=allow_partial,
                format="toon",
            )

        assert res["success"] is False
        assert res["error_code"] == "report_sections_not_found"
        assert res["invalid_sections"] == ["not-a-section"]
        assert "forecast" in res["valid_sections"]
        assert "overall_assessment" not in res
        mock_basic.assert_not_called()

    def test_forecast_selection_runs_dependency_without_summary_leak(self):
        fn = _get_report_generate()
        captured_params: Dict[str, Any] = {}

        def mock_template(_symbol, _horizon, _denoise, params):
            captured_params.update(params)
            return _make_report(
                sections={
                    "backtest": {
                        "best_method": {
                            "method": "theta",
                            "stats": {"avg_rmse": 0.001},
                        }
                    },
                    "forecast": {
                        "method": "theta",
                        "forecast": [
                            {"time": "2026-01-01T01:00Z", "value": 1.101},
                            {"time": "2026-01-01T02:00Z", "value": 1.102},
                            {"time": "2026-01-01T03:00Z", "value": 1.103},
                        ],
                    },
                    "barriers": {"long": {"best": {"ev": 0.2}}},
                }
            )

        with (
            patch("mtdata.core.report_templates.template_basic", mock_template, create=True),
            patch(_FMT_NUM, side_effect=str),
        ):
            res = fn(
                "EURUSD",
                template="basic",
                include_sections=["forecast"],
                allow_stale=True,
                format="toon",
            )

        assert captured_params["_report_execution_sections"] == ["forecast"]
        assert captured_params["allow_stale"] is True
        assert list(res["sections"]) == ["forecast"]
        assert "forecast" in res["summary_structured"]
        assert "backtest" not in res["summary_structured"]
        assert "barriers" not in res["summary_structured"]
        assert res["section_controls"]["included_sections"] == ["forecast"]
        assert "backtest" in res["section_controls"]["omitted_sections"]
        assert "barriers" in res["section_controls"]["omitted_sections"]

    def test_forecast_record_summary_promotes_decision_context(self):
        fn = _get_report_generate()

        def mock_template(_symbol, _horizon, _denoise, _params):
            return {
                "meta": {"template": "minimal"},
                "sections": {
                    "context": {"last_snapshot": {"close": 1.1}},
                    "forecast": {
                        "method": "theta",
                        "last_price_source": "candle_close",
                        "horizon": 3,
                        "forecast": [
                            {"time": "2026-01-01T01:00Z", "value": 1.101},
                            {"time": "2026-01-01T02:00Z", "value": 1.102},
                            {"time": "2026-01-01T03:00Z", "value": 1.103},
                        ],
                        "forecast_vs_last_price": {
                            "direction": "up",
                            "direction_basis": "horizon_end",
                            "horizon_delta": 0.003,
                            "horizon_delta_pct": 0.2727,
                            "direction_actionable": True,
                        },
                        "uncertainty": {
                            "status": "unavailable",
                            "mode": "point_only",
                        },
                    },
                },
            }

        with (
            patch(
                "mtdata.core.report_templates.template_minimal",
                mock_template,
                create=True,
            ),
            patch(_FMT_NUM, side_effect=str),
        ):
            res = fn("EURUSD", template="minimal", detail="compact")

        forecast = res["summary_structured"]["forecast"]
        assert res["summary_structured"]["market"] == {
            "close": 1.1,
            "price_source": "last_completed_candle_close",
        }
        assert forecast["horizon"] == 3
        assert forecast["last_price_source"] == "candle_close"
        assert forecast["terminal_value"] == 1.103
        assert forecast["direction"] == "up"
        assert forecast["direction_actionable"] is True
        assert forecast["horizon_delta_pct"] == pytest.approx(0.2727)
        assert forecast["uncertainty"] == {
            "status": "unavailable",
            "mode": "point_only",
        }
        assert "forecast" not in res.get("sections", {})

    def test_failed_sole_requested_section_cannot_use_dependency_for_success(self):
        fn = _get_report_generate()

        def mock_template(_symbol, _horizon, _denoise, _params):
            return {
                "meta": {"template": "basic"},
                "sections": {
                    "market": {
                        "bid": 1.1,
                        "ask": 1.1002,
                        "spread": 0.0002,
                    }
                },
            }

        with (
            patch(
                "mtdata.core.report_templates.template_intraday",
                mock_template,
                create=True,
            ),
            patch(_FMT_NUM, side_effect=str),
        ):
            res = fn(
                "EURUSD",
                template="intraday",
                include_sections=["execution_gates"],
            )

        assert res["sections"] == {}
        assert res["success"] is False
        assert res["section_run_status"] == "failed"
        assert res["error_code"] == "report_sections_failed"
        assert res["sections_status"]["sections"] == {"execution_gates": "error"}

    def test_temporal_mismatch_is_partial_and_omits_combined_narrative(self):
        fn = _get_report_generate()

        def mock_template(_symbol, _horizon, _denoise, _params):
            return {
                "meta": {"template": "minimal"},
                "sections": {
                    "context": {
                        "last_snapshot": {
                            "time": "2026-06-30T23:00:00Z",
                            "close": 1.1413,
                        },
                    },
                    "forecast": {
                        "method": "theta",
                        "last_observation_time": "2026-06-30T22:00:00Z",
                        "forecast": [
                            {"time": "2026-06-30T23:00:00Z", "value": 1.1418},
                        ],
                    },
                },
            }

        with patch(
            "mtdata.core.report_templates.template_minimal",
            mock_template,
            create=True,
        ):
            res = fn("EURUSD", template="minimal", detail="full")

        assert res["success"] is True
        assert res["section_run_status"] == "partial"
        assert res["as_of"] == "2026-06-30T22:00:00Z"
        assert res["as_of_basis"] == "base_timeframe_last_completed_bar_close"
        assert res["oldest_section_data_as_of"] == "2026-06-30T22:00:00Z"
        assert res["temporal_alignment"] == {
            "status": "mismatch",
            "canonical_as_of": "2026-06-30T22:00:00Z",
            "section_as_of": {
                "context": "2026-06-30T23:00:00Z",
                "forecast": "2026-06-30T22:00:00Z",
            },
            "basis": "context_last_snapshot_vs_forecast_last_bar_open",
            "timestamp_basis": {
                "context": "last_completed_bar_open",
                "forecast": "last_observation_time",
            },
        }
        assert "narrative" not in res["summary_structured"]

    def test_temporal_mismatch_compact_and_strict_name_alignment(self):
        fn = _get_report_generate()
        alignment = {
            "status": "mismatch",
            "canonical_as_of": "2026-06-30T22:00:00Z",
            "section_as_of": {
                "context": "2026-06-30T23:00:00Z",
                "forecast": "2026-06-30T22:00:00Z",
            },
            "basis": "context_last_snapshot_vs_forecast_last_bar_open",
            "timestamp_basis": {
                "context": "last_completed_bar_open",
                "forecast": "last_observation_time",
            },
        }

        def mock_template(_symbol, _horizon, _denoise, _params):
            return {
                "meta": {"template": "minimal"},
                "sections": {
                    "context": {
                        "last_snapshot": {
                            "time": "2026-06-30T23:00:00Z",
                            "close": 1.1413,
                        },
                    },
                    "forecast": {
                        "method": "theta",
                        "last_observation_time": "2026-06-30T22:00:00Z",
                        "forecast": [
                            {"time": "2026-06-30T23:00:00Z", "value": 1.1418},
                        ],
                    },
                },
            }

        with patch(
            "mtdata.core.report_templates.template_minimal",
            mock_template,
            create=True,
        ):
            res = fn(
                "EURUSD",
                template="minimal",
                detail="compact",
                allow_partial=False,
            )

        assert res["success"] is False
        assert res["section_run_status"] == "partial"
        assert res["error_code"] == "report_partial_not_allowed"
        assert res["temporal_alignment"] == alignment
        assert res["details"]["reason"] == "temporal_mismatch"
        assert res["details"]["temporal_alignment"] == alignment
        assert res["details"]["partial_sections"] == []

    def test_minimal_midbar_historical_end_aligns_context_and_forecast(self):
        fn = _get_report_generate()

        def mock_template(_symbol, _horizon, _denoise, params):
            assert params.get("end") == "2026-08-14T12:30:00Z"
            return {
                "meta": {"template": "minimal", "timeframe": "H1"},
                "sections": {
                    "context": {
                        "timeframe": "H1",
                        "last_snapshot": {
                            "time": "2026-08-14T11:00:00Z",
                            "close": 1.1500,
                        },
                    },
                    "forecast": {
                        "method": "theta",
                        "last_observation_time": "2026-08-14T11:00:00Z",
                        "forecast": [
                            {"time": "2026-08-14T12:00:00Z", "value": 1.1510},
                        ],
                    },
                },
            }

        with patch(
            "mtdata.core.report_templates.template_minimal",
            mock_template,
            create=True,
        ):
            res = fn(
                "EURUSD",
                template="minimal",
                timeframe="H1",
                end="2026-08-14T12:30:00Z",
                allow_partial=False,
            )

        assert res["success"] is True
        assert res["section_run_status"] == "complete"
        assert res["as_of"] == "2026-08-14T11:00:00Z"
        assert res["temporal_alignment"]["status"] == "aligned"
        assert res["temporal_alignment"]["section_as_of"] == {
            "context": "2026-08-14T11:00:00Z",
            "forecast": "2026-08-14T11:00:00Z",
        }

    @pytest.mark.parametrize(
        "template_name",
        ["basic", "advanced", "scalping", "intraday", "swing", "position"],
    )
    def test_effective_template_replaces_base_template_metadata(self, template_name):
        fn = _get_report_generate()
        mock_template = MagicMock(
            return_value={
                "meta": {"template": "basic"},
                "sections": {"context": {"last_snapshot": {"close": 1.1}}},
            }
        )
        template_path = f"mtdata.core.report_templates.template_{template_name}"
        with (
            patch(template_path, mock_template, create=True),
            patch(_FMT_NUM, side_effect=str),
        ):
            res = fn(
                "EURUSD",
                template=template_name,
                include_sections=["context"],
            )

        assert res["template"] == template_name
        assert res["meta"]["template"] == template_name
        assert res["executive_summary"]["template"] == template_name


    def test_report_generate_uses_base_timeframe_timestamp_for_as_of(self):
        fn = _get_report_generate()
        sec = _make_full_sections()
        sec["context"]["freshness"] = {"last_observation_time": "2026-06-01T09:30:00Z"}
        sec["forecast"]["last_observation_epoch"] = 1_780_320_000.0
        sec["contexts_multi"]["D1"] = {
            "source_bar_time": "2026-05-31T21:00:00Z"
        }
        rep = _make_report(sections=sec)
        rep["meta"] = {"generated_at": "2026-06-02T12:00:00Z"}
        mock_basic = MagicMock(return_value=rep)
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True), \
             patch(_FMT_NUM, side_effect=str):
            res = fn("EURUSD", template="basic", format="toon")

        assert res["generated_at"] == "2026-06-02T12:00:00Z"
        assert res["as_of"] == "2026-06-01T09:30:00Z"
        assert res["as_of_basis"] == "base_timeframe_last_completed_bar_close"
        assert res["oldest_section_data_as_of"] == "2026-05-31T21:00:00Z"
        assert res["as_of"] != res["generated_at"]


    def test_partial_section_marks_report_partially_complete(self):
        fn = _get_report_generate()
        sec = _make_full_sections()
        sec["barriers"] = {
            "long": {"best": {"tp": 1.5, "sl": 0.8, "edge": 0.3}},
            "short": {"error": "optimizer failed"},
        }
        rep = _make_report(sections=sec)
        mock_basic = MagicMock(return_value=rep)
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True), \
             patch(_FMT_NUM, side_effect=str):
            res = fn("EURUSD", template="basic", format="toon")

        assert res["sections_status"]["sections"]["barriers"] == "partial"
        assert res["sections_status"]["summary"]["partial"] >= 1
        assert res["overall_assessment"]["section_health"]["total"] == res["sections_status"]["summary"]["total"]
        assert res["overall_assessment"]["recommended_action"] == "review_partial_sections"
        assert res["sections_status"]["details"]["barriers"]["errors"][0]["path"] == "short"
        assert "usable data" in res["sections_status"]["definitions"]["partial"]
        assert res["section_run_status"] == "partial"
        assert res["success"] is True

    def test_sections_status_filters_placeholder_error_noise(self):
        fn = _get_report_generate()
        sec = _make_full_sections()
        sec["volatility"] = {
            "summary": {"realized_vol": 0.12},
            "error": "Volatility estimation failed.",
            "estimators": [
                {"error": "no value"},
                {"error": ""},
                {"error": "Volatility estimation failed."},
            ],
        }
        rep = _make_report(sections=sec)
        mock_basic = MagicMock(return_value=rep)
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True), \
             patch(_FMT_NUM, side_effect=str):
            res = fn("EURUSD", template="basic", format="toon")

        errors = res["sections_status"]["details"]["volatility"]["errors"]
        assert {"path": "error", "message": "Volatility estimation failed."} in errors
        assert all(item["message"] != "no value" for item in errors)

    def test_error_section_marks_otherwise_usable_report_partial(self):
        fn = _get_report_generate()
        sec = _make_full_sections()
        sec["forecast"] = {"error": "forecast failed"}
        rep = _make_report(sections=sec)
        mock_basic = MagicMock(return_value=rep)
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True), \
             patch(_FMT_NUM, side_effect=str):
            res = fn("EURUSD", template="basic", format="toon")

        assert res["sections_status"]["sections"]["forecast"] == "error"
        assert res["section_run_status"] == "partial"
        assert res["success"] is True
        assert res["sections_to_retry"] == ["forecast"]

    def test_allow_partial_false_preserves_strict_success_gate(self):
        fn = _get_report_generate()
        sec = _make_full_sections()
        sec["forecast"] = {"error": "forecast failed"}
        mock_basic = MagicMock(return_value=_make_report(sections=sec))
        with patch(
            "mtdata.core.report_templates.template_basic",
            mock_basic,
            create=True,
        ), patch(_FMT_NUM, side_effect=str):
            res = fn(
                "EURUSD",
                template="basic",
                allow_partial=False,
                format="toon",
            )

        assert res["section_run_status"] == "partial"
        assert res["success"] is False
        assert res["error_code"] == "report_partial_not_allowed"
        assert isinstance(res["request_id"], str)
        assert res["details"]["failed_sections"] == ["forecast"]

    def test_all_error_sections_mark_report_failed(self):
        fn = _get_report_generate()
        rep = _make_report(
            sections={
                "forecast": {"error": "forecast failed"},
                "context": {"error": "context failed"},
            }
        )
        mock_basic = MagicMock(return_value=rep)
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch(_FMT_NUM, side_effect=str):
            res = fn("EURUSD", template="basic", format="toon")

        assert res["section_run_status"] == "failed"
        assert res["success"] is False
        assert res["execution_progress"]["completed_sections"] == []
        assert "context" in res["execution_progress"]["failed_sections"]
        assert "forecast" in res["execution_progress"]["failed_sections"]
        narrative = (res.get("summary_structured") or {}).get("narrative")
        assert not narrative or "Forecast method" not in narrative
        assert res["sections_to_retry"] == [
            "context",
            "pivot",
            "contexts_multi",
            "pivot_multi",
            "volatility",
            "backtest",
            "forecast",
            "barriers",
            "patterns",
            "confluence",
        ]

    def test_forecast_section_without_finite_values_is_not_healthy(self):
        from mtdata.core.report.use_cases import _build_sections_status

        status = _build_sections_status(
            {
                "forecast": {
                    "method": "theta",
                    "forecast": [{"time": "2026-01-01T01:00Z", "value": None}],
                }
            }
        )

        assert status["sections"]["forecast"] == "error"
        assert status["details"]["forecast"]["errors"] == [
            {
                "path": "forecast",
                "message": "Forecast section contains no finite forecast values.",
            }
        ]


# ---------------------------------------------------------------------------
# Top-level exception
# ---------------------------------------------------------------------------


class TestReportTopLevelException:

    def test_exception_toon(self):
        fn = _get_report_generate()
        with patch.dict("sys.modules", {"mtdata.core.report_templates": None}):
            # Force a deeper exception
            res = fn("EURUSD", template="basic", format="toon")
        assert isinstance(res, dict)

# ---------------------------------------------------------------------------
# Params passthrough
# ---------------------------------------------------------------------------


class TestReportParams:

    def test_timeframe_in_params(self):
        fn = _get_report_generate()
        rep = _make_report(sections={})
        mock_basic = MagicMock(return_value=rep)
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True), \
             patch(_FMT_NUM, side_effect=str):
            fn("EURUSD", template="basic", timeframe="M15")
        # The params dict passed to template should contain 'timeframe'
        call_args = mock_basic.call_args
        p = call_args[0][3]  # 4th positional arg = params
        assert p.get("timeframe") == "M15"

    def test_methods_in_params(self):
        fn = _get_report_generate()
        rep = _make_report(sections={})
        mock_basic = MagicMock(return_value=rep)
        with patch("mtdata.core.report_templates.template_basic", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_advanced", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_scalping", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_intraday", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_swing", mock_basic, create=True), \
             patch("mtdata.core.report_templates.template_position", mock_basic, create=True), \
             patch(_FMT_NUM, side_effect=str):
            fn("EURUSD", template="basic", methods=["EMA", "ARIMA"])
        call_args = mock_basic.call_args
        p = call_args[0][3]
        assert p.get("methods") == ["EMA", "ARIMA"]


def test_compact_report_payload_includes_temporal_mismatch():
    from mtdata.core.report.use_cases import _compact_report_payload

    alignment = {
        "status": "mismatch",
        "canonical_as_of": "2026-08-14T11:00:00Z",
        "section_as_of": {
            "context": "2026-08-14T10:00:00Z",
            "forecast": "2026-08-14T11:00:00Z",
        },
    }
    out = _compact_report_payload(
        {
            "success": True,
            "section_run_status": "partial",
            "temporal_alignment": alignment,
            "summary_structured": {"market": {"close": 1.15}},
        },
        symbol="EURUSD",
        template="minimal",
    )

    assert out["temporal_alignment"] == alignment


def test_compact_report_payload_retains_as_of():
    from mtdata.core.report.use_cases import _compact_report_payload
    rep = {
        'success': True,
        'timezone': 'UTC',
        'as_of': '2026-06-05T20:00:00Z',
        'as_of_basis': 'base_timeframe_last_completed_bar_close',
        'oldest_section_data_as_of': '2026-06-04T21:00:00Z',
    }
    out = _compact_report_payload(rep, symbol='EURUSD', template='basic')
    assert out['as_of'] == '2026-06-05T20:00:00Z'
    assert out['as_of_basis'] == 'base_timeframe_last_completed_bar_close'
    assert out['oldest_section_data_as_of'] == '2026-06-04T21:00:00Z'


def test_compact_report_payload_retains_generated_at_when_distinct():
    from mtdata.core.report.use_cases import _compact_report_payload
    rep = {
        'success': True,
        'timezone': 'UTC',
        'as_of': '2026-06-05T20:00:00Z',
        'generated_at': '2026-06-05T20:05:00Z',
    }
    out = _compact_report_payload(rep, symbol='EURUSD', template='basic')
    assert out['generated_at'] == '2026-06-05T20:05:00Z'


def test_prioritize_report_payload_orders_as_of_near_top():
    from mtdata.core.report.use_cases import _prioritize_report_payload
    rep = {'sections': {}, 'as_of': 'T', 'success': True, 'timezone': 'UTC'}
    keys = list(_prioritize_report_payload(rep).keys())
    assert keys.index('as_of') < keys.index('sections')


def test_report_assessment_cannot_claim_temporal_coherence_without_as_of():
    from mtdata.core.report.use_cases import _build_overall_report_assessment

    assessment = _build_overall_report_assessment(
        {
            "data_as_of_status": "unavailable",
            "sections_status": {
                "summary": {"total": 3, "ok": 0, "partial": 0, "error": 0, "omitted": 3}
            },
        }
    )

    assert "temporally coherent" not in assessment["summary"]
    assert assessment["temporal_coherence"] == "cannot_assess"
    assert assessment["recommended_action"] == "retry_report"
    assert assessment["assembly_confidence"] == "low"


def test_report_assessment_names_section_health_confidence_explicitly():
    from mtdata.core.report.use_cases import _build_overall_report_assessment

    assessment = _build_overall_report_assessment(
        {
            "sections_status": {
                "summary": {"total": 3, "ok": 3, "partial": 0, "error": 0}
            }
        }
    )

    assert assessment["assembly_confidence"] == "high"
    assert assessment["assembly_confidence_basis"] == "report_section_health"
    assert "confidence" not in assessment
    assert assessment["is_trade_signal"] is False


def test_minimal_report_assessment_recommends_broader_template():
    from mtdata.core.report.use_cases import _build_overall_report_assessment

    assessment = _build_overall_report_assessment(
        {
            "meta": {"template": "minimal"},
            "sections_status": {
                "summary": {"total": 2, "ok": 2, "partial": 0, "error": 0}
            },
        }
    )

    assert assessment["recommended_action"] == (
        "run_basic_template_for_levels_and_risk"
    )
    assert "use template=basic" in assessment["summary"]


@pytest.mark.parametrize(
    ("selected", "completed", "not_requested"),
    [
        (["context"], "Minimal context completed", "Forecast was not requested"),
        (["forecast"], "Minimal forecast completed", "Context was not requested"),
    ],
)
def test_minimal_report_assessment_names_only_selected_sections(
    selected, completed, not_requested,
):
    from mtdata.core.report.use_cases import _build_overall_report_assessment

    assessment = _build_overall_report_assessment(
        {
            "meta": {"template": "minimal"},
            "sections_status": {
                "summary": {"total": 1, "ok": 1, "partial": 0, "error": 0},
                "sections": {selected[0]: "ok"},
            },
            "execution_progress": {"selected_sections": selected},
        }
    )

    assert completed in assessment["summary"]
    assert not_requested in assessment["summary"]


def test_report_data_as_of_prefers_base_timeframe_over_older_sections():
    from mtdata.core.report.use_cases import _derive_report_timestamp_contract

    contract = _derive_report_timestamp_contract(
        {
            "context": {
                "timeframe": "H1",
                "last_snapshot": {"time": "2026-08-18T23:00:00Z"},
            },
            "forecast": {"last_observation_time": "2026-08-18T23:00:00Z"},
            "contexts_multi": {
                "M15": {"source_bar_time": "2026-08-18T23:45:00Z"},
                "H4": {"source_bar_time": "2026-08-18T17:00:00Z"},
                "D1": {"source_bar_time": "2026-08-17T21:00:00Z"},
            },
            "pivot_multi": {
                "H4": {"source_bar_time": "2026-08-18T21:00:00Z"},
            },
        }
    )

    assert contract == {
        "as_of": "2026-08-18T23:00:00Z",
        "as_of_basis": "base_timeframe_last_completed_bar_close",
        "oldest_section_data_as_of": "2026-08-17T21:00:00Z",
    }


def test_report_data_as_of_converts_context_bar_open_to_completed_close():
    from mtdata.core.report.use_cases import _derive_report_timestamp_contract

    contract = _derive_report_timestamp_contract(
        {
            "context": {
                "timeframe": "H1",
                "last_snapshot": {"time": "2026-08-25T19:00:00Z"},
            },
            "forecast": {"last_observation_time": "2026-08-25T20:00:00Z"},
        },
        base_timeframe="H1",
    )

    assert contract["as_of"] == "2026-08-25T20:00:00Z"
    assert contract["as_of_basis"] == "base_timeframe_last_completed_bar_close"


def test_report_data_as_of_labels_oldest_section_fallback():
    from mtdata.core.report.use_cases import _derive_report_timestamp_contract

    assert _derive_report_timestamp_contract(
        {
            "contexts_multi": {
                "H4": {"source_bar_time": "2026-08-18T17:00:00Z"},
                "D1": {"source_bar_time": "2026-08-17T21:00:00Z"},
            }
        }
    ) == {
        "as_of": "2026-08-17T21:00:00Z",
        "as_of_basis": "oldest_selected_section_timestamp",
        "oldest_section_data_as_of": "2026-08-17T21:00:00Z",
    }


def test_report_data_as_of_uses_barrier_lineage_when_it_is_the_only_section():
    from mtdata.core.report.use_cases import _derive_report_timestamp_contract

    assert _derive_report_timestamp_contract(
        {
            "barriers": {
                "long": {
                    "lineage": {
                        "data_as_of": "2026-08-27T17:00:00Z",
                        "timeframe": "H1",
                    }
                }
            }
        }
    ) == {
        "as_of": "2026-08-27T17:00:00Z",
        "as_of_basis": "oldest_selected_section_timestamp",
        "oldest_section_data_as_of": "2026-08-27T17:00:00Z",
    }


def test_report_data_as_of_uses_selected_base_timeframe_context():
    from mtdata.core.report.use_cases import _derive_report_timestamp_contract

    assert _derive_report_timestamp_contract(
        {
            "contexts_multi": {
                "H4": {"source_bar_time": "2026-08-18T17:00:00Z"},
                "D1": {"source_bar_time": "2026-08-17T21:00:00Z"},
            }
        },
        base_timeframe="H4",
    ) == {
        "as_of": "2026-08-18T21:00:00Z",
        "as_of_basis": "base_timeframe_last_completed_bar_close",
        "oldest_section_data_as_of": "2026-08-17T21:00:00Z",
    }


def test_report_temporal_alignment_rejects_stale_multitimeframe_context():
    from mtdata.core.report.use_cases import _report_temporal_alignment

    alignment = _report_temporal_alignment(
        {
            "context": {
                "timeframe": "H1",
                "last_snapshot": {"time": "2026-07-31T20:00:00Z"},
            },
            "forecast": {"last_observation_time": "2026-07-31T20:00:00Z"},
            "contexts_multi": {
                "M15": {"source_bar_time": "2026-07-03T01:45:00Z"},
                "H4": {"source_bar_time": "2026-07-31T17:00:00Z"},
                "D1": {"source_bar_time": "2026-07-30T21:00:00Z"},
            },
        }
    )

    assert alignment is not None
    assert alignment["status"] == "mismatch"
    assert alignment["mismatched_sections"] == ["contexts_multi.M15"]
    assert alignment["section_as_of"]["contexts_multi"]["H4"] == (
        "2026-07-31T17:00:00Z"
    )


def test_report_temporal_alignment_accepts_timeframe_aware_source_offsets():
    from mtdata.core.report.use_cases import _report_temporal_alignment

    alignment = _report_temporal_alignment(
        {
            "context": {
                "timeframe": "H1",
                "last_snapshot": {"time": "2026-07-31T20:00:00Z"},
            },
            "forecast": {"last_observation_time": "2026-07-31T20:00:00Z"},
            "contexts_multi": {
                "M15": {"source_bar_time": "2026-07-31T20:45:00Z"},
                "H4": {"source_bar_time": "2026-07-31T17:00:00Z"},
                "D1": {"source_bar_time": "2026-07-30T21:00:00Z"},
            },
        }
    )

    assert alignment is not None
    assert alignment["status"] == "aligned"
    assert alignment["mismatched_sections"] == []


def test_report_temporal_alignment_uses_forecast_bar_open_not_close():
    from mtdata.core.report.use_cases import _report_temporal_alignment

    alignment = _report_temporal_alignment(
        {
            "context": {
                "timeframe": "H1",
                "last_snapshot": {"time": "2026-08-25T15:00:00Z"},
            },
            "forecast": {
                "last_bar_open": "2026-08-25T15:00:00Z",
                "last_observation_time": "2026-08-25T16:00:00Z",
                "data_window": {"last_bar_open": "2026-08-25T15:00:00Z"},
            },
        }
    )

    assert alignment is not None
    assert alignment["status"] == "aligned"
    assert alignment["section_as_of"] == {
        "context": "2026-08-25T15:00:00Z",
        "forecast": "2026-08-25T15:00:00Z",
    }
    assert alignment["timestamp_basis"]["forecast"] == "last_bar_open"


def test_report_assessment_elevates_closed_session_freshness():
    from mtdata.core.report.use_cases import _build_overall_report_assessment

    assessment = _build_overall_report_assessment(
        {
            "sections_status": {
                "summary": {"total": 2, "ok": 2, "partial": 0, "error": 0}
            },
            "sections": {
                "context": {
                    "freshness": {"market_status": "closed", "data_stale": True}
                },
                "forecast": {"last_observation_stale": True},
            },
        }
    )

    assert assessment["recommended_action"] == (
        "review_stale_or_closed_session_data"
    )
    assert assessment["data_trust"] == {
        "status": "closed_session",
        "affected_sections": ["context", "forecast"],
    }
    assert "closed-session data" in assessment["summary"]


def test_report_assessment_reads_public_forecast_stale_flags():
    from mtdata.core.report.use_cases import _build_overall_report_assessment

    assessment = _build_overall_report_assessment(
        {
            "sections_status": {
                "summary": {"total": 2, "ok": 2, "partial": 0, "error": 0}
            },
            "sections": {
                "context": {
                    "data_stale": True,
                    "market_status": "closed",
                },
                "forecast": {
                    "last_price_stale": True,
                    "freshness": "closed weekend, anchor 5h ago",
                },
            },
        }
    )

    assert assessment["data_trust"]["status"] == "closed_session"
    assert assessment["data_trust"]["affected_sections"] == ["context", "forecast"]


@pytest.mark.parametrize("horizon", [0, -1])
def test_report_request_rejects_nonpositive_horizon(horizon):
    from mtdata.core.report.requests import ReportGenerateRequest

    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        ReportGenerateRequest(symbol="EURUSD", horizon=horizon)


def test_report_request_rejects_nonpositive_params_horizon():
    from mtdata.core.report.requests import ReportGenerateRequest

    with pytest.raises(ValidationError, match="params.horizon"):
        ReportGenerateRequest(symbol="EURUSD", params={"horizon": 0})


def test_compact_summary_structured_drops_duplicate_aliases():
    from mtdata.core.report.use_cases import _compact_summary_structured

    compact = _compact_summary_structured(
        {
            "narrative": "Last close 1.16743.",
            "levels": [{"price": 1.16641, "role": "below"}],
            "confluence": {
                "reference_price": 1.16743,
                "levels": [{"price": 1.16641, "role": "below"}],
            },
            "patterns": {"recent": [{"name": "engulfing"}]},
            "structure": {"patterns": [{"name": "engulfing"}]},
            "volatility": {"method": "ewma"},
            "barriers": {"up": {"price": 1.17}},
            "risk": {
                "volatility": {"method": "ewma"},
                "barriers": {"up": {"price": 1.17}},
            },
        }
    )
    assert compact["levels"][0]["price"] == 1.16641
    assert compact["patterns"]["recent"][0]["name"] == "engulfing"
    assert "patterns" not in compact.get("structure", {})
    assert compact["volatility"]["method"] == "ewma"
    assert "volatility" not in compact.get("risk", {})
    assert "barriers" not in compact.get("risk", {})


def test_capped_basic_report_assessment_does_not_claim_missing_sections():
    from mtdata.core.report.use_cases import _build_overall_report_assessment

    assessment = _build_overall_report_assessment(
        {
            "template": "basic",
            "sections_status": {
                "summary": {"ok": 3, "partial": 0, "error": 0, "omitted": 0, "total": 3},
                "sections": {
                    "context": {"status": "ok"},
                    "pivot": {"status": "ok"},
                    "contexts_multi": {"status": "ok"},
                },
            },
            "execution_progress": {
                "requested_sections": [
                    "context",
                    "pivot",
                    "contexts_multi",
                    "forecast",
                    "barriers",
                ],
                "selected_sections": ["context", "pivot", "contexts_multi"],
                "capped_requested_sections": ["forecast", "barriers"],
            },
            "sections": {
                "context": {"close": 1.1},
                "pivot": {"pivot": 1.1},
                "contexts_multi": {},
            },
        }
    )

    assert assessment["assembly_confidence"] == "limited"
    assert assessment["coverage_status"] == "limited_by_max_sections"
    assert assessment["section_health"]["intentionally_omitted"] == 2
    assert "forecast" not in assessment["summary"].lower() or "excluded" in assessment["summary"].lower()
    assert "risk context" not in assessment["summary"]


def test_style_template_timeframe_compatibility_ranges():
    from mtdata.core.report.requests import template_timeframe_compatibility

    rejected = template_timeframe_compatibility("scalping", "W1")
    assert rejected is not None
    assert rejected["action"] == "reject"
    warned = template_timeframe_compatibility("scalping", "H4")
    assert warned is not None
    assert warned["action"] == "warn"
    assert warned["code"] == "template_timeframe_warning"
    position = template_timeframe_compatibility("position", "M1")
    assert position is not None
    assert position["action"] == "reject"
    compatible = template_timeframe_compatibility("intraday", "H1")
    assert compatible is None


def test_report_max_runtime_help_describes_advisory_estimates():
    from mtdata.core.param_help import COMMAND_PARAM_HELP_OVERRIDES

    help_text = COMMAND_PARAM_HELP_OVERRIDES[("report_generate", "max_runtime")]
    assert "advisory" in help_text.lower()
    assert "estimated cost does not fit are omitted" not in help_text.lower()
