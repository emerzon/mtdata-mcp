from __future__ import annotations

from typing import Any, Dict

import pytest

import mtdata.core.trading.ideas as ideas_module
from mtdata.core.trading.ideas import run_trade_idea_compose
from mtdata.core.trading.ideas_requests import TradeIdeaComposeRequest


def _session(*, usable: bool = True, tradable: bool = True) -> Dict[str, Any]:
    return {
        "success": True,
        "symbol": "EURUSD",
        "is_tradable": tradable,
        "can_open_new_positions": tradable,
        "quote": {
            "symbol": "EURUSD",
            "bid": 1.1000,
            "ask": 1.1002,
            "mid": 1.1001,
            "spread": 0.0002,
            "usable_for_live_trading": usable,
            "data_stale": not usable,
            "execution_blockers": [] if usable else ["quote_not_live_ready"],
        },
        "source": {"provider": "mt5"},
    }


def _forecast(*, trend: str = "up") -> Dict[str, Any]:
    values = [1.1001, 1.1004, 1.1008] if trend == "up" else [1.1001, 1.0998, 1.0994]
    if trend == "flat":
        values = [1.1001, 1.1001, 1.1001]
    direction = "bullish" if trend == "up" else "bearish" if trend == "down" else "neutral"
    return {
        "success": True,
        "method": "theta",
        "library": "native",
        "quantity": "price",
        "horizon": 12,
        "forecast_price": values,
        "forecast_time": [
            "2026-08-01T21:00:00Z",
            "2026-08-02T21:00:00Z",
            "2026-08-03T21:00:00Z",
        ],
        "forecast_bar_states": ["future", "future", "future"],
        "last_observation_time": "2026-07-31T20:00:00Z",
        "last_price": 1.1001,
        "last_price_source": "candle_close",
        "data_window": {
            "start": "2026-06-01T20:00:00Z",
            "end": "2026-07-31T20:00:00Z",
            "bars_used": 1000,
            "input_bar_policy": "closed_bars_only",
        },
        "trend": trend,
        "forecast_vs_last_price": {
            "direction": direction,
            "direction_basis": "horizon_end",
            "direction_actionable": trend != "flat",
            "direction_status": "interval_confirmed" if trend != "flat" else "neutral",
        },
    }


def _volatility() -> Dict[str, Any]:
    return {
        "success": True,
        "method": "ewma",
        "horizon": 12,
        "volatility_per_bar": 0.0006,
        "volatility_horizon": 0.0021,
        "volatility_annualized": 0.0474,
        "volatility_unit": "return_fraction",
        "bars_per_year": 6240.0,
        "annualization_basis": "260_fx_weekdays_24h",
        "data_as_of": "2026-07-31T20:00:00Z",
        "data_window": {
            "start": "2026-06-01T20:00:00Z",
            "end": "2026-07-31T20:00:00Z",
            "bars_used": 1000,
        },
    }


def _barriers(*, tp_first: float = 0.42, sl_first: float = 0.31) -> Dict[str, Any]:
    return {
        "success": True,
        "method": "mc_gbm_bb",
        "direction": "long",
        "horizon": 12,
        "reference_price": 1.1002,
        "last_price": 1.1002,
        "last_price_source": "candle_close",
        "data_as_of": "2026-07-31T20:00:00Z",
        "data_window": {
            "start": "2026-06-01T20:00:00Z",
            "end": "2026-07-31T20:00:00Z",
            "bars_used": 1000,
        },
        "tp_pct": 0.40,
        "sl_pct": 0.60,
        "tp_price": 1.104602,
        "sl_price": 1.0935988,
        "prob_tp_first": tp_first,
        "prob_sl_first": sl_first,
        "prob_no_hit": round(1.0 - tp_first - sl_first, 4),
        "probability_edge": tp_first - sl_first,
    }


def _confluence() -> Dict[str, Any]:
    return {
        "success": True,
        "analysis_as_of": "2026-07-31T20:00:00Z",
        "reference_price": 1.1001,
        "reference_price_source": "historical_window_close",
        "data_window": {
            "start": "2026-06-01T20:00:00Z",
            "end": "2026-07-31T20:00:00Z",
        },
        "levels": [
            {
                "type": "resistance",
                "price": 1.1044,
                "score": 4.2,
                "range": {"low": 1.1042, "high": 1.1046, "width": 0.0004},
            },
            {
                "type": "support",
                "price": 1.0938,
                "score": 3.1,
                "range": {"low": 1.0936, "high": 1.0940, "width": 0.0004},
            },
        ],
    }


def _sizing(*, volume: float = 0.12) -> Dict[str, Any]:
    return {
        "success": True,
        "candidate_valid": True,
        "position_sizing": {
            "suggested_volume": volume,
            "candidate_valid": True,
            "requested_risk_pct": 0.5,
            "status": "ok",
            "entry": 1.1002,
            "sl": 1.0935988,
            "tp": 1.104602,
        },
    }


def _preview(*, preview_ok: bool = True) -> Dict[str, Any]:
    return {
        "success": True,
        "dry_run": True,
        "preview_ok": preview_ok,
        "would_send_order": False,
        "actionability": "preview_only",
        "blockers": [] if preview_ok else ["quote_not_live_ready"],
        "validation": {"live_submission_eligible": preview_ok},
        "guardrails_preview": {"enabled": True, "blocked": False},
    }


def _caller(mapping: Dict[str, Any]):
    def _call(name: str, kwargs: Dict[str, Any]) -> Any:
        payload = mapping.get(name)
        if callable(payload):
            return payload(kwargs)
        if payload is None:
            return {"success": False, "error": f"missing section {name}"}
        return payload

    return _call


def test_trade_idea_compose_quick_preview_path() -> None:
    idea = run_trade_idea_compose(
        TradeIdeaComposeRequest(symbol="EURUSD"),
        call_section=_caller(
            {
                "session": _session(),
                "forecast": _forecast(),
                "volatility": _volatility(),
                "barriers": _barriers(),
                "sizing": _sizing(),
                "preview": _preview(),
            }
        ),
    )

    assert idea["success"] is True
    assert idea["direction"] == "long"
    assert idea["direction_basis"] == "forecast_vs_last_price"
    assert idea["suggested_direction"] == "long"
    assert idea["actionability"] == "preview_only"
    assert idea["preview"]["dry_run"] is True
    assert idea["preview"]["preview_ok"] is True
    assert idea["preview"]["would_send_order"] is False
    assert idea["sizing"]["suggested_volume"] == 0.12
    assert idea["geometry"]["take_profit"] == pytest.approx(1.104602)
    assert idea["geometry"]["stop_loss"] == pytest.approx(1.0935988)
    assert idea["gates"]["preview"]["status"] == "pass"
    assert idea["idea_eligible"] is True
    assert idea["overall_gate_status"] == "pass"
    assert idea["data_as_of"] == "2026-07-31T20:00:00Z"
    assert idea["as_of"] == idea["assembled_at"]
    assert "source_tool_calls" not in idea
    assert "not an order" in idea["narrative"]


def test_trade_idea_auto_direction_uses_calibrated_interval_vs_live_quote() -> None:
    forecast = _forecast()
    forecast.update(
        {
            "interval_usage": "calibrated",
            "calibration_sufficient": True,
            "forecast": [
                {"value": 1.1008, "lower": 1.1003, "upper": 1.1012},
            ],
        }
    )

    idea = run_trade_idea_compose(
        TradeIdeaComposeRequest(symbol="EURUSD"),
        call_section=_caller(
            {
                "session": _session(),
                "forecast": forecast,
                "volatility": _volatility(),
                "barriers": _barriers(),
                "sizing": _sizing(),
                "preview": _preview(),
            }
        ),
    )

    assert idea["direction"] == "long"
    assert idea["direction_basis"] == "forecast_vs_live_quote"
    live_context = idea["forecast"]["forecast_vs_live_quote"]
    assert live_context["direction"] == "bullish"
    assert live_context["direction_actionable"] is True
    assert live_context["direction_interval_excludes_live_quote"] is True
    assert live_context["live_ask"] == 1.1002
    assert live_context["horizon_lower_price"] == 1.1003


def test_trade_idea_stands_down_when_live_quote_is_inside_forecast_interval() -> None:
    forecast = _forecast()
    forecast.update(
        {
            "interval_usage": "calibrated",
            "calibration_sufficient": True,
            "forecast": [
                {"value": 1.1008, "lower": 1.0999, "upper": 1.1003},
            ],
        }
    )

    idea = run_trade_idea_compose(
        TradeIdeaComposeRequest(symbol="EURUSD"),
        call_section=_caller(
            {
                "session": _session(),
                "forecast": forecast,
                "volatility": _volatility(),
            }
        ),
    )

    assert idea["direction"] == "stand_down"
    assert idea["direction_basis"] == "gate_outcome"
    assert "suggested_direction" not in idea
    assert idea["forecast"]["last_price_source"] == "candle_close"
    live_context = idea["forecast"]["forecast_vs_live_quote"]
    assert live_context["direction"] == "neutral"
    assert live_context["direction_actionable"] is False
    assert live_context["direction_suppressed_reason"] == (
        "horizon_interval_contains_live_quote"
    )
    assert idea["gates"]["alignment"]["status"] == "fail"


@pytest.mark.parametrize(
    ("actionable", "expected_direction", "expect_downstream"),
    [(True, "long", True), (False, "stand_down", False)],
)
def test_trade_idea_default_auto_wiring_uses_calibrated_intervals(
    monkeypatch,
    actionable: bool,
    expected_direction: str,
    expect_downstream: bool,
) -> None:
    calls: list[str] = []
    conformal_requests: list[Any] = []

    forecast = _forecast()
    forecast.update(
        {
            "interval_method": "rolling_residual_quantiles",
            "ci_alpha": 0.05,
            "ci_status": "available" if actionable else "insufficient_calibration",
            "ci_available": actionable,
            "required_calibration_points": 30,
            "calibration_sufficient": actionable,
            "interval_usage": "calibrated" if actionable else "diagnostic_only",
            "conformal": {
                "calibration_steps": 50,
                "calibration_spacing": 20,
                "min_calibration_points": 50 if actionable else 12,
                "required_calibration_points": 30,
                "calibration_sufficient": actionable,
                "empirical_coverage": 0.94,
                "coverage_target": 0.95,
                "interval_usage": "calibrated" if actionable else "diagnostic_only",
            },
        }
    )
    if actionable:
        forecast["forecast_vs_last_price"].update(
            {
                "direction_interval_excludes_last_price": True,
                "direction_interval_basis": "horizon_interval_vs_last_price",
                "direction_interpretation": "interval_excludes_last_price",
            }
        )
    else:
        forecast["forecast_vs_last_price"] = {
            "point_estimate_direction": "bullish",
            "direction_actionable": False,
            "direction_status": "unconfirmed",
            "direction_suppressed_reason": "forecast_uncertainty_not_available",
            "direction_interval_excludes_last_price": None,
            "direction_interval_basis": "not_available",
        }

    def fake_call_tool(tool, **kwargs):
        name = tool.__name__
        calls.append(name)
        if name == "trade_session_context":
            return _session()
        if name == "forecast_conformal_intervals":
            conformal_requests.append(kwargs["request"])
            return forecast
        if name == "forecast_volatility_estimate":
            return _volatility()
        if name == "forecast_barrier_prob":
            return _barriers()
        if name == "trade_risk_analyze":
            return _sizing()
        if name == "trade_place":
            return _preview()
        raise AssertionError(f"unexpected default section tool: {name}")

    monkeypatch.setattr(ideas_module, "call_tool_sync_structured", fake_call_tool)

    idea = run_trade_idea_compose(TradeIdeaComposeRequest(symbol="EURUSD"))

    assert idea["direction"] == expected_direction
    assert len(conformal_requests) == 1
    request = conformal_requests[0]
    assert request.method == "theta"
    assert request.steps == 50
    assert request.spacing == 20
    assert request.ci_alpha == 0.05
    assert ("forecast_barrier_prob" in calls) is expect_downstream
    if actionable:
        assert idea["forecast"]["interval_method"] == "rolling_residual_quantiles"
        assert idea["forecast"]["calibration"]["min_calibration_points"] == 50
        assert (
            idea["forecast"]["forecast_vs_last_price"]["direction_interval_basis"]
            == "horizon_interval_vs_last_price"
        )


def test_trade_idea_compose_stands_down_on_stale_quote() -> None:
    idea = run_trade_idea_compose(
        TradeIdeaComposeRequest(symbol="EURUSD", direction="long"),
        call_section=_caller(
            {
                "session": _session(usable=False),
                "forecast": _forecast(),
                "volatility": _volatility(),
                "barriers": _barriers(),
            }
        ),
    )

    assert idea["direction"] == "stand_down"
    assert idea["requested_direction"] == "long"
    assert idea["evaluated_direction"] == "long"
    assert idea["action"] == "stand_down"
    assert idea["direction_basis"] == "gate_outcome"
    assert idea["actionability"] == "research"
    assert idea["sizing"]["suggested_volume"] == 0.0
    assert idea["preview"]["preview_ok"] is False
    assert idea["gates"]["quote_fresh"]["status"] == "fail"
    assert "sizing" not in idea.get("failed_sections", [])


def test_trade_idea_compose_auto_stands_down_when_barriers_disagree() -> None:
    idea = run_trade_idea_compose(
        TradeIdeaComposeRequest(symbol="EURUSD", direction="auto"),
        call_section=_caller(
            {
                "session": _session(),
                "forecast": _forecast(trend="up"),
                "volatility": _volatility(),
                "barriers": _barriers(tp_first=0.20, sl_first=0.55),
            }
        ),
    )

    assert idea["direction"] == "stand_down"
    assert idea["suggested_direction"] == "long"
    assert idea["gates"]["alignment"]["status"] == "fail"
    assert idea["actionability"] == "research"
    assert idea["sizing"]["suggested_volume"] == 0.0


@pytest.mark.parametrize(("direction", "trend"), [("long", "up"), ("short", "down")])
def test_trade_idea_compose_explicit_direction_fails_closed_when_barriers_are_weak(
    direction: str,
    trend: str,
) -> None:
    barriers = _barriers(tp_first=0.20, sl_first=0.55)
    if direction == "short":
        barriers.update(
            {
                "direction": "short",
                "tp_price": 1.0957992,
                "sl_price": 1.1068012,
            }
        )
    idea = run_trade_idea_compose(
        TradeIdeaComposeRequest(symbol="EURUSD", direction=direction),
        call_section=_caller(
            {
                "session": _session(),
                "forecast": _forecast(trend=trend),
                "volatility": _volatility(),
                "barriers": barriers,
                "sizing": _sizing(),
                "preview": _preview(),
            }
        ),
    )

    assert idea["direction"] == "stand_down"
    assert idea["gates"]["barriers"]["status"] == "fail"
    assert idea["actionability"] == "research"
    assert idea["idea_eligible"] is False
    assert idea["overall_gate_status"] == "fail"
    assert idea["sizing"]["suggested_volume"] == 0.0
    assert idea["preview"]["preview_ok"] is False
    assert idea["preview"]["live_submission_eligible"] is False


def test_trade_idea_compose_explicit_direction_fails_closed_on_alignment_gate() -> None:
    idea = run_trade_idea_compose(
        TradeIdeaComposeRequest(symbol="EURUSD", direction="long"),
        call_section=_caller(
            {
                "session": _session(),
                "forecast": _forecast(trend="flat"),
                "volatility": _volatility(),
                "barriers": _barriers(),
                "sizing": _sizing(),
                "preview": _preview(),
            }
        ),
    )

    assert idea["direction"] == "stand_down"
    assert idea["gates"]["alignment"]["status"] == "fail"
    assert idea["gates"]["barriers"]["status"] == "pass"
    assert idea["idea_eligible"] is False
    assert idea["sizing"]["suggested_volume"] == 0.0
    assert idea["preview"]["preview_ok"] is False


def test_trade_idea_compose_historical_skips_preview() -> None:
    calls: list[str] = []

    def _tracking(name: str, kwargs: Dict[str, Any]) -> Any:
        calls.append(name)
        return {
            "session": _session(),
            "forecast": _forecast(),
            "volatility": _volatility(),
            "barriers": _barriers(),
        }[name]

    idea = run_trade_idea_compose(
        TradeIdeaComposeRequest(
            symbol="EURUSD",
            as_of="2026-08-01",
            detail="full",
        ),
        call_section=_tracking,
    )

    assert "session" not in calls
    assert "sizing" not in calls
    assert "preview" not in calls
    assert "quote" not in idea
    assert idea["actionability"] == "research"
    assert idea["preview"]["skipped"] is True
    assert idea["gates"]["preview"]["status"] == "skip"
    assert idea["requested_as_of"] == "2026-08-01"
    assert idea["as_of"] == "2026-07-31T20:00:00Z"
    assert idea["data_as_of"] == "2026-07-31T20:00:00Z"
    assert idea["idea_eligible"] is False
    assert idea["overall_gate_status"] == "research_only"
    assert idea["forecast"]["points"][-1]["time"] == "2026-08-03T21:00:00Z"


def test_historical_standard_idea_keeps_structure_lineage() -> None:
    idea = run_trade_idea_compose(
        TradeIdeaComposeRequest(
            symbol="EURUSD",
            template="standard",
            as_of="2026-08-01",
            detail="full",
        ),
        call_section=_caller(
            {
                "confluence": _confluence(),
                "forecast": _forecast(),
                "volatility": _volatility(),
                "barriers": _barriers(),
            }
        ),
    )

    assert idea["lineage"]["structure"] == {
        "data_as_of": "2026-07-31T20:00:00Z",
        "data_window": {
            "start": "2026-06-01T20:00:00Z",
            "end": "2026-07-31T20:00:00Z",
        },
        "price_anchor": {
            "value": 1.1001,
            "source": "historical_window_close",
        },
    }


def test_trade_idea_compose_stands_down_on_unconfirmed_direction() -> None:
    forecast = _forecast(trend="down")
    forecast["forecast_price"] = [1.15787, 1.15780, 1.15775]
    forecast["forecast_vs_last_price"] = {
        "direction": "neutral",
        "direction_basis": "horizon_end",
        "direction_actionable": False,
        "direction_status": "neutral",
    }

    idea = run_trade_idea_compose(
        TradeIdeaComposeRequest(symbol="EURUSD"),
        call_section=_caller(
            {
                "session": _session(),
                "forecast": forecast,
                "volatility": _volatility(),
            }
        ),
    )

    assert idea["direction"] == "stand_down"
    assert idea["direction_basis"] == "gate_outcome"
    assert "suggested_direction" not in idea
    assert "trend" not in idea["forecast"]
    assert "tp_pct" not in idea.get("barriers", {})
    assert "sl_pct" not in idea.get("barriers", {})
    assert idea["gates"]["alignment"]["status"] == "fail"
    assert idea["sizing"]["suggested_volume"] == 0.0


def test_trade_idea_compose_standard_snaps_to_confluence() -> None:
    idea = run_trade_idea_compose(
        TradeIdeaComposeRequest(symbol="EURUSD", template="standard", direction="long"),
        call_section=_caller(
            {
                "session": _session(),
                "confluence": _confluence(),
                "forecast": _forecast(),
                "volatility": _volatility(),
                "barriers": _barriers(),
                "sizing": _sizing(),
                "preview": _preview(),
            }
        ),
    )

    assert idea["structure"]["levels"]
    snaps = idea["barriers"]["snapped_to_structure"]
    kinds = {row["kind"] for row in snaps}
    assert kinds == {"take_profit", "stop_loss"}
    assert idea["geometry"]["take_profit"] == pytest.approx(1.1044)
    assert idea["geometry"]["stop_loss"] == pytest.approx(1.0938)


def test_trade_idea_compose_never_accepts_live_preview() -> None:
    idea = run_trade_idea_compose(
        TradeIdeaComposeRequest(symbol="EURUSD", direction="long"),
        call_section=_caller(
            {
                "session": _session(),
                "forecast": _forecast(),
                "volatility": _volatility(),
                "barriers": _barriers(),
                "sizing": _sizing(),
                "preview": {
                    "success": True,
                    "dry_run": False,
                    "preview_ok": True,
                    "would_send_order": True,
                },
            }
        ),
    )

    assert idea["direction"] == "stand_down"
    assert idea["preview"]["dry_run"] is True
    assert idea["preview"]["preview_ok"] is False
    assert idea["gates"]["preview"]["status"] == "fail"
    assert idea["sizing"]["suggested_volume"] == 0.0


def test_trade_idea_compose_symbol_not_found_fails_closed() -> None:
    idea = run_trade_idea_compose(
        TradeIdeaComposeRequest(symbol="NOPE"),
        call_section=_caller(
            {
                "session": {
                    "success": False,
                    "error": "Symbol 'NOPE' was not found.",
                    "error_code": "symbol_not_found",
                }
            }
        ),
    )

    assert idea["success"] is False
    assert idea["error_code"] == "symbol_not_found"


def test_trade_idea_compose_full_detail_keeps_source_calls() -> None:
    idea = run_trade_idea_compose(
        TradeIdeaComposeRequest(symbol="EURUSD", detail="full"),
        call_section=_caller(
            {
                "session": _session(),
                "forecast": _forecast(),
                "volatility": _volatility(),
                "barriers": _barriers(),
                "sizing": _sizing(),
                "preview": _preview(),
            }
        ),
    )

    names = [row["name"] for row in idea["source_tool_calls"]]
    assert names == ["session", "forecast", "volatility", "barriers", "sizing", "preview"]
    assert idea["forecast"]["last_observation_time"] == "2026-07-31T20:00:00Z"
    assert idea["forecast"]["last_price"] == 1.1001
    assert idea["forecast"]["last_price_source"] == "candle_close"
    assert idea["forecast"]["points"][0] == {
        "time": "2026-08-01T21:00:00Z",
        "value": 1.1001,
        "value_semantics": "forecast_price",
        "bar_state": "future",
    }
    assert idea["lineage"]["forecast"]["target_window"] == {
        "start": "2026-08-01T21:00:00Z",
        "end": "2026-08-03T21:00:00Z",
        "bars": 3,
        "time_semantics": "bar_time",
        "value_semantics": "forecast_price",
    }
    assert idea["lineage"]["volatility"]["data_as_of"] == (
        "2026-07-31T20:00:00Z"
    )
    assert idea["volatility"]["bars_per_year"] == 6240.0
    assert idea["volatility"]["annualization_basis"] == "260_fx_weekdays_24h"
    assert idea["volatility"]["volatility_annualized"] == 0.0474
    assert idea["lineage"]["barriers"]["price_anchor"] == {
        "value": 1.1002,
        "source": "candle_close",
    }


def test_trade_idea_compose_partial_failure_when_volatility_fails() -> None:
    idea = run_trade_idea_compose(
        TradeIdeaComposeRequest(symbol="EURUSD", direction="long"),
        call_section=_caller(
            {
                "session": _session(),
                "forecast": _forecast(),
                "volatility": {"success": False, "error": "ewma unavailable"},
                "barriers": _barriers(),
                "sizing": _sizing(),
                "preview": _preview(),
            }
        ),
    )

    assert idea["success"] is True
    assert idea["partial_failure"] is True
    assert "volatility" in idea["failed_sections"]
    assert idea["direction"] == "long"


def test_trade_idea_forced_direction_uses_point_forecast_generate(monkeypatch) -> None:
    generate_requests: list[Any] = []

    def fake_call_tool(tool, **kwargs):
        name = tool.__name__
        if name == "trade_session_context":
            return _session()
        if name == "forecast_conformal_intervals":
            raise AssertionError("explicit direction must not use conformal calibration")
        if name == "forecast_generate":
            generate_requests.append(kwargs["request"])
            return _forecast()
        if name == "forecast_volatility_estimate":
            return _volatility()
        if name == "forecast_barrier_prob":
            return _barriers()
        if name == "trade_risk_analyze":
            return _sizing()
        if name == "trade_place":
            return _preview()
        raise AssertionError(f"unexpected default section tool: {name}")

    monkeypatch.setattr(ideas_module, "call_tool_sync_structured", fake_call_tool)

    idea = run_trade_idea_compose(
        TradeIdeaComposeRequest(symbol="EURUSD", direction="long")
    )

    assert len(generate_requests) == 1
    assert idea["direction"] == "long"
    assert idea["direction_basis"] == "requested"


def test_trade_idea_auto_direction_still_uses_conformal_forecast(monkeypatch) -> None:
    conformal_requests: list[Any] = []

    def fake_call_tool(tool, **kwargs):
        name = tool.__name__
        if name == "trade_session_context":
            return _session()
        if name == "forecast_conformal_intervals":
            conformal_requests.append(kwargs["request"])
            forecast = _forecast()
            forecast.update(
                {
                    "interval_method": "rolling_residual_quantiles",
                    "ci_alpha": 0.05,
                    "ci_status": "available",
                    "ci_available": True,
                }
            )
            return forecast
        if name == "forecast_generate":
            raise AssertionError("auto direction must use conformal intervals")
        if name == "forecast_volatility_estimate":
            return _volatility()
        if name == "forecast_barrier_prob":
            return _barriers()
        if name == "trade_risk_analyze":
            return _sizing()
        if name == "trade_place":
            return _preview()
        raise AssertionError(f"unexpected default section tool: {name}")

    monkeypatch.setattr(ideas_module, "call_tool_sync_structured", fake_call_tool)

    idea = run_trade_idea_compose(
        TradeIdeaComposeRequest(symbol="EURUSD", direction="auto")
    )

    assert len(conformal_requests) == 1
    assert conformal_requests[0].steps == 50
    assert idea["forecast"]["interval_method"] == "rolling_residual_quantiles"


def test_trade_idea_rejects_future_as_of_before_sections() -> None:
    idea = run_trade_idea_compose(
        TradeIdeaComposeRequest(
            symbol="EURUSD",
            as_of="2099-01-01",
        ),
        call_section=lambda name, kwargs: (_ for _ in ()).throw(
            AssertionError(f"section {name} must not run")
        ),
    )

    assert idea["success"] is False
    assert idea["error_code"] == "trade_idea_as_of_in_future"
    assert "future" in idea["error"].lower()


def test_trade_idea_barrier_widths_scale_from_horizon_volatility() -> None:
    captured: Dict[str, Any] = {}

    def call(name: str, kwargs: Dict[str, Any]) -> Any:
        captured[name] = kwargs
        if name == "barriers":
            payload = _barriers()
            payload["tp_pct"] = kwargs["take_profit_pct"]
            payload["sl_pct"] = kwargs["stop_loss_pct"]
            return payload
        mapping = {
            "session": _session(),
            "forecast": _forecast(),
            "volatility": _volatility(),
            "sizing": _sizing(),
            "preview": _preview(),
        }
        return mapping[name]

    idea = run_trade_idea_compose(
        TradeIdeaComposeRequest(symbol="EURUSD", direction="long"),
        call_section=call,
    )

    assert captured["barriers"]["take_profit_pct"] == pytest.approx(0.42)
    assert captured["barriers"]["stop_loss_pct"] == pytest.approx(0.63)
    assert idea["barriers"]["barrier_source"] == "volatility_scaled"
    assert idea["barriers"]["tp_pct"] == pytest.approx(0.42)
    assert idea["barriers"]["sl_pct"] == pytest.approx(0.63)
