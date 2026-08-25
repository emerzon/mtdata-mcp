from __future__ import annotations

import pytest

from mtdata.core.calendar import calendar
from mtdata.services.research.capabilities import CALENDAR


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def test_calendar_range_uses_finviz_adapter(monkeypatch) -> None:
    captured = {}

    def _fake_run_finviz_calendar(**kwargs):
        captured.update(kwargs)
        return {"success": True, "items": [{"title": "CPI"}], "count": 1}

    monkeypatch.setattr(
        "mtdata.core.finviz.run_finviz_calendar",
        _fake_run_finviz_calendar,
    )

    result = _unwrap(calendar)(kind="economic", impact="high", currency="USD")

    assert result["success"] is True
    assert result["providers_used"] == ["finviz"]
    assert result["provider"] == "finviz"
    assert captured["calendar"] == "economic"
    assert captured["impact"] == "high"
    assert captured["currency"] == "USD"


def test_calendar_period_view_requires_earnings() -> None:
    result = _unwrap(calendar)(kind="economic", view="period")

    assert result["success"] is False
    assert result["error_code"] == "calendar_invalid_view"


def test_calendar_mt5_pin_is_capability_unsupported() -> None:
    result = _unwrap(calendar)(source="mt5")

    assert result["success"] is False
    assert result["error_code"] == "research_capability_unsupported"
    assert result["capability"] == CALENDAR
    assert "finviz" in result["valid_values"]["source"]


def test_calendar_period_view_rejects_range_controls(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.finviz.finviz_earnings",
        lambda **_kwargs: pytest.fail("period view must not fetch with range controls"),
    )

    result = _unwrap(calendar)(
        kind="earnings",
        view="period",
        period="next-week",
        start="2000-01-01",
        end="2000-01-02",
    )

    assert result["success"] is False
    assert result["error_code"] == "incompatible_parameters"
    assert "start" in result["details"]["invalid"]
    assert "end" in result["details"]["invalid"]


def test_calendar_non_economic_kind_rejects_impact() -> None:
    result = _unwrap(calendar)(kind="earnings", impact="high")

    assert result["success"] is False
    assert result["error_code"] == "incompatible_parameters"
    assert result["details"]["invalid"] == ["impact"]
    assert result["details"]["kind"] == "earnings"
    assert "Drop impact" in result["remediation"]


def test_calendar_range_view_rejects_period_controls(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.finviz.run_finviz_calendar",
        lambda **_kwargs: pytest.fail("range view must not fetch with period controls"),
    )

    result = _unwrap(calendar)(
        kind="earnings",
        view="range",
        period="next-week",
    )

    assert result["success"] is False
    assert result["error_code"] == "incompatible_parameters"
    assert result["details"]["invalid"] == ["period"]


def test_calendar_period_view_uses_earnings_alias(monkeypatch) -> None:
    def _fake_earnings(**kwargs):
        return {
            "success": True,
            "period": kwargs.get("period"),
            "items": [{"ticker": "AAPL"}],
            "count": 1,
        }

    monkeypatch.setattr("mtdata.core.finviz.finviz_earnings", _fake_earnings)

    result = _unwrap(calendar)(
        kind="earnings",
        view="period",
        period="this-week",
    )

    assert result["success"] is True
    assert result["period"] == "this-week"
    assert result["providers_used"] == ["finviz"]


def test_calendar_period_hint_is_cli_runnable(monkeypatch) -> None:
    def _fake_earnings(**kwargs):
        return {
            "success": True,
            "period": "this-week",
            "detail": "compact",
            "hint": (
                "Period-based earnings view; use "
                "calendar --kind earnings --view range "
                "--start 2026-03-01 --end 2026-03-07 "
                "for date-range EPS/sales actuals and surprises."
            ),
            "items": [{"ticker": "AAPL"}],
            "count": 1,
        }

    monkeypatch.setattr("mtdata.core.finviz.finviz_earnings", _fake_earnings)

    result = _unwrap(calendar)(kind="earnings", view="period")

    hint = result["hint"]
    assert "finviz_calendar" not in hint
    assert "calendar --kind earnings --view range" in hint
    assert "--start" in hint
    assert "--end" in hint


def test_calendar_rejects_reversed_date_range() -> None:
    from mtdata.core.finviz.calendar import run_finviz_calendar

    result = run_finviz_calendar(start="2026-08-26", end="2026-08-25")

    assert result["success"] is False
    assert result["error_code"] == "invalid_date_range"
    assert result["details"]["start"] == "2026-08-26"
    assert result["details"]["end"] == "2026-08-25"
    assert "operation" in result
    assert result["operation"] in {"calendar", "finviz_calendar"}
