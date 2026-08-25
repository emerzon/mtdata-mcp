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


def test_economic_release_value_parses_percent_and_millions() -> None:
    from mtdata.core.finviz.calendar import parse_economic_release_value

    percent = parse_economic_release_value("3.790%")
    millions = parse_economic_release_value("1.374M")

    assert percent == {
        "value": 3.79,
        "unit": "percent",
        "scale": 1.0,
        "currency": None,
        "parse_status": "ok",
    }
    assert millions == {
        "value": 1_374_000.0,
        "unit": "count",
        "scale": 1_000_000.0,
        "currency": None,
        "parse_status": "ok",
    }


def test_economic_release_value_detects_currency_symbols() -> None:
    from mtdata.core.finviz.calendar import parse_economic_release_value

    dollars = parse_economic_release_value("$23.15T")
    euros = parse_economic_release_value("€1.2B")
    count = parse_economic_release_value("1.374M")
    percent = parse_economic_release_value("3.790%")

    assert dollars == {
        "value": 23.15e12,
        "unit": "currency",
        "scale": 1_000_000_000_000.0,
        "currency": "USD",
        "parse_status": "ok",
    }
    assert euros == {
        "value": 1.2e9,
        "unit": "currency",
        "scale": 1_000_000_000.0,
        "currency": "EUR",
        "parse_status": "ok",
    }
    assert count["unit"] == "count"
    assert count["currency"] is None
    assert percent["unit"] == "percent"
    assert percent["currency"] is None


def test_economic_calendar_keeps_raw_strings_and_adds_parsed_values() -> None:
    from mtdata.core.finviz.calendar import _normalize_finviz_calendar_payload

    result = _normalize_finviz_calendar_payload(
        {
            "success": True,
            "items": [
                {
                    "event": "CPI YoY",
                    "date": "2026-08-24T12:30:00Z",
                    "actual": "3.790%",
                    "previous": "3.780%",
                    "forecast": "3.800%",
                    "country": "United States",
                    "country_code": "US",
                },
                {
                    "event": "Crude Oil Inventories",
                    "date": "2026-08-24T14:30:00Z",
                    "previous": "1.374M",
                    "forecast": "1.443M",
                    "country": "United States",
                    "country_code": "US",
                },
            ],
            "total": 2,
        },
        calendar_type="economic",
        upcoming_only=False,
        source_is_unpaged=True,
        limit=20,
        page=1,
    )

    cpi = result["items"][0]
    inventories = result["items"][1]
    assert cpi["actual"] == "3.790%"
    assert cpi["actual_value"] == 3.79
    assert cpi["previous_value"] == 3.78
    assert cpi["forecast_value"] == 3.8
    assert cpi["unit"] == "percent"
    assert cpi["scale"] == 1.0
    assert cpi["value_parse_status"] == "ok"
    assert inventories["previous"] == "1.374M"
    assert inventories["previous_value"] == 1_374_000.0
    assert inventories["forecast_value"] == 1_443_000.0
    assert inventories["unit"] == "count"
    assert inventories["scale"] == 1_000_000.0
    assert "currency" not in inventories
    assert "actual_value" not in inventories


def test_economic_calendar_labels_currency_prints() -> None:
    from mtdata.core.finviz.calendar import _normalize_finviz_calendar_payload

    result = _normalize_finviz_calendar_payload(
        {
            "success": True,
            "items": [
                {
                    "event": "GDP",
                    "date": "2026-08-24T12:30:00Z",
                    "actual": "$23.15T",
                    "previous": "$23.00T",
                    "forecast": "$23.10T",
                    "country": "United States",
                    "country_code": "US",
                }
            ],
            "total": 1,
        },
        calendar_type="economic",
        upcoming_only=False,
        source_is_unpaged=True,
        limit=20,
        page=1,
    )

    gdp = result["items"][0]
    assert gdp["actual"] == "$23.15T"
    assert gdp["actual_value"] == 23.15e12
    assert gdp["unit"] == "currency"
    assert gdp["currency"] == "USD"
    assert gdp["scale"] == 1_000_000_000_000.0


def test_calendar_rejects_reversed_date_range() -> None:
    from mtdata.core.finviz.calendar import run_finviz_calendar

    result = run_finviz_calendar(start="2026-08-26", end="2026-08-25")

    assert result["success"] is False
    assert result["error_code"] == "invalid_date_range"
    assert result["details"]["start"] == "2026-08-26"
    assert result["details"]["end"] == "2026-08-25"
    assert "operation" in result
    assert result["operation"] in {"calendar", "finviz_calendar"}
