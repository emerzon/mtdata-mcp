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


def test_economic_release_value_parses_leading_sign_before_currency() -> None:
    from mtdata.core.finviz.calendar import parse_economic_release_value

    deficit = parse_economic_release_value("-$68.2B")

    assert deficit == {
        "value": -68.2e9,
        "unit": "currency",
        "scale": 1_000_000_000.0,
        "currency": "USD",
        "parse_status": "ok",
    }


def test_economic_release_value_unparseable_does_not_invent_count_unit() -> None:
    from mtdata.core.finviz.calendar import parse_economic_release_value

    garbled = parse_economic_release_value("n.a.")

    assert garbled["parse_status"] == "unparseable"
    assert garbled["unit"] is None
    assert garbled["value"] is None


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
        detail="standard",
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
        detail="standard",
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


def test_economic_calendar_maps_major_us_provider_ids() -> None:
    from mtdata.core.finviz.calendar import _infer_finviz_calendar_country

    mapped = {
        "GDP CQOQ": _infer_finviz_calendar_country({"source_id": "GDP CQOQ"}),
        "NFP TCH": _infer_finviz_calendar_country({"source_id": "NFP TCH"}),
        "NAPMPMI": _infer_finviz_calendar_country({"source_id": "NAPMPMI"}),
    }

    assert mapped == {
        "GDP CQOQ": ("United States", "US"),
        "NFP TCH": ("United States", "US"),
        "NAPMPMI": ("United States", "US"),
    }


def test_economic_calendar_units_follow_row_unit_not_percent() -> None:
    from mtdata.core.finviz.calendar import _normalize_finviz_calendar_payload

    result = _normalize_finviz_calendar_payload(
        {
            "success": True,
            "items": [
                {
                    "event": "Continuing Jobless Claims",
                    "date": "2026-08-26T12:30:00Z",
                    "previous": "1.799M",
                    "forecast": "1.825M",
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
        detail="compact",
    )

    assert result["items"][0]["unit"] == "count"
    assert result["units"]["previous_value"] == "parsed_numeric (count)"
    assert "percent" not in result["units"]["previous_value"]


def test_economic_calendar_mixed_units_do_not_claim_percent() -> None:
    from mtdata.core.finviz.calendar import _normalize_finviz_calendar_payload

    result = _normalize_finviz_calendar_payload(
        {
            "success": True,
            "items": [
                {
                    "event": "CPI YoY",
                    "date": "2026-08-26T12:30:00Z",
                    "previous": "3.790%",
                    "country": "United States",
                    "country_code": "US",
                },
                {
                    "event": "Continuing Jobless Claims",
                    "date": "2026-08-26T12:30:00Z",
                    "previous": "1.799M",
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
        detail="compact",
    )

    assert result["units"]["previous_value"] == "parsed_numeric (per-row unit)"


def test_economic_calendar_utc_midnight_is_date_only() -> None:
    from mtdata.core.finviz.calendar import _normalize_finviz_economic_calendar_time

    result = _normalize_finviz_economic_calendar_time(
        {"event": "Jackson Hole Symposium", "date": "2026-08-27T00:00:00Z"}
    )

    assert result["event_time_precision"] == "date_only"
    assert result["scheduled_at"] == "2026-08-27"
    assert result["date"] == "2026-08-27"
    assert "local_time" not in result


def test_economic_calendar_compact_items_include_scheduled_at() -> None:
    from mtdata.core.finviz.calendar import _compact_finviz_calendar_item

    row = _compact_finviz_calendar_item(
        {
            "event": "Money Supply",
            "date": "2026-08-25T17:00:00Z",
            "source_id": "M2",
        }
    )

    assert row["event"] == "Money Supply"
    assert row["scheduled_at"] == "2026-08-25T17:00:00Z"
    assert "date" not in row
    assert "category" not in row
    assert "value_parse_status" not in row
    assert "scale" not in row


def test_economic_calendar_compact_drops_duplicate_category_and_ok_parse() -> None:
    from mtdata.core.finviz.calendar import _normalize_finviz_calendar_payload

    result = _normalize_finviz_calendar_payload(
        {
            "success": True,
            "items": [
                {
                    "event": "CPI YoY",
                    "category": "CPI YoY",
                    "date": "2026-08-26T12:30:00Z",
                    "actual": "3.790%",
                    "previous": "3.780%",
                    "forecast": "3.800%",
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
        detail="compact",
    )

    row = result["items"][0]
    assert row["event"] == "CPI YoY"
    assert "category" not in row
    assert "actual" not in row
    assert row["actual_value"] == 3.79
    assert row["previous_value"] == 3.78
    assert row["forecast_value"] == 3.8
    assert "scale" not in row
    assert "value_parse_status" not in row
    assert "country_attribution" not in row
    assert "actual" not in result["units"]


def test_economic_calendar_compact_keeps_lossy_parse_warning() -> None:
    from mtdata.core.finviz.calendar import _compact_finviz_calendar_item

    row = _compact_finviz_calendar_item(
        {
            "event": "Odd Print",
            "scheduled_at": "2026-08-26T12:00:00Z",
            "actual": "n.a.",
            "value_parse_status": "unparseable",
        },
        mode="compact",
    )

    assert row["parse_warning"]["code"] == "calendar_value_parse_lossy"
    assert row["value_parse_status"] == "unparseable"
    assert row["raw_actual"] == "n.a."
    assert "unit" not in row


def test_economic_calendar_trade_balance_uses_currency_unit_when_unparseable() -> None:
    from mtdata.core.finviz.calendar import (
        _attach_economic_release_values,
        _compact_finviz_calendar_item,
    )

    normalized = _attach_economic_release_values(
        {
            "event": "Goods Trade Balance Adv",
            "category": "Goods Trade Balance",
            "scheduled_at": "2026-08-27T12:30:00Z",
            "previous": "n.a.",
            "forecast": "n.a.",
        }
    )
    row = _compact_finviz_calendar_item(normalized, mode="compact")

    assert normalized["unit"] == "currency"
    assert normalized["value_parse_status"] == "unparseable"
    assert row["unit"] == "currency"
    assert row["raw_previous"] == "n.a."
    assert row["raw_forecast"] == "n.a."
    assert "previous_value" not in row


def test_calendar_timestamp_bounds_filter_scheduled_at(monkeypatch) -> None:
    from mtdata.core.finviz.calendar import run_finviz_calendar

    monkeypatch.setattr(
        "mtdata.core.finviz.calendar.get_economic_calendar",
        lambda **kwargs: {
            "success": True,
            "dateFrom": kwargs.get("date_from"),
            "dateTo": kwargs.get("date_to"),
            "calendarTimezone": "America/New_York",
            "items": [
                {
                    "event": "MBA Mortgage Refinance Index",
                    "date": "2026-08-26T11:00:00Z",
                    "country": "United States",
                    "country_code": "US",
                },
                {
                    "event": "Crude Oil Inventories",
                    "date": "2026-08-26T12:30:00Z",
                    "country": "United States",
                    "country_code": "US",
                },
            ],
            "total": 2,
        },
    )

    result = run_finviz_calendar(
        start="2026-08-26T12:00:00Z",
        end="2026-08-26T13:00:00Z",
        upcoming=False,
        limit=20,
    )

    assert result["success"] is not False
    assert result["start"] == "2026-08-26T12:00:00Z"
    assert result["end"] == "2026-08-26T13:00:00Z"
    assert result["start_precision"] == "timestamp"
    assert [item["event"] for item in result["items"]] == ["Crude Oil Inventories"]
    assert result["items"][0]["scheduled_at"] == "2026-08-26T12:30:00Z"


def test_calendar_success_includes_retrieval_freshness(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.finviz.run_finviz_calendar",
        lambda **_kwargs: {"success": True, "items": [{"event": "CPI"}], "count": 1},
    )

    result = _unwrap(calendar)(kind="economic")

    assert result["success"] is True
    assert result["data_fetched_at"].endswith("Z")
    assert "T" in result["data_fetched_at"]
    assert result["is_realtime"] is False
