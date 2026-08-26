from __future__ import annotations

from typing import Annotated, get_args, get_origin, get_type_hints

import pytest

from mtdata.core.screener import screener


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def test_screener_list_filters_uses_finviz_catalog(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.finviz.finviz_filters_list",
        lambda **kwargs: {"success": True, "items": [{"filter": "Exchange"}]},
    )

    result = _unwrap(screener)(list_filters=True)

    assert result["success"] is True
    assert result["providers_used"] == ["finviz"]
    assert result["items"][0]["filter"] == "Exchange"


def test_screener_default_order_is_descending_market_cap(monkeypatch) -> None:
    captured = {}

    def _fake_screen(**kwargs):
        captured.update(kwargs)
        return {"success": True, "items": []}

    monkeypatch.setattr("mtdata.core.finviz.finviz_screen", _fake_screen)

    result = _unwrap(screener)()

    assert result["success"] is True
    assert captured["order"] == "-marketcap"


def test_screener_rejects_offset_in_results_mode(monkeypatch) -> None:
    def _fake_screen(**kwargs):
        raise AssertionError("results mode must not fetch when offset is set")

    monkeypatch.setattr("mtdata.core.finviz.finviz_screen", _fake_screen)

    result = _unwrap(screener)(offset=20)

    assert result["success"] is False
    assert result["error_code"] == "incompatible_parameters"
    assert result["details"]["invalid"] == ["offset"]
    assert "Use --page" in result["error"]


def test_screener_rejects_search_in_results_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.finviz.finviz_screen",
        lambda **_kwargs: pytest.fail("results mode must not fetch with search"),
    )

    result = _unwrap(screener)(search="dividend")

    assert result["success"] is False
    assert result["error_code"] == "incompatible_parameters"
    assert result["details"]["invalid"] == ["search"]
    assert result["details"]["mode"] == "results"


def test_screener_rejects_filter_name_in_results_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.finviz.finviz_screen",
        lambda **_kwargs: pytest.fail("results mode must not fetch with filter_name"),
    )

    result = _unwrap(screener)(filter_name="Market Cap.")

    assert result["success"] is False
    assert result["error_code"] == "incompatible_parameters"
    assert "filter_name" in result["details"]["invalid"]


def test_screener_rejects_result_controls_in_catalog_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.finviz.finviz_filters_list",
        lambda **_kwargs: pytest.fail("catalog mode must not fetch with result filters"),
    )

    result = _unwrap(screener)(
        list_filters=True,
        filters="sector=Technology",
        order="price",
        view="valuation",
        page=2,
    )

    assert result["success"] is False
    assert result["error_code"] == "incompatible_parameters"
    assert result["details"]["mode"] == "list_filters"
    assert result["details"]["invalid"] == ["filters", "order", "view", "page"]


def test_screener_normalizes_provider_error_operation(monkeypatch) -> None:
    monkeypatch.setattr(
        "mtdata.core.finviz.finviz_screen",
        lambda **_kwargs: {
            "success": False,
            "error": "bad order",
            "operation": "finviz_screen",
        },
    )

    result = _unwrap(screener)(order="nonsense")

    assert result["operation"] == "screener"
    assert result["provider_operation"] == "finviz_screen"


def test_screener_source_schema_omits_mt5() -> None:
    annotation = get_type_hints(_unwrap(screener), include_extras=True)["source"]
    source_type = (
        get_args(annotation)[0] if get_origin(annotation) is Annotated else annotation
    )
    assert set(get_args(source_type)) == {"auto", "finviz"}


def test_screener_mt5_pin_is_unsupported() -> None:
    result = _unwrap(screener)(source="mt5")

    assert result["success"] is False
    assert result["error_code"] == "research_capability_unsupported"
