from __future__ import annotations

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
