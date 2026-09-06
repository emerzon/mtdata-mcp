import pytest

from mtdata.core.indicators import _indicator_trading_context


@pytest.mark.parametrize("name, category", [
    ("ichimoku", "overlap"), ("donchian", "volatility"), ("coppock", "momentum"),
])
def test_category_fallback_does_not_invent_indicator_parameters(name, category):
    context = _indicator_trading_context({"name": name, "category": category})

    assert context["trading_styles_basis"] == "category_heuristic"
    assert context["common_use"]
    assert "typical_parameters" not in context


def test_curated_parameter_guidance_is_retained():
    context = _indicator_trading_context({"name": "rsi", "category": "momentum"})

    assert context["trading_styles_basis"] == "curated_indicator"
    assert context["typical_parameters"].startswith("rsi(14)")
