import sys
from types import SimpleNamespace

import pytest

from mtdata.core.options import _apply_options_detail
from mtdata.forecast.quantlib_tools import price_barrier_option_quantlib


@pytest.fixture(autouse=True)
def quantlib_calendar_stub(monkeypatch):
    settings = SimpleNamespace(evaluationDate=None)
    monkeypatch.setitem(sys.modules, "QuantLib", SimpleNamespace(
        NullCalendar=object, Settings=SimpleNamespace(instance=lambda: settings)
    ))


@pytest.mark.parametrize("barrier_type,barrier", [("up_out", 110), ("down_out", 90)])
@pytest.mark.parametrize("valuation_date", ["2026-09-04", "2026-09-05"])
@pytest.mark.parametrize("rebate", [0.0, 2.0])
def test_prior_knockout_excludes_assumed_settled_rebate(barrier_type, barrier, valuation_date, rebate):
    result = price_barrier_option_quantlib(
        spot=100, strike=100, barrier=barrier, maturity_days=30,
        barrier_type=barrier_type, rebate=rebate, barrier_already_hit=True,
        valuation_date=valuation_date, calendar="NullCalendar",
    )
    assert result["success"] is True
    assert result["price"] == 0.0
    assert result["barrier_state_source"] == "explicit_prior_hit"
    assert result["option_status"] == "knocked_out"
    assert result["rebate_cashflow"] == {
        "amount": rebate,
        "units": "per_underlying_unit",
        "settlement": "assumed_paid_before_valuation",
        "included_in_price": False,
    }
    assert "settled" in result["pricing_assumptions"]["model"]
    assert result["delta"] == result["gamma"] == result["vega"] == 0.0
    for detail in ("compact", "full"):
        public = _apply_options_detail(result, detail=detail, kind="barrier_price")
        assert public["price"] == 0.0
        assert public["rebate_cashflow"] == result["rebate_cashflow"]


@pytest.mark.parametrize("barrier_type", ["up_out", "down_out"])
def test_knockout_at_valuation_includes_immediately_due_rebate(barrier_type):
    result = price_barrier_option_quantlib(
        spot=100, strike=100, barrier=100, maturity_days=30,
        barrier_type=barrier_type, rebate=2.0,
        valuation_date="2026-09-04", calendar="NullCalendar",
    )
    assert result["success"] is True
    assert result["price"] == 2.0
    assert result["barrier_state_source"] == "spot_at_or_beyond_barrier"
    assert result["rebate_cashflow"]["settlement"] == "due_at_valuation"
    assert result["rebate_cashflow"]["included_in_price"] is True


@pytest.mark.parametrize("rebate", [-2.0, float("nan"), float("inf")])
def test_prior_knockout_still_rejects_invalid_rebate(rebate):
    result = price_barrier_option_quantlib(
        spot=100, strike=100, barrier=90, maturity_days=30,
        barrier_type="down_out", rebate=rebate, barrier_already_hit=True,
        valuation_date="2026-09-04", calendar="NullCalendar",
    )
    assert result["error_code"] == "invalid_rebate"
