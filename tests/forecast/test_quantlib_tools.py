from __future__ import annotations

import datetime as _dt
import types

import pytest

from mtdata.forecast import quantlib_tools as qtools


def _make_fake_quantlib():  # noqa: C901
    class _Settings:
        _instance = None

        def __init__(self):
            self.evaluationDate = None

        @classmethod
        def instance(cls):
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    class _Date:
        def __init__(self, day=None, month=None, year=None):
            if day is None or month is None or year is None:
                self.ordinal = _dt.date(2026, 1, 1).toordinal()
            else:
                self.ordinal = _dt.date(int(year), int(month), int(day)).toordinal()

        @classmethod
        def from_ordinal(cls, ordinal):
            instance = cls.__new__(cls)
            instance.ordinal = int(ordinal)
            return instance

        @staticmethod
        def todaysDate():
            return _Date.from_ordinal(_dt.date(2026, 1, 1).toordinal())

        def __add__(self, _days):
            return _Date.from_ordinal(self.ordinal + int(_days))

        def __sub__(self, other):
            return int(self.ordinal - other.ordinal)

        def year(self):
            return _dt.date.fromordinal(self.ordinal).year

        def month(self):
            return _dt.date.fromordinal(self.ordinal).month

        def dayOfMonth(self):
            return _dt.date.fromordinal(self.ordinal).day

    class _UnitedStates:
        NYSE = "NYSE"

        def __init__(self, market=None):
            self.market = market

        def advance(self, date, days, _unit):
            return date + int(days)

        def businessDaysBetween(self, start, end):
            return max(0, (end - start) - 4)

    class _NullCalendar:
        def advance(self, date, days, _unit):
            return date + int(days)

        def businessDaysBetween(self, start, end):
            return max(0, end - start)

    class _Option:
        Call = "call"
        Put = "put"

    class _Barrier:
        UpIn = "up_in"
        UpOut = "up_out"
        DownIn = "down_in"
        DownOut = "down_out"

    class _SimpleQuote:
        def __init__(self, value):
            self.value = float(value)

    class _QuoteHandle:
        def __init__(self, quote):
            self.quote = quote

    class _FlatForward:
        def __init__(self, _today, rate, _day_count):
            self.rate = float(rate)

    class _YieldTermStructureHandle:
        def __init__(self, curve):
            self.curve = curve

    class _BlackConstantVol:
        def __init__(self, _today, _calendar, vol, _day_count):
            self.vol = float(vol)

    class _BlackVolTermStructureHandle:
        def __init__(self, vol):
            self.vol = float(vol.vol)

    class _BlackScholesMertonProcess:
        def __init__(self, spot_h, _div_ts, _rf_ts, vol_ts):
            self.spot = float(spot_h.quote.value)
            self.vol = float(vol_ts.vol)

    class _AnalyticBarrierEngine:
        def __init__(self, process):
            self.process = process

    class _PlainVanillaPayoff:
        def __init__(self, option_type, strike):
            self.option_type = option_type
            self.strike = float(strike)

    class _EuropeanExercise:
        def __init__(self, maturity):
            self.maturity = maturity

    class _BarrierOption:
        def __init__(self, barrier_type, barrier, rebate, payoff, exercise):
            self.barrier_type = barrier_type
            self.barrier = float(barrier)
            self.rebate = float(rebate)
            self.payoff = payoff
            self.exercise = exercise
            self._engine = None

        def setPricingEngine(self, engine):
            self._engine = engine

        def NPV(self):
            if self._engine is None:
                return 0.0
            return max(0.0, 0.01 * self._engine.process.spot + 0.1 * self._engine.process.vol)

    class _HestonProcess:
        def __init__(self, *args, **_kwargs):
            self.spot = 0.0
            if len(args) >= 3:
                spot_h = args[2]
                try:
                    self.spot = float(spot_h.quote.value)
                except Exception:
                    self.spot = 0.0
            self.v0 = float(args[3]) if len(args) > 3 else 0.04
            self.vol = self.v0 ** 0.5

    class _HestonModel:
        def __init__(self, process):
            self.process = process
            self._kappa = 2.0
            self._theta = 0.04
            self._sigma = 0.30
            self._rho = -0.5
            self._v0 = 0.04

        def calibrate(self, _helpers, _method, _end_criteria):
            return None

        def kappa(self):
            return self._kappa

        def theta(self):
            return self._theta

        def sigma(self):
            return self._sigma

        def rho(self):
            return self._rho

        def v0(self):
            return self._v0

    class _AnalyticHestonEngine:
        def __init__(self, model):
            self.process = getattr(model, "process", None)

    class _FdHestonBarrierEngine:
        def __init__(self, model, *_args, **_kwargs):
            self.process = getattr(model, "process", None)

    class _Period:
        def __init__(self, length, unit):
            self.length = int(length)
            self.unit = unit

    class _HestonModelHelper:
        created = []

        def __init__(self, maturity, calendar, spot, strike, iv_handle, _rf_ts, _div_ts, _err_type):
            self.maturity = maturity
            self.calendar = calendar
            self.spot = float(spot)
            self.strike = float(strike)
            self.iv = float(iv_handle.quote.value)
            type(self).created.append(self)

        def setPricingEngine(self, _engine):
            return None

        def calibrationError(self):
            return abs(self.strike - self.spot) / max(self.spot, 1.0)

    class _BlackCalibrationHelper:
        ImpliedVolError = "iv_err"

    fake = types.SimpleNamespace(
        Date=_Date,
        Settings=_Settings,
        Actual365Fixed=lambda: object(),
        UnitedStates=_UnitedStates,
        NullCalendar=_NullCalendar,
        Option=_Option,
        Barrier=_Barrier,
        PlainVanillaPayoff=_PlainVanillaPayoff,
        EuropeanExercise=_EuropeanExercise,
        BarrierOption=_BarrierOption,
        QuoteHandle=_QuoteHandle,
        SimpleQuote=_SimpleQuote,
        YieldTermStructureHandle=_YieldTermStructureHandle,
        FlatForward=_FlatForward,
        BlackVolTermStructureHandle=_BlackVolTermStructureHandle,
        BlackConstantVol=_BlackConstantVol,
        BlackScholesMertonProcess=_BlackScholesMertonProcess,
        AnalyticBarrierEngine=_AnalyticBarrierEngine,
        HestonProcess=_HestonProcess,
        HestonModel=_HestonModel,
        AnalyticHestonEngine=_AnalyticHestonEngine,
        FdHestonBarrierEngine=_FdHestonBarrierEngine,
        Period=_Period,
        Days="days",
        HestonModelHelper=_HestonModelHelper,
        BlackCalibrationHelper=_BlackCalibrationHelper,
        LevenbergMarquardt=lambda: object(),
        EndCriteria=lambda *_args: object(),
    )
    return fake


def _current_chain_snapshot(
    observed_at: str = "2026-12-01T20:00:00Z",
) -> dict:
    return {
        "underlying_as_of": observed_at,
        "underlying_data_age_seconds": 15.0,
        "underlying_data_stale": False,
        "underlying_freshness": "provider_timestamped",
        "option_chain_freshness": "current",
        "option_chain_quality": "live_usable",
    }


def _qualified_contract(
    strike: float,
    *,
    implied_volatility: float = 0.25,
    side: str = "call",
    observed_at: str = "2026-12-01T20:00:00Z",
) -> dict:
    return {
        "contract": f"{side}-{strike:g}",
        "strike": strike,
        "implied_volatility": implied_volatility,
        "side": side,
        "bid": 1.0,
        "ask": 1.1,
        "contract_as_of": observed_at,
        "contract_data_age_seconds": 15.0,
        "contract_data_stale": False,
        "contract_freshness": "provider_timestamped",
        "quote_quality": "two_sided",
        "quote_usable_for_live_analysis": True,
    }


def test_price_barrier_option_quantlib_with_fake_backend(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "QuantLib", _make_fake_quantlib())
    out = qtools.price_barrier_option_quantlib(
        spot=100.0,
        strike=100.0,
        barrier=120.0,
        maturity_days=30,
        option_type="call",
        barrier_type="up_out",
        risk_free_rate=0.02,
        dividend_yield=0.0,
        volatility=0.2,
        rebate=0.0,
    )
    assert out["success"] is True
    assert out["price"] > 0.0
    assert out["params_used"]["option_type"] == "call"
    assert out["params_used"]["barrier_type"] == "up_out"
    assert out["greeks_status"] == "complete"
    assert out["greeks_spot_step"] == 0.01
    assert out["valuation_timezone"] == "America/New_York"
    assert out["valuation_date_source"] == "default_calendar_local_date"


def test_price_barrier_option_quantlib_rejects_non_positive_inputs():
    cases = (
        ("spot", {"spot": -150.0, "strike": 100.0, "barrier": 120.0, "volatility": 0.2}),
        ("strike", {"spot": 100.0, "strike": 0.0, "barrier": 120.0, "volatility": 0.2}),
        ("barrier", {"spot": 100.0, "strike": 100.0, "barrier": -1.0, "volatility": 0.2}),
        ("volatility", {"spot": 100.0, "strike": 100.0, "barrier": 120.0, "volatility": -0.5}),
    )
    for parameter, kwargs in cases:
        out = qtools.price_barrier_option_quantlib(
            maturity_days=30,
            **kwargs,
        )
        assert out["error_code"] == "invalid_parameter"
        assert out["details"]["parameter"] == parameter
        assert "options_provider_status" not in str(out)


def test_price_barrier_option_quantlib_rejects_negative_rebate():
    out = qtools.price_barrier_option_quantlib(
        spot=100.0,
        strike=100.0,
        barrier=120.0,
        maturity_days=30,
        rebate=-1.0,
    )

    assert out["error_code"] == "invalid_rebate"


def test_default_valuation_date_uses_selected_calendar_timezone():
    valuation_day, timezone_name, source, warning = qtools._default_valuation_date(
        "UnitedStates.NYSE",
        now_utc=_dt.datetime(2026, 8, 14, 1, 10, tzinfo=_dt.timezone.utc),
    )

    assert valuation_day == _dt.date(2026, 8, 13)
    assert timezone_name == "America/New_York"
    assert source == "default_calendar_local_date"
    assert warning is None


def test_price_barrier_option_quantlib_uses_safe_step_near_barrier(monkeypatch):
    fake = _make_fake_quantlib()
    base_option = fake.BarrierOption

    class _StrictBarrierOption(base_option):
        def NPV(self):
            spot = self._engine.process.spot
            if self.barrier_type.startswith("up_") and spot >= self.barrier:
                raise RuntimeError("barrier touched")
            if self.barrier_type.startswith("down_") and spot <= self.barrier:
                raise RuntimeError("barrier touched")
            return super().NPV()

    fake.BarrierOption = _StrictBarrierOption
    monkeypatch.setitem(__import__("sys").modules, "QuantLib", fake)

    out = qtools.price_barrier_option_quantlib(
        spot=1.168,
        strike=1.15,
        barrier=1.17,
        maturity_days=30,
        option_type="call",
        barrier_type="up_out",
    )

    assert out["success"] is True
    assert out["greeks_status"] == "complete"
    assert out["greeks_method"] == "central_difference"
    assert out["greeks_spot_step"] < 1.17 - 1.168
    assert out["delta"] is not None


def test_price_barrier_option_quantlib_exposes_calendar_overrides(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "QuantLib", _make_fake_quantlib())

    out = qtools.price_barrier_option_quantlib(
        spot=100.0,
        strike=100.0,
        barrier=120.0,
        maturity_days=30,
        option_type="call",
        barrier_type="up_out",
        calendar="NullCalendar",
        maturity_basis="business_days",
        valuation_date="2026-07-03",
    )

    assert out["success"] is True
    assert out["pricing_assumptions"]["calendar"] == "NullCalendar"
    assert out["pricing_assumptions"]["maturity_basis"] == "business_days"
    assert out["params_used"]["calendar"] == "NullCalendar"
    assert out["params_used"]["maturity_basis"] == "business_days"
    assert out["valuation_date"] == "2026-07-03"
    assert out["valuation_timezone"] == "UTC"
    assert out["valuation_date_source"] == "explicit"
    assert out["maturity_date"] == "2026-08-02"
    assert out["time_to_maturity_years"] == 30 / 365
    assert out["params_used"]["valuation_date"] == "2026-07-03"


def test_price_barrier_option_quantlib_returns_knocked_out_payoff(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "QuantLib", _make_fake_quantlib())
    out = qtools.price_barrier_option_quantlib(
        spot=115.0,
        strike=100.0,
        barrier=110.0,
        maturity_days=30,
        option_type="call",
        barrier_type="up_out",
        risk_free_rate=0.02,
        dividend_yield=0.0,
        volatility=0.2,
        rebate=0.0,
    )

    assert out["success"] is True
    assert out["price"] == 0.0
    assert out["status"] == "knocked_out"
    assert out["delta"] == 0.0
    assert out["gamma"] == 0.0
    assert out["vega"] == 0.0
    assert out["params_used"]["spot"] == 115.0
    assert out["params_used"]["barrier"] == 110.0


def test_calibrate_heston_quantlib_from_options_with_fake_backend(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "QuantLib", _make_fake_quantlib())
    monkeypatch.setattr(
        qtools,
        "get_options_chain",
        lambda **kwargs: {
            "success": True,
            "symbol": kwargs["symbol"],
            "expiration": "2026-12-19",
            "underlying_price": 100.0,
            **_current_chain_snapshot(),
            "underlying_price_source": "tradier_last",
            "underlying_price_session": "provider_reported_last",
            "options": [
                _qualified_contract(90.0, implied_volatility=0.35),
                _qualified_contract(95.0, implied_volatility=0.30),
                _qualified_contract(100.0, implied_volatility=0.28),
                _qualified_contract(105.0, implied_volatility=0.29),
                _qualified_contract(110.0, implied_volatility=0.33),
                _qualified_contract(115.0, implied_volatility=0.37),
            ],
        },
    )
    out = qtools.calibrate_heston_quantlib_from_options(
        symbol="AAPL",
        expiration="2026-12-19",
        option_type="call",
        risk_free_rate=0.03,
        dividend_yield=0.01,
        min_open_interest=0,
        min_volume=0,
        max_contracts=5,
    )
    assert out["success"] is True
    assert out["symbol"] == "AAPL"
    assert out["valuation_date"] == "2026-12-01"
    assert out["valuation_date_source"] == "chain_observation_date"
    assert out["spot_as_of"] == "2026-12-01T20:00:00Z"
    assert out["spot_data_age_seconds"] == 15.0
    assert out["spot_data_stale"] is False
    assert out["spot_source"] == "tradier_last"
    assert out["calibration_data_status"] == "current"
    assert out["calibration_mode"] == "single_expiry_fit"
    assert out["quote_freshness_policy"] == "last_trade_proxy"
    assert any("single_expiry_fit" in warning for warning in out["warnings"])
    assert out["days_to_expiry"] == 18
    assert out["contracts_used"] == 5
    assert out["selected_contracts_current_count"] == 5
    assert out["selected_contracts_quote_usable_count"] == 5
    assert out["selected_contract_max_spot_skew_seconds"] == 0.0
    assert out["contract_spot_skew_limit_seconds"] == 900.0
    assert out["sample_contracts"][0]["contract_as_of"] == (
        "2026-12-01T20:00:00Z"
    )
    assert set(out["params"].keys()) == {"kappa", "theta", "sigma", "rho", "v0"}
    assert out["calibration_error_rmse"] is not None
    assert out["calibration_error_rmse_unit"] == "absolute_implied_volatility"
    assert out["calibration_status"] == "accepted"
    assert out["usable_for_pricing"] is True
    assert out["pricing_usability_failures"] == []
    assert out["feller_satisfied"] is True
    assert out["rho_at_bound"] is False


def test_calibrate_heston_marks_stale_snapshot_unusable_for_pricing(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "QuantLib", _make_fake_quantlib())
    monkeypatch.setattr(
        qtools,
        "get_options_chain",
        lambda **_kwargs: {
            "success": True,
            "expiration": "2026-12-19",
            "underlying_price": 100.0,
            **_current_chain_snapshot(),
            "underlying_data_age_seconds": 3600.0,
            "underlying_data_stale": True,
            "underlying_freshness": "stale",
            "underlying_freshness_reason": "provider_snapshot_too_old",
            "options": [
                _qualified_contract(strike)
                for strike in (90, 95, 100, 105, 110)
            ],
        },
    )

    out = qtools.calibrate_heston_quantlib_from_options(
        symbol="AAPL",
        expiration="2026-12-19",
    )

    assert out["success"] is False
    assert out["error_code"] == "heston_calibration_rejected"
    assert out["calibration_data_status"] == "stale"
    assert out["calibration_status"] == "rejected"
    assert out["calibration_quality_failures"] == []
    assert out["usable_for_pricing"] is False
    assert out["pricing_usability_failures"] == ["stale_underlying_data"]
    assert any("not usable for pricing" in warning for warning in out["warnings"])


def test_calibrate_heston_rejects_stale_zero_sided_contract_surface(
    monkeypatch,
):
    fake = _make_fake_quantlib()
    monkeypatch.setitem(__import__("sys").modules, "QuantLib", fake)
    stale_contracts = []
    for strike in (90, 95, 100, 105, 110):
        contract = _qualified_contract(
            strike,
            observed_at="2026-11-30T20:00:00Z",
        )
        contract.update(
            {
                "bid": 0.0,
                "ask": 0.0,
                "contract_data_age_seconds": 86400.0,
                "contract_data_stale": True,
                "contract_freshness": "stale",
                "quote_quality": "zero_sided",
                "quote_usable_for_live_analysis": False,
            }
        )
        stale_contracts.append(contract)
    monkeypatch.setattr(
        qtools,
        "get_options_chain",
        lambda **_kwargs: {
            "success": True,
            "expiration": "2026-12-19",
            "underlying_price": 100.0,
            **_current_chain_snapshot(),
            "option_chain_freshness": "stale",
            "option_chain_quality": "unusable",
            "options": stale_contracts,
        },
    )

    out = qtools.calibrate_heston_quantlib_from_options(
        symbol="AAPL",
        expiration="2026-12-19",
    )

    assert out["success"] is False
    assert out["error_code"] == "heston_contract_inputs_rejected"
    assert out["calibration_status"] == "rejected"
    assert out["calibration_data_status"] == "unusable_contracts"
    assert out["contracts_usable"] == 0
    assert out["contract_quality_rejections"] == {
        "contract_spot_timestamp_mismatch": 5,
        "stale_contract_timestamp": 5,
        "contract_quote_not_two_sided": 5,
        "contract_quote_not_live_usable": 5,
    }
    assert out["next_tool"] == "options_provider_status"
    assert "active session" in out["remediation"]
    assert "intentionally rejected" in out["remediation"]
    assert fake.HestonModelHelper.created == []


@pytest.mark.parametrize(
    ("contract_patch", "expected_rejection"),
    [
        (
            {
                "contract_as_of": None,
                "contract_data_stale": None,
                "quote_usable_for_live_analysis": False,
            },
            "contract_timestamp_unavailable",
        ),
        (
            {"contract_as_of": "2026-12-01T19:30:00Z"},
            "contract_spot_timestamp_mismatch",
        ),
    ],
)
def test_calibrate_heston_rejects_unqualified_contract_time(
    monkeypatch,
    contract_patch,
    expected_rejection,
):
    fake = _make_fake_quantlib()
    monkeypatch.setitem(__import__("sys").modules, "QuantLib", fake)
    contracts = []
    for strike in (90, 95, 100, 105, 110):
        contract = _qualified_contract(strike)
        contract.update(contract_patch)
        contracts.append(contract)
    monkeypatch.setattr(
        qtools,
        "get_options_chain",
        lambda **_kwargs: {
            "success": True,
            "expiration": "2026-12-19",
            "underlying_price": 100.0,
            **_current_chain_snapshot(),
            "options": contracts,
        },
    )

    out = qtools.calibrate_heston_quantlib_from_options(
        symbol="AAPL",
        expiration="2026-12-19",
    )

    assert out["error_code"] == "heston_contract_inputs_rejected"
    assert out["contract_quality_rejections"][expected_rejection] == 5
    assert fake.HestonModelHelper.created == []


def test_calibrate_heston_default_selects_nearest_eligible_expiration(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "QuantLib", _make_fake_quantlib())
    monkeypatch.setattr(
        qtools,
        "get_options_expirations",
        lambda _symbol: {
            "success": True,
            "underlying_as_of": "2026-12-01T20:00:00Z",
            "expirations": ["2026-12-01", "2026-12-04", "2026-12-08"],
        },
    )
    requested: list[str] = []

    def _chain(**kwargs):
        requested.append(kwargs["expiration"])
        return {
            "success": True,
            "expiration": kwargs["expiration"],
            "underlying_price": 100.0,
            **_current_chain_snapshot(),
            "options": [
                _qualified_contract(strike)
                for strike in (90, 95, 100, 105, 110)
            ],
        }

    monkeypatch.setattr(qtools, "get_options_chain", _chain)

    out = qtools.calibrate_heston_quantlib_from_options(symbol="AAPL")

    assert out["success"] is True
    assert requested == ["2026-12-08"]
    assert out["expiration"] == "2026-12-08"
    assert out["expiration_selection"] == {
        "mode": "nearest_eligible",
        "minimum_calendar_days": 7,
        "selection_date": "2026-12-01",
        "selection_date_source": "expirations_observation_date",
        "skipped_ineligible_expirations": ["2026-12-01", "2026-12-04"],
    }


def test_calibrate_heston_rejects_subweek_maturity_before_fit(monkeypatch):
    fake = _make_fake_quantlib()
    monkeypatch.setitem(__import__("sys").modules, "QuantLib", fake)
    monkeypatch.setattr(
        qtools,
        "get_options_chain",
        lambda **_kwargs: {
            "success": True,
            "expiration": "2026-12-02",
            "underlying_price": 100.0,
            **_current_chain_snapshot(),
            "options": [
                _qualified_contract(strike)
                for strike in (90, 95, 100, 105, 110)
            ],
        },
    )

    out = qtools.calibrate_heston_quantlib_from_options(
        symbol="AAPL", expiration="2026-12-02"
    )

    assert out["error_code"] == "heston_maturity_too_short"
    assert out["calendar_days_to_expiry"] == 1
    assert out["minimum_calendar_days"] == 7
    assert fake.HestonModelHelper.created == []


def test_calibrate_heston_marks_bound_fit_unusable(monkeypatch):
    fake = _make_fake_quantlib()

    class BoundModel(fake.HestonModel):
        def __init__(self, process):
            super().__init__(process)
            self._kappa = 0.00001
            self._sigma = 1.5
            self._rho = -0.999

    fake.HestonModel = BoundModel
    monkeypatch.setitem(__import__("sys").modules, "QuantLib", fake)
    monkeypatch.setattr(
        qtools,
        "get_options_chain",
        lambda **_kwargs: {
            "success": True,
            "expiration": "2026-12-19",
            "underlying_price": 100.0,
            **_current_chain_snapshot(),
            "options": [
                _qualified_contract(strike)
                for strike in (90, 95, 100, 105, 110)
            ],
        },
    )

    out = qtools.calibrate_heston_quantlib_from_options(
        symbol="AAPL", expiration="2026-12-19"
    )

    assert out["success"] is False
    assert out["error_code"] == "heston_calibration_rejected"
    assert out["calibration_status"] == "rejected"
    assert out["usable_for_pricing"] is False
    assert out["rho_at_bound"] is True
    assert out["feller_satisfied"] is False
    assert set(out["calibration_quality_failures"]) == {
        "rho_at_calibration_bound",
        "kappa_near_zero",
    }
    assert set(out["params"]) == {"kappa", "theta", "sigma", "rho", "v0"}


def test_calibrate_heston_both_sides_use_supported_helper_signature(monkeypatch):
    fake = _make_fake_quantlib()
    monkeypatch.setitem(__import__("sys").modules, "QuantLib", fake)
    monkeypatch.setattr(
        qtools,
        "get_options_chain",
        lambda **kwargs: {
            "success": True,
            "symbol": kwargs["symbol"],
            "expiration": "2026-12-19",
            "underlying_price": 100.0,
            **_current_chain_snapshot(),
            "options": [
                _qualified_contract(strike, side=side)
                for strike, side in zip((90, 95, 100, 105, 110), ("put", "call", "put", "call", "put"))
            ],
        },
    )

    out = qtools.calibrate_heston_quantlib_from_options(
        symbol="AAPL", expiration="2026-12-19", option_type="both", valuation_date="2026-12-01"
    )

    assert out["success"] is True
    assert len(fake.HestonModelHelper.created) == 5
    assert all(
        isinstance(helper.calendar, fake.NullCalendar)
        for helper in fake.HestonModelHelper.created
    )
    assert {helper.maturity.length for helper in fake.HestonModelHelper.created} == {18}


def test_calibrate_heston_quantlib_uses_calendar_override_for_business_days(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "QuantLib", _make_fake_quantlib())
    monkeypatch.setattr(
        qtools,
        "get_options_chain",
        lambda **kwargs: {
            "success": True,
            "symbol": kwargs["symbol"],
            "expiration": "2026-12-19",
            "underlying_price": 100.0,
            **_current_chain_snapshot(),
            "options": [
                _qualified_contract(90.0, implied_volatility=0.35),
                _qualified_contract(95.0, implied_volatility=0.30),
                _qualified_contract(100.0, implied_volatility=0.28),
                _qualified_contract(105.0, implied_volatility=0.29),
                _qualified_contract(110.0, implied_volatility=0.33),
                _qualified_contract(115.0, implied_volatility=0.37),
            ],
        },
    )

    out = qtools.calibrate_heston_quantlib_from_options(
        symbol="AAPL",
        expiration="2026-12-19",
        valuation_date="2026-12-01",
        calendar="NullCalendar",
        maturity_basis="business_days",
    )

    assert out["success"] is True
    assert out["days_to_expiry"] == 18
    assert out["pricing_assumptions"]["calendar"] == "NullCalendar"
    assert out["pricing_assumptions"]["maturity_convention"] == "calendar_days_to_contract_expiry"
    assert out["pricing_assumptions"]["days_to_expiry_basis"] == "business_days"
    assert "maturity_basis" not in out["pricing_assumptions"]
    assert out["valuation_timezone"] == "UTC"
    assert out["valuation_date_source"] == "explicit_chain_observation_date"


def test_calibrate_heston_rejects_valuation_date_outside_chain_snapshot(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "QuantLib", _make_fake_quantlib())
    monkeypatch.setattr(
        qtools,
        "get_options_chain",
        lambda **_kwargs: {
            "success": True,
            "expiration": "2026-12-19",
            "underlying_price": 100.0,
            **_current_chain_snapshot("2026-12-01T23:30:00Z"),
            "options": [
                _qualified_contract(
                    strike,
                    observed_at="2026-12-01T23:30:00Z",
                )
                for strike in (90, 95, 100, 105, 110)
            ],
        },
    )

    out = qtools.calibrate_heston_quantlib_from_options(
        symbol="AAPL",
        expiration="2026-12-19",
        valuation_date="2026-11-30",
    )

    assert out["error_code"] == "valuation_date_chain_mismatch"
    assert out["chain_observation_date"] == "2026-12-01"
    assert out["spot_as_of"] == "2026-12-01T23:30:00Z"
    assert "Omit valuation_date" in out["remediation"]


def test_heston_valuation_mismatch_precedes_contract_quality_failure(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "QuantLib", _make_fake_quantlib())
    monkeypatch.setattr(
        qtools,
        "get_options_chain",
        lambda **_kwargs: {
            "success": True,
            "expiration": "2026-12-19",
            "underlying_price": 100.0,
            **_current_chain_snapshot("2026-12-01T23:30:00Z"),
            "options": [],
        },
    )

    out = qtools.calibrate_heston_quantlib_from_options(
        symbol="AAPL",
        expiration="2026-12-19",
        valuation_date="2026-11-30",
    )

    assert out["error_code"] == "valuation_date_chain_mismatch"
    assert out["chain_observation_date"] == "2026-12-01"


def test_calibrate_heston_requires_chain_observation_timestamp(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "QuantLib", _make_fake_quantlib())
    monkeypatch.setattr(
        qtools,
        "get_options_chain",
        lambda **_kwargs: {
            "success": True,
            "expiration": "2026-12-19",
            "underlying_price": 100.0,
            "underlying_as_of": None,
            "options": [
                {"strike": strike, "implied_volatility": 0.25, "side": "call"}
                for strike in (90, 95, 100, 105, 110)
            ],
        },
    )

    out = qtools.calibrate_heston_quantlib_from_options(
        symbol="AAPL", expiration="2026-12-19"
    )

    assert out["error_code"] == "chain_observation_time_unavailable"
    assert out["spot_as_of"] is None


def test_calibrate_heston_rejects_invalid_valuation_date(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "QuantLib", _make_fake_quantlib())
    monkeypatch.setattr(
        qtools,
        "get_options_chain",
        lambda **_kwargs: {
            "success": True,
            "expiration": "2026-12-19",
            "underlying_price": 100.0,
            "options": [
                {"strike": strike, "implied_volatility": 0.25, "side": "call"}
                for strike in (90, 95, 100, 105, 110)
            ],
        },
    )

    out = qtools.calibrate_heston_quantlib_from_options(
        symbol="AAPL",
        valuation_date="12/01/2026",
    )

    assert out == {
        "error": "Invalid valuation_date: 12/01/2026. Use YYYY-MM-DD."
    }


@pytest.mark.parametrize(
    ("kwargs", "parameter"),
    [
        ({"min_open_interest": -1}, "min_open_interest"),
        ({"min_volume": -1}, "min_volume"),
        ({"max_contracts": 4}, "max_contracts"),
    ],
)
def test_calibrate_heston_rejects_out_of_range_controls_before_chain_fetch(
    monkeypatch,
    kwargs,
    parameter,
):
    def fail_chain(**_kwargs):
        raise AssertionError("option chain should not be queried")

    monkeypatch.setattr(qtools, "get_options_chain", fail_chain)

    result = qtools.calibrate_heston_quantlib_from_options(
        symbol="AAPL",
        **kwargs,
    )

    assert parameter in result["error"]
    assert "greater than or equal" in result["error"]


@pytest.mark.parametrize("valuation_date", ["2026-12-19", "2026-12-20"])
def test_calibrate_heston_rejects_nonpositive_contract_maturity(
    monkeypatch, valuation_date
):
    monkeypatch.setitem(__import__("sys").modules, "QuantLib", _make_fake_quantlib())
    monkeypatch.setattr(
        qtools,
        "get_options_chain",
        lambda **_kwargs: {
            "success": True,
            "expiration": "2026-12-19",
            "underlying_price": 100.0,
            **_current_chain_snapshot(f"{valuation_date}T15:00:00Z"),
            "options": [
                _qualified_contract(
                    strike,
                    observed_at=f"{valuation_date}T15:00:00Z",
                )
                for strike in (90, 95, 100, 105, 110)
            ],
        },
    )

    out = qtools.calibrate_heston_quantlib_from_options(
        symbol="AAPL",
        expiration="2026-12-19",
        valuation_date=valuation_date,
    )

    assert out["error_code"] == "invalid_expiration_date_range"
    assert out["valuation_date"] == valuation_date
    assert out["expiration"] == "2026-12-19"


def test_price_barrier_option_validates_calendar_before_knocked_out_payoff(
    monkeypatch,
):
    monkeypatch.setitem(__import__("sys").modules, "QuantLib", _make_fake_quantlib())
    out = qtools.price_barrier_option_quantlib(
        spot=150.0,
        strike=155.0,
        barrier=140.0,
        maturity_days=30,
        barrier_type="up_out",
        calendar="DefinitelyNotARealCalendar",
    )

    assert out.get("success") is not True
    assert out["error_code"] == "invalid_calendar"
    assert "DefinitelyNotARealCalendar" in out["error"]


def test_default_valuation_date_maps_japan_and_uk_timezones():
    japan_day, japan_tz, japan_source, japan_warning = qtools._default_valuation_date(
        "Japan",
        now_utc=_dt.datetime(2026, 8, 14, 15, 10, tzinfo=_dt.timezone.utc),
    )
    uk_day, uk_tz, uk_source, uk_warning = qtools._default_valuation_date(
        "UnitedKingdom",
        now_utc=_dt.datetime(2026, 8, 14, 1, 10, tzinfo=_dt.timezone.utc),
    )

    assert japan_tz == "Asia/Tokyo"
    assert japan_source == "default_calendar_local_date"
    assert japan_warning is None
    assert japan_day == _dt.date(2026, 8, 15)
    assert uk_tz == "Europe/London"
    assert uk_source == "default_calendar_local_date"
    assert uk_warning is None
    assert uk_day == _dt.date(2026, 8, 14)


def test_unmapped_calendar_uses_utc_fallback_label():
    day, timezone_name, source, warning = qtools._default_valuation_date(
        "NullCalendar",
        now_utc=_dt.datetime(2026, 8, 14, 1, 10, tzinfo=_dt.timezone.utc),
    )

    assert timezone_name == "UTC"
    assert source == "default_calendar_local_date"
    assert warning is None
    assert day == _dt.date(2026, 8, 14)


def _yahoo_proxy_contract(
    strike: float,
    *,
    implied_volatility: float = 0.25,
    side: str = "call",
    observed_at: str = "2026-12-01T20:00:00Z",
    exercise_style: str = "american",
) -> dict:
    contract = _qualified_contract(
        strike,
        implied_volatility=implied_volatility,
        side=side,
        observed_at=observed_at,
    )
    contract.update(
        {
            "quote_usable_for_live_analysis": False,
            "last_trade_recent_and_market_two_sided": True,
            "contract_freshness": "last_trade_proxy",
            "exercise_style": exercise_style,
        }
    )
    return contract


def _yahoo_chain_payload(contracts, *, observed_at="2026-12-01T20:00:00Z"):
    return {
        "success": True,
        "provider": "yahoo",
        "providers_used": ["yahoo"],
        "cached": False,
        "retrieved_at": observed_at,
        "symbol": "AAPL",
        "expiration": "2026-12-19",
        "underlying_price": 100.0,
        **_current_chain_snapshot(observed_at),
        "underlying_price_source": "yahoo_regular_market_price",
        "underlying_price_session": "regular_market",
        "option_chain_freshness": "unknown",
        "option_chain_quality": "proxy_usable",
        "underlying_quote": {
            "scope": "underlying_quote",
            "exchange": "NMS",
            "market_state": "POSTPOST",
        },
        "options": contracts,
    }


def test_calibrate_heston_keeps_last_trade_proxy_contracts_quote_usable(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "QuantLib", _make_fake_quantlib())
    monkeypatch.setattr(
        qtools,
        "get_options_chain",
        lambda **_kwargs: _yahoo_chain_payload(
            [
                _yahoo_proxy_contract(90.0, implied_volatility=0.35),
                _yahoo_proxy_contract(95.0, implied_volatility=0.30),
                _yahoo_proxy_contract(100.0, implied_volatility=0.28),
                _yahoo_proxy_contract(105.0, implied_volatility=0.29),
                _yahoo_proxy_contract(110.0, implied_volatility=0.33),
                _yahoo_proxy_contract(115.0, implied_volatility=0.37),
                _yahoo_proxy_contract(120.0, implied_volatility=0.40),
            ]
        ),
    )

    out = qtools.calibrate_heston_quantlib_from_options(
        symbol="AAPL",
        expiration="2026-12-19",
        max_contracts=7,
    )

    assert out["success"] is True
    assert out["contracts_used"] == 7
    assert out["selected_contracts_current_count"] == 7
    assert out["selected_contracts_quote_usable_count"] == 7
    assert out["calibration_data_status"] == "current"
    assert out["quote_freshness_policy"] == "last_trade_proxy"
    assert out["usable_for_pricing"] is True
    assert out["sample_contracts"][0]["last_trade_recent_and_market_two_sided"] is True
    assert out["sample_contracts"][0]["quote_usable_for_live_analysis"] is False


def test_calibrate_heston_labels_american_surface_as_european_approximation(
    monkeypatch,
):
    monkeypatch.setitem(__import__("sys").modules, "QuantLib", _make_fake_quantlib())
    monkeypatch.setattr(
        qtools,
        "get_options_chain",
        lambda **_kwargs: _yahoo_chain_payload(
            [
                _yahoo_proxy_contract(strike, implied_volatility=iv)
                for strike, iv in (
                    (90, 0.35),
                    (95, 0.30),
                    (100, 0.28),
                    (105, 0.29),
                    (110, 0.33),
                )
            ]
        ),
    )

    out = qtools.calibrate_heston_quantlib_from_options(
        symbol="AAPL",
        expiration="2026-12-19",
    )

    assert out["american_surface_approximated_as_european"] is True
    assert out["selected_exercise_styles"] == ["american"]
    assert out["pricing_assumptions"]["american_surface_approximated_as_european"] == "true"
    assert any("American-exercise" in warning for warning in out["warnings"])
    assert any(
        "american_surface_approximated_as_european" in item
        for item in out["identification_limitations"]
    )
    assert out["sample_contracts"][0]["exercise_style"] == "american"
    assert out["usable_for_pricing"] is True


def test_calibrate_heston_rejection_keeps_provider_provenance(monkeypatch):
    fake = _make_fake_quantlib()
    monkeypatch.setitem(__import__("sys").modules, "QuantLib", fake)
    stale_contracts = []
    for strike in (90, 95, 100, 105, 110):
        contract = _yahoo_proxy_contract(strike)
        contract.update(
            {
                "contract_as_of": "2026-11-30T20:00:00Z",
                "contract_data_stale": True,
                "last_trade_recent_and_market_two_sided": False,
            }
        )
        stale_contracts.append(contract)
    monkeypatch.setattr(
        qtools,
        "get_options_chain",
        lambda **_kwargs: _yahoo_chain_payload(stale_contracts),
    )

    out = qtools.calibrate_heston_quantlib_from_options(
        symbol="AAPL",
        expiration="2026-12-19",
    )

    assert out["error_code"] == "heston_contract_inputs_rejected"
    assert out["provider"] == "yahoo"
    assert out["providers_used"] == ["yahoo"]
    assert out["cached"] is False
    assert out["retrieved_at"] == "2026-12-01T20:00:00Z"
    assert out["market_state"] == "POSTPOST"
    assert out["underlying_quote"]["market_state"] == "POSTPOST"


def test_price_barrier_option_quantlib_heston_mode(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "QuantLib", _make_fake_quantlib())
    out = qtools.price_barrier_option_quantlib(
        spot=100.0,
        strike=100.0,
        barrier=120.0,
        maturity_days=30,
        option_type="call",
        barrier_type="up_out",
        model="heston",
        heston_v0=0.04,
        heston_kappa=1.5,
        heston_theta=0.04,
        heston_sigma=0.3,
        heston_rho=-0.5,
        valuation_date="2026-07-03",
    )

    assert out["success"] is True
    assert out["price"] > 0.0
    assert "FdHestonBarrierEngine" in out["pricing_assumptions"]["model"]
    assert out["params_used"]["model"] == "heston"
    assert out["params_used"]["heston_v0"] == 0.04
    assert "volatility" not in out["params_used"]
    assert out["vega"] is None
    assert out["greeks_status"] == "complete"


def test_price_barrier_option_quantlib_rejects_heston_params_without_model():
    out = qtools.price_barrier_option_quantlib(
        spot=100.0,
        strike=100.0,
        barrier=120.0,
        maturity_days=30,
        heston_v0=0.04,
    )

    assert out["error_code"] == "invalid_parameter"
    assert out["parameter"] == "model"
