from __future__ import annotations

"""QuantLib-based pricing and calibration helpers."""

import datetime as _dt
import math
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np

from ..services.options_service import get_options_chain, get_options_expirations

_DEFAULT_QUANTLIB_CALENDAR = "UnitedStates.NYSE"
_DEFAULT_MATURITY_BASIS = "calendar_days"
_HESTON_CONTRACT_SPOT_SKEW_LIMIT_SECONDS = 900.0
_QUANTLIB_CALENDAR_TIMEZONES = {
    "NullCalendar": "UTC",
    "TARGET": "Europe/Brussels",
    "UnitedStates.NYSE": "America/New_York",
}


def _build_bs_merton_process(
    ql: Any,
    ql_today: Any,
    calendar_obj: Any,
    spot: float,
    rf: float,
    div: float,
    vol: float,
    day_count: Any,
) -> Any:
    spot_h = ql.QuoteHandle(ql.SimpleQuote(float(spot)))
    rf_ts = ql.YieldTermStructureHandle(ql.FlatForward(ql_today, float(rf), day_count))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(ql_today, float(div), day_count))
    vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(ql_today, calendar_obj, float(vol), day_count)
    )
    return ql.BlackScholesMertonProcess(spot_h, div_ts, rf_ts, vol_ts)


def _quantlib_pricing_assumptions(
    model: str,
    *,
    calendar: str,
    maturity_basis: str,
) -> Dict[str, str]:
    return {
        "source": "QuantLib",
        "model": model,
        "day_count": "Actual365Fixed",
        "calendar": calendar,
        "rate_compounding": "continuous_flat",
        "maturity_basis": maturity_basis,
    }


def _normalize_quantlib_calendar_name(calendar: Any) -> str:
    name = str(calendar or _DEFAULT_QUANTLIB_CALENDAR).strip()
    return name or _DEFAULT_QUANTLIB_CALENDAR


def _normalize_maturity_basis(maturity_basis: Any) -> str:
    value = str(maturity_basis or _DEFAULT_MATURITY_BASIS).strip().lower()
    if value not in {"calendar_days", "business_days"}:
        raise ValueError(
            f"Invalid maturity_basis: {maturity_basis}. "
            "Use calendar_days|business_days."
        )
    return value


def _valuation_timezone_for_calendar(calendar_name: Any) -> str:
    normalized_name = _normalize_quantlib_calendar_name(calendar_name)
    return _QUANTLIB_CALENDAR_TIMEZONES.get(normalized_name, "UTC")


def _default_valuation_date(
    calendar_name: Any,
    *,
    now_utc: Optional[_dt.datetime] = None,
) -> tuple[_dt.date, str]:
    timezone_name = _valuation_timezone_for_calendar(calendar_name)
    current_utc = now_utc or _dt.datetime.now(_dt.timezone.utc)
    return current_utc.astimezone(ZoneInfo(timezone_name)).date(), timezone_name


def _resolve_quantlib_calendar(ql: Any, calendar_name: Any) -> tuple[Any, str]:
    normalized_name = _normalize_quantlib_calendar_name(calendar_name)
    if "." in normalized_name:
        class_name, market_name = normalized_name.split(".", 1)
        calendar_factory = getattr(ql, class_name, None)
        market = getattr(calendar_factory, market_name, None) if calendar_factory is not None else None
        if calendar_factory is None or market is None:
            raise ValueError(
                f"Invalid calendar: {normalized_name}. "
                "Use QuantLib calendar names such as UnitedStates.NYSE or NullCalendar."
            )
        return calendar_factory(market), normalized_name
    calendar_factory = getattr(ql, normalized_name, None)
    if calendar_factory is None:
        raise ValueError(
            f"Invalid calendar: {normalized_name}. "
            "Use QuantLib calendar names such as UnitedStates.NYSE or NullCalendar."
        )
    try:
        return calendar_factory(), normalized_name
    except TypeError as exc:
        raise ValueError(
            f"Invalid calendar: {normalized_name}. "
            "Use QuantLib calendar names such as UnitedStates.NYSE or NullCalendar."
        ) from exc


def _quantlib_date(ql: Any, day: _dt.date) -> Any:
    return ql.Date(int(day.day), int(day.month), int(day.year))


def _python_date_from_quantlib(value: Any) -> _dt.date:
    return _dt.date(
        int(value.year()),
        int(value.month()),
        int(value.dayOfMonth()),
    )


def _advance_maturity_date(
    *,
    ql: Any,
    ql_today: Any,
    calendar: Any,
    maturity_days: int,
    maturity_basis: str,
) -> Any:
    if maturity_basis == "business_days":
        return calendar.advance(ql_today, int(maturity_days), ql.Days)
    return ql_today + int(maturity_days)


def _days_to_expiry(
    *,
    ql: Any,
    calendar: Any,
    valuation_day: _dt.date,
    expiry_date: _dt.date,
    maturity_basis: str,
) -> int:
    if maturity_basis == "business_days":
        return int(
            calendar.businessDaysBetween(
                _quantlib_date(ql, valuation_day),
                _quantlib_date(ql, expiry_date),
            )
        )
    return int((expiry_date - valuation_day).days)


def price_barrier_option_quantlib(
    *,
    spot: float,
    strike: float,
    barrier: float,
    maturity_days: int,
    option_type: str = "call",
    barrier_type: str = "up_out",
    risk_free_rate: float = 0.02,
    dividend_yield: float = 0.0,
    volatility: float = 0.2,
    rebate: float = 0.0,
    valuation_date: Optional[str] = None,
    calendar: str = _DEFAULT_QUANTLIB_CALENDAR,
    maturity_basis: str = _DEFAULT_MATURITY_BASIS,
) -> Dict[str, Any]:
    """Price a European barrier option with QuantLib."""
    try:
        spot_val = float(spot)
        strike_val = float(strike)
        barrier_val = float(barrier)
        maturity_val = int(maturity_days)
        rf = float(risk_free_rate)
        div = float(dividend_yield)
        vol = float(volatility)
        rebate_val = float(rebate)
    except Exception:
        return {"error": "Invalid numeric input for barrier pricing"}

    if not (spot_val > 0 and strike_val > 0 and barrier_val > 0 and maturity_val > 0 and vol > 0):
        return {"error": "spot/strike/barrier/maturity_days/volatility must be positive"}
    if not math.isfinite(rebate_val) or rebate_val < 0:
        return {
            "error": "rebate must be finite and nonnegative",
            "error_code": "invalid_rebate",
        }

    option_type_norm = str(option_type).strip().lower()
    barrier_type_norm = str(barrier_type).strip().lower()
    opt_choices = {"call", "put"}
    barrier_choices = {"up_in", "up_out", "down_in", "down_out"}
    if option_type_norm not in opt_choices:
        return {"error": f"Invalid option_type: {option_type}. Use call|put."}
    if barrier_type_norm not in barrier_choices:
        return {"error": f"Invalid barrier_type: {barrier_type}. Use up_in|up_out|down_in|down_out."}
    try:
        maturity_basis_norm = _normalize_maturity_basis(maturity_basis)
    except ValueError as ex:
        return {"error": str(ex)}
    calendar_name = _normalize_quantlib_calendar_name(calendar)
    valuation_timezone = _valuation_timezone_for_calendar(calendar_name)
    if valuation_date is None:
        valuation_day, valuation_timezone = _default_valuation_date(calendar_name)
        valuation_date_source = "default_calendar_local_date"
    else:
        try:
            valuation_day = _dt.datetime.strptime(
                str(valuation_date).strip(),
                "%Y-%m-%d",
            ).date()
        except (TypeError, ValueError):
            return {
                "error": (
                    f"Invalid valuation_date: {valuation_date}. Use YYYY-MM-DD."
                )
            }
        valuation_date_source = "explicit"

    barrier_status = _barrier_option_status(
        barrier_type=barrier_type_norm,
        spot=spot_val,
        barrier=barrier_val,
    )
    if barrier_status == "knocked_out":
        params_used = _barrier_option_params(
            spot=spot_val,
            strike=strike_val,
            barrier=barrier_val,
            maturity_days=maturity_val,
            option_type=option_type_norm,
            barrier_type=barrier_type_norm,
            risk_free_rate=rf,
            dividend_yield=div,
            volatility=vol,
            rebate=rebate_val,
            calendar=calendar_name,
            maturity_basis=maturity_basis_norm,
            valuation_date=valuation_day.isoformat(),
        )
        return {
            "success": True,
            "price": float(rebate_val),
            "status": "knocked_out",
            "option_status": "knocked_out",
            "valuation_date": valuation_day.isoformat(),
            "valuation_timezone": valuation_timezone,
            "valuation_date_source": valuation_date_source,
            "delta": 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "greeks_status": "complete",
            "greeks_method": "knocked_out_boundary",
            "pricing_assumptions": _quantlib_pricing_assumptions(
                "already-breached knock-out (rebate)",
                calendar=calendar_name,
                maturity_basis=maturity_basis_norm,
            ),
            "params_used": params_used,
        }

    try:
        import QuantLib as ql
    except Exception as ex:
        return {"error": f"QuantLib is required: {ex}"}

    try:
        calendar_obj, calendar_name = _resolve_quantlib_calendar(ql, calendar_name)
    except ValueError as ex:
        return {"error": str(ex)}

    if barrier_status == "knocked_in":
        return _price_knocked_in_as_vanilla(
            ql=ql,
            spot_val=spot_val,
            strike_val=strike_val,
            barrier_val=barrier_val,
            maturity_days=maturity_val,
            option_type_norm=option_type_norm,
            barrier_type_norm=barrier_type_norm,
            rf=rf,
            div=div,
            vol=vol,
            rebate_val=rebate_val,
            calendar_name=calendar_name,
            calendar_obj=calendar_obj,
            maturity_basis_norm=maturity_basis_norm,
            valuation_day=valuation_day,
            valuation_timezone=valuation_timezone,
            valuation_date_source=valuation_date_source,
        )

    opt_map = {"call": ql.Option.Call, "put": ql.Option.Put}
    barrier_map = {
        "up_in": ql.Barrier.UpIn,
        "up_out": ql.Barrier.UpOut,
        "down_in": ql.Barrier.DownIn,
        "down_out": ql.Barrier.DownOut,
    }

    ql_today = _quantlib_date(ql, valuation_day)
    ql.Settings.instance().evaluationDate = ql_today
    day_count = ql.Actual365Fixed()
    maturity = _advance_maturity_date(
        ql=ql,
        ql_today=ql_today,
        calendar=calendar_obj,
        maturity_days=maturity_val,
        maturity_basis=maturity_basis_norm,
    )
    maturity_day = _python_date_from_quantlib(maturity)
    time_to_maturity_years = float((maturity - ql_today) / 365.0)

    payoff = ql.PlainVanillaPayoff(opt_map[option_type_norm], float(strike_val))
    exercise = ql.EuropeanExercise(maturity)
    barrier_opt = ql.BarrierOption(
        barrier_map[barrier_type_norm],
        float(barrier_val),
        float(rebate_val),
        payoff,
        exercise,
    )

    def _price_with(spot_local: float, vol_local: float) -> float:
        process = _build_bs_merton_process(
            ql,
            ql_today,
            calendar_obj,
            spot_local,
            rf,
            div,
            vol_local,
            day_count,
        )
        barrier_opt.setPricingEngine(ql.AnalyticBarrierEngine(process))
        return float(barrier_opt.NPV())

    try:
        npv = _price_with(spot_val, vol)
    except Exception as ex:
        return {"error": f"QuantLib pricing failed: {ex}"}

    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    greeks_method: Optional[str] = None
    greeks_warnings: List[str] = []

    barrier_distance = abs(barrier_val - spot_val)
    eps_s = min(
        max(1e-6, abs(spot_val) * 1e-4),
        barrier_distance * 0.25,
        spot_val * 0.25,
    )
    try:
        p_up = _price_with(spot_val + eps_s, vol)
        p_dn = _price_with(spot_val - eps_s, vol)
        delta = (p_up - p_dn) / (2.0 * eps_s)
        gamma = (p_up - 2.0 * npv + p_dn) / (eps_s * eps_s)
        greeks_method = "central_difference"
    except Exception as central_exc:
        try:
            direction = -1.0 if barrier_type_norm.startswith("up_") else 1.0
            p_1 = _price_with(spot_val + direction * eps_s, vol)
            p_2 = _price_with(spot_val + direction * 2.0 * eps_s, vol)
            p_3 = _price_with(spot_val + direction * 3.0 * eps_s, vol)
            delta = direction * (-3.0 * npv + 4.0 * p_1 - p_2) / (2.0 * eps_s)
            gamma = (2.0 * npv - 5.0 * p_1 + 4.0 * p_2 - p_3) / (
                eps_s * eps_s
            )
            greeks_method = "one_sided_away_from_barrier"
            greeks_warnings.append(
                f"Central spot differences failed ({central_exc}); used a one-sided "
                "difference away from the barrier."
            )
        except Exception as one_sided_exc:
            greeks_warnings.append(
                "Spot Greeks unavailable: central and one-sided differences failed "
                f"({one_sided_exc})."
            )

    try:
        eps_v = max(1e-4, abs(vol) * 5e-2)
        pv_up = _price_with(spot_val, vol + eps_v)
        pv_dn = _price_with(spot_val, max(1e-6, vol - eps_v))
        vega = (pv_up - pv_dn) / (2.0 * eps_v)
    except Exception as ex:
        greeks_warnings.append(f"Vega unavailable: volatility differences failed ({ex}).")

    finite_greeks = sum(
        value is not None and math.isfinite(value)
        for value in (delta, gamma, vega)
    )
    greeks_status = "complete" if finite_greeks == 3 else "partial" if finite_greeks else "unavailable"

    return {
        "success": True,
        "price": float(npv),
        "valuation_date": valuation_day.isoformat(),
        "valuation_timezone": valuation_timezone,
        "valuation_date_source": valuation_date_source,
        "maturity_date": maturity_day.isoformat(),
        "time_to_maturity_years": time_to_maturity_years,
        "delta": float(delta) if delta is not None and math.isfinite(delta) else None,
        "gamma": float(gamma) if gamma is not None and math.isfinite(gamma) else None,
        "vega": float(vega) if vega is not None and math.isfinite(vega) else None,
        "greeks_status": greeks_status,
        "greeks_method": greeks_method,
        "greeks_spot_step": float(eps_s),
        **({"greeks_warnings": greeks_warnings} if greeks_warnings else {}),
        "pricing_assumptions": _quantlib_pricing_assumptions(
            "BlackScholesMerton analytic barrier",
            calendar=calendar_name,
            maturity_basis=maturity_basis_norm,
        ),
        "params_used": _barrier_option_params(
            spot=spot_val,
            strike=strike_val,
            barrier=barrier_val,
            maturity_days=maturity_val,
            option_type=option_type_norm,
            barrier_type=barrier_type_norm,
            risk_free_rate=rf,
            dividend_yield=div,
            volatility=vol,
            rebate=rebate_val,
            calendar=calendar_name,
            maturity_basis=maturity_basis_norm,
            valuation_date=valuation_day.isoformat(),
        ),
    }


def _price_knocked_in_as_vanilla(
    *,
    ql: Any,
    spot_val: float,
    strike_val: float,
    barrier_val: float,
    maturity_days: int,
    option_type_norm: str,
    barrier_type_norm: str,
    rf: float,
    div: float,
    vol: float,
    rebate_val: float,
    calendar_name: str,
    calendar_obj: Any,
    maturity_basis_norm: str,
    valuation_day: Any,
    valuation_timezone: str,
    valuation_date_source: str,
) -> Dict[str, Any]:
    """Price a barrier that has already knocked in as the equivalent vanilla."""
    ql_today = _quantlib_date(ql, valuation_day)
    ql.Settings.instance().evaluationDate = ql_today
    day_count = ql.Actual365Fixed()
    maturity = _advance_maturity_date(
        ql=ql,
        ql_today=ql_today,
        calendar=calendar_obj,
        maturity_days=maturity_days,
        maturity_basis=maturity_basis_norm,
    )
    maturity_day = _python_date_from_quantlib(maturity)
    time_to_maturity_years = float((maturity - ql_today) / 365.0)
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if option_type_norm == "call" else ql.Option.Put,
        float(strike_val),
    )
    exercise = ql.EuropeanExercise(maturity)
    option = ql.VanillaOption(payoff, exercise)
    process = _build_bs_merton_process(
        ql,
        ql_today,
        calendar_obj,
        spot_val,
        rf,
        div,
        vol,
        day_count,
    )
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    try:
        npv = float(option.NPV())
        delta = float(option.delta())
        gamma = float(option.gamma())
        vega = float(option.vega())
    except Exception as ex:
        return {"error": f"QuantLib pricing failed: {ex}"}
    return {
        "success": True,
        "price": npv,
        "status": "knocked_in",
        "option_status": "knocked_in",
        "valuation_date": valuation_day.isoformat(),
        "valuation_timezone": valuation_timezone,
        "valuation_date_source": valuation_date_source,
        "maturity_date": maturity_day.isoformat(),
        "time_to_maturity_years": time_to_maturity_years,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "greeks_status": "complete",
        "greeks_method": "knocked_in_vanilla",
        "pricing_assumptions": _quantlib_pricing_assumptions(
            "already-breached knock-in priced as European vanilla",
            calendar=calendar_name,
            maturity_basis=maturity_basis_norm,
        ),
        "params_used": _barrier_option_params(
            spot=spot_val,
            strike=strike_val,
            barrier=barrier_val,
            maturity_days=maturity_days,
            option_type=option_type_norm,
            barrier_type=barrier_type_norm,
            risk_free_rate=rf,
            dividend_yield=div,
            volatility=vol,
            rebate=rebate_val,
            calendar=calendar_name,
            maturity_basis=maturity_basis_norm,
            valuation_date=valuation_day.isoformat(),
        ),
    }


def _barrier_option_status(
    *,
    barrier_type: str,
    spot: float,
    barrier: float,
) -> Optional[str]:
    kind = str(barrier_type)
    if kind.startswith("up_") and barrier <= spot:
        return "knocked_out" if kind.endswith("_out") else "knocked_in"
    if kind.startswith("down_") and barrier >= spot:
        return "knocked_out" if kind.endswith("_out") else "knocked_in"
    return None


def _barrier_option_params(
    *,
    spot: float,
    strike: float,
    barrier: float,
    maturity_days: int,
    option_type: str,
    barrier_type: str,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,
    rebate: float,
    calendar: str,
    maturity_basis: str,
    valuation_date: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "spot": float(spot),
        "strike": float(strike),
        "barrier": float(barrier),
        "maturity_days": int(maturity_days),
        "option_type": str(option_type),
        "barrier_type": str(barrier_type),
        "risk_free_rate": float(risk_free_rate),
        "dividend_yield": float(dividend_yield),
        "volatility": float(volatility),
        "rebate": float(rebate),
        "calendar": str(calendar),
        "maturity_basis": str(maturity_basis),
        **({"valuation_date": valuation_date} if valuation_date else {}),
    }


def _heston_pricing_assumptions(
    *,
    calendar: str,
    days_to_expiry_basis: str,
) -> Dict[str, str]:
    assumptions = _quantlib_pricing_assumptions(
        "Heston analytic calibration",
        calendar=calendar,
        maturity_basis=days_to_expiry_basis,
    )
    assumptions.pop("maturity_basis", None)
    assumptions["maturity_convention"] = "calendar_days_to_contract_expiry"
    assumptions["days_to_expiry_basis"] = days_to_expiry_basis
    return assumptions


def _chain_observation_date(
    as_of: Any,
    *,
    timezone_name: str,
) -> Optional[_dt.date]:
    if as_of in (None, ""):
        return None
    try:
        text = str(as_of).strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        observed_at = _dt.datetime.fromisoformat(text)
        if observed_at.tzinfo is None:
            return None
        return observed_at.astimezone(ZoneInfo(timezone_name)).date()
    except (KeyError, TypeError, ValueError):
        return None


def _timezone_qualified_epoch(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        text = str(value).strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        observed_at = _dt.datetime.fromisoformat(text)
        if observed_at.tzinfo is None:
            return None
        return float(observed_at.timestamp())
    except (TypeError, ValueError):
        return None


def _heston_contract_quality(
    row: Dict[str, Any],
    *,
    spot_epoch: float,
) -> tuple[List[str], Optional[float]]:
    failures: List[str] = []
    contract_epoch = _timezone_qualified_epoch(row.get("contract_as_of"))
    if contract_epoch is None:
        failures.append("contract_timestamp_unavailable")
        skew_seconds = None
    else:
        skew_seconds = abs(float(spot_epoch) - contract_epoch)
        if skew_seconds > _HESTON_CONTRACT_SPOT_SKEW_LIMIT_SECONDS:
            failures.append("contract_spot_timestamp_mismatch")
    if row.get("contract_data_stale") is not False:
        failures.append(
            "stale_contract_timestamp"
            if row.get("contract_data_stale") is True
            else "contract_freshness_unqualified"
        )
    if row.get("quote_usable_for_live_analysis") is not True:
        failures.append("contract_quote_not_live_usable")
    return failures, skew_seconds


def calibrate_heston_quantlib_from_options(  # noqa: C901
    *,
    symbol: str,
    expiration: Optional[str] = None,
    option_type: str = "call",
    risk_free_rate: float = 0.02,
    dividend_yield: float = 0.0,
    min_open_interest: int = 0,
    min_volume: int = 0,
    max_contracts: int = 25,
    valuation_date: Optional[str] = None,
    calendar: str = _DEFAULT_QUANTLIB_CALENDAR,
    maturity_basis: str = _DEFAULT_MATURITY_BASIS,
) -> Dict[str, Any]:
    """Calibrate a Heston model from option-chain implied vols using QuantLib."""
    if int(min_open_interest) < 0:
        return {"error": "min_open_interest must be greater than or equal to 0."}
    if int(min_volume) < 0:
        return {"error": "min_volume must be greater than or equal to 0."}
    if int(max_contracts) < 5:
        return {"error": "max_contracts must be greater than or equal to 5."}
    if valuation_date is not None:
        try:
            _dt.datetime.strptime(str(valuation_date).strip(), "%Y-%m-%d")
        except (TypeError, ValueError):
            return {
                "error": (
                    f"Invalid valuation_date: {valuation_date}. Use YYYY-MM-DD."
                )
            }
    try:
        import QuantLib as ql
    except Exception as ex:
        return {"error": f"QuantLib is required: {ex}"}

    try:
        maturity_basis_norm = _normalize_maturity_basis(maturity_basis)
    except ValueError as ex:
        return {"error": str(ex)}
    calendar_name = _normalize_quantlib_calendar_name(calendar)
    try:
        calendar_obj, calendar_name = _resolve_quantlib_calendar(ql, calendar_name)
    except ValueError as ex:
        return {"error": str(ex)}

    side = str(option_type or "call").strip().lower()
    if side not in {"call", "put", "both"}:
        return {"error": f"Invalid option_type: {option_type}. Use call|put|both."}

    expiration_selection: Optional[Dict[str, Any]] = None
    if expiration is None:
        expirations_result = get_options_expirations(symbol)
        if isinstance(expirations_result, dict) and expirations_result.get("error"):
            return expirations_result
        listed_raw = (
            expirations_result.get("expirations", [])
            if isinstance(expirations_result, dict)
            else []
        )
        listed_expirations: List[_dt.date] = []
        for value in listed_raw if isinstance(listed_raw, list) else []:
            try:
                listed_expirations.append(
                    _dt.datetime.strptime(str(value).strip(), "%Y-%m-%d").date()
                )
            except (TypeError, ValueError):
                continue
        listed_expirations = sorted(set(listed_expirations))
        valuation_timezone = _valuation_timezone_for_calendar(calendar_name)
        if valuation_date is not None:
            selection_day = _dt.datetime.strptime(
                str(valuation_date).strip(), "%Y-%m-%d"
            ).date()
            selection_date_source = "explicit_valuation_date"
        else:
            selection_day = _chain_observation_date(
                (
                    expirations_result.get("underlying_as_of")
                    or expirations_result.get("as_of")
                )
                if isinstance(expirations_result, dict)
                else None,
                timezone_name=valuation_timezone,
            )
            selection_date_source = "expirations_observation_date"
        if selection_day is None:
            return {
                "error": (
                    "Options provider timestamp is required to select an eligible "
                    "default Heston expiration."
                ),
                "error_code": "expiration_observation_time_unavailable",
                "remediation": (
                    "Pass --expiration explicitly, or retry with a provider response "
                    "that includes a timezone-qualified underlying_as_of timestamp."
                ),
            }
        eligible = [
            value
            for value in listed_expirations
            if (value - selection_day).days >= 7
        ]
        if not eligible:
            return {
                "error": (
                    "No listed option expiration is at least 7 calendar days "
                    "after the provider observation date."
                ),
                "error_code": "heston_no_eligible_expiration",
                "valuation_date": selection_day.isoformat(),
                "minimum_calendar_days": 7,
                "listed_expirations": [
                    value.isoformat() for value in listed_expirations
                ],
                "remediation": (
                    "Retry when a later contract is listed, or pass --expiration "
                    "with an eligible listed date."
                ),
            }
        selected_expiration = eligible[0]
        skipped = [
            value.isoformat()
            for value in listed_expirations
            if value < selected_expiration
        ]
        expiration = selected_expiration.isoformat()
        expiration_selection = {
            "mode": "nearest_eligible",
            "minimum_calendar_days": 7,
            "selection_date": selection_day.isoformat(),
            "selection_date_source": selection_date_source,
            "skipped_ineligible_expirations": skipped,
        }

    chain = get_options_chain(
        symbol=symbol,
        expiration=expiration,
        option_type=side,
        min_open_interest=int(min_open_interest),
        min_volume=int(min_volume),
        limit=max(50, int(max_contracts) * 6),
    )
    if isinstance(chain, dict) and chain.get("error"):
        return chain

    contracts = chain.get("options", []) if isinstance(chain, dict) else []
    if not isinstance(contracts, list):
        contracts = []

    spot_val = float(chain.get("underlying_price", float("nan")))
    if not (spot_val == spot_val and spot_val > 0):
        return {"error": "Underlying spot price unavailable from options provider."}

    spot_as_of = chain.get("underlying_as_of")
    spot_epoch = _timezone_qualified_epoch(spot_as_of)
    if spot_epoch is None:
        return {
            "error": (
                "Options provider timestamp is required to anchor Heston "
                "calibration to a single market snapshot."
            ),
            "error_code": "chain_observation_time_unavailable",
            "spot_as_of": spot_as_of,
            "remediation": (
                "Retry with a provider response that includes a timezone-qualified "
                "underlying_as_of timestamp."
            ),
        }
    valuation_timezone = _valuation_timezone_for_calendar(calendar_name)
    observation_day = _chain_observation_date(
        spot_as_of,
        timezone_name=valuation_timezone,
    )
    if observation_day is None:
        return {
            "error": (
                "Options provider timestamp is required to anchor Heston "
                "calibration to a single market snapshot."
            ),
            "error_code": "chain_observation_time_unavailable",
            "spot_as_of": spot_as_of,
            "valuation_timezone": valuation_timezone,
            "remediation": (
                "Retry with a provider response that includes a timezone-qualified "
                "as_of timestamp."
            ),
        }
    if valuation_date is None:
        valuation_day = observation_day
        valuation_date_source = "chain_observation_date"
    else:
        try:
            valuation_day = _dt.datetime.strptime(
                str(valuation_date).strip(), "%Y-%m-%d"
            ).date()
        except (TypeError, ValueError):
            return {
                "error": f"Invalid valuation_date: {valuation_date}. Use YYYY-MM-DD.",
                "symbol": symbol,
                "expiration": expiration,
                "valuation_date": valuation_date,
            }
        if valuation_day != observation_day:
            return {
                "error": (
                    "valuation_date must match the options chain observation date. "
                    f"Received valuation_date={valuation_day.isoformat()} and "
                    f"chain_observation_date={observation_day.isoformat()}."
                ),
                "error_code": "valuation_date_chain_mismatch",
                "symbol": symbol,
                "expiration": expiration,
                "valuation_date": valuation_day.isoformat(),
                "chain_observation_date": observation_day.isoformat(),
                "spot_as_of": spot_as_of,
                "valuation_timezone": valuation_timezone,
                "remediation": (
                    "Omit valuation_date to use the chain observation date, or request "
                    "a chain snapshot from the intended valuation date."
                ),
            }
        valuation_date_source = "explicit_chain_observation_date"

    rows: List[Dict[str, Any]] = []
    contract_quality_rejections: Dict[str, int] = {}
    contracts_with_valid_iv = 0
    for row in contracts:
        if not isinstance(row, dict):
            continue
        try:
            strike = float(row.get("strike", float("nan")))
            iv = float(row.get("implied_volatility", float("nan")))
        except (TypeError, ValueError):
            strike = float("nan")
            iv = float("nan")
        if not (np.isfinite(strike) and strike > 0 and np.isfinite(iv) and 0.01 <= iv <= 5.0):
            contract_quality_rejections["invalid_strike_or_implied_volatility"] = (
                contract_quality_rejections.get(
                    "invalid_strike_or_implied_volatility",
                    0,
                )
                + 1
            )
            continue
        contracts_with_valid_iv += 1
        quality_failures, skew_seconds = _heston_contract_quality(
            row,
            spot_epoch=spot_epoch,
        )
        if quality_failures:
            for failure in quality_failures:
                contract_quality_rejections[failure] = (
                    contract_quality_rejections.get(failure, 0) + 1
                )
            continue
        rows.append(
            {
                "contract": row.get("contract"),
                "strike": strike,
                "iv": iv,
                "side": row.get("side"),
                "bid": row.get("bid"),
                "ask": row.get("ask"),
                "contract_as_of": row.get("contract_as_of"),
                "contract_data_age_seconds": row.get(
                    "contract_data_age_seconds"
                ),
                "contract_data_stale": row.get("contract_data_stale"),
                "contract_freshness": row.get("contract_freshness"),
                "quote_quality": row.get("quote_quality"),
                "quote_usable_for_live_analysis": row.get(
                    "quote_usable_for_live_analysis"
                ),
                "spot_contract_skew_seconds": (
                    round(float(skew_seconds), 3)
                    if skew_seconds is not None
                    else None
                ),
            }
        )

    if len(rows) < 5:
        return {
            "success": False,
            "error": (
                "Heston calibration requires at least 5 current, timestamped, "
                "two-sided option contracts from the same market snapshot."
            ),
            "error_code": "heston_contract_inputs_rejected",
            "calibration_status": "rejected",
            "calibration_data_status": "unusable_contracts",
            "usable_for_pricing": False,
            "spot_as_of": spot_as_of,
            "symbol": symbol,
            "expiration": expiration,
            "valuation_date": valuation_day.isoformat(),
            "chain_observation_date": observation_day.isoformat(),
            "contracts_available": len(contracts),
            "contracts_with_valid_implied_volatility": contracts_with_valid_iv,
            "contracts_usable": len(rows),
            "minimum_contracts_required": 5,
            "contract_spot_skew_limit_seconds": (
                _HESTON_CONTRACT_SPOT_SKEW_LIMIT_SECONDS
            ),
            "contract_quality_rejections": contract_quality_rejections,
            "pricing_usability_failures": ["unusable_option_contract_inputs"],
            "warnings": [
                (
                    "Calibration was not attempted because fewer than five "
                    "contracts passed timestamp, freshness, quote, and spot-skew checks."
                )
            ],
        }

    rows.sort(key=lambda x: abs(float(x["strike"]) - spot_val))
    contract_limit = int(max_contracts)
    if side == "both":
        calls = [row for row in rows if row.get("side") == "call"]
        puts = [row for row in rows if row.get("side") == "put"]
        if not calls or not puts:
            return {
                "error": (
                    "option_type=both requires valid implied-volatility contracts "
                    "from both calls and puts."
                ),
                "side_coverage": "call_only" if calls else "put_only" if puts else "none",
            }
        rows = []
        for index in range(max(len(calls), len(puts))):
            if index < len(calls) and len(rows) < contract_limit:
                rows.append(calls[index])
            if index < len(puts) and len(rows) < contract_limit:
                rows.append(puts[index])
            if len(rows) >= contract_limit:
                break
    else:
        rows = rows[:contract_limit]
    expiry_text = str(chain.get("expiration") or "")
    if not expiry_text:
        return {"error": "Options expiration date missing from chain output."}
    try:
        expiry_date = _dt.datetime.strptime(expiry_text, "%Y-%m-%d").date()
    except Exception:
        return {"error": f"Invalid expiration format: {expiry_text}"}
    if valuation_day >= expiry_date:
        return {
            "error": (
                "valuation_date must be before the option expiration date. "
                f"Received valuation_date={valuation_day.isoformat()} and "
                f"expiration={expiry_date.isoformat()}."
            ),
            "error_code": "invalid_expiration_date_range",
            "valuation_date": valuation_day.isoformat(),
            "expiration": expiry_date.isoformat(),
        }
    days_to_expiry = _days_to_expiry(
        ql=ql,
        calendar=calendar_obj,
        valuation_day=valuation_day,
        expiry_date=expiry_date,
        maturity_basis=maturity_basis_norm,
    )
    if days_to_expiry <= 0:
        return {
            "error": (
                "The selected calendar and maturity basis produce no positive "
                "time to expiration."
            ),
            "error_code": "invalid_expiration_maturity",
            "valuation_date": valuation_day.isoformat(),
            "expiration": expiry_date.isoformat(),
            "days_to_expiry_basis": maturity_basis_norm,
        }
    maturity_calendar_days = int((expiry_date - valuation_day).days)
    if maturity_calendar_days < 7:
        return {
            "error": (
                "Heston calibration requires at least 7 calendar days to "
                "expiration; the selected chain expires in "
                f"{maturity_calendar_days} day(s)."
            ),
            "error_code": "heston_maturity_too_short",
            "expiration": expiry_date.isoformat(),
            "valuation_date": valuation_day.isoformat(),
            "days_to_expiry": int(days_to_expiry),
            "calendar_days_to_expiry": maturity_calendar_days,
            "minimum_calendar_days": 7,
            "remediation": (
                "Pass --expiration with a listed expiry at least 7 calendar "
                "days after the chain observation date."
            ),
        }

    ql_today = _quantlib_date(ql, valuation_day)
    ql.Settings.instance().evaluationDate = ql_today
    day_count = ql.Actual365Fixed()
    rf_ts = ql.YieldTermStructureHandle(ql.FlatForward(ql_today, float(risk_free_rate), day_count))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(ql_today, float(dividend_yield), day_count))
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(float(spot_val)))

    ivs = np.asarray([float(r["iv"]) for r in rows], dtype=float)
    theta0 = float(max(1e-6, np.median(ivs) ** 2))
    v0_0 = float(theta0)
    kappa0 = 1.0
    sigma0 = float(max(0.05, np.std(ivs) * 2.0))
    rho0 = -0.5

    process = ql.HestonProcess(rf_ts, div_ts, spot_handle, v0_0, kappa0, theta0, sigma0, rho0)
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model)
    helpers: List[Any] = []
    # HestonModelHelper advances Period(Days) through the supplied calendar.
    # A calendar-day delta therefore requires NullCalendar to preserve the
    # contract's actual expiry date instead of skipping weekends and holidays.
    maturity = ql.Period(maturity_calendar_days, ql.Days)
    maturity_calendar = ql.NullCalendar()
    for row in rows:
        helper = ql.HestonModelHelper(
            maturity,
            maturity_calendar,
            float(spot_val),
            float(row["strike"]),
            ql.QuoteHandle(ql.SimpleQuote(float(row["iv"]))),
            rf_ts,
            div_ts,
            ql.BlackCalibrationHelper.ImpliedVolError,
        )
        helper.setPricingEngine(engine)
        helpers.append(helper)

    try:
        method = ql.LevenbergMarquardt()
        end_criteria = ql.EndCriteria(500, 100, 1e-8, 1e-8, 1e-8)
        model.calibrate(helpers, method, end_criteria)
    except Exception as ex:
        return {"error": f"QuantLib Heston calibration failed: {ex}"}

    errors = [float(h.calibrationError()) for h in helpers]
    rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else float("nan")
    spot_data_stale = chain.get("underlying_data_stale")
    spot_freshness = chain.get("underlying_freshness")
    selected_contracts_current_count = sum(
        1 for row in rows if row.get("contract_data_stale") is False
    )
    selected_contracts_quote_usable_count = sum(
        1
        for row in rows
        if row.get("quote_usable_for_live_analysis") is True
    )
    selected_contract_max_spot_skew_seconds = max(
        (
            float(row["spot_contract_skew_seconds"])
            for row in rows
            if row.get("spot_contract_skew_seconds") is not None
        ),
        default=None,
    )
    selected_contracts_qualified = (
        selected_contracts_current_count == len(rows)
        and selected_contracts_quote_usable_count == len(rows)
        and selected_contract_max_spot_skew_seconds is not None
        and selected_contract_max_spot_skew_seconds
        <= _HESTON_CONTRACT_SPOT_SKEW_LIMIT_SECONDS
    )
    calibration_data_stale = spot_data_stale is True
    calibration_data_status = (
        "stale"
        if calibration_data_stale
        else "current"
        if spot_data_stale is False and selected_contracts_qualified
        else "unqualified"
    )
    warnings = (
        [
            "Heston calibration used a stale underlying quote; the "
            "fitted parameters are not usable for pricing until a current "
            "snapshot is calibrated."
        ]
        if calibration_data_stale
        else []
    )
    if spot_data_stale is not False and not calibration_data_stale:
        warnings.append(
            "Underlying quote freshness is unqualified; the fitted parameters "
            "are not usable for pricing until a timestamped current snapshot is calibrated."
        )
    if not selected_contracts_qualified:
        warnings.append(
            "Selected option contracts did not all pass timestamp, freshness, "
            "two-sided quote, and spot-skew checks."
        )
    params = {
        "kappa": float(model.kappa()),
        "theta": float(model.theta()),
        "sigma": float(model.sigma()),
        "rho": float(model.rho()),
        "v0": float(model.v0()),
    }
    feller_left = 2.0 * params["kappa"] * params["theta"]
    feller_right = params["sigma"] ** 2
    feller_satisfied = bool(feller_left >= feller_right)
    rho_at_bound = bool(abs(params["rho"]) >= 0.98)
    quality_failures: List[str] = []
    if not np.isfinite(rmse) or rmse > 0.15:
        quality_failures.append("implied_vol_rmse_exceeds_0.15")
    if rho_at_bound:
        quality_failures.append("rho_at_calibration_bound")
    if params["kappa"] <= 1e-4:
        quality_failures.append("kappa_near_zero")
    if not feller_satisfied:
        warnings.append(
            "Calibrated parameters violate the Feller condition; variance can "
            "approach zero and simulation schemes require extra care."
        )
    if quality_failures:
        warnings.append(
            "Heston calibration failed pricing-usability checks: "
            + ", ".join(quality_failures)
            + "."
        )
    pricing_usability_failures = list(quality_failures)
    if calibration_data_stale:
        pricing_usability_failures.append("stale_underlying_data")
    elif spot_data_stale is not False:
        pricing_usability_failures.append("underlying_freshness_unqualified")
    if not selected_contracts_qualified:
        pricing_usability_failures.append("option_contract_inputs_unqualified")
    usable_for_pricing = not pricing_usability_failures

    return {
        "success": usable_for_pricing,
        **(
            {
                "error": (
                    "Heston calibration is not usable for pricing: "
                    + ", ".join(pricing_usability_failures)
                    + "."
                ),
                "error_code": "heston_calibration_rejected",
            }
            if not usable_for_pricing
            else {}
        ),
        "symbol": str(symbol).upper().strip(),
        "expiration": expiry_text,
        **(
            {"expiration_selection": expiration_selection}
            if expiration_selection is not None
            else {}
        ),
        "valuation_date": valuation_day.isoformat(),
        "valuation_timezone": valuation_timezone,
        "valuation_date_source": valuation_date_source,
        "days_to_expiry": int(days_to_expiry),
        "contracts_used": int(len(rows)),
        "option_type": side,
        "calls_used": sum(1 for row in rows if row.get("side") == "call"),
        "puts_used": sum(1 for row in rows if row.get("side") == "put"),
        "spot": float(spot_val),
        "spot_as_of": spot_as_of,
        "spot_data_age_seconds": chain.get("underlying_data_age_seconds"),
        "spot_data_stale": spot_data_stale,
        "spot_freshness": spot_freshness,
        "spot_freshness_reason": chain.get("underlying_freshness_reason"),
        "spot_source": chain.get("underlying_price_source"),
        "spot_session": chain.get("underlying_price_session"),
        "option_chain_freshness": chain.get("option_chain_freshness"),
        "option_chain_quality": chain.get("option_chain_quality"),
        "selected_contracts_current_count": (
            selected_contracts_current_count
        ),
        "selected_contracts_quote_usable_count": (
            selected_contracts_quote_usable_count
        ),
        "selected_contract_max_spot_skew_seconds": (
            selected_contract_max_spot_skew_seconds
        ),
        "contract_spot_skew_limit_seconds": (
            _HESTON_CONTRACT_SPOT_SKEW_LIMIT_SECONDS
        ),
        "contract_quality_rejections": contract_quality_rejections,
        "calibration_data_status": calibration_data_status,
        "warnings": warnings,
        "calibration_error_rmse": float(rmse) if np.isfinite(rmse) else None,
        "calibration_error_rmse_unit": "absolute_implied_volatility",
        "calibration_status": "accepted" if usable_for_pricing else "rejected",
        "usable_for_pricing": usable_for_pricing,
        "calibration_quality_failures": quality_failures,
        "pricing_usability_failures": pricing_usability_failures,
        "feller_satisfied": feller_satisfied,
        "feller_left": float(feller_left),
        "feller_right": float(feller_right),
        "rho_at_bound": rho_at_bound,
        "params": params,
        "pricing_assumptions": _heston_pricing_assumptions(
            calendar=calendar_name,
            days_to_expiry_basis=maturity_basis_norm,
        ),
        "risk_free_rate": float(risk_free_rate),
        "dividend_yield": float(dividend_yield),
        "sample_contracts": rows[:10],
    }
