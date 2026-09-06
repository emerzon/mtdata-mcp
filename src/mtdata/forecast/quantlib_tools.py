from __future__ import annotations

"""QuantLib-based pricing and calibration helpers."""

import datetime as _dt
import math
import threading
from functools import wraps
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np

from ..services.options_service import get_options_chain, get_options_expirations

_DEFAULT_QUANTLIB_CALENDAR = "UnitedStates.NYSE"
_DEFAULT_MATURITY_BASIS = "calendar_days"
_HESTON_CONTRACT_SPOT_SKEW_LIMIT_SECONDS = 900.0
_QUANTLIB_CALENDAR_TIMEZONES = {
    "NullCalendar": "UTC",
    "WeekendsOnly": "UTC",
    "TARGET": "Europe/Brussels",
    "UnitedStates": "America/New_York",
    "UnitedStates.NYSE": "America/New_York",
    "UnitedStates.GovernmentBond": "America/New_York",
    "UnitedStates.Settlement": "America/New_York",
    "Japan": "Asia/Tokyo",
    "UnitedKingdom": "Europe/London",
    "UnitedKingdom.Exchange": "Europe/London",
    "UnitedKingdom.Settlement": "Europe/London",
    "Canada": "America/Toronto",
    "Canada.TSX": "America/Toronto",
    "Australia": "Australia/Sydney",
    "HongKong": "Asia/Hong_Kong",
    "Switzerland": "Europe/Zurich",
    "Germany": "Europe/Berlin",
    "Germany.FrankfurtStockExchange": "Europe/Berlin",
    "SouthKorea": "Asia/Seoul",
    "China": "Asia/Shanghai",
    "India": "Asia/Kolkata",
    "Brazil": "America/Sao_Paulo",
    "Italy": "Europe/Rome",
    "France": "Europe/Paris",
    "Singapore": "Asia/Singapore",
    "Taiwan": "Asia/Taipei",
    "Poland": "Europe/Warsaw",
    "Sweden": "Europe/Stockholm",
    "Denmark": "Europe/Copenhagen",
    "Norway": "Europe/Oslo",
    "Finland": "Europe/Helsinki",
    "NewZealand": "Pacific/Auckland",
    "SouthAfrica": "Africa/Johannesburg",
    "Mexico": "America/Mexico_City",
    "Argentina": "America/Argentina/Buenos_Aires",
    "Hungary": "Europe/Budapest",
    "CzechRepublic": "Europe/Prague",
    "Iceland": "Atlantic/Reykjavik",
    "Russia": "Europe/Moscow",
    "SaudiArabia": "Asia/Riyadh",
    "Thailand": "Asia/Bangkok",
    "Turkey": "Europe/Istanbul",
    "Israel": "Asia/Jerusalem",
}
_HESTON_SINGLE_EXPIRY_LIMITATIONS = [
    "single_expiry_fit: kappa and theta are weakly identified from one smile slice",
    "do not treat fitted parameters as a general Heston term-structure calibration",
]
_HESTON_CHAIN_PROVENANCE_KEYS = (
    "provider",
    "configured_provider",
    "provider_effective",
    "cached",
    "retrieved_at",
    "provider_attempts",
    "underlying_quote",
)
_HESTON_BARRIER_T_GRID = 100
_HESTON_BARRIER_X_GRID = 100
_HESTON_BARRIER_V_GRID = 50
_BARRIER_MODELS = ("black_scholes_merton", "heston")
_QUANTLIB_SETTINGS_LOCK = threading.RLock()


def _isolated_quantlib_settings(func: Any) -> Any:
    """Serialize QuantLib global settings and restore the caller's date."""

    @wraps(func)
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            import QuantLib as ql
        except Exception:
            return func(*args, **kwargs)
        with _QUANTLIB_SETTINGS_LOCK:
            settings = ql.Settings.instance()
            previous_evaluation_date = settings.evaluationDate
            try:
                return func(*args, **kwargs)
            finally:
                settings.evaluationDate = previous_evaluation_date

    return _wrapped


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


def _build_heston_process(
    ql: Any,
    ql_today: Any,
    spot: float,
    rf: float,
    div: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    day_count: Any,
) -> Any:
    spot_h = ql.QuoteHandle(ql.SimpleQuote(float(spot)))
    rf_ts = ql.YieldTermStructureHandle(ql.FlatForward(ql_today, float(rf), day_count))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(ql_today, float(div), day_count))
    return ql.HestonProcess(
        rf_ts,
        div_ts,
        spot_h,
        float(v0),
        float(kappa),
        float(theta),
        float(sigma),
        float(rho),
    )


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


def _valuation_timezone_for_calendar(calendar_name: Any) -> Optional[str]:
    normalized_name = _normalize_quantlib_calendar_name(calendar_name)
    return _QUANTLIB_CALENDAR_TIMEZONES.get(normalized_name)


def _default_valuation_date(
    calendar_name: Any,
    *,
    now_utc: Optional[_dt.datetime] = None,
) -> tuple[_dt.date, str, str, Optional[str]]:
    mapped_timezone = _valuation_timezone_for_calendar(calendar_name)
    if mapped_timezone is None:
        timezone_name = "UTC"
        source = "utc_fallback"
        warning = (
            f"No IANA timezone is mapped for calendar {calendar_name!r}; "
            "valuation_date used UTC. Pass valuation_date explicitly."
        )
    else:
        timezone_name = mapped_timezone
        source = "default_calendar_local_date"
        warning = None
    current_utc = now_utc or _dt.datetime.now(_dt.timezone.utc)
    return (
        current_utc.astimezone(ZoneInfo(timezone_name)).date(),
        timezone_name,
        source,
        warning,
    )


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


@_isolated_quantlib_settings
def price_barrier_option_quantlib(  # noqa: C901
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
    model: str = "black_scholes_merton",
    heston_v0: Optional[float] = None,
    heston_kappa: Optional[float] = None,
    heston_theta: Optional[float] = None,
    heston_sigma: Optional[float] = None,
    heston_rho: Optional[float] = None,
    barrier_already_hit: bool = False,
) -> Dict[str, Any]:
    """Price a European barrier option with QuantLib.

    An explicit prior knock-out assumes its on-hit rebate was settled before
    valuation. Such historical cashflows are excluded from remaining premium.
    """
    try:
        model_norm = _normalize_barrier_model(model)
    except ValueError as ex:
        return {
            "error": str(ex),
            "error_code": "invalid_parameter",
            "parameter": "model",
            "value": model,
            "valid_values": list(_BARRIER_MODELS),
        }
    heston_values = (heston_v0, heston_kappa, heston_theta, heston_sigma, heston_rho)
    heston_provided = any(value is not None for value in heston_values)
    if model_norm == "black_scholes_merton" and heston_provided:
        return {
            "error": (
                "Heston parameters require --model heston. "
                "Omit heston_* or switch the pricing model."
            ),
            "error_code": "invalid_parameter",
            "parameter": "model",
            "remediation": (
                "Pass --model heston with heston_v0/kappa/theta/sigma/rho, or "
                "omit Heston parameters to keep Black-Scholes-Merton flat vol."
            ),
        }
    heston_params: Optional[Dict[str, float]] = None
    if model_norm == "heston":
        validated = _validate_heston_barrier_params(
            heston_v0=heston_v0,
            heston_kappa=heston_kappa,
            heston_theta=heston_theta,
            heston_sigma=heston_sigma,
            heston_rho=heston_rho,
        )
        if validated.get("error"):
            return validated
        heston_params = {
            "v0": float(validated["v0"]),
            "kappa": float(validated["kappa"]),
            "theta": float(validated["theta"]),
            "sigma": float(validated["sigma"]),
            "rho": float(validated["rho"]),
        }
    try:
        spot_val = float(spot)
        strike_val = float(strike)
        barrier_val = float(barrier)
        maturity_val = int(maturity_days)
        rf = float(risk_free_rate)
        div = float(dividend_yield)
        vol = float(volatility) if heston_params is None else 0.0
        rebate_val = float(rebate)
    except Exception:
        return {
            "error": "Invalid numeric input for barrier pricing",
            "error_code": "invalid_parameter",
        }

    for parameter, value in (
        ("spot", spot_val),
        ("strike", strike_val),
        ("barrier", barrier_val),
    ):
        if not math.isfinite(value) or value <= 0:
            return {
                "error": f"{parameter} must be a positive number.",
                "error_code": "invalid_parameter",
                "details": {
                    "parameter": parameter,
                    "received": value,
                    "required_minimum": 0,
                },
            }
    if maturity_val <= 0:
        return {
            "error": "maturity_days must be a positive integer.",
            "error_code": "invalid_parameter",
            "details": {
                "parameter": "maturity_days",
                "received": maturity_val,
                "required_minimum": 1,
            },
        }
    if heston_params is None and (not math.isfinite(vol) or vol <= 0):
        return {
            "error": "volatility must be a positive number.",
            "error_code": "invalid_parameter",
            "details": {
                "parameter": "volatility",
                "received": vol,
                "required_minimum": 0,
            },
        }
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
    try:
        import QuantLib as ql
    except Exception as ex:
        return {"error": f"QuantLib is required: {ex}"}
    try:
        calendar_obj, calendar_name = _resolve_quantlib_calendar(ql, calendar_name)
    except ValueError as ex:
        return {
            "error": str(ex),
            "error_code": "invalid_calendar",
            "parameter": "calendar",
            "value": calendar,
        }
    valuation_warning: Optional[str] = None
    if valuation_date is None:
        (
            valuation_day,
            valuation_timezone,
            valuation_date_source,
            valuation_warning,
        ) = _default_valuation_date(calendar_name)
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
        valuation_timezone = _valuation_timezone_for_calendar(calendar_name) or "UTC"
        valuation_date_source = "explicit"

    barrier_status = _barrier_option_status(
        barrier_type=barrier_type_norm,
        spot=spot_val,
        barrier=barrier_val,
        barrier_already_hit=barrier_already_hit,
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
            model=model_norm,
            heston_params=heston_params,
            barrier_already_hit=barrier_already_hit,
        )
        knocked_out: Dict[str, Any] = {
            "success": True,
            "price": 0.0 if barrier_already_hit else float(rebate_val),
            "rebate_cashflow": {
                "amount": float(rebate_val),
                "units": "per_underlying_unit",
                "settlement": (
                    "assumed_paid_before_valuation"
                    if barrier_already_hit
                    else "due_at_valuation"
                ),
                "included_in_price": not bool(barrier_already_hit),
            },
            "status": "knocked_out",
            "option_status": "knocked_out",
            "barrier_state_source": (
                "explicit_prior_hit"
                if barrier_already_hit
                else "spot_at_or_beyond_barrier"
            ),
            "valuation_date": valuation_day.isoformat(),
            "valuation_timezone": valuation_timezone,
            "valuation_date_source": valuation_date_source,
            "delta": 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "greeks_status": "complete",
            "greeks_method": (
                "knocked_out_settled" if barrier_already_hit else "knocked_out_boundary"
            ),
            "pricing_assumptions": _quantlib_pricing_assumptions(
                "previously knocked-out option (rebate assumed settled)"
                if barrier_already_hit
                else _barrier_model_label(
                    model=model_norm,
                    heston_params=heston_params,
                    knocked_status="knocked_out",
                ),
                calendar=calendar_name,
                maturity_basis=maturity_basis_norm,
            ),
            "params_used": params_used,
        }
        if valuation_warning:
            knocked_out["warnings"] = [valuation_warning]
        return knocked_out

    if barrier_status == "knocked_in":
        knocked_in = _price_knocked_in_as_vanilla(
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
            model_norm=model_norm,
            heston_params=heston_params,
            barrier_already_hit=barrier_already_hit,
        )
        if (
            valuation_warning
            and isinstance(knocked_in, dict)
            and knocked_in.get("success")
        ):
            knocked_in["warnings"] = [valuation_warning]
        return knocked_in

    if heston_params is not None and not hasattr(ql, "FdHestonBarrierEngine"):
        return {
            "error": (
                "QuantLib FdHestonBarrierEngine is required for --model heston "
                "barrier pricing."
            ),
            "error_code": "capability_unavailable",
            "capability": "FdHestonBarrierEngine",
            "remediation": (
                "Upgrade QuantLib, or price with --model black_scholes_merton "
                "and a strike-specific --volatility."
            ),
        }

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
        if heston_params is not None:
            process = _build_heston_process(
                ql,
                ql_today,
                spot_local,
                rf,
                div,
                heston_params["v0"],
                heston_params["kappa"],
                heston_params["theta"],
                heston_params["sigma"],
                heston_params["rho"],
                day_count,
            )
            heston_model = ql.HestonModel(process)
            barrier_opt.setPricingEngine(
                ql.FdHestonBarrierEngine(
                    heston_model,
                    _HESTON_BARRIER_T_GRID,
                    _HESTON_BARRIER_X_GRID,
                    _HESTON_BARRIER_V_GRID,
                )
            )
            return float(barrier_opt.NPV())
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

    if heston_params is None:
        try:
            eps_v = max(1e-4, abs(vol) * 5e-2)
            pv_up = _price_with(spot_val, vol + eps_v)
            pv_dn = _price_with(spot_val, max(1e-6, vol - eps_v))
            vega = (pv_up - pv_dn) / (2.0 * eps_v)
        except Exception as ex:
            greeks_warnings.append(
                f"Vega unavailable: volatility differences failed ({ex})."
            )
    else:
        greeks_warnings.append(
            "Vega is omitted for Heston barrier pricing; it is not a Black "
            "implied-volatility derivative. Spot delta/gamma use finite differences."
        )

    finite_greeks = sum(
        value is not None and math.isfinite(value)
        for value in (delta, gamma, vega)
    )
    expected_greeks = 2 if heston_params is not None else 3
    greeks_status = (
        "complete"
        if finite_greeks == expected_greeks
        else "partial"
        if finite_greeks
        else "unavailable"
    )

    return {
        "success": True,
        "price": float(npv),
        "barrier_state_source": "assumed_unhit_at_valuation",
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
        **({"warnings": [valuation_warning]} if valuation_warning else {}),
        "pricing_assumptions": {
            **_quantlib_pricing_assumptions(
                _barrier_model_label(
                    model=model_norm,
                    heston_params=heston_params,
                ),
                calendar=calendar_name,
                maturity_basis=maturity_basis_norm,
            ),
            **(
                {
                    "heston_fd_grid": (
                        f"t={_HESTON_BARRIER_T_GRID},"
                        f"x={_HESTON_BARRIER_X_GRID},"
                        f"v={_HESTON_BARRIER_V_GRID}"
                    )
                }
                if heston_params is not None
                else {}
            ),
        },
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
            model=model_norm,
            heston_params=heston_params,
            barrier_already_hit=barrier_already_hit,
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
    model_norm: str = "black_scholes_merton",
    heston_params: Optional[Dict[str, float]] = None,
    barrier_already_hit: bool = False,
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

    def _set_engine(spot_local: float) -> None:
        if heston_params is not None:
            process = _build_heston_process(
                ql,
                ql_today,
                spot_local,
                rf,
                div,
                heston_params["v0"],
                heston_params["kappa"],
                heston_params["theta"],
                heston_params["sigma"],
                heston_params["rho"],
                day_count,
            )
            option.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(process)))
            return
        process = _build_bs_merton_process(
            ql,
            ql_today,
            calendar_obj,
            spot_local,
            rf,
            div,
            vol,
            day_count,
        )
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

    def _price_with(spot_local: float) -> float:
        _set_engine(spot_local)
        return float(option.NPV())

    try:
        npv = _price_with(spot_val)
    except Exception as ex:
        return {"error": f"QuantLib pricing failed: {ex}"}

    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    greeks_warnings: List[str] = []
    if heston_params is not None:
        eps_s = max(1e-6, abs(float(spot_val)) * 1e-4)
        try:
            p_up = _price_with(float(spot_val) + eps_s)
            p_dn = _price_with(float(spot_val) - eps_s)
            delta = float((p_up - p_dn) / (2.0 * eps_s))
            gamma = float((p_up - 2.0 * npv + p_dn) / (eps_s * eps_s))
        except Exception as ex:
            greeks_warnings.append(
                f"Heston vanilla spot Greeks unavailable: finite differences failed ({ex})."
            )
        greeks_method = "knocked_in_heston_vanilla_central_difference"
    else:
        _set_engine(spot_val)
        for greek_name in ("delta", "gamma", "vega"):
            try:
                value = float(getattr(option, greek_name)())
            except Exception as ex:
                greeks_warnings.append(
                    f"{greek_name} unavailable for knocked-in vanilla pricing ({ex})."
                )
                continue
            if greek_name == "delta":
                delta = value
            elif greek_name == "gamma":
                gamma = value
            else:
                vega = value
        greeks_method = "knocked_in_vanilla_analytic"
    finite_greeks = sum(
        value is not None and math.isfinite(value)
        for value in (delta, gamma, vega)
    )
    expected_greeks = 2 if heston_params is not None else 3
    greeks_status = (
        "complete"
        if finite_greeks == expected_greeks
        else "partial"
        if finite_greeks
        else "unavailable"
    )
    return {
        "success": True,
        "price": npv,
        "status": "knocked_in",
        "option_status": "knocked_in",
        "barrier_state_source": (
            "explicit_prior_hit"
            if barrier_already_hit
            else "spot_at_or_beyond_barrier"
        ),
        "valuation_date": valuation_day.isoformat(),
        "valuation_timezone": valuation_timezone,
        "valuation_date_source": valuation_date_source,
        "maturity_date": maturity_day.isoformat(),
        "time_to_maturity_years": time_to_maturity_years,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "greeks_status": greeks_status,
        "greeks_method": greeks_method,
        **({"greeks_warnings": greeks_warnings} if greeks_warnings else {}),
        "pricing_assumptions": _quantlib_pricing_assumptions(
            _barrier_model_label(
                model=model_norm,
                heston_params=heston_params,
                knocked_status="knocked_in",
            ),
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
            model=model_norm,
            heston_params=heston_params,
            barrier_already_hit=barrier_already_hit,
        ),
    }


def _barrier_option_status(
    *,
    barrier_type: str,
    spot: float,
    barrier: float,
    barrier_already_hit: bool = False,
) -> Optional[str]:
    kind = str(barrier_type)
    if bool(barrier_already_hit):
        return "knocked_out" if kind.endswith("_out") else "knocked_in"
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
    model: str = "black_scholes_merton",
    heston_params: Optional[Dict[str, float]] = None,
    barrier_already_hit: bool = False,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "spot": float(spot),
        "strike": float(strike),
        "barrier": float(barrier),
        "maturity_days": int(maturity_days),
        "option_type": str(option_type),
        "barrier_type": str(barrier_type),
        "risk_free_rate": float(risk_free_rate),
        "dividend_yield": float(dividend_yield),
        "rebate": float(rebate),
        "calendar": str(calendar),
        "maturity_basis": str(maturity_basis),
        "model": str(model),
        "barrier_already_hit": bool(barrier_already_hit),
        **({"valuation_date": valuation_date} if valuation_date else {}),
    }
    if heston_params:
        out["heston_v0"] = float(heston_params["v0"])
        out["heston_kappa"] = float(heston_params["kappa"])
        out["heston_theta"] = float(heston_params["theta"])
        out["heston_sigma"] = float(heston_params["sigma"])
        out["heston_rho"] = float(heston_params["rho"])
    else:
        out["volatility"] = float(volatility)
    return out


def _barrier_model_label(
    *,
    model: str,
    heston_params: Optional[Dict[str, float]],
    knocked_status: Optional[str] = None,
) -> str:
    if knocked_status == "knocked_out":
        return "already-breached knock-out (rebate)"
    if knocked_status == "knocked_in":
        if heston_params:
            return "already-breached knock-in priced as Heston European vanilla"
        return "already-breached knock-in priced as European vanilla"
    if heston_params or model == "heston":
        return "Heston finite-difference barrier (FdHestonBarrierEngine)"
    return "BlackScholesMerton analytic barrier"


def _heston_pricing_assumptions(
    *,
    calendar: str,
    days_to_expiry_basis: str,
    american_surface_approximated_as_european: bool = False,
) -> Dict[str, str]:
    assumptions = _quantlib_pricing_assumptions(
        "Heston analytic calibration",
        calendar=calendar,
        maturity_basis=days_to_expiry_basis,
    )
    assumptions.pop("maturity_basis", None)
    assumptions["maturity_convention"] = "calendar_days_to_contract_expiry"
    assumptions["days_to_expiry_basis"] = days_to_expiry_basis
    if american_surface_approximated_as_european:
        assumptions["exercise_style"] = "american_approximated_as_european"
        assumptions["american_surface_approximated_as_european"] = "true"
        assumptions["pricing_engine"] = "AnalyticHestonEngine/HestonModelHelper (European)"
    else:
        assumptions["exercise_style"] = "european"
    return assumptions


def _heston_chain_provenance(source: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Copy provider/cache/retrieval envelope from a chain or catalog payload."""
    if not isinstance(source, dict):
        return {}
    out: Dict[str, Any] = {}
    for key in _HESTON_CHAIN_PROVENANCE_KEYS:
        if key not in source:
            continue
        value = source.get(key)
        if value in ("", [], {}):
            continue
        out[key] = value
    providers_used = source.get("providers_used")
    derived: List[str] = []
    if isinstance(providers_used, list):
        for item in providers_used:
            name = str(item or "").strip()
            if name and name not in derived:
                derived.append(name)
    if not derived:
        attempts = source.get("provider_attempts")
        if isinstance(attempts, list):
            for attempt in attempts:
                if not isinstance(attempt, dict):
                    continue
                name = str(attempt.get("provider") or "").strip()
                if name and name not in derived:
                    derived.append(name)
        provider = str(source.get("provider") or "").strip()
        if provider and provider not in derived:
            derived.append(provider)
    if derived:
        out["providers_used"] = derived
    envelope = out.get("underlying_quote")
    if not isinstance(envelope, dict):
        envelope = source.get("underlying_quote")
    if isinstance(envelope, dict):
        market_state = envelope.get("market_state")
        if market_state not in (None, ""):
            out["market_state"] = market_state
    return out


def _normalize_barrier_model(model: Any) -> str:
    value = str(model or "black_scholes_merton").strip().lower()
    aliases = {
        "bsm": "black_scholes_merton",
        "black_scholes": "black_scholes_merton",
        "black-scholes-merton": "black_scholes_merton",
        "black_scholes_merton": "black_scholes_merton",
        "heston": "heston",
    }
    normalized = aliases.get(value)
    if normalized is None:
        raise ValueError(
            f"Invalid model: {model}. Use black_scholes_merton|heston."
        )
    return normalized


def _validate_heston_barrier_params(
    *,
    heston_v0: Any,
    heston_kappa: Any,
    heston_theta: Any,
    heston_sigma: Any,
    heston_rho: Any,
) -> Dict[str, Any]:
    missing = [
        name
        for name, value in (
            ("heston_v0", heston_v0),
            ("heston_kappa", heston_kappa),
            ("heston_theta", heston_theta),
            ("heston_sigma", heston_sigma),
            ("heston_rho", heston_rho),
        )
        if value is None
    ]
    if missing:
        return {
            "error": (
                "Heston barrier pricing requires heston_v0, heston_kappa, "
                "heston_theta, heston_sigma, and heston_rho."
            ),
            "error_code": "invalid_parameter",
            "parameter": ",".join(missing),
            "missing_parameters": missing,
            "remediation": (
                "Pass the five calibrated Heston parameters from "
                "options_heston_calibrate, or use --model black_scholes_merton "
                "with --volatility."
            ),
        }
    try:
        v0 = float(heston_v0)
        kappa = float(heston_kappa)
        theta = float(heston_theta)
        sigma = float(heston_sigma)
        rho = float(heston_rho)
    except (TypeError, ValueError):
        return {
            "error": "Heston parameters must be numeric.",
            "error_code": "invalid_parameter",
            "parameter": "heston_v0,heston_kappa,heston_theta,heston_sigma,heston_rho",
        }
    if not all(math.isfinite(value) for value in (v0, kappa, theta, sigma, rho)):
        return {
            "error": "Heston parameters must be finite.",
            "error_code": "invalid_parameter",
        }
    if not (v0 > 0 and kappa > 0 and theta > 0 and sigma > 0):
        return {
            "error": "heston_v0, heston_kappa, heston_theta, and heston_sigma must be positive.",
            "error_code": "invalid_parameter",
        }
    if not -1.0 <= rho <= 1.0:
        return {
            "error": "heston_rho must be between -1 and 1.",
            "error_code": "invalid_parameter",
            "parameter": "heston_rho",
        }
    return {
        "v0": v0,
        "kappa": kappa,
        "theta": theta,
        "sigma": sigma,
        "rho": rho,
    }


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
    if row.get("quote_quality") not in (None, "two_sided"):
        failures.append("contract_quote_not_two_sided")
    last_trade_proxy = row.get("last_trade_recent_and_market_two_sided")
    if last_trade_proxy is False:
        failures.append("last_trade_not_recent_or_market_not_two_sided")
    elif (
        last_trade_proxy is not True
        and row.get("quote_usable_for_live_analysis") is not True
    ):
        failures.append("contract_quote_not_live_usable")
    return failures, skew_seconds


@_isolated_quantlib_settings
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
        valuation_timezone = (
            _valuation_timezone_for_calendar(calendar_name) or "UTC"
        )
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
                **_heston_chain_provenance(
                    expirations_result if isinstance(expirations_result, dict) else None
                ),
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
                **_heston_chain_provenance(
                    expirations_result if isinstance(expirations_result, dict) else None
                ),
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
    chain_provenance = _heston_chain_provenance(chain if isinstance(chain, dict) else None)

    spot_val = float(chain.get("underlying_price", float("nan")))
    if not (spot_val == spot_val and spot_val > 0):
        return {
            "error": "Underlying spot price unavailable from options provider.",
            **chain_provenance,
        }

    spot_as_of = chain.get("underlying_as_of")
    spot_epoch = _timezone_qualified_epoch(spot_as_of)
    if spot_epoch is None:
        return {
            "error": (
                "Options provider timestamp is required to anchor Heston "
                "calibration to a single market snapshot."
            ),
            "error_code": "chain_observation_time_unavailable",
            **chain_provenance,
            "spot_as_of": spot_as_of,
            "remediation": (
                "Retry with a provider response that includes a timezone-qualified "
                "underlying_as_of timestamp."
            ),
        }
    valuation_timezone = (
        _valuation_timezone_for_calendar(calendar_name) or "UTC"
    )
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
            **chain_provenance,
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
                **chain_provenance,
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
                **chain_provenance,
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
        selected_row = {
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
            "last_trade_recent_and_market_two_sided": row.get(
                "last_trade_recent_and_market_two_sided"
            ),
            "spot_contract_skew_seconds": (
                round(float(skew_seconds), 3)
                if skew_seconds is not None
                else None
            ),
        }
        if row.get("exercise_style") not in (None, ""):
            selected_row["exercise_style"] = row.get("exercise_style")
        rows.append(selected_row)

    if len(rows) < 5:
        return {
            "success": False,
            "error": (
                "Heston calibration requires at least 5 current, timestamped, "
                "two-sided option contracts from the same market snapshot."
            ),
            "error_code": "heston_contract_inputs_rejected",
            **chain_provenance,
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
            "next_tool": "options_provider_status",
            "remediation": (
                "Anonymous Yahoo last-trade data is delayed about 15 minutes, "
                "which exceeds the 900s last-trade-vs-spot gate used for "
                "Heston calibration. Set MTDATA_OPTIONS_PROVIDER=tradier and "
                "MTDATA_OPTIONS_API_KEY, then run options_provider_status to "
                "verify a real-time provider. Delayed last-trade timestamps are "
                "intentionally rejected because calibrated parameters feed "
                "pricing."
            ),
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
                **chain_provenance,
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
        return {
            "error": "Options expiration date missing from chain output.",
            **chain_provenance,
        }
    try:
        expiry_date = _dt.datetime.strptime(expiry_text, "%Y-%m-%d").date()
    except Exception:
        return {
            "error": f"Invalid expiration format: {expiry_text}",
            **chain_provenance,
        }
    if valuation_day >= expiry_date:
        return {
            "error": (
                "valuation_date must be before the option expiration date. "
                f"Received valuation_date={valuation_day.isoformat()} and "
                f"expiration={expiry_date.isoformat()}."
            ),
            "error_code": "invalid_expiration_date_range",
            **chain_provenance,
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
            **chain_provenance,
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
            **chain_provenance,
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

    selected_exercise_styles = sorted(
        {
            str(row.get("exercise_style")).strip().lower()
            for row in rows
            if row.get("exercise_style") not in (None, "")
        }
    )
    american_surface_approximated_as_european = (
        "american" in selected_exercise_styles
    )

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
        return {
            "error": f"QuantLib Heston calibration failed: {ex}",
            **chain_provenance,
        }

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
        if row.get("last_trade_recent_and_market_two_sided") is True
        or row.get("quote_usable_for_live_analysis") is True
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
    identification_limitations = list(_HESTON_SINGLE_EXPIRY_LIMITATIONS)
    warnings = []
    if american_surface_approximated_as_european:
        identification_limitations.append(
            "american_surface_approximated_as_european: AnalyticHestonEngine "
            "and HestonModelHelper ignore early exercise"
        )
        warnings.append(
            "Selected contracts include American-exercise options. Calibration "
            "uses QuantLib AnalyticHestonEngine / HestonModelHelper, which are "
            "European. american_surface_approximated_as_european=true; "
            "early-exercise premium is not modeled."
        )
    warnings.extend(
        [
            "Calibration mode is single_expiry_fit; kappa and theta are weakly "
            "identified from one expiration and are not a term-structure surface.",
            "Contract selection uses last-trade recency as a quote-freshness "
            "proxy because providers do not supply option quote timestamps.",
        ]
    )
    if calibration_data_stale:
        warnings.append(
            "Heston calibration used a stale underlying quote; the "
            "fitted parameters are not usable for pricing until a current "
            "snapshot is calibrated."
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
        **chain_provenance,
        "expiration": expiry_text,
        "calibration_mode": "single_expiry_fit",
        "identification_limitations": identification_limitations,
        "american_surface_approximated_as_european": (
            american_surface_approximated_as_european
        ),
        "selected_exercise_styles": selected_exercise_styles,
        "quote_freshness_policy": "last_trade_proxy",
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
            american_surface_approximated_as_european=(
                american_surface_approximated_as_european
            ),
        ),
        "risk_free_rate": float(risk_free_rate),
        "dividend_yield": float(dividend_yield),
        "sample_contracts": rows[:10],
    }
