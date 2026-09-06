"""Target series construction and transformation logic."""
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.indicators import _apply_ta_indicators
from ..utils.indicators import _parse_ti_specs as _parse_ti_specs_util
from ..utils.market_metadata import TICK_VOLUME_UNIT

logger = logging.getLogger(__name__)


def forecast_interval_recovery(target: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Recommend uncertainty recovery without changing the forecast target."""
    if isinstance(target, dict) and target.get("mode") == "custom":
        return {
            "recommended_tool": "forecast_list_methods",
            "recommended_params": {"supports_ci": True},
            "calibration_support": "unsupported_custom_target",
            "remediation": (
                "Custom-target conformal calibration is unsupported. Use "
                "forecast_list_methods with supports_ci=true to choose a native "
                "interval-capable method, keeping the same target_spec and history window."
            ),
        }
    return {
        "recommended_tool": "forecast_conformal_intervals",
    }


def _custom_target_contract(df: pd.DataFrame, base: str, transform: str) -> Dict[str, Any]:
    price_columns = {"open", "high", "low", "close", "typical", "tp", "hl2", "ohlc4", "ha_close", "haclose"}
    source = df.attrs.get("volume_source") if base == "volume" else base
    if base in price_columns:
        quantity, units = "price", "price"
    elif base in {"volume", "tick_volume", "real_volume"}:
        quantity = "volume"
        units = TICK_VOLUME_UNIT if source == "tick_volume" else "broker_traded_volume" if source == "real_volume" else "broker_volume_unspecified"
    else:
        quantity, units = "indicator", "indicator_units"
    base_units = units
    transform_name = transform.split("(", 1)[0]
    if transform_name in {"return", "pct_change"}:
        units = "fractional_change"
    elif transform_name == "pct":
        units = "percent_change"
    elif transform_name == "log_return":
        units = "log_ratio"
    elif transform_name == "log":
        units = "natural_log_of_" + units
    target_quantity = quantity
    if transform_name == "log":
        target_quantity = "log_" + quantity
    elif transform_name == "diff":
        target_quantity = quantity + "_change"
    elif transform_name in {"return", "pct_change", "pct", "log_return"}:
        target_quantity = "return" if quantity == "price" else quantity + "_relative_change"
    contract: Dict[str, Any] = {"quantity": target_quantity, "units": units, "base_units": base_units}
    if quantity == "volume":
        contract["volume_source"] = source or "unspecified"
        contract["volume_type"] = base_units
    return contract


def _log_return_array(prices: np.ndarray, k: int = 1) -> np.ndarray:
    """Canonical log-return computation for target series.

    A return is missing when either price is non-finite or non-positive. The
    first *k* elements are also missing because they have no lag reference.
    """
    prices = np.asarray(prices, dtype=float).reshape(-1)
    k = max(1, int(k))
    y = np.full(prices.shape, np.nan, dtype=float)
    if prices.size <= k:
        return y
    current = prices[k:]
    previous = prices[:-k]
    valid = (
        np.isfinite(current)
        & np.isfinite(previous)
        & (current > 0.0)
        & (previous > 0.0)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.log(current) - np.log(previous)
    y[k:] = np.where(valid, values, np.nan)
    return y


def _simple_return_array(values: np.ndarray, k: int = 1) -> np.ndarray:
    """Compute lagged simple returns while masking invalid denominators."""
    values = np.asarray(values, dtype=float).reshape(-1)
    k = max(1, int(k))
    y = np.full(values.shape, np.nan, dtype=float)
    if values.size <= k:
        return y
    current = values[k:]
    previous = values[:-k]
    valid = (
        np.isfinite(current)
        & np.isfinite(previous)
        & (np.abs(previous) > 1e-12)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = (current - previous) / previous
    y[k:] = np.where(valid, returns, np.nan)
    return y


def resolve_alias_base(arrs: Dict[str, np.ndarray], name: str) -> Optional[np.ndarray]:
    """Resolve alias base columns like 'typical', 'hl2', 'ohlc4'."""
    nm = name.strip().lower()
    if nm in ('typical', 'tp'):
        if all(k in arrs for k in ('high', 'low', 'close')):
            return (arrs['high'] + arrs['low'] + arrs['close']) / 3.0
        return None
    if nm == 'hl2':
        if all(k in arrs for k in ('high', 'low')):
            return (arrs['high'] + arrs['low']) / 2.0
        return None
    if nm in ('ohlc4', 'ha_close', 'haclose'):
        if all(k in arrs for k in ('open', 'high', 'low', 'close')):
            return (arrs['open'] + arrs['high'] + arrs['low'] + arrs['close']) / 4.0
        return None
    return None


def build_target_series(
    df: pd.DataFrame,
    base_col: str,
    target_spec: Optional[Dict[str, Any]] = None,
    quantity: str = 'price',
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build target series from base column with optional transformations.
    
    Returns:
        (y_array, target_info_dict)
    """
    target_info: Dict[str, Any] = {}
    
    if not target_spec or not isinstance(target_spec, dict):
        if str(quantity).strip().lower() == 'return':
            y = _log_return_array(df[base_col].astype(float).to_numpy(), k=1)
            target_info = {'mode': 'return', 'base': base_col, 'transform': 'log_return'}
        else:
            y = df[base_col].astype(float).to_numpy()
            target_info = {'mode': 'price', 'base': base_col, 'transform': 'none'}
        return y, target_info
    
    # Custom target_spec mode
    ts = dict(target_spec)
    
    # Compute indicators if requested
    ts_inds = ts.get('indicators')
    if ts_inds:
        try:
            specs = _parse_ti_specs_util(str(ts_inds)) if isinstance(ts_inds, str) else ts_inds
            try:
                _apply_ta_indicators(df, str(ts_inds) if isinstance(ts_inds, str) else ts_inds)
            except TypeError:
                _apply_ta_indicators(df, specs)
        except Exception as exc:
            logger.warning("Failed to apply target_spec indicators %r: %s", ts_inds, exc)
            raise ValueError(f"Failed to apply target_spec indicators: {exc}") from exc
    
    base_name = str(ts.get('base', ts.get('column', base_col)))
    
    # Resolve base series
    if base_name in df.columns:
        y_base = df[base_name].astype(float).to_numpy()
    else:
        # Try alias resolution
        arrs = {c: df[c].to_numpy() for c in df.columns if c in ('open', 'high', 'low', 'close')}
        y_base = resolve_alias_base(arrs, base_name)
        if y_base is None:
            raise ValueError(f"Base column '{base_name}' not found and not a recognized alias")
    
    target_info['base'] = base_name
    
    # Apply transform
    default_transform = (
        'log_return' if str(quantity).strip().lower() == 'return' else 'none'
    )
    transform = str(ts.get('transform', default_transform)).lower()
    k = int(ts.get('k', 1))
    if k < 1:
        k = 1
    
    if transform == 'none':
        y = y_base
        target_info['transform'] = 'none'
    elif transform in ('return',):
        y = _simple_return_array(y_base, k=k)
        target_info['transform'] = f'return(k={k})'
    elif transform == 'log_return':
        y = _log_return_array(y_base, k=k)
        target_info['transform'] = f'log_return(k={k})'
    elif transform == 'log':
        y = np.full(np.asarray(y_base).shape, np.nan, dtype=float)
        valid = np.isfinite(y_base) & (y_base > 0.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            y[valid] = np.log(y_base[valid])
        target_info['transform'] = 'log'
    elif transform == 'diff':
        prev = np.roll(y_base, k)
        y = y_base - prev
        y = y.astype(float, copy=False)
        y[:k] = np.nan
        target_info['transform'] = f'diff(k={k})'
    elif transform in ('pct_change', 'pct'):
        y = _simple_return_array(y_base, k=k)
        if transform == 'pct':
            y = 100.0 * y
        target_info['transform'] = f'{"pct" if transform == "pct" else "pct_change"}(k={k})'
    else:
        raise ValueError(
            f"Unsupported target transform '{transform}'. Use none, return, "
            "log_return, diff, pct_change, log, or pct."
        )
    
    target_info['mode'] = 'custom'
    target_info.update(_custom_target_contract(df, base_name, target_info['transform']))
    return y, target_info
