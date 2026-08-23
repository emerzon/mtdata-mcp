"""
Forecast optimization: genetic search for optimal settings across timeframes,
algorithms, parameters, and features.

This module provides the infrastructure for comprehensive hyperparameter optimization
via genetic algorithms, supporting multi-dimensional search over timeframes, methods,
method parameters, and optional feature indicators.
"""

import math
from typing import Any, Dict, List, Optional, Tuple


def scale_metric_to_01(value: Optional[float], vmin: float = 0.0, vmax: float = 1.0) -> float:
    """Scale a metric to [0, 1] range for fitness aggregation.

    Args:
        value: The raw metric value (may be None or NaN)
        vmin: Expected minimum (values below → 0)
        vmax: Expected maximum (values above → 1)

    Returns:
        Scaled value in [0, 1], or 0.0 if value is None/NaN/invalid
    """
    if value is None:
        return 0.0
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(fv):
        return 0.0
    if vmax <= vmin:
        return 0.5
    scaled = (fv - vmin) / (vmax - vmin)
    return max(0.0, min(1.0, scaled))


def composite_fitness_score(
    backtest_metrics: Dict[str, Any],
    *,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Compute a composite fitness score from backtest metrics.

    Default weights:
    - sharpe_ratio: 40% (risk-adjusted returns)
    - win_rate: 30% (forecast accuracy proxy)
    - inverse_max_drawdown: 20% (risk control)
    - avg_return: 10% (profitability)

    Args:
        backtest_metrics: Dict with keys like 'sharpe_ratio', 'win_rate', 'max_drawdown', etc.
        weights: Optional custom weights dict. Must sum to 1.0 (or will be auto-normalized).

    Returns:
        Composite score in [0, 1], higher is better.
    """
    if not isinstance(backtest_metrics, dict):
        return 0.0

    # Default weights
    default_weights = {
        'sharpe_ratio': 0.4,
        'win_rate': 0.3,
        'inverse_max_drawdown': 0.2,
        'avg_return': 0.1,
    }

    # Use custom weights if provided, else default
    w = dict(weights or default_weights)

    # Normalize weights to sum to 1.0
    total = sum(v for v in w.values() if v > 0)
    if total <= 0:
        return 0.0
    w = {k: v / total for k, v in w.items()}

    score = 0.0

    # Sharpe ratio: typically -5 to 5; scale to 0-1 with 0 as baseline
    if 'sharpe_ratio' in w and w['sharpe_ratio'] > 0:
        sr = backtest_metrics.get('sharpe_ratio')
        sr_scaled = scale_metric_to_01(sr, vmin=-5.0, vmax=5.0)
        score += w['sharpe_ratio'] * sr_scaled

    # Win rate: 0 to 1
    if 'win_rate' in w and w['win_rate'] > 0:
        wr = backtest_metrics.get('win_rate')
        wr_scaled = scale_metric_to_01(wr, vmin=0.3, vmax=0.7)  # 30-70% is typical
        score += w['win_rate'] * wr_scaled

    # Inverse max drawdown: penalize large drawdowns
    # Invert so that smaller drawdowns → higher score
    if 'inverse_max_drawdown' in w and w['inverse_max_drawdown'] > 0:
        md = backtest_metrics.get('max_drawdown')
        if md is not None:
            try:
                md_f = float(md)
                if math.isfinite(md_f) and md_f >= 0:
                    # Invert: 10% dd → 0.9, 50% dd → 0.5, 100% dd → 0
                    inv_dd = max(0.0, 1.0 - md_f)
                else:
                    inv_dd = 0.5
            except Exception:
                inv_dd = 0.5
        else:
            inv_dd = 0.5
        score += w['inverse_max_drawdown'] * inv_dd

    # Average return per trade: scale -10% to +10%
    if 'avg_return' in w and w['avg_return'] > 0:
        ar = backtest_metrics.get('avg_return_per_trade')
        ar_scaled = scale_metric_to_01(ar, vmin=-0.1, vmax=0.1)
        score += w['avg_return'] * ar_scaled

    return max(0.0, min(1.0, score))


def build_comprehensive_search_space(
    *,
    timeframes: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    method_search_spaces: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a comprehensive search space for genetic optimization.

    Returns a structure like:
    {
        'timeframe': {'type': 'categorical', 'choices': ['H1', 'H4', 'D1']},
        'method': {'type': 'categorical', 'choices': ['theta', 'fourier_ols', 'drift']},
        '_method_spaces': {
            'theta': {'seasonality': {'type': 'int', 'min': 8, 'max': 72}},
            'fourier_ols': {...},
        },
    }

    Args:
        timeframes: List of timeframes to search (default: ['H1', 'H4', 'D1', 'W1'])
        methods: List of methods to search (default: fast classical baselines)
        method_search_spaces: Optional dict of method-specific parameter spaces

    Returns:
        Dict defining the comprehensive search space
    """
    if not timeframes:
        timeframes = ['H1', 'H4', 'D1', 'W1']
    if not methods:
        # Keep the implicit search local and predictable. Foundation methods may
        # download models or initialize heavyweight runtimes, so callers must opt
        # into them explicitly through ``methods``.
        methods = [
            'theta',
            'fourier_ols',
            'drift',
            'naive',
            'seasonal_naive',
            'ses',
            'holt',
        ]

    # Use provided method spaces or fall back to defaults
    from .tune import default_search_space as _default_search_space
    
    if method_search_spaces is None:
        method_search_spaces = _default_search_space(methods=methods)
    else:
        # Merge provided with defaults to ensure all methods have a space
        defaults = _default_search_space(methods=methods)
        method_search_spaces = {**defaults, **method_search_spaces}

    search_space: Dict[str, Any] = {
        'timeframe': {
            'type': 'categorical',
            'choices': list(timeframes),
        },
        'method': {
            'type': 'categorical',
            'choices': list(methods),
        },
        '_method_spaces': method_search_spaces,
    }

    return search_space


def extract_method_params_from_genotype(
    genotype: Dict[str, Any],
    search_space: Dict[str, Any],
) -> Tuple[str, str, Dict[str, Any]]:
    """Extract timeframe, method, and params from a genetic genotype.

    Args:
        genotype: Individual from the population (dict with 'timeframe', 'method', param keys)
        search_space: The comprehensive search space dict

    Returns:
        Tuple of (timeframe, method, method_params_dict)
    """
    timeframe = str(genotype.get('timeframe', 'H1'))
    method = str(genotype.get('method', 'theta'))

    method_spaces = search_space.get('_method_spaces', {})
    method_space = method_spaces.get(method, {})

    params: Dict[str, Any] = {}
    for param_name, _param_spec in method_space.items():
        if param_name in genotype:
            params[param_name] = genotype[param_name]

    return timeframe, method, params


__all__ = [
    'scale_metric_to_01',
    'composite_fitness_score',
    'build_comprehensive_search_space',
    'extract_method_params_from_genotype',
]
