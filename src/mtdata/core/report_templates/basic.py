from math import isfinite
from typing import Any, Dict, List, Optional

from ...shared.schema import DenoiseSpec
from ..report.extras import (
    attach_optional_report_sections,
    extract_report_pattern_rows,
)
from ..report.trend import (
    _TREND_COMPACT_LEGEND,
    _compute_compact_trend,
)
from ..report.utils import (
    _normalize_source_bar_time,
    adapt_forecast_payload_for_report,
    attach_candle_freshness_diagnostics,
    attach_multi_timeframes,
    emit_report_progress,
    extract_report_forecast_values,
    has_complete_forecast_backtest_coverage,
    normalize_report_methods,
    now_utc_iso,
    parse_table_tail,
    pick_best_forecast_method,
    report_runtime_error,
    report_runtime_expired,
    report_section_enabled,
    resolve_report_context_end,
    resolve_report_context_indicators,
    summarize_barrier_grid,
)
from ..tool_calling import call_tool_sync_structured


def _get_raw_result(
    func: Any,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Call a tool and require a structured payload."""
    operation = getattr(func, "__name__", type(func).__name__)
    if report_runtime_expired():
        emit_report_progress(operation, "skipped_runtime_budget")
        return report_runtime_error(operation)
    emit_report_progress(operation, "started")
    try:
        result = call_tool_sync_structured(func, *args, **kwargs)
        
        # If it returns a dict, use it directly
        if isinstance(result, dict):
            return result
            
        if isinstance(result, str):
            return {
                'error': 'Expected structured tool result but received formatted text.',
                'raw_output': result[:200],
            }
        
        return {'error': f'Unexpected result type: {type(result)}'}
        
    except Exception as e:
        return {'error': f'Function call failed: {str(e)}'}
    finally:
        emit_report_progress(operation, "finished")


def _first_volatility_value(payload: Dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def _extract_forecast_values(payload: Dict[str, Any]) -> Optional[List[float]]:
    values = extract_report_forecast_values(payload)
    return values or None


def _complete_backtest_ranking(
    results: Any,
) -> tuple[List[Dict[str, Any]], List[str]]:
    """Build report rankings from methods with complete anchor coverage only."""
    ranking: List[Dict[str, Any]] = []
    excluded_incomplete: List[str] = []
    if not isinstance(results, dict):
        return ranking, excluded_incomplete
    for method, method_result in results.items():
        if not isinstance(method_result, dict):
            continue
        if not has_complete_forecast_backtest_coverage(method_result):
            excluded_incomplete.append(str(method))
            continue
        ranking.append({
            'method': method,
            'avg_rmse': method_result.get('avg_rmse'),
            'avg_mae': method_result.get('avg_mae'),
            'avg_directional_accuracy': method_result.get(
                'avg_directional_accuracy'
            ),
            'successful_tests': method_result.get('successful_tests'),
        })

    def _finite_sort_value(value: Any, fallback: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return fallback
        return numeric if isfinite(numeric) else fallback

    ranking.sort(
        key=lambda row: (
            _finite_sort_value(row.get('avg_rmse'), 1e9),
            -_finite_sort_value(row.get('avg_directional_accuracy'), 0.0),
        )
    )
    return ranking, excluded_incomplete


def _is_degenerate_forecast_payload(payload: Dict[str, Any]) -> bool:
    vals = _extract_forecast_values(payload)
    if not vals:
        return True
    if len(vals) < 3:
        return False
    first = vals[0]
    span = max(vals) - min(vals)
    tol = max(1e-9, abs(first) * 1e-6)
    return span <= tol


def template_basic(  # noqa: C901
    symbol: str,
    horizon: int,
    denoise: Optional[DenoiseSpec],
    params: Optional[Dict[str, Any]],
    *,
    include_default_timeframes: bool = True,
) -> Dict[str, Any]:
    p = dict(params or {})
    tf = str(p.get('timeframe', 'H1'))
    start = p.get('start')
    end = p.get('end')
    context_end = resolve_report_context_end(end, tf)
    
    report: Dict[str, Any] = {
        'meta': {
            'symbol': symbol,
            'timeframe': tf,
            'horizon': int(horizon),
            'template': 'basic',
            'generated_at': now_utc_iso(),
        },
        'sections': {},
    }

    # Request-scoped cache avoids re-fetching the same symbol/timeframe.
    _fetch_cache: Dict = {}

    # Context
    indicators = resolve_report_context_indicators(p)
    from ..data import data_fetch_candles
    
    ctx = (
        _get_raw_result(data_fetch_candles,
            symbol=symbol,
            timeframe=tf,
            limit=int(p.get('context_limit', 300)),
            # Request validation requires an end whenever start is supplied.
            # The context snapshot anchors at that shared report cutoff.
            start=None,
            end=context_end,
            indicators=indicators,  # type: ignore[arg-type]
            denoise=denoise,
            allow_stale=bool(p.get('allow_stale', False)),
        )
        if report_section_enabled(p, 'context')
        else {'error': 'context section not requested'}
    )
    
    if 'error' in ctx:
        report['sections']['context'] = attach_candle_freshness_diagnostics({'error': ctx['error']}, ctx)
    else:
        # Metrics require consecutive source bars. Only the snapshot is
        # projected to the requested display tail.
        context_limit = int(p.get('context_limit', 300))
        context_rows = parse_table_tail(ctx, tail=context_limit)
        tail_n = int(p.get('context_tail', 40))
        tail_rows = context_rows[-tail_n:]
        if not tail_rows:
            # Fallbacks when calling through minimal formatter
            if isinstance(ctx, dict) and isinstance(ctx.get('data'), list):     
                context_rows = list(ctx.get('data'))  # type: ignore[arg-type]
                tail_rows = context_rows[-tail_n:]
            elif isinstance(ctx, list):
                context_rows = ctx
                tail_rows = context_rows[-tail_n:]
            else:
                tail_rows = []

        if not tail_rows:
            report['sections']['context'] = attach_candle_freshness_diagnostics(
                {'error': 'No candle data available for context section.'},
                ctx,
            )
        else:
            last = tail_rows[-1] if tail_rows else {}
            compact = _compute_compact_trend(context_rows)
            ctx_obj: Dict[str, Any] = {
                'symbol': symbol,
                'timeframe': tf,
                'last_snapshot': last,
                'notes': f'Indicators included: {indicators}.',
            }
            timezone_label = ctx.get('timezone') if isinstance(ctx, dict) else None
            if timezone_label not in (None, '', [], {}):
                ctx_obj['timezone'] = timezone_label
            for key in ('price_precision', 'price_point'):
                value = ctx.get(key) if isinstance(ctx, dict) else None
                if value not in (None, '', [], {}):
                    ctx_obj[key] = value
            if compact:
                ctx_obj['trend_compact'] = compact
                ctx_obj['trend_compact_legend'] = dict(_TREND_COMPACT_LEGEND)
            report['sections']['context'] = attach_candle_freshness_diagnostics(ctx_obj, ctx)

    pivot_enabled = report_section_enabled(p, 'pivot')
    contexts_multi_enabled = include_default_timeframes and report_section_enabled(
        p, 'contexts_multi'
    )
    pivot_multi_enabled = include_default_timeframes and report_section_enabled(
        p, 'pivot_multi'
    )

    # Pivots use the latest source bar completed at the shared report cutoff.
    if pivot_enabled:
        from ..pivot import pivot_compute_points

        piv = _get_raw_result(
            pivot_compute_points, symbol=symbol, timeframe='D1', end=context_end
        )

        if 'error' in piv:
            report['sections']['pivot'] = {'error': piv['error']}
        else:
            pivot_period = piv.get('period')
            source_bar_time = _normalize_source_bar_time(
                pivot_period.get('end')
                if isinstance(pivot_period, dict)
                else pivot_period
            )
            pivot_section = {
                'levels': piv.get('levels'),
                'methods': piv.get('methods'),
                'source': piv.get('source'),
                'period': pivot_period,
                'timeframe': 'D1',
                'calculation_basis': piv.get('calculation_basis'),
                'timezone': piv.get('timezone'),
            }
            for key in ('historical_cutoff', 'analysis_as_of'):
                if piv.get(key) is not None:
                    pivot_section[key] = piv[key]
            if source_bar_time is not None:
                pivot_section['source_bar_time'] = source_bar_time
                pivot_section['source_bar_timezone'] = 'UTC'
                pivot_section['source_bar_state'] = 'completed'
            report['sections']['pivot'] = pivot_section

    if contexts_multi_enabled or pivot_multi_enabled:
        attach_multi_timeframes(
            report,
            symbol,
            denoise,
            extra_timeframes=(
                ['M15','H1','H4','D1'] if contexts_multi_enabled else []
            ),
            pivot_timeframes=(
                ['H4','D1']
                if pivot_multi_enabled
                else []
            ),
            context_indicators=indicators,
            start=None,
            end=context_end,
            allow_stale=bool(p.get('allow_stale', False)),
            _fetch_cache=_fetch_cache,
        )

    # Volatility (EWMA)
    from ..forecast import forecast_volatility_estimate
    try:
        base_h = int(horizon)
    except Exception:
        base_h = 12
    vol_horizons = []
    if report_section_enabled(p, 'volatility'):
        requested_horizons = p.get('vol_horizons')
        if isinstance(requested_horizons, (list, tuple)) and requested_horizons:
            for hh in requested_horizons:
                try:
                    horizon_value = int(hh)
                except Exception:
                    continue
                if horizon_value > 0 and horizon_value not in vol_horizons:
                    vol_horizons.append(horizon_value)
        else:
            vol_horizons.append(max(1, base_h))

    # Build method x horizon matrix (Horizon σ); keep per-bar for potential future use
    vol_method = str(p.get('vol_method') or 'yang_zhang').strip() or 'yang_zhang'
    methods = [vol_method]
    matrix_rows: List[Dict[str, Any]] = []
    vol_errors: List[Dict[str, Any]] = []
    for hh in vol_horizons:
        row: Dict[str, Any] = {'horizon': int(hh)}
        contributors: List[Dict[str, Any]] = []
        for m in methods:
            vres = _get_raw_result(
                forecast_volatility_estimate,
                symbol=symbol,
                timeframe=tf,
                horizon=int(hh),
                method=m,
                params={'lambda_': 0.94} if m == 'ewma' else None,
                start=start,
                end=end,
                denoise=denoise,
                detail='full',
            )
            if 'error' in vres:
                error_text = str(vres.get('error') or 'volatility method failed')
                row[m + '_err'] = error_text
                vol_errors.append({
                    'horizon': int(hh),
                    'method': m,
                    'error': error_text,
                })
                continue
            sh = _first_volatility_value(
                vres,
                ('volatility_horizon', 'horizon_sigma_price'),
            )
            sb = _first_volatility_value(
                vres,
                ('volatility_per_bar', 'sigma_bar_price'),
            )
            use_val = None
            try:
                fv = float(sh) if sh is not None else None
                if fv is not None and fv == fv and fv >= 0.0:  # finite check (fv==fv filters NaN)
                    use_val = fv
            except Exception:
                use_val = None
            if use_val is None:
                error_text = 'requested estimator returned no usable horizon value'
                row[m + '_err'] = error_text
                vol_errors.append({
                    'horizon': int(hh),
                    'method': m,
                    'error': error_text,
                })
            else:
                row[m] = use_val
                contributors.append({'method': m, 'value': use_val, 'weight': 1.0})
            # store bar sigma too in case renderer wants it later
            if sb is not None:
                try:
                    fb = float(sb)
                    if fb == fb and fb >= 0.0:
                        row[m + '_bar'] = fb
                    else:
                        row[m + '_bar_err'] = 'nan bar sigma'
                except Exception:
                    row[m + '_bar_err'] = 'invalid bar sigma value'
        if not contributors:
            proxy_res = _get_raw_result(
                forecast_volatility_estimate,
                symbol=symbol,
                timeframe=tf,
                horizon=int(hh),
                method='rolling_std',
                start=start,
                end=end,
                denoise=denoise,
                detail='full',
            )
            proxy_value = _first_volatility_value(
                proxy_res,
                ('volatility_horizon', 'horizon_sigma_price'),
            )
            try:
                proxy_float = float(proxy_value) if proxy_value is not None else None
            except Exception:
                proxy_float = None
            if proxy_float is not None and isfinite(proxy_float) and proxy_float >= 0.0:
                row['rolling_std_proxy'] = proxy_float
                contributors.append({
                    'method': 'rolling_std_proxy',
                    'value': proxy_float,
                    'weight': 1.0,
                    'provenance': 'fallback_when_all_requested_estimators_unavailable',
                })
        if contributors:
            values = [float(item['value']) for item in contributors]
            row['avg'] = sum(values) / len(values)
            row['avg_method'] = (
                contributors[0]['method']
                if len(contributors) == 1
                else 'ensemble_mean'
            )
            equal_weight = 1.0 / len(contributors)
            for contributor in contributors:
                contributor['weight'] = equal_weight
            row['contributors'] = contributors
            matrix_rows.append(row)
    if matrix_rows:
        report['sections']['volatility'] = {
            'methods': methods,
            'aggregate_method': 'ensemble_mean',
            'matrix': matrix_rows,
        }
    else:
        report['sections']['volatility'] = {
            'error': 'Volatility estimation failed.',
            'errors': vol_errors[:8],
            'hint': 'Run forecast_volatility_estimate directly for full diagnostics.',
        }

    # Backtest select best
    steps = int(p.get('backtest_steps', 25))
    requested_spacing = int(p.get('backtest_spacing', 10))
    spacing = requested_spacing
    if steps > 1 and spacing <= int(horizon):
        spacing = int(horizon) + 1
    try:
        rmse_tol = float(p.get('backtest_rmse_tolerance', 0.05))
    except Exception:
        rmse_tol = 0.05
    min_dir_acc_raw = p.get('backtest_min_directional_accuracy', p.get('backtest_min_accuracy'))
    try:
        min_dir_acc = float(min_dir_acc_raw) if min_dir_acc_raw is not None else None
    except Exception:
        min_dir_acc = None
    if min_dir_acc is not None:
        if not isfinite(min_dir_acc):
            min_dir_acc = None
        else:
            min_dir_acc = max(0.0, min(1.0, float(min_dir_acc)))
    from ..forecast import forecast_backtest_run
    methods = p.get('methods')
    requested_methods = normalize_report_methods(methods)
    single_method = requested_methods[0] if len(requested_methods) == 1 else None
    skip_backtest = bool(single_method) and report_section_enabled(p, 'forecast')
    bt = (
        {
            'status': 'omitted',
            'reason': 'single_method_direct_forecast',
            'method': single_method,
        }
        if skip_backtest and report_section_enabled(p, 'backtest')
        else _get_raw_result(
            forecast_backtest_run,
            symbol=symbol,
            timeframe=tf,
            horizon=int(horizon),
            steps=steps,
            spacing=spacing,
            start=start,
            end=end,
            methods=methods,
            denoise=denoise,
            detail='full',
        )
        if report_section_enabled(p, 'backtest')
        else {'error': 'backtest section not requested'}
    )
    sec_bt: Dict[str, Any]
    ranking: List[Dict[str, Any]] = []
    incomplete_backtest_methods: List[str] = []
    if bt.get('status') == 'omitted':
        sec_bt = dict(bt)
        best = None
    elif 'error' in bt:
        sec_bt = {'error': bt['error']}
        best = None
    else:
        try:
            res = bt.get('results', {})
            ranking, incomplete_backtest_methods = _complete_backtest_ranking(res)
        except Exception:
            pass
        topk = int(p.get('backtest_top_k', 3))
        sec_bt = {'ranking': ranking[:max(1, topk)], 'horizon': int(horizon), 'steps': steps, 'spacing': spacing}
        criteria_notes = 'Choose lowest RMSE; when methods are within tolerance of best RMSE, prefer higher directional accuracy.'
        if min_dir_acc is not None:
            criteria_notes += f' Require directional accuracy >= {min_dir_acc:.2f}.'
        sec_bt['selection_criteria'] = {
            'primary_metric': 'avg_rmse',
            'rmse_tolerance': float(rmse_tol),
            'rmse_tolerance_pct': float(rmse_tol * 100.0),
            'tie_breaker': 'avg_directional_accuracy',
            'secondary_tie_breaker': 'successful_tests',
            'notes': criteria_notes,
        }
        if incomplete_backtest_methods:
            sec_bt['excluded_incomplete_methods'] = incomplete_backtest_methods
        if min_dir_acc is not None:
            sec_bt['selection_criteria']['min_directional_accuracy'] = float(min_dir_acc)
            sec_bt['selection_criteria']['min_directional_accuracy_pct'] = float(min_dir_acc * 100.0)
        best = pick_best_forecast_method(
            bt,
            rmse_tolerance=rmse_tol,
            min_directional_accuracy=min_dir_acc,
        )
        if best is None and incomplete_backtest_methods and not ranking:
            sec_bt['selection_warning'] = (
                "No method had complete anchor coverage; partial aggregates were "
                "excluded from report selection."
            )
            sec_bt['selection_blocker'] = "incomplete_anchor_coverage"
        elif best is None and min_dir_acc is not None:
            sec_bt['selection_warning'] = (
                "No method met the minimum directional accuracy threshold."
            )
            sec_bt['selection_filtered_by_min_directional_accuracy'] = True
    report['sections']['backtest'] = sec_bt

    if best is not None:
        best_name, best_stats = best
        from ..forecast import forecast_generate
        bt_results = bt.get('results') if isinstance(bt, dict) else {}
        stats_by_method: Dict[str, Dict[str, Any]] = {}
        if isinstance(bt_results, dict):
            for method_name, method_stats in bt_results.items():
                if isinstance(method_stats, dict):
                    stats_by_method[str(method_name)] = method_stats

        quality_candidates: List[tuple[str, float]] = []
        for method_name, method_stats in stats_by_method.items():
            if method_stats.get('success') is not True:
                continue
            if not has_complete_forecast_backtest_coverage(method_stats):
                continue
            try:
                method_rmse = float(method_stats.get('avg_rmse'))
            except Exception:
                continue
            if not isfinite(method_rmse):
                continue
            if min_dir_acc is not None:
                try:
                    method_da = float(method_stats.get('avg_directional_accuracy'))
                except Exception:
                    continue
                if not isfinite(method_da) or method_da < float(min_dir_acc):
                    continue
            quality_candidates.append((method_name, method_rmse))
        quality_best_rmse = min(
            (item[1] for item in quality_candidates),
            default=float('inf'),
        )
        quality_limit = quality_best_rmse * (1.0 + max(0.0, float(rmse_tol)))
        eligible_methods = {
            method_name
            for method_name, method_rmse in quality_candidates
            if method_rmse <= quality_limit
        }

        ranked_methods: List[str] = []
        for row in ranking:
            if not isinstance(row, dict):
                continue
            method_name = str(row.get('method') or '').strip()
            if method_name and method_name not in ranked_methods:
                ranked_methods.append(method_name)

        candidate_methods: List[str] = [best_name]
        for method_name in ranked_methods:
            if method_name in eligible_methods and method_name not in candidate_methods:
                candidate_methods.append(method_name)

        selected_method = best_name
        selected_stats: Dict[str, Any] = dict(best_stats or {})
        selected_forecast: Optional[Dict[str, Any]] = None
        fallback_notes: List[str] = []
        first_error: Optional[str] = None
        failure_causes: Dict[str, Dict[str, str]] = {}

        if not report_section_enabled(p, 'forecast'):
            candidate_methods = []

        for method_name in candidate_methods:
            forecast_kwargs: Dict[str, Any] = {
                "symbol": symbol,
                "timeframe": tf,
                "method": method_name,
                "horizon": int(horizon),
                "start": start,
                "end": end,
                "denoise": denoise,
            }
            if p.get("forecast_ci_alpha") not in (None, ""):
                forecast_kwargs["ci_alpha"] = float(p.get("forecast_ci_alpha"))
            fc = _get_raw_result(forecast_generate, **forecast_kwargs)
            if 'error' in fc:
                if first_error is None:
                    first_error = str(fc.get('error') or '')
                failure_causes[method_name] = {
                    'code': 'forecast_error',
                    'message': str(fc.get('error') or 'forecast generation failed'),
                }
                fallback_notes.append(f"{method_name}: forecast error ({fc.get('error')})")
                continue
            if _is_degenerate_forecast_payload(fc):
                failure_causes[method_name] = {
                    'code': 'degenerate_forecast',
                    'message': 'forecast values were degenerate',
                }
                fallback_notes.append(f"{method_name}: degenerate forecast")
                continue
            selected_method = method_name
            selected_stats = dict(stats_by_method.get(method_name) or best_stats or {})
            selected_forecast = fc
            break

        if selected_forecast is None and report_section_enabled(p, 'forecast'):
            report['sections']['forecast'] = {
                'error': first_error or 'No quality-eligible method produced a usable forecast.',
                'method': best_name,
                'eligible_methods': sorted(eligible_methods),
            }
        elif selected_forecast is not None and report_section_enabled(p, 'forecast'):
            report['sections']['forecast'] = {
                'method': selected_method,
                **adapt_forecast_payload_for_report(selected_forecast),
            }
            if selected_method != best_name:
                initial_cause = failure_causes.get(best_name) or {
                    'code': 'forecast_fallback',
                    'message': 'initial method did not produce a usable forecast',
                }
                report['sections']['forecast']['fallback_from'] = best_name
                report['sections']['forecast']['fallback_reason_code'] = initial_cause['code']
                report['sections']['forecast']['fallback_reason'] = initial_cause['message']
                report['fallback_applied'] = True
                report['original_method'] = best_name
                report['fallback_method'] = selected_method
            if fallback_notes:
                report['sections']['forecast']['selection_warnings'] = fallback_notes

        best_method_payload: Dict[str, Any] = {
            'method': selected_method if selected_forecast is not None else best_name,
            'stats': {
                'avg_rmse': selected_stats.get('avg_rmse'),
                'avg_mae': selected_stats.get('avg_mae'),
                'avg_directional_accuracy': selected_stats.get('avg_directional_accuracy'),
                'successful_tests': selected_stats.get('successful_tests'),
            },
        }
        selection_basis: Dict[str, Any] = {
            'primary_metric': 'avg_rmse',
            'rmse_tolerance': float(rmse_tol),
            'rmse_tolerance_pct': float(rmse_tol * 100.0),
            'tie_breaker': 'avg_directional_accuracy',
            'secondary_tie_breaker': 'successful_tests',
            'initial_method': best_name,
            'selected_method': selected_method if selected_forecast is not None else best_name,
        }
        if min_dir_acc is not None:
            selection_basis['min_directional_accuracy'] = float(min_dir_acc)
            selection_basis['min_directional_accuracy_pct'] = float(min_dir_acc * 100.0)
        if ranking:
            try:
                best_rmse = float(ranking[0].get('avg_rmse'))
                if isfinite(best_rmse):
                    selection_basis['best_rmse'] = best_rmse
            except Exception:
                pass
        try:
            sel_rmse = float(selected_stats.get('avg_rmse'))
            if isfinite(sel_rmse):
                selection_basis['selected_rmse'] = sel_rmse
                if selection_basis.get('best_rmse') is not None:
                    tol_limit = float(selection_basis['best_rmse']) * (1.0 + float(rmse_tol))
                    selection_basis['rmse_tolerance_limit'] = tol_limit
                    selection_basis['within_rmse_tolerance'] = bool(sel_rmse <= tol_limit)
        except Exception:
            pass
        if selected_forecast is not None and selected_method != best_name:
            initial_cause = failure_causes.get(best_name) or {
                'code': 'forecast_fallback',
                'message': 'initial method did not produce a usable forecast',
            }
            selection_basis['fallback_applied'] = True
            selection_basis['fallback_reason_code'] = initial_cause['code']
            selection_basis['fallback_reason'] = initial_cause['message']
        best_method_payload['selection_basis'] = selection_basis
        if selected_forecast is not None and selected_method != best_name:
            best_method_payload['initial_method'] = best_name
            best_method_payload['selection_warning'] = (
                f"Initial best method failed ({initial_cause['code']}): "
                f"{initial_cause['message']}; fallback applied."
            )
        if fallback_notes:
            best_method_payload['selection_warnings'] = fallback_notes
        report['sections']['backtest']['best_method'] = best_method_payload

    if (
        report_section_enabled(p, 'forecast')
        and 'forecast' not in report['sections']
        and incomplete_backtest_methods
        and not ranking
    ):
        report['sections']['forecast'] = {
            'error': (
                'Forecast selection was blocked because no backtest method had '
                'complete anchor coverage.'
            ),
            'selection_mode': 'blocked_incomplete_anchor_coverage',
            'excluded_incomplete_methods': incomplete_backtest_methods,
        }

    if report_section_enabled(p, 'forecast') and 'forecast' not in report['sections']:
        from ..forecast import forecast_generate

        direct_method = single_method or (requested_methods[0] if requested_methods else "theta")
        forecast_kwargs = {
            "symbol": symbol,
            "timeframe": tf,
            "method": direct_method,
            "horizon": int(horizon),
            "start": start,
            "end": end,
            "denoise": denoise,
        }
        if p.get("forecast_ci_alpha") not in (None, ""):
            forecast_kwargs["ci_alpha"] = float(p.get("forecast_ci_alpha"))
        fc = _get_raw_result(forecast_generate, **forecast_kwargs)
        if "error" in fc:
            report["sections"]["forecast"] = {
                "error": fc.get("error") or "Direct forecast generation failed.",
                "method": direct_method,
                "selection_mode": "direct",
            }
        else:
            report["sections"]["forecast"] = {
                "method": direct_method,
                "selection_mode": "direct",
                **adapt_forecast_payload_for_report(fc),
            }

    # Barriers (grid)
    from ..forecast import forecast_barrier_optimize
    # Dynamic defaults to keep levels realistic and adaptive
    p.setdefault('grid_style', 'volatility')
    p.setdefault('vol_window', 250)
    p.setdefault('vol_min_mult', 0.6)
    p.setdefault('vol_max_mult', 2.2)
    p.setdefault('vol_sl_multiplier', 1.7)
    p.setdefault('vol_sl_steps', 9)
    # Set floors to avoid too-tight levels depending on mode
    if str(p.get('mode', 'pct')) == 'pct':
        p.setdefault('vol_floor_pct', 0.2)
    else:
        p.setdefault('vol_floor_ticks', 8.0)
    # Include trading costs to discourage too-tight levels in EV
    base_params = dict(p.get('params') or {})
    base_params.setdefault('spread_bps', 1.0)
    base_params.setdefault('slippage_bps', 0.5)
    if 'fast_defaults' in p:
        base_params.setdefault('fast_defaults', bool(p.get('fast_defaults')))
    base_params.setdefault('tp_min', float(p.get('tp_min', 0.25)))
    base_params.setdefault('tp_max', float(p.get('tp_max', 1.5)))
    base_params.setdefault('tp_steps', int(p.get('tp_steps', 7)))
    base_params.setdefault('sl_min', float(p.get('sl_min', 0.25)))
    base_params.setdefault('sl_max', float(p.get('sl_max', 2.5)))
    base_params.setdefault('sl_steps', int(p.get('sl_steps', 9)))
    base_params.setdefault('refine', bool(p.get('refine', False)))
    base_params.setdefault('refine_radius', float(p.get('refine_radius', 0.3)))
    base_params.setdefault('refine_steps', int(p.get('refine_steps', 5)))
    # Reasonable risk/reward filter defaults per template
    rr_min_default = p.get('rr_min', 0.8)
    rr_max_default = p.get('rr_max', 2.0)
    base_params.setdefault('rr_min', rr_min_default)
    base_params.setdefault('rr_max', rr_max_default)
    for barrier_key in (
        'tp_min',
        'tp_max',
        'tp_steps',
        'sl_min',
        'sl_max',
        'sl_steps',
        'vol_window',
        'vol_min_mult',
        'vol_max_mult',
        'vol_steps',
        'vol_sl_multiplier',
        'vol_sl_steps',
        'vol_floor_pct',
        'vol_floor_ticks',
        'refine',
        'refine_radius',
        'refine_steps',
    ):
        if barrier_key in p:
            base_params.setdefault(barrier_key, p.get(barrier_key))
    p['params'] = base_params

    mode_val = str(p.get('mode', 'pct'))
    barrier_method = str(p.get('barrier_method', 'auto'))
    if not report_section_enabled(p, 'barriers'):
        grid_long = grid_short = None
    else:
        grid_long = _get_raw_result(forecast_barrier_optimize,
            symbol=symbol,
            timeframe=tf,
            horizon=int(horizon),
            method=barrier_method,
            mode=mode_val,
            params=p.get('params'),
            objective=str(p.get('objective','ev')),
            top_k=int(p.get('top_k', 5)),
            grid_style=str(p.get('grid_style', 'fixed')),
            preset=p.get('grid_preset', p.get('preset')),
            search_profile=str(p.get('search_profile', 'fast')),
            denoise=denoise,
            direction='long',
            start=start,
            end=context_end,
        )
        grid_short = _get_raw_result(forecast_barrier_optimize,
            symbol=symbol,
            timeframe=tf,
            horizon=int(horizon),
            method=barrier_method,
            mode=mode_val,
            params=p.get('params'),
            objective=str(p.get('objective','ev')),
            top_k=int(p.get('top_k', 5)),
            grid_style=str(p.get('grid_style', 'fixed')),
            preset=p.get('grid_preset', p.get('preset')),
            search_profile=str(p.get('search_profile', 'fast')),
            denoise=denoise,
            direction='short',
            start=start,
            end=context_end,
        )
    sec_bar: Dict[str, Any] = {}
    if not report_section_enabled(p, 'barriers'):
        sec_bar = {'error': 'barriers section not requested'}
    elif isinstance(grid_long, dict) and isinstance(grid_short, dict) and 'error' in grid_long and 'error' in grid_short:
        sec_bar = {'error': grid_long.get('error') or grid_short.get('error') or 'Barrier optimization failed'}
    else:
        assert isinstance(grid_long, dict) and isinstance(grid_short, dict)
        if 'error' not in grid_long:
            sec_bar['long'] = summarize_barrier_grid(grid_long, top_k=int(p.get('top_k', 5)))
        else:
            sec_bar['long'] = {'error': grid_long.get('error')}
        if 'error' not in grid_short:
            sec_bar['short'] = summarize_barrier_grid(grid_short, top_k=int(p.get('top_k', 5)))
        else:
            sec_bar['short'] = {'error': grid_short.get('error')}
        conflict_directions: List[str] = []
        caution_parts: List[str] = []
        for direction in ('long', 'short'):
            sub = sec_bar.get(direction)
            if not isinstance(sub, dict):
                continue
            if bool(sub.get('ev_edge_conflict')):
                conflict_directions.append(direction)
            caution_text = sub.get('caution')
            if isinstance(caution_text, str) and caution_text.strip():
                caution_parts.append(f"{direction}: {caution_text.strip()}")
        if conflict_directions:
            sec_bar['ev_edge_conflict'] = True
            sec_bar['ev_edge_conflict_directions'] = conflict_directions
            sec_bar['ev_edge_conflict_reason'] = "ev and edge have opposite signs"
            if caution_parts:
                sec_bar['caution'] = "; ".join(caution_parts)
            else:
                sec_bar['caution'] = (
                    "EV and edge signs conflict in barrier recommendations; inspect win probability "
                    "and break-even thresholds before trading."
                )
        direction_results = [
            sec_bar.get(direction)
            for direction in ('long', 'short')
            if isinstance(sec_bar.get(direction), dict)
        ]
        if direction_results and all(
            str(result.get('status') or '').strip().lower() == 'non_viable'
            or result.get('mathematically_viable') is False
            for result in direction_results
        ):
            sec_bar['status'] = 'non_viable'
            sec_bar['status_reason'] = (
                'No barrier direction produced a mathematically viable setup.'
            )
            sec_bar['recommendation'] = 'avoid'
            sec_bar['viable_directions'] = []
    sec_bar['mode'] = mode_val
    sec_bar['method'] = barrier_method
    sec_bar['search_profile'] = str(p.get('search_profile', 'fast'))
    sec_bar['note'] = (
        "Report barriers are produced by an independent optimization run; "
        "standalone forecast_barrier_optimize may yield different candidates. "
        "edge measures win-rate margin versus breakeven, while EV also weights reward/risk."
    )
    report['sections']['barriers'] = sec_bar

    # Patterns
    from ..patterns import patterns_detect
    pattern_mode = str(p.get('patterns_mode') or 'candlestick').strip() or 'candlestick'
    extra_modes = p.get('patterns_extra_modes') or []
    if isinstance(extra_modes, str):
        extra_modes = [item.strip() for item in extra_modes.replace(',', ' ').split() if item.strip()]
    pattern_modes = [pattern_mode]
    for extra_mode in extra_modes:
        extra_name = str(extra_mode).strip()
        if extra_name and extra_name not in pattern_modes:
            pattern_modes.append(extra_name)
    last_n_bars = p.get('patterns_last_n_bars', 5 if pattern_mode == 'candlestick' else None)
    try:
        last_n_value = int(last_n_bars) if last_n_bars is not None else None
    except Exception:
        last_n_value = None
    detections: List[Dict[str, Any]] = []
    pattern_errors: List[Dict[str, str]] = []
    if report_section_enabled(p, 'patterns'):
        for mode_name in pattern_modes:
            pattern_kwargs: Dict[str, Any] = {
                "symbol": symbol,
                "timeframe": tf,
                "mode": mode_name,
                "detail": "compact",
                "lookback": int(p.get("patterns_limit", 120)),
                "start": start,
                "end": end,
                "top_k": int(p.get("patterns_top_k", 5)),
            }
            if last_n_value is not None and mode_name == "candlestick":
                pattern_kwargs["last_n_bars"] = last_n_value
            pats = _get_raw_result(patterns_detect, **pattern_kwargs)
            if "error" in pats:
                pattern_errors.append(
                    {"mode": mode_name, "error": str(pats.get("error") or "pattern detection failed")}
                )
                continue
            detections.extend(extract_report_pattern_rows(pats, limit=5))
        if detections:
            report["sections"]["patterns"] = {
                "modes": pattern_modes,
                "recent": detections[:5],
            }
        elif pattern_errors:
            report["sections"]["patterns"] = {
                "error": pattern_errors[0]["error"],
                "modes": pattern_modes,
                "errors": pattern_errors,
            }
        else:
            report["sections"]["patterns"] = {"modes": pattern_modes, "recent": []}

    attach_optional_report_sections(
        report,
        call=_get_raw_result,
        symbol=symbol,
        timeframe=tf,
        params=p,
        start=start,
        end=end,
    )

    for section_name in list(report['sections']):
        if not report_section_enabled(p, section_name):
            report['sections'].pop(section_name, None)

    return report
