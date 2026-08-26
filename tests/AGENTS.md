# tests/ — Test Suite

~269 `test_*.py` files, hybrid pytest/unittest.TestCase. GitHub Actions runs `python -m pytest tests` on Windows (plus a separate foundation-smoke job). No pytest.ini — all config in `conftest.py`.

## NAMING CONVENTIONS

Files mirror `src/mtdata/` module structure with suffix indicating scope:

| Suffix | Scope | Max Lines | Example |
|--------|-------|-----------|---------|
| `_pure` | Unit tests for pure functions, no mocks | 500 | `test_indicators_pure.py` |
| `_business_logic` | Logic tests, may mock external deps | 800 | `test_core_forecast_business_logic.py` |
| `_coverage` | Comprehensive coverage of a module | 1,000 | `test_cli_coverage.py` |
| `_extended` | Integration/extended scenarios | 600 | `test_elliott_extended.py` |
| _(no suffix)_ | General tests | 500 | `test_data_service.py` |

### Size Guidelines

- Files larger than their suffix limit should be split into subdirectories:
  - `tests/core/cli/` - CLI tests (formerly `test_cli_coverage.py`)
  - `tests/patterns/coverage/` - Pattern coverage tests (formerly `test_patterns_core_coverage.py`)
  - `tests/trading/coverage/` - Trading coverage tests (formerly `test_trading_coverage.py`)
  - `tests/services/wait_event/` - Wait event tests (formerly `test_wait_event_use_cases.py`)

### Test Domains

| Prefix | Covers | File Count |
|--------|--------|------------|
| `test_trading_*` | Trading logic | 20+ |
| `test_forecast_*` | Forecast engine | 15 |
| `test_report_*` | Report generation | 5 |
| `test_data_*` | Data service | 2 |
| `test_patterns_*` | Pattern detection | 16 |
| `test_regime_*` | Regime detection | 13 |
| `test_volatility_*` | Volatility | 2 |
| `test_indicators_*` | Indicators | 3 |
| `test_denoise_*` | Denoising | 3 |
| `test_web_api_*` | Web API | 3 |
| `test_cli_*` | CLI | 6+ (`tests/core/cli/`) |
| `test_server_*` | MCP server | 2 |

## SUBDIRECTORIES

Large modules use subdirectories for organization:

| Directory | Contains | Original File |
|-----------|----------|---------------|
| `tests/core/cli/` | 6 focused CLI test files | `test_cli_coverage.py` (6,006 lines) |
| `tests/patterns/coverage/` | 7 focused pattern test files | `test_patterns_core_coverage.py` (2,461 lines) |
| `tests/patterns/` | 3 split pattern business logic files | `test_patterns_business_logic.py` (3,736 lines) |
| `tests/trading/coverage/` | 7 focused trading test files | `test_trading_coverage.py` (2,837 lines) |
| `tests/services/wait_event/` | 4 focused wait event files | `test_wait_event_use_cases.py` (2,744 lines) |
| `tests/services/coverage/` | 6 focused data-service test files | `test_data_service_coverage.py` (2,992 lines) |
| `tests/forecast/barriers/` | 8 focused barrier test files | `test_forecast_barriers.py` (2,723 lines) |

## CONFTEST FIXTURES

`conftest.py` (245 lines) handles critical test infrastructure:

- **`sys.modules` stubbing**: Injects `MagicMock` for `MetaTrader5` into `sys.modules` at import time so tests run without a real MT5 terminal.
- **`mt5_module` fixture**: Per-test MT5 stub with automatic cleanup.
- **`pytest_runtest_setup/teardown` hooks**: Clear scipy's `_issubclass_fast` LRU cache between tests to prevent pollution from fake `torch` module injection.
- **Path setup**: Adds `src/` and project root to `sys.path`.

## MOCKING PATTERNS

- **MT5 interactions**: Always use `unittest.mock.patch` or the `mt5_module` fixture. Never call real MT5 API in tests.
- **torch/GPU**: Some `_coverage` tests inject a fake `torch` module — conftest clears scipy cache to prevent cross-test leaks.
- **Style**: Both `def test_*()` (pytest-style) and `class TestX(unittest.TestCase)` coexist. New tests can use either.

## ANTI-PATTERNS

- **Never** run tests against a live MT5 account — always mock.
- **Never** delete failing tests to "pass" — fix the underlying issue.
- **Never** import `conftest.py` directly — pytest discovers it automatically.
- **Never** let coverage files grow beyond 1,000 lines — split into subdirectories when they approach the limit.

## CLEANUP HISTORY

2026-06-15: Phase 4 test reorganization
- Split `test_forecast_barriers.py` (2,723 lines) into 8 focused files in `tests/forecast/barriers/`
  - `test_hit_probabilities.py` (318 lines): hit probabilities, closed form, GBM/HMM/bootstrap/GARCH
  - `test_optimize_basic.py` (292 lines): optimize signature, optuna, fast_defaults, bool flags
  - `test_optimize_profiles_ensemble.py` (339 lines): search profiles, ensemble, live price, geometry
  - `test_optimize_output_grid.py` (456 lines): output modes, grids, constraints, tie probs, trade gate
  - `test_optimize_guardrails.py` (382 lines): input validation, EV warnings, guardrails
  - `test_trading_costs.py` (483 lines): spread/commission/slippage, cost-adjusted metrics
  - `test_statistical_quality.py` (383 lines): statistical significance, error handling, ensemble degradation
  - `test_candidate_eval.py` (159 lines): candidate viability, unresolved PnL, barrier geometry
- All 110 tests preserved with identical assertions
- Renamed 4 files to match `_business_logic` convention
- Deleted 3 duplicate files (cli, data_service, trading business_logic)
- Split 4 large files into 19 focused subfiles in 5 subdirectories
- Total: 235 test files (was 213), +22 files but much better organization

2026-06-15: Phase 5 test reorganization
- Split `test_data_service_coverage.py` (2,992 lines) into 7 focused files in `tests/services/coverage/`
  - `_helpers.py` (141 lines): shared imports, fixture builders, patch-target constants
  - `test_internals.py` (670 lines): helper tests, TestShiftRateTimes, TestFetchRatesWithWarmup, TestBuildRatesDf, TestTrimDfToTarget
  - `test_fetch_candles_core.py` (846 lines): success paths, error paths, datetime queries, forming-candle and quality-filter behaviour
  - `test_fetch_candles_indicators.py` (497 lines): indicator specs, NaN warmup-retry variants
  - `test_fetch_candles_advanced.py` (160 lines): simplify and denoise features
  - `test_fetch_ticks.py` (667 lines): all fetch_ticks tests including volume stats and simplify
  - `test_edge_cases.py` (237 lines): edge cases from TestEdgeCases
- 157 tests preserved with identical assertions; 2 pre-existing failures carried forward unchanged
- Total: 242 test files (+7 new, 1 removed = net +6)

2026-08-12: Satellite fold (no coverage cut)
- Deleted `test_forecast_helpers_coverage.py` (strict subset of `test_forecast_common_coverage.py` / `test_forecast_common_business_logic.py`)
- Folded leftover satellite files into domain homes (wait-event, wait-candle, market_status, ticker, job_store, minimal_output, finviz symbols, CLI help/list/typed-args, remaining_logic_fixes, volume_validation_fix)
- Kept every unique assertion; dropped only literal duplicates (select-simplify ticks, selectkbest wrapper without y)
- Total: 269 test files (was 285)
