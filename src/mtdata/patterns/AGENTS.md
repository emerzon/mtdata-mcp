# patterns/ — Pattern Detection

Chart pattern and Elliott wave detection. 15 files across `patterns/` and `classic_impl/`.

## PUBLIC API

Exported from `__init__.py`:
- `detect_classic_patterns(df, config)` → `ClassicPatternResult`
- `detect_elliott_waves(df, config)` → `ElliottWaveResult`
- `detect_fractal_patterns(df, config)` → `FractalPatternResult`
- `detect_harmonic_patterns(df, config)` → `HarmonicPatternResult`
- Config classes: `ClassicDetectorConfig`, `ElliottWaveConfig`, `FractalDetectorConfig`, `HarmonicDetectorConfig`

## FILE MAP

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 31 | Public API exports (see above) |
| `classic.py` | 255 | Facade — delegates to `classic_impl/` algorithms |
| `elliott.py` | 2690 | Standalone Elliott wave detection |
| `elliott_adaptation.py` | — | Elliott adaptation helpers |
| `candlestick.py` | 1094 | Candlestick pattern detection |
| `fractal.py` | — | Fractal pattern detection |
| `harmonic.py` | — | Harmonic pattern detection |
| `common.py` | 449 | Shared pattern types and helpers |

### classic_impl/ — Pattern Algorithms

| File | Purpose |
|------|---------|
| `config.py` | Detection configuration/thresholds |
| `reversal.py` | Head-and-shoulders, double tops/bottoms |
| `continuation.py` | Flags, pennants, rectangles |
| `shapes.py` | Triangles, wedges, channels |
| `trend.py` | Trendline-based patterns |
| `utils.py` | Shared geometry/math helpers |
| `__init__.py` | Subpackage init |

## CONVENTIONS

- `classic.py` is the ONLY entry point for classic patterns — never call `classic_impl/` directly.
- Optional dependency `stock-pattern` (manual install; see `docs/SETUP.md` §6) is wired as a classic **engine** in `core/patterns.py` (`engine="stock_pattern"` / `ensemble=true`), not inside `classic.py`.
- Do NOT confuse with `utils/patterns.py` — that file has shared helpers, this package has detectors.
- Consumed by `core/patterns.py` (MCP tools) and `core/patterns_support.py` (2625 lines of helpers).
