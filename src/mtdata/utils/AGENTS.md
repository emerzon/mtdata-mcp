# utils/ — Cross-Cutting Utilities

Shared numerical and formatting helpers imported by `core/`, `forecast/`, `patterns/`, and `services/`. 39 Python files including the `denoise/` package, heavy numpy/scipy/pandas-ta code.

## FILE MAP

| File | Lines | Purpose |
|------|-------|---------|
| `indicators.py` | — | 193 technical indicators (pandas-ta; optional TA-Lib oracle backend) |
| `denoise/` | package | 10+ signal filters: wavelet, EMD, VMD, Kalman, Savgol, LOESS, etc. |
| `simplify.py` | 913 | Price series simplification/compression |
| `dimred.py` | 592 | Dimension reduction (PCA, t-SNE, UMAP wrappers) |
| `minimal_output.py` | 2467 | Compact output formatting for CLI/MCP responses |
| `patterns.py` | 804 | Pattern detection helpers (NOT the `patterns/` module) |
| `formatting.py` | — | String/number formatting utilities |
| `mt5.py` | — | MT5 connection wrapper (`mt5_connection` context manager) |
| `mt5_enums.py` | — | MT5 enum definitions |
| `barriers.py` | — | Barrier calculation helpers |
| `market_metadata.py` | — | Market metadata helpers |
| `regime_heuristics.py` | — | Regime detection helpers |
| `symbol.py` | — | Symbol normalization/lookup |
| `utils.py` | — | General-purpose helpers (`_normalize_ohlcv_arg`, etc.) |
| `__init__.py` | — | Package init (docstring only) |

## CONVENTIONS

- **`utils/patterns.py` ≠ `patterns/`**: This file has shared pattern helpers; the `patterns/` package has the actual detectors.
- **No `__init__.py` exports** — import modules directly: `from mtdata.utils.denoise import ...`
- **Heavy numerical**: Most files depend on numpy, scipy, pandas. `indicators.py` wraps pandas-ta; TA-Lib is optional.
- **`denoise/` filter methods** follow a common signature: `(series, **params) → filtered_series`. Each filter is a standalone function.
