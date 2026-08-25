# Denoising and smoothing

**Audience:** User

Prices are a mix of **structure** and **noise** (microstructure bounce, spreads, short bursts). Denoising smooths series so trends and indicators are easier to read — optionally as a preprocess step for forecasts.

**Trade-off:** more smoothing → clearer trend, more **lag**. Prefer light filters first.

**Dense terms:** [Denoising](GLOSSARY.md#denoising) · [Causal filters](GLOSSARY.md#causal-vs-non-causal-filters) · [Kalman](GLOSSARY.md#kalman-filter) · [Wavelet](GLOSSARY.md#wavelet-denoise--regimes) · [EMA](GLOSSARY.md#moving-average)

**Related:** [CLI](CLI.md) · [Indicators](TECHNICAL_INDICATORS.md) · [Forecasting](FORECAST.md) · [Simplification](SIMPLIFICATION.md) · [Glossary](GLOSSARY.md)

---

## Why denoise?

| Goal | How denoise helps |
|------|-------------------|
| Cleaner signals | Fewer false indicator flips |
| Clearer trend | Underlying path is easier to see |
| Stabler models | Less outlier-driven fit noise |
| Spike control | Median / robust filters dampen extremes |

**Simplify vs denoise:** [SIMPLIFICATION.md](SIMPLIFICATION.md) reduces *how many points* you return; denoise changes *the values*.

---

## Quick start

**Smooth closing prices:**
```bash
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 200 --denoise ema --denoise-params "alpha=0.2"
```

**Remove spikes:**
```bash
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 200 --denoise median --denoise-params "window=5"
```

## Dependencies

The default package declares the denoising libraries used by the method catalog:

- `statsmodels`: used by LOESS and STL
- `PyWavelets`, `vmdpy`, `EMD-signal`: used by wavelet, VMD, and EMD-family methods

Tip: `GET /api/denoise/methods` (see [WEB_API.md](WEB_API.md)) reports availability and required packages for the current environment.

---

## When to Apply: Pre vs Post Indicators

### Pre-Indicator (`when=pre_ti`)
Add a suffixed denoised price series, then calculate indicators on the smoothed
values while retaining broker OHLC in the canonical columns. Indicators derived
from those smoothed inputs use the same suffix (`rsi_14_dn`, for example), so
they cannot be mistaken for indicators calculated from the raw `close` column.

**Use when:** You want smoother inputs for trend estimation.

```bash
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 200 --indicators "rsi(14)" --denoise ema --denoise-params "columns=close,when=pre_ti,alpha=0.2"
```

### Post-Indicator (`when=post_ti`)
Calculate indicators on raw data, then smooth the indicator output.

**Use when:** You want to keep raw price intact but reduce indicator noise.

```bash
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 200 --indicators "rsi(14)" --denoise ema --denoise-params "columns=rsi_14,when=post_ti,alpha=0.3"
```

---

## Denoising Methods

### Moving Averages

General-purpose smoothing.

| Method | Description | Parameters |
|--------|-------------|------------|
| `ema` | Exponential Moving Average | `span` (default 10) or `alpha` (0–1; overrides span) |
| `sma` | Simple Moving Average | `window` |
| `kama` | Kaufman Adaptive Moving Average | `window` (default 10), `fast` (2), `slow` (30) |

**Example:**
```bash
--denoise ema --denoise-params "span=10"
```

### Robust Filters

Remove outliers and spikes without excessive smoothing.

| Method | Description | Parameters |
|--------|-------------|------------|
| `median` | Median filter | `window` |
| `hampel` | Hampel identifier | `window`, `n_sigmas` (default 3.0) |

**Example (spike removal):**
```bash
--denoise median --denoise-params "window=5"
```

### Frequency Filters

Separate high-frequency noise from low-frequency trend.

| Method | Description | Parameters |
|--------|-------------|------------|
| `lowpass_fft` | FFT low-pass filter | `cutoff_ratio` |
| `butterworth` | Butterworth filter | `order`, `cutoff` |
| `supersmoother` | Ehlers two-pole SuperSmoother | `period` (default 10) |

**Example:**
```bash
--denoise lowpass_fft --denoise-params "cutoff_ratio=0.1,causality=zero_phase"
```

### Trend Extractors

Isolate the slow-moving trend component.

| Method | Description | Parameters |
|--------|-------------|------------|
| `hp` | Hodrick-Prescott filter | `lamb` |
| `l1_trend` | L1 trend filter | `lamb`, `n_iter`, `rho` |
| `tv` | Total variation denoising | `weight`, `n_iter`, `tol` |

**Example:**
```bash
--denoise hp --denoise-params "lamb=1600,causality=zero_phase"
```

### Adaptive Filters

Automatically adjust smoothing based on data.

| Method | Description | Parameters |
|--------|-------------|------------|
| `kalman` | Kalman filter | `process_var`, `measurement_var` |
| `kalman_robust` | Student-t robust Kalman | `process_var`, `measurement_var`, `nu` (default 4) |
| `lms` | Least Mean Squares | `mu`, `order`, `eps`, `leak` |
| `rls` | Recursive Least Squares | `delta`, `lambda_`, `order` |

**Example:**
```bash
--denoise kalman --denoise-params "process_var=0.01"
```

### Polynomial / Local Regression

Fit local curves to smooth the data.

| Method | Description | Parameters |
|--------|-------------|------------|
| `savgol` | Savitzky-Golay smoothing (causal trailing window, or zero-phase) | `window`, `polyorder` |
| `loess` | Local polynomial regression | `frac`, `it` |

**Example:**
```bash
--denoise savgol --denoise-params "window=11,polyorder=3,causality=zero_phase"
```

### Decomposition Methods

Split into components and reconstruct smoother parts.

| Method | Description | Parameters |
|--------|-------------|------------|
| `stl` | Seasonal-Trend decomposition | required `period` (at least 2 and shorter than the series), `component` (default `trend`) |
| `ssa` | Singular Spectrum Analysis | `window` |
| `vmd` | Variational Mode Decomposition | `k`, `alpha`, `drop_modes` |
| `wavelet` | Wavelet denoising | `wavelet`, `level` |
| `wavelet_packet` | Wavelet packet denoising | `wavelet`, `level` |
| `emd` | Empirical Mode Decomposition | `drop_imfs` |
| `eemd` | Ensemble EMD | `drop_imfs`, `noise_strength` |
| `ceemdan` | Complete EEMD with Adaptive Noise | `drop_imfs` |

**Example:**
```bash
--denoise wavelet --denoise-params "wavelet=db4,level=3,causality=zero_phase"
```

### Kernel / Smoothing Filters

| Method | Description | Parameters |
|--------|-------------|------------|
| `gaussian` | Gaussian kernel smoothing | `sigma` |
| `bilateral` | Bilateral filter (edge-preserving) | `sigma_s`, `sigma_r` |
| `whittaker` | Whittaker smoother | `lamb`, `order` |
| `beta` | Robust beta smoother | `window`, `beta` |
| `preaverage` | Jacod-style pre-averaging of increments | `window` (default 10), `space` (`level` or `log`) |

---

## Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `columns` | Which columns to denoise | `close` |
| `when` | `pre_ti` or `post_ti` | `pre_ti` |
| `causality` | `causal` or explicitly opted-in `zero_phase` | `causal` |
| `keep_original` | Keep original column and add an `_dn` suffix; set `false` to overwrite explicitly | `true` |
| `alpha` | Optional smoothing factor (EMA; overrides `span`) | — |
| `window` | Window size (filters) | method-specific |

---

## Avoiding Look-Ahead Bias

**Critical for backtesting:** Some filters use future data to smooth each point (zero-phase filtering). This looks great on charts but creates unrealistic results.

**Causal-capable filters** (default to past-only processing):
- `ema`, `sma`, `median`, `butterworth`, `kalman`, `kalman_robust`, `hampel`, `bilateral`, `lms`, `rls`, `beta`, `savgol`, `supersmoother`, `kama`, `preaverage`

**Non-causal-only filters** (use past and future):
- `lowpass_fft`, `wavelet`, `wavelet_packet`, `hp`, `whittaker`, `l1_trend`, `gaussian`, `loess`, `stl`, `tv`, `ssa`, `vmd`, `emd`, `eemd`, `ceemdan`

Causal-capable filters and the public `denoise_series` helper default to causal
operation. Non-causal-only presets are rejected unless the request explicitly
sets `causality=zero_phase`. For a causal-capable filter such as Butterworth,
request `causality=zero_phase` explicitly. `denoise_list_methods` marks methods
that cannot run causally with `requires_causality_opt_in=true`; on the CLI, opt
in with `--denoise-params causality=zero_phase`. Use zero-phase methods only for
retrospective analysis, not backtesting or live trading.

---

## Method Selection Guide

| Noise Type | Recommended Method |
|------------|-------------------|
| General high-frequency noise | `ema`, `sma` |
| Spikes/outliers | `median`, `hampel` |
| Microstructure noise | `preaverage`, `kalman_robust`, `kalman` |
| Seasonal patterns | `stl` |
| Unknown/complex | Start with `ema`, try `kalman` or `kama` |

---

## Examples

### Smooth Closing Prices
```bash
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 500 --denoise ema --denoise-params "alpha=0.2"
```

### Remove Price Spikes
```bash
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 500 --denoise hampel --denoise-params "window=7,n_sigmas=3"
```

### Smooth RSI Output
```bash
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 500 --indicators "rsi(14)" --denoise ema --denoise-params "columns=rsi_14,when=post_ti,alpha=0.3"
```

### Kalman Filter (Adaptive)
```bash
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 500 --denoise kalman --denoise-params "process_var=0.01"
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Basic EMA smoothing | `--denoise ema --denoise-params "alpha=0.2"` |
| Spike removal | `--denoise median --denoise-params "window=5"` |
| Adaptive filter | `--denoise kalman` |
| Overwrite the source column explicitly | `--denoise-params "keep_original=false"` |
| Post-indicator smoothing | `--denoise-params "when=post_ti"` |

---

## Discovering Methods

Two read-only helpers describe the available denoise filters and their parameters:

```bash
mtdata-cli denoise_list_methods --json                 # all methods, dependencies, causality support
mtdata-cli denoise_list_methods --available-only --json # only methods whose dependencies are installed
mtdata-cli denoise_describe kalman --json               # parameters and defaults for one method
```

---

## See Also

- [GLOSSARY.md](GLOSSARY.md) — Term definitions
- [TECHNICAL_INDICATORS.md](TECHNICAL_INDICATORS.md) — Indicators to denoise
- [FORECAST.md](FORECAST.md) — Using denoising in forecasts
