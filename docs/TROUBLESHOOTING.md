# Troubleshooting

**Audience:** User

Something broken? Start with a **small read-only snapshot** — it usually tells you whether the problem is install, MetaTrader 5 connection, symbol visibility, or history. Then jump to the matching section below.

**Related:** [Setup](SETUP.md) · [Web UI](WEBUI.md) · [CLI](CLI.md) · [Limitations](LIMITATIONS.md) · [Timestamps](TIMESTAMPS.md) (Operator)

---

## Start here: diagnostic snapshot

```bash
python --version
mtdata-cli --help
mtdata-cli symbols_list --limit 10
mtdata-cli symbols_describe EURUSD --json
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 5 --json
```

If the Web API is involved, also check:

```bash
mtdata-webapi
curl http://127.0.0.1:8000/api/v1/health
curl http://127.0.0.1:8000/api/v1/ready
```

`/health` confirms that the Web API process responds; `/ready` also checks the
MT5 connection. For `mtdata-sse` or `mtdata-streamable-http`, use `/live` for
process liveness and `/ready` for MT5 readiness.

Keep the first troubleshooting pass read-only. Do not use `trade_place`, `trade_modify`, or `trade_close` while diagnosing setup issues unless you are intentionally testing on a demo account.

---

## Web UI

User tour: [WEBUI.md](WEBUI.md).

| What you see | What to try |
|--------------|-------------|
| Enablement page / “build the UI” instead of a chart | In `webui/`: `npm ci && npm run build`, then restart `mtdata-webapi`. |
| Chart opens but stays empty | Pick a symbol that exists in MetaTrader 5 Market Watch, then reload. |
| Connection chip is not ready | MetaTrader 5 must be running and logged in. Check `http://127.0.0.1:8000/api/v1/ready`. |
| Overlay or forecast banner error | The rest of the chart can still work. Retry after `/ready` is healthy. |
| Auth control / 401 | Enter the same token as `WEBAPI_AUTH_TOKEN`. It lives only in this tab. |
| Tools invoke of `wait_event` is refused | That tool is omitted from the Web runner. Use the [CLI wait guide](WAIT_EVENT.md) instead. |
| Tools confirm box for `trade_place` | That can send a **real** order if dry-run is off. Demo account; see [TRADING_SAFETY.md](TRADING_SAFETY.md). |

---

## Connection Issues

### "Could not connect to MT5" or Empty Data

**Checklist:**
1. MetaTrader 5 terminal is installed and running
2. Terminal is logged in to a broker account
3. Symbol is visible in Market Watch

**Test connection:**
```bash
mtdata-cli symbols_list --limit 10
```

If this works but candles fail:
- Check symbol spelling (case-sensitive for some brokers)
- Ensure symbol is added to Market Watch in MT5

### Terminal Keeps Disconnecting

**Possible causes:**
- MT5 requires periodic reconnection
- Internet stability issues

**Solution:** The library auto-reconnects. For a long-running **Operator** script, catch the error and retry:

```python
try:
    result = tool_function(...)
except Exception as e:
    # Retry logic here
```

---

## Parameter Errors

### "validation error ... Field required"

**Example:**
```
1 validation error for InputSchema
symbol
  Field required [type=missing, input_value={}, input_type=dict]
```

**Cause:** A required parameter is missing.

**Solution:** Check command help and provide required arguments:
```bash
mtdata-cli forecast_generate --help
mtdata-cli forecast_generate EURUSD --horizon 12  # symbol is positional
```

### "Invalid choice" for Timeframe/Method/Mode

**Cause:** Invalid value for a constrained parameter.

**Solution:** Use `--help` to see allowed values:
```bash
mtdata-cli forecast_volatility_estimate --help
mtdata-cli patterns_detect --help
```

**Supported timeframes:** `M1`, `M2`, `M3`, `M4`, `M5`, `M6`, `M10`, `M12`, `M15`, `M20`, `M30`, `H1`, `H2`, `H3`, `H4`, `H6`, `H8`, `H12`, `D1`, `W1`, `MN1`. Broker history availability may vary.

### "Unknown parameter" in --params

**Example:**
```bash
--params "invalid_param=5"
```

**Solution:** Check method documentation:
```bash
mtdata-cli forecast_list_methods --json
mtdata-cli indicators_describe rsi --json
```

---

## Model/Library Issues

### "Method X not available"

**Cause:** Required optional dependency not installed.

**Solution:** Check availability:
```bash
mtdata-cli forecast_list_methods --json
```

Look for `available: false` and the `requires` field. Install missing packages:

If you're installing mtdata extras, run the extra commands below from the repository root.

```bash
pip install chronos-forecasting torch  # For Chronos
pip install statsforecast              # For StatsForecast models
pip install arch                       # For GARCH
pip install statsmodels                # For ARIMA/ETS + causal_discover_signals
pip install -e ".[dimred-ext]"         # For UMAP dimred (Web UI / analysis); or: pip install umap-learn
pip install QuantLib                   # For barrier option pricing & Heston calibration
pip install optuna                     # For Bayesian hyperparameter tuning
pip install neuralforecast torch       # For neural models; fails on Windows Python 3.14 (no ray win/cp314 wheel)
pip install -e ".[forecast-timesfm]"   # From the repo root; installs TimesFM 3.x from PyPI (2.5 + 3.0 APIs)
```

### "Import error" or "Module not found"

**Solution:** Install the package or use a different method:
```bash
pip install <missing_package>
```

Or check if a similar method is available without extra dependencies:
```bash
mtdata-cli forecast_list_library_models native  # Native methods have minimal deps
```

---

## Output Issues

### Output is Hard to Parse

**Solution:** Use JSON format:
```bash
mtdata-cli symbols_describe EURUSD --json
mtdata-cli forecast_generate EURUSD --horizon 12 --json
```

**Operator:** pipe to `jq` if you already use it:

```bash
mtdata-cli forecast_generate EURUSD --json | jq '.forecast'
```

### Output is Too Verbose

**Solution:** Keep the default compact detail level, or set `--detail compact` explicitly.

### Missing Columns in Output

**Cause:** Indicator calculation failed or data insufficient.

**Solution:** Check that:
1. Enough bars are fetched for the indicator period
2. Indicator name is spelled correctly

```bash
# Check indicator syntax
mtdata-cli indicators_describe rsi

# Fetch enough bars (at least period + some buffer)
mtdata-cli data_fetch_candles EURUSD --limit 200 --indicators "rsi(14)"
```

---

## Data Issues

### No Data Returned

**Possible causes:**
- Symbol not in Market Watch
- Time range has no data (weekend, holiday)
- Broker doesn't provide history for this symbol

**Solution:**
```bash
# Check if symbol exists
mtdata-cli symbols_list --limit 100

# Try without date filters
mtdata-cli data_fetch_candles EURUSD --limit 100
```

### Timestamps Look Wrong

**Cause:** The payload may be displayed in the detected client-local timezone,
the request may not represent the intended instant, or a broker server-clock
terminal may lack the correct broker timezone configuration.

**Solution:** Pin presentation to UTC and check the payload's `timezone` field:
```ini
CLIENT_TZ=UTC
```

Request the `metadata` extra and inspect `timestamp_mode`. Native terminals use
UTC epochs directly. If `server_clock` is detected, configure `MT5_SERVER_TZ`
(preferred) or `MT5_TIME_OFFSET_MINUTES`; mtdata then normalizes at the adapter
boundary. Never manually shift an already-normalized payload. See
[TIMESTAMPS.md](TIMESTAMPS.md).

If an error says that a live quote tracks the current minute but is in the
future by whole hours, `.env` was not loaded or the broker timezone is wrong. Set
`MT5_SERVER_TZ` to the broker's IANA timezone, then restart the CLI, MCP server,
or Web API process. A running process does not adopt edited timezone settings.

### Volume is Always Zero

**Cause:** Forex spot typically has indicative volume (tick count, not real volume).

**Note:** This is expected for most forex pairs. Use volume-based indicators cautiously.

### Time Range Looks Valid but Still Returns No Rows

**Possible causes:**
- The market was closed for the requested interval.
- Your broker has limited historical depth for that symbol/timeframe.
- The symbol name differs by broker suffix, for example `EURUSD.r` or `EURUSDm`.

**Solution:**
```bash
mtdata-cli symbols_list --search-term EURUSD --limit 20
mtdata-cli data_fetch_candles EURUSD --timeframe H1 --limit 100 --json
```

If the broker uses a suffix, rerun the command with the exact symbol shown by `symbols_list`.

---

## Performance Issues

### Command is Slow

**Possible causes:**
- Large `--lookback` or `--limit`
- Complex model (Chronos, GARCH)
- First run of pre-trained model (downloading weights)

**Solutions:**
1. Reduce data size: `--limit 500` instead of `--limit 5000`
2. Use faster methods: `theta` instead of `chronos2`
3. Reduce simulation count: `--params "n_sims=1000"` instead of 5000

### Memory Error

**Cause:** Large data or too many simulations.

**Solution:**
- Reduce `--limit`
- Reduce `n_sims` for barrier analysis
- Use streaming instead of batch when possible

---

## Trading Safety Issues

### A Trading Command Might Execute Live

**Cause:** `trade_*` commands operate on the MT5 account currently logged into the terminal. `trade_place`, `trade_modify`, and `trade_close` default to preview mode, but live execution is possible when you set `--dry-run false`.

**Immediate checks:**
```bash
mtdata-cli trade_account_info --json
mtdata-cli trade_get_open --json
mtdata-cli trade_get_pending --json
```

**Safer next steps:**
1. Confirm whether the MT5 terminal is logged into demo or live.
2. Use exact tickets for `trade_close` and `trade_modify`.
3. Preview supported actions with `--dry-run true`.
4. Configure guardrails such as `MTDATA_TRADING_ENABLED=0`, `MTDATA_TRADE_ALLOWED_SYMBOLS`, and `MTDATA_TRADE_MAX_RISK_PCT_OF_EQUITY` in [ENV_VARS.md](ENV_VARS.md#trade-guardrails).

---

## Getting Help

### Search Commands by Topic
```bash
mtdata-cli --help forecast
mtdata-cli --help barrier
mtdata-cli --help indicators
```

### Get Command-Specific Help
```bash
mtdata-cli forecast_generate --help
mtdata-cli regime_detect --help
```

### Enable Debug Mode
PowerShell:
```powershell
$env:MTDATA_CLI_DEBUG = "1"
mtdata-cli forecast_generate EURUSD --horizon 12
$env:MTDATA_CLI_DEBUG = $null
```

Bash:
```bash
MTDATA_CLI_DEBUG=1 mtdata-cli forecast_generate EURUSD --horizon 12
```

---

## Optional Dependency Issues

### Manual stock-pattern Extra Fails to Install

**Symptom:** `stock-pattern` cannot be installed via pip because that repository has no packaging metadata.

**Solution:**
1. Make sure Git is installed and available on `PATH`.
2. Install the stable base stack first:
   ```bash
   pip install -r requirements.txt
   ```
3. Follow the manual copy steps in [SETUP.md](SETUP.md#6-manual-stock-pattern-installation).

If that still fails, leave it out and use the rest of mtdata without that integration. CNBC news (`ycnbc`), TimesFM, `chronos2`, and `chronos_bolt` are available from package-index extras.

### Optional hnswlib Source Build Fails

**Symptom:** `pip install -r requirements-optional-src.txt` fails while building `hnswlib` on Python 3.14.

**What this path is for:**
- `hnswlib`: optional accelerator for `search_engine=hnsw`

**Requirements by package:**
- Visual Studio Build Tools 2022 with the **Desktop development with C++** workload

**Recommended recovery steps:**
```bash
pip install hnswlib==0.8.0
```

If `hnswlib` fails, leave it out and use `search_engine=ckdtree` instead.

`tsdownsample` is no longer part of this source-build helper. It is included in the full package-index install path and can also be installed directly with `pip install "tsdownsample>=0.1.5.1"`. If it is absent, mtdata falls back to the built-in Python simplification path.

### QuantLib Import Error

**Symptom:** `ModuleNotFoundError: No module named 'QuantLib'` when using `options_barrier_price` or `options_heston_calibrate`.

**Solution:**
```bash
pip install QuantLib
```
On Windows, a pre-built wheel is available. On Linux, you may need build tools (`cmake`, `swig`).

### Finviz Errors

**Symptom:** `news`, `calendar`, `equity_profile`, `screener`, or `asset_performance` return empty Finviz-backed data or connection errors.

**Possible causes:**
- Network/firewall blocking finviz.com
- Rate limiting (finviz throttles rapid requests)

**Note:** `finvizfinance` is a core dependency — it is installed automatically with `pip install -e .`. If you see an import error, your install may be incomplete; reinstall with `pip install -e .` or `pip install -r requirements.txt`. Finviz data is US equities only and delayed 15–20 minutes.

### Optuna Not Available

**Symptom:** `forecast_tune_optuna` fails with import error.

**Solution:**
```bash
pip install optuna
```

### Neural Forecast Models Unavailable

**Symptom:** NHiTS, TFT, PatchTST, or NBEATSx show `available: false` in `forecast_list_methods`.

**Solution:**
```bash
pip install neuralforecast torch
```
These models require PyTorch and a NeuralForecast dependency stack compatible with your Python version. On Windows Python 3.14, install fails because `ray` (required by NeuralForecast) does not publish Windows wheels for Python 3.14.

---

## Quick Fixes

| Issue | Quick Fix |
|-------|-----------|
| MT5 not connecting | Restart MT5 terminal, ensure login |
| Missing parameter | Check `--help` for required arguments |
| Invalid timeframe | Use a supported MT5 interval listed in the Timeframe section above |
| Method not available | Check `forecast_list_methods` and install deps |
| Output hard to read | Add `--json` |
| Wrong timestamps | Check the payload `timezone`, set `CLIENT_TZ=UTC`, and verify that request inputs identify the intended absolute instant |
| Command slow | Reduce `--limit`, use faster method |
| QuantLib import error | `pip install QuantLib` |
| Finviz empty data | Check network / firewall (finvizfinance is pre-installed) |
| Optuna not found | `pip install optuna` |
| Neural models unavailable | Use `forecast_list_methods --json`; NeuralForecast needs `ray` Windows cp314 wheels (not published yet) |

---

## See also

- [SETUP.md](SETUP.md) — Install and first connection
- [CLI.md](CLI.md) — Commands and output
- [LIMITATIONS.md](LIMITATIONS.md) — Known caveats
- [TIMESTAMPS.md](TIMESTAMPS.md) — Shifted candle times
- [TRADING_SAFETY.md](TRADING_SAFETY.md) — Live order concerns
