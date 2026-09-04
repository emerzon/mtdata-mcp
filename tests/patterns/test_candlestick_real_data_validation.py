from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Optional, Tuple

import MetaTrader5 as mt5
import pandas as pd
import pytest

from mtdata.patterns import candlestick as candlestick_mod
from mtdata.shared.constants import TIMEFRAME_MAP
from mtdata.utils.mt5 import _mt5_copy_rates_from, _rates_to_df


def _manual_pattern_sign(
    pattern: str,
    i: int,
    o: pd.Series,
    h: pd.Series,
    l: pd.Series,
    c: pd.Series,
) -> int:
    if i <= 0:
        return 0
    op, hp, lp, cp = float(o.iloc[i - 1]), float(h.iloc[i - 1]), float(l.iloc[i - 1]), float(c.iloc[i - 1])
    oc, hc, lc, cc = float(o.iloc[i]), float(h.iloc[i]), float(l.iloc[i]), float(c.iloc[i])
    nm = str(pattern).strip().lower()

    if nm == "inside":
        if hc <= hp and lc >= lp:
            return 1
        return 0

    if nm == "outside":
        if hc >= hp and lc <= lp:
            return 1
        return 0

    if nm == "engulfing":
        return 1 if cc >= oc else -1

    if nm == "harami":
        return 1 if cp <= op else -1

    return 0


def _pick_symbol_with_data(
    *,
    timeframe: str,
    count: int,
    candidates: Iterable[str],
) -> Tuple[Optional[str], Optional[object]]:
    tf = TIMEFRAME_MAP[timeframe]
    for symbol in candidates:
        try:
            mt5.symbol_select(symbol, True)
            rates = _mt5_copy_rates_from(symbol, tf, datetime.now(timezone.utc), count)
            if rates is not None and len(rates) >= 200:
                return symbol, rates
        except Exception:
            continue
    return None, None


def test_candlestick_patterns_are_present_on_real_data(monkeypatch):
    if not mt5.initialize():
        pytest.skip("MT5 terminal not available for real-data validation")

    symbol, rates = _pick_symbol_with_data(
        timeframe="H1",
        count=1500,
        candidates=("EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD"),
    )
    if not symbol or rates is None:
        pytest.skip("No symbol with sufficient real MT5 candle data was available")

    limit = min(1000, len(rates))
    rates_slice = rates[-limit:]

    monkeypatch.setattr(
        "mtdata.services.data_service.candles.fetch_history_frame",
        lambda *_a, **_k: _rates_to_df(rates_slice),
    )
    monkeypatch.setattr(candlestick_mod, "_use_client_tz", lambda: False)

    res = candlestick_mod.detect_candlestick_patterns(
        symbol=symbol,
        timeframe="H1",
        limit=limit,
        min_strength=0.95,
        min_gap=0,
        robust_only=False,
        whitelist="engulfing,harami,inside,outside",
        top_k=4,
    )
    assert "error" not in res, f"candlestick detection error: {res.get('error')}"

    df = _rates_to_df(rates_slice)
    assert len(df) >= 200

    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    idx_by_time = {
        datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M"): i
        for i, ts in enumerate(df["time"].tolist())
    }

    rows = res.get("data", [])
    assert rows, "No patterns found in snapshot; validation did not exercise pattern checks"

    failures = []
    for row in rows:
        ts = str(row.get("time", ""))
        label = str(row.get("pattern", "")).strip()
        idx = idx_by_time.get(ts)
        if idx is None:
            failures.append(f"time not in source bars: {ts}")
            continue

        parts = label.split(" ", 1)
        if len(parts) != 2:
            failures.append(f"unexpected pattern label format: {label}")
            continue
        side, name = parts[0].lower(), parts[1].strip().lower()
        expected_side = 1 if side == "bullish" else -1 if side == "bearish" else 0
        if expected_side == 0:
            failures.append(f"unexpected direction in label: {label}")
            continue

        manual_side = _manual_pattern_sign(name, idx, o, h, l, c)
        if manual_side != expected_side:
            failures.append(
                f"{ts} {label}: manual={manual_side}, detector={expected_side}"
            )

    assert not failures, "Detected patterns failed real-ohlc validation:\n" + "\n".join(failures[:10])

    mt5.shutdown()
