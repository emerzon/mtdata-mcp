"""Deterministic microbenchmarks for market-data import and shaping hot paths."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable


def _percentile_median(values: list[float]) -> float:
    return float(statistics.median(values))


def _revision(source_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(source_root.parent), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _cold_import_benchmark(
    source_root: Path,
    *,
    samples: int,
) -> dict[str, Any]:
    program = """
import json
import sys
import time

started = time.perf_counter()
import mtdata.core.symbols.scan
elapsed = time.perf_counter() - started
modules = set(sys.modules)
print(json.dumps({
    "seconds": elapsed,
    "module_count": len(modules),
    "pyemd_loaded": any(
        name == "PyEMD" or name.startswith("PyEMD.")
        for name in modules
    ),
    "matplotlib_loaded": any(
        name == "matplotlib" or name.startswith("matplotlib.")
        for name in modules
    ),
    "candles_loaded": "mtdata.services.data_service.candles" in modules,
}))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            (str(source_root), environment.get("PYTHONPATH")),
        )
    )
    observations = []
    for _ in range(samples):
        completed = subprocess.run(
            [sys.executable, "-c", program],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        observations.append(
            json.loads(completed.stdout.strip().splitlines()[-1])
        )
    seconds = [float(item["seconds"]) for item in observations]
    return {
        "samples": observations,
        "median_seconds": _percentile_median(seconds),
        "best_seconds": min(seconds),
    }


def _timed_samples(
    operation: Callable[[], Any],
    *,
    samples: int,
) -> tuple[list[float], Any]:
    result = operation()
    observations = []
    for _ in range(samples):
        gc.collect()
        gc.disable()
        try:
            started = time.perf_counter()
            result = operation()
            observations.append(time.perf_counter() - started)
        finally:
            gc.enable()
    return observations, result


def _payload_fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _candle_shaping_benchmark(
    *,
    rows: int,
    samples: int,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    from mtdata.services.data_service import candles

    base_epoch = 1_700_000_000
    positions = np.arange(rows)
    rates = pd.DataFrame(
        {
            "time": positions.astype(float) * 60.0 + base_epoch,
            "open": 1.1 + positions * 1e-7,
            "high": 1.1002 + positions * 1e-7,
            "low": 1.0998 + positions * 1e-7,
            "close": 1.1001 + positions * 1e-7,
            "tick_volume": positions.astype(np.int64) + 100,
            "spread": np.full(rows, 10, dtype=np.int64),
            "real_volume": np.zeros(rows, dtype=np.int64),
        }
    )
    headers = [
        "time",
        "open",
        "high",
        "low",
        "close",
        "tick_volume",
        "spread",
        "real_volume",
    ]
    price_columns = frozenset({"open", "high", "low", "close", "spread"})
    columnar_table = getattr(candles, "_candle_table_from_frame", None)

    if callable(columnar_table):
        implementation = "columnar_single_time_pass"

        def shape() -> dict[str, Any]:
            frame = candles._build_rates_df(
                rates.copy(),
                False,
                format_time=False,
            )
            candles._format_candle_times(
                frame,
                headers,
                time_as_epoch=False,
                use_client_tz=False,
                client_tz=None,
            )
            return columnar_table(
                frame,
                headers,
                digits=5,
                price_columns=price_columns,
            )

    else:
        from mtdata.utils.utils import (
            _format_numeric_rows_from_df,
            _table_from_rows,
        )

        implementation = "legacy_scalar_double_time_pass"

        def shape() -> dict[str, Any]:
            frame = candles._build_rates_df(rates.copy(), False)
            candles._format_candle_times(
                frame,
                headers,
                time_as_epoch=False,
                use_client_tz=False,
                client_tz=None,
            )
            public_rows = _format_numeric_rows_from_df(
                frame,
                headers,
                stringify=False,
            )
            public_rows = candles._round_row_price_columns(
                public_rows,
                headers,
                digits=5,
                price_columns=price_columns,
            )
            return _table_from_rows(headers, public_rows)

    observations, payload = _timed_samples(shape, samples=samples)
    return {
        "implementation": implementation,
        "rows": rows,
        "samples_seconds": observations,
        "median_seconds": _percentile_median(observations),
        "best_seconds": min(observations),
        "payload_sha256": _payload_fingerprint(payload),
        "first_row": payload["data"][0],
        "last_row": payload["data"][-1],
    }


def _scan_quote_benchmark(
    *,
    symbols: int,
    ticks_per_range: int,
    samples: int,
) -> dict[str, Any]:
    from mtdata.core.symbols import scan

    now_epoch = 1_789_052_400.0  # 2026-09-10T15:00:00Z, an open weekday.
    tape = [
        {
            "time": now_epoch - index * 0.001,
            "time_msc": int((now_epoch - index * 0.001) * 1000),
            "bid": 1.10001,
            "ask": 1.10003,
            "flags": 6,
        }
        for index in range(ticks_per_range)
    ]

    class Gateway:
        COPY_TICKS_ALL = 0

        def __init__(self) -> None:
            self.range_calls = 0

        @staticmethod
        def symbol_info_tick(_symbol: str) -> Any:
            return SimpleNamespace(
                time=now_epoch - 0.5,
                bid=1.10001,
                ask=1.10003,
            )

        def copy_ticks_range(self, *_args: Any) -> list[dict[str, Any]]:
            self.range_calls += 1
            return tape

        @staticmethod
        def symbol_info(_symbol: str) -> Any:
            return SimpleNamespace(point=0.00001)

        @staticmethod
        def last_error() -> None:
            return None

    symbol_rows = [
        SimpleNamespace(
            name=f"EURUSD{index}",
            path="Forex\\Majors",
            description="",
            digits=5,
            point=0.00001,
            trade_tick_size=0.00001,
            trade_tick_value=1.0,
            currency_profit="USD",
        )
        for index in range(symbols)
    ]

    def evaluate() -> dict[str, Any]:
        gateway = Gateway()
        original_time = scan.time.time
        scan.time.time = lambda: now_epoch
        try:
            results = [
                scan._build_market_scan_spread_row(symbol, gateway)
                for symbol in symbol_rows
            ]
        finally:
            scan.time.time = original_time
        if any(error is not None for _row, error in results):
            raise RuntimeError("Synthetic scan unexpectedly failed.")
        rows_out = [row for row, _error in results]
        return {
            "range_calls": gateway.range_calls,
            "rows": rows_out,
        }

    observations, payload = _timed_samples(evaluate, samples=samples)
    return {
        "symbols": symbols,
        "ticks_per_range": ticks_per_range,
        "samples_seconds": observations,
        "median_seconds": _percentile_median(observations),
        "best_seconds": min(observations),
        "copy_ticks_range_calls": payload["range_calls"],
        "quote_source": payload["rows"][0]["quote_source"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src",
    )
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--candle-rows", type=int, default=50_000)
    parser.add_argument("--scan-symbols", type=int, default=100)
    parser.add_argument("--ticks-per-range", type=int, default=1_000)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    samples = max(1, int(args.samples))
    sys.path.insert(0, str(source_root))
    result = {
        "source_root": str(source_root),
        "revision": _revision(source_root),
        "python": sys.version.split()[0],
        "workloads": {
            "symbols_scan_cold_import": _cold_import_benchmark(
                source_root,
                samples=samples,
            ),
            "candle_response_shaping": _candle_shaping_benchmark(
                rows=max(1, int(args.candle_rows)),
                samples=samples,
            ),
            "market_scan_quote_resolution": _scan_quote_benchmark(
                symbols=max(1, int(args.scan_symbols)),
                ticks_per_range=max(1, int(args.ticks_per_range)),
                samples=samples,
            ),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
