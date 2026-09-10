from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_market_data_hotpath_benchmark_smoke() -> None:
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_market_data_hotpaths.py",
            "--samples",
            "1",
            "--candle-rows",
            "8",
            "--scan-symbols",
            "3",
            "--ticks-per-range",
            "4",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )

    workloads = json.loads(completed.stdout)["workloads"]
    cold_import = workloads["symbols_scan_cold_import"]
    candle_shaping = workloads["candle_response_shaping"]
    quote_resolution = workloads["market_scan_quote_resolution"]

    import_sample = cold_import["samples"][0]
    assert {
        key: import_sample[key]
        for key in ("pyemd_loaded", "matplotlib_loaded", "candles_loaded")
    } == {
        "pyemd_loaded": False,
        "matplotlib_loaded": False,
        "candles_loaded": False,
    }
    assert import_sample["module_count"] > 0
    assert cold_import["median_seconds"] > 0

    assert candle_shaping["implementation"] == "columnar_single_time_pass"
    assert candle_shaping["rows"] == 8
    assert candle_shaping["payload_sha256"] == (
        "06d8ac49363becc86122fe6cf9c47fe677193b2a3f89d0a4772a2dc28383f3e9"
    )

    assert quote_resolution["symbols"] == 3
    assert quote_resolution["copy_ticks_range_calls"] == 0
    assert quote_resolution["quote_source"] == "mt5.symbol_info_tick"
