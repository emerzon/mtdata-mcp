from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mtdata.core import diagnostics


@pytest.mark.parametrize("symbol", ["eurusd", "EUR/USD", "EUR.USD", "EURUSD.a"])
def test_diagnostics_resolve_symbols_before_fetching_and_preserve_identity(symbol):
    expected = "EURUSD.a" if symbol.endswith(".a") else "EURUSD"
    with (
        patch("mtdata.utils.mt5.mt5.symbols_get", return_value=[SimpleNamespace(name="EURUSD"), SimpleNamespace(name="EURUSD.a")]),
        patch.object(diagnostics, "_ensure_symbol_ready", return_value=None) as ready,
        patch.object(diagnostics, "_mt5_copy_rates_from", return_value=[{"time": 0, "close": 1.0}, {"time": 3600, "close": 1.1}]) as fetch,
    ):
        frame, error = diagnostics._fetch_diagnostic_bars(symbol, "H1", 2)
    assert error is None
    ready.assert_called_once_with(expected)
    assert fetch.call_args.args[0] == expected
    metadata = diagnostics._diagnostic_history_metadata(frame, include_incomplete=False)
    assert metadata["symbol"] == expected
    assert metadata.get("symbol_input") == (symbol if symbol != expected else None)


def test_diagnostics_preserve_unresolved_symbol_error():
    with (
        patch.object(diagnostics, "resolve_public_symbol", return_value=("MISSING", None)),
        patch.object(diagnostics, "_ensure_symbol_ready", return_value="Symbol MISSING not found"),
        patch.object(diagnostics, "_mt5_copy_rates_from") as fetch,
    ):
        _, error = diagnostics._fetch_diagnostic_bars("MISSING", "H1", 2)
    assert error == "Symbol MISSING not found"
    fetch.assert_not_called()
