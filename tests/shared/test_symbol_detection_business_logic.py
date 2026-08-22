from __future__ import annotations

from mtdata.forecast.common import bars_per_year
from mtdata.shared.symbols import (
        CRYPTO_SYMBOL_HINTS,
        is_probably_crypto_symbol,
        is_probably_forex_symbol,
        is_probably_fx_session_symbol,
)


def test_shared_crypto_symbol_hints_include_extended_tokens() -> None:
        assert {"BNB", "TRX", "NEAR", "FIL", "XLM", "USDC", "USDT"}.issubset(
            set(CRYPTO_SYMBOL_HINTS)
        )


def test_crypto_symbol_detection_stays_consistent_across_modules() -> None:
    for symbol in (
        "BNBUSDT",
        "TRXUSD",
        "NEARUSD",
        "FILUSD",
        "MATICUSD",
        "ATOMUSD",
        "XLMUSD",
        "USDCUSD",
    ):
        assert is_probably_crypto_symbol(symbol) is True

    for symbol in (
        "EURUSD",
        "SOLV",
        "ATOM",
        "UNIT",
        "LINKEDIN",
        "",
        None,
    ):
        assert is_probably_crypto_symbol(symbol) is False


def test_forex_detection_covers_extended_codes_and_broker_prefixes() -> None:
    for symbol in (
        "USDSGD",
        "USDZAR.pro",
        "FX_EURUSD",
        "mEURUSD",
        "broker_EURNOK.a",
    ):
        assert is_probably_forex_symbol(symbol) is True

    for symbol in ("BTCUSD", "SOLV", "NASDAQ", "", None):
        assert is_probably_forex_symbol(symbol) is False

    assert bars_per_year("H1", "USDSGD") == 260.0 * 24.0


def test_fx_session_detection_uses_broker_asset_paths() -> None:
    assert is_probably_fx_session_symbol("EURUSD") is True
    assert is_probably_fx_session_symbol("XAUUSD") is True
    assert is_probably_fx_session_symbol("US30", path="CFD\\Indices") is True
    assert is_probably_fx_session_symbol("AAPL", path="Stocks\\USA") is False
    assert is_probably_fx_session_symbol("BTCUSD", path="Crypto") is False
