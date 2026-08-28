"""Tests for src/mtdata/utils/symbol.py"""
from types import SimpleNamespace

from mtdata.utils.symbol import (
    _extract_group_path,
    catalog_symbol_name_matches_query,
    match_symbol_infos,
    symbol_shorthand_rank,
)


class TestExtractGroupPath:
    def test_strips_symbol_from_path(self):
        sym = SimpleNamespace(path="Forex\\Major\\EURUSD", name="EURUSD")
        assert _extract_group_path(sym) == "Forex\\Major"

    def test_case_insensitive(self):
        sym = SimpleNamespace(path="Forex\\eurusd", name="EURUSD")
        assert _extract_group_path(sym) == "Forex"

    def test_no_trailing_symbol(self):
        sym = SimpleNamespace(path="Forex\\Major", name="GBPUSD")
        assert _extract_group_path(sym) == "Forex\\Major"

    def test_empty_path(self):
        sym = SimpleNamespace(path="", name="EURUSD")
        assert _extract_group_path(sym) == "Unknown"

    def test_none_path(self):
        sym = SimpleNamespace(path=None, name="EURUSD")
        assert _extract_group_path(sym) == "Unknown"

    def test_no_name(self):
        sym = SimpleNamespace(path="Forex\\Major", name="")
        assert _extract_group_path(sym) == "Forex\\Major"

    def test_single_component_matches_name(self):
        sym = SimpleNamespace(path="EURUSD", name="EURUSD")
        assert _extract_group_path(sym) == "Unknown"

    def test_strips_symbol_suffix_variant_from_path(self):
        sym = SimpleNamespace(path="Stock CFD's\\NYSE\\24HR NYSE\\WMT.NYSE-24", name="WMT")
        assert _extract_group_path(sym) == "Stock CFD's\\NYSE\\24HR NYSE"

    def test_missing_attributes(self):
        sym = object()
        assert _extract_group_path(sym) == "Unknown"


def test_match_symbol_infos_suggests_usd_crypto_pair_for_usdt_query():
    btc = SimpleNamespace(name="BTCUSD", description="Bitcoin", path="Crypto")
    eth = SimpleNamespace(name="ETHUSD", description="Ethereum", path="Crypto")

    assert match_symbol_infos([eth, btc], "BTCUSDT") == [btc]


def test_crypto_shorthand_prefers_base_quote_pair_over_stock_prefix():
    stock = SimpleNamespace(
        name="BTCT.NAS-24",
        description="BTC Digital Ltd",
        path="Stock CFD's\\Nasdaq",
    )
    crypto = SimpleNamespace(name="BTCUSD", description="Bitcoin", path="Crypto")

    assert symbol_shorthand_rank(crypto, "BTC") == 0
    assert symbol_shorthand_rank(stock, "BTC") == 1
    assert match_symbol_infos([stock, crypto], "BTC") == [crypto, stock]


def test_extended_crypto_shorthand_prefers_pair_over_stock_prefix():
    stock = SimpleNamespace(name="ATOM.NAS-24", description="Atomera", path="Stocks")
    crypto = SimpleNamespace(name="ATOMUSD", description="Cosmos", path="Crypto")

    assert match_symbol_infos([stock, crypto], "ATOM") == [crypto, stock]


def test_catalog_symbol_name_matches_suffix_typo():
    assert catalog_symbol_name_matches_query("EURUSD", "EURUSD1") is True
    assert catalog_symbol_name_matches_query("EURUSD", "EURUSD.a") is True
    assert catalog_symbol_name_matches_query("EURUSD", "EURJPY") is False
    assert catalog_symbol_name_matches_query("EURUSD", "EURUSP", fuzzy=True) is True
    assert catalog_symbol_name_matches_query("EURUSD", "EURUSP") is False


def test_match_symbol_infos_suggests_suffix_typo():
    eurusd = SimpleNamespace(name="EURUSD", description="Euro vs US Dollar", path="Forex\\Majors")
    eurusd_m = SimpleNamespace(name="EURUSDm", description="Euro vs US Dollar mini", path="Forex")
    gbpusd = SimpleNamespace(name="GBPUSD", description="Pound vs US Dollar", path="Forex\\Majors")

    matches = match_symbol_infos([gbpusd, eurusd, eurusd_m], "EURUSD1")
    assert [item.name for item in matches] == ["EURUSD", "EURUSDm"]
