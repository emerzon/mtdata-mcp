"""Tests for candlestick lazy-load guards."""

from mtdata.patterns import candlestick as candlestick_mod


class TestEnsureCandlestickRuntime:
    def test_preserves_existing_globals(self, monkeypatch):
        monkeypatch.setattr(candlestick_mod, "ta", "ta")
        monkeypatch.setattr(candlestick_mod, "mt5", "mt5")
        monkeypatch.setattr(candlestick_mod, "TIMEFRAME_MAP", {"H1": 1})
        monkeypatch.setattr(candlestick_mod, "_mt5_copy_rates_from", "from")
        monkeypatch.setattr(candlestick_mod, "_mt5_copy_rates_range", "range")
        monkeypatch.setattr(candlestick_mod, "_rates_to_df", "df")
        monkeypatch.setattr(candlestick_mod, "_symbol_ready_guard", "guard")

        candlestick_mod._ensure_candlestick_runtime()

        assert candlestick_mod.ta == "ta"
        assert candlestick_mod.mt5 == "mt5"
        assert candlestick_mod.TIMEFRAME_MAP == {"H1": 1}
        assert candlestick_mod._mt5_copy_rates_from == "from"
        assert candlestick_mod._mt5_copy_rates_range == "range"
        assert candlestick_mod._rates_to_df == "df"
        assert candlestick_mod._symbol_ready_guard == "guard"
