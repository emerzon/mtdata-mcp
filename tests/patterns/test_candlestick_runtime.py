"""Tests for candlestick lazy-load guards."""

from mtdata.patterns import candlestick as candlestick_mod


class TestEnsureCandlestickRuntime:
    def test_preserves_existing_globals(self, monkeypatch):
        monkeypatch.setattr(candlestick_mod, "ta", "ta")
        monkeypatch.setattr(candlestick_mod, "TIMEFRAME_MAP", {"H1": 1})

        candlestick_mod._ensure_candlestick_runtime()

        assert candlestick_mod.ta == "ta"
        assert candlestick_mod.TIMEFRAME_MAP == {"H1": 1}
