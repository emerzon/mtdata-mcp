"""Tests for the candlestick technical-analysis lazy-load guard."""

from mtdata.patterns import candlestick as candlestick_mod


class TestEnsureCandlestickRuntime:
    def test_preserves_existing_ta_module(self, monkeypatch):
        monkeypatch.setattr(candlestick_mod, "ta", "ta")

        candlestick_mod._ensure_candlestick_runtime()

        assert candlestick_mod.ta == "ta"
