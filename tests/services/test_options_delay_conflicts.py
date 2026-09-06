import pytest

from mtdata.services.options_service import (
    _options_underlying_metadata,
    _underlying_quote_envelope,
)


@pytest.mark.parametrize("delay,source", [(0, "Delayed Quote"), (900, "Nasdaq Real Time Price")])
def test_conflicting_delay_indicators_do_not_claim_realtime(delay, source):
    result = _options_underlying_metadata("yahoo", {"exchangeDataDelayedBy": delay, "quoteSourceName": source})
    quote = result["underlying_quote"]
    assert quote["delay_status"] == "conflicting"
    assert quote.get("is_delayed") is None
    assert quote.get("delay_seconds") is None
    assert quote["provider_delay_indicators"]["exchangeDataDelayedBy"] == delay
    assert result["warnings"][0]["code"] == "underlying_quote_delay_conflict"


@pytest.mark.parametrize("delay,source,expected", [(0, "Nasdaq Real Time Price", False), (900, "Delayed Quote", True), (None, "Unknown", None), ("garbage", "Unknown", None), (-1, "Unknown", None)])
def test_consistent_or_missing_delay(delay, source, expected):
    quote = _underlying_quote_envelope("yahoo", {"exchangeDataDelayedBy": delay, "quoteSourceName": source})
    assert quote.get("is_delayed") is expected
    assert "delay_status" not in quote
