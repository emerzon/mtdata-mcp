import pytest

from mtdata.patterns.enrichment import (
    _config_bool,
    directional_regime_verdict,
    volume_confirmation_verdict,
)


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        (None, ("unavailable", 0.0)),
        (1.1, ("confirmed", 0.08)),
        (1.0 / 1.1, ("rejected", -0.06)),
        (1.0, ("neutral", 0.0)),
    ],
)
def test_volume_confirmation_verdict_boundaries(ratio, expected) -> None:
    assert volume_confirmation_verdict(
        ratio,
        min_ratio=1.1,
        bonus=0.08,
        penalty=0.06,
    ) == expected


@pytest.mark.parametrize(
    ("state", "direction", "expected"),
    [
        ("trending", "bullish", ("aligned", "aligned", 0.05)),
        ("trending", "bearish", ("countertrend", "countertrend", -0.04)),
        ("ranging", "bullish", ("context_only", "neutral", 0.0)),
    ],
)
def test_directional_regime_verdict(state, direction, expected) -> None:
    assert directional_regime_verdict(
        "bullish",
        state=state,
        regime_direction=direction,
        bonus=0.05,
        penalty=0.04,
    ) == expected


def test_config_bool_reuses_canonical_spellings_and_defaults() -> None:
    assert _config_bool({"enabled": "on"}, "enabled", False) is True
    assert _config_bool({"enabled": "off"}, "enabled", True) is False
    assert _config_bool({"enabled": "invalid"}, "enabled", True) is True
    assert _config_bool({"enabled": None}, "enabled", False) is False
