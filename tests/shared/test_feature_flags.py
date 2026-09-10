import pytest

from mtdata.shared.feature_flags import (
    MARKET_DEPTH_FETCH_ENV,
    MARKET_DEPTH_FETCH_FEATURE,
    feature_enable_env,
    feature_enabled,
)


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on"])
def test_market_depth_feature_accepts_canonical_true_values(value: str) -> None:
    assert feature_enabled(
        MARKET_DEPTH_FETCH_FEATURE,
        environ={MARKET_DEPTH_FETCH_ENV: value},
    )


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "enabled"])
def test_market_depth_feature_is_disabled_for_other_values(value: str) -> None:
    assert not feature_enabled(
        MARKET_DEPTH_FETCH_FEATURE,
        environ={MARKET_DEPTH_FETCH_ENV: value},
    )


def test_feature_enable_env_rejects_unknown_feature() -> None:
    with pytest.raises(ValueError, match="Unknown feature flag"):
        feature_enable_env("unknown")
