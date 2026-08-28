"""Public-contract tests for HAR-RV quality controls."""

from unittest.mock import patch

import pytest

from mtdata.forecast import volatility as vol


def test_har_quality_parameters_are_listed_in_public_method_metadata() -> None:
    har = next(
        item
        for item in vol.get_volatility_methods_data()["methods"]
        if item["method"] == "har_rv"
    )
    params = {item["name"]: item for item in har["params"]}

    assert params["minimum_daily_coverage_fraction"]["default"] == 0.9
    assert params["maximum_missing_bars_per_gap"]["default"] == 12


@pytest.mark.parametrize(
    ("params", "message"),
    [
        (
            {"minimum_daily_coverage_fraction": 0},
            "minimum_daily_coverage_fraction",
        ),
        (
            {"maximum_missing_bars_per_gap": 1.5},
            "maximum_missing_bars_per_gap",
        ),
    ],
)
def test_har_quality_parameters_fail_before_history_fetch(
    params: dict,
    message: str,
) -> None:
    with patch.object(vol, "_fetch_mt5_rates_guarded") as fetch:
        result = vol.forecast_volatility(
            "BTCUSD",
            "H1",
            method="har_rv",
            params=params,
        )

    assert message in result["error"]
    fetch.assert_not_called()
