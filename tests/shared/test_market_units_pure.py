from mtdata.shared.market_units import (
    forex_pip_size,
    forex_points_per_pip,
    price_delta_ticks,
    snap_to_increment,
)


def test_snap_to_increment_formats_integer_tick_product() -> None:
    assert snap_to_increment(1.10001, 0.00001, digits=5) == 1.10001
    assert snap_to_increment(2654.56, 0.01, digits=2) == 2654.56


def test_price_delta_ticks_removes_subtraction_residue() -> None:
    assert price_delta_ticks(1.1, 1.095, 0.00001) == 500


def test_forex_pip_helpers_tolerate_computed_point_values() -> None:
    computed_point = 0.00001 * (1.0 + 1e-12)

    assert forex_points_per_pip(
        "EURUSD",
        point=computed_point,
        digits=0,
    ) == 10.0
    assert forex_pip_size("EURUSD", point=0.00001, digits=5) == 0.0001


def test_forex_pip_helpers_do_not_assign_pips_to_cfds() -> None:
    assert forex_points_per_pip("XAUUSD", point=0.01, digits=2) is None
    assert forex_pip_size("US500", point=0.01, digits=2) is None


def test_forex_pip_helpers_accept_broker_prefixed_pairs() -> None:
    assert forex_points_per_pip("mEURUSD", point=0.00001, digits=5) == 10.0
    assert forex_points_per_pip("FX_EURUSD", point=0.00001, digits=5) == 10.0
    assert forex_pip_size("USDZAR.pro", point=0.00001, digits=5) == 0.0001
