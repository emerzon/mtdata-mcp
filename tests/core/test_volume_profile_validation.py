from __future__ import annotations

from mtdata.core import volume_profile as volume_profile_module


def test_volume_profile_rejects_mixed_window_modes_before_io(monkeypatch) -> None:
    monkeypatch.setattr(
        volume_profile_module,
        "create_mt5_gateway",
        lambda **_: (_ for _ in ()).throw(AssertionError("gateway must not run")),
    )

    result = volume_profile_module.compute_volume_profile_payload(
        symbol="EURUSD",
        start="2026-08-01",
        end="2026-08-10",
        timeframe="H1",
        lookback=100,
    )

    assert result["code"] == "volume_profile_conflicting_window_selectors"
    assert "one volume-profile window mode" in result["error"]


def test_volume_profile_rejects_reversed_range_before_io(monkeypatch) -> None:
    monkeypatch.setattr(
        volume_profile_module,
        "create_mt5_gateway",
        lambda **_: (_ for _ in ()).throw(AssertionError("gateway must not run")),
    )

    result = volume_profile_module.compute_volume_profile_payload(
        symbol="EURUSD",
        start="2026-08-20",
        end="2026-08-01",
    )

    assert result["success"] is False
    assert result["error_code"] == "invalid_date_range"
    assert result["error"] == "start must be before or equal to end."
