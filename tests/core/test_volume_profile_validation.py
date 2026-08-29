from __future__ import annotations

from types import SimpleNamespace

from mtdata.core import volume_profile as volume_profile_module


class _Guard:
    def __init__(self, err, info):
        self.err = err
        self.info = info

    def __enter__(self):
        return self.err, self.info

    def __exit__(self, *args):
        return False


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

    assert result["error_code"] == "volume_profile_conflicting_window_selectors"
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


def test_volume_profile_rejects_tick_count_for_m1_bars_before_io(monkeypatch) -> None:
    monkeypatch.setattr(
        volume_profile_module,
        "create_mt5_gateway",
        lambda **_: (_ for _ in ()).throw(AssertionError("gateway must not run")),
    )

    result = volume_profile_module.compute_volume_profile_payload(
        symbol="EURUSD",
        start="2026-08-01",
        end="2026-08-01",
        source="m1_bars",
        volume_source="tick_count",
    )

    assert result["success"] is False
    assert result["error_code"] == "volume_profile_tick_count_unavailable_for_m1_bars"
    assert result["parameter"] == "volume_source"
    assert "tick_volume" in result["remediation"]
    assert "source=ticks" in result["remediation"]


def test_volume_profile_rejects_tick_count_when_auto_selects_m1_bars(monkeypatch) -> None:
    monkeypatch.setattr(
        volume_profile_module,
        "create_mt5_gateway",
        lambda **_: SimpleNamespace(ensure_connection=lambda: None),
    )
    monkeypatch.setattr(
        volume_profile_module,
        "resolve_public_symbol",
        lambda symbol, gateway=None: (symbol, None),
    )
    monkeypatch.setattr(
        volume_profile_module,
        "_symbol_ready_guard",
        lambda symbol: _Guard(None, SimpleNamespace(point=0.0001, digits=5)),
    )
    monkeypatch.setattr(
        volume_profile_module,
        "_select_profile_rows",
        lambda **_: {"success": True, "source": "m1_bars", "rows": []},
    )

    result = volume_profile_module.compute_volume_profile_payload(
        symbol="EURUSD",
        start="2026-08-01",
        end="2026-08-01",
        source="auto",
        volume_source="tick_count",
    )

    assert result["error_code"] == "volume_profile_tick_count_unavailable_for_m1_bars"
    assert result["source"] == "m1_bars"


def test_volume_profile_rejects_unbounded_max_ticks() -> None:
    result = volume_profile_module.compute_volume_profile_payload(
        symbol="EURUSD",
        max_ticks=1_000_000_000,
    )

    assert result["error_code"] == "volume_profile_limit_exceeded"
    assert result["parameter"] == "max_ticks"
    assert result["maximum"] == volume_profile_module._MAX_TICKS
