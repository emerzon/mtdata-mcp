from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from mtdata.core.trading.execution import _close_positions_dry_run_preview


def test_close_preview_discloses_stale_market_readiness() -> None:
    position = SimpleNamespace(
        ticket=7,
        symbol="EURUSD",
        type=0,
        volume=0.5,
        profit=12.0,
        price_open=1.09,
        price_current=1.10,
    )
    gateway = SimpleNamespace(
        symbol_info_tick=lambda symbol: SimpleNamespace(
            bid=1.0999,
            ask=1.1001,
            time=1,
        )
    )

    result = _close_positions_dry_run_preview(
        [position],
        symbol="EURUSD",
        magic=None,
        profit_only=False,
        loss_only=False,
        close_priority=None,
        mt5=gateway,
    )

    assert result["success"] is False
    assert result["error_code"] == "preview_blocked"
    assert "live-ready" in result["error"].lower()
    assert result["preview_ok"] is False
    assert result["market_readiness"] == {
        "symbols_checked": 1,
        "usable_for_live_trading": False,
        "stale_or_unverified": 1,
    }
    assert result["matched_positions"][0]["quote_context"]["data_stale"] is True
    assert result["matched_positions"][0]["side"] == "BUY"
    assert "type" not in result["matched_positions"][0]


def test_close_preview_uses_reconciled_stream_quote() -> None:
    now = 1_700_000_000.0
    position = SimpleNamespace(
        ticket=7,
        symbol="EURUSD",
        type=0,
        volume=0.5,
        profit=12.0,
        price_open=1.09,
        price_current=1.10,
    )
    gateway = SimpleNamespace(
        POSITION_TYPE_BUY=0,
        ORDER_TYPE_BUY=0,
        COPY_TICKS_ALL=0,
        symbol_info_tick=lambda _symbol: SimpleNamespace(
            bid=1.0999,
            ask=1.1001,
            time=now + 30,
        ),
        copy_ticks_range=lambda *_args: [
            {
                "bid": 1.0999,
                "ask": 1.1001,
                "time": now - 1,
                "time_msc": int((now - 1) * 1000),
            }
        ],
        symbol_info=lambda _symbol: SimpleNamespace(point=0.0001),
    )

    with patch(
        "mtdata.core.trading.execution._stdlib_time.time",
        return_value=now,
    ):
        result = _close_positions_dry_run_preview(
            [position],
            symbol="EURUSD",
            magic=None,
            profit_only=False,
            loss_only=False,
            close_priority=None,
            mt5=gateway,
        )

    context = result["matched_positions"][0]["quote_context"]
    assert result["success"] is True
    assert result["preview_ok"] is True
    assert context["quote_source"] == "mt5.copy_ticks_range"
