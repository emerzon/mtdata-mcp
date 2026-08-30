from datetime import datetime, timezone

from mtdata.core.trading.requests import TradeHistoryRequest
from mtdata.core.trading.use_cases.history import (
    _decode_trade_history_cursor,
    _encode_trade_history_cursor,
    _trade_history_cursor_scope,
)
from mtdata.utils.continuation import (
    decode_continuation_cursor,
    encode_continuation_cursor,
)


def _bounds() -> tuple[datetime, datetime]:
    return (
        datetime(2026, 8, 1, tzinfo=timezone.utc),
        datetime(2026, 8, 29, tzinfo=timezone.utc),
    )


def test_trade_history_cursor_v2_uses_short_keys_and_omits_null_scope() -> None:
    request = TradeHistoryRequest(
        history_kind="deals",
        symbol="EURUSD",
        magic=42,
    )
    from_dt, to_dt = _bounds()

    cursor = _encode_trade_history_cursor(
        request,
        from_dt=from_dt,
        to_dt=to_dt,
        last_milliseconds=1_788_000_000_123.0,
        last_ticket=12_345_678,
        position=20,
        issued_at=1_788_000_000,
    )
    raw = decode_continuation_cursor(
        cursor,
        invalid_message="invalid",
        unsupported_version_message="unsupported",
        expected_versions={1, 2},
    )

    assert raw["v"] == 2
    assert raw["s"] == {"g": "42", "h": "deals", "o": "desc", "y": "EURUSD"}
    assert set(raw) == {"v", "s", "f", "t", "m", "k", "p", "i"}
    assert len(cursor) <= 190


def test_trade_history_cursor_v2_round_trips_and_v1_remains_valid() -> None:
    request = TradeHistoryRequest(
        history_kind="orders",
        start="2026-08-01",
        end="2026-08-29",
        side="sell",
        order_ticket=123,
        order="asc",
    )
    from_dt, to_dt = _bounds()
    issued_at = int(datetime.now(tz=timezone.utc).timestamp())
    state = {
        "from_dt": from_dt,
        "to_dt": to_dt,
        "last_milliseconds": 1_788_000_000_123.0,
        "last_ticket": 123,
        "position": 40,
        "issued_at": issued_at,
    }

    v2_cursor = _encode_trade_history_cursor(
        request,
        from_dt=from_dt,
        to_dt=to_dt,
        last_milliseconds=state["last_milliseconds"],
        last_ticket=state["last_ticket"],
        position=state["position"],
        issued_at=issued_at,
    )
    legacy_cursor = encode_continuation_cursor(
        {
            "v": 1,
            "scope": _trade_history_cursor_scope(request),
            "from": from_dt.isoformat(timespec="microseconds"),
            "to": to_dt.isoformat(timespec="microseconds"),
            "last_milliseconds": repr(state["last_milliseconds"]),
            "last_ticket": str(state["last_ticket"]),
            "position": state["position"],
            "issued_at": issued_at,
        }
    )

    assert _decode_trade_history_cursor(v2_cursor, request) == state
    assert _decode_trade_history_cursor(legacy_cursor, request) == state
