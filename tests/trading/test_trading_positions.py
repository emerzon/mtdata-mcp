from types import SimpleNamespace
from unittest.mock import MagicMock

from mtdata.core.trading import positions, validation


class _FakeMt5:
    ORDER_TYPE_BUY = 0
    POSITION_TYPE_BUY = 0

    def __init__(self, *, ticket_rows=None, fallback_rows=None):
        self._ticket_rows = ticket_rows or []
        self._fallback_rows = fallback_rows or []

    def positions_get(self, **kwargs):
        if "ticket" in kwargs:
            return list(self._ticket_rows)
        return list(self._fallback_rows)


def test_position_filter_uses_numeric_defaults_when_mt5_constants_are_missing():
    mt5 = SimpleNamespace()

    assert positions._position_matches_required_filters(
        SimpleNamespace(type=0), symbol=None, side="BUY", mt5=mt5
    )
    assert not positions._position_matches_required_filters(
        SimpleNamespace(type=1), symbol=None, side="BUY", mt5=mt5
    )
    assert positions._position_matches_required_filters(
        SimpleNamespace(type=1), symbol=None, side="SELL", mt5=mt5
    )


def test_position_side_resolution_accepts_text_without_mt5_constants():
    assert validation._resolve_position_side(SimpleNamespace(type="buy")) == "BUY"
    assert validation._resolve_position_side(SimpleNamespace(type="SELL_LIMIT")) == "SELL"
    assert validation._resolve_position_side(SimpleNamespace(type=0)) == "BUY"
    assert validation._resolve_position_side(SimpleNamespace(type=1)) == "SELL"


def test_ticket_resolution_uses_shared_mt5_field_priority():
    obj = SimpleNamespace(ticket=None, identifier="42", order=99, deal="bad")

    assert positions._ticket_fields(obj) == {"identifier": 42, "order": 99}
    assert positions._resolved_ticket(obj, fallback=7) == 42
    assert positions._resolved_ticket(SimpleNamespace(), fallback=7) == 7


def test_position_sort_key_normalizes_milliseconds_to_seconds():
    older = SimpleNamespace(time_update_msc=1_700_000_000_000)
    newer = SimpleNamespace(time_update=1_700_000_001)

    assert positions._position_sort_key(older) == 1_700_000_000
    assert max([older, newer], key=positions._position_sort_key) is newer


def test_normalize_trade_read_output_keeps_request_echoes_by_default():
    out = positions._normalize_trade_read_output(
        [{"ticket": 1, "symbol": "EURUSD"}],
        request=SimpleNamespace(symbol="EURUSD", limit=5),
        kind="open_positions",
    )

    assert out["symbol"] == "EURUSD"
    assert out["limit"] == 5
    assert out["scope"] == "symbol"
    assert out["count"] == 1
    assert out["row_key"] == "items"


def test_normalize_trade_read_output_compact_omits_request_echoes():
    out = positions._normalize_trade_read_output(
        [{"ticket": 1, "symbol": "EURUSD"}],
        request=SimpleNamespace(symbol="EURUSD", ticket=1, limit=5, detail="compact"),
        kind="open_positions",
    )

    assert out["count"] == 1
    assert "scope" not in out
    assert "symbol" not in out
    assert "ticket" not in out
    assert "limit" not in out


def test_normalize_trade_read_output_open_positions_includes_as_of():
    out = positions._normalize_trade_read_output(
        {"items": [{"ticket": 1, "symbol": "EURUSD"}]},
        request=SimpleNamespace(symbol="EURUSD", ticket=1, limit=5, detail="compact"),
        kind="open_positions",
    )
    assert isinstance(out.get("as_of"), str) and out["as_of"].endswith("Z")


def test_normalize_trade_read_output_error_omits_success_collection_fields():
    out = positions._normalize_trade_read_output(
        {
            "error": "Failed to get open positions",
            "error_code": "mt5_connection_error",
            "remediation": "Reconnect MT5.",
        },
        request=SimpleNamespace(symbol="EURUSD", detail="full"),
        kind="open_positions",
    )

    assert out == {
        "success": False,
        "error": "Failed to get open positions",
        "error_code": "mt5_connection_error",
        "remediation": "Reconnect MT5.",
    }
    assert not {"kind", "scope", "count", "items", "as_of", "symbol"} & set(out)


def test_normalize_trade_history_error_omits_success_collection_fields():
    out = positions._normalize_trade_read_output(
        {
            "error": "Could not parse historical datetime bound(s).",
            "error_code": "invalid_datetime",
            "remediation": "Correct the listed start/end value.",
        },
        request=SimpleNamespace(
            start="nonsense",
            end="now",
            detail="compact",
        ),
        kind="trade_history",
        account_currency="USD",
    )

    assert out["success"] is False
    assert out["error_code"] == "invalid_datetime"
    assert not {"kind", "scope", "count", "items", "row_key", "currency", "filters_applied"} & set(out)


def test_normalize_trade_read_output_unexpected_type_is_error_only():
    out = positions._normalize_trade_read_output(
        None,
        request=SimpleNamespace(detail="compact"),
        kind="open_positions",
    )

    assert out == {
        "success": False,
        "error": "Unexpected open_positions payload type: NoneType",
    }


def test_open_position_protection_summary_surfaces_missing_levels():
    out = {
        "success": True,
        "items": [
            {"ticket": 1, "sl": 0.0, "tp": 1.2},
            {"ticket": 2, "sl": None, "tp": 0.0},
            {"ticket": 3, "sl": 1.0, "tp": 1.3},
        ],
    }

    positions._attach_open_position_protection_summary(out)

    assert out["protection_summary"] == {
        "positions": 3,
        "positions_without_stop_loss": 2,
        "positions_without_take_profit": 1,
        "positions_missing_any_protection": 2,
        "fully_unprotected_positions": 1,
    }
    assert "2 open position(s)" in out["protection_warning"]


def test_normalize_trade_read_output_compact_empty_keeps_contract_shape():
    out = positions._normalize_trade_read_output(
        [],
        request=SimpleNamespace(detail="compact"),
        kind="open_positions",
    )

    as_of = out.pop("as_of", None)
    assert isinstance(as_of, str) and as_of.endswith("Z")
    assert out == {
        "success": True,
        "kind": "open_positions",
        "count": 0,
        "items": [],
        "row_key": "items",
        "empty": True,
        "message": "No open positions matched the request.",
        "hint": (
            "Normal when flat; relax symbol/ticket filters or check trade_account_info."
        ),
    }


def test_normalize_trade_read_output_adds_volume_units():
    out = positions._normalize_trade_read_output(
        [{"ticket": 1, "symbol": "EURUSD", "volume": 0.1}],
        request=SimpleNamespace(detail="compact"),
        kind="open_positions",
    )

    assert out["units"] == {"volume": "broker_lot"}


def test_normalize_trade_read_output_rounds_money_fields():
    out = positions._normalize_trade_read_output(
        [
            {
                "ticket": 1,
                "symbol": "BTCUSD",
                "profit": -1.8900000000000001,
                "swap": 0.07999999999999999,
            }
        ],
        request=SimpleNamespace(detail="compact"),
        kind="open_positions",
    )

    assert out["items"][0]["profit"] == -1.89
    assert out["items"][0]["swap"] == 0.08


def test_normalize_trade_read_output_trims_price_and_millisecond_artifacts():
    out = positions._normalize_trade_read_output(
        [
            {
                "ticket": 1,
                "symbol": "EURUSD",
                "price_open": 1.1627399999999999,
                "price_current": 1.1710099999999999,
                "sl": 1.1600000000000001,
                "tp": 1.1800000000000002,
                "time_msc": 1778822029181.0,
            }
        ],
        request=SimpleNamespace(detail="compact"),
        kind="open_positions",
    )

    row = out["items"][0]
    assert row["price_open"] == 1.16274
    assert row["price_current"] == 1.17101
    assert row["sl"] == 1.16
    assert row["tp"] == 1.18
    assert row["time_msc"] == 1778822029181


def test_normalize_trade_read_output_converts_unset_sl_tp_to_null():
    out = positions._normalize_trade_read_output(
        [{"ticket": 1, "symbol": "EURUSD", "sl": 0.0, "tp": 0}],
        request=SimpleNamespace(detail="compact"),
        kind="open_positions",
    )

    assert out["items"][0]["sl"] is None
    assert out["items"][0]["tp"] is None


def test_resolve_open_position_respects_side_filter_when_mt5_constants_are_missing():
    class _NoConstantsMt5:
        def __init__(self, rows):
            self._rows = rows

        def positions_get(self, **kwargs):
            return list(self._rows)

    rows = [
        SimpleNamespace(
            ticket=100,
            identifier=100,
            position_id=None,
            position=None,
            order=None,
            deal=None,
            symbol="EURUSD",
            type=0,
            volume=0.1,
            time_update_msc=1000,
        ),
        SimpleNamespace(
            ticket=200,
            identifier=200,
            position_id=None,
            position=None,
            order=None,
            deal=None,
            symbol="EURUSD",
            type=1,
            volume=0.1,
            time_update_msc=2000,
        ),
    ]

    pos, resolved_ticket, info = positions._resolve_open_position(
        _NoConstantsMt5(rows),
        symbol="EURUSD",
        side="BUY",
        volume=0.1,
    )

    assert pos is not None
    assert resolved_ticket == 100
    assert info["method"] == "positions_get(fallback_heuristic)"


def test_resolve_open_position_rejects_wrong_side_when_no_candidates_match():
    rows = [
        SimpleNamespace(
            ticket=200,
            identifier=200,
            position_id=None,
            position=None,
            order=None,
            deal=None,
            symbol="EURUSD",
            type=1,
            volume=0.1,
            time_update_msc=2000,
        ),
    ]

    pos, resolved_ticket, info = positions._resolve_open_position(
        _FakeMt5(fallback_rows=rows),
        symbol="EURUSD",
        side="BUY",
        volume=0.1,
    )

    assert pos is None
    assert resolved_ticket is None
    assert info["method"] == "positions_get(fallback_heuristic)"
    assert info["matched"] is False


def test_resolve_open_position_uses_candidate_ticket_when_direct_lookup_ticket_is_invalid():
    mt5 = _FakeMt5(
        ticket_rows=[
            SimpleNamespace(
                ticket=0,
                identifier=None,
                position_id=None,
                position=None,
                order=None,
                deal=None,
                symbol="EURUSD",
                type=0,
                volume=0.1,
                time_update_msc=1000,
            )
        ]
    )

    pos, resolved_ticket, info = positions._resolve_open_position(
        mt5,
        ticket_candidates=[456],
        symbol="EURUSD",
        side="BUY",
        volume=0.1,
    )

    assert pos is not None
    assert resolved_ticket == 456
    assert info["method"] == "positions_get(ticket)"


def test_resolve_open_position_uses_exact_match_value_when_ticket_field_is_invalid():
    mt5 = _FakeMt5(
        ticket_rows=[],
        fallback_rows=[
            SimpleNamespace(
                ticket=0,
                identifier=789,
                position_id=None,
                position=None,
                order=None,
                deal=None,
                symbol="EURUSD",
                type=0,
                volume=0.1,
                time_update_msc=1000,
            )
        ],
    )

    pos, resolved_ticket, info = positions._resolve_open_position(
        mt5,
        ticket_candidates=[789],
        symbol="EURUSD",
        side="BUY",
        volume=0.1,
    )

    assert pos is not None
    assert resolved_ticket == 789
    assert info["method"] == "positions_get(fallback_exact)"
    assert info["matched_field"] == "identifier"


def test_resolve_open_position_rejects_exact_match_with_symbol_mismatch():
    mt5 = _FakeMt5(
        ticket_rows=[],
        fallback_rows=[
            SimpleNamespace(
                ticket=789,
                identifier=789,
                position_id=None,
                position=None,
                order=None,
                deal=None,
                symbol="GBPUSD",
                type=0,
                volume=0.1,
                time_update_msc=1000,
            )
        ],
    )

    pos, resolved_ticket, info = positions._resolve_open_position(
        mt5,
        ticket_candidates=[789],
        symbol="EURUSD",
        side="BUY",
        volume=0.1,
    )

    assert pos is None
    assert resolved_ticket is None
    assert info["method"] == "positions_get(fallback_heuristic)"
    assert info["matched"] is False


def test_resolve_open_position_keeps_candidate_when_side_is_untyped():
    pos, resolved_ticket, info = positions._resolve_open_position(
        _FakeMt5(
            ticket_rows=[
                MagicMock(
                    symbol="EURUSD",
                    volume=0.1,
                    sl=0.0,
                    tp=0.0,
                    time_update_msc=1000,
                )
            ]
        ),
        ticket_candidates=[456],
        symbol="EURUSD",
        side="BUY",
        volume=0.1,
    )

    assert pos is not None
    assert resolved_ticket == 456
    assert info["method"] == "positions_get(ticket)"


def test_resolve_open_position_uses_other_ticket_like_fields_in_heuristic_path():
    mt5 = _FakeMt5(
        ticket_rows=[],
        fallback_rows=[
            SimpleNamespace(
                ticket=0,
                identifier=None,
                position_id=None,
                position=None,
                order=999,
                deal=None,
                symbol="EURUSD",
                type=0,
                volume=0.1,
                time_update_msc=1000,
            )
        ],
    )

    pos, resolved_ticket, info = positions._resolve_open_position(
        mt5,
        ticket_candidates=[],
        symbol="EURUSD",
        side="BUY",
        volume=0.1,
    )

    assert pos is not None
    assert resolved_ticket == 999
    assert info["method"] == "positions_get(fallback_heuristic)"


# ---------------------------------------------------------------------------
# Hedged-account tests (magic-based disambiguation)
# ---------------------------------------------------------------------------


class _HedgedFakeMt5:
    """Simulates a hedged account with multiple positions for the same symbol."""
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1

    def __init__(self, all_positions):
        self._all = all_positions

    def positions_get(self, **kwargs):
        if "ticket" in kwargs:
            return [p for p in self._all if p.ticket == kwargs["ticket"]] or None
        if "symbol" in kwargs:
            sym = kwargs["symbol"].upper()
            return [p for p in self._all if str(getattr(p, "symbol", "")).upper() == sym] or None
        return list(self._all)


def test_select_position_candidate_magic_disambiguates():
    """Magic number narrows candidates in hedged accounts."""
    positions_list = [
        SimpleNamespace(ticket=100, symbol="EURUSD", type=0, volume=0.1, magic=1000, time_update_msc=5000),
        SimpleNamespace(ticket=200, symbol="EURUSD", type=0, volume=0.1, magic=2000, time_update_msc=6000),
        SimpleNamespace(ticket=300, symbol="EURUSD", type=0, volume=0.1, magic=1000, time_update_msc=7000),
    ]
    mt5 = _HedgedFakeMt5(positions_list)

    # Without magic: picks most recent (ticket 300)
    picked = positions._select_position_candidate(
        positions_list, symbol="EURUSD", side="BUY", volume=0.1, mt5=mt5,
    )
    assert picked.ticket == 300

    # With magic=2000: narrows to ticket 200
    picked = positions._select_position_candidate(
        positions_list, symbol="EURUSD", side="BUY", volume=0.1, magic=2000, mt5=mt5,
    )
    assert picked.ticket == 200

    # With magic=1000: two match, picks most recent (ticket 300)
    picked = positions._select_position_candidate(
        positions_list, symbol="EURUSD", side="BUY", volume=0.1, magic=1000, mt5=mt5,
    )
    assert picked.ticket == 300


def test_select_position_candidate_uses_symbol_volume_step_tolerance():
    positions_list = [
        SimpleNamespace(ticket=100, symbol="XAUUSD", type=0, volume=0.3, magic=1000, time_update_msc=5000),
    ]
    mt5 = _HedgedFakeMt5(positions_list)
    mt5.symbol_info = lambda _symbol: SimpleNamespace(volume_step=0.1)

    picked = positions._select_position_candidate(
        positions_list,
        symbol="XAUUSD",
        side="BUY",
        volume=0.3000005,
        mt5=mt5,
    )

    assert picked.ticket == 100


def test_resolve_open_position_magic_filter_hedged():
    """_resolve_open_position uses magic to disambiguate hedged positions."""
    all_positions = [
        SimpleNamespace(ticket=100, identifier=100, position_id=None, position=None, order=None, deal=None,
                        symbol="EURUSD", type=0, volume=0.1, magic=1000, time_update_msc=5000),
        SimpleNamespace(ticket=200, identifier=200, position_id=None, position=None, order=None, deal=None,
                        symbol="EURUSD", type=0, volume=0.1, magic=2000, time_update_msc=6000),
    ]
    mt5 = _HedgedFakeMt5(all_positions)

    # Fallback heuristic without magic picks most recent
    pos, ticket, info = positions._resolve_open_position(
        mt5, symbol="EURUSD", side="BUY", volume=0.1,
    )
    assert pos.ticket == 200

    # With magic=1000, narrows to ticket 100
    pos, ticket, info = positions._resolve_open_position(
        mt5, symbol="EURUSD", side="BUY", volume=0.1, magic=1000,
    )
    assert pos.ticket == 100
    assert info.get("magic_filter") == 1000


def test_resolve_open_position_magic_no_match_returns_none():
    """An explicit magic number is a hard strategy-ownership filter."""
    all_positions = [
        SimpleNamespace(ticket=100, identifier=100, position_id=None, position=None, order=None, deal=None,
                        symbol="EURUSD", type=0, volume=0.1, magic=1000, time_update_msc=5000),
    ]
    mt5 = _HedgedFakeMt5(all_positions)

    pos, ticket, info = positions._resolve_open_position(
        mt5, symbol="EURUSD", side="BUY", volume=0.1, magic=9999,
    )
    assert pos is None
    assert ticket is None
    assert info["magic_filter"] == 9999


def test_select_position_candidate_ticket_candidates_disambiguates():
    """ticket_candidates narrows candidates to positions matching known tickets."""
    positions_list = [
        SimpleNamespace(ticket=100, symbol="EURUSD", type=0, volume=0.1, magic=0, time_update_msc=9000),
        SimpleNamespace(ticket=200, symbol="EURUSD", type=0, volume=0.1, magic=0, time_update_msc=5000),
        SimpleNamespace(ticket=300, symbol="EURUSD", type=0, volume=0.1, magic=0, time_update_msc=8000),
    ]
    mt5 = _HedgedFakeMt5(positions_list)

    # Without ticket_candidates: picks most recent (ticket 100, time_update_msc=9000)
    picked = positions._select_position_candidate(
        positions_list, symbol="EURUSD", side="BUY", volume=0.1, mt5=mt5,
    )
    assert picked.ticket == 100

    # With ticket_candidates=[200]: narrows to ticket 200
    picked = positions._select_position_candidate(
        positions_list, symbol="EURUSD", side="BUY", volume=0.1,
        ticket_candidates=[200], mt5=mt5,
    )
    assert picked.ticket == 200


def test_resolve_open_position_phase3_preserves_ticket_preference():
    """Phase 3 fallback passes ticket_candidates to _select_position_candidate."""
    all_positions = [
        SimpleNamespace(ticket=100, identifier=100, position_id=None, position=None, order=None, deal=None,
                        symbol="EURUSD", type=0, volume=0.1, magic=0, time_update_msc=9000),
        SimpleNamespace(ticket=200, identifier=200, position_id=None, position=None, order=None, deal=None,
                        symbol="EURUSD", type=0, volume=0.1, magic=0, time_update_msc=5000),
    ]

    class _FakeMt5SkipPhase1(_HedgedFakeMt5):
        """positions_get(ticket=X) returns None to skip Phase 1."""
        def positions_get(self, **kwargs):
            if "ticket" in kwargs:
                return None
            return super().positions_get(**kwargs)

    mt5 = _FakeMt5SkipPhase1(all_positions)

    # Phase 2 exact match still finds ticket 200 via identifier field
    pos, ticket, info = positions._resolve_open_position(
        mt5, ticket_candidates=[200], symbol="EURUSD", side="BUY", volume=0.1,
    )
    assert pos.ticket == 200
    assert info["method"] == "positions_get(fallback_exact)"

    # With unknown candidates, Phase 3 heuristic picks most recent
    pos, ticket, info = positions._resolve_open_position(
        mt5, ticket_candidates=[999], symbol="EURUSD", side="BUY", volume=0.1,
    )
    assert pos.ticket == 100
    assert info["method"] == "positions_get(fallback_heuristic)"
