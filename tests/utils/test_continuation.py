import base64
import json

import pytest

from mtdata.utils.continuation import (
    decode_continuation_cursor,
    encode_continuation_cursor,
)


def _decode(cursor: str) -> dict:
    return decode_continuation_cursor(
        cursor,
        invalid_message="invalid cursor",
        unsupported_version_message="unsupported cursor",
        expected_versions=1,
    )


def test_continuation_cursor_compresses_repetitive_state_and_round_trips() -> None:
    payload = {
        "v": 1,
        "scope": {
            "history_kind": "deals",
            "start": None,
            "end": None,
            "minutes_back": None,
            "symbol": "EURUSD",
            "magic": None,
            "side": None,
            "position_ticket": None,
            "deal_ticket": None,
            "order_ticket": None,
            "order": "asc",
        },
        "from": "2026-08-22T14:00:00.000000+00:00",
        "to": "2026-08-29T14:00:00.000000+00:00",
        "last_milliseconds": "1788012000123.0",
        "last_ticket": "123456789",
        "position": 20,
        "issued_at": 1788012000,
    }

    cursor = encode_continuation_cursor(payload)
    plain_size = len(
        base64.urlsafe_b64encode(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        )
        .decode()
        .rstrip("=")
    )

    assert cursor.startswith("z.")
    assert len(cursor) <= plain_size * 0.7
    assert _decode(cursor) == payload


def test_continuation_cursor_decoder_accepts_legacy_uncompressed_tokens() -> None:
    payload = {"v": 1, "position": 5}
    legacy = (
        base64.urlsafe_b64encode(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        )
        .decode()
        .rstrip("=")
    )

    assert _decode(legacy) == payload


@pytest.mark.parametrize(
    "cursor",
    ["z.not-valid", "x" * 32_769],
    ids=["malformed", "oversized"],
)
def test_continuation_cursor_rejects_invalid_or_oversized_tokens(cursor: str) -> None:
    with pytest.raises(ValueError, match="invalid cursor"):
        _decode(cursor)
