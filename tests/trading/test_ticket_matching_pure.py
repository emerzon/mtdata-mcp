from types import SimpleNamespace

from mtdata.core.trading.validation import _exact_ticket_row


def test_exact_ticket_row_returns_first_exact_match() -> None:
    first = SimpleNamespace(ticket="9007199254740993")
    duplicate = SimpleNamespace(ticket=9007199254740993)

    assert _exact_ticket_row([SimpleNamespace(ticket=1), first, duplicate], first.ticket) is first


def test_exact_ticket_row_rejects_invalid_or_missing_matches() -> None:
    rows = [SimpleNamespace(ticket=12), SimpleNamespace(ticket="not-a-ticket")]

    assert _exact_ticket_row(rows, 13) is None
    assert _exact_ticket_row(rows, 12.0) is None
    assert _exact_ticket_row(None, 12) is None
