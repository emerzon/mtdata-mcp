from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from mtdata.services import news_service as svc


def test_parser_memory_maps_database_instead_of_reading_it_whole(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "news.dat"
    header = bytearray(svc.MT5NewsParser.HEADER_SIZE)
    header[:4] = svc.struct.pack("<I", svc.MT5NewsParser.HEADER_SIZE)
    path.write_bytes(bytes(header) + b"record padding" * 20)
    parser = svc.MT5NewsParser(str(path))
    observed = {}

    def parse_records(data):
        observed["is_mmap"] = isinstance(data, svc.mmap.mmap)
        return []

    monkeypatch.setattr(parser, "_parse_records", parse_records)

    assert parser.parse() == []
    assert observed == {"is_mmap": True}


def _record(timestamp: datetime, subject: str = "Fed preview"):
    return SimpleNamespace(
        timestamp=timestamp,
        category="FXStreet",
        source="FXStreet",
        subject=subject,
        to_dict=lambda now=None: {
            "subject": subject,
            "category": "FXStreet",
            "source": "FXStreet",
            "published_at": timestamp.isoformat(),
        },
    )


def test_get_mt5_news_surfaces_warning_for_inverted_date_range(monkeypatch) -> None:
    class FakeParser:
        header_info = {"version": 1}
        parse_health = {"status": "ok", "candidates_scanned": 1, "records_parsed": 1,
                        "duplicates_skipped": 0, "validation_failures": 0}

        def __init__(self, path: str) -> None:
            assert path == "C:/tmp/news.dat"

        def parse(self):
            return [_record(datetime(2026, 3, 15, tzinfo=timezone.utc))]

    monkeypatch.setattr(svc, "MT5NewsParser", FakeParser)
    monkeypatch.setattr(svc, "_use_client_tz", lambda: False)

    result = svc.get_mt5_news(
        news_db_path="C:/tmp/news.dat",
        from_date="2026-03-20",
        to_date="2026-03-10",
    )

    assert result["success"] is True
    assert result["count"] == 0
    assert result["news"] == []
    assert result["warning"] == "from_date is after to_date; returning no results"


def test_mt5_news_record_preserves_absolute_published_time() -> None:
    published_at = datetime(2026, 3, 15, 12, 34, 56, tzinfo=timezone.utc)

    item = svc.MT5NewsRecord(
        timestamp=published_at,
        subject="Fed preview",
        category="FXStreet",
        source="FXStreet",
    ).to_dict(now=published_at + timedelta(minutes=10))

    assert item["published_at"] == "2026-03-15T12:34:56+00:00"
    assert item["relative_time"] == "10 minutes ago"


def test_get_mt5_news_reports_invalid_date_filter_as_input_error(monkeypatch) -> None:
    class FakeParser:
        header_info = {"version": 1}
        parse_health = {"status": "ok", "candidates_scanned": 1, "records_parsed": 1,
                        "duplicates_skipped": 0, "validation_failures": 0}

        def __init__(self, path: str) -> None:
            assert path == "C:/tmp/news.dat"

        def parse(self):
            return [_record(datetime(2026, 3, 15, tzinfo=timezone.utc))]

    monkeypatch.setattr(svc, "MT5NewsParser", FakeParser)
    monkeypatch.setattr(svc, "_use_client_tz", lambda: False)

    result = svc.get_mt5_news(
        news_db_path="C:/tmp/news.dat",
        from_date="not-a-date",
    )

    assert "Invalid news date filter" in result["error"]
    assert "ISO dates" in result["hint"]


def test_get_mt5_news_clamps_limit_at_service_layer(monkeypatch) -> None:
    class FakeParser:
        header_info = {"version": 1}
        parse_health = {"status": "ok", "candidates_scanned": 600, "records_parsed": 600,
                        "duplicates_skipped": 0, "validation_failures": 0}

        def __init__(self, path: str) -> None:
            assert path == "C:/tmp/news.dat"

        def parse(self):
            return [
                _record(datetime(2026, 3, 15, tzinfo=timezone.utc), subject=f"Item {idx}")
                for idx in range(600)
            ]

    monkeypatch.setattr(svc, "MT5NewsParser", FakeParser)
    monkeypatch.setattr(svc, "_use_client_tz", lambda: False)

    result = svc.get_mt5_news(news_db_path="C:/tmp/news.dat", limit=999)

    assert result["success"] is True
    assert result["limit"] == 500
    assert result["count"] == 500


def test_get_mt5_news_does_not_double_wrap_missing_file_error() -> None:
    result = svc.get_mt5_news(news_db_path="C:/definitely-missing/news.dat")

    assert result["error"].count("News database not found:") == 1


def test_parse_records_scans_only_candidate_prefix_offsets(monkeypatch) -> None:
    parser = object.__new__(svc.MT5NewsParser)
    parser.parse_stats = {
        "candidates_scanned": 0, "records_parsed": 0,
        "duplicates_skipped": 0, "validation_failures": 0,
    }
    offsets: list[int] = []

    def _fake_parse_inline_record(data: bytes, offset: int):
        offsets.append(offset)
        return None

    monkeypatch.setattr(parser, "_parse_inline_record", _fake_parse_inline_record)

    data = (
        b"A" * svc.MT5NewsParser.HEADER_SIZE
        + b"noise"
        + (b"\x00" * 12)
        + b"middle"
        + (b"\x00" * 12)
        + b"tail" * 32
    )

    parser._parse_records(data)

    first = svc.MT5NewsParser.HEADER_SIZE + len(b"noise")
    second = first + 12 + len(b"middle")
    assert offsets == [first, second]


def test_parse_header_rejects_unknown_layout_size() -> None:
    parser = svc.MT5NewsParser("unused.dat")
    header = (508).to_bytes(4, "little") + b"\x00" * 500

    with pytest.raises(ValueError, match="Unsupported MT5 news.dat layout"):
        parser._parse_header(header)


def test_parse_header_rejects_truncated_file() -> None:
    parser = svc.MT5NewsParser("unused.dat")

    with pytest.raises(ValueError, match="truncated"):
        parser._parse_header(b"\x00" * 100)


# ---------- parse_health diagnostics ----------

def test_parse_stats_tracks_candidates_and_failures(monkeypatch) -> None:
    """_parse_records tallies candidates, failures, duplicates correctly."""
    parser = object.__new__(svc.MT5NewsParser)
    parser.parse_stats = {
        "candidates_scanned": 0, "records_parsed": 0,
        "duplicates_skipped": 0, "validation_failures": 0,
    }

    call_count = 0

    def _alternating_parse(data: bytes, offset: int):
        nonlocal call_count
        call_count += 1
        if call_count % 2 == 0:
            return None  # simulate failure every other candidate
        ts = datetime(2026, 1, 1, call_count, tzinfo=timezone.utc)
        return SimpleNamespace(
            timestamp=ts, subject=f"news-{call_count}", source="src",
            category="cat",
        )

    monkeypatch.setattr(parser, "_parse_inline_record", _alternating_parse)

    # Build data with 4 candidate prefixes
    prefix = b"\x00" * 12
    payload = b"A" * svc.MT5NewsParser.HEADER_SIZE
    for _ in range(4):
        payload += prefix + b"X" * 80
    payload += b"Z" * 80  # trailing pad so max_offset check passes

    records = parser._parse_records(payload)

    assert parser.parse_stats["candidates_scanned"] == 4
    assert parser.parse_stats["records_parsed"] == 2
    assert parser.parse_stats["validation_failures"] == 2
    assert parser.parse_stats["duplicates_skipped"] == 0
    assert len(records) == 2


def test_parse_stats_tracks_duplicates(monkeypatch) -> None:
    """Duplicate records (same timestamp+subject+source) are counted."""
    parser = object.__new__(svc.MT5NewsParser)
    parser.parse_stats = {
        "candidates_scanned": 0, "records_parsed": 0,
        "duplicates_skipped": 0, "validation_failures": 0,
    }

    fixed_ts = datetime(2026, 6, 1, tzinfo=timezone.utc)

    def _always_same(data: bytes, offset: int):
        return SimpleNamespace(
            timestamp=fixed_ts, subject="same", source="src", category="cat",
        )

    monkeypatch.setattr(parser, "_parse_inline_record", _always_same)

    prefix = b"\x00" * 12
    payload = b"A" * svc.MT5NewsParser.HEADER_SIZE
    for _ in range(3):
        payload += prefix + b"X" * 80
    payload += b"Z" * 80

    records = parser._parse_records(payload)

    assert parser.parse_stats["candidates_scanned"] == 3
    assert parser.parse_stats["records_parsed"] == 1
    assert parser.parse_stats["duplicates_skipped"] == 2
    assert len(records) == 1


def test_parse_health_ok_when_all_candidates_succeed() -> None:
    parser = object.__new__(svc.MT5NewsParser)
    parser.parse_stats = {
        "candidates_scanned": 10, "records_parsed": 10,
        "duplicates_skipped": 0, "validation_failures": 0,
    }
    health = parser.parse_health
    assert health["status"] == "ok"
    assert health["candidates_scanned"] == 10
    assert health["records_parsed"] == 10


def test_parse_health_degraded_when_some_failures() -> None:
    parser = object.__new__(svc.MT5NewsParser)
    parser.parse_stats = {
        "candidates_scanned": 10, "records_parsed": 7,
        "duplicates_skipped": 0, "validation_failures": 3,
    }
    assert parser.parse_health["status"] == "degraded"


def test_parse_health_failed_when_all_fail() -> None:
    parser = object.__new__(svc.MT5NewsParser)
    parser.parse_stats = {
        "candidates_scanned": 5, "records_parsed": 0,
        "duplicates_skipped": 0, "validation_failures": 5,
    }
    assert parser.parse_health["status"] == "failed"


def test_parse_health_empty_when_no_candidates() -> None:
    parser = object.__new__(svc.MT5NewsParser)
    parser.parse_stats = {
        "candidates_scanned": 0, "records_parsed": 0,
        "duplicates_skipped": 0, "validation_failures": 0,
    }
    assert parser.parse_health["status"] == "empty"


def test_get_mt5_news_includes_parse_health_in_response(monkeypatch) -> None:
    class FakeParser:
        header_info = {"version": 1}
        parse_health = {"status": "degraded", "candidates_scanned": 5,
                        "records_parsed": 3, "duplicates_skipped": 0,
                        "validation_failures": 2}

        def __init__(self, path: str) -> None:
            pass

        def parse(self):
            return [_record(datetime(2026, 3, 15, tzinfo=timezone.utc))]

    monkeypatch.setattr(svc, "MT5NewsParser", FakeParser)
    monkeypatch.setattr(svc, "_use_client_tz", lambda: False)

    result = svc.get_mt5_news(news_db_path="C:/tmp/news.dat")
    assert result["parse_health"]["status"] == "degraded"
    assert result["parse_health"]["validation_failures"] == 2


def test_terminal_root_from_mt5_api_uses_shared_adapter(tmp_path, monkeypatch) -> None:
    root = tmp_path / "terminal"
    root.mkdir()
    monkeypatch.setattr(
        svc.mt5,
        "terminal_info",
        lambda: SimpleNamespace(data_path=str(root)),
    )

    assert svc._terminal_root_from_mt5_api() == root


def test_terminal_root_from_mt5_api_returns_none_without_session(monkeypatch) -> None:
    monkeypatch.setattr(svc.mt5, "terminal_info", lambda: None)

    assert svc._terminal_root_from_mt5_api() is None
