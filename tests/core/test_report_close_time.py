from zoneinfo import ZoneInfo

from mtdata.core.report.use_cases import _report_base_timestamp_candidates


def test_report_daily_close_uses_broker_calendar(monkeypatch):
    monkeypatch.setattr("mtdata.bootstrap.settings.mt5_config.get_server_tz", lambda: ZoneInfo("Europe/Nicosia"))
    monkeypatch.setattr("mtdata.bootstrap.settings.mt5_config.time_offset_minutes", 0)
    times = _report_base_timestamp_candidates({"context": {"timeframe": "D1", "last_snapshot": {"time": "2026-03-28T22:00Z"}}})
    assert times[0].isoformat() == "2026-03-29T21:00:00+00:00"


def test_report_does_not_add_a_timeframe_to_existing_close_timestamp():
    times = _report_base_timestamp_candidates({"context": {"timeframe": "H1", "data_as_of": "2026-09-04T12:00Z"}})
    assert times[0].isoformat() == "2026-09-04T12:00:00+00:00"
