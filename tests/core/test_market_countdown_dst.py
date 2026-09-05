from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from mtdata.core.market_status import _check_market_status


@pytest.mark.parametrize(("local_time", "minutes"), [("2026-10-31T12:00:00", 2790), ("2026-03-07T12:00:00", 2670), ("2026-09-05T12:00:00", 4170)])
def test_countdown_matches_elapsed_utc_time(local_time, minutes):
    now = datetime.fromisoformat(local_time).replace(tzinfo=ZoneInfo("America/New_York"))
    result = _check_market_status("NYSE", now)
    opening = datetime.fromisoformat(result["next_open"])
    assert result["minutes_until_open"] == minutes
    assert minutes == int((opening.astimezone(timezone.utc) - now.astimezone(timezone.utc)).total_seconds() // 60)
