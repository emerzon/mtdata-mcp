from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

from mtdata.services.data_service.candles import (
    _build_rates_df,
    _candle_table_from_frame,
    _format_candle_times,
    _round_row_price_columns,
)
from mtdata.utils.utils import _format_numeric_rows_from_df, _table_from_rows


def test_columnar_candle_table_matches_scalar_rounding_golden() -> None:
    headers = ["time", "open", "high", "low", "close", "tick_volume"]
    frame = pd.DataFrame(
        {
            "time": [
                "2026-09-10T10:00Z",
                "2026-09-10T10:01Z",
                "2026-09-10T10:02Z",
                "2026-09-10T10:03Z",
            ],
            "open": [2.675, 3981.575, -2.675, 1.23456],
            "high": [2.685, 3981.585, -2.665, 1.23466],
            "low": [2.665, 3981.565, -2.685, 1.23446],
            "close": [2.675, 3981.575, -2.675, 1.23456],
            "tick_volume": [1, 2, 3, 4],
        }
    )
    price_columns = frozenset({"open", "high", "low", "close"})
    scalar_rows = _round_row_price_columns(
        _format_numeric_rows_from_df(frame, headers, stringify=False),
        headers,
        digits=2,
        price_columns=price_columns,
    )
    expected = _table_from_rows(headers, scalar_rows)

    actual = _candle_table_from_frame(
        frame,
        headers,
        digits=2,
        price_columns=price_columns,
    )

    assert actual == expected
    assert actual["data"][0]["open"] == 2.67
    assert actual["data"][1]["open"] == 3981.57


def test_candle_time_formatting_is_vectorized_with_exact_utc_and_dst_values() -> None:
    epochs = pd.Series(
        [
            pd.Timestamp("2026-01-15T14:30:00Z").timestamp(),
            pd.Timestamp("2026-07-15T13:30:00Z").timestamp(),
        ]
    )
    utc_frame = pd.DataFrame({"time": epochs, "__epoch": epochs})
    local_frame = utc_frame.copy()

    _format_candle_times(
        utc_frame,
        ["time"],
        time_as_epoch=False,
        use_client_tz=False,
        client_tz=None,
    )
    _format_candle_times(
        local_frame,
        ["time"],
        time_as_epoch=False,
        use_client_tz=True,
        client_tz=ZoneInfo("America/New_York"),
    )

    assert utc_frame["time"].tolist() == [
        "2026-01-15T14:30Z",
        "2026-07-15T13:30Z",
    ]
    assert local_frame["time"].tolist() == [
        "2026-01-15T09:30-05:00",
        "2026-07-15T09:30-04:00",
    ]


def test_rate_frame_can_defer_display_time_conversion(monkeypatch) -> None:
    raw = pd.DataFrame(
        {
            "time": [1_700_000_000.0],
            "open": [1.1],
            "high": [1.2],
            "low": [1.0],
            "close": [1.15],
            "tick_volume": [10],
        }
    )
    monkeypatch.setattr(
        "mtdata.services.data_service.candles._format_rate_times",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("display time conversion must be deferred")
        ),
    )

    frame = _build_rates_df(raw, False, format_time=False)

    assert frame["time"].tolist() == [1_700_000_000.0]
    assert frame["__epoch"].tolist() == [1_700_000_000.0]
