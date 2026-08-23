import pandas as pd

from mtdata.utils import simplify as simp


def _sample_ohlc_df():
    epochs = [0, 1800, 3600, 5400, 7200]
    return pd.DataFrame(
        {
            "time": pd.to_datetime(epochs, unit="s"),
            "__epoch": epochs,
            "open": [1.0, 2.0, 3.0, 4.0, 5.0],
            "high": [3.0, 4.0, 5.0, 6.0, 7.0],
            "low": [0.5, 1.5, 2.5, 3.5, 4.5],
            "close": [2.0, 3.0, 4.0, 5.0, 6.0],
            "volume": [10, 5, 7, 8, 9],
        }
    )


def test_simplify_dataframe_rows_ext_empty_dataframe_returns_none_meta():
    empty = pd.DataFrame(columns=["time", "close"])

    result_df, meta = simp._simplify_dataframe_rows_ext(empty, ["time", "close"], {"mode": "select"})

    assert result_df.empty
    assert meta is None


def test_simplify_dataframe_rows_ext_approximate_aggregates_segments():
    df = _sample_ohlc_df()

    result_df, meta = simp._simplify_dataframe_rows_ext(
        df,
        list(df.columns),
        {"mode": "approximate", "method": "lttb", "points": 3},
    )

    assert len(result_df) == 3
    assert meta["mode"] == "approximate"
    assert meta["method"] == "lttb"
    assert result_df["volume"].sum() == df["volume"].sum()
    assert result_df["volume"].max() > df["volume"].max()
    assert result_df.iloc[1]["open"] == 2.0
    assert result_df.iloc[1]["high"] == 6.0
    assert result_df.iloc[1]["low"] == 1.5
    assert result_df.iloc[1]["close"] == 5.0


def test_handle_resample_missing_rule_returns_error_metadata():
    df = _sample_ohlc_df()

    result_df, meta = simp._handle_resample_mode(df, list(df.columns), {})

    assert result_df.equals(df)
    assert meta == {"error": "Missing rule for resample"}


def test_handle_resample_aggregates_ohlc_and_volume():
    df = _sample_ohlc_df()
    headers = ["time", "open", "high", "low", "close", "volume"]

    result_df, meta = simp._handle_resample_mode(df, headers, {"rule": "1h"})

    assert len(result_df) == 3
    assert "time" in result_df.columns
    assert "__epoch" in result_df.columns
    assert result_df.iloc[0]["time"] == df.iloc[0]["time"]
    assert result_df.iloc[0]["__epoch"] == df.iloc[0]["__epoch"]
    assert result_df.iloc[0]["open"] == 1.0
    assert result_df.iloc[0]["high"] == 4.0
    assert result_df.iloc[0]["low"] == 0.5
    assert result_df.iloc[0]["close"] == 3.0
    assert result_df.iloc[0]["volume"] == 15
    assert meta == {"mode": "resample", "rule": "1h", "rows": 3}


def test_handle_resample_supports_bucket_seconds():
    df = _sample_ohlc_df()
    headers = ["time", "open", "high", "low", "close", "volume"]

    result_df, meta = simp._handle_resample_mode(df, headers, {"bucket_seconds": 3600, "points": 3})

    assert len(result_df) == 3
    assert result_df.iloc[0]["time"] == df.iloc[0]["time"]
    assert result_df.iloc[0]["__epoch"] == df.iloc[0]["__epoch"]
    assert result_df.iloc[0]["open"] == 1.0
    assert result_df.iloc[0]["high"] == 4.0
    assert result_df.iloc[0]["low"] == 0.5
    assert result_df.iloc[0]["close"] == 3.0
    assert result_df.iloc[0]["volume"] == 15
    assert meta == {"mode": "resample", "bucket_seconds": 3600, "rows": 3}


def test_simplify_dataframe_rows_resample_uses_shared_bucketed_aggregation():
    df = _sample_ohlc_df()
    headers = ["time", "open", "high", "low", "close", "volume"]

    result_df, meta = simp._simplify_dataframe_rows_ext(
        df,
        headers,
        {"mode": "resample", "bucket_seconds": 3600},
    )

    assert len(result_df) == 3
    assert result_df.iloc[0]["time"] == df.iloc[0]["time"]
    assert result_df.iloc[0]["__epoch"] == df.iloc[0]["__epoch"]
    assert result_df.iloc[0]["open"] == 1.0
    assert result_df.iloc[0]["high"] == 4.0
    assert result_df.iloc[0]["low"] == 0.5
    assert result_df.iloc[0]["close"] == 3.0
    assert result_df.iloc[0]["volume"] == 15
    assert meta == {"mode": "resample", "bucket_seconds": 3600, "rows": 3}


def test_handle_select_returns_subset_and_metadata(monkeypatch):
    df = _sample_ohlc_df()

    monkeypatch.setattr(simp, "_choose_simplify_points", lambda original_count, spec: 3)
    monkeypatch.setattr(
        simp,
        "_select_indices_for_timeseries",
        lambda epochs, series, spec: ([0, 2, 4], "lttb", {"bucket_size": 2}),
    )

    result_df, meta = simp._handle_select_mode(df, list(df.columns), {"mode": "select", "points": 3})

    assert list(result_df.index) == [0, 2, 4]
    assert meta == {
        "mode": "select",
        "method": "lttb",
        "original_rows": 5,
        "returned_rows": 3,
        "points": 3,
        "bucket_size": 2,
    }


def test_handle_select_returns_original_when_series_unavailable(monkeypatch):
    df = pd.DataFrame({"time": ["t1", "t2", "t3"], "__epoch": [1, 2, 3]})

    monkeypatch.setattr(simp, "_choose_simplify_points", lambda original_count, spec: 2)

    result_df, meta = simp._handle_select_mode(df, ["time"], {"points": 2})

    assert result_df.equals(df)
    assert meta is None


def test_simplify_modes_no_recursive_delegation():
    df = _sample_ohlc_df()

    enc_df, enc_meta = simp._simplify_dataframe_rows_ext(df, list(df.columns), {"mode": "encode", "schema": "delta"})
    assert enc_meta is not None
    assert enc_meta.get("mode") == "encode"
    assert "encoding" in enc_df.columns

    seg_df, seg_meta = simp._simplify_dataframe_rows_ext(df, list(df.columns), {"mode": "segment", "threshold_pct": 0.01})
    assert seg_meta is not None
    assert seg_meta.get("mode") == "segment"
    assert len(seg_df) >= 2

    sym_df, sym_meta = simp._simplify_dataframe_rows_ext(df, list(df.columns), {"mode": "symbolic", "paa": 3})
    assert sym_meta is not None
    assert sym_meta.get("mode") == "symbolic"
    assert "symbolic" in sym_df.columns
