from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mtdata.core.causal import cross


@pytest.mark.parametrize(("count", "overlap", "max_lag", "valid"), [(50, 50, 3, False), (52, 50, 10, False), (50, 40, 3, True), (53, 50, 3, True)])
def test_cross_correlation_requires_overlap_for_the_entire_search(count, overlap, max_lag, valid):
    index = pd.date_range("2024-01-01", periods=count, freq="h")
    rng = np.random.default_rng(42)
    left = pd.Series(rng.normal(size=count), index=index)
    right = pd.Series(rng.normal(size=count), index=index)
    raw = cross.cross_correlation
    while hasattr(raw, "__wrapped__"):
        raw = raw.__wrapped__
    with patch.object(cross, "_causal_connection_error", return_value=None), patch.object(cross, "_fetch_series_for_window", side_effect=[(left, None), (right, None)]):
        result = raw("EURUSD,GBPUSD", window_bars=count, min_overlap=overlap, max_lag=max_lag, transform="level", bootstrap_samples=20, detail="full")
    if valid:
        assert result["success"] is True
        assert result["context"]["lag_tests"] == 2 * max_lag + 1
    else:
        assert result["error_code"] == "insufficient_overlap"
        assert "testing every lag" in result["error"]
