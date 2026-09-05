from mtdata.core._mcp_tools import shape_public_tool_output


def test_compact_outliers_preserve_scoring_window_and_cutoff():
    window = {"requested_as_of": "2024-01-02T12:30Z", "period_start": "2024-01-01T00:00Z", "period_end": "2024-01-02T11:00Z", "bars_used": 36}
    result = shape_public_tool_output(
        {"success": True, "analysis_window": window, "history_policy": "completed_bars_only", "items": []},
        tool_name="outliers_detect", detail="compact",
    )
    assert result["analysis_window"] == window
    assert result["history_policy"] == "completed_bars_only"
