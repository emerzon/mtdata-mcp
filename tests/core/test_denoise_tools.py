from __future__ import annotations

from mtdata.core import denoise as denoise_core
from mtdata.core._mcp_tools import shape_public_tool_output


def _raw_list_methods():
    raw = denoise_core.denoise_list_methods
    while hasattr(raw, "__wrapped__"):
        raw = raw.__wrapped__
    return raw


def test_denoise_list_methods_compact_lists_small_catalog_by_default(monkeypatch):
    rows = [
        {
            "method": f"m{idx}",
            "available": idx % 2 == 0,
            "requires": "pkg",
            "params": ["window", "alpha"],
            "supports": {"causality": ["causal"]},
            "description": "verbose method description",
        }
        for idx in range(12)
    ]
    monkeypatch.setattr(denoise_core, "_denoise_methods", lambda available_only=False: rows)

    result = _raw_list_methods()()

    assert result["count"] == 12
    assert result["row_key"] == "methods"
    assert not {"total", "limit", "has_more", "methods_hidden"} & result.keys()
    assert result["pagination"] == {
        "total": 12,
        "returned": 12,
        "offset": 0,
        "limit": 30,
        "has_more": False,
        "more_available": 0,
    }
    assert result["columns"] == [
        "method",
        "available",
        "causality",
        "requires_causality_opt_in",
    ]
    assert set(result["methods"][0]) == {
        "method",
        "available",
        "causality",
        "requires_causality_opt_in",
    }
    assert result["methods"][0]["causality"] == ["causal"]
    assert "list_all_hint" not in result

    standard = _raw_list_methods()(detail="standard")
    assert standard["detail"] == "standard"
    assert standard["columns"] == [
        "method",
        "available",
        "causality",
        "requires_causality_opt_in",
        "requires",
        "params",
    ]
    assert standard["methods"][0]["requires"] == "pkg"
    assert standard["methods"][0]["params"] == ["window", "alpha"]


def test_denoise_list_methods_compact_reports_hidden_catalog_hint(monkeypatch):
    rows = [{"method": f"m{idx}", "available": True} for idx in range(35)]
    monkeypatch.setattr(denoise_core, "_denoise_methods", lambda available_only=False: rows)

    result = _raw_list_methods()()

    assert result["count"] == 30
    assert not {"total", "limit", "has_more", "methods_hidden"} & result.keys()
    assert result["pagination"] == {
        "total": 35,
        "returned": 30,
        "offset": 0,
        "limit": 30,
        "has_more": True,
        "more_available": 5,
    }
    assert result["list_all_hint"] == "Pass offset=30 to continue listing methods."


def test_denoise_list_methods_consumes_reported_next_offset(monkeypatch):
    rows = [{"method": f"m{idx}", "available": True} for idx in range(35)]
    monkeypatch.setattr(denoise_core, "_denoise_methods", lambda available_only=False: rows)

    first_page = shape_public_tool_output(
        _raw_list_methods()(limit=30),
        tool_name="denoise_list_methods",
        detail="compact",
    )
    second_page = _raw_list_methods()(
        limit=30,
        offset=first_page["pagination"]["next_offset"],
    )

    assert first_page["pagination"]["next_offset"] == 30
    assert second_page["pagination"] == {
        "total": 35,
        "returned": 5,
        "offset": 30,
        "limit": 30,
        "has_more": False,
        "more_available": 0,
    }
    assert [row["method"] for row in second_page["methods"]] == [
        "m30",
        "m31",
        "m32",
        "m33",
        "m34",
    ]


def test_denoise_list_methods_full_respects_limit(monkeypatch):
    rows = [
        {
            "method": "kalman",
            "available": True,
            "requires": "",
            "params": ["process_var"],
            "supports": {"causality": ["causal"]},
            "description": "Kalman smoothing",
        }
    ]
    monkeypatch.setattr(denoise_core, "_denoise_methods", lambda available_only=False: rows)

    result = _raw_list_methods()(detail="full", limit=1)

    assert result["count"] == 1
    assert result["pagination"] == {
        "total": 1,
        "returned": 1,
        "offset": 0,
        "limit": 1,
        "has_more": False,
        "more_available": 0,
    }
    assert result["methods"] == rows
    assert "has_more" not in result


def test_denoise_list_methods_filters_for_causal_available_core_only(monkeypatch):
    rows = [
        {
            "method": "ema",
            "available": True,
            "requires": "",
            "params": ["span"],
            "supports": {"causality": ["causal", "zero_phase"]},
        },
        {
            "method": "wavelet",
            "available": True,
            "requires": "PyWavelets",
            "params": ["wavelet"],
            "supports": {"causality": ["zero_phase"]},
        },
        {
            "method": "vmd",
            "available": False,
            "requires": "vmdpy",
            "params": ["k"],
            "supports": {"causality": ["zero_phase"]},
        },
    ]
    monkeypatch.setattr(
        denoise_core,
        "_denoise_methods",
        lambda available_only=False: [row for row in rows if row["available"]]
        if available_only
        else rows,
    )

    result = _raw_list_methods()(available_only=True, causality="causal", core_only=True)

    assert result["count"] == 1
    assert result["pagination"]["total"] == 1
    assert result["available_only"] is True
    assert result["causality"] == "causal"
    assert result["core_only"] is True
    assert result["methods"][0]["method"] == "ema"
