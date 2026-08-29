from __future__ import annotations

import pytest

from mtdata.utils.level_confluence import (
    _cluster_records,
    build_level_confluence_payload,
)


def test_confluence_clusters_pivot_sr_and_fibonacci_levels():
    payload = build_level_confluence_payload(
        symbol="EURUSD",
        pivot_timeframe="D1",
        sr_timeframe="auto",
        reference_price=1.08,
        tolerance_pct=0.1,
        pivot_methods=[
            {
                "method": "classic",
                "levels": {"R2": 1.0850},
                "pivot": 1.08,
            }
        ],
        support_resistance_payload={
            "success": True,
            "symbol": "EURUSD",
            "timeframe": "auto",
            "levels": [
                {
                    "type": "resistance",
                    "value": 1.0853,
                    "score": 5.0,
                    "touches": 3,
                }
            ],
            "fibonacci": {
                "levels": [
                    {
                        "label": "61.8%",
                        "ratio": 0.618,
                        "kind": "retracement",
                        "value": 1.0848,
                    }
                ]
            },
        },
        max_distance_pct=1.0,
        detail="standard",
    )

    assert payload["success"] is True
    assert payload["levels"]
    cluster = payload["levels"][0]
    assert cluster["source_families"] == [
        "pivot_formula",
        "swing_fibonacci",
        "touch_derived",
    ]
    assert cluster["role"] == "above"
    assert {source["source"] for source in cluster["sources"]} == {
        "pivot",
        "support_resistance",
        "fibonacci",
    }
    assert cluster["reasons"]


def test_confluence_compact_omits_verbose_source_narration():
    payload = build_level_confluence_payload(
        symbol="EURUSD",
        pivot_timeframe="D1",
        sr_timeframe="H1",
        reference_price=1.08,
        tolerance_pct=0.1,
        pivot_methods=[
            {
                "method": "classic",
                "levels": {"R1": 1.0802, "R2": 1.0803},
                "pivot": 1.08,
            }
        ],
        support_resistance_payload={
            "levels": [{"type": "support", "value": 1.0801, "score": 4, "touches": 2}]
        },
        detail="compact",
    )

    cluster = payload["levels"][0]
    assert "reasons" not in cluster
    assert "source_labels" not in cluster
    assert "sources" not in cluster
    assert "score_components" not in cluster
    assert cluster["source_count"] == 2
    assert cluster["record_count"] == 3
    assert payload["tolerance"]["pct"] == 0.1
    assert "fraction" not in payload["tolerance"]
    assert "detail" not in payload
    assert payload["units"]["distance_pct"] == "percent (1.0 = 1%)"
    assert payload["units"]["score"] == "unbounded_heuristic_points"
    assert "max_distance_pct" not in payload
    assert payload["min_source_families"] == 2
    assert payload["level_coverage"] == {"above": 1, "below": 0, "inside": 0, "at": 0}
    assert "No confluence levels below" in payload["coverage_note"]


def test_confluence_standard_keeps_units_and_filter_context():
    payload = build_level_confluence_payload(
        symbol="EURUSD",
        pivot_timeframe="D1",
        sr_timeframe="H1",
        reference_price=1.08,
        tolerance_pct=0.1,
        pivot_methods=[{"method": "classic", "levels": {"PP": 1.0801}}],
        support_resistance_payload={"levels": []},
        max_distance_pct=2.0,
        min_source_families=1,
        detail="standard",
    )

    assert payload["detail"] == "standard"
    assert payload["tolerance"]["pct"] == pytest.approx(0.1)
    assert payload["units"]["tolerance.pct"] == "percent (1.0 = 1%)"
    assert "fraction" not in payload["tolerance"]
    assert payload["max_distance_pct"] == 2.0
    assert payload["min_source_families"] == 1


def test_confluence_rejects_unattainable_source_family_count():
    payload = build_level_confluence_payload(
        symbol="EURUSD",
        pivot_timeframe="D1",
        sr_timeframe="H1",
        reference_price=1.08,
        tolerance_pct=0.1,
        pivot_methods=[{"method": "classic", "levels": {"PP": 1.0801}}],
        support_resistance_payload={"levels": []},
        min_source_families=5,
        detail="compact",
    )

    assert payload["success"] is False
    assert payload["error_code"] == "invalid_parameter"
    assert payload["parameter"] == "min_source_families"
    assert payload["value"] == 5
    assert payload["enabled_source_families"] == [
        "pivot_formula",
        "touch_derived",
        "swing_fibonacci",
    ]
    assert payload["max_enabled_source_families"] == 3
    assert "volume_profile_source" in payload["remediation"]

    with_profile = build_level_confluence_payload(
        symbol="EURUSD",
        pivot_timeframe="D1",
        sr_timeframe="H1",
        reference_price=1.08,
        tolerance_pct=0.1,
        pivot_methods=[{"method": "classic", "levels": {"PP": 1.0801}}],
        support_resistance_payload={"levels": []},
        min_source_families=4,
        volume_profile_payload={"success": True, "levels": []},
        detail="compact",
    )
    assert with_profile["success"] is True
    assert with_profile["max_enabled_source_families"] == 4
    assert with_profile["levels"] == []


def test_confluence_default_requires_two_source_families():
    payload = build_level_confluence_payload(
        symbol="EURUSD",
        pivot_timeframe="D1",
        sr_timeframe="H1",
        reference_price=1.08,
        tolerance_pct=0.1,
        pivot_methods=[{"method": "classic", "levels": {"PP": 1.0801}}],
        support_resistance_payload={"levels": []},
        detail="compact",
    )

    assert payload["min_source_families"] == 2
    assert payload["levels"] == []
    assert "min_source_families=1" in payload["level_scan_note"]


def test_pivot_original_resistance_below_reference_is_role_below():
    payload = build_level_confluence_payload(
        symbol="EURUSD",
        pivot_timeframe="D1",
        sr_timeframe="H1",
        reference_price=1.10,
        tolerance_pct=0.1,
        pivot_methods=[
            {
                "method": "classic",
                "levels": {"R1": 1.09},
                "pivot": 1.085,
            }
        ],
        support_resistance_payload={"levels": []},
        max_distance_pct=2.0,
        detail="full",
    )

    candidate = payload["candidates"][0]
    assert candidate["family"] == "pivot_formula"
    assert candidate["role"] == "below"
    assert "Classic R1" in candidate["label"]


def test_single_family_clusters_are_returned_but_score_lower_than_multi_family():
    single = build_level_confluence_payload(
        symbol="EURUSD",
        pivot_timeframe="D1",
        sr_timeframe="H1",
        reference_price=1.08,
        tolerance_pct=0.1,
        pivot_methods=[{"method": "classic", "levels": {"PP": 1.0801}}],
        support_resistance_payload={"levels": []},
        min_source_families=1,
        detail="standard",
    )
    multi = build_level_confluence_payload(
        symbol="EURUSD",
        pivot_timeframe="D1",
        sr_timeframe="H1",
        reference_price=1.08,
        tolerance_pct=0.1,
        pivot_methods=[{"method": "classic", "levels": {"PP": 1.0801}}],
        support_resistance_payload={
            "levels": [{"type": "support", "value": 1.0802, "score": 4, "touches": 2}]
        },
        detail="standard",
    )

    assert single["levels"]
    assert multi["levels"]
    assert single["levels"][0]["source_families"] == ["pivot_formula"]
    assert multi["levels"][0]["score"] > single["levels"][0]["score"]


def test_confluence_cluster_width_never_exceeds_tolerance():
    records = [
        {"price": 100.0},
        {"price": 100.8},
        {"price": 101.4},
    ]

    clusters = _cluster_records(records, tolerance_abs=1.0)

    assert [[row["price"] for row in group] for group in clusters] == [
        [100.0, 100.8],
        [101.4],
    ]
    assert all(
        max(row["price"] for row in group)
        - min(row["price"] for row in group)
        <= 1.0
        for group in clusters
    )


def test_duplicate_pivot_prices_do_not_earn_method_agreement_bonus():
    payload = build_level_confluence_payload(
        symbol="EURUSD",
        pivot_timeframe="D1",
        sr_timeframe="H1",
        reference_price=1.1,
        tolerance_pct=0.1,
        pivot_methods=[
            {"method": "classic", "levels": {"PP": 1.1}},
            {"method": "fibonacci", "levels": {"PP": 1.1}},
            {"method": "camarilla", "levels": {"PP": 1.1}},
        ],
        support_resistance_payload={"levels": []},
        min_source_families=1,
        detail="standard",
    )

    components = payload["levels"][0]["score_components"]
    assert components["pivot_method_bonus"] == 0.0
    assert components["independent_pivot_prices"] == 1


def test_confluence_reference_inside_nonzero_width_cluster_is_inside():
    payload = build_level_confluence_payload(
        symbol="EURUSD",
        pivot_timeframe="D1",
        sr_timeframe="H1",
        reference_price=1.16736,
        tolerance_pct=0.2,
        pivot_methods=[
            {
                "method": "classic",
                "levels": {"R1": 1.16683, "R2": 1.16856},
                "pivot": 1.16743,
            }
        ],
        support_resistance_payload={
            "success": True,
            "levels": [
                {
                    "type": "resistance",
                    "value": 1.16743,
                    "score": 5.0,
                    "touches": 3,
                }
            ],
        },
        max_distance_pct=1.0,
        min_source_families=1,
        detail="standard",
    )

    assert payload["success"] is True
    cluster = payload["levels"][0]
    assert cluster["range"]["low"] <= 1.16736 <= cluster["range"]["high"]
    assert cluster["range"]["width"] > 0
    assert cluster["role"] == "inside"
    assert cluster["distance_pct"] == 0.0
    assert cluster["centroid_distance_pct"] != 0.0
    assert payload["level_coverage"]["inside"] >= 1


def test_confluence_source_count_is_family_count_not_record_count():
    payload = build_level_confluence_payload(
        symbol="EURUSD",
        pivot_timeframe="D1",
        sr_timeframe="H1",
        reference_price=1.08,
        tolerance_pct=0.2,
        pivot_methods=[
            {
                "method": "classic",
                "levels": {"R1": 1.0801, "R2": 1.0802, "R3": 1.0803},
                "pivot": 1.08,
            }
        ],
        support_resistance_payload={
            "levels": [{"type": "support", "value": 1.08015, "score": 4, "touches": 2}]
        },
        min_source_families=1,
        detail="compact",
    )

    cluster = payload["levels"][0]
    assert cluster["source_families"] == ["pivot_formula", "touch_derived"]
    assert cluster["source_count"] == 2
    assert cluster["record_count"] >= 2
    assert cluster["record_count"] != cluster["source_count"]


def test_identical_pivot_formula_prices_are_deduped_before_scoring():
    payload = build_level_confluence_payload(
        symbol="EURUSD",
        pivot_timeframe="D1",
        sr_timeframe="H1",
        reference_price=1.16000,
        tolerance_pct=0.1,
        pivot_methods=[
            {"method": "classic", "levels": {"R2": 1.16862}},
            {"method": "fibonacci", "levels": {"R3": 1.16862}},
            {"method": "camarilla", "levels": {"PP": 1.16000, "R1": 1.16870}},
        ],
        support_resistance_payload={"levels": []},
        min_source_families=1,
        detail="full",
    )

    cluster = next(
        item
        for item in payload["levels"]
        if item["range"]["low"] <= 1.16862 <= item["range"]["high"]
    )
    pivot_sources = [
        source for source in cluster["sources"] if source.get("family") == "pivot_formula"
    ]
    pivot_prices = {round(float(source["price"]), 10) for source in pivot_sources}
    assert 1.16862 in pivot_prices
    assert len(pivot_sources) == len(pivot_prices)
    assert not any(
        source.get("method") == "camarilla" and source.get("label") == "Camarilla PP"
        for source in cluster["sources"]
    )
    assert cluster["score_components"]["independent_pivot_prices"] == len(pivot_prices)
