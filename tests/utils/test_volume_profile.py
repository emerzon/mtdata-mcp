from typing import get_args

from mtdata.utils.volume_profile import (
    VolumeProfileConfig,
    VolumeProfilePriceSourceLiteral,
    VolumeProfileVolumeSourceLiteral,
    _bucket_prices,
    annotate_level_confluence,
    compute_volume_profile,
)


def test_volume_profile_source_literals_define_validation_options() -> None:
    assert set(get_args(VolumeProfilePriceSourceLiteral)) == {
        "mid",
        "last",
        "bid",
        "ask",
    }
    assert set(get_args(VolumeProfileVolumeSourceLiteral)) == {
        "auto",
        "real_volume",
        "tick_volume",
        "volume_real",
        "volume",
        "tick_count",
    }


def test_bucket_edges_use_same_decimal_grid_as_bucket_index() -> None:
    buckets = _bucket_prices([0.11, 0.21], [1.0, 1.0], 0.01)

    assert buckets[10]["low"] == 0.21
    assert buckets[10]["high"] == 0.22


def test_compute_volume_profile_uses_mid_and_tick_count_fallback():
    rows = [
        {"bid": 1.09995, "ask": 1.10005, "real_volume": 0, "tick_volume": 0},
        {"bid": 1.10005, "ask": 1.10015, "real_volume": 0, "tick_volume": 0},
        {"bid": 1.10005, "ask": 1.10015, "real_volume": 0, "tick_volume": 0},
        {"bid": 1.10095, "ask": 1.10105, "real_volume": 0, "tick_volume": 0},
    ]

    result = compute_volume_profile(
        rows,
        VolumeProfileConfig(
            price_source="mid",
            bucket_size=0.0001,
            value_area_fraction=0.70,
            price_digits=5,
        ),
    )

    assert result["success"] is True
    assert result["volume_kind"] == "tick_count"
    assert result["poc"]["price"] == 1.10015
    assert result["poc"]["volume"] == 2.0
    assert (
        result["poc"]["volume_share"]
        == result["poc"]["volume"] / result["total_volume"]
    )
    assert (
        result["vah"]["volume_share"]
        == result["vah"]["volume"] / result["total_volume"]
    )
    assert (
        result["val"]["volume_share"]
        == result["val"]["volume"] / result["total_volume"]
    )
    assert result["val"]["price"] <= result["poc"]["price"] <= result["vah"]["price"]


def test_compute_volume_profile_expands_value_area_ties_deterministically():
    rows = [
        {"last": 10.0, "real_volume": 1},
        {"last": 11.0, "real_volume": 4},
        {"last": 12.0, "real_volume": 10},
        {"last": 13.0, "real_volume": 4},
        {"last": 14.0, "real_volume": 1},
    ]

    result = compute_volume_profile(
        rows,
        VolumeProfileConfig(
            price_source="last",
            bucket_size=1.0,
            value_area_fraction=0.70,
            price_digits=0,
            reference_price=12.0,
        ),
    )

    assert result["volume_kind"] == "real_volume"
    assert result["poc"]["price"] == 12.0
    assert result["value_area"]["bucket_indexes"] == [1, 2, 3]
    assert result["value_area"]["volume"] == 18.0
    assert result["value_area"]["volume_share"] == 0.9
    level_shares = {row["level"]: row["volume_share"] for row in result["levels"]}
    assert level_shares == {"POC": 0.5, "VAH": 0.2, "VAL": 0.2}


def test_compute_volume_profile_expands_across_empty_price_buckets():
    rows = [
        {"last": 10.0, "real_volume": 10},
        {"last": 12.0, "real_volume": 5},
    ]

    result = compute_volume_profile(
        rows,
        VolumeProfileConfig(
            price_source="last",
            bucket_size=1.0,
            value_area_fraction=0.90,
            price_digits=0,
            reference_price=10.0,
        ),
    )

    assert result["value_area"]["bucket_indexes"] == [0, 1, 2]
    assert result["value_area"]["populated_bucket_indexes"] == [0, 2]
    assert result["value_area"]["volume_share"] == 1.0
    assert result["val"]["price"] == 10.0
    assert result["vah"]["price"] == 13.0


def test_compute_volume_profile_drops_missing_last_prices():
    rows = [
        {"bid": 1.0, "ask": 1.2, "last": 0, "tick_volume": 10},
        {"bid": 1.1, "ask": 1.3, "last": None, "tick_volume": 10},
    ]

    result = compute_volume_profile(
        rows,
        VolumeProfileConfig(price_source="last", bucket_size=0.1),
    )

    assert result["code"] == "volume_profile_no_valid_prices"
    assert result["diagnostics"]["dropped_price_rows"] == 2


def test_compute_volume_profile_rejects_invalid_bucket_size():
    result = compute_volume_profile(
        [{"last": 1.0, "tick_volume": 1}],
        VolumeProfileConfig(price_source="last", bucket_size=0),
    )

    assert result["code"] == "volume_profile_invalid_config"
    assert "bucket_size" in result["error"]


def test_compute_volume_profile_rejects_conflicting_bucket_controls():
    result = compute_volume_profile(
        [{"last": 1.0, "tick_volume": 1}],
        VolumeProfileConfig(
            price_source="last",
            bucket_size=0.1,
            bucket_points=10,
            bucket_count=20,
        ),
    )

    assert result["code"] == "volume_profile_invalid_config"
    assert "mutually exclusive" in result["error"]


def test_compute_volume_profile_respects_explicit_volume_source():
    rows = [
        {"last": 1.0, "real_volume": 0, "tick_volume": 5},
        {"last": 2.0, "real_volume": 0, "tick_volume": 5},
    ]

    result = compute_volume_profile(
        rows,
        VolumeProfileConfig(price_source="last", volume_source="real_volume"),
    )

    assert result["code"] == "volume_profile_no_positive_volume"
    assert result["diagnostics"]["dropped_volume_rows"] == 2


def test_compute_volume_profile_counts_snapshot_volume_only_on_trade_events():
    rows = [
        {"last": 1.0, "volume": 5.0, "volume_real": 0.0, "flags": 24},
        {"last": 1.1, "volume": 5.0, "volume_real": 0.0, "flags": 6},
        {"last": 1.2, "volume": 5.0, "volume_real": 0.0, "flags": 6},
    ]

    result = compute_volume_profile(
        rows,
        VolumeProfileConfig(price_source="last", volume_source="auto"),
    )

    assert result["success"] is True
    assert result["value_area_pct"] == 70.0
    assert result["volume_kind"] == "volume"
    assert result["total_volume"] == 5.0


def test_compute_volume_profile_caps_tiny_explicit_buckets():
    rows = [{"last": float(price), "tick_volume": 1} for price in range(100)]

    result = compute_volume_profile(
        rows,
        VolumeProfileConfig(
            price_source="last",
            bucket_size=0.01,
            max_buckets=10,
        ),
    )

    assert result["success"] is True
    assert result["diagnostics"]["bucket_count"] <= 10
    assert result["diagnostics"]["max_buckets_reached"] is True
    assert result["requested_bucket_size"] == 0.01
    assert result["effective_bucket_size"] == result["bucket_size"]
    assert result["effective_bucket_size"] > result["requested_bucket_size"]
    assert any("requested_bucket_size" in warning for warning in result["warnings"])


def test_compute_volume_profile_caps_sparse_price_grid():
    result = compute_volume_profile(
        [
            {"last": 1.0, "tick_volume": 1},
            {"last": 100.0, "tick_volume": 1},
        ],
        VolumeProfileConfig(
            price_source="last",
            bucket_size=0.01,
            max_buckets=10,
        ),
    )

    assert result["success"] is True
    assert result["diagnostics"]["populated_bucket_count"] == 2
    assert result["diagnostics"]["bucket_count"] <= 10
    assert result["diagnostics"]["max_buckets_reached"] is True
    assert len(result["value_area"]["bucket_indexes"]) <= 10


def test_compute_volume_profile_discloses_explicit_bucket_count_cap():
    rows = [{"last": float(price), "tick_volume": 1} for price in range(100)]

    result = compute_volume_profile(
        rows,
        VolumeProfileConfig(
            price_source="last",
            bucket_count=500,
            max_buckets=20,
        ),
    )

    assert result["success"] is True
    assert result["diagnostics"]["bucket_count_requested"] == 500
    assert result["diagnostics"]["max_buckets_reached"] is True
    assert "was capped" in result["warning"]


def test_compute_volume_profile_defaults_to_one_hundred_buckets_when_possible():
    rows = [{"last": float(price), "tick_volume": 1} for price in range(100)]

    result = compute_volume_profile(
        rows,
        VolumeProfileConfig(price_source="last", max_buckets=120),
    )

    assert result["success"] is True
    assert result["bucket_size"] < 1.0
    assert result["diagnostics"]["bucket_count"] > 80


def test_annotate_level_confluence_uses_symbol_points_tolerance():
    rows = [{"level_price": 1.0850}]
    levels = [{"level": "POC", "type": "volume_poc", "price": 1.0855}]

    enriched = annotate_level_confluence(
        rows,
        levels,
        tolerance_points=6,
        price_point=0.0001,
    )

    confluence = enriched[0]["volume_profile_confluence"]
    assert confluence["level"] == "POC"
    assert confluence["distance_points"] == 5
    assert confluence["within_tolerance"] is True
