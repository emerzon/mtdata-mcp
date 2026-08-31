from mtdata.core.output_metadata import (
    FreshnessObservation,
    OutputMetadata,
    OutputWarning,
    SourceContext,
    TimeContext,
    append_output_warning,
)


def test_source_context_has_compact_and_full_renderings() -> None:
    source = SourceContext.from_payload(
        {
            "source": {
                "provider": "mt5",
                "broker_company": "Broker Co",
                "server": "Broker-Demo",
                "source_context_id": "abc123",
                "context_available": True,
            }
        }
    )

    assert source is not None
    assert source.compact() == {"provider": "mt5"}
    assert source.full() == {
        "provider": "mt5",
        "broker_company": "Broker Co",
        "server": "Broker-Demo",
        "context_id": "abc123",
        "context_available": True,
    }


def test_time_context_consolidates_legacy_root_fields() -> None:
    context = TimeContext.from_payload(
        {
            "as_of": "2026-08-29T10:00:00Z",
            "data_as_of": "2026-08-29T09:45:00Z",
            "as_of_basis": "retrieval_time",
            "data_as_of_basis": "completed_bar_close",
            "timezone": "UTC",
            "timestamp_format": "iso_utc",
        }
    )

    assert context is not None
    assert context.to_dict()["retrieved_at"] == "2026-08-29T10:00:00Z"
    assert context.to_dict()["data_basis"] == "completed_bar_close"


def test_freshness_observation_omits_nominal_warning() -> None:
    observation = FreshnessObservation.from_payload(
        {
            "data_age_seconds": 4,
            "data_stale": False,
            "freshness_state": "live",
            "usable_for_live_trading": True,
        },
        scope="quote",
    )

    assert observation is not None
    assert observation.nominal is True
    assert observation.to_warning() is None


def test_freshness_observation_builds_one_stale_warning() -> None:
    observation = FreshnessObservation.from_payload(
        {
            "data_as_of": "2026-08-29T09:00:00Z",
            "data_age_seconds": 3600,
            "data_stale": True,
            "stale_after_seconds": 900,
            "freshness_reason": "stale_age",
        },
        scope="completed_bar",
    )

    assert observation is not None
    assert observation.to_warning() == OutputWarning(
        code="data_stale",
        scope="completed_bar",
        message="The latest data is outside the expected freshness window.",
        context={
            "data_as_of": "2026-08-29T09:00:00Z",
            "age_seconds": 3600,
        },
    )


def test_freshness_observation_maps_verified_history_to_fresh() -> None:
    observation = FreshnessObservation.from_payload(
        {
            "data_as_of": "2026-08-30T22:00:00Z",
            "data_age_seconds": 567,
            "stale_after_seconds": 2700,
            "freshness_basis": "last_completed_bar_close",
            "history_policy_ok": True,
        },
        scope="forecast_generate",
    )

    assert observation is not None
    assert observation.status == "fresh"
    assert observation.nominal is True
    assert observation.to_warning() is None


def test_freshness_observation_positive_policy_overrides_generic_unknown_state() -> None:
    observation = FreshnessObservation.from_payload(
        {
            "freshness_state": "unknown",
            "history_policy_ok": True,
        },
        scope="market_scan",
    )

    assert observation is not None
    assert observation.status == "fresh"
    assert observation.to_warning() is None


def test_freshness_observation_keeps_failed_or_missing_verification_non_nominal() -> None:
    stale = FreshnessObservation.from_payload(
        {"history_policy_ok": False},
        scope="forecast_generate",
    )
    unknown = FreshnessObservation.from_payload(
        {"freshness_state": "unknown"},
        scope="forecast_generate",
    )

    assert stale is not None
    assert stale.status == "stale"
    assert stale.to_warning() is not None
    assert stale.to_warning().code == "data_stale"
    assert unknown is not None
    assert unknown.status == "unknown"
    assert unknown.to_warning() is not None
    assert unknown.to_warning().code == "freshness_unverified"


def test_append_output_warning_deduplicates_by_condition() -> None:
    payload = {}
    warning = OutputWarning(
        code="data_stale",
        scope="quote",
        message="Quote is stale.",
    )

    append_output_warning(payload, warning)
    append_output_warning(payload, warning)

    assert payload["warnings"] == [warning.to_dict()]


def test_output_metadata_groups_canonical_sections() -> None:
    source = SourceContext(provider="mt5")
    time = TimeContext(data_as_of="2026-08-29T09:45:00Z")
    freshness = FreshnessObservation(scope="bar", status="fresh")

    rendered = OutputMetadata(
        source=source,
        time=time,
        freshness=(freshness,),
        processing={"pipeline": ["fetch", "indicators"]},
    ).to_dict()

    assert rendered == {
        "source": {"provider": "mt5"},
        "time": {"data_as_of": "2026-08-29T09:45:00Z"},
        "freshness": [{"scope": "bar", "status": "fresh"}],
        "processing": {"pipeline": ["fetch", "indicators"]},
    }
