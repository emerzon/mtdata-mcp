from __future__ import annotations

import json

from mtdata.core.cli import catalog_cache


def test_catalog_cache_round_trip_marks_replayed_source_cached(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(catalog_cache, "_cache_directory", lambda: tmp_path)
    monkeypatch.setattr(
        catalog_cache,
        "catalog_cache_fingerprint",
        lambda: "source-and-versions-a",
    )
    argv = ["forecast_list_library_models", "sktime", "--json"]
    output = json.dumps(
        {"success": True, "catalog_source": "rebuilt", "models": ["Theta"]},
        indent=2,
    )

    assert catalog_cache.store_catalog_output(
        command="forecast_list_library_models",
        argv=argv,
        program="mtdata-cli",
        output=output,
    )
    cached = catalog_cache.load_catalog_output(
        command="forecast_list_library_models",
        argv=argv,
        program="mtdata-cli",
    )

    assert cached is not None
    assert json.loads(cached)["catalog_source"] == "cached"
    assert json.loads(cached)["models"] == ["Theta"]


def test_catalog_cache_invalidates_when_source_or_versions_change(
    monkeypatch,
    tmp_path,
) -> None:
    state = {"fingerprint": "before"}
    monkeypatch.setattr(catalog_cache, "_cache_directory", lambda: tmp_path)
    monkeypatch.setattr(
        catalog_cache,
        "catalog_cache_fingerprint",
        lambda: state["fingerprint"],
    )
    argv = ["tools_list", "--json"]
    assert catalog_cache.store_catalog_output(
        command="tools_list",
        argv=argv,
        program="mtdata-cli",
        output='{"success":true,"catalog_source":"rebuilt"}\n',
    )

    state["fingerprint"] = "after"

    assert (
        catalog_cache.load_catalog_output(
            command="tools_list",
            argv=argv,
            program="mtdata-cli",
        )
        is None
    )


def test_catalog_cache_miss_still_pins_pre_bootstrap_fingerprint(
    monkeypatch,
    tmp_path,
) -> None:
    fingerprints = []
    monkeypatch.setattr(catalog_cache, "_cache_directory", lambda: tmp_path)
    monkeypatch.setattr(
        catalog_cache,
        "catalog_cache_fingerprint",
        lambda: fingerprints.append("checked") or "cold-context",
    )

    assert (
        catalog_cache.load_catalog_output(
            command="tools_list",
            argv=["tools_list", "--json"],
            program="mtdata-cli",
        )
        is None
    )
    assert fingerprints == ["checked"]


def test_catalog_and_help_invocations_are_cacheable() -> None:
    assert catalog_cache.is_cacheable_catalog_invocation(
        "forecast-list-methods",
        ["forecast-list-methods", "--json"],
    )
    assert catalog_cache.is_cacheable_catalog_invocation(
        "forecast_list_methods",
        ["forecast_list_methods", "--help"],
    )
    assert catalog_cache.is_cacheable_catalog_invocation(
        "market_ticker",
        ["market_ticker", "--help"],
    )
    assert not catalog_cache.is_cacheable_catalog_invocation(
        "market_ticker",
        ["market_ticker", "EURUSD"],
    )
