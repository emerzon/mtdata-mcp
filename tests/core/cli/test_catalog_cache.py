from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

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


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("CLIENT_TZ", "America/New_York"),
        ("MT5_CLIENT_TZ", "Europe/London"),
        ("MT5_SERVER_TZ", "Europe/Athens"),
        ("MT5_TIME_OFFSET_MINUTES", "120"),
        ("MTDATA_OUTPUT_FORMAT", "json"),
        ("TZ", "Pacific/Auckland"),
    ],
)
def test_catalog_fingerprint_invalidates_for_provenance_environment(
    monkeypatch,
    tmp_path,
    name,
    value,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        catalog_cache,
        "_installed_distributions_fingerprint",
        lambda: "d" * 64,
    )
    monkeypatch.setattr(
        catalog_cache,
        "_runtime_context",
        lambda: {"timezone": "stable"},
    )

    before = catalog_cache.catalog_cache_fingerprint()
    monkeypatch.setenv(name, value)
    after = catalog_cache.catalog_cache_fingerprint()

    assert after != before


def test_catalog_fingerprint_invalidates_for_runtime_timezone(
    monkeypatch,
    tmp_path,
) -> None:
    state = {"timezone": "UTC"}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        catalog_cache,
        "_installed_distributions_fingerprint",
        lambda: "d" * 64,
    )
    monkeypatch.setattr(
        catalog_cache,
        "_runtime_context",
        lambda: dict(state),
    )

    before = catalog_cache.catalog_cache_fingerprint()
    state["timezone"] = "America/Chicago"
    after = catalog_cache.catalog_cache_fingerprint()

    assert after != before


def test_catalog_fingerprint_invalidates_when_env_file_appears(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        catalog_cache,
        "_installed_distributions_fingerprint",
        lambda: "d" * 64,
    )
    monkeypatch.setattr(catalog_cache, "_runtime_context", lambda: {})

    before = catalog_cache.catalog_cache_fingerprint()
    (tmp_path / ".env").write_text("CLIENT_TZ=UTC\n", encoding="utf-8")
    after = catalog_cache.catalog_cache_fingerprint()

    assert after != before


def test_distribution_manifest_is_reused_and_invalidated(
    monkeypatch,
    tmp_path,
) -> None:
    dist_info = tmp_path / "demo-1.0.dist-info"
    dist_info.mkdir()
    metadata_path = dist_info / "METADATA"
    metadata_path.write_text("Name: demo\nVersion: 1.0\n", encoding="utf-8")
    manifest_path = tmp_path / catalog_cache._DISTRIBUTION_MANIFEST_NAME
    manifest_path.write_text("{corrupt", encoding="utf-8")
    reads: list[str] = []

    class FakeDistribution:
        _path = dist_info

        def __init__(self, version: str, *, readable: bool = True) -> None:
            self._version = version
            self._readable = readable

        @property
        def metadata(self):
            if not self._readable:
                raise AssertionError("cached metadata should not be reparsed")
            reads.append(self._version)
            return {"Name": "demo", "Version": self._version}

        @property
        def version(self):
            return self._version

    monkeypatch.setattr(catalog_cache, "_cache_directory", lambda: tmp_path)
    current = {"distribution": FakeDistribution("1.0")}
    monkeypatch.setattr(
        catalog_cache.metadata,
        "distributions",
        lambda: (current["distribution"],),
    )

    first = catalog_cache._installed_distributions_fingerprint()
    current["distribution"] = FakeDistribution("1.0", readable=False)
    second = catalog_cache._installed_distributions_fingerprint()

    assert first == second
    assert reads == ["1.0"]
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["kind"] == (
        catalog_cache._DISTRIBUTION_MANIFEST_KIND
    )

    metadata_path.write_text(
        "Name: demo\nVersion: 2.0\nSummary: changed\n",
        encoding="utf-8",
    )
    current["distribution"] = FakeDistribution("2.0")

    assert catalog_cache._installed_distributions_fingerprint() != first
    assert reads == ["1.0", "2.0"]


def _write_owned_entry(
    directory,
    *,
    index: int,
    size: int,
    mtime: float,
):
    path = directory / f"catalog-v2-tools_list-{index:024x}.json"
    path.write_bytes(b"x" * size)
    os.utime(path, (mtime, mtime))
    return path


def test_catalog_cache_prunes_expired_entries_only(
    monkeypatch,
    tmp_path,
) -> None:
    now = 10_000.0
    monkeypatch.setattr(catalog_cache, "_cache_directory", lambda: tmp_path)
    monkeypatch.setattr(catalog_cache, "CATALOG_CACHE_MAX_AGE_SECONDS", 100)
    monkeypatch.setattr(catalog_cache, "CATALOG_CACHE_MAX_ENTRIES", 10)
    monkeypatch.setattr(catalog_cache, "CATALOG_CACHE_MAX_BYTES", 10_000)
    expired = _write_owned_entry(
        tmp_path,
        index=1,
        size=10,
        mtime=now - 101,
    )
    fresh = _write_owned_entry(
        tmp_path,
        index=2,
        size=10,
        mtime=now - 99,
    )
    unrelated = tmp_path / "application-state.json"
    unrelated.write_text("{}", encoding="utf-8")
    os.utime(unrelated, (now - 1_000, now - 1_000))

    assert catalog_cache._prune_catalog_cache(now=now) == 1
    assert not expired.exists()
    assert fresh.exists()
    assert unrelated.exists()


def test_catalog_cache_prunes_to_newest_count(
    monkeypatch,
    tmp_path,
) -> None:
    now = 20_000.0
    monkeypatch.setattr(catalog_cache, "_cache_directory", lambda: tmp_path)
    monkeypatch.setattr(catalog_cache, "CATALOG_CACHE_MAX_AGE_SECONDS", 10_000)
    monkeypatch.setattr(catalog_cache, "CATALOG_CACHE_MAX_ENTRIES", 2)
    monkeypatch.setattr(catalog_cache, "CATALOG_CACHE_MAX_BYTES", 10_000)
    paths = [
        _write_owned_entry(
            tmp_path,
            index=index,
            size=10,
            mtime=now - index,
        )
        for index in range(5)
    ]

    assert catalog_cache._prune_catalog_cache(now=now) == 3
    assert [path.exists() for path in paths] == [
        True,
        True,
        False,
        False,
        False,
    ]


def test_catalog_cache_prunes_to_newest_byte_prefix(
    monkeypatch,
    tmp_path,
) -> None:
    now = 30_000.0
    monkeypatch.setattr(catalog_cache, "_cache_directory", lambda: tmp_path)
    monkeypatch.setattr(catalog_cache, "CATALOG_CACHE_MAX_AGE_SECONDS", 10_000)
    monkeypatch.setattr(catalog_cache, "CATALOG_CACHE_MAX_ENTRIES", 10)
    monkeypatch.setattr(catalog_cache, "CATALOG_CACHE_MAX_BYTES", 10)
    newest = _write_owned_entry(tmp_path, index=1, size=6, mtime=now)
    older = _write_owned_entry(tmp_path, index=2, size=6, mtime=now - 1)
    oldest = _write_owned_entry(tmp_path, index=3, size=2, mtime=now - 2)

    assert catalog_cache._prune_catalog_cache(now=now) == 2
    assert newest.exists()
    assert not older.exists()
    assert not oldest.exists()


def test_catalog_cache_prunes_identified_legacy_entries_only(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(catalog_cache, "_cache_directory", lambda: tmp_path)
    monkeypatch.setattr(catalog_cache, "CATALOG_CACHE_MAX_ENTRIES", 0)
    valid_legacy = tmp_path / f"removed_command-{'a' * 24}.json"
    valid_legacy.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fingerprint": "b" * 64,
                "created_at_epoch": 1.0,
                "output": "cached output",
            }
        ),
        encoding="utf-8",
    )
    corrupt_lookalike = tmp_path / f"notes-{'c' * 24}.json"
    corrupt_lookalike.write_text("{not-cache", encoding="utf-8")
    unrelated = tmp_path / "notes.json"
    unrelated.write_text("keep", encoding="utf-8")

    assert catalog_cache._prune_catalog_cache() == 1
    assert not valid_legacy.exists()
    assert corrupt_lookalike.exists()
    assert unrelated.exists()


def test_corrupt_cache_entry_misses_then_round_trips(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(catalog_cache, "_cache_directory", lambda: tmp_path)
    argv = ["tools_list", "--json"]
    path = catalog_cache._cache_path(
        command="tools_list",
        argv=argv,
        program="mtdata-cli",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{partial", encoding="utf-8")

    assert (
        catalog_cache.load_catalog_output(
            command="tools_list",
            argv=argv,
            program="mtdata-cli",
            fingerprint="stable",
        )
        is None
    )
    assert catalog_cache.store_catalog_output(
        command="tools_list",
        argv=argv,
        program="mtdata-cli",
        output='{"catalog_source":"rebuilt","tools":[]}',
        fingerprint="stable",
    )
    cached = catalog_cache.load_catalog_output(
        command="tools_list",
        argv=argv,
        program="mtdata-cli",
        fingerprint="stable",
    )

    assert json.loads(cached or "{}")["catalog_source"] == "cached"


def test_concurrent_cache_writes_remain_atomic(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(catalog_cache, "_cache_directory", lambda: tmp_path)
    argv = ["tools_list", "--json"]
    outputs = [
        json.dumps(
            {
                "catalog_source": "rebuilt",
                "generation": generation,
                "tools": ["x" * 10_000],
            }
        )
        for generation in range(6)
    ]

    def store(output: str) -> bool:
        return catalog_cache.store_catalog_output(
            command="tools_list",
            argv=argv,
            program="mtdata-cli",
            output=output,
            fingerprint="concurrent",
        )

    with ThreadPoolExecutor(max_workers=6) as executor:
        stored = list(executor.map(store, outputs))

    cached = catalog_cache.load_catalog_output(
        command="tools_list",
        argv=argv,
        program="mtdata-cli",
        fingerprint="concurrent",
    )
    payload = json.loads(cached or "{}")

    assert all(stored)
    assert payload["generation"] in range(6)
    assert payload["catalog_source"] == "cached"
    assert not list(tmp_path.glob("*.tmp"))


def test_abandoned_prune_lock_file_does_not_block(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(catalog_cache, "_cache_directory", lambda: tmp_path)
    lock_path = tmp_path / catalog_cache._PRUNE_LOCK_NAME
    lock_path.write_text("abandoned", encoding="ascii")

    assert catalog_cache.store_catalog_output(
        command="tools_list",
        argv=["tools_list", "--json"],
        program="mtdata-cli",
        output='{"catalog_source":"rebuilt","tools":[]}',
        fingerprint="stable",
    )
    assert lock_path.exists()
