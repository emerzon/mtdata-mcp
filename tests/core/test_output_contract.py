from unittest.mock import patch

import pytest

from mtdata.core.output_contract import (
    _coerce_optional_verbose_flag,
    apply_output_verbosity,
    attach_collection_contract,
    ensure_common_meta,
    normalize_output_detail,
    normalize_output_extras,
    normalize_output_verbosity_detail,
    output_extras_shape_detail,
    resolve_output_contract,
)


def test_normalize_output_detail_preserves_summary_and_standard() -> None:
    assert normalize_output_detail(" Summary ") == "summary"
    assert normalize_output_detail(" Standard ") == "standard"


def test_normalize_output_detail_rejects_removed_summary_only_alias() -> None:
    with pytest.raises(ValueError, match="Invalid detail"):
        normalize_output_detail(" Summary_Only ")


def test_normalize_output_detail_normalizes_custom_alias_mapping_keys_and_values() -> None:
    assert (
        normalize_output_detail(
            " brief ",
            aliases={" Brief ": " Summary "},
        )
        == "summary"
    )


def test_normalize_output_verbosity_detail_is_strict_compact_or_full() -> None:
    assert normalize_output_verbosity_detail(" summary ") == "compact"
    assert normalize_output_verbosity_detail(" standard ") == "compact"
    assert normalize_output_verbosity_detail(" FULL ") == "full"


def test_resolve_output_contract_tracks_full_detail_only() -> None:
    assert resolve_output_contract({"detail": " summary "}).verbose is False
    assert resolve_output_contract({"detail": " full "}).verbose is True


def test_resolve_output_contract_maps_full_detail_to_full_state() -> None:
    state = resolve_output_contract({"detail": "full"})

    assert state.detail == "full"
    assert state.shape_detail == "full"
    assert state.verbose is True


def test_resolve_output_contract_preserves_summary_detail() -> None:
    state = resolve_output_contract(
        {"detail": " summary "},
    )

    assert state.detail == "summary"
    assert state.shape_detail == "compact"
    assert state.verbose is False


def test_normalize_output_extras_accepts_comma_lists_and_full_aliases() -> None:
    assert normalize_output_extras("metadata, diagnostics") == (
        "metadata",
        "diagnostics",
    )
    assert set(normalize_output_extras("all")) >= {"metadata", "diagnostics", "raw"}


def test_normalize_output_extras_accepts_bool_like_full_shortcut() -> None:
    np = pytest.importorskip("numpy")

    full_extras = normalize_output_extras(True)

    assert normalize_output_extras(np.bool_(True)) == full_extras
    assert normalize_output_extras("true") == full_extras
    assert normalize_output_extras(np.bool_(False)) == ()
    assert _coerce_optional_verbose_flag(np.bool_(True)) is True


def test_normalize_output_extras_rejects_legacy_detail_assignments() -> None:
    with pytest.raises(ValueError, match="Invalid extras value"):
        normalize_output_extras("detail=full")
    with pytest.raises(ValueError, match="Invalid extras value"):
        normalize_output_extras("detail=compact")
    with pytest.raises(ValueError, match="Invalid extras value"):
        normalize_output_extras(["metadata", "verbose=true"])


def test_output_extras_shape_detail_is_compact_by_default_and_full_when_requested() -> None:
    assert output_extras_shape_detail(None) == "compact"
    assert output_extras_shape_detail("") == "compact"
    assert output_extras_shape_detail("metadata") == "full"
    with pytest.raises(ValueError, match="Invalid extras value"):
        output_extras_shape_detail("detail=full")


def test_resolve_output_contract_prefers_explicit_verbose_when_detail_is_none() -> None:
    state = resolve_output_contract(
        {"detail": "summary"},
        detail=None,
        verbose=True,
    )

    assert state.detail == "full"
    assert state.shape_detail == "full"
    assert state.verbose is True


def test_ensure_common_meta_adds_tool_and_runtime_timezone() -> None:
    timezone_meta = {"used": {"tz": "UTC"}}
    with patch(
        "mtdata.core.output_contract.build_runtime_timezone_meta",
        return_value=timezone_meta,
    ):
        out = ensure_common_meta({"success": True}, tool_name="market_ticker")

    assert out["meta"]["tool"] == "market_ticker"
    assert out["meta"]["runtime"]["timezone"] == timezone_meta


def test_ensure_common_meta_preserves_existing_tool_and_timezone() -> None:
    existing_timezone = {"client": {"tz": "US/Central"}}
    with patch("mtdata.core.output_contract.build_runtime_timezone_meta") as build_meta:
        out = ensure_common_meta(
            {
                "meta": {
                    "tool": "existing_tool",
                    "runtime": {"timezone": existing_timezone},
                }
            },
            tool_name="market_ticker",
        )

    build_meta.assert_not_called()
    assert out["meta"]["tool"] == "existing_tool"
    assert out["meta"]["runtime"]["timezone"] == existing_timezone


def test_attach_collection_contract_avoids_duplicate_rows_for_legacy_data() -> None:
    rows = [{"symbol": "EURUSD"}]

    out = attach_collection_contract(
        {"success": True, "data": rows},
        collection_kind="table",
        rows=rows,
    )

    assert out["data"] == rows
    assert out["collection_kind"] == "table"
    assert out["collection_contract_version"] == "collection.v1"
    assert out["canonical_source"] == "data"
    assert out["row_key"] == "data"
    assert "rows" not in out


def test_attach_collection_contract_short_circuits_identical_collection_equality() -> None:
    class IdentityOnlyRows(list):
        def __eq__(self, other):
            raise AssertionError("identity match should not invoke collection equality")

    rows = IdentityOnlyRows([{"symbol": "EURUSD"}])

    out = attach_collection_contract(
        {"success": True, "data": rows},
        collection_kind="table",
        rows=rows,
    )

    assert out["canonical_source"] == "data"
    assert "rows" not in out


def test_compact_output_reuses_unchanged_collection_branches() -> None:
    rows = [{"time": 1, "close": 1.25}]
    payload = {
        "success": True,
        "data": rows,
        "meta": {"diagnostics": {"latency_ms": 1.0}},
    }

    out = apply_output_verbosity(payload, detail="compact")

    assert out is not payload
    assert out["data"] is rows
    assert "meta" not in out
    assert "meta" in payload


def test_compact_output_copies_only_nested_branches_that_change() -> None:
    unchanged_row = {"time": 1, "close": 1.25}
    changed_row = {"time": 2, "close": 1.26, "debug": {"source": "test"}}
    rows = [unchanged_row, changed_row]

    out = apply_output_verbosity({"data": rows}, detail="compact")

    assert out["data"] is not rows
    assert out["data"][0] is unchanged_row
    assert out["data"][1] is not changed_row
    assert "debug" not in out["data"][1]
    assert "debug" in changed_row


def test_attach_collection_contract_can_omit_contract_metadata() -> None:
    rows = [{"symbol": "EURUSD"}]

    out = attach_collection_contract(
        {"success": True, "data": rows},
        collection_kind="table",
        rows=rows,
        include_contract_meta=False,
    )

    assert out["data"] == rows
    assert out["row_key"] == "data"
    assert "rows" not in out
    assert "collection_kind" not in out
    assert "collection_contract_version" not in out


def test_attach_collection_contract_keeps_compact_alias_when_no_legacy_collection() -> None:
    rows = [{"symbol": "EURUSD"}]

    out = attach_collection_contract(
        {"success": True, "data": {"table": {"columns": ["symbol"]}}},
        collection_kind="table",
        rows=rows,
        include_contract_meta=False,
    )

    assert out["rows"] == rows


def test_attach_collection_contract_avoids_duplicate_groups_for_results() -> None:
    groups = {"lowest_spread": {"data": [["EURUSD"]]}}

    out = attach_collection_contract(
        {"success": True, "results": groups},
        collection_kind="groups",
        groups=groups,
    )

    assert out["results"] == groups
    assert out["collection_kind"] == "groups"
    assert out["collection_contract_version"] == "collection.v1"
    assert out["canonical_source"] == "results"
    assert "groups" not in out


def test_attach_collection_contract_preserves_error_payloads() -> None:
    payload = {"error": "failed"}

    assert attach_collection_contract(payload, collection_kind="table", rows=[]) == payload
