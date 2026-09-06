from mtdata.core._mcp_tools import _select_output_fields


def test_partial_projection_keeps_warning_objects_and_source_warning():
    warning = {"code": "stale_quote", "scope": "quote", "message": "Quote is stale."}
    result = _select_output_fields({"success": True, "value": 1, "warnings": [warning]}, "value,missing")
    assert result["success"] is True
    assert result["warnings"][0] == warning
    assert result["warnings"][1]["code"] == "output_fields_partial"
    assert result["warnings"][1]["details"]["unresolved_output_fields"] == ["missing"]
    assert all({"code", "scope", "message"} <= item.keys() for item in result["warnings"])
