import json

from mtdata.core.options import _apply_options_detail, _compact_option_contract


def _row(strike):
    return {"strike": strike, "bid": 1, "ask": 2, "contract_as_of": "2026-09-05T12:00:00Z",
            "quote_usable_for_live_analysis": False, "quote_freshness": "unknown", "greeks_available": False,
            "greeks_unavailable_reason": "provider_does_not_supply_greeks",
            "quote_freshness_reason": "provider_quote_timestamp_unavailable", "quote_usability_reason": "quote_timestamp_unavailable"}


def test_compact_shares_repeated_reasons_without_losing_row_safety():
    rows = [_row(strike) for strike in range(20)]
    payload = {"success": True, "options": rows, "count": 20, "option_contract_count": 20}
    compact = _apply_options_detail(payload, detail="compact", kind="chain")
    previous = {**compact, "options": [_compact_option_contract(row, include_freshness=True) for row in rows]}
    previous.pop("shared_contract_status")
    assert len(json.dumps(compact)) < len(json.dumps(previous)) * 0.75
    for source, row in zip(rows, compact["options"]):
        assert row["quote_usable_for_live_analysis"] is False
        assert row["greeks_available"] is False
        assert {**compact["shared_contract_status"], **row} == source
    assert _apply_options_detail(payload, detail="full", kind="chain")["options"] == rows


def test_distinct_reasons_remain_on_each_contract():
    rows = [_row(100), _row(110)]
    rows[1]["quote_usability_reason"] = "quote_one_sided"
    compact = _apply_options_detail({"success": True, "options": rows}, detail="compact", kind="chain")
    assert "quote_usability_reason" not in compact["shared_contract_status"]
    assert compact["options"][1]["quote_usability_reason"] == "quote_one_sided"
