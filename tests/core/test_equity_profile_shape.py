import pytest

from mtdata.core.equity_profile import _compose_profile


@pytest.mark.parametrize("extra", [None, "ratings", "peers", "description", "insider", "failed"])
def test_section_paths_do_not_depend_on_other_sections(extra):
    fundamentals = {"success": True, "symbol": "AAPL", "fundamentals": {"price": 120}, "units": {"price": "USD"}}
    payloads = {"fundamentals": fundamentals}
    if extra:
        payloads[extra] = {"success": False, "error": "Unavailable"} if extra == "failed" else {"success": True, extra: []}
    result = _compose_profile(payloads, sections=tuple(payloads), provider="finviz")
    assert result["fundamentals"]["price"] == 120
    assert result["units"] == {"price": "USD"}
    if extra == "failed":
        assert result["partial_failure"] is True


@pytest.mark.parametrize("section,data_key,content", [
    ("ratings", "ratings", [{"rating": "Buy"}]),
    ("peers", "peers", ["MSFT"]),
    ("insider", "items", [{"transaction": "Sale"}]),
    ("description", "description", "Company description"),
])
def test_section_content_and_metadata_have_stable_paths(section, data_key, content):
    payload = {"success": True, "symbol": "AAPL", data_key: content, "pagination": {"returned": 1}, "provider": "finviz"}
    alone = _compose_profile({section: payload}, sections=(section,), provider="finviz")
    combined = _compose_profile({section: payload, "fundamentals": {"fundamentals": {"price": 120}}}, sections=(section, "summary"), provider="finviz")
    key = "text" if section == "description" else "items"
    assert alone[section] == combined[section]
    assert alone[section][key] == content
    assert alone[section]["pagination"] == {"returned": 1}
    assert alone[section]["provider"] == "finviz"
    assert section not in alone[section]
