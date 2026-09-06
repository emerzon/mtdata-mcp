"""Report confluence prose describes the score-ranked level selection."""

import inspect

import pytest

from mtdata.core import report, report_templates
from mtdata.core.report.requests import ReportGenerateRequest


@pytest.mark.parametrize("levels", [
    [
        {"price": 1.16443, "score": 35.51, "distance_pct": 0.1877, "role": "above"},
        {"price": 1.16215, "score": 22.83, "distance_pct": 0.0034, "role": "above"},
    ],
    [{"price": 1.16443, "score": 35.51, "distance_pct": 0.1877, "role": "above"}],
    [],
])
def test_confluence_narrative_matches_score_order(monkeypatch, levels):
    monkeypatch.setattr(report, "ensure_mt5_connection_or_raise", lambda: None)
    monkeypatch.setattr(
        report_templates, "template_basic",
        lambda *args, **kwargs: {
            "sections": {
                "context": {"price_precision": 5, "last_snapshot": {"close": 1.16146}},
                "confluence": {"reference_price": 1.16146, "levels": levels},
            },
        },
    )
    out = inspect.unwrap(report.report_generate)(request=ReportGenerateRequest(
        symbol="EURUSD", template="basic", detail="full",
        include_sections=["context", "confluence"],
    ))
    narrative = out["summary_structured"]["narrative"]
    assert "Nearest confluence" not in narrative
    if levels:
        assert "Highest-scoring confluence is 1.16443 (above)." in narrative
    else:
        assert "confluence" not in narrative
