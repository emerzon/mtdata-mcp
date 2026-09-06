from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mtdata.core._mcp_tools import shape_public_tool_output
from mtdata.core.forecast_tasks import _serialize_model_handle
from mtdata.forecast.model_store import ModelStore


@pytest.mark.parametrize("detail", ["compact", "standard", "full"])
def test_model_inventory_keeps_training_cutoff_and_expiration(detail):
    handle = SimpleNamespace(
        model_id="theta/EURUSD_H1/id", method="theta", data_scope="EURUSD_H1",
        params_hash="id", created_at=1788091200.0,
        metadata={"training_context": {"training_end_epoch": 1788004800.0},
                  "reuse_request": {"horizon": 12}},
        store_metadata={},
    )
    store = Mock()
    store.describe_model.return_value = {"expires_in_seconds": 43200.0, "expired": False}
    row = _serialize_model_handle(handle, detail=detail, store=store)
    output = shape_public_tool_output({"models": [row]}, tool_name="forecast_models_list", detail=detail)
    model = output["models"][0]
    assert model["training_end"] == "2026-08-29T12:00:00Z"
    assert model["expires_in_days"] == 0.5
    assert model["horizon"] == 12
    store.describe_model.assert_called_once_with(handle, include_size=detail == "full")


def test_model_inventory_omits_unknown_freshness_instead_of_inventing_it():
    output = shape_public_tool_output(
        {"models": [{"model_id": "legacy/id"}]}, tool_name="forecast_models_list", detail="compact"
    )
    assert output["models"] == [{"model_id": "legacy/id"}]


def test_model_freshness_description_avoids_scanning_artifacts(tmp_path, monkeypatch):
    store = ModelStore(root=tmp_path, ttl_seconds=86400)
    handle = store.save(method="theta", data_scope="EURUSD_H1", params_hash="id", artifact_bytes=b"model")
    monkeypatch.setattr("pathlib.Path.rglob", Mock(side_effect=AssertionError("unexpected artifact scan")))
    info = store.describe_model(handle, include_size=False)
    assert 86000 < info["expires_in_seconds"] <= 86400
    assert info["expired"] is False
    assert "file_count" not in info
