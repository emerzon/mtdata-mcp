import json
from datetime import datetime, timezone

import pytest

from mtdata.core.cli import _print_missing_command_error
from mtdata.core.cli.api import _write_shell_batch_record


def test_shell_json_record_sanitizes_nested_nonfinite_numbers(capsys):
    _write_shell_batch_record({
        "result": {"price": 0.00001, "values": [float("nan"), float("inf")]},
        "time": datetime(2026, 9, 4, tzinfo=timezone.utc),
    })

    output = capsys.readouterr().out
    assert len(output.splitlines()) == 1
    assert '"price":0.00001' in output
    payload = json.loads(output)
    assert payload["result"]["values"] == [None, None]
    assert payload["time"] == "2026-09-04T00:00:00+00:00"


def test_lightweight_cli_error_uses_shared_serializer(monkeypatch, capsys):
    monkeypatch.setattr("mtdata.core.cli.build_error_payload", lambda *args, **kwargs: {
        "error": "missing command", "details": {"price": 0.00001, "value": float("nan")},
    })

    assert _print_missing_command_error("mtdata-cli", as_json=True) == 1

    output = capsys.readouterr().out
    assert "NaN" not in output
    assert "1e-05" not in output
    assert json.loads(output)["details"]["value"] is None
