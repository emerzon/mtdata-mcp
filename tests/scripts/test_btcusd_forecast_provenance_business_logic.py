from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

from scripts import btcusd_forecast_experiment as experiment


def _clean_source() -> dict[str, object]:
    return {
        "git_head": "a" * 40,
        "tracked_tree_dirty": False,
        "untracked_runtime_files": [],
        "source_tree_dirty": False,
        "source_status_sha256": "b" * 64,
        "captured_at": "2026-08-27T00:00:00Z",
    }


def _context(tmp_path: Path) -> experiment.RunContext:
    args = experiment.build_parser().parse_args(
        ["--run-dir", str(tmp_path / "study"), "audit"]
    )
    return experiment.prepare_context(args)


def test_project_distribution_snapshot_uses_installed_name() -> None:
    assert experiment._PACKAGE_DISTRIBUTIONS[0] == "mtdata-mcp"


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1.25, True), (float("nan"), False), (float("inf"), False), (True, False)],
)
def test_finite_number_contract(value: object, expected: bool) -> None:
    assert experiment._finite_number(value) is expected


def test_volatility_screen_accepts_finite_full_detail() -> None:
    metadata = {
        "kind": "baseline_screen",
        "methods": ["ewma"],
        "steps": 1,
        "horizon": 6,
        "lookback": 720,
        "quantity": "volatility",
    }
    payload = {
        "complete_success": True,
        "status": "complete",
        "methods_total": 1,
        "methods_succeeded": 1,
        "methods_complete": 1,
        "methods_partial": 0,
        "methods_failed": 0,
        "anchor_tests_planned": 1,
        "anchor_tests_succeeded": 1,
        "anchor_tests_failed": 0,
        "results": {
            "ewma": {
                "success": True,
                "complete_success": True,
                "status": "complete",
                "num_tests": 1,
                "successful_tests": 1,
                "details": [
                    {
                        "success": True,
                        "anchor": "2024-01-01T00:00:00Z",
                        "training_bars_used": 720,
                        "forecast_sigma": 0.02,
                        "realized_sigma": 0.03,
                    }
                ],
            }
        },
    }

    assert experiment._screen_collection_contract_error(metadata, payload) is None


def test_resume_rejects_changed_dependency_versions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "study"
    run_dir.mkdir()
    current_runtime = experiment._runtime_identity_snapshot()
    recorded_runtime = copy.deepcopy(current_runtime)
    recorded_runtime["package_versions"]["numpy"] = {
        "status": "installed",
        "version": "0.0-different",
    }
    (run_dir / "manifest.json").write_text(
        json.dumps({"source": _clean_source(), "runtime": recorded_runtime}),
        encoding="utf-8",
    )
    args = experiment.build_parser().parse_args(
        ["--run-dir", str(run_dir), "--resume", "audit"]
    )
    monkeypatch.setattr(experiment, "_git_source_state", _clean_source)
    monkeypatch.setattr(
        experiment,
        "_runtime_identity_snapshot",
        lambda: current_runtime,
    )

    with pytest.raises(experiment.HarnessError, match="package_versions"):
        experiment._enforce_source_integrity(args)


def test_source_is_rechecked_after_each_child_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    context.enforce_execution_integrity = True
    clean = _clean_source()
    dirty = {**clean, "tracked_tree_dirty": True, "source_tree_dirty": True}
    context.manifest["source"] = clean
    context.manifest["runtime"].update(experiment._runtime_identity_snapshot())
    observed = iter((clean, dirty))
    monkeypatch.setattr(experiment, "_git_source_state", lambda: next(observed))
    runtime = experiment._runtime_identity_snapshot()
    monkeypatch.setattr(experiment, "_runtime_identity_snapshot", lambda: runtime)
    monkeypatch.setattr(
        experiment.subprocess,
        "run",
        lambda invocation, **kwargs: subprocess.CompletedProcess(
            invocation,
            0,
            stdout=json.dumps({"success": True}),
            stderr="",
        ),
    )
    spec = experiment.CommandSpec("catalog", ("forecast_list_methods",), {})
    experiment._register_specs(context, "audit", [spec])

    assert experiment.execute_spec(context, "audit", spec) is False
    record = context.manifest["stages"]["audit"]["commands"][spec.command_id]
    assert record["status"] == "failed"
    assert "source tree changed" in record["error"]


def test_untracked_runtime_module_marks_source_tree_dirty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = iter(("a" * 40, "", "src/mtdata/untracked_runtime.py\n"))

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 0, stdout=next(outputs), stderr="")

    monkeypatch.setattr(experiment.subprocess, "run", fake_run)

    state = experiment._git_source_state()

    assert state["source_tree_dirty"] is True
    assert state["untracked_runtime_files"] == ["src/mtdata/untracked_runtime.py"]


def test_tampered_prior_attempt_blocks_completed_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    spec = experiment.CommandSpec("catalog", ("forecast_list_methods",), {})
    experiment._register_specs(context, "audit", [spec])
    calls = 0

    def fake_run(invocation: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(
            invocation,
            1 if calls == 1 else 0,
            stdout=json.dumps({"success": calls > 1}),
            stderr="",
        )

    monkeypatch.setattr(experiment.subprocess, "run", fake_run)
    assert experiment.execute_spec(context, "audit", spec) is False
    assert experiment.execute_spec(context, "audit", spec) is True
    record = context.manifest["stages"]["audit"]["commands"][spec.command_id]
    first_path = context.run_dir / record["attempt_artifacts"][0]["raw_path"]
    first_path.write_text("tampered", encoding="utf-8")

    with pytest.raises(experiment.HarnessError, match="BTC-RAW-INTEGRITY"):
        experiment.execute_spec(context, "audit", spec)


def test_empty_stage_plan_is_rejected(tmp_path: Path) -> None:
    context = _context(tmp_path)

    with pytest.raises(experiment.HarnessError, match="empty command plan"):
        experiment._run_command_stage(context, "screen", [])


def test_issue_list_ignores_incompatible_stored_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "study"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "legacy", "source": {}}),
        encoding="utf-8",
    )
    (run_dir / "resolved_config.json").write_text("[]", encoding="utf-8")

    result = experiment.main(
        ["--run-dir", str(run_dir), "--resume", "issue", "list"]
    )

    assert result == 0
    assert json.loads(capsys.readouterr().out)["issues"] == []
