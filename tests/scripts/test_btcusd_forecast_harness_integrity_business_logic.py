from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import btcusd_forecast_experiment as experiment


def _context(tmp_path: Path) -> experiment.RunContext:
    args = experiment.build_parser().parse_args(
        ["--run-dir", str(tmp_path / "study"), "audit"]
    )
    return experiment.prepare_context(args)


def _completed(
    invocation: list[str],
    payload: dict[str, object],
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(invocation, 0, stdout=json.dumps(payload), stderr="")


def _feature_spec(
    *,
    methods: list[str] | None = None,
    steps: int = 2,
    horizon: int = 6,
) -> experiment.CommandSpec:
    planned_methods = methods or ["mlf_lightgbm"]
    features = {
        "indicators": "rsi(14)",
        "future_covariates": ["hour"],
        "observed_future_policy": "carry_forward",
    }
    return experiment.CommandSpec(
        "feature-screen",
        (
            "forecast_backtest_run",
            "BTCUSD",
            "--horizon",
            str(horizon),
            "--steps",
            str(steps),
            "--methods",
            *planned_methods,
            "--features",
            json.dumps(features),
            "--detail",
            "full",
        ),
        {
            "kind": "baseline_screen",
            "methods": planned_methods,
            "horizon": horizon,
            "steps": steps,
            "lookback": 720,
            "detail": "full",
            "feature_contract_required": True,
            "features": features,
            "feature_expectations": experiment._feature_expectations(features),
            "params_per_method": {"mlf_lightgbm": {"lags": [1, 2]}},
        },
    )


def _feature_payload(
    *,
    methods: list[str] | None = None,
    steps: int = 2,
    horizon: int = 6,
) -> dict[str, object]:
    planned_methods = methods or ["mlf_lightgbm"]
    results = {
        method: {
            "success": True,
            "complete_success": True,
            "status": "complete",
            "num_tests": steps,
            "successful_tests": steps,
            "failed_tests": 0,
            "feature_usage": {
                "status": "consumed",
                "historical_consumed": True,
                "future_consumed": True,
                "anchors_verified": steps,
                "historical_rows_min": 720,
                "historical_rows_max": 720,
                "future_rows": horizon,
                "n_features": 3,
                "selected_columns": ["RSI_14", "hr_sin", "hr_cos"],
                "observed_feature_lag_bars": 1,
                "observed_future_policy": "carry_forward",
            },
            "details": [
                {
                    "anchor": f"2024-01-{index + 1:02d}T00:00:00Z",
                    "success": True,
                    "training_bars_used": 720,
                    "forecast": [0.001] * horizon,
                    "actual": [0.002] * horizon,
                    "feature_usage": {
                        "status": "consumed",
                        "historical_consumed": True,
                        "future_consumed": True,
                        "historical_rows": 720,
                        "future_rows": horizon,
                        "n_features": 3,
                        "selected_columns": ["RSI_14", "hr_sin", "hr_cos"],
                        "observed_feature_lag_bars": 1,
                        "observed_future_policy": "carry_forward",
                    },
                    "params_used": {"lags": [1, 2]},
                }
                for index in range(steps)
            ],
        }
        for method in planned_methods
    }
    return {
        "success": True,
        "complete_success": True,
        "status": "complete",
        "methods_total": len(planned_methods),
        "methods_succeeded": len(planned_methods),
        "methods_complete": len(planned_methods),
        "methods_partial": 0,
        "methods_failed": 0,
        "anchor_tests_planned": len(planned_methods) * steps,
        "anchor_tests_succeeded": len(planned_methods) * steps,
        "anchor_tests_failed": 0,
        "results": results,
    }


def test_window_end_sharding_and_custom_spacing_build_one_exact_shard() -> None:
    config = copy.deepcopy(experiment.DEFAULT_CONFIG)
    config["screen"].update(
        {
            "sharding": "window_end",
            "spacing_bars": 168,
            "timeframes": ["H1"],
            "horizons": [6, 12, 24],
            "quantities": ["return"],
            "lookbacks": [720],
            "methods": ["naive"],
            "steps_per_shard": 52,
            "variants": [{"id": "raw"}],
        }
    )

    specs = experiment.build_screen_specs(config)

    assert len(specs) == 3
    assert {spec.metadata["shard"]["id"] for spec in specs} == {"window-end"}
    assert {spec.metadata["shard"]["kind"] for spec in specs} == {
        "exact_window_end"
    }
    assert {spec.metadata["shard"]["start"] for spec in specs} == {"2022-06-01"}
    assert {spec.metadata["shard"]["end"] for spec in specs} == {"2024-06-30"}
    assert all(spec.metadata["spacing"] == 168 for spec in specs)
    assert all(spec.argv[spec.argv.index("--spacing") + 1] == "168" for spec in specs)


def test_window_end_auto_steps_reserve_lookback_and_use_custom_spacing() -> None:
    config = copy.deepcopy(experiment.DEFAULT_CONFIG)
    config["screen"].update(
        {
            "sharding": "window_end",
            "spacing_bars": 168,
            "timeframes": ["H1"],
            "horizons": [6],
            "quantities": ["return"],
            "lookbacks": [720],
            "methods": ["naive"],
            "steps_per_shard": "auto_month",
            "variants": [{"id": "raw"}],
        }
    )

    spec = experiment.build_screen_specs(config)[0]
    available = experiment._continuous_bars("2022-06-01", "2024-06-30", "H1")
    expected_steps = (available - 720 - 6) // 168

    assert spec.metadata["steps"] == expected_steps
    assert 720 + expected_steps * 168 + 6 <= available
    assert spec.argv[spec.argv.index("--steps") + 1] == str(expected_steps)


@pytest.mark.parametrize("spacing", [0, -1, 1.5, True, "168"])
def test_screen_spacing_bars_requires_positive_integer(spacing: object) -> None:
    config = copy.deepcopy(experiment.DEFAULT_CONFIG)
    config["screen"]["spacing_bars"] = spacing

    with pytest.raises(experiment.HarnessError, match="positive integer"):
        experiment.build_screen_specs(config)


def test_screen_rejects_unknown_sharding_and_overlapping_custom_spacing() -> None:
    config = copy.deepcopy(experiment.DEFAULT_CONFIG)
    config["screen"]["sharding"] = "weekly"
    with pytest.raises(experiment.HarnessError, match="screen.sharding"):
        experiment.build_screen_specs(config)

    config = copy.deepcopy(experiment.DEFAULT_CONFIG)
    config["screen"].update(
        {
            "spacing_bars": 5,
            "horizons": [6],
            "steps_per_shard": 2,
        }
    )
    with pytest.raises(experiment.HarnessError, match="greater than or equal"):
        experiment.build_screen_specs(config)


def test_screen_rejects_unverifiable_features_and_non_full_evidence() -> None:
    config = copy.deepcopy(experiment.DEFAULT_CONFIG)
    config["screen"]["variants"] = [
        {
            "id": "unknown-indicator",
            "features": {
                "indicators": "ema(14)",
                "observed_future_policy": "carry_forward",
            },
        }
    ]
    with pytest.raises(experiment.HarnessError, match="cannot be fully verified"):
        experiment.build_screen_specs(config)

    config = copy.deepcopy(experiment.DEFAULT_CONFIG)
    config["screen"]["detail"] = "compact"
    with pytest.raises(experiment.HarnessError, match="screen.detail must be 'full'"):
        experiment.build_screen_specs(config)

def test_default_screen_plan_remains_backward_compatible() -> None:
    specs = experiment.build_screen_specs(copy.deepcopy(experiment.DEFAULT_CONFIG))

    assert len(specs) == 534
    assert all(
        spec.argv[spec.argv.index("--spacing") + 1]
        == str(spec.metadata["horizon"])
        for spec in specs
    )


def test_audit_adds_full_indicator_description_commands() -> None:
    config = copy.deepcopy(experiment.DEFAULT_CONFIG)
    config["audit"]["indicator_descriptions"] = ["rsi", "natr"]

    descriptions = [
        spec
        for spec in experiment.build_audit_specs(config)
        if spec.metadata.get("kind") == "indicator_description"
    ]

    assert [spec.command_id for spec in descriptions] == ["indicator-rsi", "indicator-natr"]
    assert [spec.argv for spec in descriptions] == [
        ("indicators_describe", "rsi", "--detail", "full"),
        ("indicators_describe", "natr", "--detail", "full"),
    ]


@pytest.mark.parametrize("descriptions", ["rsi", [""], ["RSI", "rsi"]])
def test_audit_rejects_invalid_indicator_description_config(
    descriptions: object,
) -> None:
    config = copy.deepcopy(experiment.DEFAULT_CONFIG)
    config["audit"]["indicator_descriptions"] = descriptions

    with pytest.raises(experiment.HarnessError, match="indicator_descriptions"):
        experiment.build_audit_specs(config)


def test_feature_screen_accepts_complete_consumption_and_hashes_attempt_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    spec = _feature_spec()
    experiment._register_specs(context, "screen", [spec])

    def fake_run(invocation: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        model_file = Path(environment["MTDATA_MODEL_STORE"]) / "model.bin"
        model_file.write_bytes(b"model")
        jobs_file = Path(environment["MTDATA_FORECAST_JOBS_DB"])
        jobs_file.write_bytes(b"jobs")
        Path(f"{jobs_file}-wal").write_bytes(b"wal")
        Path(f"{jobs_file}-shm").write_bytes(b"shm")
        return _completed(invocation, _feature_payload())

    monkeypatch.setattr(experiment.subprocess, "run", fake_run)

    assert experiment.execute_spec(context, "screen", spec) is True
    record = context.manifest["stages"]["screen"]["commands"][spec.command_id]
    raw_path = context.run_dir / record["raw_path"]
    assert record["raw_sha256"] == hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert record["model_store_path"].endswith("screen/feature-screen/attempt-1")
    assert record["jobs_db_path"].endswith(
        "jobs/screen/feature-screen/attempt-1.sqlite"
    )
    inventory = record["post_call_inventory"]
    assert inventory["complete"] is True
    assert inventory["files_observed"] == 4
    assert all(set(row) == {"path", "size", "sha256"} for row in inventory["files"])
    normalized = experiment.write_normalized_stage(context, "screen")
    assert normalized["status"] != "failed"
    assert normalized["commands"][0]["raw_sha256"] == record["raw_sha256"]
    assert normalized["commands"][0]["status"] == "completed"


@pytest.mark.parametrize(
    ("usage_field", "value", "message"),
    [
        ("status", "prepared", "status=consumed"),
        ("historical_consumed", False, "historical features"),
        ("future_consumed", False, "future features"),
        ("anchors_verified", 1, "anchor counts"),
        ("future_rows", 5, "future_rows"),
    ],
)
def test_feature_screen_contract_rejects_incomplete_attestation(
    usage_field: str,
    value: object,
    message: str,
) -> None:
    spec = _feature_spec()
    payload = _feature_payload()
    payload["results"]["mlf_lightgbm"]["feature_usage"][usage_field] = value

    error = experiment._screen_feature_contract_error(
        spec.metadata,
        spec.argv,
        payload,
    )

    assert message in (error or "")


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("feature_count", "n_features"),
        ("semantic_columns", "semantic feature set"),
        ("semantic_columns_missing", "semantic feature set"),
        ("semantic_columns_prefix", "semantic feature set"),
        ("observed_lag", "one-bar"),
        ("observed_policy", "carry_forward"),
        ("training_bars", "training bars"),
        ("forecast_path", "forecast path"),
        ("actual_path", "actual path"),
        ("anchor_usage", "per-anchor feature_usage"),
        ("historical_rows", "historical feature rows"),
        ("params_used", "params_used"),
        ("params_mismatch", "preregistered parameter"),
    ],
)
def test_feature_screen_contract_checks_semantics_and_full_anchor_paths(
    case: str,
    message: str,
) -> None:
    spec = _feature_spec()
    payload = _feature_payload()
    method = payload["results"]["mlf_lightgbm"]
    usage = method["feature_usage"]
    detail = method["details"][0]
    if case == "feature_count":
        usage["n_features"] = 2
    elif case == "semantic_columns":
        usage["selected_columns"] = ["EMA_14", "hr_sin", "hr_cos"]
    elif case == "semantic_columns_missing":
        usage.pop("selected_columns")
    elif case == "semantic_columns_prefix":
        usage["selected_columns"] = ["RSI_140", "hr_sin", "hr_cos"]
    elif case == "observed_lag":
        usage["observed_feature_lag_bars"] = 0
    elif case == "observed_policy":
        usage["observed_future_policy"] = "unknown"
    elif case == "training_bars":
        detail["training_bars_used"] = 719
    elif case == "forecast_path":
        detail["forecast"] = [float("nan")] * 6
    elif case == "actual_path":
        detail["actual"] = [0.0] * 5
    elif case == "anchor_usage":
        detail.pop("feature_usage")
    elif case == "historical_rows":
        detail["feature_usage"]["historical_rows"] = 719
    elif case == "params_used":
        detail.pop("params_used")
    else:
        detail["params_used"]["lags"] = [1, 3]

    error = experiment._screen_feature_contract_error(
        spec.metadata,
        spec.argv,
        payload,
    )

    assert message in (error or "")


def test_raw_screen_contract_rejects_partial_anchor_coverage() -> None:
    spec = _feature_spec()
    payload = _feature_payload()
    payload["complete_success"] = False
    payload["status"] = "partial"
    payload["methods_complete"] = 0
    payload["methods_partial"] = 1
    payload["anchor_tests_succeeded"] = 1
    payload["anchor_tests_failed"] = 1

    error = experiment._screen_collection_contract_error(spec.metadata, payload)

    assert "complete_success" in (error or "")


def test_feature_screen_command_and_normalization_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    spec = _feature_spec()
    experiment._register_specs(context, "screen", [spec])
    payload = _feature_payload()
    payload["results"]["mlf_lightgbm"].pop("feature_usage")
    monkeypatch.setattr(
        experiment.subprocess,
        "run",
        lambda invocation, **kwargs: _completed(invocation, payload),
    )

    assert experiment.execute_spec(context, "screen", spec) is False
    normalized = experiment.write_normalized_stage(context, "screen")
    row = normalized["commands"][0]
    assert row["status"] == "failed"
    assert "omitted feature_usage" in row["contract_error"]
    assert normalized["status"] == "failed"


def test_feature_contract_failure_stops_stage_without_fail_fast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    first = _feature_spec()
    second = experiment.CommandSpec("feature-screen-2", first.argv, first.metadata)
    payload = _feature_payload()
    payload["results"]["mlf_lightgbm"].pop("feature_usage")
    calls = 0

    def fake_run(invocation: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        return _completed(invocation, payload)

    monkeypatch.setattr(experiment.subprocess, "run", fake_run)

    assert experiment._run_command_stage(context, "screen", [first, second]) == 1
    assert calls == 1
    commands = context.manifest["stages"]["screen"]["commands"]
    assert commands[first.command_id]["status"] == "failed"
    assert commands[second.command_id]["status"] == "preregistered"
    assert "BTC-FEATURE-CONSUMPTION" in commands[first.command_id]["error"]


def test_normalization_rechecks_legacy_completed_feature_envelope(tmp_path: Path) -> None:
    context = _context(tmp_path)
    spec = _feature_spec()
    experiment._register_specs(context, "screen", [spec])
    payload = _feature_payload()
    payload["results"]["mlf_lightgbm"]["feature_usage"]["anchors_verified"] = 1
    raw_path = experiment._raw_path(context, "screen", spec.command_id)
    record = context.manifest["stages"]["screen"]["commands"][spec.command_id]
    experiment._atomic_write_json(
        raw_path,
        {
            "stage": "screen",
            "command_id": spec.command_id,
            "status": "completed",
            "invocation": [
                sys.executable,
                "-m",
                "mtdata",
                "--json",
                *record["argv"],
            ],
            "metadata": record["metadata"],
            "returncode": 0,
            "payload": payload,
        },
    )
    record.update(
        {
            "status": "completed",
            "raw_path": experiment._relative(raw_path, context.run_dir),
            "raw_sha256": experiment._sha256_file(raw_path),
            "attempt_artifacts": [
                {
                    "raw_path": experiment._relative(raw_path, context.run_dir),
                    "raw_sha256": experiment._sha256_file(raw_path),
                }
            ],
        }
    )

    normalized = experiment.write_normalized_stage(context, "screen")

    assert normalized["status"] == "failed"
    assert normalized["commands"][0]["status"] == "failed"
    assert "anchor counts" in normalized["commands"][0]["contract_error"]


def test_completed_raw_hash_is_verified_before_resume_skip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    spec = experiment.CommandSpec("catalog", ("forecast_list_methods",), {})
    experiment._register_specs(context, "audit", [spec])
    monkeypatch.setattr(
        experiment.subprocess,
        "run",
        lambda invocation, **kwargs: _completed(invocation, {"success": True}),
    )
    assert experiment.execute_spec(context, "audit", spec) is True
    record = context.manifest["stages"]["audit"]["commands"][spec.command_id]
    raw_path = context.run_dir / record["raw_path"]
    raw_path.write_text("tampered", encoding="utf-8")

    with pytest.raises(experiment.HarnessError, match="BTC-RAW-INTEGRITY"):
        experiment.execute_spec(context, "audit", spec)


def test_retry_uses_new_attempt_paths_without_reusing_first_attempt(
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
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        Path(environment["MTDATA_MODEL_STORE"]).joinpath("marker").write_text(
            str(calls),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            invocation,
            1 if calls == 1 else 0,
            stdout=json.dumps({"success": calls > 1}),
            stderr="",
        )

    monkeypatch.setattr(experiment.subprocess, "run", fake_run)

    assert experiment.execute_spec(context, "audit", spec) is False
    first_record = context.manifest["stages"]["audit"]["commands"][spec.command_id]
    first_path = context.run_dir / first_record["raw_path"]
    first_hash = first_record["raw_sha256"]
    first_contents = first_path.read_bytes()
    assert experiment.execute_spec(context, "audit", spec) is True
    record = context.manifest["stages"]["audit"]["commands"][spec.command_id]
    assert [row["attempt"] for row in record["attempt_artifacts"]] == [1, 2]
    first_store = context.run_dir / record["attempt_artifacts"][0]["model_store_path"]
    second_store = context.run_dir / record["attempt_artifacts"][1]["model_store_path"]
    assert first_store != second_store
    assert first_store.joinpath("marker").read_text(encoding="utf-8") == "1"
    assert second_store.joinpath("marker").read_text(encoding="utf-8") == "2"
    assert first_path != context.run_dir / record["raw_path"]
    assert first_path.read_bytes() == first_contents
    assert experiment._sha256_file(first_path) == first_hash
    assert record["attempt_artifacts"][0]["raw_path"] != record["attempt_artifacts"][1]["raw_path"]


def test_materialize_and_shadow_keep_shared_lifecycle_store_exemption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    environments: list[dict[str, str]] = []

    def fake_run(invocation: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        environments.append(environment)
        return _completed(invocation, {"success": True})

    monkeypatch.setattr(experiment.subprocess, "run", fake_run)
    for stage in ("materialize", "shadow"):
        spec = experiment.CommandSpec(f"{stage}-command", ("forecast_generate", "BTCUSD"), {})
        experiment._register_specs(context, stage, [spec], allow_append=True)
        assert experiment.execute_spec(context, stage, spec) is True
        record = context.manifest["stages"][stage]["commands"][spec.command_id]
        assert record["post_call_inventory"]["exempt"] is True

    assert all(
        Path(environment["MTDATA_MODEL_STORE"])
        == context.run_dir / "model_store" / "lifecycle"
        for environment in environments
    )
    assert all(
        Path(environment["MTDATA_FORECAST_JOBS_DB"])
        == context.run_dir / "forecast_jobs.sqlite"
        for environment in environments
    )
    assert context.manifest["safety"]["lifecycle_store_exempt_stages"] == [
        "materialize",
        "shadow",
    ]


def test_manifest_captures_source_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = {
        "git_head": "a" * 40,
        "tracked_tree_dirty": False,
        "untracked_runtime_files": [],
        "source_tree_dirty": False,
        "source_status_sha256": hashlib.sha256(b"").hexdigest(),
        "captured_at": "2026-08-27T00:00:00Z",
    }
    versions = {
        name: {"status": "missing", "version": None}
        for name in experiment._PACKAGE_DISTRIBUTIONS
    }
    monkeypatch.setattr(experiment, "_git_source_state", lambda: source)
    monkeypatch.setattr(experiment, "_package_version_snapshot", lambda: versions)

    context = _context(tmp_path)

    assert context.manifest["source"] == source
    assert context.manifest["runtime"]["package_versions"] == versions


def test_package_version_snapshot_explicitly_marks_missing_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_version(distribution: str) -> str:
        if distribution == "numpy":
            return "2.3.0"
        raise experiment.importlib.metadata.PackageNotFoundError(distribution)

    monkeypatch.setattr(experiment.importlib.metadata, "version", fake_version)

    versions = experiment._package_version_snapshot()

    assert set(versions) == set(experiment._PACKAGE_DISTRIBUTIONS)
    assert versions["numpy"] == {"status": "installed", "version": "2.3.0"}
    assert versions["TA-Lib"] == {"status": "missing", "version": None}


def test_source_gate_rejects_dirty_new_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = experiment.build_parser().parse_args(
        ["--run-dir", str(tmp_path / "study"), "audit"]
    )
    monkeypatch.setattr(
        experiment,
        "_git_source_state",
        lambda: {"git_head": "a" * 40, "tracked_tree_dirty": True},
    )

    with pytest.raises(experiment.HarnessError, match="BTC-SOURCE-INTEGRITY"):
        experiment._enforce_source_integrity(args)


@pytest.mark.parametrize(
    ("recorded_head", "recorded_dirty", "current_head", "current_dirty"),
    [
        ("a" * 40, False, "b" * 40, False),
        ("a" * 40, True, "a" * 40, False),
        ("a" * 40, False, "a" * 40, True),
    ],
)
def test_source_gate_rejects_mixed_or_dirty_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recorded_head: str,
    recorded_dirty: bool,
    current_head: str,
    current_dirty: bool,
) -> None:
    run_dir = tmp_path / "study"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source": {
                    "git_head": recorded_head,
                    "tracked_tree_dirty": recorded_dirty,
                }
            }
        ),
        encoding="utf-8",
    )
    args = experiment.build_parser().parse_args(
        ["--run-dir", str(run_dir), "--resume", "audit"]
    )
    monkeypatch.setattr(
        experiment,
        "_git_source_state",
        lambda: {
            "git_head": current_head,
            "tracked_tree_dirty": current_dirty,
        },
    )

    with pytest.raises(experiment.HarnessError, match="BTC-SOURCE-INTEGRITY"):
        experiment._enforce_source_integrity(args)


def test_issue_list_remains_read_only_across_source_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = experiment.build_parser().parse_args(
        ["--run-dir", str(tmp_path / "study"), "--resume", "issue", "list"]
    )
    monkeypatch.setattr(
        experiment,
        "_git_source_state",
        lambda: pytest.fail("read-only issue list should not inspect current source"),
    )

    experiment._enforce_source_integrity(args)
