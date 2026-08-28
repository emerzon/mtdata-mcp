from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from scripts import btcusd_forecast_smoke_validate as validator

PAIRS = {
    "lgbm-momentum": "lgbm-raw",
    "lgbm-volatility-time": "lgbm-raw",
    "rf-momentum": "rf-raw",
    "rf-volatility-time": "rf-raw",
}
FEATURES = {
    "momentum": {
        "indicators": "rsi(14),roc(12)",
        "observed_future_policy": "carry_forward",
    },
    "volatility-time": {
        "future_covariates": ["hour", "dow"],
        "indicators": "natr(14)",
        "observed_future_policy": "carry_forward",
    },
}


def _source() -> dict[str, Any]:
    status = {"tracked_status": "", "untracked_runtime_files": []}
    return {
        "git_head": "a" * 40,
        "source_tree_dirty": False,
        "source_status_sha256": validator._sha256_json(status),
    }


def _runtime() -> dict[str, Any]:
    return {
        "python": "test-python",
        "python_executable": "test-python.exe",
        "platform": "test-platform",
        "package_versions": {"mtdata-mcp": {"status": "installed", "version": "1"}},
    }


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _inventory(run_dir: Path, model_store: Path, jobs_db: Path) -> dict[str, Any]:
    files = sorted(
        [path for path in model_store.rglob("*") if path.is_file()],
        key=lambda path: path.as_posix(),
    )
    files.extend(
        sorted(
            [
                path
                for path in jobs_db.parent.iterdir()
                if path.name.startswith(jobs_db.name) and path.is_file()
            ],
            key=lambda path: path.as_posix(),
        )
    )
    return {
        "complete": True,
        "max_files": 256,
        "files_observed": len(files),
        "files": [
            {
                "path": path.relative_to(run_dir).as_posix(),
                "size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in files
        ],
    }


def _usage(selected: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    calendar = [
        column
        for column in selected
        if validator._semantic_token(column) in {"hrsin", "hrcos", "dowsin", "dowcos"}
    ]
    indicators = [column for column in selected if column not in calendar]
    method = {
        "status": "consumed",
        "historical_consumed": True,
        "future_consumed": True,
        "anchors_verified": 1,
        "historical_rows_min": 720,
        "historical_rows_max": 720,
        "future_rows": 12,
        "n_features": len(selected),
        "adapter_columns": [f"x{index}" for index in range(len(selected))],
        "selected_columns": selected,
        "include_columns": [],
        "indicator_columns": indicators,
        "calendar_columns": calendar,
        "observed_feature_lag_bars": 1,
        "observed_future_policy": "carry_forward",
    }
    detail = {
        key: copy.deepcopy(value)
        for key, value in method.items()
        if key not in {"anchors_verified", "historical_rows_min", "historical_rows_max"}
    }
    detail["historical_rows"] = 720
    return method, detail


def _payload(method: str, *, feature_family: str | None, delta: float) -> dict[str, Any]:
    actual = [0.0002 * (index - 5) for index in range(12)]
    forecast = [0.0001 * (index + 1) + delta for index in range(12)]
    detail: dict[str, Any] = {
        "anchor": "2024-05-30T12:00:00Z",
        "success": True,
        "training_bars_used": 720,
        "forecast": forecast,
        "actual": actual,
        "entry_price": 68000.25,
        "signal_reference_price": 67999.75,
        "entry_time": "2024-05-30T13:00:00Z",
        "entry_price_source": "next_bar_open",
        "params_used": copy.deepcopy(validator.EXPECTED_PARAMS[method]),
    }
    result: dict[str, Any] = {
        "success": True,
        "complete_success": True,
        "status": "complete",
        "num_tests": 1,
        "successful_tests": 1,
        "failed_tests": 0,
        "details": [detail],
    }
    if feature_family is not None:
        selected = (
            ["RSI_14", "ROC_12"]
            if feature_family == "momentum"
            else ["NATR_14", "hr_sin", "hr_cos", "dow_sin", "dow_cos"]
        )
        method_usage, detail_usage = _usage(selected)
        result["feature_usage"] = method_usage
        detail["feature_usage"] = detail_usage
    return {
        "success": True,
        "complete_success": True,
        "status": "complete",
        "detail": "full",
        "methods_total": 1,
        "methods_succeeded": 1,
        "methods_complete": 1,
        "methods_partial": 0,
        "methods_failed": 0,
        "anchor_tests_planned": 1,
        "anchor_tests_succeeded": 1,
        "anchor_tests_failed": 0,
        "symbol": "BTCUSD",
        "timeframe": "H1",
        "slippage_bps": 0.5,
        "spread_bps": 0.625,
        "commission_bps_per_side": 0.0,
        "backtest_plan": {
            "model": "rolling_origin_fixed_window",
            "anchor_mode": "rolling",
            "runs_requested": 1,
            "runs_used": 1,
            "horizon_bars": 12,
            "history_bars_used": 744,
            "method_selection": "explicit",
            "methods_planned": [method],
            "method_count": 1,
            "fits_planned": 1,
            "model_lookback_bars": 720,
            "anchor_spacing_bars": 12,
            "validation_span_bars": 12,
        },
        "analysis_time_window": {
            "history_start": "2024-05-01T00:00:00Z",
            "history_end": "2024-05-31T23:00:00Z",
            "evaluation_start": "2024-05-30T13:00:00Z",
            "evaluation_end": "2024-05-31T00:00:00Z",
            "first_anchor": "2024-05-30T12:00:00Z",
            "last_anchor": "2024-05-30T12:00:00Z",
            "timezone": "UTC",
            "timestamp_basis": "bar_open_time",
            "input_bar_policy": "closed_bars_only",
            "evaluation_target_policy": "next_bar_through_horizon_bar",
        },
        "results": {method: result},
    }


def _variant_parts(variant: str) -> tuple[str, str | None]:
    method = "mlf_lightgbm" if variant.startswith("lgbm") else "mlf_rf"
    if variant.endswith("raw"):
        return method, None
    return method, variant.split("-", 1)[1]


def _make_run(
    tmp_path: Path,
    *,
    completed: set[str] | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    run_dir = (tmp_path / "smoke").resolve()
    all_variants = set(PAIRS) | set(PAIRS.values())
    completed_variants = all_variants if completed is None else completed
    config: dict[str, Any] = {
        "schema_version": 1,
        "symbol": "BTCUSD",
        "seed": 42,
        "smoke_validation": {
            "pairs": [
                {"feature_variant": feature, "raw_variant": raw}
                for feature, raw in PAIRS.items()
            ]
        },
    }
    commands: dict[str, Any] = {}
    for variant in validator.EXPECTED_VARIANT_ORDER:
        method, family = _variant_parts(variant)
        params = copy.deepcopy(validator.EXPECTED_PARAMS[method])
        argv = [
            "forecast_backtest_run",
            "BTCUSD",
            "--timeframe",
            "H1",
            "--horizon",
            "12",
            "--steps",
            "1",
            "--spacing",
            "12",
            "--lookback",
            "720",
            "--methods",
            method,
            "--quantity",
            "return",
            "--start",
            "2024-05-01",
            "--end",
            "2024-05-31",
            "--detail",
            "full",
            "--params",
            json.dumps(params, sort_keys=True, separators=(",", ":")),
        ]
        features = copy.deepcopy(FEATURES[family]) if family is not None else None
        if features is not None:
            argv += ["--features", json.dumps(features, sort_keys=True, separators=(",", ":"))]
        argv += [
            "--slippage-bps",
            "0.5",
            "--spread-bps",
            "0.625",
            "--commission-bps-per-side",
            "0.0",
            "--trade-threshold",
            "0.0",
        ]
        command_id = f"development-window-end-h1-h12-return-lb720-{variant}"
        metadata = {
            "kind": "baseline_screen",
            "window": "development",
            "shard": {
                "id": "window-end",
                "kind": "exact_window_end",
                "start": "2024-05-01",
                "end": "2024-05-31",
            },
            "timeframe": "H1",
            "horizon": 12,
            "quantity": "return",
            "lookback": 720,
            "steps": 1,
            "spacing": 12,
            "detail": "full",
            "training_floor": "2024-05-01",
            "methods": [method],
            "variant": variant,
            "feature_contract_required": family is not None,
            "features": features,
            "feature_expectations": {"complete": True} if family else None,
            "params": params,
            "params_per_method": None,
            "denoise": None,
            "dimred": None,
        }
        record: dict[str, Any] = {
            "argv": argv,
            "metadata": metadata,
            "status": "preregistered",
            "attempts": 0,
        }
        if variant in completed_variants:
            command_slug = validator._slug(command_id)
            model_relative = f"model_store/screen/{command_slug}/attempt-1"
            jobs_relative = f"jobs/screen/{command_slug}/attempt-1.sqlite"
            model_store = run_dir / model_relative
            jobs_db = run_dir / jobs_relative
            model_store.mkdir(parents=True, exist_ok=True)
            jobs_db.parent.mkdir(parents=True, exist_ok=True)
            (model_store / "model.bin").write_bytes(f"model-{variant}".encode())
            jobs_db.write_bytes(f"jobs-{variant}".encode())
            inventory = _inventory(run_dir, model_store, jobs_db)
            isolation = {
                "attempt": 1,
                "policy": "fresh_command_attempt",
                "model_store_path": model_relative,
                "jobs_db_path": jobs_relative,
                "preexisting_files": 0,
                "attempt_isolation_exempt": False,
                "post_call_inventory": inventory,
            }
            raw_relative = f"raw/screen/{command_slug}/attempt-1.json"
            raw_path = run_dir / raw_relative
            delta = 0.0 if family is None else (1e-6 if family == "momentum" else -2e-6)
            envelope = {
                "schema_version": 1,
                "stage": "screen",
                "command_id": command_id,
                "status": "completed",
                "invocation": ["test-python.exe", "-m", "mtdata", "--json", *argv],
                "metadata": metadata,
                "returncode": 0,
                "payload": _payload(method, feature_family=family, delta=delta),
                "attempt_isolation": isolation,
            }
            _json_write(raw_path, envelope)
            raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
            attempt = {**isolation, "raw_path": raw_relative, "raw_sha256": raw_hash}
            record.update(
                {
                    "status": "completed",
                    "attempts": 1,
                    "raw_path": raw_relative,
                    "raw_sha256": raw_hash,
                    "post_call_inventory": inventory,
                    "attempt_artifacts": [attempt],
                }
            )
        commands[command_id] = record
    config_hash = validator._sha256_json(config)
    revision_relative = f"config_revisions/{config_hash}.json"
    _json_write(
        run_dir / revision_relative,
        {"schema_version": 1, "config_hash": config_hash, "config": config},
    )
    manifest = {
        "schema_version": 1,
        "harness_version": "0.2.0",
        "run_id": "test-smoke",
        "source": _source(),
        "runtime": _runtime(),
        "active_config_hash": config_hash,
        "config_revisions": [{"config_hash": config_hash, "path": revision_relative}],
        "stages": {
            "screen": {
                "status": "completed" if completed_variants == all_variants else "partial",
                "plan_digest": validator._sha256_json(
                    [
                        {
                            "command_id": command_id,
                            "argv": record["argv"],
                            "metadata": record["metadata"],
                        }
                        for command_id, record in commands.items()
                    ]
                ),
                "commands": commands,
            }
        },
    }
    _json_write(run_dir / "resolved_config.json", config)
    _json_write(run_dir / "manifest.json", manifest)
    return run_dir, manifest, config


def _rewrite_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    _json_write(run_dir / "manifest.json", manifest)


def _mutate_envelope(
    run_dir: Path,
    manifest: dict[str, Any],
    variant: str,
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    command_id, record = next(
        (command_id, record)
        for command_id, record in manifest["stages"]["screen"]["commands"].items()
        if record["metadata"]["variant"] == variant
    )
    raw_path = run_dir / record["raw_path"]
    envelope = json.loads(raw_path.read_text(encoding="utf-8"))
    mutation(envelope)
    _json_write(raw_path, envelope)
    digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    record["raw_sha256"] = digest
    record["attempt_artifacts"][-1]["raw_sha256"] = digest
    _rewrite_manifest(run_dir, manifest)


def _validate(run_dir: Path, *, pair: str | None = None) -> dict[str, Any]:
    return validator.validate_smoke_run(
        run_dir,
        pair=pair,
        current_source=_source(),
        current_runtime=_runtime(),
        generated_at="2026-08-27T00:00:00Z",
    )


def test_aggregate_validates_six_commands_and_four_pairs(tmp_path: Path) -> None:
    run_dir, _, _ = _make_run(tmp_path)

    artifact = _validate(run_dir)

    assert artifact["status"] == "passed"
    assert artifact["mode"] == "aggregate"
    assert artifact["outcome_metrics_computed"] is False
    assert len(artifact["diagnostics"]["pairs_validated"]) == 4
    assert len(artifact["input_hashes"]) == 22
    assert "actual" not in json.dumps(artifact)


@pytest.mark.parametrize("harness_version", ["0.2.0", "0.3.0"])
def test_validator_accepts_preserved_and_current_harness_versions(
    tmp_path: Path,
    harness_version: str,
) -> None:
    run_dir, manifest, _ = _make_run(tmp_path / harness_version.replace(".", "-"))
    manifest["harness_version"] = harness_version
    _rewrite_manifest(run_dir, manifest)

    assert _validate(run_dir)["status"] == "passed"


def test_validator_rejects_unknown_harness_version(tmp_path: Path) -> None:
    run_dir, manifest, _ = _make_run(tmp_path)
    manifest["harness_version"] = "9.0.0"
    _rewrite_manifest(run_dir, manifest)

    artifact = _validate(run_dir)

    assert artifact["status"] == "failed"
    assert artifact["diagnostics"]["errors"][0]["code"] == validator.CONFIG_ERROR


def test_pair_mode_passes_before_other_commands_run(tmp_path: Path) -> None:
    completed = {"lgbm-raw", "lgbm-momentum"}
    run_dir, _, _ = _make_run(tmp_path, completed=completed)

    pair_artifact = _validate(run_dir, pair="lgbm-momentum")
    aggregate_artifact = _validate(run_dir)

    assert pair_artifact["status"] == "passed"
    assert pair_artifact["mode"] == "pair"
    assert len(pair_artifact["diagnostics"]["pairs_validated"]) == 1
    assert aggregate_artifact["status"] == "failed"


@pytest.mark.parametrize("field", ["actual", "entry_price", "entry_time"])
def test_pair_rejects_mismatched_actual_or_entry_evidence(
    tmp_path: Path, field: str
) -> None:
    run_dir, manifest, _ = _make_run(tmp_path)

    def mutate(envelope: dict[str, Any]) -> None:
        detail = envelope["payload"]["results"]["mlf_lightgbm"]["details"][0]
        if field == "actual":
            detail[field][0] += 1e-9
        elif field == "entry_price":
            detail[field] += 1.0
        else:
            detail[field] = "2024-05-30T14:00:00Z"

    _mutate_envelope(run_dir, manifest, "lgbm-momentum", mutate)

    artifact = _validate(run_dir, pair="lgbm-momentum")

    assert artifact["status"] == "failed"
    expected_code = validator.RESULT_ERROR if field == "entry_time" else validator.PAIR_ERROR
    assert artifact["diagnostics"]["errors"][0]["code"] == expected_code


def test_pair_rejects_feature_forecast_noop_at_tolerance(tmp_path: Path) -> None:
    run_dir, manifest, _ = _make_run(tmp_path)
    raw_record = next(
        record
        for record in manifest["stages"]["screen"]["commands"].values()
        if record["metadata"]["variant"] == "rf-raw"
    )
    raw_envelope = json.loads((run_dir / raw_record["raw_path"]).read_text(encoding="utf-8"))
    raw_forecast = raw_envelope["payload"]["results"]["mlf_rf"]["details"][0]["forecast"]

    def mutate(envelope: dict[str, Any]) -> None:
        envelope["payload"]["results"]["mlf_rf"]["details"][0]["forecast"] = [
            value + 1e-13 for value in raw_forecast
        ]

    _mutate_envelope(run_dir, manifest, "rf-momentum", mutate)

    artifact = _validate(run_dir, pair="rf-momentum")

    error = artifact["diagnostics"]["errors"][0]
    assert artifact["status"] == "failed"
    assert error["code"] == validator.NOOP_ERROR
    assert error["context"]["absolute_tolerance"] == 1e-12


def test_aggregate_rejects_cross_adapter_evaluation_drift(tmp_path: Path) -> None:
    run_dir, manifest, _ = _make_run(tmp_path)

    def mutate(envelope: dict[str, Any]) -> None:
        detail = envelope["payload"]["results"]["mlf_rf"]["details"][0]
        detail["actual"][0] += 1e-9

    for variant in ("rf-raw", "rf-momentum", "rf-volatility-time"):
        _mutate_envelope(run_dir, manifest, variant, mutate)

    artifact = _validate(run_dir)

    assert artifact["status"] == "failed"
    assert artifact["diagnostics"]["errors"][0]["code"] == validator.PAIR_ERROR


def test_raw_control_rejects_feature_usage(tmp_path: Path) -> None:
    run_dir, manifest, _ = _make_run(tmp_path)

    def mutate(envelope: dict[str, Any]) -> None:
        envelope["payload"]["results"]["mlf_rf"]["feature_usage"] = {}

    _mutate_envelope(run_dir, manifest, "rf-raw", mutate)

    artifact = _validate(run_dir, pair="rf-momentum")

    assert artifact["status"] == "failed"
    assert artifact["diagnostics"]["errors"][0]["code"] == validator.RESULT_ERROR


@pytest.mark.parametrize("case", ["count", "finite", "usage"])
def test_feature_requires_complete_finite_consumed_evidence(
    tmp_path: Path, case: str
) -> None:
    run_dir, manifest, _ = _make_run(tmp_path)

    def mutate(envelope: dict[str, Any]) -> None:
        result = envelope["payload"]["results"]["mlf_lightgbm"]
        if case == "count":
            envelope["payload"]["anchor_tests_succeeded"] = 0
        elif case == "finite":
            result["details"][0]["forecast"][3] = float("nan")
        else:
            result["feature_usage"]["historical_consumed"] = False

    _mutate_envelope(run_dir, manifest, "lgbm-volatility-time", mutate)

    artifact = _validate(run_dir, pair="lgbm-volatility-time")

    assert artifact["status"] == "failed"
    assert artifact["diagnostics"]["errors"][0]["code"] == validator.RESULT_ERROR


def test_feature_accepts_aligned_post_transform_history_rows(tmp_path: Path) -> None:
    run_dir, manifest, _ = _make_run(tmp_path)

    def mutate(envelope: dict[str, Any]) -> None:
        result = envelope["payload"]["results"]["mlf_lightgbm"]
        result["feature_usage"]["historical_rows_min"] = 706
        result["feature_usage"]["historical_rows_max"] = 706
        result["details"][0]["feature_usage"]["historical_rows"] = 706

    _mutate_envelope(run_dir, manifest, "lgbm-momentum", mutate)

    artifact = _validate(run_dir, pair="lgbm-momentum")

    assert artifact["status"] == "passed"


def test_feature_accepts_semantically_equivalent_column_order_and_case(
    tmp_path: Path,
) -> None:
    run_dir, manifest, _ = _make_run(tmp_path)

    def mutate(envelope: dict[str, Any]) -> None:
        result = envelope["payload"]["results"]["mlf_lightgbm"]
        for usage in (result["feature_usage"], result["details"][0]["feature_usage"]):
            usage["selected_columns"] = ["roc_12", "rsi_14"]
            usage["indicator_columns"] = ["roc_12", "rsi_14"]

    _mutate_envelope(run_dir, manifest, "lgbm-momentum", mutate)

    assert _validate(run_dir, pair="lgbm-momentum")["status"] == "passed"


@pytest.mark.parametrize("case", ["columns", "params"])
def test_feature_requires_exact_columns_and_fitted_params(
    tmp_path: Path, case: str
) -> None:
    run_dir, manifest, _ = _make_run(tmp_path)

    def mutate(envelope: dict[str, Any]) -> None:
        result = envelope["payload"]["results"]["mlf_rf"]
        detail = result["details"][0]
        if case == "columns":
            result["feature_usage"]["selected_columns"] = ["EMA_14", "ROC_12"]
            detail["feature_usage"]["selected_columns"] = ["EMA_14", "ROC_12"]
        else:
            detail["params_used"]["n_estimators"] = 99

    _mutate_envelope(run_dir, manifest, "rf-momentum", mutate)

    artifact = _validate(run_dir, pair="rf-momentum")

    assert artifact["status"] == "failed"
    assert artifact["diagnostics"]["errors"][0]["code"] == validator.RESULT_ERROR


def test_feature_requires_adapter_and_category_attestation(tmp_path: Path) -> None:
    run_dir, manifest, _ = _make_run(tmp_path)

    def mutate(envelope: dict[str, Any]) -> None:
        result = envelope["payload"]["results"]["mlf_rf"]
        for usage in (result["feature_usage"], result["details"][0]["feature_usage"]):
            usage.pop("adapter_columns")
            usage.pop("indicator_columns")

    _mutate_envelope(run_dir, manifest, "rf-momentum", mutate)

    artifact = _validate(run_dir, pair="rf-momentum")
    assert artifact["status"] == "failed"
    assert artifact["diagnostics"]["errors"][0]["code"] == validator.RESULT_ERROR


@pytest.mark.parametrize("field", ["returncode", "anchor_tests_planned"])
def test_boolean_counters_fail_closed(tmp_path: Path, field: str) -> None:
    run_dir, manifest, _ = _make_run(tmp_path)

    def mutate(envelope: dict[str, Any]) -> None:
        if field == "returncode":
            envelope[field] = False
        else:
            envelope["payload"][field] = True

    _mutate_envelope(run_dir, manifest, "lgbm-momentum", mutate)

    assert _validate(run_dir, pair="lgbm-momentum")["status"] == "failed"


def test_out_of_contract_evaluation_context_and_invocation_fail(
    tmp_path: Path,
) -> None:
    run_dir, manifest, _ = _make_run(tmp_path)

    def mutate_context(envelope: dict[str, Any]) -> None:
        envelope["payload"]["analysis_time_window"]["evaluation_end"] = (
            "2024-06-01T00:00:00Z"
        )

    _mutate_envelope(run_dir, manifest, "lgbm-momentum", mutate_context)
    assert _validate(run_dir, pair="lgbm-momentum")["status"] == "failed"

    run_dir, manifest, _ = _make_run(tmp_path / "invocation")

    def mutate_invocation(envelope: dict[str, Any]) -> None:
        envelope["invocation"][0] = "another-python.exe"

    _mutate_envelope(run_dir, manifest, "lgbm-momentum", mutate_invocation)
    assert _validate(run_dir, pair="lgbm-momentum")["status"] == "failed"


def test_rehashes_raw_attempt_and_inventory_files(tmp_path: Path) -> None:
    run_dir, manifest, _ = _make_run(tmp_path)
    record = next(iter(manifest["stages"]["screen"]["commands"].values()))
    raw_path = run_dir / record["raw_path"]
    raw_path.write_text("tampered", encoding="utf-8")

    raw_artifact = _validate(run_dir)

    assert raw_artifact["status"] == "failed"
    assert raw_artifact["diagnostics"]["errors"][0]["code"] == validator.RAW_ERROR

    run_dir, manifest, _ = _make_run(tmp_path / "inventory")
    record = next(iter(manifest["stages"]["screen"]["commands"].values()))
    inventory_path = run_dir / record["post_call_inventory"]["files"][0]["path"]
    inventory_path.write_bytes(b"tampered")
    inventory_artifact = _validate(run_dir)
    assert inventory_artifact["status"] == "failed"
    assert inventory_artifact["diagnostics"]["errors"][0]["code"] == validator.RAW_ERROR


def test_rejects_dirty_source_runtime_change_and_noncanonical_store(tmp_path: Path) -> None:
    run_dir, manifest, _ = _make_run(tmp_path)
    dirty = {**_source(), "source_tree_dirty": True}
    source_artifact = validator.validate_smoke_run(
        run_dir, current_source=dirty, current_runtime=_runtime()
    )
    changed_runtime = {**_runtime(), "python": "different"}
    runtime_artifact = validator.validate_smoke_run(
        run_dir, current_source=_source(), current_runtime=changed_runtime
    )
    assert source_artifact["diagnostics"]["errors"][0]["code"] == validator.SOURCE_ERROR
    assert runtime_artifact["diagnostics"]["errors"][0]["code"] == validator.SOURCE_ERROR

    record = next(iter(manifest["stages"]["screen"]["commands"].values()))
    record["attempt_artifacts"][0]["model_store_path"] = "model_store/reused"
    _rewrite_manifest(run_dir, manifest)
    store_artifact = _validate(run_dir)
    assert store_artifact["diagnostics"]["errors"][0]["code"] == validator.ATTEMPT_ERROR


def test_plan_requires_exact_six_and_frozen_pair_context(tmp_path: Path) -> None:
    run_dir, manifest, _ = _make_run(tmp_path)
    commands = manifest["stages"]["screen"]["commands"]
    commands.pop(next(iter(commands)))
    _rewrite_manifest(run_dir, manifest)

    count_artifact = _validate(run_dir)

    assert count_artifact["diagnostics"]["errors"][0]["code"] == validator.PLAN_ERROR

    run_dir, manifest, _ = _make_run(tmp_path / "context")
    feature = next(
        record
        for record in manifest["stages"]["screen"]["commands"].values()
        if record["metadata"]["variant"] == "lgbm-momentum"
    )
    feature["argv"][feature["argv"].index("--spread-bps") + 1] = "0.7"
    _rewrite_manifest(run_dir, manifest)
    context_artifact = _validate(run_dir)
    assert context_artifact["diagnostics"]["errors"][0]["code"] == validator.PLAN_ERROR


def test_unknown_pair_fails_and_artifact_writer_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _, _ = _make_run(tmp_path)
    unknown = _validate(run_dir, pair="missing")
    assert unknown["status"] == "failed"
    assert unknown["diagnostics"]["errors"][0]["code"] == validator.CONFIG_ERROR

    monkeypatch.setattr(validator, "_git_source_state", _source)
    monkeypatch.setattr(validator, "_runtime_identity", _runtime)
    path, artifact = validator.validate_and_write(run_dir, pair="lgbm-momentum")
    assert path.name == "smoke_validation_lgbm-momentum.json"
    assert path.is_file()
    assert artifact["status"] == "passed"
    with pytest.raises(validator.SmokeValidationError, match="Refusing to overwrite"):
        validator.validate_and_write(run_dir, pair="lgbm-momentum")


def test_failed_validation_is_also_written_immutably(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, manifest, _ = _make_run(tmp_path)
    manifest["stages"]["screen"]["status"] = "partial"
    _rewrite_manifest(run_dir, manifest)
    monkeypatch.setattr(validator, "_git_source_state", _source)
    monkeypatch.setattr(validator, "_runtime_identity", _runtime)

    path, artifact = validator.validate_and_write(run_dir)

    assert path.name == "smoke_validation.json"
    assert artifact["status"] == "failed"
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "failed"


def test_manifest_and_validation_directory_symlinks_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _, _ = _make_run(tmp_path / "manifest")
    manifest_path = run_dir / "manifest.json"
    outside_manifest = tmp_path / "outside-manifest.json"
    outside_manifest.write_bytes(manifest_path.read_bytes())
    manifest_path.unlink()
    try:
        manifest_path.symlink_to(outside_manifest)
    except OSError as exc:
        pytest.skip(f"Symbolic links unavailable: {exc}")
    artifact = _validate(run_dir)
    assert artifact["status"] == "failed"
    assert artifact["diagnostics"]["errors"][0]["code"] == validator.RAW_ERROR

    run_dir, _, _ = _make_run(tmp_path / "output")
    outside_dir = tmp_path / "outside-validation"
    outside_dir.mkdir()
    (run_dir / "validation").symlink_to(outside_dir, target_is_directory=True)
    monkeypatch.setattr(validator, "_git_source_state", _source)
    monkeypatch.setattr(validator, "_runtime_identity", _runtime)
    with pytest.raises(validator.SmokeValidationError, match="symbolic link"):
        validator.validate_and_write(run_dir)
    assert not any(outside_dir.iterdir())
