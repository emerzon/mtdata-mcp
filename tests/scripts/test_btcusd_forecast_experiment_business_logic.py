from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import btcusd_forecast_experiment as experiment


def _completed(invocation: list[str], payload: dict | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        invocation,
        0,
        stdout=json.dumps(payload or {"success": True}),
        stderr="",
    )


def _context(tmp_path: Path, *, dry_run: bool = False) -> experiment.RunContext:
    argv = ["--run-dir", str(tmp_path / "study")]
    if dry_run:
        argv.append("--dry-run")
    argv.append("audit")
    args = experiment.build_parser().parse_args(argv)
    return experiment.prepare_context(args)


def _candidate(**overrides: object) -> dict[str, object]:
    candidate: dict[str, object] = {
        "symbol": "BTCUSD",
        "method": "theta",
        "timeframe": "H1",
        "horizon": 12,
        "quantity": "price",
        "lookback": 720,
        "params": {},
        "pipeline": {
            "denoise": None,
            "features": None,
            "dimred": None,
            "target_spec": None,
        },
        "costs": {
            "slippage_bps": 0.5,
            "spread_bps": 0.625,
            "commission_bps_per_side": 0.0,
            "trade_threshold": 0.0,
        },
        "calibration": {
            "method": "conformal",
            "ci_alpha": 0.10,
            "steps": 100,
            "max_mean_relative_width": 0.10,
        },
        "provenance": {
            "selection_artifacts": ["normalized/screen.json"],
            "baseline": {
                "method": "naive",
                "params": {},
                "minimum_accuracy_improvement": 0.02,
            },
        },
    }
    candidate.update(overrides)
    return candidate


def _successful_backtest_payload(
    invocation: list[str],
    *,
    candidate_method: str = "theta",
    baseline_method: str = "naive",
) -> dict[str, object]:
    end = invocation[invocation.index("--end") + 1]
    steps = int(invocation[invocation.index("--steps") + 1])
    candidate_details = [
        {
            "success": True,
            "forecast": [101.0],
            "actual": [101.0],
            "signal_reference_price": 100.0,
            "entry_price": 100.0,
        }
        for _ in range(steps)
    ]
    baseline_details = [
        {
            "success": True,
            "forecast": [99.0],
            "actual": [101.0],
            "signal_reference_price": 100.0,
            "entry_price": 100.0,
        }
        for _ in range(steps)
    ]
    return {
        "success": True,
        "ranked_methods": [],
        "methods_failed": 0,
        "results": {
            candidate_method: {
                "success": True,
                "num_tests": steps,
                "successful_tests": steps,
                "details": candidate_details,
            },
            baseline_method: {
                "success": True,
                "num_tests": steps,
                "successful_tests": steps,
                "details": baseline_details,
            },
        },
        "analysis_time_window": {
            "evaluation_start": f"{end[:7]}-01T00:00:00Z",
            "evaluation_end": f"{end}T00:00:00Z",
        },
    }


def _write_review_artifacts(
    tmp_path: Path,
    assessment_path: Path,
    *,
    approval_field: str,
    mean_relative_width: float = 0.05,
) -> Path:
    assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
    stage = str(assessment["stage"])
    interval_path = tmp_path / f"{stage}_interval_evidence.json"
    by_window = {
        window: {
            "sample_count": 100,
            "covered_count": 90,
            "empirical_coverage": 0.90,
            "mean_relative_width": mean_relative_width,
        }
        for window in assessment["windows"]
    }
    interval_payload = {
        "candidate_hash": assessment["candidate_hash"],
        "pipeline_hash": assessment["candidate_spec"]["pipeline_hash"],
        "timeframe": assessment["candidate_spec"]["timeframe"],
        "horizon": assessment["candidate_spec"]["horizon"],
        "quantity": assessment["candidate_spec"]["quantity"],
        "source": "mtdata_cli_oos_replay",
        "out_of_sample": True,
        "causal_actuals_after_forecast": True,
        "forecast_tool": "forecast_conformal_intervals",
        "actuals_tool": "data_fetch_candles",
        "nominal_coverage": 0.90,
        "width_unit": "price",
        "relative_width_definition": experiment.RELATIVE_WIDTH_DEFINITION,
        "windows": assessment["windows"],
        "sample_count": 100 * len(by_window),
        "covered_count": 90 * len(by_window),
        "empirical_coverage": 0.90,
        "mean_width": 125.0,
        "mean_relative_width": mean_relative_width,
        "by_window": by_window,
        "raw_sha256": {window: "a" * 64 for window in assessment["windows"]},
    }
    interval_path.write_text(json.dumps(interval_payload), encoding="utf-8")
    review_path = tmp_path / f"{stage}_review.json"
    review_path.write_text(
        json.dumps(
            {
                "candidate_hash": assessment["candidate_hash"],
                "assessment_sha256": hashlib.sha256(assessment_path.read_bytes()).hexdigest(),
                "reviewer": "test-reviewer",
                "interval_evidence": {
                    "status": "pass",
                    "artifact": interval_path.name,
                    "empirical_coverage": interval_payload["empirical_coverage"],
                    "sample_count": interval_payload["sample_count"],
                    "covered_count": interval_payload["covered_count"],
                    "mean_width": interval_payload["mean_width"],
                    "mean_relative_width": interval_payload["mean_relative_width"],
                    "by_window": interval_payload["by_window"],
                    "evidence_sha256": hashlib.sha256(interval_path.read_bytes()).hexdigest(),
                },
                approval_field: True,
            }
        ),
        encoding="utf-8",
    )
    return review_path


def test_default_protocol_preregisters_full_month_screen_without_prewindow_data() -> None:
    config = copy.deepcopy(experiment.DEFAULT_CONFIG)

    specs = experiment.build_screen_specs(config)

    assert len(specs) == 534
    assert len({spec.command_id for spec in specs}) == len(specs)
    assert {spec.metadata["horizon"] for spec in specs} == {6, 12, 24}
    assert {spec.metadata["quantity"] for spec in specs} == {"price", "return"}
    assert {spec.metadata["lookback"] for spec in specs} == {336, 720, 2160, 4320}
    assert all(spec.metadata["training_floor"] == "2022-06-01" for spec in specs)
    assert all("--start" in spec.argv for spec in specs)
    assert all(spec.argv[spec.argv.index("--start") + 1] == "2022-06-01" for spec in specs)
    assert all(spec.argv[spec.argv.index("--spacing") + 1] == str(spec.metadata["horizon"]) for spec in specs)
    assert all(int(spec.argv[spec.argv.index("--steps") + 1]) > 1 for spec in specs)
    assert all(spec.argv[spec.argv.index("--slippage-bps") + 1] == "0.5" for spec in specs)
    assert all(spec.argv[spec.argv.index("--spread-bps") + 1] == "0.625" for spec in specs)
    assert all(spec.argv[spec.argv.index("--commission-bps-per-side") + 1] == "0.0" for spec in specs)
    assert not any(spec.metadata["shard"]["id"] == "2022-06" for spec in specs)

    july_h6 = next(
        spec
        for spec in specs
        if spec.metadata["shard"]["id"] == "2022-07"
        and spec.metadata["horizon"] == 6
        and spec.metadata["lookback"] == 336
    )
    assert july_h6.argv[july_h6.argv.index("--steps") + 1] == "123"
    params = json.loads(july_h6.argv[july_h6.argv.index("--params-per-method") + 1])
    assert params == {
        "fourier_ols": {"seasonality": 24, "terms": 3, "trend": True},
        "seasonal_naive": {"seasonality": 24},
    }


def test_dry_run_preregisters_then_resume_skips_completed_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "study"
    calls: list[tuple[list[str], dict]] = []
    clean_source = {
        "git_head": "a" * 40,
        "tracked_tree_dirty": False,
        "untracked_runtime_files": [],
        "source_tree_dirty": False,
        "source_status_sha256": hashlib.sha256(b"").hexdigest(),
        "captured_at": "2026-08-27T00:00:00Z",
    }

    def fake_run(invocation: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((list(invocation), dict(kwargs)))
        return _completed(invocation, {"success": True, "symbol": "BTCUSD"})

    monkeypatch.setattr(experiment, "_git_source_state", lambda: clean_source)
    monkeypatch.setattr(experiment.subprocess, "run", fake_run)

    assert (
        experiment.main(
            ["--run-dir", str(run_dir), "--dry-run", "--max-commands", "1", "audit"]
        )
        == 0
    )
    assert calls == []
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"]["screen"]["commands_planned"] == 534
    assert manifest["safety"]["mt5_market_data_only"] is True
    assert "timesfm" in manifest["safety"]["external_pretrained_methods_forbidden"]
    assert manifest["protocol"]["candidate_gate"]["interval_requires_candidate_width_ceiling"] is True
    assert manifest["protocol"]["candidate_gate"]["interval_max_mean_relative_width_cap"] == 0.10
    assert manifest["protocol"]["candidate_gate"]["interval_approval_enabled"] is False
    assert (
        manifest["protocol"]["candidate_gate"]["interval_approval_disabled_reason"]
        == experiment.INTERVAL_APPROVAL_DISABLED_REASON
    )
    assert (
        manifest["protocol"]["candidate_gate"][
            "self_reported_interval_summaries_are_approval_evidence"
        ]
        is False
    )
    assert manifest["protocol"]["research_windows"]["locked_holdout"] == {
        "end": "2026-08-26",
        "locked": True,
        "role": "single-use final holdout",
        "start": "2026-07-01",
    }
    for key in (
        "interval_approval_enabled",
        "interval_approval_disabled_reason",
        "self_reported_interval_summaries_are_approval_evidence",
    ):
        manifest["protocol"]["candidate_gate"].pop(key)
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert (
        experiment.main(
            ["--run-dir", str(run_dir), "--resume", "--max-commands", "1", "audit"]
        )
        == 0
    )
    assert len(calls) == 1
    resumed_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert resumed_manifest["protocol"]["candidate_gate"]["interval_approval_enabled"] is False
    assert calls[0][0][:4] == [sys.executable, "-m", "mtdata", "--json"]
    assert calls[0][0][4] == "symbols_describe"
    assert calls[0][1]["shell"] is False
    assert Path(calls[0][1]["env"]["MTDATA_MODEL_STORE"]) == (
        run_dir / "model_store" / "audit" / "symbol" / "attempt-1"
    )
    assert Path(calls[0][1]["env"]["MTDATA_FORECAST_JOBS_DB"]) == (
        run_dir / "jobs" / "audit" / "symbol" / "attempt-1.sqlite"
    )
    assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"

    first_record = resumed_manifest["stages"]["audit"]["commands"]["symbol"]
    first_raw = run_dir / first_record["raw_path"]
    first_contents = first_raw.read_text(encoding="utf-8")
    assert json.loads(first_contents)["payload"]["success"] is True
    assert (run_dir / "normalized" / "audit.json").exists()

    assert (
        experiment.main(
            ["--run-dir", str(run_dir), "--resume", "--max-commands", "1", "audit"]
        )
        == 0
    )
    assert len(calls) == 2
    assert calls[1][0][4] == "forecast_list_methods"
    assert first_raw.read_text(encoding="utf-8") == first_contents
    capsys.readouterr()


def test_research_command_guard_rejects_trade_and_shell() -> None:
    with pytest.raises(experiment.HarnessError, match="refuses"):
        experiment._validate_research_command(("trade_place", "BTCUSD"))
    with pytest.raises(experiment.HarnessError, match="refuses"):
        experiment._validate_research_command(("trade-close", "1"))
    with pytest.raises(experiment.HarnessError, match="refuses"):
        experiment._validate_research_command(("shell",))


def test_cli_result_and_issue_ledger_redact_secrets_and_dedupe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "study"
    run_dir.mkdir()
    monkeypatch.setenv("BROKER_API_TOKEN", "environment-secret")
    monkeypatch.setenv("MT5_LOGIN", "12345678")
    issue = {
        "severity": "high",
        "category": "cli_contract",
        "reproduction_command": [
            "mtdata-cli",
            "example",
            "--token",
            "argument-secret",
            "--storage",
            "postgresql://alice:db-password@example.test/study",
            "--params",
            '{"password":"nested-secret","window":24}',
        ],
        "context": {
            "password": "context-secret",
            "account_number": "87654321",
            "login": "12345678",
            "note": "environment-secret",
        },
        "impact": "Cannot resume the study",
        "workaround": "Use a local SQLite store",
        "suggested_fix": "Return a stable storage error code",
        "status": "open",
    }

    first, created = experiment.append_issue(run_dir, issue, run_id="study")
    second, created_again = experiment.append_issue(run_dir, issue, run_id="study")

    assert created is True
    assert created_again is False
    assert first["id"].startswith("MTDATA-")
    assert second["id"] == first["id"]
    assert second["occurrences"] == 2
    persisted = (run_dir / "issues.json").read_text(encoding="utf-8")
    for secret in (
        "environment-secret",
        "argument-secret",
        "db-password",
        "context-secret",
        "nested-secret",
        "12345678",
        "87654321",
    ):
        assert secret not in persisted
    saved = json.loads(persisted)["issues"][0]
    assert saved["reproduction_command"][3] == "[REDACTED]"
    assert "nested-secret" not in saved["reproduction_command"][7]
    assert saved["context"]["password"] == "[REDACTED]"
    assert saved["context"]["login"] == "[REDACTED]"
    assert saved["context"]["account_number"] == "[REDACTED]"
    for field in (
        "id",
        "severity",
        "category",
        "observed_at",
        "reproduction_command",
        "context",
        "impact",
        "workaround",
        "suggested_fix",
        "status",
    ):
        assert field in saved


def test_failed_non_json_cli_call_persists_raw_output_and_automatic_issue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    spec = experiment.CommandSpec("bad-json", ("forecast_list_methods", "--limit", "1"), {})
    experiment._register_specs(context, "audit", [spec])

    def fake_run(invocation: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(invocation, 2, stdout="not json", stderr="broken")

    monkeypatch.setattr(experiment.subprocess, "run", fake_run)

    assert experiment.execute_spec(context, "audit", spec) is False
    record = context.manifest["stages"]["audit"]["commands"][spec.command_id]
    raw = json.loads((context.run_dir / record["raw_path"]).read_text(encoding="utf-8"))
    assert raw["status"] == "failed"
    assert "not valid JSON" in raw["parse_error"]
    ledger = json.loads((context.run_dir / "issues.json").read_text(encoding="utf-8"))
    assert len(ledger["issues"]) == 1
    assert ledger["issues"][0]["context"]["command_id"] == "bad-json"


def test_interval_approval_disabled_keeps_holdout_unopened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    with pytest.raises(
        experiment.IntervalApprovalDisabledError,
        match="BTC-INTERVAL-VERIFIER-REQUIRED",
    ):
        experiment.run_holdout(context)
    assert not (context.run_dir / "holdout_lock.json").exists()

    context.manifest["stages"]["screen"]["status"] = "completed"
    experiment._save_manifest(context)
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(_candidate()), encoding="utf-8")
    with pytest.raises(
        experiment.IntervalApprovalDisabledError,
        match="BTC-INTERVAL-VERIFIER-REQUIRED",
    ):
        experiment.freeze_candidate(context, candidate_path)

    validation_calls: list[list[str]] = []

    def validation_run(invocation: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        validation_calls.append(list(invocation))
        return _completed(invocation, _successful_backtest_payload(invocation))

    monkeypatch.setattr(experiment.subprocess, "run", validation_run)
    assert experiment.run_validate(context, candidate_path) == 1
    assert len(validation_calls) == 24
    assert (context.run_dir / "validation_evidence.json").exists()
    assert not (context.run_dir / "validation_decision.json").exists()
    assert context.manifest["stages"]["validate"]["status"] == "awaiting_interval_verifier"

    validation_review = _write_review_artifacts(
        tmp_path,
        context.run_dir / "validation_assessment.json",
        approval_field="eligible_for_holdout",
    )
    with pytest.raises(
        experiment.IntervalApprovalDisabledError,
        match="recomputes coverage and width from hashed raw MT5",
    ):
        experiment.run_validate(context, candidate_path, validation_review)
    assert len(validation_calls) == 24
    assert context.manifest["stages"]["validate"]["status"] == "awaiting_interval_verifier"
    assert context.manifest["stages"]["validate"]["eligible_for_holdout"] is False
    assert not (context.run_dir / "validation_decision.json").exists()
    assert not (context.run_dir / "validation_interval_evidence.json").exists()
    with pytest.raises(
        experiment.IntervalApprovalDisabledError,
        match="BTC-INTERVAL-VERIFIER-REQUIRED",
    ):
        experiment.freeze_candidate(context, candidate_path)
    with pytest.raises(
        experiment.IntervalApprovalDisabledError,
        match="BTC-INTERVAL-VERIFIER-REQUIRED",
    ):
        experiment.run_materialize(context, None)
    with pytest.raises(
        experiment.IntervalApprovalDisabledError,
        match="BTC-INTERVAL-VERIFIER-REQUIRED",
    ):
        experiment.run_shadow(context, None)
    assert not (context.run_dir / "frozen_candidate.json").exists()
    assert not (context.run_dir / "holdout_lock.json").exists()

    assert experiment.generate_report(context) == 0
    report = json.loads((context.run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["interval_approval_enabled"] is False
    assert report["interval_approval_disabled_reason"] == experiment.INTERVAL_APPROVAL_DISABLED_REASON
    assert "Interval approval enabled: no" in (context.run_dir / "report.md").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "candidate",
    [
        _candidate(quantity="return"),
        _candidate(
            pipeline={
                "denoise": None,
                "features": {"indicators": "rsi(14)"},
                "dimred": None,
                "target_spec": None,
            }
        ),
        _candidate(
            pipeline={
                "denoise": None,
                "features": None,
                "dimred": {"method": "pca", "params": {"n_components": 3}},
                "target_spec": None,
            }
        ),
    ],
)
def test_shadow_refuses_mismatched_conformal_calibration(candidate: dict[str, object]) -> None:
    with pytest.raises(experiment.HarnessError, match="cannot reproduce the frozen pipeline"):
        experiment.build_shadow_specs(experiment.DEFAULT_CONFIG, candidate, "2026-08-26")


def test_price_shadow_plan_is_forecast_plus_matching_conformal() -> None:
    specs = experiment.build_shadow_specs(experiment.DEFAULT_CONFIG, _candidate(), "2026-08-26")

    assert [spec.argv[0] for spec in specs] == ["forecast_generate", "forecast_conformal_intervals"]
    assert all("BTCUSD" in spec.argv for spec in specs)
    assert all("--as-of" in spec.argv for spec in specs)
    assert "--quantity" not in specs[1].argv
    assert specs[0].argv[specs[0].argv.index("--ci-alpha") + 1] == "0.1"
    assert specs[1].argv[specs[1].argv.index("--steps") + 1] == "100"


def test_candidate_contract_rejects_silently_dropped_and_unproven_fields() -> None:
    with pytest.raises(experiment.HarnessError, match="unsupported fields"):
        experiment._validate_frozen_candidate(_candidate(target_spec={"column": "close"}))

    noncanonical_pipeline = _candidate()
    noncanonical_pipeline["pipeline"] = {
        "denoise": None,
        "features": None,
        "dimred_method": "pca",
        "target_spec": None,
    }
    with pytest.raises(experiment.HarnessError, match="pipeline is missing: dimred"):
        experiment._validate_frozen_candidate(noncanonical_pipeline)

    unknown_cost = _candidate()
    unknown_cost["costs"] = {**unknown_cost["costs"], "broker_markup_bps": 1.0}
    with pytest.raises(experiment.HarnessError, match="costs have unsupported fields"):
        experiment._validate_frozen_candidate(unknown_cost)

    unknown_calibration = _candidate()
    unknown_calibration["calibration"] = {
        **unknown_calibration["calibration"],
        "unexecuted_option": True,
    }
    with pytest.raises(experiment.HarnessError, match="calibration has unsupported fields"):
        experiment._validate_frozen_candidate(unknown_calibration)

    unusable_width = _candidate()
    unusable_width["calibration"] = {
        **unusable_width["calibration"],
        "max_mean_relative_width": 0.11,
    }
    with pytest.raises(experiment.HarnessError, match="study usability protocol"):
        experiment._validate_frozen_candidate(unusable_width)

    foundation = _candidate(method="chronos_bolt")
    foundation["provenance"] = {
        **foundation["provenance"],
        "external_model_id": "amazon/chronos-bolt-small",
    }
    with pytest.raises(experiment.HarnessError, match="MT5-provided market data"):
        experiment._validate_frozen_candidate(foundation)

    external_screen = copy.deepcopy(experiment.DEFAULT_CONFIG)
    external_screen["screen"]["methods"] = ["timesfm"]
    with pytest.raises(experiment.HarnessError, match="Externally pretrained"):
        experiment.build_screen_specs(external_screen)

    external_variant = copy.deepcopy(experiment.DEFAULT_CONFIG)
    external_variant["screen"]["variants"] = [
        {"id": "external", "methods": ["pretrained-acme"]}
    ]
    with pytest.raises(experiment.HarnessError, match="Externally pretrained"):
        experiment.build_screen_specs(external_variant)

    external_tune = copy.deepcopy(experiment.DEFAULT_CONFIG)
    external_tune["tune"]["experiments"] = [{"id": "external", "method": "chronos2"}]
    with pytest.raises(experiment.HarnessError, match="Externally pretrained"):
        experiment.build_tune_specs(external_tune, Path("study"))

    external_baseline = _candidate()
    external_baseline["provenance"]["baseline"]["method"] = "timesfm"
    with pytest.raises(experiment.HarnessError, match="Externally pretrained"):
        experiment._validate_frozen_candidate(external_baseline)


def test_validation_decision_rejects_directionally_and_economically_bad_candidate(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    candidate = experiment._validate_frozen_candidate(_candidate())
    specs = experiment.build_validation_specs(context.config, candidate)
    experiment._register_specs(
        context,
        "validate",
        specs,
        extra_hash_input=experiment._sha256_json(candidate),
    )
    for spec in specs:
        raw_path = experiment._raw_path(context, "validate", spec.command_id)
        experiment._atomic_write_json(
            raw_path,
            {
                "payload": {
                    "success": True,
                    "results": {
                        "theta": {
                            "success": True,
                            "num_tests": 5,
                            "successful_tests": 5,
                            "details": [
                                {
                                    "success": True,
                                    "forecast": [101.0],
                                    "actual": [99.0],
                                    "signal_reference_price": 100.0,
                                    "entry_price": 100.0,
                                }
                                for _ in range(5)
                            ],
                        },
                        "naive": {
                            "success": True,
                            "num_tests": 5,
                            "successful_tests": 5,
                            "details": [
                                {
                                    "success": True,
                                    "forecast": [99.0],
                                    "actual": [99.0],
                                    "signal_reference_price": 100.0,
                                    "entry_price": 100.0,
                                }
                                for _ in range(5)
                            ],
                        },
                    },
                }
            },
        )

    decision = experiment._compute_validation_decision(
        context,
        candidate,
        experiment._sha256_json(candidate),
        specs,
    )

    assert decision["gates"]["minimum_evidence"]["status"] == "pass"
    assert decision["gates"]["statistical"]["status"] == "fail"
    assert decision["gates"]["economic"]["status"] == "fail"
    assert decision["gates"]["interval"]["status"] == "not_yet_available"
    assert decision["eligible_for_holdout"] is False


def test_shard_contract_rejects_anchor_leakage() -> None:
    spec = next(iter(experiment.build_screen_specs(experiment.DEFAULT_CONFIG)))
    shard = spec.metadata["shard"]
    payload = {
        "analysis_time_window": {
            "evaluation_start": "2022-05-31T23:00:00Z",
            "evaluation_end": f"{shard['end']}T00:00:00Z",
        }
    }

    assert "precedes registered shard start" in (experiment._shard_contract_error(spec, payload) or "")


def test_materialization_and_shadow_cache_contracts_and_staleness() -> None:
    candidate = experiment._validate_frozen_candidate(_candidate())
    materialize = experiment.build_materialize_specs(
        experiment.DEFAULT_CONFIG,
        candidate,
        "2026-08-26T00:00:00Z",
    )[0]
    assert materialize.argv[materialize.argv.index("--model-cache") + 1] == "reuse"

    shadow = experiment.build_shadow_specs(
        experiment.DEFAULT_CONFIG,
        candidate,
        "2026-08-27T00:00:00Z",
        model_id="mlf_rf/BTCUSD-H1/hash",
    )[0]
    assert shadow.argv[shadow.argv.index("--model-cache") + 1] == "require_existing"
    assert shadow.argv[shadow.argv.index("--model-id") + 1] == "mlf_rf/BTCUSD-H1/hash"

    artifact = {"as_of": "2026-08-26T00:00:00Z", "refit_interval_bars": 24}
    experiment._enforce_materialization_freshness(candidate, artifact, "2026-08-27T00:00:00Z")
    with pytest.raises(experiment.HarnessError, match="above the 24-bar limit"):
        experiment._enforce_materialization_freshness(candidate, artifact, "2026-08-27T01:00:00Z")


def test_shadow_command_stage_remains_active_append_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    spec = experiment.CommandSpec("shadow-observation", ("forecast_list_methods", "--limit", "1"), {})
    monkeypatch.setattr(
        experiment.subprocess,
        "run",
        lambda invocation, **kwargs: _completed(invocation),
    )

    assert (
        experiment._run_command_stage(
            context,
            "shadow",
            [spec],
            allow_append=True,
            continuous=True,
        )
        == 0
    )
    assert context.manifest["stages"]["shadow"]["status"] == "active"
    assert context.manifest["stages"]["shadow"]["lifecycle"] == "append_only"


def test_history_quality_is_preserved_in_normalized_summary() -> None:
    history_quality = {"status": "sufficient", "minimum_bars": 336, "observed_bars": 720}

    assert experiment._summarize_payload({"success": True, "history_quality": history_quality})[
        "history_quality"
    ] == history_quality
