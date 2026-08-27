from __future__ import annotations

import json
import subprocess
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


def _explicit_payload(*, anchor_tests_failed: int = 11) -> dict[str, object]:
    return {
        "success": True,
        "complete_success": False,
        "status": "partial",
        "methods_total": 2,
        "methods_succeeded": 2,
        "methods_complete": 1,
        "methods_partial": 1,
        "methods_failed": 0,
        "complete_methods": ["naive"],
        "partial_methods": ["arima"],
        "anchor_tests_planned": 62,
        "anchor_tests_succeeded": 51,
        "anchor_tests_failed": anchor_tests_failed,
        "results": {
            "arima": {
                "success": True,
                "complete_success": False,
                "status": "partial",
                "num_tests": 31,
                "successful_tests": 20,
                "failed_tests": 11,
            },
            "naive": {
                "success": True,
                "complete_success": True,
                "status": "complete",
                "num_tests": 31,
                "successful_tests": 31,
                "failed_tests": 0,
            },
        },
    }


def _sampling_inputs(
    by_window: dict[str, dict[str, float | int]],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    sample_count = sum(int(row["sample_count"]) for row in by_window.values())
    covered_count = sum(int(row["covered_count"]) for row in by_window.values())
    mean_relative_width = sum(
        float(row["mean_relative_width"]) * int(row["sample_count"])
        for row in by_window.values()
    ) / sample_count
    artifact: dict[str, object] = {
        "windows": list(by_window),
        "sample_count": sample_count,
        "covered_count": covered_count,
        "empirical_coverage": covered_count / sample_count,
        "mean_relative_width": mean_relative_width,
        "by_window": by_window,
    }
    assessment = {
        "windows": list(by_window),
        "thresholds": {"minimum_opportunities": 100},
    }
    interval = {
        "sample_count": sample_count,
        "covered_count": covered_count,
        "by_window": by_window,
    }
    return assessment, artifact, interval


def test_legacy_partial_anchor_failures_are_normalized_and_auto_logged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    spec = experiment.CommandSpec(
        "partial-anchors",
        ("forecast_backtest_run", "BTCUSD", "--methods", "arima"),
        {},
    )
    experiment._register_specs(context, "screen", [spec])
    payload = {
        "success": True,
        "methods_failed": 0,
        "results": {
            "arima": {"success": True, "num_tests": 31, "successful_tests": 0},
            "naive": {"success": True, "num_tests": 31, "successful_tests": 31},
        },
    }
    monkeypatch.setattr(
        experiment.subprocess,
        "run",
        lambda invocation, **kwargs: _completed(invocation, payload),
    )

    assert experiment.execute_spec(context, "screen", spec) is True
    result = experiment.write_normalized_stage(context, "screen")["commands"][0]["result"]

    assert result["derived_anchor_status"]["partial_anchor_failures"] == 31
    assert result["derived_anchor_status"]["legacy_methods_failed_ambiguity"] is True
    ledger = json.loads((context.run_dir / "issues.json").read_text(encoding="utf-8"))
    assert ledger["issues"][0]["id"] == "BTC-R017"


def test_explicit_partial_contract_suppresses_legacy_r017(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    spec = experiment.CommandSpec(
        "explicit-partial-anchors",
        ("forecast_backtest_run", "BTCUSD", "--methods", "arima", "naive"),
        {},
    )
    experiment._register_specs(context, "screen", [spec])
    monkeypatch.setattr(
        experiment.subprocess,
        "run",
        lambda invocation, **kwargs: _completed(invocation, _explicit_payload()),
    )

    assert experiment.execute_spec(context, "screen", spec) is True
    result = experiment.write_normalized_stage(context, "screen")["commands"][0]["result"]

    assert result["status"] == "partial"
    assert result["complete_success"] is False
    assert result["partial_methods"] == ["arima"]
    assert result["anchor_tests_failed"] == 11
    assert result["derived_anchor_status"]["explicit_partial_contract_consistent"] is True
    assert result["derived_anchor_status"]["legacy_methods_failed_ambiguity"] is False
    ledger = json.loads((context.run_dir / "issues.json").read_text(encoding="utf-8"))
    assert ledger["issues"] == []


def test_inconsistent_explicit_partial_counts_fail_command_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    spec = experiment.CommandSpec(
        "inconsistent-partial-anchors",
        ("forecast_backtest_run", "BTCUSD", "--methods", "arima", "naive"),
        {},
    )
    experiment._register_specs(context, "screen", [spec])
    monkeypatch.setattr(
        experiment.subprocess,
        "run",
        lambda invocation, **kwargs: _completed(
            invocation,
            _explicit_payload(anchor_tests_failed=10),
        ),
    )

    assert experiment.execute_spec(context, "screen", spec) is False
    record = context.manifest["stages"]["screen"]["commands"][spec.command_id]
    raw = json.loads(
        (context.run_dir / record["raw_path"]).read_text(encoding="utf-8")
    )
    assert "disagree" in raw["contract_error"]
    ledger = json.loads((context.run_dir / "issues.json").read_text(encoding="utf-8"))
    assert ledger["issues"][0]["severity"] == "high"


def test_interval_approval_rejects_non_price_quantity() -> None:
    assessment = {
        "candidate_hash": "candidate-hash",
        "candidate_spec": {
            "pipeline_hash": "candidate-hash",
            "timeframe": "H1",
            "horizon": 12,
            "quantity": "return",
        },
    }
    artifact = {
        "candidate_hash": "candidate-hash",
        "pipeline_hash": "candidate-hash",
        "timeframe": "H1",
        "horizon": 12,
        "quantity": "return",
    }

    with pytest.raises(experiment.HarnessError, match="quantity=price only"):
        experiment._validate_interval_candidate_identity(assessment, artifact)


def test_interval_sampling_rejects_bad_window_width_hidden_by_aggregate() -> None:
    assessment, artifact, interval = _sampling_inputs(
        {
            "validation": {
                "sample_count": 100,
                "covered_count": 90,
                "empirical_coverage": 0.90,
                "mean_relative_width": 0.11,
            },
            "confirmation": {
                "sample_count": 100,
                "covered_count": 90,
                "empirical_coverage": 0.90,
                "mean_relative_width": 0.01,
            },
        }
    )

    with pytest.raises(experiment.HarnessError, match="window 'validation'.*usability ceiling"):
        experiment._validated_interval_sampling(
            assessment,
            artifact,
            interval,
            max_mean_relative_width=0.10,
        )


def test_interval_sampling_requires_coverage_wilson_lower_bound() -> None:
    assessment, artifact, interval = _sampling_inputs(
        {
            "validation": {
                "sample_count": 100,
                "covered_count": 80,
                "empirical_coverage": 0.80,
                "mean_relative_width": 0.05,
            }
        }
    )

    with pytest.raises(experiment.HarnessError, match="Wilson 95% coverage lower bound"):
        experiment._validated_interval_sampling(
            assessment,
            artifact,
            interval,
            max_mean_relative_width=0.10,
        )


def test_interval_sampling_rejects_inconsistent_aggregate_counts() -> None:
    assessment, artifact, interval = _sampling_inputs(
        {
            "validation": {
                "sample_count": 100,
                "covered_count": 90,
                "empirical_coverage": 0.90,
                "mean_relative_width": 0.05,
            }
        }
    )
    artifact["sample_count"] = 101
    artifact["empirical_coverage"] = 90 / 101
    interval["sample_count"] = 101

    with pytest.raises(experiment.HarnessError, match="Aggregate interval counts"):
        experiment._validated_interval_sampling(
            assessment,
            artifact,
            interval,
            max_mean_relative_width=0.10,
        )
