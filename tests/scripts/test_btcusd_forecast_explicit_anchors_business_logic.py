from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts import btcusd_forecast_experiment as experiment


def _explicit_config(
    *,
    every_days: int = 7,
    grid_start: str = "2022-07-04",
    grid_end: str = "2022-12-26",
    horizons: list[int] | None = None,
) -> dict[str, object]:
    config = copy.deepcopy(experiment.DEFAULT_CONFIG)
    config["research_windows"]["a1"] = {
        "start": "2022-07-01",
        "end": "2022-12-31",
        "role": "development screen",
        "locked": False,
    }
    config["screen"].update(
        {
            "window": "a1",
            "timeframes": ["H1"],
            "horizons": horizons or [6, 12, 24],
            "quantities": ["return"],
            "lookbacks": [720],
            "methods": ["naive"],
            "variants": [{"id": "raw"}, {"id": "control"}],
            "anchor_grid": {
                "id": "a1-weekly",
                "start": grid_start,
                "end": grid_end,
                "every_days": every_days,
                "hour_utc": 0,
                "history_start": "2022-06-01",
            },
        }
    )
    return config


def _explicit_payload(spec: experiment.CommandSpec) -> dict[str, object]:
    anchors = list(spec.metadata["expected_anchors"])
    horizon = int(spec.metadata["horizon"])
    details = []
    for anchor in anchors:
        anchor_at = datetime.fromisoformat(anchor.replace("Z", "+00:00"))
        details.append(
            {
                "success": True,
                "anchor": anchor,
                "training_bars_used": int(spec.metadata["lookback"]),
                "forecast": [0.001] * horizon,
                "actual": [0.002] * horizon,
                "actual_timestamps": [
                    (anchor_at + timedelta(hours=step))
                    .isoformat(timespec="seconds")
                    .replace("+00:00", "Z")
                    for step in range(1, horizon + 1)
                ],
            }
        )
    method = str(spec.metadata["methods"][0])
    count = len(anchors)
    return {
        "success": True,
        "complete_success": True,
        "status": "complete",
        "methods_total": 1,
        "methods_succeeded": 1,
        "methods_complete": 1,
        "methods_partial": 0,
        "methods_failed": 0,
        "anchor_tests_planned": count,
        "anchor_tests_succeeded": count,
        "anchor_tests_failed": 0,
        "backtest_plan": {
            "anchor_mode": "explicit",
            "anchor_resolution": "exact_bar_open",
            "requested_anchors": anchors,
            "resolved_anchors": anchors,
            "runs_requested": count,
            "runs_used": count,
        },
        "results": {
            method: {
                "success": True,
                "complete_success": True,
                "status": "complete",
                "num_tests": count,
                "successful_tests": count,
                "failed_tests": 0,
                "details": details,
            }
        },
    }


def test_explicit_grid_is_generated_once_and_shared_by_every_screen_command() -> None:
    specs = experiment.build_screen_specs(_explicit_config())

    assert len(specs) == 6
    expected = [
        (
            datetime(2022, 7, 4, tzinfo=timezone.utc) + timedelta(days=7 * index)
        )
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
        for index in range(26)
    ]
    assert {tuple(spec.metadata["expected_anchors"]) for spec in specs} == {
        tuple(expected)
    }
    assert {spec.metadata["expected_anchors_sha256"] for spec in specs} == {
        experiment._sha256_json(expected)
    }
    assert {spec.metadata["anchor_grid_id"] for spec in specs} == {"a1-weekly"}
    assert {spec.metadata["steps"] for spec in specs} == {26}
    assert {spec.metadata["spacing"] for spec in specs} == {168}
    for spec in specs:
        assert "--steps" not in spec.argv
        assert "--spacing" not in spec.argv
        assert spec.argv[spec.argv.index("--start") + 1] == "2022-06-01"
        assert json.loads(spec.argv[spec.argv.index("--anchors") + 1]) == expected


def test_dense_grid_reaches_183_preregistered_daily_origins() -> None:
    specs = experiment.build_screen_specs(
        _explicit_config(
            every_days=1,
            grid_start="2022-07-01",
            grid_end="2022-12-30",
            horizons=[24],
        )
    )

    assert len(specs) == 2
    assert all(spec.metadata["steps"] == 183 for spec in specs)
    assert all(spec.metadata["spacing"] == 24 for spec in specs)
    assert all(
        spec.metadata["expected_anchors"][-1] == "2022-12-30T00:00:00Z"
        for spec in specs
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"history_start": "2022-07-01"}, "does not provide lookback"),
        ({"start": "2022-06-30"}, "inside the registered"),
        ({"end": "2023-01-02"}, "inside the registered"),
        ({"every_days": 0}, "positive integer"),
    ],
)
def test_explicit_grid_rejects_invalid_or_leaky_schedules(
    change: dict[str, object],
    message: str,
) -> None:
    config = _explicit_config()
    config["screen"]["anchor_grid"].update(change)

    with pytest.raises(experiment.HarnessError, match=message):
        experiment.build_screen_specs(config)


def test_explicit_grid_rejects_overlap_and_out_of_role_targets() -> None:
    overlap = _explicit_config(every_days=1, horizons=[25])
    with pytest.raises(experiment.HarnessError, match="must not overlap"):
        experiment.build_screen_specs(overlap)

    outside = _explicit_config(
        every_days=1,
        grid_start="2022-07-01",
        grid_end="2022-12-31",
        horizons=[24],
    )
    with pytest.raises(experiment.HarnessError, match="outside"):
        experiment.build_screen_specs(outside)


def test_stage_registration_preserves_canonical_anchor_inventory(tmp_path: Path) -> None:
    config = _explicit_config(horizons=[24])
    specs = experiment.build_screen_specs(config)
    run_dir = tmp_path / "study"
    context = experiment.RunContext(
        run_dir=run_dir,
        manifest_path=run_dir / "manifest.json",
        manifest={"stages": {}},
        config=config,
        dry_run=True,
        timeout=30.0,
        max_commands=None,
        fail_fast=True,
        enforce_execution_integrity=False,
    )

    experiment._register_specs(context, "screen", specs)

    stage = context.manifest["stages"]["screen"]
    grid = stage["anchor_grids"]["a1-weekly"]
    assert grid["count"] == 26
    assert grid["anchors"] == specs[0].metadata["expected_anchors"]
    assert grid["sha256"] == experiment._sha256_json(grid["anchors"])
    assert all(
        command["metadata"]["expected_anchors_sha256"] == grid["sha256"]
        for command in stage["commands"].values()
    )


def test_explicit_manifest_records_history_only_training_floor_policy(
    tmp_path: Path,
) -> None:
    config = _explicit_config()
    config["screen"]["spacing_policy"] = "equal_to_horizon"
    config["screen"]["steps_per_shard"] = "auto_month"

    manifest = experiment._new_manifest(
        "a1-weekly",
        config,
        tmp_path / "study",
        source_state={"git_head": "test"},
        runtime_identity={"python": "test"},
    )

    protocol = manifest["protocol"]
    baseline = protocol["baseline_matrix"]
    assert baseline["anchor_grid"] == config["screen"]["anchor_grid"]
    assert baseline["spacing_policy"] == "explicit_anchor_grid"
    assert baseline["steps_per_shard"] is None
    assert "anchor_grid.history_start" in protocol["screen_training_floor_policy"]
    assert "inside the registered role window" in protocol[
        "screen_training_floor_policy"
    ]


def test_explicit_screen_contract_accepts_exact_plan_and_timestamp_paths() -> None:
    specs = experiment.build_screen_specs(_explicit_config(horizons=[6, 24]))
    by_horizon = {
        int(spec.metadata["horizon"]): spec
        for spec in specs
        if spec.metadata["variant"] == "raw"
    }
    payloads = {horizon: _explicit_payload(spec) for horizon, spec in by_horizon.items()}

    for horizon, spec in by_horizon.items():
        assert (
            experiment._screen_collection_contract_error(
                spec.metadata,
                payloads[horizon],
            )
            is None
        )
    h6_times = payloads[6]["results"]["naive"]["details"][0][
        "actual_timestamps"
    ]
    h24_times = payloads[24]["results"]["naive"]["details"][0][
        "actual_timestamps"
    ]
    assert h6_times == h24_times[:6]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("requested", "requested anchors"),
        ("resolved", "resolved anchors"),
        ("resolution", "exact_bar_open"),
        ("timestamps_missing", "actual timestamps"),
        ("timestamps_shifted", "shifted or gapped"),
    ],
)
def test_explicit_screen_contract_rejects_shifted_or_incomplete_evidence(
    mutation: str,
    message: str,
) -> None:
    spec = next(
        spec
        for spec in experiment.build_screen_specs(_explicit_config(horizons=[6]))
        if spec.metadata["variant"] == "raw"
    )
    payload = _explicit_payload(spec)
    plan = payload["backtest_plan"]
    details = payload["results"]["naive"]["details"]
    if mutation == "requested":
        plan["requested_anchors"] = plan["requested_anchors"][:-1]
    elif mutation == "resolved":
        plan["resolved_anchors"] = list(reversed(plan["resolved_anchors"]))
    elif mutation == "resolution":
        plan["anchor_resolution"] = "nearest"
    elif mutation == "timestamps_missing":
        details[0].pop("actual_timestamps")
    else:
        details[0]["actual_timestamps"][0] = "2022-07-04T02:00:00Z"

    error = experiment._screen_collection_contract_error(spec.metadata, payload)

    assert message in (error or "")
