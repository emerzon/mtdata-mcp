"""Validate Iteration 4 feature-smoke evidence without calling MT5.

The validator reads only an existing BTCUSD experiment bundle.  It verifies the
source/runtime pin, immutable raw responses, attempt-local stores, and matched
feature/raw forecast evidence.  It deliberately computes no performance metric.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
VALIDATOR_VERSION = "1.0.0"
VALIDATOR_NAME = "btcusd_iteration4_smoke"
SUPPORTED_HARNESS_VERSIONS = frozenset({"0.2.0", "0.3.0"})
SOURCE_ERROR = "BTC-SMOKE-SOURCE"
CONFIG_ERROR = "BTC-SMOKE-CONFIG"
PLAN_ERROR = "BTC-SMOKE-PLAN"
RAW_ERROR = "BTC-SMOKE-RAW"
ATTEMPT_ERROR = "BTC-SMOKE-ATTEMPT"
RESULT_ERROR = "BTC-SMOKE-RESULT"
PAIR_ERROR = "BTC-SMOKE-PAIR"
NOOP_ERROR = "BTC-SMOKE-FEATURE-NOOP"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
PACKAGE_DISTRIBUTIONS = (
    "mtdata-mcp",
    "numpy",
    "pandas",
    "mlforecast",
    "lightgbm",
    "scikit-learn",
    "pandas-ta-classic",
    "TA-Lib",
)
EXPECTED_PARAMS: dict[str, dict[str, Any]] = {
    "mlf_lightgbm": {
        "lags": [1, 2, 3, 6, 12, 24],
        "learning_rate": 0.05,
        "max_depth": 5,
        "n_estimators": 50,
        "num_leaves": 15,
    },
    "mlf_rf": {
        "lags": [1, 2, 3, 6, 12, 24],
        "max_depth": 8,
        "n_estimators": 100,
    },
}
EXPECTED_FEATURES = (
    {
        "indicators": "rsi(14),roc(12)",
        "observed_future_policy": "carry_forward",
    },
    {
        "future_covariates": ["hour", "dow"],
        "indicators": "natr(14)",
        "observed_future_policy": "carry_forward",
    },
)
EXPECTED_FEATURE_COLUMNS = (
    (EXPECTED_FEATURES[0], ["RSI_14", "ROC_12"]),
    (EXPECTED_FEATURES[1], ["NATR_14", "hr_sin", "hr_cos", "dow_sin", "dow_cos"]),
)
EXPECTED_PAIRS = {
    "lgbm-momentum": "lgbm-raw",
    "lgbm-volatility-time": "lgbm-raw",
    "rf-momentum": "rf-raw",
    "rf-volatility-time": "rf-raw",
}
EXPECTED_VARIANT_METHOD = {
    "lgbm-raw": "mlf_lightgbm",
    "lgbm-momentum": "mlf_lightgbm",
    "lgbm-volatility-time": "mlf_lightgbm",
    "rf-raw": "mlf_rf",
    "rf-momentum": "mlf_rf",
    "rf-volatility-time": "mlf_rf",
}
EXPECTED_VARIANT_ORDER = tuple(EXPECTED_VARIANT_METHOD)
EXPECTED_VARIANT_FEATURE = {
    "lgbm-momentum": EXPECTED_FEATURES[0],
    "lgbm-volatility-time": EXPECTED_FEATURES[1],
    "rf-momentum": EXPECTED_FEATURES[0],
    "rf-volatility-time": EXPECTED_FEATURES[1],
}
ENTRY_FIELDS = (
    "entry_price",
    "signal_reference_price",
    "entry_time",
    "entry_price_source",
)


class SmokeValidationError(RuntimeError):
    """A fail-closed protocol or evidence violation."""

    def __init__(
        self,
        code: str,
        message: str,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.context = dict(context or {})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, code: str) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SmokeValidationError(code, f"Could not read valid JSON from {path}: {exc}")


def _slug(value: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value).strip()).strip("-._")
    return text.lower() or "item"


def _inside_path(run_dir: Path, raw_path: Any, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path or Path(raw_path).is_absolute():
        raise SmokeValidationError(RAW_ERROR, f"{label} is not a relative path")
    relative = Path(raw_path)
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise SmokeValidationError(RAW_ERROR, f"{label} is not a canonical relative path")
    candidate = run_dir
    for part in relative.parts:
        candidate /= part
        if candidate.is_symlink():
            raise SmokeValidationError(RAW_ERROR, f"{label} traverses a symbolic link")
    path = candidate.resolve()
    try:
        path.relative_to(run_dir)
    except ValueError as exc:
        raise SmokeValidationError(RAW_ERROR, f"{label} escapes the run directory") from exc
    return path


def _require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise SmokeValidationError(RAW_ERROR, f"{label} is not a SHA-256 digest")
    return value


def _exact_scalar(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, int):
        return type(actual) is int and actual == expected
    if isinstance(expected, float):
        return type(actual) in {int, float} and actual == expected
    return actual == expected


def _strict_int(value: Any, *, minimum: int | None = None) -> int | None:
    if type(value) is not int or (minimum is not None and value < minimum):
        return None
    return value


def _semantic_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _semantic_columns_match(actual: Any, expected: Sequence[str]) -> bool:
    if not isinstance(actual, list) or not all(
        isinstance(item, str) and item for item in actual
    ):
        return False
    return Counter(_semantic_token(item) for item in actual) == Counter(
        _semantic_token(item) for item in expected
    )


def _file_and_hash(
    run_dir: Path,
    raw_path: Any,
    expected_hash: Any,
    label: str,
    hashes: dict[str, str],
) -> Path:
    path = _inside_path(run_dir, raw_path, label)
    expected = _require_hash(expected_hash, f"{label} hash")
    if not path.is_file() or path.is_symlink():
        raise SmokeValidationError(RAW_ERROR, f"{label} is missing or is not a regular file")
    actual = _sha256_file(path)
    hashes[path.relative_to(run_dir).as_posix()] = actual
    if actual != expected:
        raise SmokeValidationError(RAW_ERROR, f"{label} SHA-256 does not match")
    return path


def _package_versions() -> dict[str, dict[str, str | None]]:
    versions: dict[str, dict[str, str | None]] = {}
    for distribution in PACKAGE_DISTRIBUTIONS:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = {"status": "missing", "version": None}
        else:
            versions[distribution] = {"status": "installed", "version": version}
    return versions


def _runtime_identity() -> dict[str, Any]:
    return {
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "package_versions": _package_versions(),
    }


def _git_source_state() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            shell=False,
        )
        if completed.returncode:
            detail = (completed.stderr or completed.stdout or "git failed").strip()
            raise SmokeValidationError(SOURCE_ERROR, f"Could not inspect source: {detail}")
        return completed.stdout.strip()

    head = git("rev-parse", "HEAD")
    tracked = git("status", "--porcelain", "--untracked-files=no")
    untracked = git(
        "ls-files",
        "--others",
        "--exclude-standard",
        "--",
        "src",
        "scripts",
        "mtdata",
        "mtdata.py",
        "sitecustomize.py",
        "usercustomize.py",
    ).splitlines()
    status = {"tracked_status": tracked, "untracked_runtime_files": untracked}
    return {
        "git_head": head,
        "tracked_tree_dirty": bool(tracked),
        "untracked_runtime_files": untracked,
        "source_tree_dirty": bool(tracked or untracked),
        "source_status_sha256": _sha256_json(status),
    }


def _verify_source_runtime(
    manifest: Mapping[str, Any],
    *,
    current_source: Mapping[str, Any] | None,
    current_runtime: Mapping[str, Any] | None,
) -> None:
    recorded_source = manifest.get("source")
    recorded_runtime = manifest.get("runtime")
    source = dict(current_source or _git_source_state())
    runtime = dict(current_runtime or _runtime_identity())
    if not isinstance(recorded_source, Mapping) or not isinstance(
        recorded_runtime, Mapping
    ):
        raise SmokeValidationError(SOURCE_ERROR, "Manifest provenance is incomplete")
    if recorded_source.get("source_tree_dirty") is not False:
        raise SmokeValidationError(SOURCE_ERROR, "Run was not pinned to a clean source tree")
    if source.get("source_tree_dirty") is not False:
        raise SmokeValidationError(SOURCE_ERROR, "Current source tree is dirty")
    if recorded_source.get("git_head") != source.get("git_head"):
        raise SmokeValidationError(SOURCE_ERROR, "Current git HEAD differs from the run")
    if recorded_source.get("source_status_sha256") != source.get(
        "source_status_sha256"
    ):
        raise SmokeValidationError(SOURCE_ERROR, "Current source status differs from the run")
    for key, value in runtime.items():
        if recorded_runtime.get(key) != value:
            raise SmokeValidationError(
                SOURCE_ERROR,
                f"Current runtime {key!r} differs from the run",
            )


def _flag_value(argv: Sequence[Any], flag: str) -> Any:
    try:
        index = argv.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(argv):
        return None
    return argv[index + 1]


def _without_features(argv: Sequence[Any]) -> list[Any]:
    values = list(argv)
    try:
        index = values.index("--features")
    except ValueError:
        return values
    if index + 1 >= len(values):
        raise SmokeValidationError(PLAN_ERROR, "--features has no JSON value")
    return values[:index] + values[index + 2 :]


def _json_flag(argv: Sequence[Any], flag: str, *, required: bool = True) -> Any:
    raw = _flag_value(argv, flag)
    if raw is None:
        if required:
            raise SmokeValidationError(PLAN_ERROR, f"Command omitted {flag}")
        return None
    if not isinstance(raw, str):
        raise SmokeValidationError(PLAN_ERROR, f"{flag} is not encoded JSON")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SmokeValidationError(PLAN_ERROR, f"{flag} is not valid JSON") from exc


def _validate_flag_shape(command_id: str, argv: Sequence[str], *, feature: bool) -> None:
    tokens = list(argv[2:])
    if len(tokens) % 2:
        raise SmokeValidationError(PLAN_ERROR, f"{command_id} has an unpaired CLI flag")
    flags = tokens[::2]
    values = tokens[1::2]
    if not all(flag.startswith("--") for flag in flags) or any(
        value.startswith("--") for value in values
    ):
        raise SmokeValidationError(PLAN_ERROR, f"{command_id} CLI arguments are malformed")
    expected = {
        "--timeframe",
        "--horizon",
        "--steps",
        "--spacing",
        "--lookback",
        "--methods",
        "--quantity",
        "--start",
        "--end",
        "--detail",
        "--params",
        "--slippage-bps",
        "--spread-bps",
        "--commission-bps-per-side",
        "--trade-threshold",
    }
    if feature:
        expected.add("--features")
    if set(flags) != expected or any(count != 1 for count in Counter(flags).values()):
        raise SmokeValidationError(
            PLAN_ERROR,
            f"{command_id} CLI flags do not exactly match the frozen smoke protocol",
        )


def _validate_command_plan(command_id: str, record: Mapping[str, Any]) -> dict[str, Any]:
    argv = record.get("argv")
    metadata = record.get("metadata")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise SmokeValidationError(PLAN_ERROR, f"{command_id} has invalid argv")
    if not isinstance(metadata, Mapping):
        raise SmokeValidationError(PLAN_ERROR, f"{command_id} has invalid metadata")
    expected_metadata = {
        "kind": "baseline_screen",
        "window": "development",
        "timeframe": "H1",
        "horizon": 12,
        "quantity": "return",
        "lookback": 720,
        "steps": 1,
        "spacing": 12,
        "detail": "full",
        "training_floor": "2024-05-01",
    }
    for key, expected in expected_metadata.items():
        if not _exact_scalar(metadata.get(key), expected):
            raise SmokeValidationError(
                PLAN_ERROR,
                f"{command_id} metadata {key}={metadata.get(key)!r}, expected {expected!r}",
            )
    if argv[:2] != ["forecast_backtest_run", "BTCUSD"]:
        raise SmokeValidationError(PLAN_ERROR, f"{command_id} is not a BTCUSD backtest")
    expected_shard = {
        "id": "window-end",
        "kind": "exact_window_end",
        "start": "2024-05-01",
        "end": "2024-05-31",
    }
    if _canonical_json(metadata.get("shard")) != _canonical_json(expected_shard):
        raise SmokeValidationError(PLAN_ERROR, f"{command_id} shard is not frozen")
    expected_flags = {
        "--timeframe": "H1",
        "--horizon": "12",
        "--steps": "1",
        "--spacing": "12",
        "--lookback": "720",
        "--quantity": "return",
        "--detail": "full",
        "--start": "2024-05-01",
        "--end": "2024-05-31",
        "--slippage-bps": "0.5",
        "--spread-bps": "0.625",
        "--commission-bps-per-side": "0.0",
        "--trade-threshold": "0.0",
    }
    for flag, expected in expected_flags.items():
        if _flag_value(argv, flag) != expected:
            raise SmokeValidationError(
                PLAN_ERROR, f"{command_id} {flag} must equal {expected}"
            )
    methods = metadata.get("methods")
    if not isinstance(methods, list) or len(methods) != 1 or methods[0] not in EXPECTED_PARAMS:
        raise SmokeValidationError(PLAN_ERROR, f"{command_id} must plan one Iteration 4 adapter")
    method = methods[0]
    if _flag_value(argv, "--methods") != method:
        raise SmokeValidationError(PLAN_ERROR, f"{command_id} method argv disagrees with metadata")
    if "--params-per-method" in argv or "--denoise" in argv or "--dimred" in argv:
        raise SmokeValidationError(PLAN_ERROR, f"{command_id} contains a forbidden smoke transform")
    params = _json_flag(argv, "--params")
    if _canonical_json(params) != _canonical_json(
        EXPECTED_PARAMS[method]
    ) or _canonical_json(metadata.get("params")) != _canonical_json(params):
        raise SmokeValidationError(PLAN_ERROR, f"{command_id} parameters are not frozen")
    variant = metadata.get("variant")
    if not isinstance(variant, str) or not variant:
        raise SmokeValidationError(PLAN_ERROR, f"{command_id} omitted its variant")
    features = _json_flag(argv, "--features", required=False)
    _validate_flag_shape(command_id, argv, feature=features is not None)
    if _canonical_json(metadata.get("features")) != _canonical_json(features):
        raise SmokeValidationError(
            PLAN_ERROR, f"{command_id} feature argv disagrees with metadata"
        )
    return {
        "argv": argv,
        "metadata": metadata,
        "method": method,
        "params": params,
        "variant": variant,
        "features": features,
    }


def _validate_plan(  # noqa: C901 - fail-closed protocol validation is intentionally linear
    manifest: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[dict[str, Mapping[str, Any]], dict[str, str], dict[str, dict[str, Any]]]:
    if config.get("symbol") != "BTCUSD" or not _exact_scalar(config.get("seed"), 42):
        raise SmokeValidationError(CONFIG_ERROR, "Smoke config must use BTCUSD and seed 42")
    if not _exact_scalar(config.get("schema_version"), 1):
        raise SmokeValidationError(CONFIG_ERROR, "Smoke config schema version is unsupported")
    smoke = config.get("smoke_validation")
    raw_pairs = smoke.get("pairs") if isinstance(smoke, Mapping) else None
    pairs: dict[str, str] = {}
    if isinstance(raw_pairs, Mapping):
        pair_rows = list(raw_pairs.items())
    elif isinstance(raw_pairs, list):
        pair_rows = []
        for row in raw_pairs:
            if not isinstance(row, Mapping) or set(row) != {
                "feature_variant",
                "raw_variant",
            }:
                raise SmokeValidationError(
                    CONFIG_ERROR,
                    "Each smoke pair must contain only feature_variant and raw_variant",
                )
            pair_rows.append((row["feature_variant"], row["raw_variant"]))
    else:
        pair_rows = []
    if len(pair_rows) != 4:
        raise SmokeValidationError(CONFIG_ERROR, "smoke_validation.pairs must define four pairs")
    for feature_variant, raw_variant in pair_rows:
        if not isinstance(feature_variant, str) or not feature_variant:
            raise SmokeValidationError(CONFIG_ERROR, "Feature variant names must be nonempty strings")
        if not isinstance(raw_variant, str) or not raw_variant:
            raise SmokeValidationError(CONFIG_ERROR, "Raw variant names must be nonempty strings")
        if feature_variant in pairs:
            raise SmokeValidationError(CONFIG_ERROR, "Feature variants must be unique")
        pairs[feature_variant] = raw_variant
    if pairs != EXPECTED_PAIRS:
        raise SmokeValidationError(
            CONFIG_ERROR, "Smoke pairs do not match the frozen Iteration 4 mapping"
        )
    if set(pairs) & set(pairs.values()):
        raise SmokeValidationError(CONFIG_ERROR, "Feature and raw variant names must be disjoint")
    if len(set(pairs.values())) != 2 or set(Counter(pairs.values()).values()) != {2}:
        raise SmokeValidationError(CONFIG_ERROR, "Each of two raw controls must have two feature pairs")

    stages = manifest.get("stages")
    screen = stages.get("screen") if isinstance(stages, Mapping) else None
    commands = screen.get("commands") if isinstance(screen, Mapping) else None
    if not isinstance(commands, Mapping) or len(commands) != 6:
        raise SmokeValidationError(PLAN_ERROR, "Screen stage must contain exactly six commands")
    command_records: dict[str, Mapping[str, Any]] = {}
    plans: dict[str, dict[str, Any]] = {}
    by_variant: dict[str, str] = {}
    for raw_id, raw_record in commands.items():
        if not isinstance(raw_id, str) or not isinstance(raw_record, Mapping):
            raise SmokeValidationError(PLAN_ERROR, "Screen command records are malformed")
        plan = _validate_command_plan(raw_id, raw_record)
        variant = plan["variant"]
        if variant in by_variant:
            raise SmokeValidationError(PLAN_ERROR, f"Variant {variant!r} is not unique")
        by_variant[variant] = raw_id
        expected_command_id = (
            f"development-window-end-h1-h12-return-lb720-{variant}"
        )
        if raw_id != expected_command_id:
            raise SmokeValidationError(
                PLAN_ERROR, f"Variant {variant!r} has a noncanonical command id"
            )
        if plan["method"] != EXPECTED_VARIANT_METHOD.get(variant):
            raise SmokeValidationError(
                PLAN_ERROR, f"Variant {variant!r} uses the wrong adapter"
            )
        command_records[raw_id] = raw_record
        plans[raw_id] = plan
    if set(by_variant) != set(pairs) | set(pairs.values()):
        raise SmokeValidationError(PLAN_ERROR, "Pair mapping does not cover the six screen variants")
    ordered_plan = [
        {
            "command_id": command_id,
            "argv": list(record["argv"]),
            "metadata": record["metadata"],
        }
        for variant in EXPECTED_VARIANT_ORDER
        for command_id, record in [
            (by_variant[variant], command_records[by_variant[variant]])
        ]
    ]
    if not isinstance(screen, Mapping) or screen.get("plan_digest") != _sha256_json(
        ordered_plan
    ):
        raise SmokeValidationError(PLAN_ERROR, "Screen plan digest does not match its commands")
    features_by_method: dict[str, list[Any]] = {method: [] for method in EXPECTED_PARAMS}
    raw_methods: dict[str, str] = {}
    for variant, command_id in by_variant.items():
        plan = plans[command_id]
        if variant in pairs:
            if _canonical_json(plan["features"]) != _canonical_json(
                EXPECTED_VARIANT_FEATURE.get(variant)
            ):
                raise SmokeValidationError(PLAN_ERROR, f"{variant} has an unexpected feature family")
            if plan["metadata"].get("feature_contract_required") is not True:
                raise SmokeValidationError(PLAN_ERROR, f"{variant} does not require feature evidence")
            features_by_method[plan["method"]].append(plan["features"])
        else:
            if plan["features"] is not None or "--features" in plan["argv"]:
                raise SmokeValidationError(PLAN_ERROR, f"Raw control {variant} requests features")
            if plan["metadata"].get("feature_contract_required") is not False:
                raise SmokeValidationError(PLAN_ERROR, f"Raw control {variant} has invalid feature metadata")
            raw_methods[variant] = plan["method"]
    expected_feature_hashes = {_sha256_json(value) for value in EXPECTED_FEATURES}
    for method, features in features_by_method.items():
        if {_sha256_json(value) for value in features} != expected_feature_hashes:
            raise SmokeValidationError(PLAN_ERROR, f"{method} does not have both feature families")
    if set(raw_methods.values()) != set(EXPECTED_PARAMS):
        raise SmokeValidationError(PLAN_ERROR, "Raw controls are not one per adapter")
    for feature_variant, raw_variant in pairs.items():
        feature = plans[by_variant[feature_variant]]
        raw = plans[by_variant[raw_variant]]
        if feature["method"] != raw["method"]:
            raise SmokeValidationError(PAIR_ERROR, f"{feature_variant} is paired across adapters")
        if _without_features(feature["argv"]) != raw["argv"]:
            raise SmokeValidationError(
                PAIR_ERROR,
                f"{feature_variant} and {raw_variant} differ by more than --features",
            )
        comparable_metadata = (
            "window",
            "shard",
            "timeframe",
            "horizon",
            "quantity",
            "lookback",
            "steps",
            "spacing",
            "detail",
            "training_floor",
            "methods",
            "params",
            "params_per_method",
            "denoise",
            "dimred",
        )
        for key in comparable_metadata:
            if _canonical_json(feature["metadata"].get(key)) != _canonical_json(
                raw["metadata"].get(key)
            ):
                raise SmokeValidationError(
                    PAIR_ERROR, f"{feature_variant}/{raw_variant} metadata differs at {key}"
                )
    return command_records, pairs, {variant: plans[command_id] for variant, command_id in by_variant.items()}


def _inventory_files(run_dir: Path, model_store: Path, jobs_db: Path) -> list[Path]:
    model_files = [path for path in model_store.rglob("*") if path.is_file()]
    job_files = [
        path
        for path in jobs_db.parent.iterdir()
        if path.name.startswith(jobs_db.name) and path.is_file()
    ]
    return sorted(model_files + job_files, key=lambda path: path.relative_to(run_dir).as_posix())


def _validate_inventory(
    run_dir: Path,
    command_id: str,
    attempt_number: int,
    attempt: Mapping[str, Any],
    hashes: dict[str, str],
    stores_seen: set[str],
) -> None:
    expected_model = f"model_store/screen/{_slug(command_id)}/attempt-{attempt_number}"
    expected_jobs = f"jobs/screen/{_slug(command_id)}/attempt-{attempt_number}.sqlite"
    if not _exact_scalar(attempt.get("attempt"), attempt_number):
        raise SmokeValidationError(ATTEMPT_ERROR, f"{command_id} attempt sequence is invalid")
    if attempt.get("policy") != "fresh_command_attempt":
        raise SmokeValidationError(ATTEMPT_ERROR, f"{command_id} did not use a fresh attempt")
    if attempt.get("attempt_isolation_exempt") is not False:
        raise SmokeValidationError(ATTEMPT_ERROR, f"{command_id} was improperly store-exempt")
    if not _exact_scalar(attempt.get("preexisting_files"), 0):
        raise SmokeValidationError(ATTEMPT_ERROR, f"{command_id} attempt store was not empty")
    if attempt.get("model_store_path") != expected_model or attempt.get(
        "jobs_db_path"
    ) != expected_jobs:
        raise SmokeValidationError(ATTEMPT_ERROR, f"{command_id} attempt paths are not canonical")
    for value in (expected_model, expected_jobs):
        if value in stores_seen:
            raise SmokeValidationError(ATTEMPT_ERROR, f"Attempt path {value} is reused")
        stores_seen.add(value)
    model_store = _inside_path(run_dir, expected_model, "model store")
    jobs_db = _inside_path(run_dir, expected_jobs, "jobs database")
    if not model_store.is_dir() or model_store.is_symlink() or not jobs_db.parent.is_dir():
        raise SmokeValidationError(ATTEMPT_ERROR, f"{command_id} attempt store is missing")
    inventory = attempt.get("post_call_inventory")
    if not isinstance(inventory, Mapping) or inventory.get("complete") is not True:
        raise SmokeValidationError(ATTEMPT_ERROR, f"{command_id} inventory is incomplete")
    rows = inventory.get("files")
    if not isinstance(rows, list) or not _exact_scalar(
        inventory.get("files_observed"), len(rows)
    ):
        raise SmokeValidationError(ATTEMPT_ERROR, f"{command_id} inventory count is invalid")
    actual_files = _inventory_files(run_dir, model_store, jobs_db)
    actual_paths = [path.relative_to(run_dir).as_posix() for path in actual_files]
    recorded_paths: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise SmokeValidationError(ATTEMPT_ERROR, f"{command_id} inventory row is malformed")
        path = _file_and_hash(
            run_dir,
            row.get("path"),
            row.get("sha256"),
            f"{command_id} inventory file",
            hashes,
        )
        if not _exact_scalar(row.get("size"), path.stat().st_size):
            raise SmokeValidationError(ATTEMPT_ERROR, f"{command_id} inventory size differs")
        relative = path.relative_to(run_dir).as_posix()
        if not (
            relative.startswith(f"{expected_model}/")
            or (
                path.parent == jobs_db.parent
                and path.name.startswith(jobs_db.name)
            )
        ):
            raise SmokeValidationError(ATTEMPT_ERROR, f"{command_id} inventories another store")
        recorded_paths.append(relative)
    if set(recorded_paths) != set(actual_paths) or len(set(recorded_paths)) != len(
        recorded_paths
    ):
        raise SmokeValidationError(ATTEMPT_ERROR, f"{command_id} inventory is not exhaustive")


def _validate_attempts(
    run_dir: Path,
    command_id: str,
    record: Mapping[str, Any],
    hashes: dict[str, str],
    stores_seen: set[str],
) -> Mapping[str, Any]:
    attempts = record.get("attempt_artifacts")
    count = record.get("attempts")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise SmokeValidationError(ATTEMPT_ERROR, f"{command_id} has no attempts")
    if not isinstance(attempts, list) or len(attempts) != count:
        raise SmokeValidationError(ATTEMPT_ERROR, f"{command_id} attempt history is incomplete")
    latest_envelope: Mapping[str, Any] | None = None
    for attempt_number, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt, Mapping):
            raise SmokeValidationError(ATTEMPT_ERROR, f"{command_id} attempt is malformed")
        expected_raw = f"raw/screen/{_slug(command_id)}/attempt-{attempt_number}.json"
        if attempt.get("raw_path") != expected_raw:
            raise SmokeValidationError(RAW_ERROR, f"{command_id} raw attempt path is not canonical")
        path = _file_and_hash(
            run_dir,
            attempt.get("raw_path"),
            attempt.get("raw_sha256"),
            f"{command_id} attempt {attempt_number}",
            hashes,
        )
        envelope = _load_json(path, RAW_ERROR)
        if not isinstance(envelope, Mapping):
            raise SmokeValidationError(RAW_ERROR, f"{command_id} envelope is not an object")
        if not _exact_scalar(envelope.get("schema_version"), 1):
            raise SmokeValidationError(RAW_ERROR, f"{command_id} envelope schema is unsupported")
        if envelope.get("stage") != "screen" or envelope.get("command_id") != command_id:
            raise SmokeValidationError(RAW_ERROR, f"{command_id} envelope identity differs")
        if attempt_number < count and envelope.get("status") != "failed":
            raise SmokeValidationError(RAW_ERROR, f"{command_id} prior attempt was not failed")
        _validate_inventory(run_dir, command_id, attempt_number, attempt, hashes, stores_seen)
        envelope_isolation = envelope.get("attempt_isolation")
        if not isinstance(envelope_isolation, Mapping):
            raise SmokeValidationError(ATTEMPT_ERROR, f"{command_id} envelope lacks isolation")
        for key in (
            "attempt",
            "policy",
            "model_store_path",
            "jobs_db_path",
            "preexisting_files",
            "attempt_isolation_exempt",
            "post_call_inventory",
        ):
            if _canonical_json(envelope_isolation.get(key)) != _canonical_json(
                attempt.get(key)
            ):
                raise SmokeValidationError(
                    ATTEMPT_ERROR, f"{command_id} envelope isolation differs at {key}"
                )
        latest_envelope = envelope
    latest = attempts[-1]
    if record.get("raw_path") != latest.get("raw_path") or record.get(
        "raw_sha256"
    ) != latest.get("raw_sha256"):
        raise SmokeValidationError(RAW_ERROR, f"{command_id} latest attempt is not current")
    if _canonical_json(record.get("post_call_inventory")) != _canonical_json(
        latest.get("post_call_inventory")
    ):
        raise SmokeValidationError(ATTEMPT_ERROR, f"{command_id} current inventory differs")
    assert latest_envelope is not None
    return latest_envelope


def _finite_path(value: Any, expected: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != expected:
        raise SmokeValidationError(RESULT_ERROR, f"{label} must contain {expected} values")
    numbers: list[float] = []
    for item in value:
        if isinstance(item, bool):
            raise SmokeValidationError(RESULT_ERROR, f"{label} contains a non-number")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise SmokeValidationError(RESULT_ERROR, f"{label} contains a non-number") from exc
        if not math.isfinite(number):
            raise SmokeValidationError(RESULT_ERROR, f"{label} contains a non-finite value")
        numbers.append(number)
    return numbers


def _valid_utc(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SmokeValidationError(RESULT_ERROR, f"{label} is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SmokeValidationError(RESULT_ERROR, f"{label} is not ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise SmokeValidationError(RESULT_ERROR, f"{label} is not UTC")
    return value


def _utc_datetime(value: Any, label: str) -> datetime:
    text_value = _valid_utc(value, label)
    return datetime.fromisoformat(text_value.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, Mapping):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def _validate_feature_usage(
    method_usage: Any,
    detail_usage: Any,
    command_id: str,
    expected_features: Mapping[str, Any],
) -> None:
    if not isinstance(method_usage, Mapping) or not isinstance(detail_usage, Mapping):
        raise SmokeValidationError(RESULT_ERROR, f"{command_id} omitted feature usage")
    expected = {
        "status": "consumed",
        "historical_consumed": True,
        "future_consumed": True,
        "anchors_verified": 1,
        "future_rows": 12,
        "observed_feature_lag_bars": 1,
        "observed_future_policy": "carry_forward",
    }
    for key, value in expected.items():
        if not _exact_scalar(method_usage.get(key), value):
            raise SmokeValidationError(
                RESULT_ERROR, f"{command_id} feature_usage.{key} is not {value!r}"
            )
    n_features = method_usage.get("n_features")
    selected = method_usage.get("selected_columns")
    expected_selected = next(
        (
            columns
            for feature_spec, columns in EXPECTED_FEATURE_COLUMNS
            if _canonical_json(feature_spec) == _canonical_json(expected_features)
        ),
        None,
    )
    if expected_selected is None:
        raise SmokeValidationError(RESULT_ERROR, f"{command_id} feature family is unknown")
    if _canonical_json(expected_features) == _canonical_json(EXPECTED_FEATURES[0]):
        expected_indicators = ["RSI_14", "ROC_12"]
        expected_calendar: list[str] = []
    else:
        expected_indicators = ["NATR_14"]
        expected_calendar = ["hr_sin", "hr_cos", "dow_sin", "dow_cos"]
    if (
        not _exact_scalar(n_features, len(expected_selected))
        or not _semantic_columns_match(selected, expected_selected)
        or method_usage.get("adapter_columns")
        != [f"x{index}" for index in range(len(expected_selected))]
        or not _semantic_columns_match(method_usage.get("include_columns"), [])
        or not _semantic_columns_match(
            method_usage.get("indicator_columns"), expected_indicators
        )
        or not _semantic_columns_match(
            method_usage.get("calendar_columns"), expected_calendar
        )
        or method_usage.get("dimred_method") is not None
        or method_usage.get("dimred_n_features") is not None
    ):
        raise SmokeValidationError(RESULT_ERROR, f"{command_id} feature columns are invalid")
    for key in (
        "status",
        "historical_consumed",
        "future_consumed",
        "future_rows",
        "n_features",
        "adapter_columns",
        "selected_columns",
        "include_columns",
        "indicator_columns",
        "calendar_columns",
        "observed_feature_lag_bars",
        "observed_future_policy",
        "dimred_method",
        "dimred_n_features",
    ):
        if _canonical_json(detail_usage.get(key)) != _canonical_json(method_usage.get(key)):
            raise SmokeValidationError(RESULT_ERROR, f"{command_id} per-anchor usage differs")
    historical_rows = detail_usage.get("historical_rows")
    historical_min = method_usage.get("historical_rows_min")
    historical_max = method_usage.get("historical_rows_max")
    if (
        _strict_int(historical_rows, minimum=1) is None
        or historical_rows > 720
        or not _exact_scalar(historical_min, historical_rows)
        or not _exact_scalar(historical_max, historical_rows)
    ):
        raise SmokeValidationError(RESULT_ERROR, f"{command_id} historical feature rows differ")


def _validate_backtest_context(
    payload: Mapping[str, Any],
    detail: Mapping[str, Any],
    method: str,
    command_id: str,
) -> tuple[str, str]:
    plan = payload.get("backtest_plan")
    analysis = payload.get("analysis_time_window")
    if not isinstance(plan, Mapping) or not isinstance(analysis, Mapping):
        raise SmokeValidationError(RESULT_ERROR, f"{command_id} omitted evaluation context")
    expected_plan = {
        "model": "rolling_origin_fixed_window",
        "anchor_mode": "rolling",
        "runs_requested": 1,
        "runs_used": 1,
        "horizon_bars": 12,
        "method_selection": "explicit",
        "methods_planned": [method],
        "method_count": 1,
        "fits_planned": 1,
        "model_lookback_bars": 720,
        "anchor_spacing_bars": 12,
        "validation_span_bars": 12,
    }
    for key, expected in expected_plan.items():
        if not _exact_scalar(plan.get(key), expected):
            raise SmokeValidationError(
                RESULT_ERROR, f"{command_id} backtest_plan.{key} differs"
            )
    history_bars = _strict_int(plan.get("history_bars_used"), minimum=733)
    if history_bars is None:
        raise SmokeValidationError(
            RESULT_ERROR, f"{command_id} history coverage is too short"
        )
    expected_analysis = {
        "timezone": "UTC",
        "timestamp_basis": "bar_open_time",
        "input_bar_policy": "closed_bars_only",
        "evaluation_target_policy": "next_bar_through_horizon_bar",
    }
    for key, expected in expected_analysis.items():
        if analysis.get(key) != expected:
            raise SmokeValidationError(
                RESULT_ERROR, f"{command_id} analysis_time_window.{key} differs"
            )
    anchor = _valid_utc(detail.get("anchor"), f"{command_id} anchor")
    entry_time = _valid_utc(detail.get("entry_time"), f"{command_id} entry_time")
    anchor_dt = _utc_datetime(anchor, f"{command_id} anchor")
    entry_dt = _utc_datetime(entry_time, f"{command_id} entry_time")
    first_anchor = _utc_datetime(
        analysis.get("first_anchor"), f"{command_id} first_anchor"
    )
    last_anchor = _utc_datetime(
        analysis.get("last_anchor"), f"{command_id} last_anchor"
    )
    evaluation_start = _utc_datetime(
        analysis.get("evaluation_start"), f"{command_id} evaluation_start"
    )
    evaluation_end = _utc_datetime(
        analysis.get("evaluation_end"), f"{command_id} evaluation_end"
    )
    history_start = _utc_datetime(
        analysis.get("history_start"), f"{command_id} history_start"
    )
    history_end = _utc_datetime(
        analysis.get("history_end"), f"{command_id} history_end"
    )
    window_start = datetime(2024, 5, 1, tzinfo=timezone.utc)
    window_end = datetime(2024, 5, 31, 23, 59, 59, tzinfo=timezone.utc)
    if not (
        first_anchor == last_anchor == anchor_dt
        and evaluation_start == entry_dt
        and entry_dt - anchor_dt == timedelta(hours=1)
        and evaluation_end - anchor_dt == timedelta(hours=12)
        and window_start <= history_start <= anchor_dt
        and evaluation_end <= history_end <= window_end
    ):
        raise SmokeValidationError(
            RESULT_ERROR, f"{command_id} evaluation timestamps are inconsistent"
        )
    return anchor, entry_time


def _validate_payload(
    command_id: str,
    record: Mapping[str, Any],
    envelope: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    feature: bool,
    expected_python: str,
) -> dict[str, Any]:
    if (
        record.get("status") != "completed"
        or envelope.get("status") != "completed"
        or not _exact_scalar(envelope.get("returncode"), 0)
        or "contract_error" in envelope
        or "parse_error" in envelope
    ):
        raise SmokeValidationError(RESULT_ERROR, f"{command_id} did not complete cleanly")
    invocation = envelope.get("invocation")
    argv = record.get("argv")
    expected_invocation = [expected_python, "-m", "mtdata", "--json", *argv]
    if invocation != expected_invocation:
        raise SmokeValidationError(RAW_ERROR, f"{command_id} invocation differs from manifest")
    if _canonical_json(envelope.get("metadata")) != _canonical_json(record.get("metadata")):
        raise SmokeValidationError(RAW_ERROR, f"{command_id} envelope metadata differs")
    payload = envelope.get("payload")
    if not isinstance(payload, Mapping):
        raise SmokeValidationError(RESULT_ERROR, f"{command_id} payload is not an object")
    expected_top = {
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
    }
    for key, value in expected_top.items():
        if not _exact_scalar(payload.get(key), value):
            raise SmokeValidationError(RESULT_ERROR, f"{command_id} payload {key} differs")
    results = payload.get("results")
    method = plan["method"]
    if not isinstance(results, Mapping) or set(results) != {method}:
        raise SmokeValidationError(RESULT_ERROR, f"{command_id} result method differs")
    result = results[method]
    if not isinstance(result, Mapping):
        raise SmokeValidationError(RESULT_ERROR, f"{command_id} method result is invalid")
    expected_method = {
        "success": True,
        "complete_success": True,
        "status": "complete",
        "num_tests": 1,
        "successful_tests": 1,
        "failed_tests": 0,
    }
    for key, value in expected_method.items():
        if not _exact_scalar(result.get(key), value):
            raise SmokeValidationError(RESULT_ERROR, f"{command_id} result {key} differs")
    details = result.get("details")
    if not isinstance(details, list) or len(details) != 1 or not isinstance(details[0], Mapping):
        raise SmokeValidationError(RESULT_ERROR, f"{command_id} omitted its full detail")
    detail = details[0]
    if detail.get("success") is not True or not _exact_scalar(
        detail.get("training_bars_used"), 720
    ):
        raise SmokeValidationError(RESULT_ERROR, f"{command_id} detail is incomplete")
    anchor, entry_time = _validate_backtest_context(payload, detail, method, command_id)
    forecast = _finite_path(detail.get("forecast"), 12, f"{command_id} forecast")
    _finite_path(detail.get("actual"), 12, f"{command_id} actual")
    for key in ("entry_price", "signal_reference_price"):
        value = detail.get(key)
        if isinstance(value, bool):
            raise SmokeValidationError(RESULT_ERROR, f"{command_id} {key} is invalid")
        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            raise SmokeValidationError(RESULT_ERROR, f"{command_id} {key} is invalid")
    if detail.get("entry_price_source") != "next_bar_open":
        raise SmokeValidationError(RESULT_ERROR, f"{command_id} entry source differs")
    if feature:
        _validate_feature_usage(
            result.get("feature_usage"),
            detail.get("feature_usage"),
            command_id,
            plan["features"],
        )
        params_used = detail.get("params_used")
        if not isinstance(params_used, Mapping):
            raise SmokeValidationError(RESULT_ERROR, f"{command_id} omitted params_used")
        for key, expected in plan["params"].items():
            if _canonical_json(params_used.get(key)) != _canonical_json(expected):
                raise SmokeValidationError(
                    RESULT_ERROR,
                    f"{command_id} did not use preregistered parameter {key!r}",
                )
    elif _contains_key(payload, "feature_usage"):
        raise SmokeValidationError(RESULT_ERROR, f"Raw control {command_id} contains feature usage")
    return {
        "anchor": anchor,
        "forecast": forecast,
        "actual_json": _canonical_json(detail["actual"]),
        "entry_json": _canonical_json({key: detail.get(key) for key in ENTRY_FIELDS}),
    }


def _validate_completed(
    run_dir: Path,
    command_id: str,
    record: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    feature: bool,
    hashes: dict[str, str],
    stores_seen: set[str],
    expected_python: str,
) -> dict[str, Any]:
    envelope = _validate_attempts(
        run_dir, command_id, record, hashes, stores_seen
    )
    return _validate_payload(
        command_id,
        record,
        envelope,
        plan,
        feature=feature,
        expected_python=expected_python,
    )


def _validate_pair(
    feature_variant: str,
    raw_variant: str,
    *,
    by_variant: Mapping[str, tuple[str, Mapping[str, Any]]],
    plans: Mapping[str, Mapping[str, Any]],
    run_dir: Path,
    hashes: dict[str, str],
    stores_seen: set[str],
    evidence_cache: dict[str, dict[str, Any]],
    expected_python: str,
) -> dict[str, Any]:
    feature_id, feature_record = by_variant[feature_variant]
    raw_id, raw_record = by_variant[raw_variant]
    if feature_id not in evidence_cache:
        evidence_cache[feature_id] = _validate_completed(
            run_dir,
            feature_id,
            feature_record,
            plans[feature_variant],
            feature=True,
            hashes=hashes,
            stores_seen=stores_seen,
            expected_python=expected_python,
        )
    if raw_id not in evidence_cache:
        evidence_cache[raw_id] = _validate_completed(
            run_dir,
            raw_id,
            raw_record,
            plans[raw_variant],
            feature=False,
            hashes=hashes,
            stores_seen=stores_seen,
            expected_python=expected_python,
        )
    feature_evidence = evidence_cache[feature_id]
    raw_evidence = evidence_cache[raw_id]
    if feature_evidence["anchor"] != raw_evidence["anchor"]:
        raise SmokeValidationError(PAIR_ERROR, f"{feature_variant} anchor differs from raw")
    if feature_evidence["actual_json"] != raw_evidence["actual_json"]:
        raise SmokeValidationError(PAIR_ERROR, f"{feature_variant} actual path differs from raw")
    if feature_evidence["entry_json"] != raw_evidence["entry_json"]:
        raise SmokeValidationError(PAIR_ERROR, f"{feature_variant} entry evidence differs from raw")
    deltas = [
        abs(feature_value - raw_value)
        for feature_value, raw_value in zip(
            feature_evidence["forecast"], raw_evidence["forecast"]
        )
    ]
    max_delta = max(deltas)
    if all(delta <= 1e-12 for delta in deltas):
        raise SmokeValidationError(
            NOOP_ERROR,
            f"{feature_variant} forecast is numerically identical to {raw_variant}",
            {"absolute_tolerance": 1e-12, "max_absolute_delta": max_delta},
        )
    return {
        "feature_variant": feature_variant,
        "feature_command_id": feature_id,
        "raw_variant": raw_variant,
        "raw_command_id": raw_id,
        "method": plans[feature_variant]["method"],
        "anchor": feature_evidence["anchor"],
        "max_absolute_forecast_delta": max_delta,
        "no_op_tolerance": 1e-12,
    }


def _verify_config_revision(
    run_dir: Path,
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    hashes: dict[str, str],
) -> None:
    digest = _sha256_json(config)
    if manifest.get("active_config_hash") != digest:
        raise SmokeValidationError(CONFIG_ERROR, "Active config hash differs from resolved config")
    revisions = manifest.get("config_revisions")
    matching = [
        row
        for row in revisions
        if isinstance(row, Mapping) and row.get("config_hash") == digest
    ] if isinstance(revisions, list) else []
    if len(matching) != 1:
        raise SmokeValidationError(CONFIG_ERROR, "Active config revision is missing or ambiguous")
    revision_path = _inside_path(run_dir, matching[0].get("path"), "config revision")
    if not revision_path.is_file() or revision_path.is_symlink():
        raise SmokeValidationError(CONFIG_ERROR, "Active config revision file is missing")
    hashes[revision_path.relative_to(run_dir).as_posix()] = _sha256_file(revision_path)
    revision = _load_json(revision_path, CONFIG_ERROR)
    if (
        not isinstance(revision, Mapping)
        or not _exact_scalar(revision.get("schema_version"), 1)
        or revision.get("config_hash") != digest
        or _canonical_json(revision.get("config")) != _canonical_json(config)
    ):
        raise SmokeValidationError(CONFIG_ERROR, "Active config revision content differs")


def validate_smoke_run(
    run_dir: Path,
    *,
    pair: str | None = None,
    current_source: Mapping[str, Any] | None = None,
    current_runtime: Mapping[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Return a pass/fail artifact; never mutate the run directory."""

    resolved_run_dir = run_dir.resolve()
    hashes: dict[str, str] = {}
    pair_diagnostics: list[dict[str, Any]] = []
    run_id = resolved_run_dir.name
    error: SmokeValidationError | None = None
    mode = "pair" if pair is not None else "aggregate"
    try:
        if not resolved_run_dir.is_dir():
            raise SmokeValidationError(RAW_ERROR, "Run directory does not exist")
        manifest_path = _inside_path(
            resolved_run_dir, "manifest.json", "manifest"
        )
        config_path = _inside_path(
            resolved_run_dir, "resolved_config.json", "resolved config"
        )
        if not manifest_path.is_file() or not config_path.is_file():
            raise SmokeValidationError(
                RAW_ERROR, "Canonical manifest or resolved config is missing"
            )
        if manifest_path.is_file():
            hashes["manifest.json"] = _sha256_file(manifest_path)
        if config_path.is_file():
            hashes["resolved_config.json"] = _sha256_file(config_path)
        validator_path = Path(__file__).resolve()
        hashes["validator_source"] = _sha256_file(validator_path)
        manifest = _load_json(manifest_path, RAW_ERROR)
        config = _load_json(config_path, CONFIG_ERROR)
        if not isinstance(manifest, Mapping) or not isinstance(config, Mapping):
            raise SmokeValidationError(CONFIG_ERROR, "Manifest and config must be objects")
        if not _exact_scalar(manifest.get("schema_version"), 1) or manifest.get(
            "harness_version"
        ) not in SUPPORTED_HARNESS_VERSIONS:
            raise SmokeValidationError(CONFIG_ERROR, "Manifest schema or harness is unsupported")
        run_id = str(manifest.get("run_id") or run_id)
        _verify_source_runtime(
            manifest,
            current_source=current_source,
            current_runtime=current_runtime,
        )
        expected_python = manifest["runtime"].get("python_executable")
        if not isinstance(expected_python, str) or not expected_python:
            raise SmokeValidationError(SOURCE_ERROR, "Run Python executable is missing")
        _verify_config_revision(resolved_run_dir, manifest, config, hashes)
        records, pairs, plans = _validate_plan(manifest, config)
        by_variant = {
            str(record["metadata"]["variant"]): (command_id, record)
            for command_id, record in records.items()
        }
        screen = manifest["stages"]["screen"]
        statuses = {command_id: record.get("status") for command_id, record in records.items()}
        if any(status in {"failed", "running"} for status in statuses.values()):
            raise SmokeValidationError(RESULT_ERROR, "Screen stage contains a failed/running command")
        selected_pairs: list[tuple[str, str]]
        if pair is not None:
            if pair not in pairs:
                raise SmokeValidationError(CONFIG_ERROR, f"Unknown feature variant {pair!r}")
            selected_pairs = [(pair, pairs[pair])]
        else:
            if screen.get("status") != "completed" or any(
                status != "completed" for status in statuses.values()
            ):
                raise SmokeValidationError(RESULT_ERROR, "Aggregate smoke stage is not complete")
            selected_pairs = list(pairs.items())
        stores_seen: set[str] = set()
        evidence_cache: dict[str, dict[str, Any]] = {}
        for feature_variant, raw_variant in selected_pairs:
            pair_diagnostics.append(
                _validate_pair(
                    feature_variant,
                    raw_variant,
                    by_variant=by_variant,
                    plans=plans,
                    run_dir=resolved_run_dir,
                    hashes=hashes,
                    stores_seen=stores_seen,
                    evidence_cache=evidence_cache,
                    expected_python=expected_python,
                )
            )
        if pair is None:
            contexts = {
                (
                    evidence["anchor"],
                    evidence["actual_json"],
                    evidence["entry_json"],
                )
                for evidence in evidence_cache.values()
            }
            if len(contexts) != 1:
                raise SmokeValidationError(
                    PAIR_ERROR,
                    "The six smoke commands do not share identical evaluation evidence",
                )
    except SmokeValidationError as exc:
        error = exc
    return {
        "schema_version": SCHEMA_VERSION,
        "validator": VALIDATOR_NAME,
        "validator_version": VALIDATOR_VERSION,
        "generated_at": generated_at or _utc_now(),
        "run_id": run_id,
        "mode": mode,
        "requested_pair": pair,
        "status": "failed" if error else "passed",
        "outcome_metrics_computed": False,
        "input_hashes": dict(sorted(hashes.items())),
        "diagnostics": {
            "pairs_validated": pair_diagnostics,
            "errors": []
            if error is None
            else [
                {
                    "code": error.code,
                    "message": str(error),
                    "context": error.context,
                }
            ],
        },
    }


def _artifact_path(run_dir: Path, pair: str | None) -> Path:
    name = (
        f"smoke_validation_{_slug(pair)}.json"
        if pair is not None
        else "smoke_validation.json"
    )
    resolved_run_dir = run_dir.resolve()
    return _inside_path(resolved_run_dir, f"validation/{name}", "validation artifact")


def validate_and_write(run_dir: Path, *, pair: str | None = None) -> tuple[Path, dict[str, Any]]:
    """Validate and exclusively create the immutable pass/fail artifact."""

    output_path = _artifact_path(run_dir, pair)
    if output_path.parent.exists() and not output_path.parent.is_dir():
        raise SmokeValidationError(RAW_ERROR, "Validation output path is not a directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.parent.is_symlink():
        raise SmokeValidationError(RAW_ERROR, "Validation output directory is a symbolic link")
    if output_path.exists():
        raise SmokeValidationError(
            RAW_ERROR, f"Refusing to overwrite immutable artifact: {output_path}"
        )
    artifact = validate_smoke_run(run_dir, pair=pair)
    try:
        with output_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(artifact, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise SmokeValidationError(
            RAW_ERROR, f"Refusing to overwrite immutable artifact: {output_path}"
        ) from exc
    return output_path, artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate saved Iteration 4 BTCUSD feature-smoke evidence."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--pair",
        help="Validate one completed feature variant/raw pair before later MT5 calls.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        path, artifact = validate_and_write(args.run_dir, pair=args.pair)
    except SmokeValidationError as exc:
        print(f"{exc.code}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"artifact": str(path), "status": artifact["status"]}))
    return 0 if artifact["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
