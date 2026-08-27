"""Resumable, research-only BTCUSD forecasting experiments through mtdata CLI.

The harness never imports forecast internals and never invokes trading commands.  Every
mtdata call is ``sys.executable -m mtdata --json ...`` in a run-local model/job store.
Raw command envelopes, normalized stage summaries, protocol locks, and a friction ledger
are persisted below the study directory so long experiments can be audited and resumed.
"""

from __future__ import annotations

import argparse
import calendar
import copy
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
HARNESS_VERSION = "0.1.0"
RESEARCH_STAGES = (
    "audit",
    "screen",
    "tune",
    "validate",
    "freeze",
    "holdout",
    "materialize",
    "shadow",
    "report",
)
FORBIDDEN_COMMAND_PREFIXES = ("trade_", "trade-")
FORBIDDEN_COMMANDS = {"shell"}
DEFAULT_OUTPUT_ROOT = Path("backtests") / "btcusd_forecast"

RESEARCH_WINDOWS: dict[str, dict[str, Any]] = {
    "stress": {
        "start": "2017-05-01",
        "end": "2022-05-31",
        "role": "pre-development stress window",
        "locked": False,
    },
    "development": {
        "start": "2022-06-01",
        "end": "2024-06-30",
        "role": "model screening and tuning",
        "locked": False,
    },
    "validation": {
        "start": "2024-07-01",
        "end": "2025-06-30",
        "role": "candidate validation",
        "locked": False,
    },
    "confirmation": {
        "start": "2025-07-01",
        "end": "2026-06-30",
        "role": "pre-freeze confirmation",
        "locked": False,
    },
    "locked_holdout": {
        "start": "2026-07-01",
        "end": "2026-08-26",
        "role": "single-use final holdout",
        "locked": True,
    },
}

BASELINE_METHODS = ["naive", "drift", "seasonal_naive", "theta", "fourier_ols"]
EXTERNALLY_PRETRAINED_METHODS = {"chronos2", "chronos_bolt", "pretrained", "timesfm"}
RELATIVE_WIDTH_DEFINITION = "mean((upper-lower)/abs(reference_price))"
MAX_MEAN_RELATIVE_WIDTH_CAP = 0.10
INTERVAL_APPROVAL_ENABLED = False
INTERVAL_APPROVAL_DISABLED_REASON = (
    "BTC-INTERVAL-VERIFIER-REQUIRED: interval approval is disabled until a first-class "
    "verifier recomputes coverage and width from hashed raw MT5 forecast and actual "
    "envelopes; self-reported interval summaries and raw_sha256 labels are not approval evidence."
)
BASELINE_MATRIX = {
    "timeframes": ["H1"],
    "horizons": [6, 12, 24],
    "quantities": ["price", "return"],
    "lookbacks": [336, 720, 2160, 4320],
    "methods": BASELINE_METHODS,
    "spacing_policy": "equal_to_horizon",
    "sharding": "month_end_plus_exact_window_end",
}

DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "symbol": "BTCUSD",
    "seed": 42,
    "runtime": {"cuda_visible_devices": "0,1"},
    "research_windows": RESEARCH_WINDOWS,
    "audit": {
        "as_of": RESEARCH_WINDOWS["confirmation"]["end"],
        "timeframes": ["H1", "H4", "D1"],
        "lookback": 1500,
        "candle_probe_limit": 500,
    },
    "screen": {
        "window": "development",
        **BASELINE_MATRIX,
        "steps_per_shard": "auto_month",
        "detail": "full",
        "variants": [
            {
                "id": "raw",
                "denoise": None,
                "features": None,
                "params_per_method": {
                    "seasonal_naive": {"seasonality": 24},
                    "fourier_ols": {"seasonality": 24, "terms": 3, "trend": True},
                },
            }
        ],
        "slippage_bps": 0.5,
        "spread_bps": 0.625,
        "commission_bps_per_side": 0.0,
        "trade_threshold": 0.0,
    },
    "tune": {
        "experiments": [],
        "detail": "full",
    },
    "validate": {
        "windows": ["validation", "confirmation"],
        "training_floor": RESEARCH_WINDOWS["development"]["start"],
        "steps_per_shard": "auto_month",
        "detail": "full",
    },
    "holdout": {
        "window": "locked_holdout",
        "steps_per_shard": "auto_month",
        "minimum_opportunities": 50,
        "minimum_signals": 25,
        "detail": "full",
    },
    "shadow": {
        "ci_alpha": 0.10,
        "conformal_steps": 100,
        "model_cache": "require_existing",
        "refit_interval_bars": 24,
        "detail": "full",
    },
}

_SENSITIVE_KEY_RE = re.compile(
    r"(?:^|[_-])(account(?:[_-](?:id|number))?|api[_-]?key|authorization|cookie|credential|login|passwd|password|secret|token)(?:$|[_-])",
    re.IGNORECASE,
)
_URL_USERINFO_RE = re.compile(r"(?P<scheme>[a-z][a-z0-9+.-]*://)(?P<userinfo>[^/@\s:]+:[^/@\s]+)@", re.IGNORECASE)
_INLINE_SECRET_RE = re.compile(
    r"(?P<prefix>\b(?:account(?:[_-](?:id|number))?|api[_-]?key|authorization|cookie|credential|login|passwd|password|secret|token)\s*[:=]\s*)(?P<value>[^\s,;]+)",
    re.IGNORECASE,
)
_SECRET_FLAG_RE = re.compile(
    r"(?P<prefix>--(?:account(?:[_-](?:id|number))?|api[_-]?key|authorization|cookie|credential|login|passwd|password|secret|token)\s+)(?P<value>[^\s]+)",
    re.IGNORECASE,
)


class HarnessError(RuntimeError):
    """Expected protocol/configuration failure suitable for a concise CLI error."""


class CommandFailed(HarnessError):
    """One or more mtdata commands did not produce a successful JSON result."""


class IntervalApprovalDisabledError(HarnessError):
    """Approval cannot proceed until interval metrics are recomputed from raw evidence."""


@dataclass(frozen=True)
class CommandSpec:
    """One deterministic mtdata CLI invocation."""

    command_id: str
    argv: tuple[str, ...]
    metadata: Mapping[str, Any]


@dataclass
class RunContext:
    """Resolved study state used by stage functions."""

    run_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    config: dict[str, Any]
    dry_run: bool
    timeout: float
    max_commands: int | None
    fail_fast: bool


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _slug(value: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value).strip()).strip("-._")
    return text.lower() or "item"


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _secret_values(environment: Mapping[str, str] | None = None) -> list[str]:
    env = environment if environment is not None else os.environ
    values: list[str] = []
    for key, value in env.items():
        if _SENSITIVE_KEY_RE.search(str(key)) and len(str(value)) >= 4:
            values.append(str(value))
    return sorted(set(values), key=len, reverse=True)


def _redact_text(value: str, known_secrets: Sequence[str] = ()) -> str:
    text = _URL_USERINFO_RE.sub(lambda match: f"{match.group('scheme')}***:***@", value)
    text = _INLINE_SECRET_RE.sub(lambda match: f"{match.group('prefix')}[REDACTED]", text)
    text = _SECRET_FLAG_RE.sub(lambda match: f"{match.group('prefix')}[REDACTED]", text)
    for secret in known_secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text


def redact(value: Any, *, known_secrets: Sequence[str] = ()) -> Any:
    """Return a JSON-compatible copy with secret keys, URLs, and values redacted."""
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _SENSITIVE_KEY_RE.search(key_text):
                out[key_text] = "[REDACTED]"
            else:
                out[key_text] = redact(item, known_secrets=known_secrets)
        return out
    if isinstance(value, (list, tuple, set)):
        return [redact(item, known_secrets=known_secrets) for item in value]
    if isinstance(value, str):
        return _redact_text(value, known_secrets)
    return value


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _write_new_json(path: Path, value: Any) -> None:
    """Write an immutable protocol artifact exactly once."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise HarnessError(f"Refusing to overwrite immutable artifact: {path}") from exc


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"Could not read valid JSON from {path}: {exc}") from exc


def _relative(path: Path, run_dir: Path) -> str:
    try:
        return path.relative_to(run_dir).as_posix()
    except ValueError:
        return str(path)


def _validate_research_command(argv: Sequence[str]) -> None:
    if not argv:
        raise HarnessError("An empty mtdata command is not allowed")
    command = str(argv[0]).strip().lower()
    if command in FORBIDDEN_COMMANDS or command.startswith(FORBIDDEN_COMMAND_PREFIXES):
        raise HarnessError(f"Research harness refuses command {argv[0]!r}")


def _reject_external_pretrained_method(method: Any) -> None:
    normalized = str(method or "").strip().lower()
    if normalized in EXTERNALLY_PRETRAINED_METHODS or normalized.startswith(
        ("chronos", "timesfm", "pretrained")
    ):
        raise HarnessError(
            f"Externally pretrained/foundation method {method!r} is forbidden; "
            "this study may learn only from MT5-provided market data"
        )


def _require_interval_approval_enabled() -> None:
    if not INTERVAL_APPROVAL_ENABLED:
        raise IntervalApprovalDisabledError(INTERVAL_APPROVAL_DISABLED_REASON)


def _redact_invocation(argv: Sequence[str], known_secrets: Sequence[str]) -> list[str]:
    result: list[str] = []
    redact_next = False
    for token in argv:
        text = str(token)
        if redact_next:
            result.append("[REDACTED]")
            redact_next = False
            continue
        safe_text = _redact_text(text, known_secrets)
        if text.lstrip().startswith(("{", "[")):
            try:
                structured = json.loads(text)
            except json.JSONDecodeError:
                pass
            else:
                safe_text = _canonical_json(redact(structured, known_secrets=known_secrets))
        result.append(safe_text)
        normalized = text.strip().lower().lstrip("-").replace("-", "_")
        if _SENSITIVE_KEY_RE.search(normalized):
            redact_next = "=" not in text
    return result


def _month_end_shards(window_name: str, window: Mapping[str, Any]) -> list[dict[str, str]]:
    start = date.fromisoformat(str(window["start"]))
    end = date.fromisoformat(str(window["end"]))
    if end < start:
        raise HarnessError(f"Window {window_name!r} ends before it starts")
    cursor = date(start.year, start.month, 1)
    shards: list[dict[str, str]] = []
    while cursor <= end:
        last_day = calendar.monthrange(cursor.year, cursor.month)[1]
        natural_end = date(cursor.year, cursor.month, last_day)
        shard_end = min(natural_end, end)
        if shard_end >= start:
            kind = "month_end" if shard_end == natural_end else "exact_window_end"
            shards.append(
                {
                    "id": f"{cursor.year:04d}-{cursor.month:02d}",
                    "start": max(start, cursor).isoformat(),
                    "end": shard_end.isoformat(),
                    "kind": kind,
                }
            )
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return shards


_TIMEFRAME_MINUTES = {
    "M1": 1,
    "M2": 2,
    "M3": 3,
    "M4": 4,
    "M5": 5,
    "M6": 6,
    "M10": 10,
    "M12": 12,
    "M15": 15,
    "M20": 20,
    "M30": 30,
    "H1": 60,
    "H2": 120,
    "H3": 180,
    "H4": 240,
    "H6": 360,
    "H8": 480,
    "H12": 720,
    "D1": 1440,
    "W1": 10080,
}


def _continuous_bars(start: str, end: str, timeframe: str) -> int:
    minutes = _TIMEFRAME_MINUTES.get(timeframe.upper())
    if minutes is None:
        raise HarnessError(f"Automatic shard sizing does not support timeframe {timeframe!r}")
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    inclusive_minutes = int(((end_date + timedelta(days=1)) - start_date).total_seconds() // 60)
    return inclusive_minutes // minutes


def _resolve_shard_steps(value: Any, shard: Mapping[str, str], timeframe: str, horizon: int) -> int:
    if str(value).lower() != "auto_month":
        steps = int(value)
    else:
        shard_bars = _continuous_bars(shard["start"], shard["end"], timeframe)
        # Leave one complete horizon as a feed-gap margin.  BTC brokers can omit
        # maintenance bars; consuming the theoretical 24/7 maximum could make the
        # rolling-origin selector reach back into the preceding monthly shard.
        steps = max(1, (shard_bars // horizon) - 1)
    if steps < 1:
        raise HarnessError(f"Shard {shard['id']} is shorter than horizon={horizon}")
    if steps > 200:
        raise HarnessError(
            f"Full shard {shard['id']} needs {steps} anchors, above mtdata's 200-step limit; use a coarser timeframe or split the shard"
        )
    return steps


def _json_arg(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _append_optional_pipeline(argv: list[str], config: Mapping[str, Any]) -> None:
    for key, flag in (("params", "--params"), ("denoise", "--denoise"), ("features", "--features"), ("dimred", "--dimred")):
        value = config.get(key)
        if value is not None:
            argv.extend([flag, _json_arg(value) if isinstance(value, (dict, list)) else str(value)])


def _append_costs(argv: list[str], config: Mapping[str, Any]) -> None:
    for key, flag in (
        ("slippage_bps", "--slippage-bps"),
        ("spread_bps", "--spread-bps"),
        ("commission_bps_per_side", "--commission-bps-per-side"),
        ("trade_threshold", "--trade-threshold"),
    ):
        value = config.get(key)
        if value is not None:
            argv.extend([flag, str(value)])


def build_audit_specs(config: Mapping[str, Any]) -> list[CommandSpec]:
    symbol = str(config["symbol"])
    audit = dict(config.get("audit") or {})
    as_of = str(audit.get("as_of") or RESEARCH_WINDOWS["confirmation"]["end"])
    lookback = int(audit.get("lookback", 1500))
    candle_limit = int(audit.get("candle_probe_limit", 500))
    timeframes = [str(item) for item in audit.get("timeframes", ["H1"])]
    specs = [
        CommandSpec("symbol", ("symbols_describe", symbol, "--detail", "full"), {"kind": "symbol"}),
        CommandSpec(
            "forecast-methods",
            ("forecast_list_methods", "--profile", "all", "--show-unavailable", "true", "--limit", "250", "--detail", "full"),
            {"kind": "catalog"},
        ),
        CommandSpec(
            "denoise-methods",
            ("denoise_list_methods", "--limit", "100", "--detail", "full"),
            {"kind": "catalog"},
        ),
        CommandSpec(
            "indicators",
            ("indicators_list", "--limit", "500", "--detail", "full"),
            {"kind": "catalog"},
        ),
    ]
    for timeframe in timeframes:
        suffix = _slug(timeframe)
        common = (symbol, "--timeframe", timeframe)
        specs.extend(
            [
                CommandSpec(
                    f"candles-{suffix}",
                    (
                        "data_fetch_candles",
                        *common,
                        "--end",
                        as_of,
                        "--limit",
                        str(candle_limit),
                        "--include-spread",
                        "true",
                        "--include-incomplete",
                        "false",
                        "--timestamp-format",
                        "iso_utc",
                        "--detail",
                        "full",
                    ),
                    {"kind": "data_probe", "timeframe": timeframe, "as_of": as_of},
                ),
                CommandSpec(
                    f"stationarity-{suffix}",
                    (
                        "stationarity_test",
                        *common,
                        "--lookback",
                        str(lookback),
                        "--target",
                        "log_return",
                        "--as-of",
                        as_of,
                        "--detail",
                        "full",
                    ),
                    {"kind": "diagnostic", "timeframe": timeframe},
                ),
                CommandSpec(
                    f"seasonality-{suffix}",
                    (
                        "seasonality_detect",
                        *common,
                        "--lookback",
                        str(lookback),
                        "--target",
                        "log_return",
                        "--top-n",
                        "10",
                        "--as-of",
                        as_of,
                        "--detail",
                        "full",
                    ),
                    {"kind": "diagnostic", "timeframe": timeframe},
                ),
                CommandSpec(
                    f"outliers-{suffix}",
                    (
                        "outliers_detect",
                        *common,
                        "--lookback",
                        str(lookback),
                        "--method",
                        "mad",
                        "--limit",
                        "100",
                        "--as-of",
                        as_of,
                        "--detail",
                        "full",
                    ),
                    {"kind": "diagnostic", "timeframe": timeframe},
                ),
                CommandSpec(
                    f"temporal-{suffix}",
                    (
                        "temporal_analyze",
                        *common,
                        "--lookback",
                        str(lookback),
                        "--end",
                        as_of,
                        "--group-by",
                        "all",
                        "--session-calendar",
                        "continuous_24_7",
                        "--detail",
                        "full",
                    ),
                    {"kind": "diagnostic", "timeframe": timeframe},
                ),
                CommandSpec(
                    f"regime-{suffix}",
                    (
                        "regime_detect",
                        *common,
                        "--end",
                        as_of,
                        "--method",
                        "rule_based",
                        "--lookback",
                        str(min(lookback, 1000)),
                        "--detail",
                        "full",
                    ),
                    {"kind": "diagnostic", "timeframe": timeframe},
                ),
            ]
        )
    return specs


def build_screen_specs(config: Mapping[str, Any]) -> list[CommandSpec]:
    symbol = str(config["symbol"])
    screen = dict(config.get("screen") or {})
    windows = dict(config.get("research_windows") or {})
    window_name = str(screen.get("window", "development"))
    if window_name not in windows:
        raise HarnessError(f"Unknown screen window {window_name!r}")
    if bool(windows[window_name].get("locked")):
        raise HarnessError("The locked holdout cannot be used by the screen stage")
    shards = _month_end_shards(window_name, windows[window_name])
    timeframes = [str(item) for item in screen.get("timeframes", ["H1"])]
    horizons = [int(item) for item in screen.get("horizons", [12])]
    quantities = [str(item) for item in screen.get("quantities", ["price"])]
    lookbacks = [int(item) for item in screen.get("lookbacks", [720])]
    methods = [str(item) for item in screen.get("methods", BASELINE_METHODS)]
    for method in methods:
        _reject_external_pretrained_method(method)
    variants = list(screen.get("variants") or [{"id": "raw"}])
    for raw_variant in variants:
        variant = dict(raw_variant or {})
        for method in variant.get("methods", methods):
            _reject_external_pretrained_method(method)
    steps_setting = screen.get("steps_per_shard", "auto_month")
    detail = str(screen.get("detail", "full"))
    specs: list[CommandSpec] = []
    for shard in shards:
        for timeframe in timeframes:
            for horizon in horizons:
                if horizon < 1:
                    raise HarnessError("Screen horizons must be positive")
                steps = _resolve_shard_steps(steps_setting, shard, timeframe, horizon)
                for quantity in quantities:
                    for lookback in lookbacks:
                        available_bars = _continuous_bars(
                            str(windows[window_name]["start"]),
                            shard["end"],
                            timeframe,
                        )
                        required_bars = lookback + (steps * horizon) + horizon
                        if available_bars < required_bars:
                            # Never fill an early development shard with observations
                            # from the stress window merely to satisfy a long lookback.
                            continue
                        for raw_variant in variants:
                            variant = dict(raw_variant or {})
                            variant_id = _slug(variant.get("id", "raw"))
                            command_id = _slug(
                                f"{window_name}-{shard['id']}-{timeframe}-h{horizon}-{quantity}-lb{lookback}-{variant_id}"
                            )
                            variant_methods = [str(item) for item in variant.get("methods", methods)]
                            argv = [
                                "forecast_backtest_run",
                                symbol,
                                "--timeframe",
                                timeframe,
                                "--horizon",
                                str(horizon),
                                "--steps",
                                str(steps),
                                "--spacing",
                                str(horizon),
                                "--lookback",
                                str(lookback),
                                "--methods",
                                *variant_methods,
                                "--quantity",
                                quantity,
                                "--start",
                                str(windows[window_name]["start"]),
                                "--end",
                                shard["end"],
                                "--detail",
                                detail,
                            ]
                            if variant.get("params_per_method") is not None:
                                argv.extend(["--params-per-method", _json_arg(variant["params_per_method"])])
                            _append_optional_pipeline(argv, variant)
                            _append_costs(argv, {**screen, **variant})
                            specs.append(
                                CommandSpec(
                                    command_id,
                                    tuple(argv),
                                    {
                                        "kind": "baseline_screen",
                                        "window": window_name,
                                        "shard": shard,
                                        "timeframe": timeframe,
                                        "horizon": horizon,
                                        "quantity": quantity,
                                        "lookback": lookback,
                                        "steps": steps,
                                        "training_floor": str(windows[window_name]["start"]),
                                        "methods": variant_methods,
                                        "variant": variant_id,
                                    },
                                )
                            )
    return specs


def build_tune_specs(config: Mapping[str, Any], run_dir: Path) -> list[CommandSpec]:
    symbol = str(config["symbol"])
    tune = dict(config.get("tune") or {})
    windows = dict(config.get("research_windows") or {})
    experiments = list(tune.get("experiments") or [])
    specs: list[CommandSpec] = []
    for index, raw_experiment in enumerate(experiments):
        experiment = dict(raw_experiment or {})
        experiment_id = _slug(experiment.get("id", f"tune-{index + 1}"))
        engine = str(experiment.get("engine", "optuna")).lower()
        if engine not in {"optuna", "genetic"}:
            raise HarnessError(f"Tune experiment {experiment_id!r} has unsupported engine {engine!r}")
        window_name = str(experiment.get("window", "development"))
        if window_name not in windows or bool(windows[window_name].get("locked")):
            raise HarnessError(f"Tune experiment {experiment_id!r} must use an unlocked registered window")
        window = windows[window_name]
        horizon = int(experiment.get("horizon", 12))
        method = str(experiment.get("method", "fourier_ols"))
        _reject_external_pretrained_method(method)
        argv = [
            f"forecast_tune_{engine}",
            symbol,
            "--timeframe",
            str(experiment.get("timeframe", "H1")),
            "--horizon",
            str(horizon),
            "--steps",
            str(int(experiment.get("steps", 20))),
            "--spacing",
            str(int(experiment.get("spacing", horizon))),
            "--lookback",
            str(int(experiment.get("lookback", 720))),
            "--methods",
            method,
            "--quantity",
            str(experiment.get("quantity", "return")),
            "--start",
            str(window["start"]),
            "--end",
            str(window["end"]),
            "--metric",
            str(experiment.get("metric", "avg_rmse")),
            "--mode",
            str(experiment.get("mode", "auto")),
            "--seed",
            str(int(experiment.get("seed", config.get("seed", 42)))),
            "--detail",
            str(experiment.get("detail", tune.get("detail", "full"))),
        ]
        if experiment.get("search_space") is not None:
            argv.extend(["--search-space", _json_arg(experiment["search_space"])])
        if engine == "optuna":
            study_db = (run_dir / "tuning" / "optuna.sqlite").resolve()
            argv.extend(
                [
                    "--n-trials",
                    str(int(experiment.get("n_trials", 40))),
                    "--n-jobs",
                    str(int(experiment.get("n_jobs", 1))),
                    "--sampler",
                    str(experiment.get("sampler", "tpe")),
                    "--study-name",
                    f"{_slug(run_dir.name)}-{experiment_id}",
                    "--storage",
                    f"sqlite:///{study_db.as_posix()}",
                ]
            )
            if experiment.get("timeout") is not None:
                argv.extend(["--timeout", str(experiment["timeout"])])
        else:
            argv.extend(
                [
                    "--population",
                    str(int(experiment.get("population", 12))),
                    "--generations",
                    str(int(experiment.get("generations", 10))),
                ]
            )
            if experiment.get("max_search_time_seconds") is not None:
                argv.extend(["--max-search-time-seconds", str(experiment["max_search_time_seconds"])])
        _append_optional_pipeline(argv, experiment)
        _append_costs(argv, experiment)
        specs.append(
            CommandSpec(
                experiment_id,
                tuple(argv),
                {
                    "kind": "tuning",
                    "engine": engine,
                    "window": window_name,
                    "method": method,
                    "timeframe": str(experiment.get("timeframe", "H1")),
                    "horizon": horizon,
                },
            )
        )
    return specs


def build_validation_specs(config: Mapping[str, Any], candidate: Mapping[str, Any]) -> list[CommandSpec]:
    validation = dict(config.get("validate") or {})
    windows = dict(config.get("research_windows") or {})
    window_names = [str(item) for item in validation.get("windows", ["validation", "confirmation"])]
    if window_names != ["validation", "confirmation"]:
        raise HarnessError("Validation must cover the registered validation and confirmation windows in order")
    training_floor = str(validation.get("training_floor", windows["development"]["start"]))
    specs: list[CommandSpec] = []
    candidate_hash = _sha256_json(candidate)
    baseline = dict(candidate.get("provenance", {}).get("baseline") or {})
    baseline_method = str(baseline["method"])
    for window_name in window_names:
        window = windows.get(window_name)
        if not isinstance(window, Mapping) or bool(window.get("locked")):
            raise HarnessError(f"Validation window {window_name!r} is missing or locked")
        for shard in _month_end_shards(window_name, window):
            steps = _resolve_shard_steps(
                validation.get("steps_per_shard", "auto_month"),
                shard,
                str(candidate["timeframe"]),
                int(candidate["horizon"]),
            )
            candidate_args = _candidate_argv(candidate, include_params=False)
            candidate_args.insert(candidate_args.index("--quantity"), baseline_method)
            argv = [
                "forecast_backtest_run",
                "BTCUSD",
                *candidate_args,
                "--params-per-method",
                _json_arg(
                    {
                        str(candidate["method"]): dict(candidate.get("params") or {}),
                        baseline_method: dict(baseline.get("params") or {}),
                    }
                ),
                "--steps",
                str(steps),
                "--spacing",
                str(candidate["horizon"]),
                "--start",
                training_floor,
                "--end",
                shard["end"],
                "--detail",
                str(validation.get("detail", "full")),
            ]
            _append_costs(argv, {**dict(candidate.get("costs") or {}), **validation})
            specs.append(
                CommandSpec(
                    _slug(f"validate-{window_name}-{shard['id']}-{candidate['method']}"),
                    tuple(argv),
                    {
                        "kind": "candidate_validation",
                        "window": window_name,
                        "shard": shard,
                        "training_floor": training_floor,
                        "steps": steps,
                        "candidate_hash": candidate_hash,
                    },
                )
            )
    return specs


_CANDIDATE_REQUIRED_FIELDS = (
    "method",
    "timeframe",
    "horizon",
    "quantity",
    "lookback",
    "params",
    "pipeline",
    "costs",
    "calibration",
    "provenance",
)


def _validate_candidate_pipeline(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HarnessError("Frozen candidate pipeline must be an explicit JSON object")
    pipeline = dict(value)
    missing_pipeline = [key for key in ("denoise", "features", "dimred", "target_spec") if key not in pipeline]
    if missing_pipeline:
        raise HarnessError(f"Frozen candidate pipeline is missing: {', '.join(missing_pipeline)}")
    if pipeline.get("target_spec") is not None:
        raise HarnessError("Backtest validation cannot reproduce a non-null target_spec; use an explicit supported target quantity")
    unknown_pipeline_keys = sorted(set(pipeline) - {"denoise", "features", "dimred", "target_spec"})
    if unknown_pipeline_keys:
        raise HarnessError(f"Frozen candidate pipeline has unsupported fields: {', '.join(unknown_pipeline_keys)}")
    if pipeline.get("features") is not None and not isinstance(pipeline.get("features"), Mapping):
        raise HarnessError("Frozen candidate pipeline.features must be a JSON object or null")
    dimred = pipeline.get("dimred")
    if dimred is not None:
        if not isinstance(dimred, Mapping):
            raise HarnessError("Frozen candidate pipeline.dimred must use the canonical JSON object form or null")
        dimred = dict(dimred)
        unknown_dimred_keys = sorted(set(dimred) - {"method", "params"})
        if unknown_dimred_keys:
            raise HarnessError(
                "Frozen candidate pipeline.dimred has unsupported fields: "
                + ", ".join(unknown_dimred_keys)
            )
        if not str(dimred.get("method") or "").strip():
            raise HarnessError("Frozen candidate pipeline.dimred.method is required")
        if "params" not in dimred or not isinstance(dimred.get("params"), Mapping):
            raise HarnessError("Frozen candidate pipeline.dimred.params must be an explicit JSON object")
        dimred["params"] = dict(dimred["params"])
        pipeline["dimred"] = dimred
    return pipeline


def _validate_candidate_costs(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise HarnessError("Frozen candidate costs must be an explicit JSON object")
    costs = dict(value)
    cost_keys = ("slippage_bps", "spread_bps", "commission_bps_per_side", "trade_threshold")
    unknown_cost_keys = sorted(set(costs) - set(cost_keys))
    if unknown_cost_keys:
        raise HarnessError(f"Frozen candidate costs have unsupported fields: {', '.join(unknown_cost_keys)}")
    missing_costs = [key for key in cost_keys if key not in costs]
    if missing_costs:
        raise HarnessError(f"Frozen candidate costs are missing: {', '.join(missing_costs)}")
    for key in cost_keys:
        try:
            costs[key] = float(costs[key])
        except (TypeError, ValueError) as exc:
            raise HarnessError(f"Frozen candidate cost {key} must be numeric") from exc
        if costs[key] < 0:
            raise HarnessError(f"Frozen candidate cost {key} must not be negative")
    return costs


def _validate_candidate_calibration(
    value: Any,
    *,
    quantity: str,
    pipeline: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HarnessError("Frozen candidate calibration must be an explicit JSON object")
    calibration = dict(value)
    calibration_keys = {"method", "ci_alpha", "steps", "max_mean_relative_width"}
    unknown_calibration_keys = sorted(set(calibration) - calibration_keys)
    if unknown_calibration_keys:
        raise HarnessError(
            "Frozen candidate calibration has unsupported fields: "
            + ", ".join(unknown_calibration_keys)
        )
    missing_calibration = sorted(calibration_keys - set(calibration))
    if missing_calibration:
        raise HarnessError(
            "Frozen candidate calibration is missing: " + ", ".join(missing_calibration)
        )
    method = str(calibration.get("method") or "").strip()
    if not method:
        raise HarnessError("Frozen candidate calibration.method is required")
    if quantity == "price" and not pipeline.get("features") and not pipeline.get("dimred") and method != "conformal":
        raise HarnessError("Univariate price candidates require conformal calibration")
    try:
        ci_alpha = float(calibration["ci_alpha"])
        steps = int(calibration["steps"])
        max_mean_relative_width = float(calibration["max_mean_relative_width"])
    except (TypeError, ValueError) as exc:
        raise HarnessError(
            "Frozen candidate calibration ci_alpha/steps/max_mean_relative_width must be numeric"
        ) from exc
    if abs(ci_alpha - 0.10) > 1e-12:
        raise HarnessError("The approved confidence band is nominal 90% (ci_alpha=0.10)")
    if steps < 100:
        raise HarnessError("Confidence calibration requires at least 100 out-of-sample steps")
    if not 0.0 < max_mean_relative_width <= MAX_MEAN_RELATIVE_WIDTH_CAP:
        raise HarnessError(
            "Calibration max_mean_relative_width must be in "
            f"(0, {MAX_MEAN_RELATIVE_WIDTH_CAP:g}] under the study usability protocol"
        )
    calibration.update(
        {
            "method": method,
            "ci_alpha": 0.10,
            "steps": steps,
            "max_mean_relative_width": max_mean_relative_width,
        }
    )
    return calibration


def _validate_candidate_provenance(value: Any, *, method: str) -> dict[str, Any]:
    _reject_external_pretrained_method(method)
    if not isinstance(value, Mapping):
        raise HarnessError("Frozen candidate provenance must be an explicit JSON object")
    provenance = dict(value)
    evidence = provenance.get("selection_artifacts")
    if not isinstance(evidence, list) or not evidence:
        raise HarnessError("Frozen candidate provenance.selection_artifacts must name the selection evidence")
    baseline = provenance.get("baseline")
    if not isinstance(baseline, Mapping):
        raise HarnessError("Frozen candidate provenance.baseline must preregister a comparison model")
    baseline = dict(baseline)
    unknown_baseline_keys = sorted(
        set(baseline) - {"method", "params", "minimum_accuracy_improvement"}
    )
    if unknown_baseline_keys:
        raise HarnessError(
            "Frozen candidate provenance.baseline has unsupported fields: "
            + ", ".join(unknown_baseline_keys)
        )
    if not str(baseline.get("method") or "").strip():
        raise HarnessError("Frozen candidate provenance.baseline.method is required")
    _reject_external_pretrained_method(baseline["method"])
    if str(baseline["method"]) == method:
        raise HarnessError("Frozen candidate baseline method must differ from the candidate method")
    if not isinstance(baseline.get("params"), Mapping):
        raise HarnessError("Frozen candidate provenance.baseline.params must be an explicit JSON object")
    baseline["params"] = dict(baseline["params"])
    minimum_improvement = _numeric(baseline.get("minimum_accuracy_improvement", 0.02))
    if minimum_improvement is None or minimum_improvement < 0:
        raise HarnessError("Frozen candidate baseline.minimum_accuracy_improvement must be non-negative")
    baseline["minimum_accuracy_improvement"] = minimum_improvement
    provenance["baseline"] = baseline
    if any(provenance.get(key) for key in ("external_model_id", "external_prior", "model_identifier")):
        raise HarnessError("External model/prior identifiers are forbidden by the MT5-only study contract")
    return provenance


def _validate_frozen_candidate(candidate: Any) -> dict[str, Any]:
    if not isinstance(candidate, Mapping):
        raise HarnessError("Frozen candidate input must be a JSON object")
    normalized = dict(candidate)
    if isinstance(normalized.get("candidate"), Mapping):
        normalized = dict(normalized["candidate"])
    missing = [key for key in _CANDIDATE_REQUIRED_FIELDS if normalized.get(key) is None]
    if missing:
        raise HarnessError(f"Frozen candidate is missing required fields: {', '.join(missing)}")
    allowed_candidate_keys = {"symbol", "candidate_id", "description", *_CANDIDATE_REQUIRED_FIELDS}
    unknown_candidate_keys = sorted(set(normalized) - allowed_candidate_keys)
    if unknown_candidate_keys:
        raise HarnessError(
            "Frozen candidate contains unsupported fields that would not reach mtdata: "
            + ", ".join(unknown_candidate_keys)
        )
    if str(normalized.get("symbol", "BTCUSD")).upper() != "BTCUSD":
        raise HarnessError("This harness only accepts a BTCUSD frozen candidate")
    normalized["symbol"] = "BTCUSD"
    normalized["method"] = str(normalized["method"])
    normalized["timeframe"] = str(normalized["timeframe"])
    normalized["horizon"] = int(normalized["horizon"])
    normalized["lookback"] = int(normalized["lookback"])
    normalized["quantity"] = str(normalized["quantity"])
    if normalized["horizon"] < 1 or normalized["lookback"] < 1:
        raise HarnessError("Frozen candidate horizon and lookback must be positive")
    if normalized["quantity"] not in {"price", "return", "volatility"}:
        raise HarnessError("Frozen candidate quantity must be price, return, or volatility")
    if not isinstance(normalized["params"], Mapping):
        raise HarnessError("Frozen candidate params must be an explicit JSON object (use {} when empty)")
    normalized["params"] = dict(normalized["params"])
    normalized["pipeline"] = _validate_candidate_pipeline(normalized["pipeline"])
    normalized["costs"] = _validate_candidate_costs(normalized["costs"])
    normalized["calibration"] = _validate_candidate_calibration(
        normalized["calibration"],
        quantity=normalized["quantity"],
        pipeline=normalized["pipeline"],
    )
    normalized["provenance"] = _validate_candidate_provenance(
        normalized["provenance"],
        method=normalized["method"],
    )
    return normalized


def _candidate_argv(
    candidate: Mapping[str, Any],
    *,
    plural_method: bool = True,
    include_params: bool = True,
) -> list[str]:
    method_flag = "--methods" if plural_method else "--method"
    argv = [
        "--timeframe",
        str(candidate["timeframe"]),
        "--horizon",
        str(candidate["horizon"]),
        "--lookback",
        str(candidate["lookback"]),
        method_flag,
        str(candidate["method"]),
        "--quantity",
        str(candidate["quantity"]),
    ]
    pipeline = dict(candidate.get("pipeline") or {})
    _append_optional_pipeline(
        argv,
        {
            "params": candidate.get("params") if include_params else None,
            "denoise": pipeline.get("denoise"),
            "features": pipeline.get("features"),
            "dimred": pipeline.get("dimred"),
        },
    )
    return argv


def build_holdout_specs(config: Mapping[str, Any], candidate: Mapping[str, Any]) -> list[CommandSpec]:
    holdout = dict(config.get("holdout") or {})
    windows = dict(config.get("research_windows") or {})
    window_name = str(holdout.get("window", "locked_holdout"))
    window = windows.get(window_name)
    if not isinstance(window, Mapping) or not bool(window.get("locked")):
        raise HarnessError("Holdout stage requires a registered locked window")
    baseline = dict(candidate.get("provenance", {}).get("baseline") or {})
    baseline_method = str(baseline["method"])
    specs: list[CommandSpec] = []
    for shard in _month_end_shards(window_name, window):
        steps = _resolve_shard_steps(
            holdout.get("steps_per_shard", "auto_month"),
            shard,
            str(candidate["timeframe"]),
            int(candidate["horizon"]),
        )
        candidate_args = _candidate_argv(candidate, include_params=False)
        candidate_args.insert(candidate_args.index("--quantity"), baseline_method)
        argv = [
            "forecast_backtest_run",
            "BTCUSD",
            *candidate_args,
            "--params-per-method",
            _json_arg(
                {
                    str(candidate["method"]): dict(candidate.get("params") or {}),
                    baseline_method: dict(baseline.get("params") or {}),
                }
            ),
            "--steps",
            str(steps),
            "--spacing",
            str(int(candidate["horizon"])),
            "--end",
            shard["end"],
            "--detail",
            str(holdout.get("detail", "full")),
        ]
        _append_costs(argv, {**dict(candidate.get("costs") or {}), **holdout})
        specs.append(
            CommandSpec(
                _slug(f"holdout-{shard['id']}-{candidate['method']}"),
                tuple(argv),
                {
                    "kind": "locked_holdout",
                    "window": window_name,
                    "shard": shard,
                    "candidate_hash": _sha256_json(candidate),
                },
            )
        )
    return specs


def build_shadow_specs(
    config: Mapping[str, Any],
    candidate: Mapping[str, Any],
    as_of: str | None,
    *,
    model_id: str | None = None,
) -> list[CommandSpec]:
    shadow = dict(config.get("shadow") or {})
    incompatible = []
    if str(candidate.get("quantity")) != "price":
        incompatible.append(f"quantity={candidate.get('quantity')}")
    pipeline = dict(candidate.get("pipeline") or {})
    if pipeline.get("features"):
        incompatible.append("features")
    if pipeline.get("dimred"):
        incompatible.append("dimred")
    if incompatible:
        raise HarnessError(
            "Shadow conformal calibration cannot reproduce the frozen pipeline "
            f"({', '.join(incompatible)}). Provide an external pipeline-matched calibration path "
            "before shadow forecasting, or freeze a univariate price candidate."
        )
    observation = as_of or utc_now()
    observation_id = _slug(observation)
    generate_argv = [
        "forecast_generate",
        "BTCUSD",
        *_candidate_argv(candidate, plural_method=False),
        "--ci-alpha",
        str(candidate.get("calibration", {}).get("ci_alpha", shadow.get("ci_alpha", 0.10))),
        "--model-cache",
        str(shadow.get("model_cache", "require_existing")),
        "--detail",
        str(shadow.get("detail", "full")),
    ]
    conformal_argv = [
        "forecast_conformal_intervals",
        "BTCUSD",
        "--timeframe",
        str(candidate["timeframe"]),
        "--horizon",
        str(candidate["horizon"]),
        "--lookback",
        str(candidate["lookback"]),
        "--method",
        str(candidate["method"]),
        "--steps",
        str(int(candidate.get("calibration", {}).get("steps", shadow.get("conformal_steps", 100)))),
        "--spacing",
        str(int(candidate["horizon"])),
        "--ci-alpha",
        str(candidate.get("calibration", {}).get("ci_alpha", shadow.get("ci_alpha", 0.10))),
        "--detail",
        str(shadow.get("detail", "full")),
    ]
    if as_of:
        generate_argv.extend(["--as-of", as_of])
        conformal_argv.extend(["--as-of", as_of])
    if model_id:
        generate_argv.extend(["--model-id", model_id])
    for key, value in (
        ("params", candidate.get("params")),
        ("denoise", pipeline.get("denoise")),
    ):
        if value is not None:
            conformal_argv.extend([f"--{key}", _json_arg(value) if isinstance(value, (dict, list)) else str(value)])
    metadata = {"kind": "shadow", "as_of": observation, "candidate_hash": _sha256_json(candidate)}
    return [
        CommandSpec(f"shadow-{observation_id}-forecast", tuple(generate_argv), metadata),
        CommandSpec(f"shadow-{observation_id}-conformal", tuple(conformal_argv), metadata),
    ]


def build_materialize_specs(
    config: Mapping[str, Any],
    candidate: Mapping[str, Any],
    as_of: str | None,
) -> list[CommandSpec]:
    shadow = dict(config.get("shadow") or {})
    observation = as_of or utc_now()
    argv = [
        "forecast_generate",
        "BTCUSD",
        *_candidate_argv(candidate, plural_method=False),
        "--ci-alpha",
        str(candidate.get("calibration", {}).get("ci_alpha", shadow.get("ci_alpha", 0.10))),
        "--model-cache",
        "reuse",
        "--detail",
        str(shadow.get("detail", "full")),
    ]
    if as_of:
        argv.extend(["--as-of", as_of])
    return [
        CommandSpec(
            f"materialize-{_slug(observation)}",
            tuple(argv),
            {
                "kind": "model_materialization",
                "as_of": observation,
                "candidate_hash": _sha256_json(candidate),
                "refit_interval_bars": int(shadow.get("refit_interval_bars", 24)),
                "cache_policy": "reuse",
            },
        )
    ]


def _stage_config_hash(stage: str, config: Mapping[str, Any], extra: Any = None) -> str:
    stage_config = config.get(stage)
    value = {
        "symbol": config.get("symbol"),
        "seed": config.get("seed"),
        "runtime": config.get("runtime"),
        "research_windows": config.get("research_windows"),
        "stage": stage_config,
        "extra": extra,
    }
    return _sha256_json(redact(value))


def _plan_digest(specs: Sequence[CommandSpec]) -> str:
    return _sha256_json(
        [
            {
                "command_id": spec.command_id,
                "argv": list(spec.argv),
                "metadata": spec.metadata,
            }
            for spec in specs
        ]
    )


def _new_manifest(run_id: str, config: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    now = utc_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "harness_version": HARNESS_VERSION,
        "run_id": run_id,
        "symbol": "BTCUSD",
        "status": "active",
        "created_at": now,
        "updated_at": now,
        "runtime": {
            "python": sys.version,
            "python_executable": str(Path(sys.executable).resolve()),
            "platform": platform.platform(),
            "mtdata_invocation": [str(Path(sys.executable).resolve()), "-m", "mtdata", "--json"],
            "cuda_visible_devices": str(config.get("runtime", {}).get("cuda_visible_devices", "0,1")),
            "cuda_visibility_scope": "child_process_only",
            "gpu_device_probe": "forecast catalog only; mtdata has no dedicated GPU inventory CLI command",
        },
        "safety": {
            "research_only": True,
            "symbol_allowlist": ["BTCUSD"],
            "trade_command_prefixes_forbidden": list(FORBIDDEN_COMMAND_PREFIXES),
            "shell_command_forbidden": True,
            "subprocess_shell": False,
            "isolated_model_store": "model_store",
            "isolated_jobs_db": "forecast_jobs.sqlite",
            "no_trade_execution": True,
            "mt5_market_data_only": True,
            "external_pretrained_methods_forbidden": sorted(EXTERNALLY_PRETRAINED_METHODS),
        },
        "protocol": {
            "research_windows": redact(config.get("research_windows", {})),
            "baseline_matrix": redact(
                {
                    key: config.get("screen", {}).get(key)
                    for key in (
                        "timeframes",
                        "horizons",
                        "quantities",
                        "lookbacks",
                        "methods",
                        "spacing_policy",
                        "sharding",
                        "steps_per_shard",
                        "slippage_bps",
                        "spread_bps",
                        "commission_bps_per_side",
                        "trade_threshold",
                        "variants",
                    )
                }
            ),
            "screen_training_floor_policy": (
                "Every command passes the registered window start; shards without "
                "lookback plus full evaluation history inside that window are omitted."
            ),
            "candidate_gate": {
                "validation_windows": ["validation", "confirmation"],
                "validation_candidate_write_once": True,
                "interval_approval_enabled": INTERVAL_APPROVAL_ENABLED,
                "interval_approval_disabled_reason": INTERVAL_APPROVAL_DISABLED_REASON,
                "freeze_requires_validation_evidence": True,
                "freeze_requires_terminal_path_baseline_significance": True,
                "freeze_requires_oos_interval_coverage_and_width": True,
                "interval_nominal_coverage": 0.90,
                "interval_relative_width_definition": RELATIVE_WIDTH_DEFINITION,
                "interval_requires_candidate_width_ceiling": True,
                "interval_max_mean_relative_width_cap": MAX_MEAN_RELATIVE_WIDTH_CAP,
                "interval_native_diagnostic_is_not_a_guarantee": True,
                "external_interval_replay_artifact_required": True,
                "self_reported_interval_summaries_are_approval_evidence": False,
                "confidence_approval_quantity": "price_only_until_pipeline_matched_support_exists",
                "freeze_requires_same_candidate_hash": True,
            },
            "locked_holdout_policy": {
                "requires_frozen_candidate": True,
                "requires_candidate_hash": True,
                "writes_once": "holdout_lock.json",
                "opened_at_recorded_before_access": True,
                "post_holdout_decision": "holdout_decision.json",
                "materialize_requires_approved_for_shadow": True,
                "no_reselection_after_open": True,
            },
            "shadow_policy": {
                "materialize_before_shadow": True,
                "materialize_cache_policy": "reuse",
                "shadow_cache_policy": "require_existing",
                "refit_interval_bars": int(config.get("shadow", {}).get("refit_interval_bars", 24)),
                "lifecycle": "append_only_active",
                "ci_alpha": float(config.get("shadow", {}).get("ci_alpha", 0.10)),
                "conformal_steps": int(config.get("shadow", {}).get("conformal_steps", 100)),
            },
        },
        "paths": {
            "run_dir": str(run_dir.resolve()),
            "manifest": "manifest.json",
            "issues": "issues.json",
            "raw": "raw",
            "normalized": "normalized",
            "model_store": "model_store",
            "jobs_db": "forecast_jobs.sqlite",
        },
        "config_revisions": [],
        "stages": {stage: {"status": "not_started", "commands": {}} for stage in RESEARCH_STAGES},
    }


def _save_manifest(context: RunContext) -> None:
    context.manifest["updated_at"] = utc_now()
    secrets = _secret_values()
    _atomic_write_json(context.manifest_path, redact(context.manifest, known_secrets=secrets))


def _record_config_revision(context: RunContext) -> None:
    secrets = _secret_values()
    safe_config = redact(context.config, known_secrets=secrets)
    digest = _sha256_json(safe_config)
    revision_path = context.run_dir / "config_revisions" / f"{digest}.json"
    if not revision_path.exists():
        _write_new_json(
            revision_path,
            {
                "schema_version": SCHEMA_VERSION,
                "recorded_at": utc_now(),
                "config_hash": digest,
                "config": safe_config,
            },
        )
    revisions = context.manifest.setdefault("config_revisions", [])
    if not any(item.get("config_hash") == digest for item in revisions if isinstance(item, Mapping)):
        revisions.append(
            {
                "config_hash": digest,
                "recorded_at": utc_now(),
                "path": _relative(revision_path, context.run_dir),
            }
        )
    context.manifest["active_config_hash"] = digest
    _atomic_write_json(context.run_dir / "resolved_config.json", safe_config)


def _empty_issue_ledger(run_id: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "updated_at": utc_now(),
        "issues": [],
    }


def _load_issue_ledger(run_dir: Path, run_id: str | None = None) -> dict[str, Any]:
    path = run_dir / "issues.json"
    if not path.exists():
        return _empty_issue_ledger(run_id or run_dir.name)
    value = _load_json(path)
    if not isinstance(value, Mapping) or not isinstance(value.get("issues"), list):
        raise HarnessError(f"Invalid issue ledger: {path}")
    return dict(value)


def _issue_fingerprint(issue: Mapping[str, Any]) -> str:
    return _sha256_json(
        {
            "category": issue.get("category"),
            "reproduction_command": issue.get("reproduction_command"),
            "context": issue.get("context"),
            "impact": issue.get("impact"),
            "suggested_fix": issue.get("suggested_fix"),
        }
    )


def append_issue(run_dir: Path, issue: Mapping[str, Any], *, run_id: str | None = None) -> tuple[dict[str, Any], bool]:
    """Append or deterministically deduplicate one redacted friction record."""
    secrets = _secret_values()
    now = utc_now()
    safe = redact(dict(issue), known_secrets=secrets)
    safe.setdefault("severity", "medium")
    safe.setdefault("category", "mtdata_cli")
    safe.setdefault("observed_at", now)
    safe.setdefault("reproduction_command", [])
    safe.setdefault("context", {})
    safe.setdefault("impact", "")
    safe.setdefault("workaround", "")
    safe.setdefault("suggested_fix", "")
    safe.setdefault("status", "open")
    reproduction = safe.get("reproduction_command")
    if isinstance(reproduction, list):
        safe["reproduction_command"] = _redact_invocation(
            [str(item) for item in reproduction],
            secrets,
        )
    fingerprint = _issue_fingerprint(safe)
    if not safe.get("id"):
        safe["id"] = f"MTDATA-{fingerprint[:12].upper()}"
    safe["fingerprint"] = fingerprint
    safe["last_observed_at"] = safe.get("observed_at", now)
    safe["occurrences"] = 1

    ledger = _load_issue_ledger(run_dir, run_id)
    issues = ledger["issues"]
    for existing in issues:
        if not isinstance(existing, dict):
            continue
        if existing.get("id") == safe["id"] or existing.get("fingerprint") == fingerprint:
            existing["last_observed_at"] = safe["last_observed_at"]
            existing["occurrences"] = int(existing.get("occurrences", 1)) + 1
            for key in ("severity", "status", "workaround", "suggested_fix"):
                if safe.get(key):
                    existing[key] = safe[key]
            ledger["updated_at"] = now
            _atomic_write_json(run_dir / "issues.json", ledger)
            return existing, False
    issues.append(safe)
    ledger["updated_at"] = now
    _atomic_write_json(run_dir / "issues.json", ledger)
    return safe, True


def _register_specs(
    context: RunContext,
    stage: str,
    specs: Sequence[CommandSpec],
    *,
    extra_hash_input: Any = None,
    allow_append: bool = False,
) -> None:
    stage_record = context.manifest.setdefault("stages", {}).setdefault(stage, {"status": "not_started", "commands": {}})
    config_hash = _stage_config_hash(stage, context.config, extra_hash_input)
    previous_hash = stage_record.get("config_hash")
    commands = stage_record.setdefault("commands", {})
    has_execution = any(
        isinstance(record, Mapping) and record.get("status") not in {"preregistered", "planned"}
        for record in commands.values()
    )
    if previous_hash and previous_hash != config_hash and has_execution:
        raise HarnessError(
            f"Stage {stage!r} configuration is locked after execution; start a new study for changed inputs"
        )
    stage_record["config_hash"] = config_hash
    stage_record["plan_digest"] = _plan_digest(specs)
    stage_record["commands_planned"] = len(specs)
    known_secrets = _secret_values()
    expected_ids: set[str] = set()
    for spec in specs:
        _validate_research_command(spec.argv)
        expected_ids.add(spec.command_id)
        safe_argv = _redact_invocation(spec.argv, known_secrets)
        existing = commands.get(spec.command_id)
        if existing is not None:
            if existing.get("argv") != safe_argv:
                raise HarnessError(f"Command {spec.command_id!r} changed after preregistration")
            existing["metadata"] = redact(dict(spec.metadata), known_secrets=known_secrets)
            continue
        commands[spec.command_id] = {
            "status": "preregistered",
            "argv": safe_argv,
            "metadata": redact(dict(spec.metadata), known_secrets=known_secrets),
            "attempts": 0,
        }
    unexpected = [command_id for command_id in commands if command_id not in expected_ids]
    if unexpected and has_execution and not allow_append:
        raise HarnessError(f"Stage {stage!r} plan removed registered commands: {', '.join(unexpected[:5])}")
    if not allow_append:
        for command_id in unexpected:
            commands.pop(command_id, None)
    else:
        stage_record["commands_planned"] = len(commands)
        stage_record["plan_digest"] = _sha256_json(
            [
                {
                    "command_id": command_id,
                    "argv": command.get("argv"),
                    "metadata": command.get("metadata"),
                }
                for command_id, command in sorted(commands.items())
            ]
        )


def _preregister_default_plan(context: RunContext) -> None:
    _register_specs(context, "audit", build_audit_specs(context.config))
    _register_specs(context, "screen", build_screen_specs(context.config))
    tune_specs = build_tune_specs(context.config, context.run_dir)
    if tune_specs:
        _register_specs(context, "tune", tune_specs)


def _resolve_config(config_path: Path | None, existing_run_dir: Path | None = None) -> dict[str, Any]:
    if config_path is not None:
        loaded = _load_json(config_path)
        if not isinstance(loaded, Mapping):
            raise HarnessError("Experiment configuration must be a JSON object")
        base: Mapping[str, Any] = DEFAULT_CONFIG
        stored_path = existing_run_dir / "resolved_config.json" if existing_run_dir is not None else None
        if stored_path is not None and stored_path.exists():
            stored = _load_json(stored_path)
            if not isinstance(stored, Mapping):
                raise HarnessError("Stored resolved configuration must be a JSON object")
            base = stored
        config = _deep_merge(base, loaded)
    elif existing_run_dir is not None and (existing_run_dir / "resolved_config.json").exists():
        loaded = _load_json(existing_run_dir / "resolved_config.json")
        if not isinstance(loaded, Mapping):
            raise HarnessError("Stored resolved configuration must be a JSON object")
        config = dict(loaded)
    else:
        config = copy.deepcopy(DEFAULT_CONFIG)
    if str(config.get("symbol", "")).upper() != "BTCUSD":
        raise HarnessError("This research harness is restricted to BTCUSD")
    config["symbol"] = "BTCUSD"
    windows = config.get("research_windows")
    if not isinstance(windows, Mapping):
        raise HarnessError("research_windows must be a JSON object")
    locked = windows.get("locked_holdout")
    if not isinstance(locked, Mapping) or not bool(locked.get("locked")):
        raise HarnessError("The locked_holdout window must exist and remain locked")
    return config


def prepare_context(args: argparse.Namespace) -> RunContext:
    if args.resume and args.run_dir is None:
        raise HarnessError("--resume requires --run-dir")
    now_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = _slug(args.run_id or now_id)
    run_dir = Path(args.run_dir) if args.run_dir is not None else DEFAULT_OUTPUT_ROOT / run_id
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise HarnessError(f"Study already exists at {run_dir}; pass --resume to use it")
        loaded_manifest = _load_json(manifest_path)
        if not isinstance(loaded_manifest, Mapping):
            raise HarnessError("Stored manifest must be a JSON object")
        manifest = dict(loaded_manifest)
        run_id = str(manifest.get("run_id") or run_id)
        config = _resolve_config(Path(args.config) if args.config else None, run_dir)
    else:
        if args.resume:
            raise HarnessError(f"No study manifest exists at {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        config = _resolve_config(Path(args.config) if args.config else None)
        manifest = _new_manifest(run_id, config, run_dir)
    context = RunContext(
        run_dir=run_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        config=config,
        dry_run=bool(args.dry_run),
        timeout=float(args.timeout),
        max_commands=args.max_commands,
        fail_fast=bool(args.fail_fast),
    )
    protocol = context.manifest.setdefault("protocol", {})
    if not isinstance(protocol, dict):
        raise HarnessError("Stored manifest protocol must be a JSON object")
    candidate_gate = protocol.setdefault("candidate_gate", {})
    if not isinstance(candidate_gate, dict):
        raise HarnessError("Stored manifest candidate_gate must be a JSON object")
    candidate_gate.update(
        {
            "interval_approval_enabled": INTERVAL_APPROVAL_ENABLED,
            "interval_approval_disabled_reason": INTERVAL_APPROVAL_DISABLED_REASON,
            "self_reported_interval_summaries_are_approval_evidence": False,
        }
    )
    for directory in ("raw", "normalized", "model_store", "tuning", "materializations", "config_revisions"):
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    if not (run_dir / "issues.json").exists():
        _atomic_write_json(run_dir / "issues.json", _empty_issue_ledger(run_id))
    _record_config_revision(context)
    _preregister_default_plan(context)
    _save_manifest(context)
    return context


def _command_environment(context: RunContext) -> dict[str, str]:
    environment = os.environ.copy()
    environment["MTDATA_MODEL_STORE"] = str((context.run_dir / "model_store").resolve())
    environment["MTDATA_FORECAST_JOBS_DB"] = str((context.run_dir / "forecast_jobs.sqlite").resolve())
    environment["MTDATA_MODEL_TTL_DAYS"] = "3650"
    environment["MTDATA_OUTPUT_FORMAT"] = "json"
    environment["PYTHONUNBUFFERED"] = "1"
    cuda_visible_devices = context.config.get("runtime", {}).get("cuda_visible_devices")
    if cuda_visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    return environment


def _payload_succeeded(payload: Any, returncode: int) -> bool:
    if returncode != 0:
        return False
    if isinstance(payload, Mapping):
        if payload.get("success") is False:
            return False
        if payload.get("error") not in (None, "", False):
            return False
    return True


def _shard_contract_error(spec: CommandSpec, payload: Any) -> str | None:
    shard = spec.metadata.get("shard") if isinstance(spec.metadata, Mapping) else None
    if not isinstance(shard, Mapping):
        return None
    if not isinstance(payload, Mapping):
        return "A sharded backtest did not return a JSON object"
    analysis = payload.get("analysis_time_window")
    if not isinstance(analysis, Mapping):
        return "A sharded backtest omitted analysis_time_window"
    evaluation_start = str(analysis.get("evaluation_start") or "")[:10]
    evaluation_end = str(analysis.get("evaluation_end") or "")[:10]
    shard_start = str(shard.get("start") or "")[:10]
    shard_end = str(shard.get("end") or "")[:10]
    try:
        evaluation_start_date = date.fromisoformat(evaluation_start)
        evaluation_end_date = date.fromisoformat(evaluation_end)
        shard_start_date = date.fromisoformat(shard_start)
        shard_end_date = date.fromisoformat(shard_end)
    except ValueError:
        return "A sharded backtest returned invalid or missing evaluation timestamps"
    if evaluation_start_date < shard_start_date:
        return (
            f"Backtest evaluation_start={evaluation_start} precedes registered shard start={shard_start}; "
            "feed gaps may have duplicated anchors from the prior shard"
        )
    if evaluation_end_date > shard_end_date:
        return f"Backtest evaluation_end={evaluation_end} exceeds registered shard end={shard_end}"
    return None


def _raw_path(context: RunContext, stage: str, command_id: str) -> Path:
    return context.run_dir / "raw" / stage / f"{_slug(command_id)}.json"


def _record_automatic_issue(
    context: RunContext,
    stage: str,
    spec: CommandSpec,
    impact: str,
    raw_path: Path,
) -> None:
    append_issue(
        context.run_dir,
        {
            "severity": "high",
            "category": "mtdata_cli",
            "observed_at": utc_now(),
            "reproduction_command": [sys.executable, "-m", "mtdata", "--json", *spec.argv],
            "context": {"stage": stage, "command_id": spec.command_id, "raw_output": _relative(raw_path, context.run_dir)},
            "impact": impact,
            "workaround": "Inspect the persisted raw envelope and resume after correcting the underlying issue.",
            "suggested_fix": "Make the CLI failure actionable and preserve a stable machine-readable error code.",
            "status": "open",
        },
        run_id=str(context.manifest.get("run_id")),
    )


def _backtest_anchor_status(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("results"), Mapping):
        return None
    total_anchors = 0
    successful_anchors = 0
    methods_with_partial_failures: dict[str, int] = {}
    derived_methods_succeeded = 0
    derived_complete_methods = 0
    derived_partial_methods = 0
    derived_failed_methods = 0
    methods_observed = 0
    for method, result in payload["results"].items():
        if not isinstance(result, Mapping):
            continue
        try:
            num_tests = int(result["num_tests"])
            successful_tests = int(result["successful_tests"])
        except (KeyError, TypeError, ValueError):
            continue
        methods_observed += 1
        total_anchors += max(0, num_tests)
        successful_anchors += max(0, min(successful_tests, num_tests))
        failures = max(0, num_tests - successful_tests)
        if result.get("success") is True:
            derived_methods_succeeded += 1
            if failures:
                derived_partial_methods += 1
            else:
                derived_complete_methods += 1
        else:
            derived_failed_methods += 1
        if failures:
            methods_with_partial_failures[str(method)] = failures
    if not methods_observed:
        return None
    partial_failures = max(0, total_anchors - successful_anchors)
    count_keys = (
        "methods_total",
        "methods_succeeded",
        "methods_complete",
        "methods_partial",
        "methods_failed",
        "anchor_tests_planned",
        "anchor_tests_succeeded",
        "anchor_tests_failed",
    )
    try:
        reported_counts = {key: int(payload[key]) for key in count_keys}
    except (KeyError, TypeError, ValueError):
        reported_counts = {}
    reported_methods_failed = payload.get("methods_failed")
    explicit_partial_contract = (
        bool(reported_counts)
        and payload.get("status") in {"complete", "partial", "failed"}
        and isinstance(payload.get("complete_success"), bool)
    )
    expected_complete_success = methods_observed > 0 and derived_complete_methods == methods_observed
    expected_status = (
        "complete"
        if expected_complete_success
        else "partial"
        if derived_methods_succeeded
        else "failed"
    )
    explicit_contract_consistent = (
        reported_counts
        == {
            "methods_total": methods_observed,
            "methods_succeeded": derived_methods_succeeded,
            "methods_complete": derived_complete_methods,
            "methods_partial": derived_partial_methods,
            "methods_failed": derived_failed_methods,
            "anchor_tests_planned": total_anchors,
            "anchor_tests_succeeded": successful_anchors,
            "anchor_tests_failed": partial_failures,
        }
        and payload.get("complete_success") is expected_complete_success
        and payload.get("status") == expected_status
        if explicit_partial_contract
        else None
    )
    legacy_ambiguity = (
        not explicit_partial_contract
        and reported_methods_failed == 0
        and partial_failures > 0
    )
    return {
        "total_anchors": total_anchors,
        "successful_anchors": successful_anchors,
        "partial_anchor_failures": partial_failures,
        "methods_with_partial_anchor_failures": methods_with_partial_failures,
        "reported_methods_failed": reported_methods_failed,
        "reported_counts": reported_counts,
        "explicit_partial_contract": explicit_partial_contract,
        "explicit_partial_contract_consistent": explicit_contract_consistent,
        "legacy_methods_failed_ambiguity": legacy_ambiguity,
        "methods_failed_contract_disagreement": legacy_ambiguity,
    }


def _record_anchor_contract_issue(
    context: RunContext,
    stage: str,
    spec: CommandSpec,
    payload: Any,
    raw_path: Path,
) -> None:
    anchor_status = _backtest_anchor_status(payload)
    if not anchor_status or anchor_status["legacy_methods_failed_ambiguity"] is not True:
        return
    append_issue(
        context.run_dir,
        {
            "id": "BTC-R017",
            "severity": "high",
            "category": "cli_contract",
            "observed_at": utc_now(),
            "reproduction_command": [sys.executable, "-m", "mtdata", "--json", *spec.argv],
            "context": {
                "stage": stage,
                "command_id": spec.command_id,
                "raw_output": _relative(raw_path, context.run_dir),
                "derived_anchor_status": anchor_status,
            },
            "impact": (
                "Top-level methods_failed=0 masks anchor-level failures and can make an incomplete "
                "candidate appear fully evaluated."
            ),
            "workaround": "Use derived partial_anchor_failures from each method's num_tests and successful_tests.",
            "suggested_fix": (
                "Expose anchor_failures and partial_methods at the top level, or make methods_failed include "
                "methods with any failed evaluation anchors."
            ),
            "status": "open",
        },
        run_id=str(context.manifest.get("run_id")),
    )


def _explicit_anchor_contract_error(payload: Any) -> str | None:
    anchor_status = _backtest_anchor_status(payload)
    if (
        anchor_status
        and anchor_status["explicit_partial_contract"] is True
        and anchor_status["explicit_partial_contract_consistent"] is False
    ):
        return (
            "Explicit top-level anchor/method partial counts disagree with per-method "
            "num_tests and successful_tests"
        )
    return None


def execute_spec(context: RunContext, stage: str, spec: CommandSpec) -> bool | None:
    """Execute one registered command; return True, False, or None for dry-run."""
    _validate_research_command(spec.argv)
    stage_record = context.manifest["stages"][stage]
    command_record = stage_record["commands"][spec.command_id]
    raw_path = _raw_path(context, stage, spec.command_id)
    if command_record.get("status") == "completed" and raw_path.exists():
        return True
    invocation = [sys.executable, "-m", "mtdata", "--json", *spec.argv]
    known_secrets = _secret_values()
    if context.dry_run:
        envelope = {
            "schema_version": SCHEMA_VERSION,
            "stage": stage,
            "command_id": spec.command_id,
            "status": "planned",
            "dry_run": True,
            "invocation": _redact_invocation(invocation, known_secrets),
            "metadata": redact(dict(spec.metadata), known_secrets=known_secrets),
            "planned_at": utc_now(),
        }
        _atomic_write_json(raw_path, envelope)
        command_record["status"] = "planned"
        command_record["raw_path"] = _relative(raw_path, context.run_dir)
        _save_manifest(context)
        return None

    command_record["status"] = "running"
    command_record["attempts"] = int(command_record.get("attempts", 0)) + 1
    command_record["started_at"] = utc_now()
    _save_manifest(context)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            invocation,
            cwd=Path(__file__).resolve().parents[1],
            env=_command_environment(context),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=context.timeout,
            check=False,
            shell=False,
        )
        returncode = int(completed.returncode)
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        parse_error: str | None = None
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            payload = None
            parse_error = f"mtdata stdout was not valid JSON: {exc}"
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout = str(exc.stdout or "")
        stderr = str(exc.stderr or "")
        payload = None
        parse_error = f"mtdata command exceeded {context.timeout:g} seconds"
    finished = utc_now()
    contract_error = None
    if parse_error is None:
        contract_error = _shard_contract_error(spec, payload) or _explicit_anchor_contract_error(payload)
    success = parse_error is None and contract_error is None and _payload_succeeded(payload, returncode)
    safe_payload = redact(payload, known_secrets=known_secrets)
    envelope = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "command_id": spec.command_id,
        "status": "completed" if success else "failed",
        "invocation": _redact_invocation(invocation, known_secrets),
        "metadata": redact(dict(spec.metadata), known_secrets=known_secrets),
        "started_at": command_record["started_at"],
        "finished_at": finished,
        "duration_seconds": round(time.perf_counter() - started, 6),
        "returncode": returncode,
        "payload": safe_payload,
        "stderr": _redact_text(stderr, known_secrets),
    }
    if parse_error:
        envelope["parse_error"] = parse_error
        envelope["stdout"] = _redact_text(stdout, known_secrets)
    if contract_error:
        envelope["contract_error"] = contract_error
    _atomic_write_json(raw_path, envelope)
    _record_anchor_contract_issue(context, stage, spec, safe_payload, raw_path)
    command_record.update(
        {
            "status": envelope["status"],
            "finished_at": finished,
            "duration_seconds": envelope["duration_seconds"],
            "returncode": returncode,
            "raw_path": _relative(raw_path, context.run_dir),
        }
    )
    if not success:
        error_text = parse_error or contract_error
        if error_text is None and isinstance(payload, Mapping):
            error_text = str(payload.get("error") or f"mtdata exited with {returncode}")
        error_text = error_text or f"mtdata exited with {returncode}"
        command_record["error"] = _redact_text(error_text, known_secrets)
        _record_automatic_issue(context, stage, spec, command_record["error"], raw_path)
    else:
        command_record.pop("error", None)
    _save_manifest(context)
    return success


_SUMMARY_KEYS = {
    "analysis_time_window",
    "anchor_tests_failed",
    "anchor_tests_planned",
    "anchor_tests_succeeded",
    "available",
    "backtest_plan",
    "best_method",
    "best_params",
    "best_score",
    "best_trial",
    "best_value",
    "candidate",
    "candidates",
    "cost_assumptions",
    "complete_methods",
    "complete_success",
    "data_quality",
    "denoise_used",
    "detail",
    "error",
    "error_code",
    "failed_tests",
    "failed_methods",
    "forecast_price",
    "forecast_return",
    "forecast_time",
    "horizon",
    "history_quality",
    "method",
    "methods",
    "methods_complete",
    "methods_failed",
    "methods_partial",
    "methods_succeeded",
    "methods_total",
    "pagination",
    "partial_methods",
    "quantity",
    "ranked_methods",
    "ranking",
    "results",
    "study_name",
    "status",
    "success",
    "symbol",
    "timeframe",
    "units",
    "warnings",
}


def _summarize_payload(payload: Any) -> Any:
    if not isinstance(payload, Mapping):
        return payload
    summary: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "results" and isinstance(value, Mapping):
            summary["results"] = {
                str(method): (
                    {str(field): item for field, item in result.items() if field != "details"}
                    if isinstance(result, Mapping)
                    else result
                )
                for method, result in value.items()
            }
            continue
        if key in _SUMMARY_KEYS or isinstance(value, (str, int, float, bool)) or value is None:
            summary[str(key)] = value
    anchor_status = _backtest_anchor_status(payload)
    if anchor_status is not None:
        summary["derived_anchor_status"] = anchor_status
    return summary


def write_normalized_stage(context: RunContext, stage: str) -> dict[str, Any]:
    stage_record = context.manifest["stages"][stage]
    rows: list[dict[str, Any]] = []
    for command_id, record in stage_record.get("commands", {}).items():
        row: dict[str, Any] = {
            "command_id": command_id,
            "status": record.get("status"),
            "argv": record.get("argv"),
            "metadata": record.get("metadata", {}),
            "attempts": record.get("attempts", 0),
            "raw_path": record.get("raw_path"),
        }
        raw_path_text = record.get("raw_path")
        if raw_path_text:
            raw_path = context.run_dir / str(raw_path_text)
            if raw_path.exists():
                envelope = _load_json(raw_path)
                if isinstance(envelope, Mapping):
                    row["returncode"] = envelope.get("returncode")
                    row["duration_seconds"] = envelope.get("duration_seconds")
                    if envelope.get("parse_error"):
                        row["parse_error"] = envelope.get("parse_error")
                    if envelope.get("contract_error"):
                        row["contract_error"] = envelope.get("contract_error")
                    row["result"] = _summarize_payload(envelope.get("payload"))
        rows.append(row)
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    normalized = {
        "schema_version": SCHEMA_VERSION,
        "run_id": context.manifest.get("run_id"),
        "stage": stage,
        "generated_at": utc_now(),
        "status": stage_record.get("status"),
        "config_hash": stage_record.get("config_hash"),
        "plan_digest": stage_record.get("plan_digest"),
        "counts": counts,
        "commands": rows,
    }
    output_path = context.run_dir / "normalized" / f"{stage}.json"
    _atomic_write_json(output_path, normalized)
    stage_record["normalized_path"] = _relative(output_path, context.run_dir)
    return normalized


def _run_command_stage(
    context: RunContext,
    stage: str,
    specs: Sequence[CommandSpec],
    *,
    extra_hash_input: Any = None,
    allow_append: bool = False,
    continuous: bool = False,
) -> int:
    _register_specs(
        context,
        stage,
        specs,
        extra_hash_input=extra_hash_input,
        allow_append=allow_append,
    )
    stage_record = context.manifest["stages"][stage]
    stage_record["attempts"] = int(stage_record.get("attempts", 0)) + 1
    stage_record.setdefault("started_at", utc_now())
    stage_record["status"] = "planned" if context.dry_run else "running"
    if continuous:
        stage_record["lifecycle"] = "append_only"
    _save_manifest(context)

    pending = [
        spec
        for spec in specs
        if stage_record["commands"][spec.command_id].get("status") != "completed"
    ]
    selected = pending[: context.max_commands] if context.max_commands is not None else pending
    failures = 0
    for spec in selected:
        outcome = execute_spec(context, stage, spec)
        if outcome is False:
            failures += 1
            if context.fail_fast:
                break

    statuses = {
        command_id: record.get("status")
        for command_id, record in stage_record.get("commands", {}).items()
    }
    if context.dry_run:
        stage_record["status"] = "planned"
    elif all(status == "completed" for status in statuses.values()):
        stage_record["status"] = "active" if continuous else "completed"
        stage_record["last_observation_at" if continuous else "completed_at"] = utc_now()
    elif any(status == "failed" for status in statuses.values()):
        stage_record["status"] = "failed"
    else:
        stage_record["status"] = "partial"
    write_normalized_stage(context, stage)
    _save_manifest(context)
    if failures:
        return 1
    return 0


def _require_stage(context: RunContext, stage: str) -> None:
    status = context.manifest.get("stages", {}).get(stage, {}).get("status")
    if status != "completed":
        raise HarnessError(f"Stage {stage!r} must be completed first (current status: {status})")


def _require_holdout_unopened(context: RunContext, action: str) -> None:
    if (context.run_dir / "holdout_lock.json").exists():
        raise HarnessError(
            f"Cannot {action} after the locked holdout was opened; this study is sealed against reselection"
        )


def run_audit(context: RunContext) -> int:
    return _run_command_stage(context, "audit", build_audit_specs(context.config))


def run_screen(context: RunContext) -> int:
    if not context.dry_run:
        _require_stage(context, "audit")
    return _run_command_stage(context, "screen", build_screen_specs(context.config))


def run_tune(context: RunContext) -> int:
    if not context.dry_run:
        _require_holdout_unopened(context, "tune or reselect candidates")
        _require_stage(context, "screen")
    specs = build_tune_specs(context.config, context.run_dir)
    if not specs:
        raise HarnessError(
            "No tune experiments are configured. Add tune.experiments to a config revision after reviewing screen results."
        )
    return _run_command_stage(context, "tune", specs)


def _validation_candidate(
    context: RunContext,
    candidate_file: Path | None,
) -> tuple[dict[str, Any], str]:
    artifact_path = context.run_dir / "validation_candidate.json"
    if artifact_path.exists():
        artifact = _load_json(artifact_path)
        if not isinstance(artifact, Mapping):
            raise HarnessError("Validation candidate artifact is invalid")
        candidate = _validate_frozen_candidate(artifact.get("candidate"))
        candidate_hash = _sha256_json(redact(candidate))
        if artifact.get("candidate_hash") != candidate_hash:
            raise HarnessError("Validation candidate hash verification failed")
        if candidate_file is not None:
            proposed = _validate_frozen_candidate(_load_json(candidate_file))
            if _sha256_json(redact(proposed)) != candidate_hash:
                raise HarnessError("Validation candidate is locked; start a new study to test a different candidate")
        return candidate, candidate_hash
    if candidate_file is None:
        raise HarnessError("validate requires --candidate-file on the first invocation")
    candidate = _validate_frozen_candidate(_load_json(candidate_file))
    safe_candidate = redact(candidate, known_secrets=_secret_values())
    candidate_hash = _sha256_json(safe_candidate)
    if not context.dry_run:
        _write_new_json(
            artifact_path,
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": context.manifest.get("run_id"),
                "registered_at": utc_now(),
                "immutable": True,
                "candidate_hash": candidate_hash,
                "candidate": safe_candidate,
                "source": candidate_file.name,
                "windows": list(context.config.get("validate", {}).get("windows", [])),
            },
        )
    return candidate, candidate_hash


def _write_validation_evidence(
    context: RunContext,
    candidate_hash: str,
    specs: Sequence[CommandSpec],
    *,
    stage: str = "validate",
    filename: str = "validation_evidence.json",
    windows: Sequence[str] = ("validation", "confirmation"),
) -> None:
    evidence_path = context.run_dir / filename
    raw_hashes: dict[str, str] = {}
    for spec in specs:
        raw_path = _raw_path(context, stage, spec.command_id)
        if not raw_path.exists():
            raise HarnessError(f"Validation evidence is missing {raw_path.name}")
        raw_hashes[spec.command_id] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    expected = {
        "schema_version": SCHEMA_VERSION,
        "run_id": context.manifest.get("run_id"),
        "immutable": True,
        "candidate_hash": candidate_hash,
        "windows": list(windows),
        "plan_digest": _plan_digest(specs),
        "raw_sha256": raw_hashes,
    }
    if evidence_path.exists():
        existing = _load_json(evidence_path)
        comparable = {key: existing.get(key) for key in expected} if isinstance(existing, Mapping) else None
        if comparable != expected:
            raise HarnessError("Immutable validation evidence no longer matches its raw results")
        return
    _write_new_json(evidence_path, {**expected, "completed_at": utc_now()})


def _numeric(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _sign(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def _terminal_moves(
    row: Mapping[str, Any],
    quantity: str,
) -> tuple[float, float, float, float] | None:
    forecast = row.get("forecast")
    actual = row.get("actual")
    if not isinstance(forecast, list) or not isinstance(actual, list) or not forecast or not actual:
        return None
    forecast_values = [_numeric(item) for item in forecast]
    actual_values = [_numeric(item) for item in actual]
    if any(item is None for item in forecast_values + actual_values):
        return None
    predicted = [float(item) for item in forecast_values if item is not None]
    realized = [float(item) for item in actual_values if item is not None]
    if quantity == "return":
        predicted_move = sum(predicted)
        actual_move = sum(realized)
        reference = _numeric(row.get("signal_reference_price"))
        entry = _numeric(row.get("entry_price"))
        if reference in (None, 0.0) or entry in (None, 0.0):
            return None
        predicted_return = pow(2.718281828459045, predicted_move) - 1.0
        terminal_price = float(reference) * pow(2.718281828459045, actual_move)
        gross_return = ((terminal_price - float(entry)) / float(entry)) * _sign(predicted_move)
    else:
        reference = _numeric(row.get("signal_reference_price"))
        entry = _numeric(row.get("entry_price"))
        if reference in (None, 0.0) or entry in (None, 0.0):
            return None
        predicted_move = predicted[-1] - reference
        actual_move = realized[-1] - reference
        predicted_return = predicted_move / float(reference)
        gross_return = ((realized[-1] - float(entry)) / float(entry)) * _sign(predicted_move)
    return predicted_move, actual_move, gross_return, predicted_return


def _empty_model_evidence() -> dict[str, Any]:
    return {
        "opportunities": 0,
        "signals": 0,
        "correct": 0,
        "total_tests": 0,
        "successful_tests": 0,
        "missing_detail_anchors": 0,
        "fixed_horizon_net_returns": [],
        "by_window": {},
    }


def _window_evidence(evidence: dict[str, Any], window: str) -> dict[str, Any]:
    return evidence["by_window"].setdefault(window, _empty_model_evidence() | {"by_window": {}})


def _record_terminal_evidence(
    evidence: dict[str, Any],
    *,
    window: str,
    row: Mapping[str, Any],
    quantity: str,
    round_trip_cost_fraction: float,
    trade_threshold: float,
) -> None:
    target = _window_evidence(evidence, window)
    moves = _terminal_moves(row, quantity)
    if moves is None:
        evidence["missing_detail_anchors"] += 1
        target["missing_detail_anchors"] += 1
        return
    predicted_move, actual_move, gross_return, predicted_return = moves
    actual_direction = _sign(actual_move)
    predicted_direction = _sign(predicted_move) if abs(predicted_return) > trade_threshold else 0
    if actual_direction == 0:
        return
    for bucket in (evidence, target):
        bucket["opportunities"] += 1
        if predicted_direction != 0:
            bucket["signals"] += 1
            bucket["correct"] += int(predicted_direction == actual_direction)
            bucket["fixed_horizon_net_returns"].append(gross_return - round_trip_cost_fraction)


def _wilson_lower(successes: int, total: int, z: float = 1.959963984540054) -> float | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1.0 + (z * z / total)
    centre = proportion + (z * z / (2.0 * total))
    margin = z * ((proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) ** 0.5)
    return (centre - margin) / denominator


def _summarize_model_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    signals = int(evidence.get("signals") or 0)
    opportunities = int(evidence.get("opportunities") or 0)
    correct = int(evidence.get("correct") or 0)
    returns = [float(item) for item in evidence.get("fixed_horizon_net_returns", [])]
    sorted_returns = sorted(returns)
    midpoint = len(sorted_returns) // 2
    if not sorted_returns:
        median_return = None
    elif len(sorted_returns) % 2:
        median_return = sorted_returns[midpoint]
    else:
        median_return = (sorted_returns[midpoint - 1] + sorted_returns[midpoint]) / 2.0
    positive_returns = sum(item > 0.0 for item in returns)
    return {
        "opportunities": opportunities,
        "signals": signals,
        "signal_coverage": signals / opportunities if opportunities else 0.0,
        "correct": correct,
        "directional_accuracy": correct / signals if signals else None,
        "wilson_95_lower": _wilson_lower(correct, signals),
        "mean_fixed_horizon_net_return": sum(returns) / len(returns) if returns else None,
        "median_fixed_horizon_net_return": median_return,
        "positive_fixed_horizon_returns": positive_returns,
        "positive_return_rate": positive_returns / len(returns) if returns else None,
        "positive_return_wilson_95_lower": _wilson_lower(positive_returns, len(returns)),
        "fixed_horizon_return_observations": len(returns),
        "total_tests": int(evidence.get("total_tests") or 0),
        "successful_tests": int(evidence.get("successful_tests") or 0),
        "missing_detail_anchors": int(evidence.get("missing_detail_anchors") or 0),
    }


def _collect_validation_evidence(
    context: RunContext,
    candidate: Mapping[str, Any],
    specs: Sequence[CommandSpec],
    *,
    stage: str = "validate",
) -> tuple[dict[str, Any], dict[str, Any], set[str], list[str]]:
    candidate_method = str(candidate["method"])
    baseline_method = str(candidate["provenance"]["baseline"]["method"])
    evidence_by_method = {
        candidate_method: _empty_model_evidence(),
        baseline_method: _empty_model_evidence(),
    }
    costs = candidate["costs"]
    round_trip_cost_fraction = (
        2.0 * float(costs["slippage_bps"])
        + float(costs["spread_bps"])
        + 2.0 * float(costs["commission_bps_per_side"])
    ) / 10000.0
    trade_threshold = float(costs["trade_threshold"])
    windows_seen: set[str] = set()
    missing_results: list[str] = []
    for spec in specs:
        window = str(spec.metadata.get("window"))
        raw = _load_json(_raw_path(context, stage, spec.command_id))
        payload = raw.get("payload") if isinstance(raw, Mapping) else None
        results = payload.get("results") if isinstance(payload, Mapping) else None
        for method, evidence in evidence_by_method.items():
            result = results.get(method) if isinstance(results, Mapping) else None
            if not isinstance(result, Mapping) or result.get("success") is not True:
                missing_results.append(f"{spec.command_id}:{method}")
                continue
            windows_seen.add(window)
            shard_tests = int(result.get("num_tests") or 0)
            shard_successes = int(result.get("successful_tests") or 0)
            evidence["total_tests"] += shard_tests
            evidence["successful_tests"] += shard_successes
            window_bucket = _window_evidence(evidence, window)
            window_bucket["total_tests"] += shard_tests
            window_bucket["successful_tests"] += shard_successes
            details = result.get("details")
            if not isinstance(details, list):
                evidence["missing_detail_anchors"] += shard_successes
                window_bucket["missing_detail_anchors"] += shard_successes
                continue
            successful_detail_rows = 0
            for row in details:
                if isinstance(row, Mapping) and row.get("success") is not False:
                    successful_detail_rows += 1
                    _record_terminal_evidence(
                        evidence,
                        window=window,
                        row=row,
                        quantity=str(candidate["quantity"]),
                        round_trip_cost_fraction=round_trip_cost_fraction,
                        trade_threshold=trade_threshold,
                    )
            omitted_successes = max(0, shard_successes - successful_detail_rows)
            evidence["missing_detail_anchors"] += omitted_successes
            window_bucket["missing_detail_anchors"] += omitted_successes
    return evidence_by_method[candidate_method], evidence_by_method[baseline_method], windows_seen, missing_results


def _compute_validation_decision(
    context: RunContext,
    candidate: Mapping[str, Any],
    candidate_hash: str,
    specs: Sequence[CommandSpec],
    *,
    stage: str = "validate",
    required_windows: set[str] | None = None,
    minimum_opportunities: int = 100,
    minimum_signals: int = 50,
) -> dict[str, Any]:
    candidate_evidence, baseline_evidence, windows_seen, missing_results = _collect_validation_evidence(
        context,
        candidate,
        specs,
        stage=stage,
    )
    candidate_summary = _summarize_model_evidence(candidate_evidence)
    baseline_summary = _summarize_model_evidence(baseline_evidence)
    expected_windows = required_windows or {"validation", "confirmation"}
    failed_anchors = max(
        0,
        candidate_summary["total_tests"] - candidate_summary["successful_tests"],
    )
    baseline_failed_anchors = max(
        0,
        baseline_summary["total_tests"] - baseline_summary["successful_tests"],
    )
    candidate_windows = set(candidate_evidence.get("by_window", {}))
    baseline_windows = set(baseline_evidence.get("by_window", {}))
    evidence_checks = {
        "minimum_opportunities": candidate_summary["opportunities"] >= minimum_opportunities,
        "minimum_signals": candidate_summary["signals"] >= minimum_signals,
        "signal_coverage_at_least_10pct": candidate_summary["signal_coverage"] >= 0.10,
        "no_failed_anchors": failed_anchors == 0 and not missing_results,
        "all_windows_represented": expected_windows.issubset(candidate_windows),
        "terminal_paths_available": candidate_summary["missing_detail_anchors"] == 0,
        "baseline_all_windows_represented": expected_windows.issubset(baseline_windows),
        "baseline_no_failed_anchors": baseline_failed_anchors == 0,
        "baseline_terminal_paths_available": baseline_summary["missing_detail_anchors"] == 0,
        "baseline_same_opportunities": (
            baseline_summary["opportunities"] == candidate_summary["opportunities"]
        ),
    }
    evidence_status = "pass" if all(evidence_checks.values()) else "fail"
    minimum_improvement = float(candidate["provenance"]["baseline"]["minimum_accuracy_improvement"])
    candidate_accuracy = candidate_summary["directional_accuracy"]
    baseline_accuracy = baseline_summary["directional_accuracy"]
    wilson_lower = candidate_summary["wilson_95_lower"]
    if candidate_accuracy is None or baseline_accuracy is None or wilson_lower is None:
        statistical_status = "not_yet_available"
    else:
        statistical_status = (
            "pass"
            if wilson_lower > 0.50
            and wilson_lower - baseline_accuracy >= minimum_improvement
            else "fail"
        )
    per_window: dict[str, Any] = {}
    for window in sorted(expected_windows):
        candidate_window = _summarize_model_evidence(candidate_evidence.get("by_window", {}).get(window, {}))
        baseline_window = _summarize_model_evidence(baseline_evidence.get("by_window", {}).get(window, {}))
        accuracy = candidate_window["directional_accuracy"]
        baseline_window_accuracy = baseline_window["directional_accuracy"]
        net_return = candidate_window["mean_fixed_horizon_net_return"]
        median_return = candidate_window["median_fixed_horizon_net_return"]
        positive_return_lower = candidate_window["positive_return_wilson_95_lower"]
        window_wilson_lower = candidate_window["wilson_95_lower"]
        consistent = (
            accuracy is not None
            and baseline_window_accuracy is not None
            and window_wilson_lower is not None
            and window_wilson_lower > 0.50
            and window_wilson_lower - baseline_window_accuracy >= minimum_improvement
            and net_return is not None
            and net_return > 0.0
            and median_return is not None
            and median_return > 0.0
            and positive_return_lower is not None
            and positive_return_lower > 0.50
        )
        per_window[window] = {
            "candidate": candidate_window,
            "baseline": baseline_window,
            "consistent": consistent,
        }
    stability_status = "pass" if per_window and all(item["consistent"] for item in per_window.values()) else "fail"
    mean_net_return = candidate_summary["mean_fixed_horizon_net_return"]
    median_net_return = candidate_summary["median_fixed_horizon_net_return"]
    positive_return_lower = candidate_summary["positive_return_wilson_95_lower"]
    if mean_net_return is None or median_net_return is None or positive_return_lower is None:
        economic_status = "not_yet_available"
    else:
        economic_status = (
            "pass"
            if mean_net_return > 0.0
            and median_net_return > 0.0
            and positive_return_lower > 0.50
            else "fail"
        )
    gates = {
        "minimum_evidence": {"status": evidence_status, "checks": evidence_checks},
        "statistical": {
            "status": statistical_status,
            "test": "candidate Wilson 95% lower bound > 0.50 and preregistered improvement over baseline",
        },
        "stability": {"status": stability_status, "per_window": per_window},
        "economic": {
            "status": economic_status,
            "metric": (
                "positive mean and median fixed-horizon return net of preregistered costs, "
                "with Wilson 95% lower bound for positive-return rate above 0.50"
            ),
        },
        "interval": {
            "status": "not_yet_available",
            "max_mean_relative_width": candidate["calibration"]["max_mean_relative_width"],
            "relative_width_definition": RELATIVE_WIDTH_DEFINITION,
            "note": "forecast_backtest_run does not expose pipeline-matched interval coverage/width; a separately hashed review artifact is required.",
        },
    }
    computed_gates_pass = all(
        gates[name]["status"] == "pass"
        for name in ("minimum_evidence", "statistical", "stability", "economic")
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": context.manifest.get("run_id"),
        "immutable": True,
        "candidate_hash": candidate_hash,
        "assessed_at": utc_now(),
        "stage": stage,
        "candidate_spec": {
            "method": candidate["method"],
            "timeframe": candidate["timeframe"],
            "horizon": candidate["horizon"],
            "quantity": candidate["quantity"],
            "lookback": candidate["lookback"],
            "pipeline_hash": candidate_hash,
            "calibration": candidate["calibration"],
        },
        "candidate": candidate_summary,
        "baseline": {
            "method": candidate["provenance"]["baseline"]["method"],
            **baseline_summary,
        },
        "windows": sorted(candidate_windows),
        "missing_results": missing_results,
        "thresholds": {
            "minimum_opportunities": minimum_opportunities,
            "minimum_signals": minimum_signals,
            "minimum_signal_coverage": 0.10,
        },
        "gates": gates,
        "computed_gates_pass": computed_gates_pass,
        "review_required": True,
        "eligible_for_holdout": False,
    }


def _write_assessment(path: Path, assessment: Mapping[str, Any]) -> dict[str, Any]:
    if path.exists():
        existing = _load_json(path)
        if not isinstance(existing, Mapping) or existing.get("candidate_hash") != assessment.get("candidate_hash"):
            raise HarnessError(f"Immutable assessment does not match its candidate: {path}")
        return dict(existing)
    _write_new_json(path, assessment)
    return dict(assessment)


def _assessment_with_evidence(
    context: RunContext,
    assessment: Mapping[str, Any],
    evidence_name: str,
) -> dict[str, Any]:
    evidence_path = context.run_dir / evidence_name
    return {
        **dict(assessment),
        "evidence": evidence_name,
        "evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
    }


def _validate_interval_candidate_identity(
    assessment: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> Mapping[str, Any]:
    if artifact.get("candidate_hash") != assessment.get("candidate_hash"):
        raise HarnessError("Interval evidence candidate_hash does not match the assessment")
    candidate_spec = assessment.get("candidate_spec")
    if not isinstance(candidate_spec, Mapping):
        raise HarnessError("Assessment candidate specification is missing")
    if artifact.get("pipeline_hash") != candidate_spec.get("pipeline_hash"):
        raise HarnessError("Interval evidence does not match the frozen pipeline hash")
    for key in ("timeframe", "horizon", "quantity"):
        if artifact.get(key) != candidate_spec.get(key):
            raise HarnessError(f"Interval evidence {key} does not match the assessed candidate")
    if candidate_spec.get("quantity") != "price":
        raise HarnessError(
            "Confidence approval currently supports quantity=price only; native conformal/shadow "
            "cannot reproduce return or volatility candidates"
        )
    return candidate_spec


def _validated_coverage_row(
    row: Mapping[str, Any],
    *,
    label: str,
    max_mean_relative_width: float,
) -> dict[str, Any]:
    sample_number = _numeric(row.get("sample_count"))
    covered_number = _numeric(row.get("covered_count"))
    if (
        sample_number is None
        or covered_number is None
        or sample_number < 1
        or int(sample_number) != sample_number
        or int(covered_number) != covered_number
        or not 0 <= covered_number <= sample_number
    ):
        raise HarnessError(f"{label} sample_count/covered_count must be consistent non-negative integers")
    sample_count = int(sample_number)
    covered_count = int(covered_number)
    empirical_coverage = _numeric(row.get("empirical_coverage"))
    mean_relative_width = _numeric(row.get("mean_relative_width"))
    expected_coverage = covered_count / sample_count
    if empirical_coverage is None or abs(empirical_coverage - expected_coverage) > 1e-12:
        raise HarnessError(f"{label} empirical_coverage does not equal covered_count/sample_count")
    if mean_relative_width is None or mean_relative_width <= 0.0:
        raise HarnessError(f"{label} mean_relative_width must be positive")
    if mean_relative_width > max_mean_relative_width:
        raise HarnessError(f"{label} mean_relative_width exceeds the preregistered usability ceiling")
    coverage_wilson_95_lower = _wilson_lower(covered_count, sample_count)
    if coverage_wilson_95_lower is None or coverage_wilson_95_lower < 0.80:
        raise HarnessError(f"{label} Wilson 95% coverage lower bound is below 0.80")
    return {
        "sample_count": sample_count,
        "covered_count": covered_count,
        "empirical_coverage": empirical_coverage,
        "mean_relative_width": mean_relative_width,
        "coverage_wilson_95_lower": coverage_wilson_95_lower,
    }


def _validated_interval_sampling(
    assessment: Mapping[str, Any],
    artifact: Mapping[str, Any],
    interval: Mapping[str, Any],
    *,
    max_mean_relative_width: float,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = _validated_coverage_row(
        artifact,
        label="Aggregate interval evidence",
        max_mean_relative_width=max_mean_relative_width,
    )
    if (
        interval.get("sample_count") != artifact.get("sample_count")
        or interval.get("covered_count") != artifact.get("covered_count")
        or interval.get("by_window") != artifact.get("by_window")
    ):
        raise HarnessError("Review interval counts/windows do not match the immutable evidence artifact")
    required_windows = set(assessment.get("windows") or [])
    artifact_windows = set(artifact.get("windows") or [])
    by_window = artifact.get("by_window")
    if not isinstance(by_window, Mapping) or set(by_window) != required_windows:
        raise HarnessError("Interval evidence must contain exact per-window metrics for every assessed window")
    if artifact_windows != required_windows:
        raise HarnessError("Interval evidence windows must exactly match the assessed out-of-sample windows")
    normalized_windows: dict[str, dict[str, Any]] = {}
    for window in sorted(required_windows):
        row = by_window.get(window)
        if not isinstance(row, Mapping):
            raise HarnessError(f"Interval evidence window {window!r} is invalid")
        normalized_windows[window] = _validated_coverage_row(
            row,
            label=f"Interval evidence window {window!r}",
            max_mean_relative_width=max_mean_relative_width,
        )
    sample_total = sum(row["sample_count"] for row in normalized_windows.values())
    covered_total = sum(row["covered_count"] for row in normalized_windows.values())
    weighted_width = sum(
        row["mean_relative_width"] * row["sample_count"]
        for row in normalized_windows.values()
    ) / sample_total
    if root["sample_count"] != sample_total or root["covered_count"] != covered_total:
        raise HarnessError("Aggregate interval counts do not equal the per-window totals")
    if abs(root["mean_relative_width"] - weighted_width) > 1e-12:
        raise HarnessError("Aggregate mean_relative_width is not the sample-weighted per-window mean")
    minimum_samples = int(assessment.get("thresholds", {}).get("minimum_opportunities", 100))
    if root["sample_count"] < minimum_samples:
        raise HarnessError(f"Interval evidence requires at least {minimum_samples} out-of-sample observations")
    return root, normalized_windows


def _check_self_reported_interval_summary(
    assessment: Mapping[str, Any],
    review_file: Path,
    interval: Any,
) -> tuple[
    dict[str, Any],
    float,
    float,
    float,
    float,
    float,
    dict[str, dict[str, Any]],
    str,
]:
    if not isinstance(interval, Mapping) or interval.get("status") != "pass":
        raise HarnessError("Review requires passing pipeline-matched interval_evidence")
    artifact_name = str(interval.get("artifact") or "")
    if not artifact_name:
        raise HarnessError(
            "interval_evidence.artifact must name an out-of-sample coverage/width JSON artifact"
        )
    artifact_source = Path(artifact_name)
    if not artifact_source.is_absolute():
        artifact_source = review_file.parent / artifact_source
    interval_artifact = _load_json(artifact_source)
    if not isinstance(interval_artifact, Mapping):
        raise HarnessError("Interval evidence artifact must be a JSON object")
    artifact = dict(interval_artifact)
    artifact_bytes_sha256 = hashlib.sha256(artifact_source.read_bytes()).hexdigest()
    empirical_coverage = _numeric(interval.get("empirical_coverage"))
    mean_width = _numeric(interval.get("mean_width"))
    mean_relative_width = _numeric(interval.get("mean_relative_width"))
    evidence_sha256 = str(interval.get("evidence_sha256") or "")
    if empirical_coverage is None or not 0.0 <= empirical_coverage <= 1.0:
        raise HarnessError("interval_evidence.empirical_coverage must be between 0 and 1")
    if mean_width is None or mean_width <= 0.0:
        raise HarnessError("interval_evidence.mean_width must be positive")
    if mean_relative_width is None or mean_relative_width <= 0.0:
        raise HarnessError("interval_evidence.mean_relative_width must be positive")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", evidence_sha256):
        raise HarnessError("interval_evidence.evidence_sha256 must be a SHA-256 hex digest")
    if evidence_sha256.lower() != artifact_bytes_sha256:
        raise HarnessError("interval_evidence.evidence_sha256 does not match its artifact bytes")
    candidate_spec = _validate_interval_candidate_identity(assessment, artifact)
    if artifact.get("source") != "mtdata_cli_oos_replay" or artifact.get("out_of_sample") is not True:
        raise HarnessError("Interval evidence must be an out-of-sample mtdata CLI replay")
    if artifact.get("causal_actuals_after_forecast") is not True:
        raise HarnessError("Interval replay must prove actuals were observed after each forecast cutoff")
    if artifact.get("forecast_tool") != "forecast_conformal_intervals":
        raise HarnessError("Interval evidence must replay forecast_conformal_intervals")
    if artifact.get("actuals_tool") != "data_fetch_candles":
        raise HarnessError("Interval evidence actuals must come from data_fetch_candles")
    nominal_coverage = _numeric(artifact.get("nominal_coverage"))
    if nominal_coverage is None or abs(nominal_coverage - 0.90) > 1e-12:
        raise HarnessError("Interval evidence must assess the preregistered nominal 90% band")
    expected_width_unit = {
        "price": "price",
        "return": "log_return",
        "volatility": "volatility",
    }.get(str(candidate_spec.get("quantity")))
    if artifact.get("width_unit") != expected_width_unit:
        raise HarnessError("Interval evidence width_unit does not match the forecast quantity")
    if artifact.get("relative_width_definition") != RELATIVE_WIDTH_DEFINITION:
        raise HarnessError("Interval evidence must use the preregistered relative-width definition")
    artifact_coverage = _numeric(artifact.get("empirical_coverage"))
    artifact_width = _numeric(artifact.get("mean_width"))
    artifact_relative_width = _numeric(artifact.get("mean_relative_width"))
    if (
        artifact_coverage != empirical_coverage
        or artifact_width != mean_width
        or artifact_relative_width != mean_relative_width
    ):
        raise HarnessError("Review interval metrics do not match the immutable evidence artifact")
    calibration = candidate_spec.get("calibration")
    max_mean_relative_width = (
        _numeric(calibration.get("max_mean_relative_width"))
        if isinstance(calibration, Mapping)
        else None
    )
    if max_mean_relative_width is None or max_mean_relative_width <= 0.0:
        raise HarnessError("Assessment lacks a valid preregistered relative-width ceiling")
    if mean_relative_width > max_mean_relative_width:
        raise HarnessError(
            "Empirical mean_relative_width exceeds the candidate's preregistered usability ceiling"
        )
    aggregate, by_window = _validated_interval_sampling(
        assessment,
        artifact,
        interval,
        max_mean_relative_width=max_mean_relative_width,
    )
    raw_hashes = artifact.get("raw_sha256")
    if not isinstance(raw_hashes, Mapping) or not raw_hashes:
        raise HarnessError("Interval evidence must retain hashes of its mtdata CLI replay outputs")
    if not all(re.fullmatch(r"[0-9a-fA-F]{64}", str(value)) for value in raw_hashes.values()):
        raise HarnessError("Every interval replay raw_sha256 value must be a SHA-256 hex digest")
    if abs(empirical_coverage - 0.90) > 0.10:
        raise HarnessError("Empirical interval coverage must remain within 10 percentage points of nominal 90%")
    return (
        artifact,
        empirical_coverage,
        mean_width,
        mean_relative_width,
        max_mean_relative_width,
        aggregate["coverage_wilson_95_lower"],
        by_window,
        artifact_bytes_sha256,
    )


def _load_review_decision_with_hash_checks(
    context: RunContext,
    *,
    assessment_path: Path,
    decision_path: Path,
    approval_field: str,
) -> dict[str, Any]:
    _require_interval_approval_enabled()
    assessment = _load_json(assessment_path)
    decision = _load_json(decision_path)
    if not isinstance(assessment, Mapping) or not isinstance(decision, Mapping):
        raise HarnessError("Immutable assessment/review decision is invalid")
    assessment_sha256 = hashlib.sha256(assessment_path.read_bytes()).hexdigest()
    if decision.get("assessment_sha256") != assessment_sha256:
        raise HarnessError("Immutable review decision assessment hash verification failed")
    if (
        assessment.get("computed_gates_pass") is not True
        or decision.get("candidate_hash") != assessment.get("candidate_hash")
        or decision.get("computed_gates") != assessment.get("gates")
        or decision.get(approval_field) is not True
    ):
        raise HarnessError("Immutable review decision does not preserve passing gates and candidate identity")
    evidence_text = str(assessment.get("evidence") or "")
    evidence_path = (context.run_dir / evidence_text).resolve()
    try:
        evidence_path.relative_to(context.run_dir.resolve())
    except ValueError as exc:
        raise HarnessError("Assessment evidence must remain inside the study directory") from exc
    if not evidence_path.is_file():
        raise HarnessError("Assessment command-evidence artifact is missing")
    if hashlib.sha256(evidence_path.read_bytes()).hexdigest() != assessment.get("evidence_sha256"):
        raise HarnessError("Assessment command-evidence hash verification failed")
    evidence = _load_json(evidence_path)
    raw_hashes = evidence.get("raw_sha256") if isinstance(evidence, Mapping) else None
    if not isinstance(raw_hashes, Mapping) or evidence.get("candidate_hash") != assessment.get("candidate_hash"):
        raise HarnessError("Assessment command evidence is invalid")
    for command_id, expected_sha256 in raw_hashes.items():
        raw_path = _raw_path(context, str(assessment.get("stage")), str(command_id))
        if not raw_path.is_file():
            raise HarnessError("Assessment raw mtdata evidence is missing")
        if hashlib.sha256(raw_path.read_bytes()).hexdigest() != expected_sha256:
            raise HarnessError("Assessment raw mtdata evidence hash verification failed")
    interval_gate = decision.get("interval_gate")
    if not isinstance(interval_gate, Mapping) or interval_gate.get("status") != "pass":
        raise HarnessError("Immutable review decision has no passing interval gate")
    artifact_text = str(interval_gate.get("artifact") or "")
    artifact_path = (context.run_dir / artifact_text).resolve()
    try:
        artifact_path.relative_to(context.run_dir.resolve())
    except ValueError as exc:
        raise HarnessError("Reviewed interval artifact must remain inside the study directory") from exc
    if not artifact_path.is_file():
        raise HarnessError("Copied interval evidence artifact is missing")
    copied_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if copied_sha256 != interval_gate.get("copied_artifact_sha256"):
        raise HarnessError("Copied interval evidence hash verification failed")
    return dict(decision)


def _review_decision(
    context: RunContext,
    *,
    assessment_path: Path,
    decision_path: Path,
    review_file: Path | None,
    approval_field: str,
) -> dict[str, Any] | None:
    assessment = _load_json(assessment_path)
    if not isinstance(assessment, Mapping):
        raise HarnessError("Assessment artifact is invalid")
    if assessment.get("computed_gates_pass") is not True:
        return None
    if decision_path.exists():
        _require_interval_approval_enabled()
        return _load_review_decision_with_hash_checks(
            context,
            assessment_path=assessment_path,
            decision_path=decision_path,
            approval_field=approval_field,
        )
    if review_file is None:
        return None
    _require_interval_approval_enabled()
    review = _load_json(review_file)
    if not isinstance(review, Mapping):
        raise HarnessError("Review artifact must be a JSON object")
    assessment_sha256 = hashlib.sha256(assessment_path.read_bytes()).hexdigest()
    if review.get("candidate_hash") != assessment.get("candidate_hash"):
        raise HarnessError("Review candidate_hash does not match the assessment")
    if review.get("assessment_sha256") != assessment_sha256:
        raise HarnessError("Review assessment_sha256 does not match the immutable assessment")
    if not str(review.get("reviewer") or "").strip():
        raise HarnessError("Review artifact must identify a reviewer")
    interval = review.get("interval_evidence")
    (
        interval_artifact,
        empirical_coverage,
        mean_width,
        mean_relative_width,
        max_mean_relative_width,
        coverage_wilson_95_lower,
        interval_by_window,
        artifact_bytes_sha256,
    ) = _check_self_reported_interval_summary(assessment, review_file, interval)
    evidence_sha256 = artifact_bytes_sha256
    if review.get(approval_field) is not True:
        raise HarnessError(f"Review must explicitly set {approval_field}=true")
    safe_review = redact(dict(review), known_secrets=_secret_values())
    copied_interval_path = context.run_dir / decision_path.name.replace("decision", "interval_evidence")
    if copied_interval_path.exists():
        existing_interval = _load_json(copied_interval_path)
        if _sha256_json(existing_interval) != _sha256_json(redact(interval_artifact)):
            raise HarnessError("Immutable copied interval evidence differs from the reviewed artifact")
    else:
        _write_new_json(copied_interval_path, redact(interval_artifact, known_secrets=_secret_values()))
    copied_artifact_sha256 = hashlib.sha256(copied_interval_path.read_bytes()).hexdigest()
    decision = {
        "schema_version": SCHEMA_VERSION,
        "run_id": context.manifest.get("run_id"),
        "immutable": True,
        "candidate_hash": assessment.get("candidate_hash"),
        "assessment": _relative(assessment_path, context.run_dir),
        "assessment_sha256": assessment_sha256,
        "review": safe_review,
        "review_sha256": _sha256_json(safe_review),
        "computed_gates": assessment.get("gates"),
        "interval_gate": {
            "status": "pass",
            "sample_count": int(interval_artifact["sample_count"]),
            "covered_count": int(interval_artifact["covered_count"]),
            "empirical_coverage": empirical_coverage,
            "mean_width": mean_width,
            "mean_relative_width": mean_relative_width,
            "max_mean_relative_width": max_mean_relative_width,
            "relative_width_definition": RELATIVE_WIDTH_DEFINITION,
            "coverage_wilson_95_lower": coverage_wilson_95_lower,
            "by_window": interval_by_window,
            "evidence_sha256": evidence_sha256.lower(),
            "artifact": _relative(copied_interval_path, context.run_dir),
            "source_artifact_sha256": artifact_bytes_sha256,
            "copied_artifact_sha256": copied_artifact_sha256,
        },
        approval_field: True,
        "decided_at": utc_now(),
    }
    _write_new_json(decision_path, decision)
    return decision


def run_validate(
    context: RunContext,
    candidate_file: Path | None,
    review_file: Path | None = None,
) -> int:
    if not context.dry_run:
        _require_holdout_unopened(context, "validate a candidate")
        _require_stage(context, "screen")
    candidate, candidate_hash = _validation_candidate(context, candidate_file)
    specs = build_validation_specs(context.config, candidate)
    result = _run_command_stage(
        context,
        "validate",
        specs,
        extra_hash_input=candidate_hash,
    )
    if not context.dry_run and context.manifest["stages"]["validate"].get("status") == "completed":
        _write_validation_evidence(context, candidate_hash, specs)
        assessment_path = context.run_dir / "validation_assessment.json"
        assessment = _write_assessment(
            assessment_path,
            _assessment_with_evidence(
                context,
                _compute_validation_decision(context, candidate, candidate_hash, specs),
                "validation_evidence.json",
            ),
        )
        try:
            decision = _review_decision(
                context,
                assessment_path=assessment_path,
                decision_path=context.run_dir / "validation_decision.json",
                review_file=review_file,
                approval_field="eligible_for_holdout",
            )
        except IntervalApprovalDisabledError:
            context.manifest["stages"]["validate"].update(
                {
                    "status": "awaiting_interval_verifier",
                    "candidate_hash": candidate_hash,
                    "evidence": "validation_evidence.json",
                    "assessment": "validation_assessment.json",
                    "decision": None,
                    "eligible_for_holdout": False,
                    "approval_blocker": INTERVAL_APPROVAL_DISABLED_REASON,
                }
            )
            _save_manifest(context)
            raise
        eligible = bool(decision and decision.get("eligible_for_holdout") is True)
        if assessment.get("computed_gates_pass") is not True:
            status = "rejected"
        elif not eligible:
            status = "awaiting_interval_verifier"
        else:
            status = "completed"
        context.manifest["stages"]["validate"].update(
            {
                "status": status,
                "candidate_hash": candidate_hash,
                "evidence": "validation_evidence.json",
                "assessment": "validation_assessment.json",
                "decision": "validation_decision.json" if decision else None,
                "eligible_for_holdout": eligible,
                "approval_blocker": (
                    None if INTERVAL_APPROVAL_ENABLED else INTERVAL_APPROVAL_DISABLED_REASON
                ),
            }
        )
        _save_manifest(context)
        if not eligible:
            return 1
    return result


def freeze_candidate(context: RunContext, candidate_file: Path | None) -> int:
    if not context.dry_run:
        _require_interval_approval_enabled()
        _require_stage(context, "validate")
    validation_path = context.run_dir / "validation_candidate.json"
    evidence_path = context.run_dir / "validation_evidence.json"
    assessment_path = context.run_dir / "validation_assessment.json"
    decision_path = context.run_dir / "validation_decision.json"
    if not context.dry_run and (
        not validation_path.exists()
        or not evidence_path.exists()
        or not assessment_path.exists()
        or not decision_path.exists()
    ):
        raise HarnessError("Freeze requires immutable validation, confirmation, and eligibility evidence")
    validated_candidate: dict[str, Any] | None = None
    validated_hash: str | None = None
    if validation_path.exists():
        validation_artifact = _load_json(validation_path)
        if not isinstance(validation_artifact, Mapping):
            raise HarnessError("Validation candidate artifact is invalid")
        validated_candidate = _validate_frozen_candidate(validation_artifact.get("candidate"))
        validated_hash = _sha256_json(redact(validated_candidate))
        if validation_artifact.get("candidate_hash") != validated_hash:
            raise HarnessError("Validation candidate hash verification failed")
    if decision_path.exists():
        validation_specs = build_validation_specs(context.config, validated_candidate or {})
        validation_evidence = _load_json(evidence_path)
        if (
            not isinstance(validation_evidence, Mapping)
            or validation_evidence.get("plan_digest") != _plan_digest(validation_specs)
            or context.manifest["stages"]["validate"].get("config_hash")
            != _stage_config_hash("validate", context.config, validated_hash)
        ):
            raise HarnessError("Validation plan/config no longer matches its immutable evidence")
        decision = _load_review_decision_with_hash_checks(
            context,
            assessment_path=assessment_path,
            decision_path=decision_path,
            approval_field="eligible_for_holdout",
        )
        if decision.get("candidate_hash") != validated_hash:
            raise HarnessError("Validation decision hash does not match the candidate")
    path = context.run_dir / "frozen_candidate.json"
    if path.exists():
        frozen = _load_json(path)
        if not isinstance(frozen, Mapping):
            raise HarnessError("Frozen candidate artifact is invalid")
        if candidate_file is not None:
            candidate = _validate_frozen_candidate(_load_json(candidate_file))
            if _sha256_json(redact(candidate)) != frozen.get("candidate_hash"):
                raise HarnessError("Candidate differs from the immutable frozen candidate; start a new study")
        context.manifest["stages"]["freeze"].update(
            {
                "status": "completed",
                "candidate_hash": frozen.get("candidate_hash"),
                "artifact": "frozen_candidate.json",
            }
        )
        _save_manifest(context)
        return 0
    if candidate_file is None:
        if validated_candidate is None:
            raise HarnessError("freeze requires --candidate-file for dry-run planning")
        candidate = validated_candidate
    else:
        candidate = _validate_frozen_candidate(_load_json(candidate_file))
    safe_candidate = redact(candidate, known_secrets=_secret_values())
    candidate_hash = _sha256_json(safe_candidate)
    if validated_hash is not None and candidate_hash != validated_hash:
        raise HarnessError("Freeze candidate must exactly match the candidate that passed validation and confirmation")
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "run_id": context.manifest.get("run_id"),
        "frozen_at": None if context.dry_run else utc_now(),
        "dry_run": context.dry_run,
        "immutable": not context.dry_run,
        "candidate_hash": candidate_hash,
        "candidate": safe_candidate,
        "source": str(candidate_file.name),
        "protocol_hash": _sha256_json(context.manifest.get("protocol", {})),
    }
    if context.dry_run:
        _atomic_write_json(context.run_dir / "normalized" / "freeze.json", artifact)
        context.manifest["stages"]["freeze"]["status"] = "planned"
    else:
        _write_new_json(path, artifact)
        _atomic_write_json(context.run_dir / "normalized" / "freeze.json", artifact)
        context.manifest["stages"]["freeze"].update(
            {
                "status": "completed",
                "completed_at": utc_now(),
                "candidate_hash": candidate_hash,
                "artifact": "frozen_candidate.json",
            }
        )
    _save_manifest(context)
    return 0


def _load_frozen_candidate(context: RunContext) -> tuple[dict[str, Any], str]:
    path = context.run_dir / "frozen_candidate.json"
    if not path.exists():
        raise HarnessError("A frozen candidate is required before this stage")
    artifact = _load_json(path)
    if not isinstance(artifact, Mapping):
        raise HarnessError("Frozen candidate artifact is invalid")
    candidate = _validate_frozen_candidate(artifact.get("candidate"))
    candidate_hash = _sha256_json(redact(candidate))
    if artifact.get("candidate_hash") != candidate_hash:
        raise HarnessError("Frozen candidate hash verification failed")
    return candidate, candidate_hash


def _open_holdout(context: RunContext, specs: Sequence[CommandSpec], candidate_hash: str) -> None:
    lock_path = context.run_dir / "holdout_lock.json"
    plan_digest = _plan_digest(specs)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "run_id": context.manifest.get("run_id"),
        "immutable": True,
        "window": redact(context.config["research_windows"][context.config["holdout"]["window"]]),
        "candidate_hash": candidate_hash,
        "holdout_config_hash": _stage_config_hash("holdout", context.config, candidate_hash),
        "plan_digest": plan_digest,
        "command_ids": [spec.command_id for spec in specs],
    }
    if lock_path.exists():
        existing = _load_json(lock_path)
        if not isinstance(existing, Mapping):
            raise HarnessError("Holdout lock is invalid")
        comparable = {key: existing.get(key) for key in expected}
        if comparable != expected:
            raise HarnessError("Holdout lock does not match the frozen candidate and plan")
        return
    lock = {**expected, "opened_at": utc_now()}
    _write_new_json(lock_path, lock)
    context.manifest["protocol"]["holdout_opened_at"] = lock["opened_at"]
    context.manifest["protocol"]["holdout_lock"] = "holdout_lock.json"
    _save_manifest(context)


def run_holdout(context: RunContext, review_file: Path | None = None) -> int:
    if not context.dry_run:
        _require_interval_approval_enabled()
    candidate, candidate_hash = _load_frozen_candidate(context)
    specs = build_holdout_specs(context.config, candidate)
    _register_specs(context, "holdout", specs, extra_hash_input=candidate_hash)
    if not context.dry_run:
        _require_stage(context, "freeze")
        _open_holdout(context, specs, candidate_hash)
    result = _run_command_stage(context, "holdout", specs, extra_hash_input=candidate_hash)
    if not context.dry_run and context.manifest["stages"]["holdout"].get("status") == "completed":
        _write_validation_evidence(
            context,
            candidate_hash,
            specs,
            stage="holdout",
            filename="holdout_evidence.json",
            windows=("locked_holdout",),
        )
        holdout_config = dict(context.config.get("holdout") or {})
        assessment_path = context.run_dir / "holdout_assessment.json"
        assessment = _write_assessment(
            assessment_path,
            _assessment_with_evidence(
                context,
                _compute_validation_decision(
                    context,
                    candidate,
                    candidate_hash,
                    specs,
                    stage="holdout",
                    required_windows={"locked_holdout"},
                    minimum_opportunities=int(holdout_config.get("minimum_opportunities", 50)),
                    minimum_signals=int(holdout_config.get("minimum_signals", 25)),
                ),
                "holdout_evidence.json",
            ),
        )
        decision = _review_decision(
            context,
            assessment_path=assessment_path,
            decision_path=context.run_dir / "holdout_decision.json",
            review_file=review_file,
            approval_field="approved_for_shadow",
        )
        approved = bool(decision and decision.get("approved_for_shadow") is True)
        status = "completed" if approved else "rejected" if assessment.get("computed_gates_pass") is not True else "awaiting_review"
        context.manifest["stages"]["holdout"].update(
            {
                "status": status,
                "candidate_hash": candidate_hash,
                "evidence": "holdout_evidence.json",
                "assessment": "holdout_assessment.json",
                "decision": "holdout_decision.json" if decision else None,
                "approved_for_shadow": approved,
                "no_reselection": True,
            }
        )
        if status == "rejected":
            context.manifest.update(
                {
                    "status": "terminated_after_holdout",
                    "terminated_at": utc_now(),
                    "termination_reason": (
                        "Frozen candidate failed the immutable locked-holdout assessment; "
                        "reselection is prohibited in this study."
                    ),
                }
            )
        _save_manifest(context)
        if not approved:
            return 1
    return result


def _find_model_id(value: Any) -> str | None:
    if isinstance(value, Mapping):
        model_id = value.get("model_id")
        if isinstance(model_id, str) and model_id.strip():
            return model_id.strip()
        for item in value.values():
            found = _find_model_id(item)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_model_id(item)
            if found:
                return found
    return None


def _candidate_is_trainable(candidate: Mapping[str, Any]) -> bool:
    method = str(candidate.get("method", "")).lower()
    return method.startswith(("mlf_", "sf_", "skt_")) or method in {
        "mlforecast",
        "statsforecast",
        "sktime",
        "nhits",
        "nbeatsx",
        "patchtst",
        "tft",
    }


def _materialization_artifact(
    context: RunContext,
    spec: CommandSpec,
    candidate: Mapping[str, Any],
    candidate_hash: str,
) -> tuple[dict[str, Any], Path]:
    raw_path = _raw_path(context, "materialize", spec.command_id)
    envelope = _load_json(raw_path)
    payload = envelope.get("payload") if isinstance(envelope, Mapping) else None
    model_id = _find_model_id(payload)
    if _candidate_is_trainable(candidate) and not model_id:
        error = "Trainable candidate materialization succeeded without a model_id; require-existing shadow reuse cannot be proven"
        command_record = context.manifest["stages"]["materialize"]["commands"][spec.command_id]
        command_record.update({"status": "failed", "error": error})
        context.manifest["stages"]["materialize"]["status"] = "failed"
        if isinstance(envelope, dict):
            envelope["status"] = "failed"
            envelope["contract_error"] = error
            _atomic_write_json(raw_path, envelope)
        append_issue(
            context.run_dir,
            {
                "severity": "high",
                "category": "model_lifecycle",
                "reproduction_command": [sys.executable, "-m", "mtdata", "--json", *spec.argv],
                "context": {"stage": "materialize", "raw_output": _relative(raw_path, context.run_dir)},
                "impact": error,
                "workaround": "Do not start shadow mode until a persisted exact-pipeline model_id is returned.",
                "suggested_fix": "Expose model_info.model_id for every cache-persisted trainable forecast.",
                "status": "open",
            },
            run_id=str(context.manifest.get("run_id")),
        )
        write_normalized_stage(context, "materialize")
        _save_manifest(context)
        raise HarnessError(error)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "run_id": context.manifest.get("run_id"),
        "immutable": True,
        "materialized_at": utc_now(),
        "as_of": spec.metadata.get("as_of"),
        "candidate_hash": candidate_hash,
        "pipeline_hash": candidate_hash,
        "model_id": model_id,
        "trainable": _candidate_is_trainable(candidate),
        "cache_policy": "reuse",
        "refit_interval_bars": int(context.config.get("shadow", {}).get("refit_interval_bars", 24)),
        "raw_path": _relative(raw_path, context.run_dir),
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    }
    artifact_path = context.run_dir / "materializations" / f"{_slug(spec.command_id)}.json"
    if artifact_path.exists():
        existing = _load_json(artifact_path)
        for key in ("candidate_hash", "pipeline_hash", "model_id", "raw_sha256"):
            if not isinstance(existing, Mapping) or existing.get(key) != artifact.get(key):
                raise HarnessError("Immutable materialization artifact does not match its command result")
        artifact = dict(existing)
    else:
        _write_new_json(artifact_path, artifact)
    _atomic_write_json(
        context.run_dir / "materialization_latest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "artifact": _relative(artifact_path, context.run_dir),
            "candidate_hash": candidate_hash,
            "model_id": model_id,
            "updated_at": utc_now(),
        },
    )
    return artifact, artifact_path


def run_materialize(context: RunContext, as_of: str | None) -> int:
    if not context.dry_run:
        _require_interval_approval_enabled()
        _require_stage(context, "holdout")
    candidate, candidate_hash = _load_frozen_candidate(context)
    if not context.dry_run:
        holdout_specs = build_holdout_specs(context.config, candidate)
        if not (context.run_dir / "holdout_lock.json").exists():
            raise HarnessError("Materialization requires the immutable holdout lock")
        _open_holdout(context, holdout_specs, candidate_hash)
        holdout_decision_path = context.run_dir / "holdout_decision.json"
        holdout_assessment_path = context.run_dir / "holdout_assessment.json"
        if not holdout_decision_path.exists():
            raise HarnessError("Materialization requires an immutable approved holdout decision")
        holdout_decision = _load_review_decision_with_hash_checks(
            context,
            assessment_path=holdout_assessment_path,
            decision_path=holdout_decision_path,
            approval_field="approved_for_shadow",
        )
        if holdout_decision.get("candidate_hash") != candidate_hash:
            raise HarnessError("Holdout decision is not approved_for_shadow for this frozen candidate")
    specs = build_materialize_specs(context.config, candidate, as_of)
    result = _run_command_stage(
        context,
        "materialize",
        specs,
        extra_hash_input=candidate_hash,
        allow_append=True,
    )
    if not context.dry_run and result == 0:
        artifact, artifact_path = _materialization_artifact(
            context,
            specs[0],
            candidate,
            candidate_hash,
        )
        context.manifest["stages"]["materialize"].update(
            {
                "status": "completed",
                "candidate_hash": candidate_hash,
                "latest_artifact": _relative(artifact_path, context.run_dir),
                "model_id": artifact.get("model_id"),
            }
        )
        _save_manifest(context)
    return result


def _load_latest_materialization(context: RunContext, candidate_hash: str) -> dict[str, Any]:
    pointer_path = context.run_dir / "materialization_latest.json"
    if not pointer_path.exists():
        raise HarnessError("Shadow forecasting requires an exact-pipeline materialize stage first")
    pointer = _load_json(pointer_path)
    if not isinstance(pointer, Mapping) or pointer.get("candidate_hash") != candidate_hash:
        raise HarnessError("Latest materialization does not match the frozen candidate hash")
    artifact_path = context.run_dir / str(pointer.get("artifact") or "")
    artifact = _load_json(artifact_path)
    if not isinstance(artifact, Mapping) or artifact.get("candidate_hash") != candidate_hash:
        raise HarnessError("Latest materialization artifact failed candidate-hash verification")
    return dict(artifact)


def _parse_reference_time(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise HarnessError("Materialization/shadow reference time is missing")
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise HarnessError(
            f"Reference time {value!r} is not ISO-8601; use an explicit timestamp so refit staleness can be enforced"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _enforce_materialization_freshness(
    candidate: Mapping[str, Any],
    materialization: Mapping[str, Any],
    shadow_as_of: str | None,
) -> None:
    materialized_reference = _parse_reference_time(materialization.get("as_of"))
    shadow_reference = _parse_reference_time(shadow_as_of or utc_now())
    if shadow_reference < materialized_reference:
        raise HarnessError("Shadow as_of precedes model materialization and would introduce look-ahead leakage")
    timeframe = str(candidate.get("timeframe", "")).upper()
    timeframe_minutes = _TIMEFRAME_MINUTES.get(timeframe)
    if timeframe_minutes is None:
        raise HarnessError(f"Cannot enforce model staleness for unsupported timeframe {timeframe!r}")
    elapsed_bars = (shadow_reference - materialized_reference).total_seconds() / (timeframe_minutes * 60.0)
    limit = int(materialization.get("refit_interval_bars") or 24)
    if elapsed_bars > limit:
        raise HarnessError(
            f"Materialized model is approximately {elapsed_bars:.1f} {timeframe} bars old, above the {limit}-bar limit; run materialize again before shadow"
        )


def run_shadow(context: RunContext, as_of: str | None) -> int:
    if not context.dry_run:
        _require_interval_approval_enabled()
        _require_stage(context, "holdout")
    candidate, candidate_hash = _load_frozen_candidate(context)
    lock_path = context.run_dir / "holdout_lock.json"
    if not context.dry_run and not lock_path.exists():
        raise HarnessError("Shadow forecasting requires a verified holdout lock")
    materialization: dict[str, Any] | None = None
    if not context.dry_run:
        _open_holdout(context, build_holdout_specs(context.config, candidate), candidate_hash)
        holdout_decision = _load_review_decision_with_hash_checks(
            context,
            assessment_path=context.run_dir / "holdout_assessment.json",
            decision_path=context.run_dir / "holdout_decision.json",
            approval_field="approved_for_shadow",
        )
        if holdout_decision.get("candidate_hash") != candidate_hash:
            raise HarnessError("Shadow holdout decision does not match the frozen candidate")
        _require_stage(context, "materialize")
        materialization = _load_latest_materialization(context, candidate_hash)
        if _candidate_is_trainable(candidate) and not materialization.get("model_id"):
            raise HarnessError("Trainable frozen candidate has no persisted materialization model_id")
        _enforce_materialization_freshness(candidate, materialization, as_of)
    specs = build_shadow_specs(
        context.config,
        candidate,
        as_of,
        model_id=str(materialization["model_id"]) if materialization and materialization.get("model_id") else None,
    )
    _register_specs(context, "shadow", specs, extra_hash_input=candidate_hash, allow_append=True)
    return _run_command_stage(
        context,
        "shadow",
        specs,
        extra_hash_input=candidate_hash,
        allow_append=True,
        continuous=True,
    )


def _stage_counts(manifest: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stage, record in manifest.get("stages", {}).items():
        commands = record.get("commands", {}) if isinstance(record, Mapping) else {}
        counts: dict[str, int] = {}
        for command in commands.values():
            status = str(command.get("status", "unknown")) if isinstance(command, Mapping) else "unknown"
            counts[status] = counts.get(status, 0) + 1
        result[str(stage)] = {"status": record.get("status"), "commands": counts}
    return result


def generate_report(context: RunContext) -> int:
    ledger = _load_issue_ledger(context.run_dir, str(context.manifest.get("run_id")))
    frozen_path = context.run_dir / "frozen_candidate.json"
    validation_decision_path = context.run_dir / "validation_decision.json"
    holdout_lock_path = context.run_dir / "holdout_lock.json"
    holdout_decision_path = context.run_dir / "holdout_decision.json"
    materialization_path = context.run_dir / "materialization_latest.json"
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "run_id": context.manifest.get("run_id"),
        "symbol": "BTCUSD",
        "research_only": True,
        "disclaimer": "Historical performance and forecast intervals do not establish live-trading profitability.",
        "interval_approval_enabled": INTERVAL_APPROVAL_ENABLED,
        "interval_approval_disabled_reason": INTERVAL_APPROVAL_DISABLED_REASON,
        "protocol": context.manifest.get("protocol"),
        "stages": _stage_counts(context.manifest),
        "frozen_candidate": _load_json(frozen_path) if frozen_path.exists() else None,
        "validation_decision": _load_json(validation_decision_path) if validation_decision_path.exists() else None,
        "holdout_lock": _load_json(holdout_lock_path) if holdout_lock_path.exists() else None,
        "holdout_decision": _load_json(holdout_decision_path) if holdout_decision_path.exists() else None,
        "latest_materialization": _load_json(materialization_path) if materialization_path.exists() else None,
        "issues": {
            "count": len(ledger.get("issues", [])),
            "open": sum(1 for issue in ledger.get("issues", []) if isinstance(issue, Mapping) and issue.get("status") == "open"),
            "path": "issues.json",
        },
    }
    _atomic_write_json(context.run_dir / "report.json", redact(report, known_secrets=_secret_values()))
    lines = [
        "# BTCUSD forecast experiment",
        "",
        f"Run: `{report['run_id']}`",
        "",
        "Research only. Historical performance and forecast intervals do not establish live-trading profitability.",
        "",
        "## Protocol status",
        "",
        "| Stage | Status | Completed | Failed | Remaining |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage in RESEARCH_STAGES:
        info = report["stages"].get(stage, {})
        counts = info.get("commands", {})
        remaining = sum(
            int(count)
            for status, count in counts.items()
            if status not in {"completed"}
        )
        lines.append(
            f"| {stage} | {info.get('status', 'not_started')} | {counts.get('completed', 0)} | {counts.get('failed', 0)} | {remaining} |"
        )
    lines.extend(
        [
            "",
            "## Integrity artifacts",
            "",
            f"- Interval approval enabled: {'yes' if INTERVAL_APPROVAL_ENABLED else 'no'}",
            f"- Interval approval blocker: {INTERVAL_APPROVAL_DISABLED_REASON}",
            f"- Frozen candidate: {'present' if frozen_path.exists() else 'not yet frozen'}",
            f"- Validation decision: {'eligible' if (report.get('validation_decision') or {}).get('eligible_for_holdout') else 'not eligible/not yet available'}",
            f"- Locked holdout opened: {'yes' if holdout_lock_path.exists() else 'no'}",
            f"- Holdout approved for shadow: {'yes' if (report.get('holdout_decision') or {}).get('approved_for_shadow') else 'no'}",
            f"- Exact-pipeline materialization: {'present' if materialization_path.exists() else 'not yet materialized'}",
            f"- Issues/friction: {report['issues']['count']} total, {report['issues']['open']} open",
            "",
            "See `report.json`, `manifest.json`, `normalized/`, and `raw/` for machine-readable evidence.",
            "",
        ]
    )
    report_md_path = context.run_dir / "report.md"
    report_md_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    context.manifest["stages"]["report"].update(
        {
            "status": "completed",
            "completed_at": utc_now(),
            "artifacts": ["report.json", "report.md"],
        }
    )
    _save_manifest(context)
    return 0


def _print_json(value: Any) -> None:
    json.dump(value, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def _run_issue_command(context: RunContext, args: argparse.Namespace) -> int:
    if args.issue_action == "list":
        ledger = _load_issue_ledger(context.run_dir, str(context.manifest.get("run_id")))
        issues = list(ledger.get("issues", []))
        if args.status:
            issues = [issue for issue in issues if isinstance(issue, Mapping) and issue.get("status") == args.status]
        _print_json({**ledger, "issues": issues})
        return 0
    reproduction: Any = args.reproduction_command
    try:
        parsed_reproduction = json.loads(reproduction)
        if isinstance(parsed_reproduction, (str, list)):
            reproduction = parsed_reproduction
    except json.JSONDecodeError:
        pass
    context_value: Any = args.context
    try:
        context_value = json.loads(context_value)
    except json.JSONDecodeError:
        pass
    issue, created = append_issue(
        context.run_dir,
        {
            "id": args.issue_id,
            "severity": args.severity,
            "category": args.category,
            "observed_at": args.observed_at or utc_now(),
            "reproduction_command": reproduction,
            "context": context_value,
            "impact": args.impact,
            "workaround": args.workaround,
            "suggested_fix": args.suggested_fix,
            "status": args.status,
        },
        run_id=str(context.manifest.get("run_id")),
    )
    _print_json({"created": created, "issue": issue})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Study directory; required with --resume")
    parser.add_argument("--run-id", help="Stable study identifier for a new run")
    parser.add_argument("--config", type=Path, help="JSON override merged over the preregistered protocol defaults")
    parser.add_argument("--resume", action="store_true", help="Resume an existing study without rerunning completed shards")
    parser.add_argument("--dry-run", action="store_true", help="Persist and display plans without invoking mtdata")
    parser.add_argument("--timeout", type=float, default=3600.0, help="Per-mtdata-command timeout in seconds")
    parser.add_argument("--max-commands", type=int, help="Run at most this many pending shards in this invocation")
    parser.add_argument("--fail-fast", action="store_true", help="Stop a stage invocation after the first failed command")
    subparsers = parser.add_subparsers(dest="action", required=True)
    for stage in ("audit", "screen", "tune", "report"):
        subparsers.add_parser(stage, help=f"Run or resume the {stage} stage")
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate one locked candidate across both pre-holdout windows",
    )
    validate_parser.add_argument("--candidate-file", type=Path, help="Exact candidate JSON; required initially")
    validate_parser.add_argument(
        "--review-file",
        type=Path,
        help="Reserved for the future raw-envelope interval verifier; approval is currently disabled",
    )
    freeze_parser = subparsers.add_parser(
        "freeze",
        help="Write-once candidate freeze; disabled until raw interval evidence can be recomputed",
    )
    freeze_parser.add_argument("--candidate-file", type=Path, help="Optional exact copy of the validated candidate JSON")
    holdout_parser = subparsers.add_parser(
        "holdout",
        help="Locked holdout stage; opening is disabled until raw interval evidence can be recomputed",
    )
    holdout_parser.add_argument(
        "--review-file",
        type=Path,
        help="Reserved for the future raw-envelope interval verifier; approval is currently disabled",
    )
    materialize_parser = subparsers.add_parser(
        "materialize",
        help="Materialize the exact frozen pipeline in the isolated model store before shadow use",
    )
    materialize_parser.add_argument("--as-of", help="Reference cutoff for this write-once materialization observation")
    shadow_parser = subparsers.add_parser("shadow", help="Generate a no-trade forecast and conformal interval snapshot")
    shadow_parser.add_argument("--as-of", help="Historical or live reference cutoff; omit for the current closed-bar view")
    issue_parser = subparsers.add_parser("issue", help="Append to or inspect the per-study issues/friction ledger")
    issue_subparsers = issue_parser.add_subparsers(dest="issue_action", required=True)
    issue_add = issue_subparsers.add_parser("add", help="Append or deduplicate one issue")
    issue_add.add_argument("--id", dest="issue_id", help="Stable issue ID; generated deterministically when omitted")
    issue_add.add_argument("--severity", choices=("low", "medium", "high", "critical"), default="medium")
    issue_add.add_argument("--category", required=True)
    issue_add.add_argument("--observed-at")
    issue_add.add_argument("--reproduction-command", required=True, help="Command string or JSON argv list")
    issue_add.add_argument("--context", default="{}", help="Context text or JSON value")
    issue_add.add_argument("--impact", required=True)
    issue_add.add_argument("--workaround", default="")
    issue_add.add_argument("--suggested-fix", required=True)
    issue_add.add_argument("--status", choices=("open", "accepted", "resolved", "wont_fix"), default="open")
    issue_list = issue_subparsers.add_parser("list", help="Print the redacted ledger as JSON")
    issue_list.add_argument("--status", choices=("open", "accepted", "resolved", "wont_fix"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_commands is not None and args.max_commands < 1:
        parser.error("--max-commands must be at least 1")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        context = prepare_context(args)
        if args.action == "audit":
            result = run_audit(context)
        elif args.action == "screen":
            result = run_screen(context)
        elif args.action == "tune":
            result = run_tune(context)
        elif args.action == "validate":
            result = run_validate(context, args.candidate_file, args.review_file)
        elif args.action == "freeze":
            result = freeze_candidate(context, args.candidate_file)
        elif args.action == "holdout":
            result = run_holdout(context, args.review_file)
        elif args.action == "materialize":
            result = run_materialize(context, args.as_of)
        elif args.action == "shadow":
            result = run_shadow(context, args.as_of)
        elif args.action == "report":
            result = generate_report(context)
        elif args.action == "issue":
            result = _run_issue_command(context, args)
        else:  # pragma: no cover - argparse prevents this
            raise HarnessError(f"Unsupported action {args.action!r}")
        if args.action not in {"issue"}:
            _print_json(
                {
                    "success": result == 0,
                    "run_id": context.manifest.get("run_id"),
                    "run_dir": str(context.run_dir),
                    "action": args.action,
                    "stage_status": context.manifest.get("stages", {}).get(args.action, {}).get("status"),
                    "manifest": str(context.manifest_path),
                }
            )
        return result
    except HarnessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
