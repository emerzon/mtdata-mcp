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
import importlib.metadata
import json
import math
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
HARNESS_VERSION = "0.3.0"
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
    enforce_execution_integrity: bool


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


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


_PACKAGE_DISTRIBUTIONS = (
    "mtdata-mcp",
    "numpy",
    "pandas",
    "mlforecast",
    "lightgbm",
    "scikit-learn",
    "pandas-ta-classic",
    "TA-Lib",
)


def _package_version_snapshot() -> dict[str, dict[str, str | None]]:
    versions: dict[str, dict[str, str | None]] = {}
    for distribution in _PACKAGE_DISTRIBUTIONS:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = {"status": "missing", "version": None}
        else:
            versions[distribution] = {"status": "installed", "version": version}
    return versions


def _runtime_identity_snapshot() -> dict[str, Any]:
    return {
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "package_versions": _package_version_snapshot(),
    }


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


def _screen_shards(
    window_name: str,
    window: Mapping[str, Any],
    sharding: Any,
) -> list[dict[str, str]]:
    policy = str(sharding or "month_end").strip().lower()
    if policy in {"month_end", "month_end_plus_exact_window_end"}:
        return _month_end_shards(window_name, window)
    if policy != "window_end":
        raise HarnessError(
            "screen.sharding must be 'month_end' or 'window_end' "
            "('month_end_plus_exact_window_end' remains a backward-compatible alias)"
        )
    start = date.fromisoformat(str(window["start"]))
    end = date.fromisoformat(str(window["end"]))
    if end < start:
        raise HarnessError(f"Window {window_name!r} ends before it starts")
    return [
        {
            "id": "window-end",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "kind": "exact_window_end",
        }
    ]


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


def _resolve_shard_steps(
    value: Any,
    shard: Mapping[str, str],
    timeframe: str,
    horizon: int,
    spacing: int,
    *,
    history_reserve: int = 0,
) -> int:
    if str(value).lower() != "auto_month":
        steps = int(value)
    else:
        shard_bars = (
            _continuous_bars(shard["start"], shard["end"], timeframe)
            - int(history_reserve)
        )
        # The CLI fetches lookback + steps*spacing + horizon bars.  Reserving one
        # full spacing interval beyond the first anchor also gives monthly shards
        # the same feed-gap margin as the historical spacing==horizon behavior.
        steps = max(1, (shard_bars - horizon) // spacing)
    if steps < 1:
        raise HarnessError(f"Shard {shard['id']} is shorter than horizon={horizon}")
    if steps > 200:
        raise HarnessError(
            f"Full shard {shard['id']} needs {steps} anchors, above mtdata's 200-step limit; use a coarser timeframe or split the shard"
        )
    return steps


def _positive_screen_spacing(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise HarnessError("screen.spacing_bars must be a positive integer")
    return value


def _strict_iso_date(value: Any, field: str) -> date:
    if not isinstance(value, str):
        raise HarnessError(f"{field} must be an ISO date in YYYY-MM-DD form")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise HarnessError(
            f"{field} must be an ISO date in YYYY-MM-DD form"
        ) from exc
    if parsed.isoformat() != value:
        raise HarnessError(f"{field} must be an ISO date in YYYY-MM-DD form")
    return parsed


def _resolve_screen_anchor_grid(
    screen: Mapping[str, Any],
    window_name: str,
    window: Mapping[str, Any],
) -> dict[str, Any] | None:
    raw_grid = screen.get("anchor_grid")
    if raw_grid is None:
        return None
    if not isinstance(raw_grid, Mapping):
        raise HarnessError("screen.anchor_grid must be a JSON object")

    grid_id_raw = raw_grid.get("id")
    if not isinstance(grid_id_raw, str) or not grid_id_raw.strip():
        raise HarnessError("screen.anchor_grid.id must be a non-empty string")
    grid_id = _slug(grid_id_raw)
    history_start = _strict_iso_date(
        raw_grid.get("history_start"),
        "screen.anchor_grid.history_start",
    )
    role_start = _strict_iso_date(
        window.get("start"),
        f"research_windows.{window_name}.start",
    )
    role_end = _strict_iso_date(
        window.get("end"),
        f"research_windows.{window_name}.end",
    )
    if role_end < role_start:
        raise HarnessError(f"Window {window_name!r} ends before it starts")

    literal_anchors = raw_grid.get("anchors")
    generator_keys = {"start", "end", "every_days", "hour_utc"}
    generator_present = any(key in raw_grid for key in generator_keys)
    anchors: list[str] = []
    if literal_anchors is not None:
        if generator_present:
            raise HarnessError(
                "screen.anchor_grid must use either anchors or the date generator, not both"
            )
        if not isinstance(literal_anchors, list):
            raise HarnessError("screen.anchor_grid.anchors must be a JSON array")
        for value in literal_anchors:
            canonical = _canonical_utc_timestamp(value)
            if canonical is None:
                raise HarnessError(
                    "screen.anchor_grid.anchors must contain timezone-aware UTC ISO timestamps"
                )
            anchors.append(canonical)
    else:
        grid_start = _strict_iso_date(
            raw_grid.get("start"),
            "screen.anchor_grid.start",
        )
        grid_end = _strict_iso_date(
            raw_grid.get("end"),
            "screen.anchor_grid.end",
        )
        every_days = raw_grid.get("every_days")
        hour_utc = raw_grid.get("hour_utc", 0)
        if (
            isinstance(every_days, bool)
            or not isinstance(every_days, int)
            or every_days < 1
        ):
            raise HarnessError("screen.anchor_grid.every_days must be a positive integer")
        if (
            isinstance(hour_utc, bool)
            or not isinstance(hour_utc, int)
            or not 0 <= hour_utc <= 23
        ):
            raise HarnessError("screen.anchor_grid.hour_utc must be an integer from 0 to 23")
        if grid_end < grid_start:
            raise HarnessError("screen.anchor_grid.end must not precede start")
        current = grid_start
        while current <= grid_end:
            anchor = datetime.combine(
                current,
                datetime.min.time(),
                tzinfo=timezone.utc,
            ) + timedelta(hours=hour_utc)
            anchors.append(
                anchor.isoformat(timespec="seconds").replace("+00:00", "Z")
            )
            if len(anchors) > 200:
                raise HarnessError("screen.anchor_grid cannot contain more than 200 anchors")
            current += timedelta(days=every_days)

    if not anchors:
        raise HarnessError("screen.anchor_grid must contain at least one anchor")
    if len(anchors) > 200:
        raise HarnessError("screen.anchor_grid cannot contain more than 200 anchors")
    parsed_anchors = [
        datetime.fromisoformat(value.replace("Z", "+00:00")) for value in anchors
    ]
    if any(
        current <= previous
        for previous, current in zip(parsed_anchors, parsed_anchors[1:])
    ):
        raise HarnessError(
            "screen.anchor_grid anchors must be strictly increasing and unique"
        )
    role_start_at = datetime.combine(
        role_start,
        datetime.min.time(),
        tzinfo=timezone.utc,
    )
    role_limit = datetime.combine(
        role_end + timedelta(days=1),
        datetime.min.time(),
        tzinfo=timezone.utc,
    )
    if any(anchor < role_start_at or anchor >= role_limit for anchor in parsed_anchors):
        raise HarnessError(
            "screen.anchor_grid anchors must remain inside the registered screen window"
        )
    history_start_at = datetime.combine(
        history_start,
        datetime.min.time(),
        tzinfo=timezone.utc,
    )
    if history_start_at > parsed_anchors[0]:
        raise HarnessError(
            "screen.anchor_grid.history_start must not follow the first anchor"
        )
    return {
        "id": grid_id,
        "anchors": anchors,
        "sha256": _sha256_json(anchors),
        "count": len(anchors),
        "history_start": history_start.isoformat(),
        "role_window": {
            "name": window_name,
            "start": role_start.isoformat(),
            "end": role_end.isoformat(),
        },
    }


def _explicit_grid_spacing_bars(
    anchor_grid: Mapping[str, Any],
    timeframe: str,
) -> int | None:
    timeframe_minutes = _TIMEFRAME_MINUTES.get(timeframe.upper())
    if timeframe_minutes is None:
        raise HarnessError(
            f"Explicit anchor grids do not support timeframe {timeframe!r}"
        )
    anchors = list(anchor_grid["anchors"])
    parsed = [datetime.fromisoformat(value.replace("Z", "+00:00")) for value in anchors]
    timeframe_seconds = timeframe_minutes * 60
    if any(int(value.timestamp()) % timeframe_seconds != 0 for value in parsed):
        raise HarnessError(
            f"screen.anchor_grid contains an anchor that is not aligned to {timeframe}"
        )
    if len(parsed) < 2:
        return None
    deltas = [int((current - previous).total_seconds()) for previous, current in zip(parsed, parsed[1:])]
    if any(delta <= 0 or delta % timeframe_seconds for delta in deltas):
        raise HarnessError(
            f"screen.anchor_grid spacing is not an exact number of {timeframe} bars"
        )
    spacing_values = {delta // timeframe_seconds for delta in deltas}
    return spacing_values.pop() if len(spacing_values) == 1 else None


def _validate_explicit_grid_for_command(
    anchor_grid: Mapping[str, Any],
    shard: Mapping[str, str],
    timeframe: str,
    horizon: int,
    lookback: int,
) -> tuple[int, int | None]:
    anchors = list(anchor_grid["anchors"])
    timeframe_minutes = _TIMEFRAME_MINUTES[timeframe.upper()]
    parsed_anchors = [
        datetime.fromisoformat(value.replace("Z", "+00:00")) for value in anchors
    ]
    if any(
        current - previous < timedelta(minutes=horizon * timeframe_minutes)
        for previous, current in zip(parsed_anchors, parsed_anchors[1:])
    ):
        raise HarnessError(
            "screen.anchor_grid validation windows must not overlap "
            f"for horizon={horizon}"
        )
    history_start_at = datetime.combine(
        date.fromisoformat(str(anchor_grid["history_start"])),
        datetime.min.time(),
        tzinfo=timezone.utc,
    )
    history_bars = int(
        (parsed_anchors[0] - history_start_at).total_seconds()
        // (timeframe_minutes * 60)
    ) + 1
    if history_bars < lookback:
        raise HarnessError(
            "screen.anchor_grid.history_start does not provide "
            f"lookback={lookback} {timeframe} bars before the first anchor"
        )
    role_end_limit = datetime.combine(
        date.fromisoformat(str(shard["end"])) + timedelta(days=1),
        datetime.min.time(),
        tzinfo=timezone.utc,
    )
    if (
        parsed_anchors[-1] + timedelta(minutes=horizon * timeframe_minutes)
        >= role_end_limit
    ):
        raise HarnessError(
            "screen.anchor_grid final realized target falls outside "
            f"the registered window for horizon={horizon}"
        )
    return len(anchors), _explicit_grid_spacing_bars(anchor_grid, timeframe)


def _has_nonempty_features(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    try:
        return bool(len(value))
    except (TypeError, ValueError):
        return bool(value)


def _split_feature_tokens(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    if not isinstance(value, str):
        return []
    tokens: list[str] = []
    current: list[str] = []
    depth = 0
    for character in value:
        if character == "(":
            depth += 1
        elif character == ")" and depth:
            depth -= 1
        if depth == 0 and character in {",", " "}:
            token = "".join(current).strip()
            if token:
                tokens.append(token)
            current = []
        else:
            current.append(character)
    token = "".join(current).strip()
    if token:
        tokens.append(token)
    return tokens


def _semantic_feature_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _feature_expectations(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping) or not value:
        return None
    semantic_columns: list[dict[str, str]] = []
    complete = True
    observed = False

    include_value = value.get("include")
    if include_value is None:
        include_value = value.get("exog")
    observed = bool(include_value)
    include_tokens = _split_feature_tokens(include_value)
    if include_value and not include_tokens:
        complete = False
    for token in include_tokens:
        if token.lower() in {"all", "ohlcv", "price", "volume"}:
            complete = False
            continue
        observed = True
        semantic_columns.append(
            {
                "kind": "observed_column",
                "requested": token,
                "semantic_id": _semantic_feature_token(token),
            }
        )

    indicators_value = value.get("indicators")
    if indicators_value is None:
        indicators_value = value.get("ti")
    observed = observed or bool(indicators_value)
    indicator_tokens = _split_feature_tokens(indicators_value)
    if indicators_value and not indicator_tokens:
        complete = False
    simple_indicators = {"rsi", "roc", "natr"}
    for indicator in indicator_tokens:
        match = re.fullmatch(
            r"(?P<name>[a-zA-Z][a-zA-Z0-9_]*)\(\s*(?P<period>[0-9]+)\s*\)",
            indicator,
        )
        if match is None or match.group("name").lower() not in simple_indicators:
            complete = False
            continue
        observed = True
        semantic_columns.append(
            {
                "kind": "indicator",
                "requested": indicator,
                "semantic_id": _semantic_feature_token(
                    f"{match.group('name')}{match.group('period')}"
                ),
            }
        )

    calendar_columns = {
        "hour": ("hr_sin", "hr_cos"),
        "hr": ("hr_sin", "hr_cos"),
        "dow": ("dow_sin", "dow_cos"),
        "wday": ("dow_sin", "dow_cos"),
        "weekday": ("dow_sin", "dow_cos"),
        "dayofweek": ("dow_sin", "dow_cos"),
    }
    future_value = value.get("future_covariates")
    future_tokens = _split_feature_tokens(future_value)
    if future_value and not future_tokens:
        complete = False
    for requested in future_tokens:
        columns = calendar_columns.get(requested.lower())
        if columns is None:
            complete = False
            continue
        semantic_columns.extend(
            {
                "kind": "calendar",
                "requested": requested,
                "semantic_id": _semantic_feature_token(column),
            }
            for column in columns
        )

    return {
        "complete": complete,
        "n_features": len(semantic_columns) if complete else None,
        "semantic_columns": semantic_columns,
        "has_observed_features": observed,
        "observed_future_policy": value.get("observed_future_policy"),
    }


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
    raw_indicator_descriptions = audit.get("indicator_descriptions", [])
    if not isinstance(raw_indicator_descriptions, list):
        raise HarnessError("audit.indicator_descriptions must be a list of indicator names")
    indicator_command_ids: set[str] = set()
    for raw_name in raw_indicator_descriptions:
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise HarnessError(
                "audit.indicator_descriptions entries must be non-empty strings"
            )
        name = raw_name.strip()
        command_id = f"indicator-{_slug(name)}"
        if command_id in indicator_command_ids:
            raise HarnessError(
                "audit.indicator_descriptions entries must have distinct canonical names"
            )
        indicator_command_ids.add(command_id)
        specs.append(
            CommandSpec(
                command_id,
                ("indicators_describe", name, "--detail", "full"),
                {"kind": "indicator_description", "indicator": name},
            )
        )
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
    window = windows[window_name]
    anchor_grid = _resolve_screen_anchor_grid(screen, window_name, window)
    sharding_policy = str(screen.get("sharding") or "month_end").strip().lower()
    if anchor_grid is None:
        shards = _screen_shards(
            window_name,
            window,
            sharding_policy,
        )
    else:
        if screen.get("spacing_bars") is not None:
            raise HarnessError(
                "screen.spacing_bars is rolling-only and cannot be combined with anchor_grid"
            )
        role_window = anchor_grid["role_window"]
        shards = [
            {
                "id": str(anchor_grid["id"]),
                "start": str(role_window["start"]),
                "end": str(role_window["end"]),
                "kind": "explicit_anchor_grid",
            }
        ]
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
    explicit_spacing = (
        _positive_screen_spacing(screen["spacing_bars"])
        if anchor_grid is None and screen.get("spacing_bars") is not None
        else None
    )
    detail = str(screen.get("detail", "full"))
    if detail != "full":
        raise HarnessError(
            "screen.detail must be 'full' so every research command preserves "
            "anchor-level forecast and actual paths"
        )
    specs: list[CommandSpec] = []
    for shard in shards:
        for timeframe in timeframes:
            for horizon in horizons:
                if horizon < 1:
                    raise HarnessError("Screen horizons must be positive")
                for quantity in quantities:
                    for lookback in lookbacks:
                        if anchor_grid is not None:
                            steps, spacing = _validate_explicit_grid_for_command(
                                anchor_grid,
                                shard,
                                timeframe,
                                horizon,
                                lookback,
                            )
                        else:
                            spacing = (
                                explicit_spacing
                                if explicit_spacing is not None
                                else horizon
                            )
                            steps = _resolve_shard_steps(
                                steps_setting,
                                shard,
                                timeframe,
                                horizon,
                                int(spacing),
                                history_reserve=(
                                    lookback if sharding_policy == "window_end" else 0
                                ),
                            )
                            if steps > 1 and int(spacing) < horizon:
                                raise HarnessError(
                                    "screen.spacing_bars must be greater than or equal to "
                                    f"horizon={horizon} when steps_per_shard is greater than 1"
                                )
                            available_bars = _continuous_bars(
                                str(window["start"]),
                                shard["end"],
                                timeframe,
                            )
                            required_bars = lookback + (steps * int(spacing)) + horizon
                            if available_bars < required_bars:
                                # Never fill an early development shard with observations
                                # from the stress window merely to satisfy a long lookback.
                                continue
                        for raw_variant in variants:
                            variant = dict(raw_variant or {})
                            variant_id = _slug(variant.get("id", "raw"))
                            variant_features = variant.get("features")
                            feature_contract_required = _has_nonempty_features(
                                variant_features
                            )
                            feature_expectations = _feature_expectations(
                                variant_features
                            )
                            if feature_contract_required:
                                if detail != "full":
                                    raise HarnessError(
                                        "Feature-bearing screen variants require "
                                        "screen.detail='full' so per-anchor paths and "
                                        "consumption evidence can be verified"
                                    )
                                if (
                                    not isinstance(feature_expectations, Mapping)
                                    or feature_expectations.get("complete") is not True
                                ):
                                    raise HarnessError(
                                        f"Feature-bearing screen variant {variant_id!r} "
                                        "cannot be fully verified by the harness; use "
                                        "explicit supported include columns, rsi/roc/natr "
                                        "periods, and hour/dow calendar covariates"
                                    )
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
                                "--lookback",
                                str(lookback),
                                "--methods",
                                *variant_methods,
                                "--quantity",
                                quantity,
                                "--start",
                                (
                                    str(anchor_grid["history_start"])
                                    if anchor_grid is not None
                                    else str(window["start"])
                                ),
                                "--end",
                                shard["end"],
                                "--detail",
                                detail,
                            ]
                            if anchor_grid is None:
                                argv[argv.index("--lookback"):argv.index("--lookback")] = [
                                    "--steps",
                                    str(steps),
                                    "--spacing",
                                    str(spacing),
                                ]
                            else:
                                argv[argv.index("--lookback"):argv.index("--lookback")] = [
                                    "--anchors",
                                    _json_arg(anchor_grid["anchors"]),
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
                                        "spacing": spacing,
                                        "anchor_mode": (
                                            "explicit"
                                            if anchor_grid is not None
                                            else "rolling"
                                        ),
                                        "anchor_grid_id": (
                                            anchor_grid["id"]
                                            if anchor_grid is not None
                                            else None
                                        ),
                                        "expected_anchors": (
                                            list(anchor_grid["anchors"])
                                            if anchor_grid is not None
                                            else None
                                        ),
                                        "expected_anchors_sha256": (
                                            anchor_grid["sha256"]
                                            if anchor_grid is not None
                                            else None
                                        ),
                                        "history_start": (
                                            anchor_grid["history_start"]
                                            if anchor_grid is not None
                                            else str(window["start"])
                                        ),
                                        "role_window": (
                                            dict(anchor_grid["role_window"])
                                            if anchor_grid is not None
                                            else {
                                                "name": window_name,
                                                "start": str(window["start"]),
                                                "end": str(window["end"]),
                                            }
                                        ),
                                        "detail": detail,
                                        "training_floor": (
                                            str(anchor_grid["history_start"])
                                            if anchor_grid is not None
                                            else str(window["start"])
                                        ),
                                        "methods": variant_methods,
                                        "variant": variant_id,
                                        "feature_contract_required": feature_contract_required,
                                        "features": variant_features,
                                        "feature_expectations": feature_expectations,
                                        "params": variant.get("params"),
                                        "params_per_method": variant.get(
                                            "params_per_method"
                                        ),
                                        "denoise": variant.get("denoise"),
                                        "dimred": variant.get("dimred"),
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


_SOURCE_INTEGRITY_ERROR_CODE = "BTC-SOURCE-INTEGRITY"


def _git_source_state() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]

    def _git(*arguments: str) -> str:
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
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "git command failed").strip()
            raise HarnessError(
                f"{_SOURCE_INTEGRITY_ERROR_CODE}: could not inspect repository source: {detail}"
            )
        return completed.stdout.strip()

    head = _git("rev-parse", "HEAD")
    tracked_status = _git("status", "--porcelain", "--untracked-files=no")
    untracked_runtime_files = _git(
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
    source_status = {
        "tracked_status": tracked_status,
        "untracked_runtime_files": untracked_runtime_files,
    }
    return {
        "git_head": head,
        "tracked_tree_dirty": bool(tracked_status),
        "untracked_runtime_files": untracked_runtime_files,
        "source_tree_dirty": bool(tracked_status or untracked_runtime_files),
        "source_status_sha256": _sha256_json(source_status),
        "captured_at": utc_now(),
    }


def _source_integrity_error(reason: str) -> HarnessError:
    return HarnessError(
        f"{_SOURCE_INTEGRITY_ERROR_CODE}: {reason}. Remediation: start a new run "
        "from one clean committed HEAD; use only issue list to inspect an old run"
    )


def _is_read_only_issue_list(args: argparse.Namespace) -> bool:
    return args.action == "issue" and getattr(args, "issue_action", None) == "list"


def _enforce_source_integrity(args: argparse.Namespace) -> None:
    if _is_read_only_issue_list(args):
        return
    current = _git_source_state()
    current_runtime = _runtime_identity_snapshot()
    args._verified_source_state = current
    args._verified_runtime_identity = current_runtime
    if args.resume:
        if args.run_dir is None:
            return
        manifest_path = Path(args.run_dir).resolve() / "manifest.json"
        if not manifest_path.exists():
            return
        manifest = _load_json(manifest_path)
        recorded = manifest.get("source") if isinstance(manifest, Mapping) else None
        if not isinstance(recorded, Mapping):
            raise _source_integrity_error("resumed run has no recorded source state")
        if recorded.get("source_tree_dirty") is not False:
            raise _source_integrity_error("resumed run was created from a dirty source tree")
        if current.get("source_tree_dirty") is not False:
            raise _source_integrity_error("current source tree is dirty")
        if recorded.get("git_head") != current.get("git_head"):
            raise _source_integrity_error(
                "current git HEAD differs from the run's recorded source HEAD"
            )
        recorded_runtime = manifest.get("runtime")
        if not isinstance(recorded_runtime, Mapping):
            raise _source_integrity_error("resumed run has no recorded runtime identity")
        for key, expected in current_runtime.items():
            if recorded_runtime.get(key) != expected:
                raise _source_integrity_error(
                    f"current runtime {key} differs from the run's recorded value"
                )
    elif not args.dry_run and current.get("source_tree_dirty") is not False:
        raise _source_integrity_error("new executable run requires a clean source tree")
    args._execution_integrity_verified = True


def _execution_integrity_error(context: RunContext) -> str | None:
    if not context.enforce_execution_integrity:
        return None
    recorded_source = context.manifest.get("source")
    recorded_runtime = context.manifest.get("runtime")
    if not isinstance(recorded_source, Mapping) or not isinstance(
        recorded_runtime, Mapping
    ):
        return f"{_SOURCE_INTEGRITY_ERROR_CODE}: study provenance is incomplete"
    try:
        current_source = _git_source_state()
        current_runtime = _runtime_identity_snapshot()
    except HarnessError as exc:
        return str(exc)
    if recorded_source.get("source_tree_dirty") is not False:
        return (
            f"{_SOURCE_INTEGRITY_ERROR_CODE}: study was not pinned to a clean "
            "source tree"
        )
    if current_source.get("source_tree_dirty") is not False:
        return f"{_SOURCE_INTEGRITY_ERROR_CODE}: source tree changed during execution"
    if current_source.get("git_head") != recorded_source.get("git_head"):
        return f"{_SOURCE_INTEGRITY_ERROR_CODE}: git HEAD changed during execution"
    for key, expected in current_runtime.items():
        if recorded_runtime.get(key) != expected:
            return (
                f"{_SOURCE_INTEGRITY_ERROR_CODE}: runtime {key} changed during "
                "execution"
            )
    return None


def _assert_execution_integrity(context: RunContext) -> None:
    if error := _execution_integrity_error(context):
        raise _source_integrity_error(error.split(": ", 1)[-1])


def _new_manifest(
    run_id: str,
    config: Mapping[str, Any],
    run_dir: Path,
    *,
    source_state: Mapping[str, Any] | None = None,
    runtime_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    now = utc_now()
    runtime = dict(runtime_identity or _runtime_identity_snapshot())
    runtime.update(
        {
            "mtdata_invocation": [
                str(Path(sys.executable).resolve()),
                "-m",
                "mtdata",
                "--json",
            ],
            "cuda_visible_devices": str(
                config.get("runtime", {}).get("cuda_visible_devices", "0,1")
            ),
            "cuda_visibility_scope": "child_process_only",
            "gpu_device_probe": (
                "forecast catalog only; mtdata has no dedicated GPU inventory CLI command"
            ),
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "harness_version": HARNESS_VERSION,
        "run_id": run_id,
        "symbol": "BTCUSD",
        "status": "active",
        "created_at": now,
        "updated_at": now,
        "source": dict(source_state or _git_source_state()),
        "runtime": runtime,
        "safety": {
            "research_only": True,
            "symbol_allowlist": ["BTCUSD"],
            "trade_command_prefixes_forbidden": list(FORBIDDEN_COMMAND_PREFIXES),
            "shell_command_forbidden": True,
            "subprocess_shell": False,
            "isolated_model_store": "model_store/<stage>/<command>/attempt-N",
            "isolated_jobs_db": "jobs/<stage>/<command>/attempt-N.sqlite",
            "attempt_store_reuse": False,
            "attempt_isolation_stages": [
                "audit",
                "screen",
                "tune",
                "validate",
                "holdout",
            ],
            "lifecycle_store_exempt_stages": ["materialize", "shadow"],
            "lifecycle_model_store": "model_store/lifecycle",
            "lifecycle_jobs_db": "forecast_jobs.sqlite",
            "lifecycle_store_exemption_reason": (
                "shadow require_existing intentionally reads the model materialized "
                "in the shared lifecycle store"
            ),
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
            "lifecycle_model_store": "model_store/lifecycle",
            "jobs": "jobs",
            "lifecycle_jobs_db": "forecast_jobs.sqlite",
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


def _registered_anchor_grids(
    specs: Sequence[CommandSpec],
) -> dict[str, dict[str, Any]]:
    grids: dict[str, dict[str, Any]] = {}
    for spec in specs:
        metadata = spec.metadata
        if metadata.get("anchor_mode") != "explicit":
            continue
        grid_id = metadata.get("anchor_grid_id")
        anchors = metadata.get("expected_anchors")
        expected_hash = metadata.get("expected_anchors_sha256")
        if not isinstance(grid_id, str) or not grid_id:
            raise HarnessError("Explicit-anchor command omitted anchor_grid_id")
        if not isinstance(anchors, list) or not anchors:
            raise HarnessError("Explicit-anchor command omitted expected_anchors")
        actual_hash = _sha256_json(anchors)
        if expected_hash != actual_hash:
            raise HarnessError("Explicit-anchor command has an invalid anchor-grid hash")
        registration = {
            "id": grid_id,
            "sha256": actual_hash,
            "count": len(anchors),
            "anchors": list(anchors),
            "history_start": metadata.get("history_start"),
            "role_window": metadata.get("role_window"),
        }
        prior = grids.get(grid_id)
        if prior is not None and prior != registration:
            raise HarnessError(
                f"Explicit anchor grid {grid_id!r} differs across planned commands"
            )
        grids[grid_id] = registration
    return grids


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
    anchor_grids = _registered_anchor_grids(specs)
    if anchor_grids:
        stage_record["anchor_grids"] = anchor_grids
    else:
        stage_record.pop("anchor_grids", None)
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
    read_only_issue_list = _is_read_only_issue_list(args)
    if read_only_issue_list and not manifest_path.exists():
        raise HarnessError("issue list requires an existing study manifest")
    if manifest_path.exists():
        if not args.resume:
            raise HarnessError(f"Study already exists at {run_dir}; pass --resume to use it")
        loaded_manifest = _load_json(manifest_path)
        if not isinstance(loaded_manifest, Mapping):
            raise HarnessError("Stored manifest must be a JSON object")
        manifest = dict(loaded_manifest)
        run_id = str(manifest.get("run_id") or run_id)
        config = (
            {}
            if read_only_issue_list
            else _resolve_config(Path(args.config) if args.config else None, run_dir)
        )
    else:
        if args.resume:
            raise HarnessError(f"No study manifest exists at {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        config = _resolve_config(Path(args.config) if args.config else None)
        verified_source = getattr(args, "_verified_source_state", None)
        verified_runtime = getattr(args, "_verified_runtime_identity", None)
        manifest = _new_manifest(
            run_id,
            config,
            run_dir,
            source_state=(
                verified_source if isinstance(verified_source, Mapping) else None
            ),
            runtime_identity=(
                verified_runtime if isinstance(verified_runtime, Mapping) else None
            ),
        )
    context = RunContext(
        run_dir=run_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        config=config,
        dry_run=bool(args.dry_run),
        timeout=float(args.timeout),
        max_commands=args.max_commands,
        fail_fast=bool(args.fail_fast),
        enforce_execution_integrity=bool(
            getattr(args, "_execution_integrity_verified", False)
        )
        and not bool(args.dry_run),
    )
    if read_only_issue_list:
        return context
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
    for directory in (
        "raw",
        "normalized",
        "model_store",
        "model_store/lifecycle",
        "jobs",
        "tuning",
        "materializations",
        "config_revisions",
    ):
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    if not (run_dir / "issues.json").exists():
        _atomic_write_json(run_dir / "issues.json", _empty_issue_ledger(run_id))
    _record_config_revision(context)
    _preregister_default_plan(context)
    _save_manifest(context)
    return context


_ATTEMPT_ISOLATION_ERROR_CODE = "BTC-ATTEMPT-ISOLATION"
_ATTEMPT_INVENTORY_MAX_FILES = 256
_ATTEMPT_ISOLATION_EXEMPT_STAGES = frozenset({"materialize", "shadow"})


def _prepare_attempt_isolation(
    context: RunContext,
    stage: str,
    command_id: str,
    attempt: int,
) -> dict[str, Any]:
    stage_slug = _slug(stage)
    command_slug = _slug(command_id)
    model_store = (
        context.run_dir
        / "model_store"
        / stage_slug
        / command_slug
        / f"attempt-{attempt}"
    )
    jobs_db = (
        context.run_dir
        / "jobs"
        / stage_slug
        / command_slug
        / f"attempt-{attempt}.sqlite"
    )
    if model_store.exists() or jobs_db.exists():
        raise HarnessError(
            f"{_ATTEMPT_ISOLATION_ERROR_CODE}: attempt paths already exist for "
            f"{stage}/{command_id} attempt {attempt}; prior attempts are never reused"
        )
    model_store.mkdir(parents=True, exist_ok=False)
    jobs_db.parent.mkdir(parents=True, exist_ok=True)
    if any(model_store.iterdir()) or jobs_db.exists():
        raise HarnessError(
            f"{_ATTEMPT_ISOLATION_ERROR_CODE}: fresh attempt paths contain preexisting files"
        )
    return {
        "attempt": attempt,
        "model_store_path": _relative(model_store, context.run_dir),
        "jobs_db_path": _relative(jobs_db, context.run_dir),
        "preexisting_files": 0,
        "model_store": model_store,
        "jobs_db": jobs_db,
    }


def _attempt_post_call_inventory(
    context: RunContext,
    isolation: Mapping[str, Any],
) -> tuple[dict[str, Any], str | None]:
    model_store = Path(isolation["model_store"])
    jobs_db = Path(isolation["jobs_db"])
    files = sorted(
        (path for path in model_store.rglob("*") if path.is_file()),
        key=lambda path: path.as_posix(),
    )
    job_entries = sorted(
        (
            path
            for path in jobs_db.parent.iterdir()
            if path.name.startswith(jobs_db.name)
        ),
        key=lambda path: path.as_posix(),
    )
    invalid_job_entries = [
        path for path in job_entries if path.is_symlink() or not path.is_file()
    ]
    if invalid_job_entries:
        return (
            {
                "max_files": _ATTEMPT_INVENTORY_MAX_FILES,
                "files_observed": len(files) + len(job_entries),
                "files": [],
                "complete": False,
            },
            f"{_ATTEMPT_ISOLATION_ERROR_CODE}: jobs attempt output contains "
            "a symbolic link or non-file entry",
        )
    files.extend(job_entries)
    if len(files) > _ATTEMPT_INVENTORY_MAX_FILES:
        return (
            {
                "max_files": _ATTEMPT_INVENTORY_MAX_FILES,
                "files_observed": len(files),
                "files": [],
                "complete": False,
            },
            f"{_ATTEMPT_ISOLATION_ERROR_CODE}: post-call attempt inventory has "
            f"{len(files)} files, above the {_ATTEMPT_INVENTORY_MAX_FILES}-file bound",
        )
    entries: list[dict[str, Any]] = []
    try:
        for path in files:
            if path.is_symlink():
                return (
                    {
                        "max_files": _ATTEMPT_INVENTORY_MAX_FILES,
                        "files_observed": len(files),
                        "files": entries,
                        "complete": False,
                    },
                    f"{_ATTEMPT_ISOLATION_ERROR_CODE}: attempt output contains a symbolic link",
                )
            entries.append(
                {
                    "path": _relative(path, context.run_dir),
                    "size": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    except OSError as exc:
        return (
            {
                "max_files": _ATTEMPT_INVENTORY_MAX_FILES,
                "files_observed": len(files),
                "files": entries,
                "complete": False,
            },
            f"{_ATTEMPT_ISOLATION_ERROR_CODE}: could not inventory attempt output: {exc}",
        )
    return (
        {
            "max_files": _ATTEMPT_INVENTORY_MAX_FILES,
            "files_observed": len(files),
            "files": entries,
            "complete": True,
        },
        None,
    )


def _command_environment(
    context: RunContext,
    isolation: Mapping[str, Any] | None,
) -> dict[str, str]:
    environment = os.environ.copy()
    if isolation is None:
        model_store = context.run_dir / "model_store" / "lifecycle"
        jobs_db = context.run_dir / "forecast_jobs.sqlite"
    else:
        model_store = Path(isolation["model_store"])
        jobs_db = Path(isolation["jobs_db"])
    environment["MTDATA_MODEL_STORE"] = str(model_store.resolve())
    environment["MTDATA_FORECAST_JOBS_DB"] = str(jobs_db.resolve())
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


def _planned_raw_path(context: RunContext, stage: str, command_id: str) -> Path:
    return context.run_dir / "raw" / stage / f"{_slug(command_id)}.json"


def _attempt_raw_path(
    context: RunContext,
    stage: str,
    command_id: str,
    attempt: int,
) -> Path:
    return (
        context.run_dir
        / "raw"
        / _slug(stage)
        / _slug(command_id)
        / f"attempt-{attempt}.json"
    )


def _raw_path(context: RunContext, stage: str, command_id: str) -> Path:
    record = (
        context.manifest.get("stages", {})
        .get(stage, {})
        .get("commands", {})
        .get(command_id, {})
    )
    raw_path_text = record.get("raw_path") if isinstance(record, Mapping) else None
    if isinstance(raw_path_text, str) and raw_path_text:
        return context.run_dir / raw_path_text
    return _planned_raw_path(context, stage, command_id)


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


def _screen_features_requested(
    metadata: Mapping[str, Any],
    argv: Sequence[str],
) -> bool:
    if metadata.get("feature_contract_required") is True:
        return True
    if str(metadata.get("kind") or "") != "baseline_screen":
        return False
    try:
        raw_features = argv[argv.index("--features") + 1]
    except (ValueError, IndexError):
        return False
    try:
        parsed_features = json.loads(str(raw_features))
    except json.JSONDecodeError:
        parsed_features = raw_features
    return _has_nonempty_features(parsed_features)


def _strict_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _finite_horizon_vector(value: Any, horizon: int) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != horizon:
        return False
    for item in value:
        if isinstance(item, bool):
            return False
        try:
            numeric = float(item)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(numeric):
            return False
    return True


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _canonical_utc_timestamp(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text_value = value.strip()
    try:
        parsed = datetime.fromisoformat(text_value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        return None
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _semantic_columns_match(
    selected_columns: Sequence[str],
    expected_columns: Sequence[Mapping[str, str]],
) -> bool:
    remaining = [_semantic_feature_token(column) for column in selected_columns]
    for expected in expected_columns:
        token = str(expected.get("semantic_id") or "")
        match_index = next(
            (
                index
                for index, actual in enumerate(remaining)
                if actual == token
            ),
            None,
        )
        if match_index is None:
            return False
        remaining.pop(match_index)
    return not remaining


def _feature_collection_contract_error(
    payload: Mapping[str, Any],
    results: Mapping[str, Any],
    planned_methods: Sequence[str],
    expected_tests: int,
) -> str | None:
    expected_method_count = len(planned_methods)
    expected_anchor_count = expected_method_count * expected_tests
    top_level_counts = {
        "methods_total": expected_method_count,
        "methods_succeeded": expected_method_count,
        "methods_complete": expected_method_count,
        "methods_partial": 0,
        "methods_failed": 0,
        "anchor_tests_planned": expected_anchor_count,
        "anchor_tests_succeeded": expected_anchor_count,
        "anchor_tests_failed": 0,
    }
    for key, expected in top_level_counts.items():
        if _strict_nonnegative_int(payload.get(key)) != expected:
            return (
                f"Feature-bearing screen result reports {key}={payload.get(key)!r}, "
                f"expected {expected}"
            )
    if {str(method) for method in results} != {
        str(method) for method in planned_methods
    }:
        return "Feature-bearing screen result method set differs from the planned methods"
    return None


def _screen_detail_evidence(
    method_name: str,
    detail_row: Any,
    *,
    quantity: str,
    timeframe: str,
    horizon: int,
    lookback: int,
    require_actual_timestamps: bool,
) -> tuple[str | None, str | None, Any]:
    if not isinstance(detail_row, Mapping) or detail_row.get("success") is not True:
        return f"Screen method {method_name!r} has a non-successful detail", None, None
    if _strict_nonnegative_int(detail_row.get("training_bars_used")) != lookback:
        return f"Screen method {method_name!r} used unexpected training bars", None, None
    anchor = _canonical_utc_timestamp(detail_row.get("anchor"))
    if anchor is None:
        return f"Screen method {method_name!r} has an invalid UTC anchor", None, None
    if quantity == "volatility":
        if not _finite_number(detail_row.get("forecast_sigma")):
            return f"Screen method {method_name!r} has an invalid forecast value", None, None
        if not _finite_number(detail_row.get("realized_sigma")):
            return f"Screen method {method_name!r} has an invalid actual value", None, None
        return None, anchor, float(detail_row["realized_sigma"])
    if not _finite_horizon_vector(detail_row.get("forecast"), horizon):
        return f"Screen method {method_name!r} has an invalid forecast path", None, None
    if not _finite_horizon_vector(detail_row.get("actual"), horizon):
        return f"Screen method {method_name!r} has an invalid actual path", None, None
    actual = [float(item) for item in detail_row["actual"]]
    if not require_actual_timestamps:
        return None, anchor, actual
    timeframe_minutes = _TIMEFRAME_MINUTES.get(timeframe.upper())
    if timeframe_minutes is None:
        return f"Screen method {method_name!r} has an unsupported timeframe", None, None
    raw_timestamps = detail_row.get("actual_timestamps")
    if not isinstance(raw_timestamps, list) or len(raw_timestamps) != horizon:
        return (
            f"Screen method {method_name!r} omitted full actual timestamps",
            None,
            None,
        )
    timestamps = [_canonical_utc_timestamp(value) for value in raw_timestamps]
    if any(value is None for value in timestamps):
        return f"Screen method {method_name!r} has invalid actual timestamps", None, None
    anchor_at = datetime.fromisoformat(anchor.replace("Z", "+00:00"))
    expected_timestamps = [
        (
            anchor_at + timedelta(minutes=timeframe_minutes * step)
        ).isoformat(timespec="seconds").replace("+00:00", "Z")
        for step in range(1, horizon + 1)
    ]
    if timestamps != expected_timestamps:
        return (
            f"Screen method {method_name!r} has shifted or gapped actual timestamps",
            None,
            None,
        )
    return None, anchor, {"values": actual, "timestamps": timestamps}


def _screen_anchor_grid_error(
    metadata: Mapping[str, Any],
    method_name: str,
    anchors: Sequence[str],
    horizon: int,
) -> str | None:
    if len(set(anchors)) != len(anchors):
        return f"Screen method {method_name!r} contains duplicate anchors"
    timeframe = str(metadata.get("timeframe") or "")
    spacing = _strict_nonnegative_int(metadata.get("spacing"))
    timeframe_minutes = _TIMEFRAME_MINUTES.get(timeframe)
    parsed = [datetime.fromisoformat(value.replace("Z", "+00:00")) for value in anchors]
    if len(parsed) > 1 and spacing and timeframe_minutes:
        expected_delta = timedelta(minutes=spacing * timeframe_minutes)
        if any(
            current - previous != expected_delta
            for previous, current in zip(parsed, parsed[1:])
        ):
            return f"Screen method {method_name!r} has a shifted or gapped anchor grid"
    shard = metadata.get("shard")
    if isinstance(shard, Mapping):
        try:
            shard_start = datetime.combine(
                date.fromisoformat(str(shard["start"])),
                datetime.min.time(),
                tzinfo=timezone.utc,
            )
            shard_limit = datetime.combine(
                date.fromisoformat(str(shard["end"])) + timedelta(days=1),
                datetime.min.time(),
                tzinfo=timezone.utc,
            )
        except (KeyError, ValueError):
            return "Screen command has malformed shard bounds"
        target_delta = timedelta(minutes=horizon * (timeframe_minutes or 0))
        if any(anchor < shard_start or anchor + target_delta >= shard_limit for anchor in parsed):
            return (
                f"Screen method {method_name!r} has an anchor or target outside "
                "its registered shard"
            )
    expected_anchors = metadata.get("expected_anchors")
    if expected_anchors is not None and (
        not isinstance(expected_anchors, list) or list(anchors) != expected_anchors
    ):
        return f"Screen method {method_name!r} differs from the frozen anchor grid"
    return None


def _screen_method_evidence(
    metadata: Mapping[str, Any],
    method_name: str,
    result: Any,
    *,
    expected_tests: int,
    horizon: int,
    lookback: int,
) -> tuple[str | None, list[str], list[Any]]:
    if not isinstance(result, Mapping):
        return f"Screen result omitted planned method {method_name!r}", [], []
    complete = (
        result.get("success") is True
        and result.get("complete_success") is True
        and result.get("status") == "complete"
        and _strict_nonnegative_int(result.get("num_tests")) == expected_tests
        and _strict_nonnegative_int(result.get("successful_tests")) == expected_tests
    )
    if not complete:
        return f"Screen method {method_name!r} was not complete on every anchor", [], []
    details = result.get("details")
    if not isinstance(details, list) or len(details) != expected_tests:
        return f"Screen method {method_name!r} omitted full anchor details", [], []
    anchors: list[str] = []
    actuals: list[Any] = []
    for detail_row in details:
        error, anchor, actual = _screen_detail_evidence(
            method_name,
            detail_row,
            quantity=str(metadata.get("quantity") or "price"),
            timeframe=str(metadata.get("timeframe") or ""),
            horizon=horizon,
            lookback=lookback,
            require_actual_timestamps=metadata.get("anchor_mode") == "explicit",
        )
        if error or anchor is None:
            return error or "Screen detail evidence is incomplete", [], []
        anchors.append(anchor)
        actuals.append(actual)
    return _screen_anchor_grid_error(metadata, method_name, anchors, horizon), anchors, actuals


def _explicit_screen_plan_error(
    metadata: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> str | None:
    if metadata.get("anchor_mode") != "explicit":
        return None
    expected_anchors = metadata.get("expected_anchors")
    expected_hash = metadata.get("expected_anchors_sha256")
    if not isinstance(expected_anchors, list) or not expected_anchors:
        return "Explicit screen metadata omitted its frozen anchor grid"
    if expected_hash != _sha256_json(expected_anchors):
        return "Explicit screen metadata has an invalid anchor-grid hash"
    plan = payload.get("backtest_plan")
    if not isinstance(plan, Mapping):
        return "Explicit screen result omitted backtest_plan"
    if plan.get("anchor_mode") != "explicit":
        return "Explicit screen result did not report anchor_mode=explicit"
    if plan.get("anchor_resolution") != "exact_bar_open":
        return "Explicit screen result did not attest exact_bar_open resolution"
    if plan.get("requested_anchors") != expected_anchors:
        return "Explicit screen requested anchors differ from the frozen grid"
    if plan.get("resolved_anchors") != expected_anchors:
        return "Explicit screen resolved anchors differ from the frozen grid"
    expected_count = len(expected_anchors)
    if (
        _strict_nonnegative_int(plan.get("runs_requested")) != expected_count
        or _strict_nonnegative_int(plan.get("runs_used")) != expected_count
    ):
        return "Explicit screen result has inconsistent requested/resolved anchor counts"
    role_window = metadata.get("role_window")
    horizon = _strict_nonnegative_int(metadata.get("horizon"))
    timeframe_minutes = _TIMEFRAME_MINUTES.get(str(metadata.get("timeframe") or "").upper())
    if (
        not isinstance(role_window, Mapping)
        or horizon is None
        or timeframe_minutes is None
    ):
        return "Explicit screen metadata has invalid role-window timing"
    try:
        role_end_limit = datetime.combine(
            date.fromisoformat(str(role_window["end"])) + timedelta(days=1),
            datetime.min.time(),
            tzinfo=timezone.utc,
        )
        last_anchor = datetime.fromisoformat(
            str(expected_anchors[-1]).replace("Z", "+00:00")
        )
    except (KeyError, ValueError):
        return "Explicit screen metadata has malformed role-window timing"
    if last_anchor + timedelta(minutes=horizon * timeframe_minutes) >= role_end_limit:
        return "Explicit screen final target falls outside its registered role window"
    return None


def _screen_collection_contract_error(
    metadata: Mapping[str, Any],
    payload: Any,
) -> str | None:
    if str(metadata.get("kind") or "") != "baseline_screen":
        return None
    if not isinstance(payload, Mapping):
        return "Screen result is not a JSON object"
    if payload.get("complete_success") is not True or payload.get("status") != "complete":
        return "Screen result did not report complete_success/status=complete"
    explicit_plan_error = _explicit_screen_plan_error(metadata, payload)
    if explicit_plan_error:
        return explicit_plan_error
    results = payload.get("results")
    planned_methods = metadata.get("methods")
    expected_tests = _strict_nonnegative_int(metadata.get("steps"))
    horizon = _strict_nonnegative_int(metadata.get("horizon"))
    lookback = _strict_nonnegative_int(metadata.get("lookback"))
    if not isinstance(results, Mapping):
        return "Screen result omitted per-method results"
    if not isinstance(planned_methods, list) or not planned_methods:
        return "Screen command omitted its planned method list"
    if expected_tests is None or expected_tests < 1:
        return "Screen command has an invalid planned anchor count"
    if horizon is None or horizon < 1 or lookback is None or lookback < 1:
        return "Screen command has invalid horizon/lookback metadata"
    collection_error = _feature_collection_contract_error(
        payload, results, planned_methods, expected_tests
    )
    if collection_error:
        return collection_error.replace("Feature-bearing screen", "Screen")
    reference: tuple[list[str], list[Any]] | None = None
    for method in planned_methods:
        error, anchors, actuals = _screen_method_evidence(
            metadata,
            str(method),
            results.get(str(method)),
            expected_tests=expected_tests,
            horizon=horizon,
            lookback=lookback,
        )
        if error:
            return error
        if reference is None:
            reference = anchors, actuals
        elif reference != (anchors, actuals):
            return "Screen methods do not share identical anchor/actual evidence"
    return None


def _feature_usage_contract_error(
    method_name: str,
    usage: Any,
    *,
    num_tests: int | None,
    successful_tests: int | None,
    expected_tests: int,
    horizon: int,
    expectations: Any,
) -> str | None:
    if not isinstance(usage, Mapping):
        return f"Feature-bearing screen method {method_name!r} omitted feature_usage"
    anchors_verified = _strict_nonnegative_int(usage.get("anchors_verified"))
    future_rows = _strict_nonnegative_int(usage.get("future_rows"))
    n_features = _strict_nonnegative_int(usage.get("n_features"))
    if usage.get("status") != "consumed":
        return f"Feature-bearing screen method {method_name!r} did not attest status=consumed"
    if usage.get("historical_consumed") is not True:
        return f"Feature-bearing screen method {method_name!r} did not consume historical features"
    if usage.get("future_consumed") is not True:
        return f"Feature-bearing screen method {method_name!r} did not consume future features"
    if not (anchors_verified == num_tests == successful_tests == expected_tests):
        return (
            f"Feature-bearing screen method {method_name!r} has inconsistent verified, "
            "planned, or successful anchor counts"
        )
    if future_rows != horizon:
        return (
            f"Feature-bearing screen method {method_name!r} reports future_rows="
            f"{future_rows!r}, expected horizon={horizon}"
        )
    if not isinstance(expectations, Mapping):
        return "Feature-bearing screen command omitted semantic feature expectations"
    if expectations.get("complete") is not True:
        return "Feature-bearing screen feature expectations are incomplete"
    expected_n_features = _strict_nonnegative_int(expectations.get("n_features"))
    if n_features != expected_n_features:
        return (
            f"Feature-bearing screen method {method_name!r} reports n_features="
            f"{n_features!r}, expected {expected_n_features} from the variant"
        )
    selected_columns = usage.get("selected_columns")
    expected_columns = expectations.get("semantic_columns")
    if (
        not isinstance(selected_columns, list)
        or not isinstance(expected_columns, list)
        or not _semantic_columns_match(selected_columns, expected_columns)
    ):
        return (
            f"Feature-bearing screen method {method_name!r} selected columns "
            "do not match the variant's semantic feature set"
        )
    if expectations.get("has_observed_features") is not True:
        return None
    if expectations.get("observed_future_policy") != "carry_forward":
        return "Observed screen features must preregister carry_forward policy"
    if usage.get("observed_feature_lag_bars") != 1:
        return (
            f"Feature-bearing screen method {method_name!r} did not attest "
            "a one-bar observed feature lag"
        )
    if usage.get("observed_future_policy") != "carry_forward":
        return (
            f"Feature-bearing screen method {method_name!r} did not attest "
            "carry_forward observed features"
        )
    return None


def _feature_detail_contract_error(
    method_name: str,
    details: Any,
    *,
    detail_mode: str,
    successful_tests: int | None,
    lookback: int,
    horizon: int,
    method_usage: Mapping[str, Any],
    expected_params: Mapping[str, Any] | None,
) -> str | None:
    if detail_mode != "full":
        return (
            f"Feature-bearing screen method {method_name!r} was not requested "
            "with detail=full"
        )
    if not isinstance(details, list):
        return f"Feature-bearing screen method {method_name!r} omitted full details"
    if len(details) != successful_tests:
        return (
            f"Feature-bearing screen method {method_name!r} detail count differs "
            "from successful_tests"
        )
    for detail_row in details:
        if not isinstance(detail_row, Mapping) or detail_row.get("success") is not True:
            return f"Feature-bearing screen method {method_name!r} has a non-successful detail"
        if _strict_nonnegative_int(detail_row.get("training_bars_used")) != lookback:
            return (
                f"Feature-bearing screen method {method_name!r} used an unexpected "
                "number of training bars"
            )
        if not _finite_horizon_vector(detail_row.get("forecast"), horizon):
            return f"Feature-bearing screen method {method_name!r} has an invalid forecast path"
        if not _finite_horizon_vector(detail_row.get("actual"), horizon):
            return f"Feature-bearing screen method {method_name!r} has an invalid actual path"
        anchor_usage = detail_row.get("feature_usage")
        if not isinstance(anchor_usage, Mapping):
            return (
                f"Feature-bearing screen method {method_name!r} omitted per-anchor "
                "feature_usage"
            )
        for key in (
            "status",
            "historical_consumed",
            "future_consumed",
            "future_rows",
            "n_features",
            "selected_columns",
            "observed_feature_lag_bars",
            "observed_future_policy",
        ):
            if method_usage.get(key) != anchor_usage.get(key):
                return (
                    f"Feature-bearing screen method {method_name!r} has inconsistent "
                    f"per-anchor feature_usage.{key}"
                )
        historical_rows = _strict_nonnegative_int(anchor_usage.get("historical_rows"))
        historical_min = _strict_nonnegative_int(method_usage.get("historical_rows_min"))
        historical_max = _strict_nonnegative_int(method_usage.get("historical_rows_max"))
        if (
            historical_rows is None
            or historical_min is None
            or historical_max is None
            or not historical_min <= historical_rows <= historical_max
        ):
            return (
                f"Feature-bearing screen method {method_name!r} has inconsistent "
                "per-anchor historical feature rows"
            )
        params_used = detail_row.get("params_used")
        if not isinstance(params_used, Mapping):
            return f"Feature-bearing screen method {method_name!r} omitted params_used"
        if isinstance(expected_params, Mapping):
            for key, expected_value in expected_params.items():
                if params_used.get(key) != expected_value:
                    return (
                        f"Feature-bearing screen method {method_name!r} did not use "
                        f"the preregistered parameter {key!r}"
                    )
    return None


def _screen_feature_contract_error(
    metadata: Mapping[str, Any],
    argv: Sequence[str],
    payload: Any,
) -> str | None:
    if not _screen_features_requested(metadata, argv):
        return None
    if not isinstance(payload, Mapping):
        return "Feature-bearing screen result is not a JSON object"
    if payload.get("complete_success") is not True or payload.get("status") != "complete":
        return "Feature-bearing screen result did not report complete_success/status=complete"
    results = payload.get("results")
    if not isinstance(results, Mapping):
        return "Feature-bearing screen result omitted per-method results"
    planned_methods = metadata.get("methods")
    if not isinstance(planned_methods, list) or not planned_methods:
        return "Feature-bearing screen command omitted its planned method list"
    horizon = _strict_nonnegative_int(metadata.get("horizon"))
    expected_tests = _strict_nonnegative_int(metadata.get("steps"))
    lookback = _strict_nonnegative_int(metadata.get("lookback"))
    if (
        horizon is None
        or horizon < 1
        or expected_tests is None
        or expected_tests < 1
        or lookback is None
        or lookback < 1
    ):
        return "Feature-bearing screen command has invalid horizon/lookback/anchor counts"
    collection_error = _feature_collection_contract_error(
        payload,
        results,
        planned_methods,
        expected_tests,
    )
    if collection_error:
        return collection_error
    expectations = metadata.get("feature_expectations")
    raw_params_per_method = metadata.get("params_per_method")
    params_per_method = (
        raw_params_per_method if isinstance(raw_params_per_method, Mapping) else {}
    )
    global_params = metadata.get("params")
    for method in planned_methods:
        method_name = str(method)
        result = results.get(method_name)
        if not isinstance(result, Mapping):
            return f"Feature-bearing screen result omitted planned method {method_name!r}"
        if (
            result.get("success") is not True
            or result.get("complete_success") is not True
            or result.get("status") != "complete"
        ):
            return f"Feature-bearing screen method {method_name!r} was not complete"
        num_tests = _strict_nonnegative_int(result.get("num_tests"))
        successful_tests = _strict_nonnegative_int(result.get("successful_tests"))
        usage_error = _feature_usage_contract_error(
            method_name,
            result.get("feature_usage"),
            num_tests=num_tests,
            successful_tests=successful_tests,
            expected_tests=expected_tests,
            horizon=horizon,
            expectations=expectations,
        )
        if usage_error:
            return usage_error
        detail_error = _feature_detail_contract_error(
            method_name,
            result.get("details"),
            detail_mode=str(metadata.get("detail") or "full"),
            successful_tests=successful_tests,
            lookback=lookback,
            horizon=horizon,
            method_usage=result["feature_usage"],
            expected_params=(
                params_per_method.get(method_name)
                if isinstance(params_per_method.get(method_name), Mapping)
                else global_params
                if isinstance(global_params, Mapping)
                else None
            ),
        )
        if detail_error:
            return detail_error
    return None


_FEATURE_CONSUMPTION_ERROR_CODE = "BTC-FEATURE-CONSUMPTION"
_SCREEN_EVIDENCE_ERROR_CODE = "BTC-SCREEN-EVIDENCE"
_RAW_INTEGRITY_ERROR_CODE = "BTC-RAW-INTEGRITY"


def _verified_recorded_raw_path(
    context: RunContext,
    stage: str,
    command_id: str,
    record: Mapping[str, Any],
) -> Path:
    raw_path_text = record.get("raw_path")
    expected_hash = record.get("raw_sha256")
    if not isinstance(raw_path_text, str) or not raw_path_text:
        raise HarnessError(
            f"{_RAW_INTEGRITY_ERROR_CODE}: recorded command "
            f"{stage}/{command_id} has no raw_path"
        )
    if not isinstance(expected_hash, str) or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
        raise HarnessError(
            f"{_RAW_INTEGRITY_ERROR_CODE}: recorded command "
            f"{stage}/{command_id} has no valid raw_sha256"
        )
    raw_path = (context.run_dir / raw_path_text).resolve()
    try:
        raw_path.relative_to(context.run_dir.resolve())
    except ValueError as exc:
        raise HarnessError(
            f"{_RAW_INTEGRITY_ERROR_CODE}: raw_path for {stage}/{command_id} "
            "escapes the study directory"
        ) from exc
    if not raw_path.is_file():
        raise HarnessError(
            f"{_RAW_INTEGRITY_ERROR_CODE}: raw output for recorded command "
            f"{stage}/{command_id} is missing"
        )
    actual_hash = _sha256_file(raw_path)
    if actual_hash != expected_hash:
        raise HarnessError(
            f"{_RAW_INTEGRITY_ERROR_CODE}: raw SHA-256 mismatch for completed "
            f"command {stage}/{command_id}"
        )
    return raw_path


def _verify_command_raw_history(
    context: RunContext,
    stage: str,
    command_id: str,
    record: Mapping[str, Any],
) -> Path:
    current_path = _verified_recorded_raw_path(
        context,
        stage,
        command_id,
        record,
    )
    attempts = record.get("attempt_artifacts")
    if not isinstance(attempts, list) or not attempts:
        raise HarnessError(
            f"{_RAW_INTEGRITY_ERROR_CODE}: recorded command {stage}/{command_id} "
            "has no immutable attempt history"
        )
    observed_paths: set[str] = set()
    for index, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt, Mapping):
            raise HarnessError(
                f"{_RAW_INTEGRITY_ERROR_CODE}: attempt history entry {index} for "
                f"{stage}/{command_id} is malformed"
            )
        attempt_path = _verified_recorded_raw_path(
            context,
            stage,
            f"{command_id} attempt {index}",
            attempt,
        )
        relative_path = _relative(attempt_path, context.run_dir)
        if relative_path in observed_paths:
            raise HarnessError(
                f"{_RAW_INTEGRITY_ERROR_CODE}: attempt history for "
                f"{stage}/{command_id} reuses a raw path"
            )
        observed_paths.add(relative_path)
    latest = attempts[-1]
    if (
        latest.get("raw_path") != record.get("raw_path")
        or latest.get("raw_sha256") != record.get("raw_sha256")
    ):
        raise HarnessError(
            f"{_RAW_INTEGRITY_ERROR_CODE}: current raw pointer for "
            f"{stage}/{command_id} does not match its latest attempt"
        )
    return current_path


def _recorded_success_contract_error(
    stage: str,
    command_id: str,
    record: Mapping[str, Any],
    envelope: Any,
) -> str | None:
    if not isinstance(envelope, Mapping):
        return "Recorded raw envelope is not a JSON object"
    if envelope.get("status") != "completed":
        return "Recorded completed command has a non-completed raw envelope"
    if envelope.get("stage") != stage or envelope.get("command_id") != command_id:
        return "Recorded raw envelope stage/command identity does not match the manifest"
    invocation = envelope.get("invocation")
    argv = record.get("argv")
    if (
        not isinstance(invocation, list)
        or not isinstance(argv, list)
        or invocation[-len(argv) :] != argv
    ):
        return "Recorded raw invocation does not match the manifest command"
    if envelope.get("metadata") != record.get("metadata", {}):
        return "Recorded raw metadata does not match the manifest command"
    returncode = _strict_nonnegative_int(envelope.get("returncode"))
    payload = envelope.get("payload")
    if returncode != 0 or not _payload_succeeded(payload, returncode):
        return "Recorded completed command does not contain a successful payload"
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        return "Recorded command metadata is malformed"
    spec = CommandSpec(
        command_id,
        tuple(str(item) for item in argv),
        metadata,
    )
    shard_error = _shard_contract_error(spec, payload)
    if shard_error:
        return shard_error
    screen_error = (
        _screen_collection_contract_error(metadata, payload)
        if stage == "screen"
        else None
    )
    if screen_error:
        return f"{_SCREEN_EVIDENCE_ERROR_CODE}: {screen_error}"
    feature_error = (
        _screen_feature_contract_error(metadata, spec.argv, payload)
        if stage == "screen"
        else None
    )
    if feature_error:
        return f"{_FEATURE_CONSUMPTION_ERROR_CODE}: {feature_error}"
    return _explicit_anchor_contract_error(payload)


def execute_spec(context: RunContext, stage: str, spec: CommandSpec) -> bool | None:
    """Execute one registered command; return True, False, or None for dry-run."""
    _validate_research_command(spec.argv)
    _assert_execution_integrity(context)
    stage_record = context.manifest["stages"][stage]
    command_record = stage_record["commands"][spec.command_id]
    prior_status = command_record.get("status")
    if prior_status == "running":
        raise HarnessError(
            f"{_RAW_INTEGRITY_ERROR_CODE}: command {stage}/{spec.command_id} has "
            "an interrupted attempt without a finalized raw envelope; start a new run"
        )
    if prior_status in {"completed", "failed"}:
        _verify_command_raw_history(
            context,
            stage,
            spec.command_id,
            command_record,
        )
    if prior_status == "completed":
        return True
    invocation = [sys.executable, "-m", "mtdata", "--json", *spec.argv]
    known_secrets = _secret_values()
    if context.dry_run:
        raw_path = _planned_raw_path(context, stage, spec.command_id)
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

    attempt = int(command_record.get("attempts", 0)) + 1
    raw_path = _attempt_raw_path(context, stage, spec.command_id, attempt)
    if raw_path.exists():
        raise HarnessError(
            f"{_RAW_INTEGRITY_ERROR_CODE}: immutable raw attempt path already "
            f"exists for {stage}/{spec.command_id} attempt {attempt}"
        )
    isolation = (
        _prepare_attempt_isolation(
            context,
            stage,
            spec.command_id,
            attempt,
        )
        if stage not in _ATTEMPT_ISOLATION_EXEMPT_STAGES
        else None
    )
    if isolation is None:
        isolation_record: dict[str, Any] = {
            "attempt": attempt,
            "policy": "shared_materialize_shadow_lifecycle_store",
            "model_store_path": "model_store/lifecycle",
            "jobs_db_path": "forecast_jobs.sqlite",
            "attempt_isolation_exempt": True,
        }
    else:
        isolation_record = {
            "attempt": attempt,
            "policy": "fresh_command_attempt",
            "model_store_path": isolation["model_store_path"],
            "jobs_db_path": isolation["jobs_db_path"],
            "preexisting_files": isolation["preexisting_files"],
            "attempt_isolation_exempt": False,
        }
    command_record["status"] = "running"
    command_record["attempts"] = attempt
    command_record["started_at"] = utc_now()
    command_record["model_store_path"] = isolation_record["model_store_path"]
    command_record["jobs_db_path"] = isolation_record["jobs_db_path"]
    attempt_artifacts = command_record.setdefault("attempt_artifacts", [])
    if not isinstance(attempt_artifacts, list):
        raise HarnessError(
            f"{_ATTEMPT_ISOLATION_ERROR_CODE}: command attempt history is malformed"
        )
    attempt_artifacts.append(isolation_record)
    _save_manifest(context)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            invocation,
            cwd=Path(__file__).resolve().parents[1],
            env=_command_environment(context, isolation),
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
    if isolation is None:
        post_call_inventory = {
            "complete": False,
            "exempt": True,
            "reason": "shared_materialize_shadow_lifecycle_store",
        }
        inventory_error = None
    else:
        post_call_inventory, inventory_error = _attempt_post_call_inventory(
            context,
            isolation,
        )
    post_execution_integrity_error = _execution_integrity_error(context)
    isolation_record["post_call_inventory"] = post_call_inventory
    if inventory_error:
        isolation_record["inventory_error"] = inventory_error
    finished = utc_now()
    contract_error = post_execution_integrity_error
    if parse_error is None and contract_error is None:
        screen_contract_error = (
            _screen_collection_contract_error(spec.metadata, payload)
            if stage == "screen"
            else None
        )
        feature_contract_error = (
            _screen_feature_contract_error(spec.metadata, spec.argv, payload)
            if stage == "screen"
            else None
        )
        contract_error = (
            inventory_error
            or _shard_contract_error(spec, payload)
            or (
                f"{_SCREEN_EVIDENCE_ERROR_CODE}: {screen_contract_error}"
                if screen_contract_error
                else None
            )
            or (
                f"{_FEATURE_CONSUMPTION_ERROR_CODE}: {feature_contract_error}"
                if feature_contract_error
                else None
            )
            or _explicit_anchor_contract_error(payload)
        )
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
        "attempt_isolation": isolation_record,
    }
    if parse_error:
        envelope["parse_error"] = parse_error
        envelope["stdout"] = _redact_text(stdout, known_secrets)
    if contract_error:
        envelope["contract_error"] = contract_error
    _atomic_write_json(raw_path, envelope)
    raw_sha256 = _sha256_file(raw_path)
    isolation_record["raw_path"] = _relative(raw_path, context.run_dir)
    isolation_record["raw_sha256"] = raw_sha256
    _record_anchor_contract_issue(context, stage, spec, safe_payload, raw_path)
    command_record.update(
        {
            "status": envelope["status"],
            "finished_at": finished,
            "duration_seconds": envelope["duration_seconds"],
            "returncode": returncode,
            "raw_path": _relative(raw_path, context.run_dir),
            "raw_sha256": raw_sha256,
            "post_call_inventory": post_call_inventory,
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
            "raw_sha256": record.get("raw_sha256"),
            "model_store_path": record.get("model_store_path"),
            "jobs_db_path": record.get("jobs_db_path"),
            "post_call_inventory": record.get("post_call_inventory"),
        }
        raw_path_text = record.get("raw_path")
        if raw_path_text:
            raw_path = (
                _verify_command_raw_history(context, stage, command_id, record)
                if record.get("status") in {"completed", "failed"}
                else context.run_dir / str(raw_path_text)
            )
            if raw_path.exists():
                envelope = _load_json(raw_path)
                if isinstance(envelope, Mapping):
                    row["returncode"] = envelope.get("returncode")
                    row["duration_seconds"] = envelope.get("duration_seconds")
                    if envelope.get("parse_error"):
                        row["parse_error"] = envelope.get("parse_error")
                    if envelope.get("contract_error"):
                        row["contract_error"] = envelope.get("contract_error")
                    payload = envelope.get("payload")
                    recorded_contract_error = (
                        _recorded_success_contract_error(
                            stage,
                            command_id,
                            record,
                            envelope,
                        )
                        if record.get("status") == "completed"
                        else None
                    )
                    if recorded_contract_error:
                        row["status"] = "failed"
                        row["contract_error"] = recorded_contract_error
                        if isinstance(record, dict):
                            record["status"] = "failed"
                            record["error"] = recorded_contract_error
                    row["result"] = _summarize_payload(payload)
        rows.append(row)
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    normalized_status = stage_record.get("status")
    if any(row.get("status") == "failed" for row in rows):
        normalized_status = "failed"
        stage_record["status"] = "failed"
    normalized = {
        "schema_version": SCHEMA_VERSION,
        "run_id": context.manifest.get("run_id"),
        "stage": stage,
        "generated_at": utc_now(),
        "status": normalized_status,
        "config_hash": stage_record.get("config_hash"),
        "plan_digest": stage_record.get("plan_digest"),
        "anchor_grids": stage_record.get("anchor_grids", {}),
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
    if not specs:
        raise HarnessError(f"Stage {stage!r} produced an empty command plan")
    _register_specs(
        context,
        stage,
        specs,
        extra_hash_input=extra_hash_input,
        allow_append=allow_append,
    )
    stage_record = context.manifest["stages"][stage]
    for command_id, record in stage_record.get("commands", {}).items():
        if isinstance(record, Mapping) and record.get("status") in {
            "completed",
            "failed",
        }:
            raw_path = _verify_command_raw_history(
                context,
                stage,
                str(command_id),
                record,
            )
            if record.get("status") == "completed":
                stored_error = _recorded_success_contract_error(
                    stage,
                    str(command_id),
                    record,
                    _load_json(raw_path),
                )
                if stored_error:
                    if isinstance(record, dict):
                        record["status"] = "failed"
                        record["error"] = stored_error
                    stage_record["status"] = "failed"
                    write_normalized_stage(context, stage)
                    _save_manifest(context)
                    raise HarnessError(
                        f"Recorded command {stage}/{command_id} failed "
                        f"revalidation: {stored_error}"
                    )
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
        _enforce_source_integrity(args)
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
