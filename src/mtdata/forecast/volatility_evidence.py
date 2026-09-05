"""Deterministic, bounded evidence for volatility model inputs.

The public volatility tools expose these summaries only in full-detail output.
They intentionally fingerprint source prices instead of returning raw price
vectors, while still making the exact effective calculation inputs auditable.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

VOLATILITY_INPUT_EVIDENCE_SCHEMA_VERSION = 1
VOLATILITY_DIGEST_ALGORITHM = "sha256"
VOLATILITY_DIGEST_ENCODING = "canonical_float64_le_v1"


def _canonical_float64_array(values: Any) -> np.ndarray:
    """Return C-contiguous little-endian floats with stable NaN/zero bits."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        array = array.reshape(1)
    array = np.array(array, dtype=np.float64, order="C", copy=True)
    if array.size:
        array[array == 0.0] = 0.0
        array[np.isnan(array)] = np.nan
    return np.asarray(array, dtype="<f8", order="C")


def volatility_array_sha256(
    values: Any,
    *,
    domain: str,
    context: dict[str, Any] | None = None,
) -> str:
    """Hash a numeric array with an explicit domain, shape, and encoding."""
    if not str(domain).strip():
        raise ValueError("volatility evidence digest domain must not be empty")
    array = _canonical_float64_array(values)
    header_payload = {
        "schema_version": VOLATILITY_INPUT_EVIDENCE_SCHEMA_VERSION,
        "digest_algorithm": VOLATILITY_DIGEST_ALGORITHM,
        "domain": str(domain),
        "dtype": "float64",
        "encoding": VOLATILITY_DIGEST_ENCODING,
        "shape": [int(size) for size in array.shape],
    }
    if context:
        header_payload["context"] = context
    header = json.dumps(
        header_payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\n")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def source_positions_for_returns(return_positions: Any) -> np.ndarray:
    """Map diff-result positions to the unique price rows they depend on."""
    raw_positions = np.asarray(return_positions).reshape(-1)
    if raw_positions.size == 0:
        return np.asarray([], dtype=np.int64)
    try:
        numeric_positions = raw_positions.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("return positions must be finite integers") from exc
    if not bool(np.all(np.isfinite(numeric_positions))) or not bool(
        np.all(numeric_positions == np.floor(numeric_positions))
    ):
        raise ValueError("return positions must be finite integers")
    positions = numeric_positions.astype(np.int64)
    if int(positions[0]) < 0 or bool(np.any(np.diff(positions) <= 0)):
        raise ValueError("return positions must be nonnegative and strictly increasing")
    return np.unique(np.concatenate((positions, positions + 1)))


def _numeric_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame.columns:
        return np.full(len(frame), np.nan, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)


def _timestamp_bound(value: float) -> str | None:
    if not math.isfinite(float(value)):
        return None
    try:
        return (
            datetime.fromtimestamp(float(value), tz=timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
    except (OverflowError, TypeError, ValueError):
        return None


def _array_descriptor(
    values: Any,
    *,
    domain: str,
    operation: str,
    fields: Sequence[str],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    array = _canonical_float64_array(values)
    ordered_fields = [str(field) for field in fields]
    if not str(domain).strip():
        raise ValueError("volatility evidence digest domain must not be empty")
    if not str(operation).strip():
        raise ValueError("volatility evidence operation must not be empty")
    if (
        not ordered_fields
        or any(not field.strip() for field in ordered_fields)
        or len(ordered_fields) != len(set(ordered_fields))
    ):
        raise ValueError("volatility evidence fields must be nonempty and unique")
    if array.ndim == 1 and len(ordered_fields) != 1:
        raise ValueError(
            "one-dimensional volatility evidence requires exactly one field"
        )
    if array.ndim == 2 and len(ordered_fields) != int(array.shape[1]):
        raise ValueError(
            "two-dimensional volatility evidence fields must match its columns"
        )
    if array.ndim not in {1, 2}:
        raise ValueError("volatility evidence arrays must be one- or two-dimensional")
    digest_context = {
        **dict(context or {}),
        "operation": str(operation),
        "fields": ordered_fields,
    }
    return {
        "operation": str(operation),
        "fields": ordered_fields,
        "shape": [int(size) for size in array.shape],
        "count": int(array.shape[0]) if array.ndim else int(array.size),
        "sha256": volatility_array_sha256(
            array,
            domain=domain,
            context=digest_context,
        ),
    }


def build_volatility_input_evidence(
    frame: pd.DataFrame,
    *,
    method: str,
    timeframe: str,
    operation: str,
    value_columns: Sequence[str],
    raw_value_columns: Sequence[str] | None = None,
    raw_source_columns: Sequence[str] | None = None,
    source_positions: Iterable[int] | np.ndarray,
    returns: Any,
    return_start_timestamps: Any,
    return_timestamps: Any,
    return_operation: str,
    return_timestamp_policy: str,
    transformed_input: Any,
    transformed_fields: Sequence[str],
    transformed_operation: str,
) -> dict[str, Any]:
    """Describe and fingerprint exactly the rows and arrays a method used."""
    if not str(method).strip() or not str(timeframe).strip():
        raise ValueError("volatility evidence method and timeframe are required")
    if (
        not str(operation).strip()
        or not str(return_operation).strip()
        or not str(return_timestamp_policy).strip()
    ):
        raise ValueError("volatility evidence operations must not be empty")
    raw_positions = np.asarray(list(source_positions)).reshape(-1)
    try:
        numeric_positions = raw_positions.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "volatility evidence source positions must be finite integers"
        ) from exc
    if numeric_positions.size and (
        not bool(np.all(np.isfinite(numeric_positions)))
        or not bool(np.all(numeric_positions == np.floor(numeric_positions)))
    ):
        raise ValueError("volatility evidence source positions must be finite integers")
    positions = numeric_positions.astype(np.int64)
    if positions.size:
        if bool(np.any(np.diff(positions) <= 0)):
            raise ValueError(
                "volatility evidence source positions must be strictly increasing"
            )
        if int(positions[0]) < 0 or int(positions[-1]) >= len(frame):
            raise ValueError("volatility evidence source position is out of range")
        source = frame.iloc[positions]
    else:
        source = frame.iloc[0:0]

    timestamps = _numeric_column(source, "time")
    columns = [str(column) for column in value_columns]
    raw_columns = (
        [str(column) for column in raw_value_columns]
        if raw_value_columns is not None
        else list(columns)
    )
    raw_storage_columns = (
        [str(column) for column in raw_source_columns]
        if raw_source_columns is not None
        else list(raw_columns)
    )
    if not columns or any(not column.strip() for column in columns):
        raise ValueError("volatility evidence value columns must be nonempty")
    if not raw_columns or any(not column.strip() for column in raw_columns):
        raise ValueError("raw volatility evidence value columns must be nonempty")
    if any(not column.strip() for column in raw_storage_columns):
        raise ValueError("raw source columns must be nonempty")
    if len(raw_columns) != len(columns):
        raise ValueError(
            "raw and effective volatility evidence columns must have equal length"
        )
    if len(raw_storage_columns) != len(raw_columns):
        raise ValueError("raw source and raw label columns must have equal length")
    if len(columns) != len(set(columns)):
        raise ValueError("volatility evidence value columns must be unique")
    if "time" in columns:
        raise ValueError("time cannot also be a volatility evidence value column")
    if len(raw_columns) != len(set(raw_columns)):
        raise ValueError("raw volatility evidence value columns must be unique")
    if len(raw_storage_columns) != len(set(raw_storage_columns)):
        raise ValueError("raw source columns must be unique")
    if "time" in raw_columns or "time" in raw_storage_columns:
        raise ValueError("time cannot also be a raw volatility value column")
    missing_columns = [
        column
        for column in ["time", *columns, *raw_storage_columns]
        if column not in frame
    ]
    if missing_columns:
        raise ValueError(
            "volatility evidence source columns are missing: "
            + ", ".join(missing_columns)
        )
    values = (
        np.column_stack([_numeric_column(source, column) for column in columns])
        if columns
        else np.empty((len(source), 0), dtype=float)
    )
    raw_values = (
        np.column_stack(
            [_numeric_column(source, column) for column in raw_storage_columns]
        )
        if raw_columns
        else np.empty((len(source), 0), dtype=float)
    )
    rows = np.column_stack((timestamps, values))
    raw_rows = np.column_stack((timestamps, raw_values))
    first_timestamp = float(timestamps[0]) if timestamps.size else float("nan")
    last_timestamp = float(timestamps[-1]) if timestamps.size else float("nan")

    return_values = _canonical_float64_array(returns).reshape(-1)
    return_start_times = _canonical_float64_array(return_start_timestamps).reshape(-1)
    return_times = _canonical_float64_array(return_timestamps).reshape(-1)
    if not (return_values.size == return_start_times.size == return_times.size):
        raise ValueError(
            "return values and start/end timestamps must have the same length"
        )
    transformed_values = _canonical_float64_array(transformed_input)
    for label, values_to_check in (
        ("source timestamps", timestamps),
        ("source values", values),
        ("raw source values", raw_values),
        ("returns", return_values),
        ("return start timestamps", return_start_times),
        ("return timestamps", return_times),
        ("transformed input", transformed_values),
    ):
        if values_to_check.size and not bool(np.all(np.isfinite(values_to_check))):
            raise ValueError(f"volatility evidence {label} must be finite")
    base_context = {
        "method": str(method),
        "timeframe": str(timeframe),
        "operation": str(operation),
    }
    source_fields = ["time", *columns]
    timestamp_deltas = np.diff(timestamps)

    return {
        "schema_version": VOLATILITY_INPUT_EVIDENCE_SCHEMA_VERSION,
        "digest_algorithm": VOLATILITY_DIGEST_ALGORITHM,
        "digest_encoding": VOLATILITY_DIGEST_ENCODING,
        "method": str(method),
        "timeframe": str(timeframe),
        "operation": str(operation),
        "source": {
            "columns": source_fields,
            "raw_value_columns": raw_columns,
            "effective_value_columns": columns,
            "raw_effective_values_equal": bool(
                raw_values.shape == values.shape
                and np.array_equal(raw_values, values, equal_nan=True)
            ),
            "row_count": int(len(source)),
            "timestamps_strictly_increasing": bool(
                not timestamp_deltas.size or np.all(timestamp_deltas > 0.0)
            ),
            "duplicate_timestamp_intervals": int(
                np.count_nonzero(timestamp_deltas == 0.0)
            ),
            "out_of_order_timestamp_intervals": int(
                np.count_nonzero(timestamp_deltas < 0.0)
            ),
            "start": _timestamp_bound(first_timestamp),
            "end": _timestamp_bound(last_timestamp),
            "row_sha256": volatility_array_sha256(
                rows,
                domain="volatility_source_rows",
                context={**base_context, "fields": source_fields},
            ),
            "effective_row_sha256": volatility_array_sha256(
                rows,
                domain="volatility_effective_source_rows",
                context={**base_context, "fields": source_fields},
            ),
            "raw_row_sha256": volatility_array_sha256(
                raw_rows,
                domain="volatility_raw_source_rows",
                context={
                    **base_context,
                    "fields": ["time", *raw_columns],
                },
            ),
            "timestamp_sha256": volatility_array_sha256(
                timestamps,
                domain="volatility_source_timestamps",
                context={**base_context, "fields": ["time"]},
            ),
            "value_sha256": volatility_array_sha256(
                values,
                domain="volatility_source_values",
                context={**base_context, "fields": columns},
            ),
            "effective_value_sha256": volatility_array_sha256(
                values,
                domain="volatility_effective_source_values",
                context={**base_context, "fields": columns},
            ),
            "raw_value_sha256": volatility_array_sha256(
                raw_values,
                domain="volatility_raw_source_values",
                context={**base_context, "fields": raw_columns},
            ),
        },
        "returns": {
            "operation": str(return_operation),
            "timestamp_policy": str(return_timestamp_policy),
            "count": int(return_values.size),
            "sha256": volatility_array_sha256(
                return_values,
                domain="volatility_effective_returns",
                context={
                    **base_context,
                    "operation": str(return_operation),
                    "fields": ["return"],
                },
            ),
            "timestamp_sha256": volatility_array_sha256(
                return_times,
                domain="volatility_effective_return_timestamps",
                context={
                    **base_context,
                    "operation": str(return_operation),
                    "fields": ["return_time"],
                },
            ),
            "start_timestamp_sha256": volatility_array_sha256(
                return_start_times,
                domain="volatility_effective_return_start_timestamps",
                context={
                    **base_context,
                    "operation": str(return_operation),
                    "fields": ["return_start_time"],
                },
            ),
            "pair_sha256": volatility_array_sha256(
                np.column_stack((return_start_times, return_times, return_values)),
                domain="volatility_effective_return_pairs",
                context={
                    **base_context,
                    "operation": str(return_operation),
                    "timestamp_policy": str(return_timestamp_policy),
                    "fields": [
                        "previous_time",
                        "current_time",
                        "return",
                    ],
                },
            ),
        },
        "transformed_input": _array_descriptor(
            transformed_values,
            domain="volatility_effective_transformed_input",
            operation=transformed_operation,
            fields=transformed_fields,
            context=base_context,
        ),
    }


def build_array_evidence(
    values: Any,
    *,
    domain: str,
    operation: str,
    fields: Sequence[str],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a public descriptor for a secondary derived numeric array."""
    return _array_descriptor(
        values,
        domain=domain,
        operation=operation,
        fields=fields,
        context=context,
    )
