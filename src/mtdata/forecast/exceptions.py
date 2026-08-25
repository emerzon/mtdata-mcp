"""Forecast-domain exception types."""

from __future__ import annotations

from typing import Any, Dict, List


class ForecastError(RuntimeError):
    """Raised when forecast execution fails outside normal result handling."""


class UnknownFeatureColumnError(ForecastError):
    """Raised when an explicit include/exog token is not a usable column."""

    error_code = "unknown_feature_column"

    def __init__(
        self,
        unknown_columns: List[str],
        available_columns: List[str],
    ) -> None:
        self.unknown_columns = [str(column) for column in unknown_columns]
        self.available_columns = [str(column) for column in available_columns]
        unknown = ", ".join(self.unknown_columns)
        available = ", ".join(self.available_columns) or "(none)"
        super().__init__(
            f"Unknown feature column(s): {unknown}. Available columns: {available}."
        )

    def details(self) -> Dict[str, Any]:
        return {
            "error_code": self.error_code,
            "unknown_columns": list(self.unknown_columns),
            "available_columns": list(self.available_columns),
        }


class ForecastResultError(ForecastError):
    """Raised when a legacy forecast payload encodes an error dict."""


class ModelCompatibilityError(ForecastError):
    """Raised when a requested model ID has a different training identity."""

    def __init__(
        self,
        message: str,
        *,
        model_id: str,
        stored_fingerprint: Any,
        requested_fingerprint: Any,
        mismatches: Any,
    ) -> None:
        super().__init__(message)
        self.model_id = str(model_id)
        self.stored_fingerprint = stored_fingerprint
        self.requested_fingerprint = requested_fingerprint
        self.mismatches = mismatches

    def details(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "mismatches": self.mismatches,
            "stored_fingerprint": self.stored_fingerprint,
            "requested_fingerprint": self.requested_fingerprint,
        }


def raise_if_error_result(result: Any) -> Any:
    """Raise ForecastResultError when a legacy forecast payload encodes an error dict."""
    if isinstance(result, dict):
        error = result.get("error")
        if isinstance(error, str) and error.strip():
            raise ForecastResultError(error)
    return result
