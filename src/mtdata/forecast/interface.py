import json
import struct
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

_ARTIFACT_ENVELOPE_MAGIC = b"MTDATA-ARTIFACT\x00"
_ARTIFACT_FORMAT_VERSION = 2
_ARTIFACT_HEADER_LIMIT = 64 * 1024
_ARTIFACT_DEPENDENCY_DISTRIBUTIONS = {
    "lightgbm": "lightgbm",
    "mlforecast": "mlforecast",
    "neuralforecast": "neuralforecast",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "sktime": "sktime",
    "statsforecast": "statsforecast",
    "statsmodels": "statsmodels",
    "torch": "torch",
    "transformers": "transformers",
}
_APPLICATION_DISTRIBUTION = "mtdata-mcp"


class ArtifactCompatibilityError(ValueError):
    """Raised before deserialization when a stored artifact is incompatible."""


def _python_runtime_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _distribution_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _artifact_dependency_roots(artifact: Any) -> set[str]:
    roots: set[str] = set()
    seen: set[int] = set()
    stack: List[tuple[Any, int]] = [(artifact, 0)]
    while stack and len(seen) < 512:
        value, depth = stack.pop()
        value_id = id(value)
        if value_id in seen:
            continue
        seen.add(value_id)
        module_name = str(type(value).__module__ or "").split(".", 1)[0]
        if module_name in _ARTIFACT_DEPENDENCY_DISTRIBUTIONS:
            roots.add(module_name)
        if depth >= 4:
            continue
        if isinstance(value, dict):
            children = value.values()
        elif isinstance(value, (list, tuple, set, frozenset)):
            children = value
        else:
            try:
                children = vars(value).values()
            except (TypeError, AttributeError):
                continue
        stack.extend((child, depth + 1) for child in children)
    return roots


def _artifact_runtime_header(artifact: Any) -> Dict[str, Any]:
    dependencies: Dict[str, str] = {}
    for module_root in sorted(_artifact_dependency_roots(artifact)):
        distribution = _ARTIFACT_DEPENDENCY_DISTRIBUTIONS[module_root]
        version = _distribution_version(distribution)
        if version is not None:
            dependencies[distribution] = version
    return {
        "format_version": _ARTIFACT_FORMAT_VERSION,
        "python": _python_runtime_version(),
        "application": _distribution_version(_APPLICATION_DISTRIBUTION),
        "dependencies": dependencies,
    }


def _validate_artifact_runtime_header(header: Dict[str, Any]) -> None:
    if header.get("format_version") != _ARTIFACT_FORMAT_VERSION:
        raise ArtifactCompatibilityError(
            "Stored model uses an unsupported artifact format; retrain it with "
            "the current mtdata runtime."
        )
    mismatches: List[str] = []
    stored_python = str(header.get("python") or "")
    current_python = _python_runtime_version()
    if stored_python != current_python:
        mismatches.append(f"python {stored_python or '<missing>'} != {current_python}")
    stored_application = header.get("application")
    current_application = _distribution_version(_APPLICATION_DISTRIBUTION)
    if (
        stored_application
        and current_application
        and str(stored_application) != current_application
    ):
        mismatches.append(
            f"{_APPLICATION_DISTRIBUTION} {stored_application} != {current_application}"
        )
    dependencies = header.get("dependencies")
    if not isinstance(dependencies, dict):
        mismatches.append("dependency fingerprint is missing")
    else:
        for distribution, stored_version in sorted(dependencies.items()):
            current_version = _distribution_version(str(distribution))
            if current_version != str(stored_version):
                mismatches.append(
                    f"{distribution} {stored_version} != {current_version or '<missing>'}"
                )
    if mismatches:
        raise ArtifactCompatibilityError(
            "Stored model runtime is incompatible ("
            + "; ".join(mismatches)
            + "); retrain it before prediction."
        )


def _pack_artifact_envelope(artifact: Any, pickle_payload: bytes) -> bytes:
    header_bytes = json.dumps(
        _artifact_runtime_header(artifact),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return (
        _ARTIFACT_ENVELOPE_MAGIC
        + struct.pack(">I", len(header_bytes))
        + header_bytes
        + pickle_payload
    )


def _unpack_artifact_envelope(data: bytes) -> bytes:
    if not data.startswith(_ARTIFACT_ENVELOPE_MAGIC):
        raise ArtifactCompatibilityError(
            "Stored model is an unversioned legacy artifact; retrain it with the "
            "current mtdata runtime."
        )
    length_start = len(_ARTIFACT_ENVELOPE_MAGIC)
    length_end = length_start + 4
    if len(data) < length_end:
        raise ArtifactCompatibilityError("Stored model artifact header is truncated.")
    header_length = struct.unpack(">I", data[length_start:length_end])[0]
    if header_length <= 0 or header_length > _ARTIFACT_HEADER_LIMIT:
        raise ArtifactCompatibilityError("Stored model artifact header length is invalid.")
    payload_start = length_end + header_length
    if payload_start >= len(data):
        raise ArtifactCompatibilityError("Stored model artifact payload is truncated.")
    try:
        header = json.loads(data[length_end:payload_start].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactCompatibilityError(
            "Stored model artifact header is unreadable."
        ) from exc
    if not isinstance(header, dict):
        raise ArtifactCompatibilityError("Stored model artifact header is invalid.")
    _validate_artifact_runtime_header(header)
    return data[payload_start:]


@dataclass
class ForecastResult:
    forecast: np.ndarray
    ci_values: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (lower, upper)
    params_used: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TrainingProgress:
    """Progress update emitted during model training via ``progress_callback``."""

    step: int
    total_steps: int
    loss: Optional[float] = None
    metrics: Optional[Dict[str, float]] = None
    eta_seconds: Optional[float] = None
    message: Optional[str] = None

    @property
    def fraction(self) -> float:
        if self.total_steps <= 0:
            return 0.0
        return min(1.0, max(0.0, self.step / self.total_steps))


TrainingCategory = Literal["instant", "fast", "moderate", "heavy"]
ProgressCallback = Callable[[TrainingProgress], None]


class TrainingCancelledError(RuntimeError):
    """Raised when a background training job is cancelled cooperatively."""


class CancelToken:
    """Simple cooperative cancellation token for long-running training."""

    def __init__(self, checker: Callable[[], bool]) -> None:
        self._checker = checker

    @property
    def cancelled(self) -> bool:
        return bool(self._checker())

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise TrainingCancelledError("Training cancelled")


class ProgressReporter:
    """Throttled helper for emitting monotonic training progress updates."""

    def __init__(
        self,
        callback: Optional[ProgressCallback],
        *,
        total_steps: int,
        throttle_seconds: float = 1.0,
    ) -> None:
        self._callback = callback
        self._total_steps = max(1, int(total_steps))
        self._throttle_seconds = max(0.0, float(throttle_seconds))
        self._started_at = time.monotonic()
        self._last_emit_at = 0.0
        self._last_step = -1

    def emit(
        self,
        step: int,
        *,
        loss: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
        message: Optional[str] = None,
        force: bool = False,
    ) -> None:
        if self._callback is None:
            return
        now = time.monotonic()
        clamped_step = min(self._total_steps, max(0, int(step)))
        if (
            not force
            and clamped_step <= self._last_step
            and (now - self._last_emit_at) < self._throttle_seconds
        ):
            return
        elapsed = max(0.0, now - self._started_at)
        eta_seconds: Optional[float] = None
        if clamped_step > 0 and clamped_step < self._total_steps and elapsed > 0:
            rate = elapsed / clamped_step
            eta_seconds = max(0.0, rate * (self._total_steps - clamped_step))
        progress = TrainingProgress(
            step=clamped_step,
            total_steps=self._total_steps,
            loss=loss,
            metrics=dict(metrics) if isinstance(metrics, dict) else None,
            eta_seconds=eta_seconds,
            message=message,
        )
        self._callback(progress)
        self._last_emit_at = now
        self._last_step = clamped_step

    def stage(self, step: int, message: str, *, force: bool = False) -> None:
        self.emit(step, message=message, force=force)


@dataclass
class TrainedModelHandle:
    """Opaque reference to a persisted trained model."""

    model_id: str
    method: str
    data_scope: str
    params_hash: str
    created_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    store_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainResult:
    """Structured return value from :meth:`ForecastMethod.train`.

    *artifact_bytes* is the method-serialized model (bytes).  The model
    store persists this opaque blob; deserialization is method-owned via
    :meth:`ForecastMethod.deserialize_artifact`.
    """

    artifact_bytes: bytes
    params_used: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ForecastCapabilityDescriptor:
    capability_id: str
    method: str
    adapter_method: str
    library: str
    namespace: str
    concept: str
    display_name: str
    category: str
    description: str = ""
    available: bool = True
    requires: Tuple[str, ...] = ()
    supports: Dict[str, bool] = field(default_factory=dict)
    params: Tuple[Dict[str, Any], ...] = ()
    aliases: Tuple[str, ...] = ()
    selector_key: Optional[str] = None
    selector_value: Optional[str] = None
    selector_mode: str = "method"
    source: str = "registry"
    notes: Optional[str] = None

    def selector(self) -> Dict[str, Any]:
        selector: Dict[str, Any] = {"mode": self.selector_mode}
        if self.selector_key is not None:
            selector["key"] = self.selector_key
        if self.selector_value is not None:
            selector["value"] = self.selector_value
        if self.aliases:
            selector["aliases"] = list(self.aliases)
        return selector

    def execution(self) -> Dict[str, Any]:
        execution: Dict[str, Any] = {
            "library": self.library,
            "method": self.adapter_method,
        }
        if self.selector_key is not None and self.selector_value is not None:
            execution["params"] = {self.selector_key: self.selector_value}
        return execution

    def to_record(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "capability_id": self.capability_id,
            "method": self.method,
            "adapter_method": self.adapter_method,
            "library": self.library,
            "namespace": self.namespace,
            "concept": self.concept,
            "display_name": self.display_name,
            "category": self.category,
            "description": self.description,
            "available": self.available,
            "requires": list(self.requires),
            "supports": dict(self.supports),
            "params": [dict(param) for param in self.params],
            "aliases": list(self.aliases),
            "selector": self.selector(),
            "execution": self.execution(),
            "source": self.source,
        }
        if self.notes:
            record["notes"] = self.notes
        return record


@dataclass(frozen=True)
class ForecastCallContext:
    method: str
    symbol: str
    timeframe: str
    quantity: str
    horizon: int
    seasonality: int
    base_col: str
    ci_alpha: Optional[float]
    as_of: Optional[str]
    denoise_spec_used: Optional[Any]
    history_df: pd.DataFrame
    target_series: pd.Series
    exog_used: Optional[np.ndarray]
    future_exog: Optional[np.ndarray]
    features: Optional[Dict[str, Any]] = None
    feature_info: Optional[Dict[str, Any]] = None

class ForecastMethod(ABC):
    """Abstract base class for all forecasting methods."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of the method."""
        ...
    
    @property
    def category(self) -> str:
        """Category of the method (e.g., 'classical', 'neural', 'ensemble')."""
        return "unknown"
        
    @property
    def required_packages(self) -> List[str]:
        """List of Python packages required for this method."""
        return []

    @property
    def supports_features(self) -> Dict[str, bool]:
        """Dictionary of supported features (price, return, volatility, ci)."""
        return {"price": True, "return": True, "volatility": False, "ci": False}

    @property
    def supports_historical_exog(self) -> bool:
        """Whether the adapter is audited to consume historical exogenous inputs."""
        return False

    @property
    def supports_future_exog(self) -> bool:
        """Whether the adapter is audited to consume future exogenous inputs."""
        return False

    def forecast(
        self,
        series: pd.Series,
        horizon: int,
        seasonality: int,
        params: Dict[str, Any],
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> ForecastResult:
        """Generate a forecast.

        Trainable methods use an ephemeral ``train()`` + ``predict_with_model()``
        path. One-shot methods must override this.
        """
        if not self.supports_training:
            raise NotImplementedError(f"{self.name} must implement forecast()")
        train_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in {"progress_callback", "cancel_token", "timeframe"}
        }
        exog = kwargs.get("exog_used")
        if exog is None:
            exog = (params or {}).get("exog_used")
        live_artifact: list[Any] = []
        original_serialize = self.serialize_artifact

        def _capture_artifact(artifact: Any) -> bytes:
            live_artifact.append(artifact)
            try:
                return original_serialize(artifact)
            except Exception:
                return b""

        self.serialize_artifact = _capture_artifact  # type: ignore[method-assign]
        try:
            trained = self.train(
                series,
                horizon,
                seasonality,
                params,
                exog=exog,
                **train_kwargs,
            )
        finally:
            self.serialize_artifact = original_serialize  # type: ignore[method-assign]
        artifact = (
            live_artifact[0]
            if live_artifact
            else self.deserialize_artifact(trained.artifact_bytes)
        )
        result = self.predict_with_model(
            artifact,
            series,
            horizon,
            seasonality,
            params,
            exog_future=exog_future,
            **kwargs,
        )
        if getattr(trained, "params_used", None):
            merged = dict(trained.params_used)
            merged.update(result.params_used or {})
            result.params_used = merged
        return result
    
    def validate_params(self, params: Dict[str, Any]) -> List[str]:
        """
        Validate method-specific parameters.
        
        Args:
            params: Dictionary of parameters to validate.
            
        Returns:
            List of error messages (empty if valid).
        """
        return []

    def prepare_forecast_call(
        self,
        params: Dict[str, Any],
        call_kwargs: Dict[str, Any],
        context: ForecastCallContext,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Allow methods to request engine context without engine name checks."""
        return params, call_kwargs

    # ------------------------------------------------------------------
    # Train / predict lifecycle (opt-in for methods with heavy training)
    # ------------------------------------------------------------------

    @property
    def supports_training(self) -> bool:
        """Whether this method supports a separate train → predict lifecycle."""
        return False

    @property
    def supports_live_model_update(self) -> bool:
        """Whether a stored artifact can safely incorporate newer history.

        A trainable method may support a persisted train → predict lifecycle
        without being able to advance a fitted artifact to a newer series.  The
        forecast engine uses this capability to distinguish safe live reuse from
        historical exact-anchor prediction.
        """
        return False

    @property
    def train_supports_cancel(self) -> bool:
        """Whether ``train()`` checks a cooperative cancellation token."""
        return False

    @property
    def train_supports_progress(self) -> bool:
        """Whether ``train()`` emits meaningful progress updates."""
        return False

    @property
    def training_category(self) -> TrainingCategory:
        """Hint for task routing: 'instant', 'fast', 'moderate', or 'heavy'."""
        return "instant"

    def train(
        self,
        series: pd.Series,
        horizon: int,
        seasonality: int,
        params: Dict[str, Any],
        *,
        progress_callback: Optional[ProgressCallback] = None,
        cancel_token: Optional[CancelToken] = None,
        exog: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> TrainResult:
        """Train and return a serialized model artifact.

        The returned :class:`TrainResult` contains method-serialized bytes
        that :meth:`deserialize_artifact` can reconstruct.  Override in
        methods that support separate training.
        """
        raise NotImplementedError(
            f"{self.name} does not support separate training"
        )

    def predict_with_model(
        self,
        model: Any,
        series: pd.Series,
        horizon: int,
        seasonality: int,
        params: Dict[str, Any],
        *,
        exog_future: Optional[pd.DataFrame] = None,
        **kwargs: Any,
    ) -> ForecastResult:
        """Generate a forecast using a previously trained model artifact.

        *model* is the deserialized object returned by
        :meth:`deserialize_artifact`.  Default implementation ignores
        *model* and falls back to :meth:`forecast`.
        """
        return self.forecast(
            series, horizon, seasonality, params,
            exog_future=exog_future, **kwargs,
        )

    def serialize_artifact(self, artifact: Any) -> bytes:
        """Serialize a trained model artifact to bytes.

        Default wraps pickle in a versioned runtime-compatibility envelope.
        Override both serialization methods for framework-specific formats.
        """
        import pickle

        payload = pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL)
        return _pack_artifact_envelope(artifact, payload)

    def deserialize_artifact(self, data: bytes) -> Any:
        """Deserialize bytes produced by :meth:`serialize_artifact`.

        Runtime compatibility is checked before pickle is invoked. Override to
        match :meth:`serialize_artifact` for method-owned formats.
        """
        import pickle

        payload = _unpack_artifact_envelope(data)
        return pickle.loads(payload)  # noqa: S301

    def training_fingerprint(
        self,
        horizon: int,
        seasonality: int,
        params: Dict[str, Any],
        *,
        timeframe: Optional[str] = None,
        has_exog: bool = False,
    ) -> Dict[str, Any]:
        """Return a canonical dict of training-relevant attributes.

        This fingerprint is hashed to produce the ``params_hash`` used
        for model identity.  Override to include method-specific training
        parameters that affect the trained artifact.
        """
        _PREDICTION_ONLY_KEYS = frozenset({
            "ci_alpha", "as_of", "seasonality",
        })
        filtered_params = {
            k: v for k, v in sorted((params or {}).items())
            if k not in _PREDICTION_ONLY_KEYS
        }
        training_context = filtered_params.get("_training_context")
        if isinstance(training_context, dict):
            window_mode = str(
                training_context.get("training_window_mode") or "latest"
            ).strip().lower()
            if window_mode == "latest":
                # Latest-window models intentionally refresh in place as new
                # bars arrive. Historical windows retain their observed bounds
                # below so replay artifacts cannot overwrite the live cache.
                filtered_params["_training_context"] = {
                    key: value
                    for key, value in sorted(training_context.items())
                    if key
                    not in {
                        "target_points",
                        "history_start_epoch",
                        "training_end_epoch",
                        "training_window_mode",
                    }
                }
        return {
            "method": self.name,
            "horizon": int(horizon),
            "seasonality": int(seasonality),
            "timeframe": timeframe,
            "has_exog": bool(has_exog),
            "params": filtered_params,
        }

    @staticmethod
    def hash_fingerprint(fingerprint: Dict[str, Any]) -> str:
        """Compute a stable short hash from a training fingerprint dict."""
        import hashlib
        import json
        blob = json.dumps(fingerprint, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]
