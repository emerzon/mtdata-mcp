from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..common import (
    _NF_ENV_LOCK,
    _extract_forecast_values,
    _nf_resolve_accelerator,
    _NfEnvGuard,
    nf_build_model_kwargs,
    nf_create_and_fit,
    nf_predict_from_fitted,
)
from ..common import edge_pad_to_length as _edge_pad_to_length  # noqa: F401
from ..forecast_registry import ForecastRegistry
from ..interface import (
    CancelToken,
    ForecastMethod,
    ForecastResult,
    ProgressCallback,
    ProgressReporter,
    TrainResult,
)


def _ensure_pytorch_lightning_distributed_compat() -> None:
    """Provide the legacy Lightning distributed logger expected by old deps."""
    try:
        import logging
        import sys
        import types

        import pytorch_lightning as _pl  # type: ignore

        utilities = getattr(_pl, 'utilities', None)
        if utilities is None or hasattr(utilities, 'distributed'):
            return

        distributed = types.ModuleType('pytorch_lightning.utilities.distributed')
        distributed.log = logging.getLogger('pytorch_lightning.utilities.distributed')  # type: ignore[attr-defined]
        utilities.distributed = distributed
        sys.modules.setdefault('pytorch_lightning.utilities.distributed', distributed)
    except Exception:
        pass


def _import_neuralforecast_model_classes() -> Dict[str, Any]:
    """Import NeuralForecast model classes from the supported Nixtla API."""
    _ensure_pytorch_lightning_distributed_compat()
    try:
        from neuralforecast.models import NHITS as _NF_NHITS  # type: ignore
        from neuralforecast.models import TFT as _NF_TFT  # type: ignore
        from neuralforecast.models import NBEATSx as _NF_NBEATSX  # type: ignore
        from neuralforecast.models import PatchTST as _NF_PATCHTST  # type: ignore
    except Exception as ex:
        raise RuntimeError(
            "Installed neuralforecast package is not compatible with mtdata neural "
            "methods. Install a modern Nixtla neuralforecast release in a "
            "Python/Torch environment supported by that package."
        ) from ex

    return {
        'nhits': _NF_NHITS,
        'nbeatsx': _NF_NBEATSX,
        'tft': _NF_TFT,
        'patchtst': _NF_PATCHTST,
    }


def _resolve_nf_model_class(method_name: str):
    """Return the NeuralForecast model class for *method_name*."""
    model_map = _import_neuralforecast_model_classes()
    cls = model_map.get(str(method_name).lower().strip())
    if cls is None:
        raise RuntimeError(f"Unknown neural method: {method_name}")
    return cls


def _neural_resolve_hyperparams(
    params: Dict[str, Any], n: int, fh: int, m: int,
) -> Tuple[int, int, int, Optional[float]]:
    """Return (input_size, steps, batch_size, learning_rate)."""
    h = int(fh)
    available_context = int(max(1, (n - h) if n > h else n))
    if params.get('input_size') is not None:
        requested = int(params['input_size'])
        input_size = int(max(1, min(requested, available_context)))
    else:
        base = max(64, (int(m) * 3) if m and int(m) > 0 else 96)
        input_size = int(max(1, min(available_context, base)))
    steps = int(params.get('max_steps', params.get('max_epochs', 50)))
    batch_size = int(params.get('batch_size', 32))
    lr = params.get('learning_rate', None)
    return input_size, steps, batch_size, float(lr) if lr is not None else None


def _neural_resolve_validation_settings(
    params: Dict[str, Any], n: int, fh: int, steps: int,
) -> Tuple[int, Optional[int]]:
    raw_val_size = params.get("val_size")
    if raw_val_size is None:
        available = max(0, int(n) - int(fh) - 1)
        if available <= 0:
            val_size = 0
        else:
            val_size = min(available, max(int(fh), int(n) // 5))
    else:
        val_size = max(0, min(int(raw_val_size), max(0, int(n) - 1)))

    raw_patience = params.get("early_stop_patience_steps")
    if val_size <= 0:
        return 0, None
    if raw_patience is None:
        patience = min(max(5, int(steps) // 5), max(1, int(steps) - 1))
    else:
        patience = max(0, int(raw_patience))
    return int(val_size), (int(patience) if patience > 0 else None)


class NeuralForecastMethod(ForecastMethod):
    PARAMS: List[Dict[str, Any]] = [
        {"name": "input_size", "type": "int|null", "description": "Lookback context for the model (auto if omitted)."},
        {"name": "max_steps", "type": "int|null", "description": "Training steps (fallback to max_epochs, default: 50)."},
        {"name": "max_epochs", "type": "int|null", "description": "Alias for max_steps."},
        {"name": "batch_size", "type": "int", "description": "Batch size (default: 32)."},
        {"name": "learning_rate", "type": "float|null", "description": "Learning rate (model default if omitted)."},
        {"name": "val_size", "type": "int|null", "description": "Validation window for early stopping (auto if omitted)."},
        {
            "name": "early_stop_patience_steps",
            "type": "int|null",
            "description": "Stop training early after this many non-improving validation checks (auto if omitted).",
        },
    ]

    @property
    def category(self) -> str:
        return "neural"

    @property
    def required_packages(self) -> List[str]:
        return ["neuralforecast"]

    @property
    def supports_features(self) -> Dict[str, bool]:
        return {"price": True, "return": True, "volatility": False, "ci": False}

    # ------------------------------------------------------------------
    # Train / predict lifecycle
    # ------------------------------------------------------------------

    @property
    def supports_training(self) -> bool:
        return True

    @property
    def training_category(self):
        return "heavy"

    @property
    def train_supports_cancel(self) -> bool:
        return True

    @property
    def train_supports_progress(self) -> bool:
        return True

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
        from ..common import _create_training_dataframes

        p = dict(params or {})
        x = np.asarray(series.values, dtype=float)
        n = int(x.size)
        if n < 5:
            raise ValueError(f"{self.name} requires at least 5 observations")

        exog_used = exog if exog is not None else p.get("exog_used")
        exog_future_arr = p.get("exog_future")
        reporter = ProgressReporter(progress_callback, total_steps=3)
        reporter.stage(0, f"Preparing {self.name} training data", force=True)
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()

        input_size, steps, batch_size, lr = _neural_resolve_hyperparams(
            p, n, int(horizon), int(seasonality or 0),
        )
        val_size, early_stop_patience_steps = _neural_resolve_validation_settings(
            p, n, int(horizon), steps,
        )
        timeframe = str(p.get("timeframe") or kwargs.get("timeframe") or "H1")
        model_class = _resolve_nf_model_class(self.name)

        Y_df, _, _ = _create_training_dataframes(x, int(horizon), exog_used, exog_future_arr)

        reporter.stage(1, f"Fitting {self.name} model", force=True)
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()

        accel = _nf_resolve_accelerator()
        model_kwargs = nf_build_model_kwargs(
            model_class=model_class,
            fh=int(horizon),
            input_size=input_size,
            batch_size=batch_size,
            steps=steps,
            learning_rate=lr,
            accel=accel,
            early_stop_patience_steps=early_stop_patience_steps,
        )

        with _NF_ENV_LOCK:
            with _NfEnvGuard(accel):
                nf = nf_create_and_fit(
                    model_class=model_class,
                    model_kwargs=model_kwargs,
                    timeframe=timeframe,
                    Y_df=Y_df,
                    val_size=val_size,
                    exog_used=exog_used,
                )

        reporter.stage(3, "Training complete", force=True)

        artifact_bytes = self.serialize_artifact(nf)
        params_used = {
            'max_epochs': steps, 'input_size': input_size,
            'batch_size': batch_size,
        }
        if val_size > 0:
            params_used['val_size'] = val_size
        if early_stop_patience_steps is not None:
            params_used['early_stop_patience_steps'] = early_stop_patience_steps
        return TrainResult(
            artifact_bytes=artifact_bytes,
            params_used=params_used,
            metadata={"accelerator": accel, "timeframe": timeframe},
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
        nf = model  # deserialized NeuralForecast object
        p = dict(params or {})
        exog_future_arr = kwargs.get("exog_future")
        if exog_future_arr is None:
            exog_future_arr = exog_future if exog_future is not None else p.get("exog_future")

        accel = _nf_resolve_accelerator()
        with _NF_ENV_LOCK:
            with _NfEnvGuard(accel):
                Yf = nf_predict_from_fitted(
                    nf,
                    fh=int(horizon),
                    exog_future=exog_future_arr if isinstance(exog_future_arr, np.ndarray) else None,
                    future_times=kwargs.get("future_times"),
                )

        try:
            Yf = Yf[Yf['unique_id'] == 'ts']
        except Exception:
            pass
        f_vals = _extract_forecast_values(
            Yf, int(horizon),
            f"{self.name.upper()} forecast",
            allow_actual_fallback=False,
        )
        params_used = dict(p)
        return ForecastResult(forecast=f_vals.astype(float, copy=False), params_used=params_used)

    def training_fingerprint(
        self,
        horizon: int,
        seasonality: int,
        params: Dict[str, Any],
        *,
        timeframe: Optional[str] = None,
        has_exog: bool = False,
    ) -> Dict[str, Any]:
        fp = super().training_fingerprint(
            horizon, seasonality, params,
            timeframe=timeframe, has_exog=has_exog,
        )
        # Neural models also depend on input_size and batch_size
        p = params or {}
        fp["input_size"] = p.get("input_size")
        fp["batch_size"] = int(p.get("batch_size", 32))
        return fp


@ForecastRegistry.register("nhits")
class NHITSMethod(NeuralForecastMethod):
    @property
    def name(self) -> str:
        return "nhits"


@ForecastRegistry.register("nbeatsx")
class NBEATSXMethod(NeuralForecastMethod):
    @property
    def name(self) -> str:
        return "nbeatsx"


@ForecastRegistry.register("tft")
class TFTMethod(NeuralForecastMethod):
    @property
    def name(self) -> str:
        return "tft"


@ForecastRegistry.register("patchtst")
class PatchTSTMethod(NeuralForecastMethod):
    @property
    def name(self) -> str:
        return "patchtst"
