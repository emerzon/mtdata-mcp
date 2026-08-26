"""Runtime annotation resolution helpers."""

from __future__ import annotations

import annotationlib
import inspect
from typing import Any, Dict, get_type_hints

_ANNOTATION_VALUE_FORMAT = annotationlib.Format.VALUE


def get_runtime_signature(obj: Any) -> inspect.Signature:
    """Resolve a signature with evaluated annotations when available."""
    try:
        return inspect.signature(
            obj,
            eval_str=True,
            annotation_format=_ANNOTATION_VALUE_FORMAT,
        )
    except Exception:
        return inspect.signature(obj)


def get_runtime_annotations(obj: Any) -> Dict[str, Any]:
    """Resolve annotations using Python 3.14 annotationlib."""
    try:
        resolved = annotationlib.get_annotations(
            obj,
            eval_str=True,
            format=_ANNOTATION_VALUE_FORMAT,
        )
        if isinstance(resolved, dict):
            return resolved
    except Exception:
        pass
    try:
        # ``Annotated`` metadata carries public validation constraints used by
        # dynamic transports (for example ``Field(ge=1)``).  The default
        # behavior strips that metadata.
        resolved = get_type_hints(obj, include_extras=True)
        if isinstance(resolved, dict):
            return resolved
    except Exception:
        pass
    raw = getattr(obj, "__annotations__", None)
    return raw if isinstance(raw, dict) else {}
