from __future__ import annotations

import asyncio
import inspect
import threading
from typing import Any

from pydantic import BaseModel, ValidationError

from ._mcp_tools import _get_pydantic_model_fields


def unwrap_tool_callable(func: Any) -> Any:
    raw = getattr(func, "__wrapped__", None)
    return raw if callable(raw) else func


def resolve_sync_tool_result(result: Any) -> Any:
    if not inspect.isawaitable(result):
        return result

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(result)

    state: dict[str, Any] = {}

    def _runner() -> None:
        try:
            state["result"] = asyncio.run(result)
        except BaseException as exc:  # pragma: no cover
            state["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()

    if "error" in state:
        raise state["error"]
    return state.get("result")


def _coerce_tool_kwargs(target: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    coerced = dict(kwargs)
    if not coerced:
        return coerced
    try:
        sig = inspect.signature(target)
        annotations = inspect.get_annotations(target, eval_str=True)
        params = list(sig.parameters.values())
        if len(params) == 1:
            request_param = params[0]
            request_type = annotations.get(request_param.name, request_param.annotation)
            if (
                request_param.name not in coerced
                and isinstance(request_type, type)
                and issubclass(request_type, BaseModel)
            ):
                model_fields = _get_pydantic_model_fields(request_type)
                field_names = set(model_fields.keys())
                alias_map: dict[str, str] = {}
                for name, info in model_fields.items():
                    for attr in ("alias", "validation_alias"):
                        val = getattr(info, attr, None)
                        if isinstance(val, str) and val != name:
                            alias_map[val] = name
                        elif hasattr(val, "choices"):
                            for choice in val.choices:
                                if isinstance(choice, str) and choice != name:
                                    alias_map[choice] = name
                payload: dict[str, Any] = {}
                for key in list(coerced.keys()):
                    if key in field_names or key in alias_map:
                        payload[alias_map.get(key, key)] = coerced.pop(key)
                if payload:
                    coerced[request_param.name] = request_type.model_validate(payload)
    except ValidationError:
        raise
    except Exception:
        pass
    try:
        from ._mcp_tools import _coerce_kwargs_for_callable

        _coerce_kwargs_for_callable(target, coerced)
    except Exception:
        pass
    return coerced


def call_tool_sync_structured(
    func: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    target = unwrap_tool_callable(func)
    coerced_kwargs = _coerce_tool_kwargs(target, kwargs)
    return resolve_sync_tool_result(target(*args, **coerced_kwargs))

