"""Evaluate public MCP input schemas against their runtime callables.

The checks in this module deliberately operate on the final schemas attached to
FastMCP tools.  That makes the report cover the same contract seen by MCP, the
dynamic CLI, and the generic Web API catalog.
"""

from __future__ import annotations

import inspect
import os
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from pydantic import BaseModel

from ..shared.annotations import get_runtime_signature


@dataclass(frozen=True, order=True)
class SchemaFinding:
    """One deterministic schema evaluation finding."""

    severity: str
    code: str
    tool: str
    parameter: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class SchemaEvaluationReport:
    """Sorted result of evaluating the complete public tool surface."""

    tool_count: int
    expected_tool_count: int
    findings: tuple[SchemaFinding, ...]

    @property
    def errors(self) -> tuple[SchemaFinding, ...]:
        return tuple(item for item in self.findings if item.severity == "error")

    @property
    def warnings(self) -> tuple[SchemaFinding, ...]:
        return tuple(item for item in self.findings if item.severity == "warning")

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "tool_count": self.tool_count,
            "expected_tool_count": self.expected_tool_count,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "findings": [item.to_dict() for item in self.findings],
        }


# market_depth_fetch stays registered while gated so CLI help and schema
# evaluation still see it; tools_list keeps it out of the enabled catalog.
_EXPECTED_DEFAULT_TOOL_COUNT = 83
_EXPECTED_GATED_TOOL_COUNT = 83
_OUTPUT_CONTROLS = frozenset({"json", "output_fields"})
_LEGACY_OUTPUT_CONTROLS = frozenset({"extras"})
_FINVIZ_DOMAIN_FIELDS = frozenset({"equity_profile"})
_PYDANTIC_CONSTRAINT_KEYWORDS = frozenset(
    {"allow_inf_nan", "ge", "gt", "le", "lt", "max_length", "min_length"}
)

# These are intentional breaking migrations.  Keeping them here makes the scan
# catch accidental reintroduction of aliases after the implementation is
# simplified.
_SIMPLIFIED_PARAMETERS: Mapping[str, tuple[frozenset[str], frozenset[str]]] = {
    "denoise_list_methods": (frozenset({"no_extras"}), frozenset({"core_only"})),
    "forecast_backtest_run": (
        frozenset({"dimred_method", "dimred_params", "method"}),
        frozenset({"dimred", "methods"}),
    ),
    "forecast_barrier_optimize": (
        frozenset({"min_prob", "max_expected_time", "require_finite_time"}),
        frozenset({"candidate_filter"}),
    ),
    "forecast_barrier_prob": (
        frozenset(
            {
                "barrier_level",
                "tp_abs",
                "sl_abs",
                "tp_pct",
                "sl_pct",
                "tp_ticks",
                "sl_ticks",
                "tp_unit",
                "tp_value",
                "sl_unit",
                "sl_value",
            }
        ),
        frozenset({"barrier"}),
    ),
    "forecast_generate": (
        frozenset({"dimred_method", "dimred_params"}),
        frozenset({"dimred"}),
    ),
    "forecast_optimize_hints": (
        frozenset({"timeframe", "method", "dimred_method", "dimred_params"}),
        frozenset({"timeframes", "methods", "dimred"}),
    ),
    "forecast_tune_genetic": (
        frozenset({"method", "dimred_method", "dimred_params"}),
        frozenset({"methods", "dimred"}),
    ),
    "forecast_tune_optuna": (
        frozenset({"method", "dimred_method", "dimred_params"}),
        frozenset({"methods", "dimred"}),
    ),
    "labels_triple_barrier": (
        frozenset(
            {
                "tp_abs",
                "sl_abs",
                "tp_pct",
                "sl_pct",
                "tp_ticks",
                "sl_ticks",
                "tp_unit",
                "tp_value",
                "sl_unit",
                "sl_value",
            }
        ),
        frozenset({"barrier"}),
    ),
    "patterns_detect": (frozenset({"include_confirmed"}), frozenset()),
    "trade_close": (
        frozenset({"profit_only", "loss_only"}),
        frozenset({"pnl_filter"}),
    ),
    "trade_get_open": (
        frozenset({"profit_only", "loss_only"}),
        frozenset({"pnl_filter"}),
    ),
    "trade_place": (
        frozenset({"auto_close", "auto_close_on_sl_tp_fail", "sl", "tp"}),
        frozenset(),
    ),
    "trade_risk_analyze": (
        frozenset(
            {
                "desired_risk_pct",
                "sizing_method",
                "kelly_win_rate",
                "kelly_avg_win",
                "kelly_avg_loss",
                "kelly_fraction_multiplier",
                "kelly_max_risk_pct",
                "proposed_entry",
                "proposed_sl",
                "proposed_tp",
                "sl",
                "tp",
            }
        ),
        frozenset({"sizing"}),
    ),
}

_NUMERIC_NAMES_REQUIRING_LOWER_BOUND = frozenset(
    {
        "breakdown_limit",
        "horizon",
        "limit",
        "lookback",
        "max_wait_seconds",
        "min_overlap",
        "min_sample",
        "offset",
        "poll_interval_seconds",
        "timeout_seconds",
        "window",
    }
)


def _finding(
    findings: list[SchemaFinding],
    severity: str,
    code: str,
    tool: str,
    parameter: str,
    message: str,
) -> None:
    findings.append(SchemaFinding(severity, code, tool, parameter, message))


def _local_ref_names(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, dict):
        ref = value.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            refs.add(ref.rsplit("/", 1)[-1])
        for item in value.values():
            refs.update(_local_ref_names(item))
    elif isinstance(value, list):
        for item in value:
            refs.update(_local_ref_names(item))
    return refs


def _schema_allows_null(value: Any, defs: Mapping[str, Any]) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("type") == "null" or value.get("const") is None and "const" in value:
        return True
    enum = value.get("enum")
    if isinstance(enum, list) and None in enum:
        return True
    ref = value.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        return _schema_allows_null(defs.get(ref.rsplit("/", 1)[-1]), defs)
    for union_key in ("anyOf", "oneOf"):
        options = value.get(union_key)
        if isinstance(options, list) and any(_schema_allows_null(item, defs) for item in options):
            return True
    return False


def _schema_has_numeric_bound(value: Any, defs: Mapping[str, Any]) -> bool:
    if not isinstance(value, dict):
        return False
    if "minimum" in value or "exclusiveMinimum" in value:
        return True
    ref = value.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        return _schema_has_numeric_bound(defs.get(ref.rsplit("/", 1)[-1]), defs)
    return any(
        _schema_has_numeric_bound(item, defs)
        for key in ("anyOf", "oneOf", "allOf")
        for item in value.get(key, [])
        if isinstance(value.get(key), list)
    )


def _pydantic_constraint_paths(
    value: Any,
    path: tuple[str, ...] = (),
) -> list[tuple[tuple[str, ...], str]]:
    findings: list[tuple[tuple[str, ...], str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child_path = (*path, str(key))
            if key in _PYDANTIC_CONSTRAINT_KEYWORDS:
                findings.append((child_path, key))
            findings.extend(_pydantic_constraint_paths(item, child_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            findings.extend(_pydantic_constraint_paths(item, (*path, str(index))))
    return findings


def _runtime_parameters(func: Any) -> dict[str, inspect.Parameter]:
    signature = get_runtime_signature(func)
    return {
        name: parameter
        for name, parameter in signature.parameters.items()
        if parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }


def _defaults_equal(schema_default: Any, runtime_default: Any) -> bool:
    if isinstance(runtime_default, BaseModel):
        return schema_default == runtime_default.model_dump(mode="json")
    if schema_default == runtime_default:
        return True
    # Pydantic and JSON Schema can expose tuple defaults as JSON arrays.
    if isinstance(runtime_default, tuple) and isinstance(schema_default, list):
        return schema_default == list(runtime_default)
    return False


def _evaluate_tool(  # noqa: C901
    name: str,
    schema: Mapping[str, Any],
    func: Any,
    findings: list[SchemaFinding],
) -> None:
    properties_value = schema.get("properties")
    properties = properties_value if isinstance(properties_value, dict) else {}
    required_value = schema.get("required")
    required = set(required_value) if isinstance(required_value, list) else set()
    defs_value = schema.get("$defs")
    defs = defs_value if isinstance(defs_value, dict) else {}

    if schema.get("type") != "object":
        _finding(findings, "error", "top_level_type", name, "", "Schema must be an object.")
    if schema.get("additionalProperties") is not False:
        _finding(
            findings,
            "error",
            "unknown_parameters_allowed",
            name,
            "",
            "Top-level schema must reject undeclared parameters.",
        )

    for path, keyword in _pydantic_constraint_paths(schema):
        parameter = path[1] if len(path) > 1 and path[0] == "properties" else ""
        _finding(
            findings,
            "error",
            "non_json_schema_constraint",
            name,
            parameter,
            (
                f"Constraint {keyword!r} at /{'/'.join(path)} is a Pydantic "
                "keyword and is ignored by JSON Schema validators."
            ),
        )

    for control in sorted(_OUTPUT_CONTROLS - properties.keys()):
        _finding(
            findings,
            "error",
            "missing_output_control",
            name,
            control,
            "Every public tool must expose the shared output control.",
        )
    for control in sorted(_LEGACY_OUTPUT_CONTROLS & properties.keys()):
        _finding(
            findings,
            "error",
            "legacy_output_control",
            name,
            control,
            "Legacy output controls must not be public.",
        )
    if "fields" in properties and name not in _FINVIZ_DOMAIN_FIELDS:
        _finding(
            findings,
            "error",
            "ambiguous_fields_parameter",
            name,
            "fields",
            "Use output_fields for response projection; fields is reserved for domain selection.",
        )

    for parameter, property_schema in sorted(properties.items()):
        if not isinstance(property_schema, dict):
            _finding(
                findings,
                "error",
                "invalid_property_schema",
                name,
                parameter,
                "Property schema must be an object.",
            )
            continue
        description = " ".join(str(property_schema.get("description") or "").split())
        if not description:
            _finding(
                findings,
                "error",
                "missing_description",
                name,
                parameter,
                "Public parameters require a concise description.",
            )
        elif description.startswith("Value for "):
            _finding(
                findings,
                "error",
                "placeholder_description",
                name,
                parameter,
                "Public parameters require semantic help, not generated placeholder text.",
            )
        elif len(description) > 180:
            _finding(
                findings,
                "error",
                "long_description",
                name,
                parameter,
                f"Description is {len(description)} characters; maximum is 180.",
            )
        if parameter in required and _schema_allows_null(property_schema, defs):
            _finding(
                findings,
                "error",
                "required_nullable",
                name,
                parameter,
                "A required parameter must not accept null.",
            )
        default = property_schema.get("default", inspect._empty)
        enum = property_schema.get("enum")
        if default is not inspect._empty and isinstance(enum, list) and default not in enum:
            _finding(
                findings,
                "error",
                "default_outside_enum",
                name,
                parameter,
                "The declared default is not an allowed enum value.",
            )
        if (
            parameter in _NUMERIC_NAMES_REQUIRING_LOWER_BOUND
            and not _schema_has_numeric_bound(property_schema, defs)
        ):
            _finding(
                findings,
                "warning",
                "unbounded_numeric_parameter",
                name,
                parameter,
                "Common count, duration, or window parameter has no lower bound.",
            )

    unresolved = sorted(_local_ref_names(schema) - defs.keys())
    for ref_name in unresolved:
        _finding(
            findings,
            "error",
            "unresolved_ref",
            name,
            ref_name,
            "Local schema reference does not resolve.",
        )

    try:
        runtime = _runtime_parameters(func)
    except Exception as exc:
        _finding(
            findings,
            "error",
            "signature_unavailable",
            name,
            "",
            f"Could not inspect runtime callable: {exc}",
        )
        return

    for parameter in sorted(properties.keys() - runtime.keys()):
        _finding(
            findings,
            "error",
            "schema_only_parameter",
            name,
            parameter,
            "Parameter is declared but absent from the public runtime signature.",
        )
    for parameter in sorted(runtime.keys() - properties.keys()):
        _finding(
            findings,
            "error",
            "runtime_only_parameter",
            name,
            parameter,
            "Public runtime parameter is absent from the schema.",
        )

    runtime_required = {
        parameter
        for parameter, spec in runtime.items()
        if spec.default is inspect._empty
    }
    for parameter in sorted(required - runtime_required):
        _finding(
            findings,
            "error",
            "schema_required_runtime_optional",
            name,
            parameter,
            "Schema requires a parameter that has a runtime default.",
        )
    for parameter in sorted(runtime_required - required):
        _finding(
            findings,
            "error",
            "runtime_required_schema_optional",
            name,
            parameter,
            "Runtime requires a parameter that the schema marks optional.",
        )

    for parameter in sorted(properties.keys() & runtime.keys()):
        runtime_default = runtime[parameter].default
        if runtime_default is inspect._empty or runtime_default is None:
            continue
        property_schema = properties[parameter]
        if not isinstance(property_schema, dict):
            continue
        schema_default = property_schema.get("default", inspect._empty)
        if schema_default is inspect._empty:
            _finding(
                findings,
                "error",
                "missing_declared_default",
                name,
                parameter,
                "Runtime has a non-null default that the schema does not declare.",
            )
        elif not _defaults_equal(schema_default, runtime_default):
            _finding(
                findings,
                "error",
                "default_mismatch",
                name,
                parameter,
                f"Schema default {schema_default!r} differs from runtime default {runtime_default!r}.",
            )

    old_parameters, replacement_parameters = _SIMPLIFIED_PARAMETERS.get(
        name,
        (frozenset(), frozenset()),
    )
    for parameter in sorted(old_parameters & properties.keys()):
        _finding(
            findings,
            "error",
            "removed_parameter_reintroduced",
            name,
            parameter,
            "Removed compatibility parameter was reintroduced.",
        )
    for parameter in sorted(replacement_parameters - properties.keys()):
        _finding(
            findings,
            "error",
            "replacement_parameter_missing",
            name,
            parameter,
            "Canonical replacement parameter is missing.",
        )

    property_names = set(properties)
    allowed_singular_plural_pairs = {("symbol", "symbols")} if name == "wait_event" else set()
    for plural in sorted(item for item in property_names if item.endswith("s") and len(item) > 1):
        singular = plural[:-1]
        if singular in property_names and (singular, plural) not in allowed_singular_plural_pairs:
            _finding(
                findings,
                "warning",
                "singular_plural_pair",
                name,
                f"{singular},{plural}",
                "Sibling singular/plural parameters may express the same intent twice.",
            )
    for left, right in (("profit_only", "loss_only"), ("include", "exclude")):
        if {left, right}.issubset(property_names):
            _finding(
                findings,
                "warning",
                "contradictory_boolean_pair",
                name,
                f"{left},{right}",
                "Mutually exclusive booleans should usually be one enum parameter.",
            )
    for parameter in sorted(item for item in property_names if item.endswith("_method")):
        params_name = f"{parameter[:-7]}_params"
        if params_name in property_names:
            _finding(
                findings,
                "warning",
                "method_params_pair",
                name,
                f"{parameter},{params_name}",
                "Method and parameter-map siblings may be clearer as one nested object.",
            )


def evaluate_public_tool_schemas(*, include_gated: bool = False) -> SchemaEvaluationReport:
    """Bootstrap and evaluate all public tool schemas.

    Set ``include_gated`` only in a fresh process; gated tool registration occurs
    when its module is first imported.
    """

    if include_gated:
        os.environ["MTDATA_ENABLE_MARKET_DEPTH_FETCH"] = "1"

    from ..bootstrap.tools import bootstrap_tools
    from ._mcp_tools import get_tool_functions
    from .schema_attach import get_public_tool_schemas

    bootstrap_tools()
    schemas = get_public_tool_schemas()
    functions = get_tool_functions()
    findings: list[SchemaFinding] = []
    expected_count = (
        _EXPECTED_GATED_TOOL_COUNT if include_gated else _EXPECTED_DEFAULT_TOOL_COUNT
    )

    if len(schemas) != expected_count:
        _finding(
            findings,
            "error",
            "tool_count_mismatch",
            "*",
            "",
            f"Expected {expected_count} public tools but found {len(schemas)}.",
        )
    for name in sorted(schemas.keys() - functions.keys()):
        _finding(
            findings,
            "error",
            "schema_without_callable",
            name,
            "",
            "Schema has no registered runtime callable.",
        )
    for name in sorted(functions.keys() - schemas.keys()):
        _finding(
            findings,
            "error",
            "callable_without_schema",
            name,
            "",
            "Runtime callable has no canonical schema.",
        )
    for name in sorted(schemas.keys() & functions.keys()):
        _evaluate_tool(name, schemas[name], functions[name], findings)

    return SchemaEvaluationReport(
        tool_count=len(schemas),
        expected_tool_count=expected_count,
        findings=tuple(sorted(findings)),
    )


def format_schema_evaluation(report: SchemaEvaluationReport) -> str:
    """Render a stable, human-readable evaluator report."""

    status = "PASS" if report.ok else "FAIL"
    lines = [
        (
            f"Schema evaluation {status}: {report.tool_count}/"
            f"{report.expected_tool_count} tools, {len(report.errors)} errors, "
            f"{len(report.warnings)} warnings"
        )
    ]
    for finding in report.findings:
        location = finding.tool
        if finding.parameter:
            location = f"{location}.{finding.parameter}"
        lines.append(
            f"{finding.severity.upper():7} {finding.code:34} {location}: {finding.message}"
        )
    return "\n".join(lines)
