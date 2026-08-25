from mtdata.core.error_envelope import (
    build_error_payload,
    canonical_documentation_url,
    normalize_error_payload,
)
from mtdata.core.request_context import (
    current_request_id,
    ensure_request_id_scope,
    request_id_scope,
)


def test_build_error_payload_adds_common_remediation():
    out = build_error_payload(
        "MT5 connection failed",
        code="mt5_connection_error",
        operation="data_fetch_candles",
        request_id="req123",
    )

    assert out["request_id"] == "req123"
    assert "MetaTrader 5 is running" in out["remediation"]
    assert out["related_tools"] == ["symbols_list"]


def test_build_error_payload_uses_bound_request_id():
    with request_id_scope("bound-request-7"):
        out = build_error_payload("broken", code="test_error")

    assert out["request_id"] == "bound-request-7"


def test_ensure_request_id_scope_generates_and_cleans_up_identifier():
    assert current_request_id() is None

    with ensure_request_id_scope() as request_id:
        assert len(request_id) == 12
        assert current_request_id() == request_id

    assert current_request_id() is None


def test_ensure_request_id_scope_preserves_transport_identifier():
    with request_id_scope("transport-request-9"):
        with ensure_request_id_scope() as request_id:
            assert request_id == "transport-request-9"
            assert current_request_id() == request_id


def test_build_error_payload_keeps_explicit_guidance():
    out = build_error_payload(
        "No such method",
        code="forecast_generate_error",
        operation="forecast_generate",
        request_id="req123",
        remediation="Choose theta.",
        related_tools=["forecast_list_methods"],
        valid_values={"method": ["theta"]},
        example="mtdata-cli forecast_generate EURUSD --method theta",
    )

    assert out["remediation"] == "Choose theta."
    assert out["valid_values"] == {"method": ["theta"]}
    assert out["example"].endswith("--method theta")


def test_forecast_train_errors_point_to_trainable_method_discovery():
    out = build_error_payload(
        "Method 'ets' does not support separate training.",
        code="tool_error",
        operation="forecast_train",
    )

    assert out["remediation"] == (
        "Choose a trainable method with forecast_list_methods "
        "--supports-training true, then retry forecast_train."
    )
    assert out["related_tools"] == ["forecast_list_methods"]


def test_canonical_documentation_url_maps_repo_paths_to_github():
    url = canonical_documentation_url("docs/CLI.md")
    assert url.startswith("https://github.com/emerzon/mtdata-mcp/blob/")
    assert url.endswith("/docs/CLI.md")

    with_fragment = canonical_documentation_url(
        "docs/FORECAST.md#background-training--model-store"
    )
    assert with_fragment.endswith(
        "/docs/FORECAST.md#background-training--model-store"
    )

    already_https = canonical_documentation_url("https://example.com/guide")
    assert already_https == "https://example.com/guide"


def test_build_error_payload_rewrites_documentation_to_https():
    out = build_error_payload(
        "Unknown command",
        code="cli_unknown_command",
        operation="cli",
        documentation="docs/CLI.md",
    )

    assert out["documentation"].startswith("https://github.com/emerzon/mtdata-mcp/blob/")
    assert out["documentation"].endswith("/docs/CLI.md")


def test_missing_forecast_task_id_points_to_task_discovery():
    out = build_error_payload(
        "Missing required argument: task_id",
        code="cli_missing_required",
        operation="forecast_task_cancel",
    )

    assert "Use forecast_task_list to find a task_id" in out["remediation"]
    assert out["related_tools"] == ["forecast_task_list"]


def test_missing_forecast_model_id_points_to_model_store_discovery():
    out = build_error_payload(
        "Missing required argument: model_id",
        code="cli_missing_required",
        operation="forecast_models_delete",
    )

    assert "forecast_models_list" in out["remediation"]
    assert out["related_tools"] == ["forecast_models_list"]


def test_barrier_usage_errors_point_to_barrier_help():
    out = build_error_payload(
        "Invalid method: closed_form",
        code="cli_invalid_arguments",
        operation="forecast_barrier_optimize",
    )

    assert "forecast_barrier_optimize --help" in out["remediation"]
    assert out["related_tools"] == ["forecast_barrier_prob"]


def test_generic_method_errors_use_the_failing_operation_help():
    out = build_error_payload(
        "Invalid method. Valid options: pearson, spearman",
        code="invalid_method",
        operation="correlation_matrix",
    )

    assert out["remediation"] == (
        "Use this operation's --help and choose one of the listed method values."
    )
    assert "related_tools" not in out


def test_normalize_error_payload_adds_symbol_lookup_guidance():
    out = normalize_error_payload(
        {
            "error": "Symbol not found",
            "error_code": "symbol_not_found",
            "request_id": "req123",
        },
        operation="symbols_describe",
    )

    assert out["remediation"].startswith("Use symbols_list")
    assert out["related_tools"] == ["symbols_list"]


def test_finviz_symbol_errors_use_provider_specific_discovery_guidance():
    out = normalize_error_payload(
        {
            "error": "Symbol not found",
            "error_code": "finviz_symbol_not_found",
        },
        operation="news",
    )

    assert "standard US equity ticker" in out["remediation"]
    assert "MT5 broker suffix" in out["remediation"]
    assert out["related_tools"] == ["screener"]


def test_normalize_error_payload_preserves_specific_code_from_warnings():
    out = normalize_error_payload(
        {
            "error": "Not enough valid symbol data fetched.",
            "error_code": "insufficient_symbols",
            "warnings": [
                "Symbol NOTAREALSYM was not found in MT5.",
                "Symbol NOTAREALSYM was not found in MT5.",
            ],
            "remediation": "Increase the lookback.",
        },
        operation="correlation_matrix",
    )

    assert out["error_code"] == "insufficient_symbols"
    assert out["warnings"] == ["Symbol NOTAREALSYM was not found in MT5."]
    assert out["remediation"] == "Increase the lookback."
    assert "related_tools" not in out


def test_normalize_error_payload_classifies_generic_code_from_warnings():
    out = normalize_error_payload(
        {
            "error": "Not enough valid symbol data fetched.",
            "error_code": "tool_error",
            "warnings": ["Symbol NOTAREALSYM was not found in MT5."],
        },
        operation="correlation_matrix",
    )

    assert out["error_code"] == "symbol_not_found"
    assert out["remediation"].startswith("Use symbols_list")
    assert out["related_tools"] == ["symbols_list"]


def test_normalize_error_payload_classifies_forecast_symbol_failure():
    out = normalize_error_payload(
        {
            "error": "Symbol 'NOTAREALSYM' not found in MT5 terminal.",
            "error_code": "forecast_generate_error",
        },
        operation="forecast_generate",
    )

    assert out["error_code"] == "symbol_not_found"
    assert out["remediation"].startswith("Use symbols_list")
    assert out["related_tools"] == ["symbols_list"]


def test_normalize_error_payload_classifies_any_forecast_operation_catch_all():
    out = normalize_error_payload(
        {
            "error": "Symbol 'NOTAREALSYM' was not found in MT5.",
            "error_code": "forecast_conformal_intervals_error",
        },
        operation="forecast_conformal_intervals",
    )

    assert out["error_code"] == "symbol_not_found"
    assert out["remediation"].startswith("Use symbols_list")
    assert out["related_tools"] == ["symbols_list"]


def test_forecast_operation_catch_all_canonicalizes_reversed_date_range():
    out = normalize_error_payload(
        {
            "error": "start must be before or equal to end.",
            "error_code": "forecast_conformal_intervals_error",
        },
        operation="forecast_conformal_intervals",
    )

    assert out["error_code"] == "invalid_date_range"
    assert out["remediation"] == (
        "Set start to a timestamp earlier than or equal to end."
    )


def test_normalize_error_payload_does_not_override_dependency_code():
    out = normalize_error_payload(
        build_error_payload(
            "finviz: symbol XYZ not found in screener",
            code="dependency_missing",
            operation="news_fetch",
            details={"provider": "finviz"},
        )
    )

    assert out["error_code"] == "dependency_missing"
    assert "optional dependency group" in out["remediation"]


def test_normalize_error_payload_canonicalizes_date_ranges_and_details():
    out = normalize_error_payload(
        {
            "error": "Error detecting regimes: start_datetime must be before end_datetime",
            "error_code": "tool_error",
            "details": [
                "start_datetime must be before end_datetime",
                "start_datetime must be before end_datetime",
            ],
            "remediation": "Run forecast_list_methods.",
        },
        operation="regime_detect",
    )

    assert out["error_code"] == "invalid_date_range"
    assert out["error"] == "start must be before or equal to end."
    assert out["details"] == ["start_datetime must be before end_datetime"]
    assert out["remediation"] == "Set start to a timestamp earlier than or equal to end."


def test_normalize_error_payload_preserves_malformed_datetime_diagnosis():
    out = normalize_error_payload(
        {
            "error": "Could not parse historical datetime bound(s): start='bad'.",
            "error_code": "invalid_datetime",
            "details": {
                "invalid_fields": [{"field": "start", "value": "bad"}]
            },
        },
        operation="correlation_matrix",
    )

    assert out["error_code"] == "invalid_datetime"
    assert "start='bad'" in out["error"]
    assert out["details"]["invalid_fields"][0]["field"] == "start"
    assert "ISO 8601" in out["remediation"]
