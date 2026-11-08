import logging
import time

import pytest

from mtdata.shared.execution_logging import (
    log_operation_finish,
    log_operation_start,
    run_logged_operation,
)
from mtdata.shared.request_context import request_id_scope


def test_run_logged_operation_logs_finish_event(caplog):
    with caplog.at_level(logging.DEBUG, logger="mtdata.test.exec"):
        result = run_logged_operation(
            logging.getLogger("mtdata.test.exec"),
            operation="sample_op",
            item="abc",
            func=lambda: {"success": True},
        )

    assert result["success"] is True
    assert any(
        "event=finish operation=sample_op success=True" in record.message
        for record in caplog.records
    )


def test_request_context_is_added_to_operation_logs(caplog):
    with (
        request_id_scope("web-request-42"),
        caplog.at_level(logging.DEBUG, logger="mtdata.test.exec"),
    ):
        run_logged_operation(
            logging.getLogger("mtdata.test.exec"),
            operation="sample_op",
            func=lambda: {"success": True},
        )

    assert any(
        "request_id=web-request-42" in record.message
        for record in caplog.records
        if "operation=sample_op" in record.message
    )


def test_structured_failure_logs_warning_with_actionable_fields(caplog):
    with caplog.at_level(logging.WARNING, logger="mtdata.test.exec"):
        result = run_logged_operation(
            logging.getLogger("mtdata.test.exec"),
            operation="trade_place",
            symbol="EURUSD",
            func=lambda: {
                "success": False,
                "error": "Broker rejected order\nrequest",
                "error_code": "order_rejected",
                "retcode": 10016,
                "retcode_name": "TRADE_RETCODE_INVALID_STOPS",
            },
        )

    assert result["success"] is False
    record = next(
        record
        for record in caplog.records
        if "event=finish operation=trade_place success=False" in record.message
    )
    assert record.levelno == logging.WARNING
    assert "error=Broker rejected order request" in record.message
    assert "error_code=order_rejected" in record.message
    assert "retcode=10016" in record.message
    assert "retcode_name=TRADE_RETCODE_INVALID_STOPS" in record.message


def test_successful_operation_remains_debug_only(caplog):
    with caplog.at_level(logging.DEBUG, logger="mtdata.test.exec"):
        run_logged_operation(
            logging.getLogger("mtdata.test.exec"),
            operation="sample_op",
            func=lambda: {"success": True},
        )

    record = next(
        record
        for record in caplog.records
        if "event=finish operation=sample_op success=True" in record.message
    )
    assert record.levelno == logging.DEBUG


def test_run_logged_operation_logs_exception_and_reraises(caplog):
    with caplog.at_level(logging.ERROR, logger="mtdata.test.exec"), pytest.raises(RuntimeError, match="boom"):
        run_logged_operation(
            logging.getLogger("mtdata.test.exec"),
            operation="sample_fail",
            func=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )

    assert any(
        "event=error operation=sample_fail" in record.message
        for record in caplog.records
    )


def test_nested_same_operation_logs_single_finish_event(caplog):
    logger = logging.getLogger("mtdata.test.exec")

    with caplog.at_level(logging.DEBUG, logger="mtdata.test.exec"):
        result = run_logged_operation(
            logger,
            operation="sample_op",
            func=lambda: run_logged_operation(
                logger,
                operation="sample_op",
                func=lambda: {"success": True},
            ),
        )

    assert result["success"] is True
    finish_records = [
        record
        for record in caplog.records
        if "event=finish operation=sample_op success=True" in record.message
    ]
    assert len(finish_records) == 1


def test_nested_different_operations_still_log_both_finish_events(caplog):
    logger = logging.getLogger("mtdata.test.exec")

    with caplog.at_level(logging.DEBUG, logger="mtdata.test.exec"):
        result = run_logged_operation(
            logger,
            operation="outer_op",
            func=lambda: run_logged_operation(
                logger,
                operation="inner_op",
                func=lambda: {"success": True},
            ),
        )

    assert result["success"] is True
    assert any("event=finish operation=outer_op success=True" in record.message for record in caplog.records)
    assert any("event=finish operation=inner_op success=True" in record.message for record in caplog.records)


def test_manual_nested_same_operation_logs_single_finish_event(caplog):
    logger = logging.getLogger("mtdata.test.exec")
    outer_started = time.perf_counter()
    inner_started = time.perf_counter()

    with caplog.at_level(logging.DEBUG, logger="mtdata.test.exec"):
        log_operation_start(logger, operation="manual_op")
        log_operation_start(logger, operation="manual_op")
        log_operation_finish(logger, operation="manual_op", started_at=inner_started, success=True)
        log_operation_finish(logger, operation="manual_op", started_at=outer_started, success=True)

    finish_records = [
        record
        for record in caplog.records
        if "event=finish operation=manual_op success=True" in record.message
    ]
    assert len(finish_records) == 1
