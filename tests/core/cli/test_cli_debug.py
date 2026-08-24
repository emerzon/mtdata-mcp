"""Debug and logging tests for mtdata.core.cli module.

Tests debug utilities, logging configuration, and environment handling.
"""

import argparse
import copy
import json
import logging
import os
import sys
import types
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from unittest.mock import MagicMock, call, patch

import pytest

from mtdata.core.data.requests import DataFetchCandlesRequest
from mtdata.core.trading.requests import (
    TradeGetOpenRequest,
    TradeHistoryRequest,
    TradeRiskAnalyzeRequest,
)
from mtdata.forecast.requests import ForecastGenerateRequest

# ---------------------------------------------------------------------------
# Fixture: ensure the cli module is importable with heavy deps mocked
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Clear env vars that influence debug/colour behaviour between tests."""
    monkeypatch.delenv("MTDATA_CLI_DEBUG", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("MT5_TIME_OFFSET_MINUTES", raising=False)


# We import lazily inside tests where heavy server machinery is needed,
# but the pure-logic helpers can be imported directly.
from mtdata.core.cli.api import (
    _configure_cli_logging,
    _debug,
    _debug_enabled,
)

# ========================================================================
# _debug_enabled / _debug
# ========================================================================


class TestDebugEnabled:
    @pytest.mark.parametrize(
        ("env_value", "expected"),
        [
            (None, False),
            ("1", True),
            ("true", True),
            ("0", False),
            ("false", False),
            ("no", False),
            ("", False),
            ("yes", True),
        ],
    )
    def test_debug_env_values(self, monkeypatch, env_value, expected):
        if env_value is None:
            monkeypatch.delenv("MTDATA_CLI_DEBUG", raising=False)
        else:
            monkeypatch.setenv("MTDATA_CLI_DEBUG", env_value)
        assert _debug_enabled() is expected


class TestDebug:
    def test_debug_prints_when_enabled(self, monkeypatch, capsys):
        monkeypatch.setenv("MTDATA_CLI_DEBUG", "1")
        _debug("test message")
        assert "test message" in capsys.readouterr().err

    def test_debug_silent_when_disabled(self, monkeypatch, capsys):
        monkeypatch.delenv("MTDATA_CLI_DEBUG", raising=False)
        _debug("should not appear")
        assert capsys.readouterr().err == ""


class TestConfigureCliLogging:
    def test_default_cli_logging_suppresses_mtdata_info(self):
        logger = logging.getLogger("mtdata")
        previous = logger.level
        previous_propagate = logger.propagate
        try:
            logger.setLevel(logging.NOTSET)
            logger.propagate = True
            _configure_cli_logging(verbose=False)
            assert logger.level == logging.WARNING
            assert logger.propagate is False
            assert any(
                isinstance(handler, logging.NullHandler) for handler in logger.handlers
            )
        finally:
            logger.setLevel(previous)
            logger.propagate = previous_propagate

    def test_verbose_cli_logging_restores_mtdata_info(self, monkeypatch):
        # --verbose no longer enables INFO logging; operation logs should only
        # stream when MTDATA_CLI_DEBUG is set. Validate both branches.
        logger = logging.getLogger("mtdata")
        previous = logger.level
        previous_propagate = logger.propagate
        try:
            logger.setLevel(logging.WARNING)
            logger.propagate = False
            monkeypatch.delenv("MTDATA_CLI_DEBUG", raising=False)
            _configure_cli_logging(verbose=True)
            assert logger.level == logging.WARNING
            assert logger.propagate is False

            monkeypatch.setenv("MTDATA_CLI_DEBUG", "1")
            _configure_cli_logging(verbose=False)
            assert logger.level == logging.INFO
            assert logger.propagate is True
        finally:
            logger.setLevel(previous)
            logger.propagate = previous_propagate


def test_cli_finish_log_matches_exit_predicate(caplog) -> None:
    from mtdata.core.cli.api import _invoke_cli_tool_function

    logger = logging.getLogger("mtdata.core.cli.api")
    previous_propagate = logger.propagate
    previous_level = logger.level
    logger.propagate = True
    logger.setLevel(logging.WARNING)
    try:
        with caplog.at_level(logging.WARNING, logger="mtdata.core.cli.api"):
            result = _invoke_cli_tool_function(
                lambda **_kwargs: {"message": "No action taken", "no_action": True},
                args=argparse.Namespace(),
                cmd_name="noop_tool",
                kwargs={},
            )
    finally:
        logger.propagate = previous_propagate
        logger.setLevel(previous_level)

    assert result == {"message": "No action taken", "no_action": True}
    assert any(
        "event=finish" in record.message and "success=False" in record.message
        for record in caplog.records
    )
