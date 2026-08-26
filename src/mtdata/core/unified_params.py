"""
Simple global parameter definitions for mtdata functions.

This module provides global parameters that work across all functions.
"""

import argparse
from typing import Optional

from ..shared.constants import DEFAULT_TIMEFRAME
from ..shared.output_precision import PRECISION_CHOICES
from ..shared.parameter_contracts import PARAMETER_HELP


def add_global_args_to_parser(
    parser: argparse.ArgumentParser,
    exclude_params: Optional[list] = None,
    *,
    suppress_defaults: bool = False,
) -> None:
    """Add all global parameters to an argument parser"""
    
    exclude_params = exclude_params or []
    
    # Timeframe
    if 'timeframe' not in exclude_params:
        timeframe_kwargs = {
            "help": PARAMETER_HELP["timeframe"],
        }
        if suppress_defaults:
            timeframe_kwargs["default"] = argparse.SUPPRESS
        else:
            timeframe_kwargs["default"] = DEFAULT_TIMEFRAME
        parser.add_argument(
            '--timeframe',
            **timeframe_kwargs,
        )
    
    # Output format: TOON by default, JSON when explicitly requested.
    if 'json' not in exclude_params:
        json_kwargs = {
            "action": "store_true",
            "dest": "json",
            "help": (
                "Output structured JSON. Default is TOON text unless "
                "MTDATA_OUTPUT_FORMAT=json is set."
            ),
        }
        if suppress_defaults:
            json_kwargs["default"] = argparse.SUPPRESS
        parser.add_argument(
            '--json',
            **json_kwargs,
        )

    if 'output_fields' not in exclude_params:
        fields_kwargs = {
            "dest": "output_fields",
            "default": None,
            "metavar": "FIELD[,FIELD...]",
            "help": (
                "Return only the selected output fields, plus success/error, "
                "symbol/timeframe, pagination, and warnings. Does not keep "
                "freshness, source, or other unselected trust fields."
            ),
        }
        if suppress_defaults:
            fields_kwargs["default"] = argparse.SUPPRESS
        parser.add_argument("--output-fields", **fields_kwargs)

    if 'precision' not in exclude_params:
        precision_kwargs = {
            "choices": PRECISION_CHOICES,
            "default": "auto",
            "help": (
                "TOON numeric display precision: auto (compact for most tools, "
                "full for forecast/trade analytics), compact/display to force "
                "token-saving output, or full/raw to disable rounding. JSON output "
                "is always full precision."
            ),
        }
        if suppress_defaults:
            precision_kwargs["default"] = argparse.SUPPRESS
        parser.add_argument("--precision", **precision_kwargs)

