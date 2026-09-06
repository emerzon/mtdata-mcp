"""Pure tests for indicators helpers, core wrappers, and shared schema.

Covers:
  - mtdata.utils.indicators  (TA indicator helpers)
  - mtdata.core.indicators    (thin wrappers / delegates)
  - mtdata.shared.schema      (schema validation/parsing)

Utility helpers (utils.utils, formatting, parse_kv) live in dedicated test modules.
"""

import logging
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import pytest
from typing_extensions import TypedDict

from mtdata.shared.schema import (
    _DENOISE_METHODS,
    _PIVOT_METHODS,
    _SIMPLIFY_METHODS,
    _SIMPLIFY_MODES,
    PARAM_HINTS,
    _allow_null,
    _ensure_defs,
    _is_typed_dict_type,
    _parameters_obj,
    _type_hint_to_schema,
    apply_param_hints,
    apply_timeframe_ref,
    build_minimal_schema,
    complex_defs,
    enrich_schema_with_shared_defs,
    get_function_info,
    get_shared_enum_lists,
    shared_defs,
)

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------
from mtdata.utils.indicators import (
    _apply_ta_indicators,
    _estimate_warmup_bars,
    _find_unknown_ta_indicators,
    _parse_doc_default_value,
    _parse_ti_number,
    _parse_ti_specs,
    clean_help_text,
    infer_defaults_from_doc,
    list_ta_indicators,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RS = np.random.RandomState(42)


def _make_ohlcv_df(n: int = 100) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame with deterministic data."""
    close = 100.0 + RS.standard_normal(n).cumsum()
    high = close + RS.uniform(0.5, 2.0, n)
    low = close - RS.uniform(0.5, 2.0, n)
    open_ = close + RS.uniform(-1.0, 1.0, n)
    volume = RS.randint(100, 10000, n).astype(float)
    time_vals = np.arange(n) * 3600 + 1_600_000_000
    return pd.DataFrame({
        "time": time_vals,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


# ===================================================================
# 1. mtdata.utils.indicators
# ===================================================================
class TestCleanHelpText:
    def test_empty_string(self):
        assert clean_help_text("") == ""

    def test_non_string_returns_empty(self):
        assert clean_help_text(None) == ""  # type: ignore[arg-type]
        assert clean_help_text(123) == ""  # type: ignore[arg-type]

    def test_strips_backspace_overstrikes(self):
        # "A\x08B" should become "B" after overstrike removal
        result = clean_help_text("A\x08B")
        assert "\x08" not in result

    def test_with_func_name_finds_signature(self):
        text = "some preamble\nema(close, length=10)\n  description here"
        result = clean_help_text(text, func_name="ema")
        assert result.startswith("ema(")

    def test_removes_method_of_suffix(self):
        text = "sma(close) method of pandas_ta.overlap\n  desc"
        result = clean_help_text(text, func_name="sma")
        assert "method of" not in result.split("\n")[0]

    def test_removes_method_of_on_second_line(self):
        text = "sma(close)\n  method of pandas_ta.core\n  desc"
        result = clean_help_text(text, func_name="sma")
        assert "method of" not in result


class TestParseDocDefaultValue:
    def test_integer(self):
        assert _parse_doc_default_value("42") == 42
        assert isinstance(_parse_doc_default_value("42"), int)

    def test_float(self):
        assert _parse_doc_default_value("3.14") == pytest.approx(3.14)

    def test_non_numeric(self):
        assert _parse_doc_default_value("hello") == "hello"

    def test_negative(self):
        assert _parse_doc_default_value("-5") == -5


class TestParseTiNumber:
    def test_integer_float_normalized(self):
        assert _parse_ti_number("20.0") == 20
        assert isinstance(_parse_ti_number("20.0"), int)

    def test_true_float(self):
        assert _parse_ti_number("0.5") == pytest.approx(0.5)

    def test_non_numeric(self):
        with pytest.raises(ValueError, match="must be a finite number"):
            _parse_ti_number("abc")

    def test_int_string(self):
        assert _parse_ti_number("14") == 14


class TestInferDefaultsFromDoc:
    def test_from_signature_line(self):
        params = [{"name": "length"}, {"name": "offset"}]
        doc = "ema(close, length=20, offset=0)\n  desc"
        infer_defaults_from_doc("ema", doc, params)
        assert params[0].get("default") == 20
        assert params[1].get("default") == 0

    def test_from_body_default_keyword(self):
        params = [{"name": "length"}]
        doc = "sma(close)\n  length: Default: 14"
        infer_defaults_from_doc("sma", doc, params)
        assert params[0].get("default") == 14

    def test_empty_doc(self):
        params = [{"name": "length"}]
        infer_defaults_from_doc("sma", "", params)
        assert "default" not in params[0]

    def test_existing_default_not_overwritten(self):
        params = [{"name": "length", "default": 99}]
        doc = "ema(close, length=20)\n  desc"
        infer_defaults_from_doc("ema", doc, params)
        assert params[0]["default"] == 99

    def test_parameter_names_are_line_anchored(self):
        params = [{"name": "d"}, {"name": "c"}, {"name": "length"}]
        doc = """Indicator description mentions default factor of 4.236.
        length (int): Period. Default: 14
        d (int): Smoothing. Default: 3
        c (float): Multiplier. Default: 1
        """

        infer_defaults_from_doc("indicator", doc, params)

        assert params == [
            {"name": "d", "default": 3},
            {"name": "c", "default": 1},
            {"name": "length", "default": 14},
        ]

    def test_wrapped_parameter_description_keeps_default(self):
        params = [{"name": "talib"}, {"name": "offset"}]
        doc = """Args:
        talib (bool): If TA Lib is installed, returns the TA Lib
            version. Default: True
        offset (int): Bars to offset. Default: 0
        """

        infer_defaults_from_doc("stoch", doc, params)

        assert params == [
            {"name": "talib", "default": True},
            {"name": "offset", "default": 0},
        ]


class TestParseTiSpecs:
    def test_empty_string(self):
        assert _parse_ti_specs("") == []

    def test_simple_name(self):
        specs = _parse_ti_specs("rsi")
        assert len(specs) == 1
        assert specs[0][0] == "rsi"

    def test_name_with_args(self):
        specs = _parse_ti_specs("ema(20)")
        assert specs[0][0] == "ema"
        # 20 is parsed as a positional arg (not kwargs since it's inside parens)
        assert 20 in specs[0][1]

    def test_name_with_kwargs(self):
        specs = _parse_ti_specs("macd(fast=12,slow=26)")
        name, args, kwargs = specs[0]
        assert name == "macd"
        assert kwargs["fast"] == 12
        assert kwargs["slow"] == 26

    def test_multiple_specs(self):
        specs = _parse_ti_specs("rsi(14),ema(20),macd")
        assert len(specs) == 3

    def test_trailing_number_in_name(self):
        specs = _parse_ti_specs("EMA21")
        name, args, kwargs = specs[0]
        assert name == "ema"
        assert kwargs.get("length") == 21

    def test_literal_digit_suffixed_indicator_takes_precedence(self, monkeypatch):
        import mtdata.utils.indicators as indicators_mod

        monkeypatch.setattr(indicators_mod.pta, "hlc3", lambda: None, raising=False)
        indicators_mod._is_available_ta_indicator.cache_clear()

        assert _parse_ti_specs("hlc3") == [("hlc3", [], {})]

    def test_cdl_name_with_trailing_digits_is_not_rewritten(self):
        specs = _parse_ti_specs("CDL_FAKE12")
        name, args, kwargs = specs[0]
        assert name == "cdl_fake12"
        assert args == []
        assert kwargs == {}

    def test_cdl_name_with_trailing_digits_is_reported_as_unknown(self):
        assert _find_unknown_ta_indicators("CDL_FAKE12") == ["cdl_fake12"]

    def test_nested_parens_not_split(self):
        specs = _parse_ti_specs("rsi(14),macd(12,26,9)")
        assert len(specs) == 2

    def test_name_with_positional_args(self):
        specs = _parse_ti_specs("stoch(14,3,3)")
        name, args, kwargs = specs[0]
        assert name == "stoch"
        assert 14 in args or kwargs.get("length") == 14

    def test_canonical_bbands_name_is_preserved(self):
        specs = _parse_ti_specs("bbands(20)")
        assert [name for name, _args, _kwargs in specs] == ["bbands"]

    def test_historical_bollinger_nicknames_are_not_aliased(self):
        specs = _parse_ti_specs("bb(20),boll(20),bollinger_bands(20)")
        assert [name for name, _args, _kwargs in specs] == ["bb", "boll", "bollinger_bands"]
        unknown = _find_unknown_ta_indicators("bb(20),boll(20),bollinger_bands(20)")
        assert set(unknown) >= {"bb", "boll", "bollinger_bands"}

    def test_indicator_call_failure_is_not_retried_with_different_bindings(self, monkeypatch):
        import mtdata.utils.indicators as indicators_mod

        calls = []

        def broken_indicator(close, length=14):
            calls.append((close, length))
            raise RuntimeError("invalid calculation")

        monkeypatch.setattr(
            indicators_mod,
            "pta",
            type("PtaStub", (), {"rsi": staticmethod(broken_indicator)})(),
        )
        # Availability is cached against the module-level pta object; clear so the
        # stubbed rsi is visible to unknown-indicator checks.
        indicators_mod._is_available_ta_indicator.cache_clear()

        with pytest.raises(
            ValueError,
            match=r"Indicator 'rsi' failed with parameters close, length: invalid calculation",
        ):
            _apply_ta_indicators(_make_ohlcv_df(20), "rsi(10)")

        assert len(calls) == 1

    def test_indicator_lookup_results_are_cached(self, monkeypatch):
        import mtdata.utils.indicators as indicators_mod

        class _ProxyPta:
            def __init__(self) -> None:
                self.lookups = 0

            def __getattr__(self, name: str):
                self.lookups += 1
                if name == "ema":
                    return lambda *args, **kwargs: None
                raise AttributeError(name)

        proxy = _ProxyPta()
        monkeypatch.setattr(indicators_mod, "pta", proxy)
        indicators_mod._is_available_ta_indicator.cache_clear()

        assert indicators_mod._find_unknown_ta_indicators("ema(20),ema(50)") == []
        assert indicators_mod._find_unknown_ta_indicators("ema(20)") == []
        assert proxy.lookups == 1

        indicators_mod._is_available_ta_indicator.cache_clear()

    def test_indicator_lookup_cache_is_bounded(self, monkeypatch):
        import mtdata.utils.indicators as indicators_mod

        class _ProxyPta:
            def __getattr__(self, name: str):
                raise AttributeError(name)

        monkeypatch.setattr(indicators_mod, "pta", _ProxyPta())
        indicators_mod._is_available_ta_indicator.cache_clear()

        for idx in range(600):
            indicators_mod._is_available_ta_indicator(f"unknown_indicator_{idx}")

        assert indicators_mod._is_available_ta_indicator.cache_info().currsize <= 512

        indicators_mod._is_available_ta_indicator.cache_clear()

    def test_list_ta_indicators_uses_native_category_catalog(self):
        items = list_ta_indicators(detailed=False)
        by_name = {str(item.get("name")): item for item in items}

        assert "cdl" not in by_name
        assert "cdl_pattern" in by_name
        for name in ("atr", "natr", "bbands", "kc", "donchian"):
            assert name in by_name
            assert by_name[name]["category"] == "volatility"

    def test_list_ta_indicators_caches_discovery_without_sharing_mutations(self):
        import mtdata.utils.indicators as indicators_mod

        indicators_mod._list_ta_indicators_cached.cache_clear()
        first = indicators_mod.list_ta_indicators(detailed=True)
        first[0]["name"] = "mutated"
        second = indicators_mod.list_ta_indicators(detailed=True)

        assert second[0]["name"] != "mutated"
        assert indicators_mod._list_ta_indicators_cached.cache_info().hits == 1
        indicators_mod._list_ta_indicators_cached.cache_clear()


class TestEstimateWarmupBars:
    def test_empty_spec(self):
        assert _estimate_warmup_bars("") == 0
        assert _estimate_warmup_bars(None) == 0

    def test_sma(self):
        result = _estimate_warmup_bars("sma(20)")
        assert result >= 50  # min(max_warmup*3, 50)

    def test_rsi(self):
        result = _estimate_warmup_bars("rsi(14)")
        assert result == 350

    def test_macd(self):
        result = _estimate_warmup_bars("macd(12,26,9)")
        assert result == 650

    def test_macd_uses_default_signal_when_missing(self):
        result = _estimate_warmup_bars("macd(fast=12,slow=26)")
        assert result == 650

    def test_bbands(self):
        result = _estimate_warmup_bars("bbands(20)")
        assert result >= 50

    def test_stoch(self):
        result = _estimate_warmup_bars("stoch(14,3,3)")
        assert result >= 50

    def test_unknown_indicator(self):
        result = _estimate_warmup_bars("someunknown")
        assert result >= 50  # default warmup 50 * 3

    def test_multiple_indicators_takes_max(self):
        result = _estimate_warmup_bars("sma(50),rsi(14)")
        assert result == 350

    def test_rsi_tip_converges_across_requested_limits(self):
        count = 1200
        idx = np.arange(count, dtype=float)
        close = 100.0 + np.cumsum(
            0.02 * np.sin(idx / 7.0)
            + 0.01 * np.cos(idx / 13.0)
            + 0.003
        )
        source = pd.DataFrame(
            {
                "time": idx,
                "open": close,
                "high": close + 0.1,
                "low": close - 0.1,
                "close": close,
                "tick_volume": 100,
            }
        )
        warmup = _estimate_warmup_bars("rsi(14)")
        tips = []

        for limit in (3, 5, 10, 50):
            frame = source.iloc[-(limit + warmup):].copy()
            columns = _apply_ta_indicators(frame, "rsi(14)")
            tips.append(float(frame[columns[0]].iloc[-1]))

        assert max(tips) - min(tips) < 1e-9


# ===================================================================
# 2. mtdata.core.indicators  (canonical utility imports)
# ===================================================================
class TestCoreIndicatorsWrappers:
    def test_doc_defaults_preserve_unknown_identifiers(self):
        assert _parse_doc_default_value("42") == 42
        assert _parse_doc_default_value("3.14") == pytest.approx(3.14)
        assert _parse_doc_default_value("bad") == "bad"

    def test_clean_help_text_delegation(self):
        """core.indicators._clean_help_text uses the canonical indicators utility."""
        from mtdata.core.indicators import _clean_help_text as core_clean
        assert core_clean("") == ""
        assert core_clean(None) == ""  # type: ignore[arg-type]

    def test_indicators_describe_returns_structured_documentation(self, monkeypatch):
        from mtdata.core import indicators as core_indicators

        sample_doc = """rsi(close, length=14) method of pandas_ta.momentum
Relative Strength Index for momentum.
Sources:
https://example.com/rsi
Calculation:
1) Compute average gains and losses over length.
Args:
length (int): Window length.
scalar (float): Optional scalar multiplier.
Interpretation:
Values above 70 often indicate overbought conditions.
"""
        monkeypatch.setattr(
            core_indicators,
            "_list_ta_indicators",
            lambda detailed=True: [
                {
                    "name": "rsi",
                    "category": "momentum",
                    "params": [{"name": "length", "default": 14}, {"name": "scalar"}],
                    "description": sample_doc,
                }
            ],
        )

        raw_describe = getattr(core_indicators.indicators_describe, "__wrapped__", core_indicators.indicators_describe)
        out = raw_describe("rsi", detail="full")
        assert out["success"] is True
        assert out["detail"] == "full"
        indicator = out["indicator"]
        assert "method of" not in indicator["description"].lower()
        docs = indicator["documentation"]
        assert docs["calculation"]
        assert docs["interpretation"]
        assert docs["sources"] == ["https://example.com/rsi"]
        assert "parameters" not in docs
        params = {p["name"]: p for p in indicator["params"]}
        assert params["length"]["description"] == "Window length."
        assert params["scalar"]["description"] == "Optional scalar multiplier."
        assert "usage_examples" not in indicator

    def test_indicators_describe_cleans_signature_and_preserves_multiline_docs(self, monkeypatch):
        from mtdata.core import indicators as core_indicators

        sample_doc = """Python Library Documentation: function rsi in module pandas_ta_classic.momentum.rsi

rsi(
close: Series,
length: Optional[int] = None,
talib: Optional[bool] = None
) -> Optional[Series]
Relative Strength Index
Tracks momentum across the selected lookback window.
Useful for overbought and oversold analysis.
Parameters:
talib (bool): If TA Lib is installed and talib is True,
    Returns the TA Lib version. Default: True
Interpretation:
Values above 70 often indicate overbought conditions.
Values below 30 often indicate oversold conditions.
"""
        monkeypatch.setattr(
            core_indicators,
            "_list_ta_indicators",
            lambda detailed=True: [
                {
                    "name": "rsi",
                    "category": "momentum",
                    "params": [{"name": "talib"}],
                    "description": sample_doc,
                }
            ],
        )

        raw_describe = getattr(core_indicators.indicators_describe, "__wrapped__", core_indicators.indicators_describe)
        out = raw_describe("rsi", detail="full")
        indicator = out["indicator"]
        docs = indicator["documentation"]
        assert "parameters" not in docs
        params = {p["name"]: p for p in indicator["params"]}

        assert "pandas_ta_classic" not in indicator["description"]
        assert "rsi(" not in indicator["description"]
        assert "Useful for overbought and oversold analysis." in indicator["description"]
        assert docs["interpretation"] == (
            "Values above 70 often indicate overbought conditions.\n"
            "Values below 30 often indicate oversold conditions."
        )
        assert params["talib"]["description"] == (
            "If TA Lib is installed and talib is True, Returns the TA Lib version. Default: True"
        )

    def test_indicators_describe_compact_and_standard_detail(self, monkeypatch):
        from mtdata.core import indicators as core_indicators

        sample_doc = """Relative Strength Index
Tracks momentum.
Parameters:
length (int): Window length.
scalar (float): Optional scalar multiplier.
Interpretation:
Values above 70 often indicate overbought conditions.
"""
        monkeypatch.setattr(
            core_indicators,
            "_list_ta_indicators",
            lambda detailed=True: [
                {
                    "name": "rsi",
                    "category": "momentum",
                    "params": [{"name": "length", "default": 14}, {"name": "scalar"}],
                    "description": sample_doc,
                }
            ],
        )

        raw_describe = getattr(core_indicators.indicators_describe, "__wrapped__", core_indicators.indicators_describe)
        compact = raw_describe("rsi")
        standard = raw_describe("rsi", detail="standard")

        assert compact["detail"] == "compact"
        indicator = compact["indicator"]
        assert indicator["name"] == "rsi"
        assert indicator["category"] == "momentum"
        assert indicator["description"] == "Relative Strength Index"
        assert indicator["params"] == [
            {"name": "length", "default": 14},
            {"name": "scalar"},
        ]
        assert indicator["interpretation"] == (
            "Values above 70 often indicate overbought conditions."
        )
        assert indicator["see_also"] == ["stochrsi", "tsi", "mfi"]
        assert indicator["trading_context"]["common_use"] == (
            "overbought/oversold, momentum reversal, and divergence checks"
        )
        assert "compact_spec" in indicator["usage"]
        assert "documentation" not in indicator
        assert standard["detail"] == "standard"
        params = {p["name"]: p for p in standard["indicator"]["params"]}
        assert params["length"]["description"] == "Window length."
        assert "documentation" not in standard["indicator"]
        assert "trading_context" in standard["indicator"]
        assert "usage" in standard["indicator"]

    def test_indicators_describe_logs_finish_event(self, monkeypatch, caplog):
        from mtdata.core import indicators as core_indicators

        monkeypatch.setattr(
            core_indicators,
            "_list_ta_indicators",
            lambda detailed=True: [
                {
                    "name": "rsi",
                    "category": "momentum",
                    "params": [],
                    "description": "Relative Strength Index.",
                }
            ],
        )

        raw_describe = getattr(core_indicators.indicators_describe, "__wrapped__", core_indicators.indicators_describe)
        with caplog.at_level(logging.DEBUG, logger=core_indicators.logger.name):
            out = raw_describe("rsi")

        assert out["success"] is True
        assert any(
            "event=finish" in record.message and "operation=indicators_describe" in record.message
            for record in caplog.records
        )

    def test_indicators_list_reports_when_results_are_truncated(self, monkeypatch):
        from mtdata.core import indicators as core_indicators

        monkeypatch.setattr(
            core_indicators,
            "_list_ta_indicators",
            lambda detailed=False: [
                {
                    "name": f"ind_{i:02d}",
                    "category": "momentum",
                    "description": "",
                    "params": [{"name": "length", "default": 14}],
                }
                for i in range(30)
            ],
        )

        raw_list = getattr(core_indicators.indicators_list, "__wrapped__", core_indicators.indicators_list)
        out = raw_list(category="momentum", limit=25)

        assert out["success"] is True
        assert out["count"] == 25
        assert out["data"][0]["params_count"] == 1
        assert set(out["data"][0]) == {"name", "category", "params_count"}
        assert out["pagination"] == {
            "total": 30,
            "returned": 25,
            "offset": 0,
            "limit": 25,
            "has_more": True,
            "more_available": 5,
        }
        assert not {
            "total_count",
            "offset",
            "limit",
            "has_more",
            "more_available",
            "truncated",
        } & out.keys()
        assert out["search_hint"] == (
            "Use search_term to match indicator names, "
            "categories, or docs."
        )
        assert "show_all_hint" not in out

        standard = raw_list(category="momentum", limit=25, detail="standard")
        assert standard["detail"] == "standard"
        assert standard["data"][0]["params"] == "length=14"
        assert "description" in standard["data"][0]

        summary = raw_list(category="momentum", limit=25, detail="summary")
        assert summary["detail"] == "summary"
        assert set(summary["data"][0]) == {
            "name",
            "category",
            "description",
            "params_count",
        }
        assert "params" not in summary["data"][0]

    def test_indicators_list_rejects_invalid_limits(self):
        from mtdata.core import indicators as core_indicators

        raw_list = getattr(
            core_indicators.indicators_list,
            "__wrapped__",
            core_indicators.indicators_list,
        )

        assert raw_list(limit=0) == {"error": "Invalid limit: 0. Must be >= 1."}
        assert raw_list(limit=-3) == {"error": "Invalid limit: -3. Must be >= 1."}
        assert raw_list(limit="many") == {
            "error": "Invalid limit: many. Must be an integer >= 1."
        }

    def test_indicators_list_supports_offset_pagination(self, monkeypatch):
        from mtdata.core import indicators as core_indicators

        monkeypatch.setattr(
            core_indicators,
            "_list_ta_indicators",
            lambda detailed=False: [
                {
                    "name": f"ind_{i:02d}",
                    "category": "momentum",
                    "description": "",
                    "params": [],
                }
                for i in range(10)
            ],
        )

        raw_list = getattr(core_indicators.indicators_list, "__wrapped__", core_indicators.indicators_list)
        out = raw_list(category="momentum", limit=3, offset=4)

        assert out["success"] is True
        assert out["row_key"] == "data"
        assert out["count"] == 3
        assert [row["name"] for row in out["data"]] == ["ind_04", "ind_05", "ind_06"]
        assert out["pagination"] == {
            "total": 10,
            "returned": 3,
            "offset": 4,
            "limit": 3,
            "has_more": True,
            "more_available": 3,
        }
        assert not {"total_count", "offset", "limit", "has_more", "more_available"} & out.keys()

    def test_indicators_list_full_detail_includes_descriptions(self, monkeypatch):
        from mtdata.core import indicators as core_indicators

        monkeypatch.setattr(
            core_indicators,
            "_list_ta_indicators",
            lambda detailed=False: [
                {
                    "name": "bbands",
                    "category": "volatility",
                    "description": "Bollinger Bands volatility envelope.",
                    "params": [{"name": "length", "default": 20}],
                }
            ],
        )

        raw_list = getattr(core_indicators.indicators_list, "__wrapped__", core_indicators.indicators_list)
        out = raw_list(search_term="bbands", detail="full")

        assert out["success"] is True
        assert out["detail"] == "full"
        assert out["data"][0]["summary"] == "Bollinger Bands volatility envelope."
        assert out["data"][0]["params_count"] == 1
        assert out["data"][0]["params"] == [{"name": "length", "default": 20}]
        assert "aliases" not in out["data"][0]
        assert "Bollinger Bands" in out["data"][0]["description"]

    def test_indicators_list_full_detail_strips_signature_and_doc_sections(self, monkeypatch):
        from mtdata.core import indicators as core_indicators

        sample_doc = (
            "adx(high, low, close, length=14) -> pandas.core.frame.DataFrame\n"
            "Average Directional Movement\n\n"
            "This indicator attempts to quantify trend strength.\n\n"
            "Parameters:\n"
            "    length (int): Window length.\n"
            "Sources:\n"
            "    * https://example.com/adx\n"
        )

        monkeypatch.setattr(
            core_indicators,
            "_list_ta_indicators",
            lambda detailed=False: [
                {
                    "name": "adx",
                    "category": "trend",
                    "description": sample_doc,
                    "aliases": [],
                    "params": [{"name": "length", "default": 14}],
                }
            ],
        )

        raw_list = getattr(core_indicators.indicators_list, "__wrapped__", core_indicators.indicators_list)
        out = raw_list(search_term="adx", detail="full")

        assert out["success"] is True
        description = out["data"][0]["description"]
        assert "Average Directional Movement" in description
        assert "quantify trend strength" in description
        assert "adx(" not in description
        assert "Parameters:" not in description
        assert "Sources:" not in description
        assert out["data"][0]["params"] == [
            {"name": "length", "default": 14, "description": "Window length."}
        ]

    def test_indicators_list_search_matches_names_aliases_and_categories(self, monkeypatch):
        from mtdata.core import indicators as core_indicators

        monkeypatch.setattr(
            core_indicators,
            "_list_ta_indicators",
            lambda detailed=False: [
                {"name": "rsi", "category": "momentum", "description": "Relative Strength Index", "aliases": []},
                {"name": "lrsi", "category": "momentum", "description": "Laguerre RSI", "aliases": []},
                {"name": "stochrsi", "category": "momentum", "description": "Stochastic RSI", "aliases": []},
                {"name": "obv", "category": "volume", "description": "Volume tool that references RSI in docs", "aliases": []},
            ],
        )

        raw_list = getattr(core_indicators.indicators_list, "__wrapped__", core_indicators.indicators_list)
        out = raw_list(search_term="rsi", detail="full")

        assert out["success"] is True
        assert [row["name"] for row in out["data"]] == ["rsi", "lrsi", "stochrsi"]

        category_out = raw_list(search_term="momentum", detail="full")
        assert category_out["success"] is True
        assert [row["name"] for row in category_out["data"]] == ["lrsi", "rsi", "stochrsi"]

    def test_indicators_describe_requires_canonical_name(self, monkeypatch):
        from mtdata.core import indicators as core_indicators

        monkeypatch.setattr(
            core_indicators,
            "_list_ta_indicators",
            lambda detailed=True: [
                {
                    "name": "bbands",
                    "category": "volatility",
                    "params": [{"name": "length", "default": 20}],
                    "description": "Bollinger Bands volatility envelope.",
                }
            ],
        )

        raw_describe = getattr(core_indicators.indicators_describe, "__wrapped__", core_indicators.indicators_describe)
        out = raw_describe("bbands")

        assert out["success"] is True
        assert out["indicator"]["name"] == "bbands"
        assert "usage_examples" not in out["indicator"]

        missing = raw_describe("bb")
        assert missing["success"] is False
        assert missing["error"] == "Indicator 'bb' not found"
        assert missing["error_code"] == "indicator_not_found"
        assert missing["details"] == {"name": "bb"}
        assert missing["related_tools"] == ["indicators_list"]
        assert "canonical catalog names" in missing["remediation"]

    def test_indicators_describe_suggests_catalog_name_for_fetch_spec(self, monkeypatch):
        from mtdata.core import indicators as core_indicators

        monkeypatch.setattr(
            core_indicators,
            "_list_ta_indicators",
            lambda detailed=True: [
                {
                    "name": "rsi",
                    "category": "momentum",
                    "params": [{"name": "length", "default": 14}],
                    "description": "Relative Strength Index.",
                }
            ],
        )

        raw_describe = getattr(core_indicators.indicators_describe, "__wrapped__", core_indicators.indicators_describe)
        missing = raw_describe("rsi_14")

        assert missing["success"] is False
        assert missing["error_code"] == "indicator_not_found"
        assert missing["details"] == {"name": "rsi_14", "suggested_name": "rsi"}
        assert "describe 'rsi' instead" in missing["error"]
        assert "rsi_14" in missing["remediation"]


# ===================================================================
# 3. mtdata.shared.schema
# ===================================================================
class TestSharedDefs:
    def test_returns_timeframe_spec(self):
        defs = shared_defs()
        assert "TimeframeSpec" in defs
        assert defs["TimeframeSpec"]["type"] == "string"
        assert "enum" in defs["TimeframeSpec"]

    def test_timeframe_enum_sorted(self):
        defs = shared_defs()
        vals = defs["TimeframeSpec"]["enum"]
        assert vals == sorted(vals)



class TestComplexDefs:
    def test_has_expected_keys(self):
        defs = complex_defs()
        assert "IndicatorSpec" in defs
        assert "DenoiseSpec" in defs
        assert "SimplifySpec" in defs

    def test_indicator_spec_structure(self):
        defs = complex_defs()
        spec = defs["IndicatorSpec"]
        assert spec["type"] == "object"
        assert spec["required"] == ["name"]
        assert spec["additionalProperties"] is False
        assert spec["properties"]["params"]["oneOf"][0]["type"] == "array"
        assert spec["properties"]["params"]["oneOf"][1]["type"] == "object"

    def test_denoise_columns_accept_string_or_list(self):
        defs = complex_defs()
        spec = defs["DenoiseSpec"]
        assert spec["required"] == ["method"]
        assert spec["properties"]["columns"]["oneOf"][0]["type"] == "string"
        assert spec["properties"]["columns"]["oneOf"][1]["type"] == "array"

    def test_simplify_schema_includes_sax(self):
        from mtdata.shared.schema import EncodeSchemaLiteral
        assert "sax" in EncodeSchemaLiteral.__args__

    def test_simplify_algorithm_uses_canonical_name(self):
        defs = complex_defs()

        assert defs["SimplifySpec"]["properties"]["algo"]["enum"] == ["zigzag"]


class TestEnsureDefs:
    def test_adds_defs_if_missing(self):
        schema = {}
        _ensure_defs(schema)
        assert "$defs" in schema
        assert "TimeframeSpec" in schema["$defs"]

    def test_does_not_overwrite_existing(self):
        schema = {"$defs": {"Custom": {"type": "string"}}}
        _ensure_defs(schema)
        assert "Custom" in schema["$defs"]
        assert "TimeframeSpec" in schema["$defs"]


class TestParametersObj:
    def test_creates_if_missing(self):
        schema = {}
        params = _parameters_obj(schema)
        assert params["type"] == "object"
        assert "properties" in params

    def test_returns_existing(self):
        schema = {"parameters": {"type": "object", "properties": {"a": {"type": "string"}}}}
        params = _parameters_obj(schema)
        assert "a" in params["properties"]


class TestApplyParamHints:
    def test_adds_description(self):
        schema = {"parameters": {"type": "object", "properties": {
            "symbol": {"type": "string"},
        }}}
        apply_param_hints(schema)
        assert schema["parameters"]["properties"]["symbol"]["description"] == PARAM_HINTS["symbol"]

    def test_no_overwrite_existing_description(self):
        schema = {"parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "custom"},
        }}}
        apply_param_hints(schema)
        assert schema["parameters"]["properties"]["symbol"]["description"] == "custom"


class TestApplyTimeframeRef:
    def test_replaces_timeframe_prop(self):
        schema = {"parameters": {"type": "object", "properties": {
            "timeframe": {"type": "string"},
        }}}
        apply_timeframe_ref(schema)
        assert schema["parameters"]["properties"]["timeframe"] == {"$ref": "#/$defs/TimeframeSpec"}

    def test_ignores_non_timeframe(self):
        schema = {"parameters": {"type": "object", "properties": {
            "symbol": {"type": "string"},
        }}}
        apply_timeframe_ref(schema)
        assert schema["parameters"]["properties"]["symbol"] == {"type": "string"}


class TestAllowNull:
    def test_simple_type(self):
        result = _allow_null({"type": "string"})
        assert result["type"] == ["string", "null"]

    def test_already_null(self):
        result = _allow_null({"type": "null"})
        assert result["type"] == "null"

    def test_list_type(self):
        result = _allow_null({"type": ["string", "integer"]})
        assert "null" in result["type"]

    def test_list_type_already_has_null(self):
        result = _allow_null({"type": ["string", "null"]})
        assert result["type"].count("null") == 1

    def test_no_type_key(self):
        result = _allow_null({"enum": ["a", "b"]})
        assert "type" not in result


class TestTypeHintToSchema:
    def test_none(self):
        assert _type_hint_to_schema(None) == {"type": "string"}

    def test_any(self):
        assert _type_hint_to_schema(Any) == {}

    def test_str(self):
        assert _type_hint_to_schema(str) == {"type": "string"}

    def test_int(self):
        assert _type_hint_to_schema(int) == {"type": "integer"}

    def test_float(self):
        assert _type_hint_to_schema(float) == {"type": "number"}

    def test_bool(self):
        assert _type_hint_to_schema(bool) == {"type": "boolean"}

    def test_bare_dict(self):
        result = _type_hint_to_schema(dict)
        assert result["type"] == "object"

    def test_bare_list(self):
        result = _type_hint_to_schema(list)
        assert result["type"] == "array"

    def test_list_of_str(self):
        result = _type_hint_to_schema(List[str])
        assert result["type"] == "array"
        assert result["items"] == {"type": "string"}

    def test_dict_str_int(self):
        result = _type_hint_to_schema(Dict[str, int])
        assert result["type"] == "object"
        assert result["additionalProperties"] == {"type": "integer"}

    def test_optional_str(self):
        result = _type_hint_to_schema(Optional[str])
        assert "null" in str(result.get("type", ""))

    def test_literal_strings(self):
        result = _type_hint_to_schema(Literal["a", "b"])
        assert result["type"] == "string"
        assert set(result["enum"]) == {"a", "b"}

    def test_literal_ints(self):
        result = _type_hint_to_schema(Literal[1, 2, 3])
        assert result["type"] == "integer"
        assert result["enum"] == [1, 2, 3]

    def test_literal_bools(self):
        result = _type_hint_to_schema(Literal[True, False])
        assert result["type"] == "boolean"

    def test_union_type(self):
        result = _type_hint_to_schema(Union[str, int])
        assert "oneOf" in result


class TestIsTypedDictType:
    def test_regular_class_false(self):
        class Foo:
            pass
        assert _is_typed_dict_type(Foo) is False

    def test_typed_dict_true(self):
        class Bar(TypedDict):
            name: str
        assert _is_typed_dict_type(Bar) is True


class TestBuildMinimalSchema:
    def test_basic(self):
        func_info = {
            "params": [
                {"name": "symbol", "required": True, "type": str, "default": None},
                {"name": "limit", "required": False, "type": int, "default": 25},
            ]
        }
        schema = build_minimal_schema(func_info)
        props = schema["parameters"]["properties"]
        assert "symbol" in props
        assert "limit" in props
        assert props["limit"].get("default") == 25
        assert "symbol" in schema["parameters"]["required"]
        assert "$defs" in schema

    def test_empty_params(self):
        schema = build_minimal_schema({"params": []})
        assert schema["parameters"]["properties"] == {}


class TestEnrichSchemaWithSharedDefs:
    def test_empty_schema_builds_from_func_info(self):
        func_info = {"params": [{"name": "x", "required": True, "type": str, "default": None}]}
        schema = enrich_schema_with_shared_defs({}, func_info)
        assert "$defs" in schema
        assert "x" in schema["parameters"]["properties"]

    def test_existing_schema_gets_defs(self):
        schema = {"parameters": {"type": "object", "properties": {"timeframe": {"type": "string"}}}}
        enriched = enrich_schema_with_shared_defs(schema, {"params": []})
        assert "$defs" in enriched
        tf_prop = enriched["parameters"]["properties"]["timeframe"]
        assert tf_prop["$ref"] == "#/$defs/TimeframeSpec"


class TestGetSharedEnumLists:
    def test_has_expected_keys(self):
        enums = get_shared_enum_lists()
        assert "DENOISE_METHODS" in enums
        assert "SIMPLIFY_MODES" in enums
        assert "SIMPLIFY_METHODS" in enums
        assert "PIVOT_METHODS" in enums
        assert "FORECAST_METHODS" in enums

    def test_values_match_tuples(self):
        enums = get_shared_enum_lists()
        assert enums["DENOISE_METHODS"] == list(_DENOISE_METHODS)
        assert enums["SIMPLIFY_MODES"] == list(_SIMPLIFY_MODES)
        assert enums["SIMPLIFY_METHODS"] == list(_SIMPLIFY_METHODS)
        assert enums["PIVOT_METHODS"] == list(_PIVOT_METHODS)

    def test_denoise_enum_matches_live_registry(self):
        from mtdata.utils.denoise.base import list_filters

        enums = get_shared_enum_lists()
        assert set(enums["DENOISE_METHODS"]) == {"none"} | set(list_filters())


class TestGetFunctionInfo:
    def test_simple_function(self):
        def example(x: int, y: str = "hello") -> bool:
            """A test function."""
            return True

        info = get_function_info(example)
        assert info["name"] == "example"
        assert info["doc"] == "A test function."
        params = info["params"]
        assert len(params) == 2
        assert params[0]["name"] == "x"
        assert params[0]["required"] is True
        assert params[1]["name"] == "y"
        assert params[1]["required"] is False
        assert params[1]["default"] == "hello"

    def test_no_params(self):
        def noop():
            pass
        info = get_function_info(noop)
        assert info["params"] == []

    def test_skips_self(self):
        class Foo:
            def bar(self, x: int) -> None:
                pass
        info = get_function_info(Foo.bar)
        names = [p["name"] for p in info["params"]]
        assert "self" not in names
        assert "x" in names


class TestSchemaConstants:
    """Smoke-test that schema-level constants are well-formed."""

    def test_denoise_methods_are_strings(self):
        assert all(isinstance(m, str) for m in _DENOISE_METHODS)
        assert "ema" in _DENOISE_METHODS

    def test_simplify_modes(self):
        assert "select" in _SIMPLIFY_MODES
        assert "resample" in _SIMPLIFY_MODES

    def test_pivot_methods(self):
        assert "classic" in _PIVOT_METHODS
        assert "fibonacci" in _PIVOT_METHODS
