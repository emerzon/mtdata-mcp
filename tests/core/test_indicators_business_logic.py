from mtdata.core import indicators as core_indicators


def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def test_indicators_list_full_uses_cleaned_summary(monkeypatch):
    monkeypatch.setattr(
        core_indicators,
        "_list_ta_indicators",
        lambda detailed=False: [
            {
                "name": "rsi",
                "category": "momentum",
                "description": (
                    "Python Library Documentation: function rsi in module pandas_ta\n"
                    "rsi(close, length=14)\n"
                    "Relative Strength Index (RSI)\n"
                    "Measures momentum by comparing recent average gains and losses."
                ),
                "params": [{"name": "length", "default": 14}],
                "aliases": [],
            }
        ],
    )

    raw = _unwrap(core_indicators.indicators_list)
    result = raw(search_term="rsi", detail="full")

    row = result["data"][0]
    assert row["summary"] == "Relative Strength Index (RSI)"
    assert "Python Library Documentation" not in row["summary"]
    assert "Python Library Documentation" not in row["description"]


def test_indicators_list_default_prioritizes_common_indicators(monkeypatch):
    monkeypatch.setattr(
        core_indicators,
        "_list_ta_indicators",
        lambda detailed=False: [
            {"name": "cdl_doji", "category": "candles", "description": "", "params": []},
            {"name": "ema", "category": "overlap", "description": "", "params": []},
            {"name": "rsi", "category": "momentum", "description": "", "params": []},
            {"name": "zscore", "category": "statistics", "description": "", "params": []},
        ],
    )

    raw = _unwrap(core_indicators.indicators_list)

    default = raw()
    filtered = raw(category="candles")

    assert [row["name"] for row in default["data"]] == ["rsi", "ema", "cdl_doji", "zscore"]
    assert [row["name"] for row in filtered["data"]] == ["cdl_doji"]


def test_indicators_list_discloses_trading_style_filter_basis(monkeypatch):
    monkeypatch.setattr(
        core_indicators,
        "_list_ta_indicators",
        lambda detailed=False: [
            {
                "name": "rsi",
                "category": "momentum",
                "description": "Relative Strength Index.",
                "params": [],
            },
            {
                "name": "coppock",
                "category": "momentum",
                "description": "Designed for use on a monthly time scale.",
                "params": [],
            },
            {
                "name": "sma",
                "category": "overlap",
                "description": "Simple moving average.",
                "params": [],
            },
        ],
    )

    raw = _unwrap(core_indicators.indicators_list)
    result = raw(trading_style="intraday", detail="full", limit=20)

    assert result["trading_style_filter"] == {
        "requested": "intraday",
        "semantics": "broad_workflow_tag_not_performance_recommendation",
        "curated_indicator_matches": 1,
        "category_heuristic_matches": 1,
        "unknown_basis_matches": 0,
    }
    assert "1 match(es) inherit" in result["warnings"][0]
    rows = {row["name"]: row for row in result["data"]}
    assert rows["rsi"]["trading_context"]["trading_styles_basis"] == (
        "curated_indicator"
    )
    assert rows["coppock"]["trading_context"]["trading_styles_basis"] == (
        "category_heuristic"
    )
    assert "not an indicator-specific recommendation" in rows["coppock"][
        "trading_context"
    ]["trading_styles_note"]


def test_indicators_describe_rsi_and_macd_include_calculation(monkeypatch):
    monkeypatch.setattr(
        core_indicators,
        "_list_ta_indicators",
        lambda detailed=False: [
            {"name": "rsi", "category": "momentum", "description": "", "params": []},
            {"name": "macd", "category": "momentum", "description": "", "params": []},
            {"name": "obscure_tail", "category": "statistics", "description": "", "params": []},
        ],
    )
    raw = _unwrap(core_indicators.indicators_describe)

    rsi = raw("rsi", detail="full")
    macd = raw("macd", detail="full")
    tail = raw("obscure_tail", detail="full")

    assert rsi["indicator"]["documentation"]["calculation"]
    assert "avg_gain" in rsi["indicator"]["documentation"]["calculation"]
    assert macd["indicator"]["documentation"]["calculation"]
    assert "EMA" in macd["indicator"]["documentation"]["calculation"]
    assert "pandas-ta-classic" in tail["indicator"]["documentation"]["calculation"]
    assert "macd_h_{fast}_{slow}_{signal}" in macd["indicator"]["documentation"]["calculation"]
    assert "macdh_" not in macd["indicator"]["documentation"]["calculation"]


def test_indicators_describe_bbands_calculation_matches_backend_default(monkeypatch):
    monkeypatch.setattr(
        core_indicators,
        "_list_ta_indicators",
        lambda detailed=False: [
            {
                "name": "bbands",
                "category": "overlap",
                "description": "",
                "params": [{"name": "length", "default": 5}],
            },
        ],
    )
    raw = _unwrap(core_indicators.indicators_describe)
    bbands = raw("bbands", detail="full")
    calculation = bbands["indicator"]["documentation"]["calculation"]
    assert "backend default 5" in calculation
    assert "bbands(20,2)" in calculation
    assert "default 20" not in calculation


def test_indicator_engine_provenance_is_compact():
    from mtdata.utils.indicators import indicator_engine_provenance

    provenance = indicator_engine_provenance()
    assert provenance["pandas_ta"]["name"] == "pandas-ta-classic"
    assert "version" in provenance["pandas_ta"]
    assert "available" in provenance["talib"]
    assert provenance["effective_backend"] in {
        "pandas-ta-classic",
        "pandas-ta-classic+talib",
    }


def test_indicators_describe_full_includes_indicator_engine(monkeypatch):
    monkeypatch.setattr(
        core_indicators,
        "_list_ta_indicators",
        lambda detailed=True: [
            {"name": "rsi", "category": "momentum", "description": "", "params": []},
        ],
    )
    raw = _unwrap(core_indicators.indicators_describe)
    out = raw("rsi", detail="full")

    assert out["indicator_engine"]["pandas_ta"]["name"] == "pandas-ta-classic"
    assert "effective_backend" in out["indicator_engine"]


def test_indicators_describe_vwap_usage_has_no_params(monkeypatch):
    monkeypatch.setattr(
        core_indicators,
        "_list_ta_indicators",
        lambda detailed=True: [
            {
                "name": "vwap",
                "category": "overlap",
                "description": "Volume Weighted Average Price",
                "params": [{"name": "anchor", "default": 0}],
            }
        ],
    )
    raw = _unwrap(core_indicators.indicators_describe)
    out = raw("vwap")

    assert out["indicator"]["usage"]["compact_spec"] == "vwap"
    assert out["indicator"]["usage"]["cli"] == '--indicators "vwap"'


def test_indicators_describe_cdl_pattern_is_not_compact_cli(monkeypatch):
    monkeypatch.setattr(
        core_indicators,
        "_list_ta_indicators",
        lambda detailed=True: [
            {
                "name": "cdl_pattern",
                "category": "candles",
                "description": "Candlestick patterns",
                "params": [{"name": "name", "default": "all"}],
            }
        ],
    )
    raw = _unwrap(core_indicators.indicators_describe)
    out = raw("cdl_pattern")
    usage = out["indicator"]["usage"]

    assert usage["cli_supported"] is False
    assert usage["compact_spec"] is None
    assert "patterns_detect" in usage["alternative"]["tool"]
    assert "cdl_pattern(all)" not in str(usage)
