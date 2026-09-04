"""Candlestick pattern tests."""

from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import mtdata.core.patterns_support as patterns_support_mod
import mtdata.patterns.candlestick as candlestick_mod
import mtdata.services.data_service.candles as data_service_mod
from mtdata.core import patterns as core_patterns
from mtdata.core.patterns_requests import PatternsDetectRequest
from mtdata.patterns.candlestick import (
    _extract_candlestick_rows,
    _get_candlestick_pattern_methods,
    _is_candlestick_allowed,
    _normalize_candlestick_name,
)


def patterns_detect(**kwargs):
    raw_output = bool(kwargs.pop("__cli_raw", True))
    request = kwargs.pop("request", None)
    if request is None:
        request = PatternsDetectRequest(**kwargs)
    return core_patterns.patterns_detect(request=request, __cli_raw=raw_output)


def _install_fake_aggregate_backend(monkeypatch, *, catalog, signals):
    calls = []

    class _FakeFrame(pd.DataFrame):
        @property
        def _constructor(self):
            return _FakeFrame

        @property
        def ta(self):
            frame = self

            class _Accessor:
                def cdl_pattern(self, name="all", append=True):
                    assert append is True
                    requested = list(catalog) if name == "all" else list(name)
                    calls.append(("cdl_pattern", tuple(requested)))
                    for detector in requested:
                        if detector in catalog:
                            frame[f"CDL_{detector.upper()}"] = signals.get(
                                detector, [0.0, 0.0]
                            )

                def cdl_doji(self, append=True):
                    assert append is True
                    calls.append(("cdl_doji", ()))
                    frame["CDL_DOJI"] = signals.get("doji", [0.0, 0.0])

            return _Accessor()

    monkeypatch.setattr(candlestick_mod, "_ensure_candlestick_runtime", lambda: None)
    monkeypatch.setattr(candlestick_mod, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(
        candlestick_mod,
        "_get_candlestick_pattern_methods",
        lambda _temp: ["cdl_doji", "cdl_pattern"],
    )
    monkeypatch.setattr(
        candlestick_mod,
        "ta",
        SimpleNamespace(CDL_PATTERN_NAMES=list(catalog)),
    )
    monkeypatch.setattr(
        data_service_mod,
        "fetch_history_frame",
        lambda *_a, **_k: _FakeFrame(
            {
                "time": [1_700_000_000.0, 1_700_003_600.0],
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
            }
        ),
    )
    return calls


def test_candlestick_allowed_respects_whitelist_when_not_robust_only():
    robust_set = {"engulfing", "harami"}
    whitelist = {"engulfing"}

    assert _is_candlestick_allowed(
        "engulfing",
        robust_only=False,
        robust_set=robust_set,
        whitelist_set=whitelist,
    )
    assert not _is_candlestick_allowed(
        "doji",
        robust_only=False,
        robust_set=robust_set,
        whitelist_set=whitelist,
    )


def test_candlestick_allowed_requires_robust_when_enabled():
    robust_set = {"engulfing", "harami"}

    assert _is_candlestick_allowed(
        "engulfing",
        robust_only=True,
        robust_set=robust_set,
        whitelist_set=None,
    )
    assert not _is_candlestick_allowed(
        "doji",
        robust_only=True,
        robust_set=robust_set,
        whitelist_set=None,
    )


def test_candlestick_name_normalization_canonicalizes_common_variants():
    assert _normalize_candlestick_name("cdl_closing_marubozu") == "closingmarubozu"
    assert _is_candlestick_allowed(
        "closing_marubozu",
        robust_only=False,
        robust_set=set(),
        whitelist_set={"closingmarubozu"},
    )


def test_get_candlestick_pattern_methods_caches_discovery(monkeypatch):
    class _FakeTA:
        def __init__(self):
            self.dir_calls = 0

        def __dir__(self):
            self.dir_calls += 1
            return ["cdl_alpha", "not_a_pattern"]

        def __getattr__(self, name):
            if name == "cdl_alpha":
                return lambda *args, **kwargs: None
            raise AttributeError(name)

    fake_temp = SimpleNamespace(ta=_FakeTA())
    monkeypatch.setattr(candlestick_mod, "_CANDLESTICK_PATTERN_METHOD_CACHE", None)

    out_first = _get_candlestick_pattern_methods(fake_temp)
    out_second = _get_candlestick_pattern_methods(fake_temp)

    assert out_first == ["cdl_alpha"]
    assert out_second == ["cdl_alpha"]
    assert fake_temp.ta.dir_calls == 1


def test_get_candlestick_pattern_methods_invalidates_cache_on_accessor_type_change(
    monkeypatch,
):
    class _AlphaTA:
        def __init__(self):
            self.dir_calls = 0

        def __dir__(self):
            self.dir_calls += 1
            return ["cdl_alpha"]

        def __getattr__(self, name):
            if name == "cdl_alpha":
                return lambda *args, **kwargs: None
            raise AttributeError(name)

    class _BetaTA:
        def __init__(self):
            self.dir_calls = 0

        def __dir__(self):
            self.dir_calls += 1
            return ["cdl_beta"]

        def __getattr__(self, name):
            if name == "cdl_beta":
                return lambda *args, **kwargs: None
            raise AttributeError(name)

    alpha_temp = SimpleNamespace(ta=_AlphaTA())
    beta_temp = SimpleNamespace(ta=_BetaTA())
    monkeypatch.setattr(candlestick_mod, "_CANDLESTICK_PATTERN_METHOD_CACHE", None)
    monkeypatch.setattr(candlestick_mod, "_CANDLESTICK_PATTERN_METHOD_CACHE_KEY", None)

    out_alpha = _get_candlestick_pattern_methods(alpha_temp)
    out_beta = _get_candlestick_pattern_methods(beta_temp)

    assert out_alpha == ["cdl_alpha"]
    assert out_beta == ["cdl_beta"]
    assert alpha_temp.ta.dir_calls == 1
    assert beta_temp.ta.dir_calls == 1


def test_extract_candlestick_rows_prefers_non_deprioritized_hits():
    df_tail = pd.DataFrame({"time": ["T0", "T1"]})
    temp_tail = pd.DataFrame(
        {
            "cdl_doji": [0.0, 100.0],
            "cdl_engulfing": [0.0, 100.0],
            "cdl_longline": [0.0, 80.0],
        }
    )

    rows = _extract_candlestick_rows(
        df_tail,
        temp_tail,
        ["cdl_doji", "cdl_engulfing", "cdl_longline"],
        threshold=0.95,
        robust_only=False,
        robust_set={"engulfing"},
        whitelist_set=None,
        min_gap=0,
        top_k=1,
        deprioritize={"doji", "longline"},
    )

    assert rows == [["T1", "Bullish ENGULFING"]]


def test_extract_candlestick_rows_includes_metrics_when_enabled():
    df_tail = pd.DataFrame(
        {"time": ["T0", "T1"], "close": [100.0, 101.50000000000001]}
    )
    temp_tail = pd.DataFrame({"cdl_engulfing": [0.0, 100.0]})

    rows = _extract_candlestick_rows(
        df_tail,
        temp_tail,
        ["cdl_engulfing"],
        threshold=0.95,
        robust_only=False,
        robust_set={"engulfing"},
        whitelist_set=None,
        min_gap=0,
        top_k=1,
        deprioritize=set(),
        include_metrics=True,
    )

    assert len(rows) == 1
    assert rows[0][0] == "T1"
    assert rows[0][1] == "Bullish ENGULFING"
    assert rows[0][2] == "bullish"
    assert rows[0][3] == pytest.approx(1.0)
    assert rows[0][4:] == [100, 101.5, "T0", "T1", 2, 0, 1]


def test_extract_candlestick_rows_normalizes_detector_signal_scale():
    df_tail = pd.DataFrame(
        {"time": ["T0", "T1"], "close": [100.0, 101.0]}
    )
    temp_tail = pd.DataFrame({"cdl_alpha": [50.0, 100.0]})

    rows = _extract_candlestick_rows(
        df_tail,
        temp_tail,
        ["cdl_alpha"],
        threshold=0.0,
        robust_only=False,
        robust_set=set(),
        whitelist_set=None,
        min_gap=0,
        top_k=1,
        deprioritize=set(),
        include_metrics=True,
    )

    assert [row[3] for row in rows] == pytest.approx([0.75, 0.75])


def test_detect_candlestick_patterns_rejects_out_of_range_min_strength():
    res = candlestick_mod.detect_candlestick_patterns(
        symbol="EURUSD",
        timeframe="H1",
        limit=10,
        min_strength=1.5,
        min_gap=0,
        robust_only=False,
        whitelist=None,
        top_k=1,
    )

    assert res == {"error": "min_strength must be between 0.0 and 1.0."}


def test_detect_candlestick_patterns_does_not_flatten_unexpected_errors(monkeypatch):
    monkeypatch.setattr(candlestick_mod, "_ensure_candlestick_runtime", lambda: None)
    monkeypatch.setattr(candlestick_mod, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(
        data_service_mod,
        "fetch_history_frame",
        lambda *_a, **_k: pd.DataFrame(
            {
                "time": [1_700_000_000.0, 1_700_003_600.0],
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
            }
        ),
    )
    monkeypatch.setattr(
        candlestick_mod,
        "_get_candlestick_pattern_methods",
        lambda _temp: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        candlestick_mod.detect_candlestick_patterns(
            symbol="EURUSD",
            timeframe="H1",
            limit=10,
            min_strength=0.95,
            min_gap=0,
            robust_only=False,
            whitelist=None,
            top_k=1,
        )


def test_detect_candlestick_patterns_prefilters_methods_by_whitelist(monkeypatch):
    calls = []

    class _FakeFrame(pd.DataFrame):
        _metadata = ["_calls"]

        @property
        def _constructor(self):
            return _FakeFrame

        @property
        def ta(self):
            frame = self

            class _Accessor:
                def cdl_alpha(self, append=True):
                    _ = append
                    frame._calls.append("cdl_alpha")
                    frame["cdl_alpha"] = [0.0, 100.0]

                def cdl_beta(self, append=True):
                    _ = append
                    frame._calls.append("cdl_beta")
                    frame["cdl_beta"] = [0.0, 100.0]

            return _Accessor()

    monkeypatch.setattr(candlestick_mod, "_ensure_candlestick_runtime", lambda: None)
    monkeypatch.setattr(candlestick_mod, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(
        candlestick_mod,
        "_get_candlestick_pattern_methods",
        lambda _temp: ["cdl_alpha", "cdl_beta"],
    )

    def _fake_rates_to_df(*_a, **_k):
        frame = _FakeFrame(
            {
                "time": [1_700_000_000.0, 1_700_003_600.0],
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
            }
        )
        frame._calls = calls
        return frame

    monkeypatch.setattr(data_service_mod, "fetch_history_frame", _fake_rates_to_df)

    res = candlestick_mod.detect_candlestick_patterns(
        symbol="EURUSD",
        timeframe="H1",
        limit=10,
        min_strength=0.95,
        min_gap=0,
        robust_only=False,
        whitelist="alpha",
        top_k=1,
    )

    assert res["success"] is True
    assert calls == ["cdl_alpha"]
    assert res["min_strength"] == pytest.approx(0.95)
    assert res["strength_scale"] == "ohlc_geometry_and_pattern_reliability_v3"
    assert res["signal_scale"] == "backend_native_cdl_signal"


def test_detect_candlestick_patterns_whitelist_accepts_display_names(monkeypatch):
    calls = []

    class _FakeFrame(pd.DataFrame):
        _metadata = ["_calls"]

        @property
        def _constructor(self):
            return _FakeFrame

        @property
        def ta(self):
            frame = self

            class _Accessor:
                def cdl_belthold(self, append=True):
                    _ = append
                    frame._calls.append("cdl_belthold")
                    frame["cdl_belthold"] = [0.0, 200.0]

                def cdl_doji(self, append=True):
                    _ = append
                    frame._calls.append("cdl_doji")
                    frame["cdl_doji"] = [0.0, 200.0]

            return _Accessor()

    monkeypatch.setattr(candlestick_mod, "_ensure_candlestick_runtime", lambda: None)
    monkeypatch.setattr(candlestick_mod, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(
        candlestick_mod,
        "_get_candlestick_pattern_methods",
        lambda _temp: ["cdl_belthold", "cdl_doji"],
    )

    def _fake_rates_to_df(*_a, **_k):
        frame = _FakeFrame(
            {
                "time": [1_700_000_000.0, 1_700_003_600.0],
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
            }
        )
        frame._calls = calls
        return frame

    monkeypatch.setattr(data_service_mod, "fetch_history_frame", _fake_rates_to_df)

    res = candlestick_mod.detect_candlestick_patterns(
        symbol="EURUSD",
        timeframe="H1",
        limit=10,
        min_strength=0.70,
        min_gap=0,
        robust_only=False,
        whitelist="Bullish BELTHOLD",
        top_k=1,
    )

    assert res["success"] is True
    assert calls == ["cdl_belthold"]
    assert [row["pattern"] for row in res["data"]] == ["Bullish BELTHOLD"]


def test_detect_candlestick_patterns_whitelist_error_lists_detectors(monkeypatch):
    class _FakeFrame(pd.DataFrame):
        @property
        def _constructor(self):
            return _FakeFrame

    monkeypatch.setattr(candlestick_mod, "_ensure_candlestick_runtime", lambda: None)
    monkeypatch.setattr(candlestick_mod, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(
        candlestick_mod,
        "_get_candlestick_pattern_methods",
        lambda _temp: ["cdl_doji", "cdl_hammer"],
    )
    monkeypatch.setattr(
        data_service_mod,
        "fetch_history_frame",
        lambda *_a, **_k: _FakeFrame(
            {
                "time": [1_700_000_000.0, 1_700_003_600.0],
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
            }
        ),
    )

    res = candlestick_mod.detect_candlestick_patterns(
        symbol="EURUSD",
        timeframe="H1",
        limit=10,
        min_strength=0.70,
        min_gap=0,
        robust_only=False,
        whitelist="foo",
        top_k=1,
    )

    assert "No candlestick detectors match whitelist 'foo'" in res["error"]
    assert "DOJI" in res["error"]
    assert "HAMMER" in res["error"]


def test_aggregate_dispatcher_evaluates_supported_zero_hit_whitelist(monkeypatch):
    calls = _install_fake_aggregate_backend(
        monkeypatch,
        catalog=["doji", "engulfing", "hammer"],
        signals={"engulfing": [0.0, 0.0]},
    )

    result = candlestick_mod.detect_candlestick_patterns(
        symbol="EURUSD",
        timeframe="H1",
        limit=10,
        min_strength=0.0,
        min_gap=0,
        robust_only=False,
        whitelist="engulfing",
        top_k=3,
    )

    assert result["success"] is True
    assert result["data"] == []
    assert result["requested_detectors"] == ["engulfing"]
    assert result["detectors_evaluated"] == ["engulfing"]
    assert "unsupported_detectors" not in result
    assert calls == [("cdl_pattern", ("engulfing",))]


def test_candlestick_default_denoise_expands_to_complete_ohlc(monkeypatch):
    _install_fake_aggregate_backend(
        monkeypatch,
        catalog=["doji"],
        signals={"doji": [0.0, 0.0]},
    )
    captured = {}

    def _apply(frame, spec):
        captured.update(spec)
        for column in ("open", "high", "low", "close"):
            frame[f"{column}_dn"] = frame[column]
        frame.attrs["denoise_last_application"] = {
            "added_columns": [
                "open_dn",
                "high_dn",
                "low_dn",
                "close_dn",
            ],
            "overwrote_columns": [],
        }

    monkeypatch.setattr("mtdata.utils.denoise.apply_denoise", _apply)

    result = candlestick_mod.detect_candlestick_patterns(
        symbol="EURUSD",
        timeframe="H1",
        limit=10,
        min_strength=0.0,
        min_gap=0,
        robust_only=False,
        whitelist="doji",
        top_k=1,
        denoise={"method": "ema"},
    )

    assert result["success"] is True
    assert captured["columns"] == "ohlc"


def test_candlestick_partial_ohlc_denoise_falls_back_with_warning(monkeypatch):
    _install_fake_aggregate_backend(
        monkeypatch,
        catalog=["doji"],
        signals={"doji": [0.0, 0.0]},
    )

    result = candlestick_mod.detect_candlestick_patterns(
        symbol="EURUSD",
        timeframe="H1",
        limit=10,
        min_strength=0.0,
        min_gap=0,
        robust_only=False,
        whitelist="doji",
        top_k=1,
        denoise={"method": "ema", "columns": ["close"]},
    )

    assert result["success"] is True
    assert any(
        "requires all OHLC columns" in str(warning)
        for warning in result["warnings"]
    )


def test_aggregate_dispatcher_discloses_mixed_whitelist_coverage(monkeypatch):
    calls = _install_fake_aggregate_backend(
        monkeypatch,
        catalog=["doji", "engulfing", "hammer"],
        signals={
            "doji": [0.0, 100.0],
            "engulfing": [0.0, 0.0],
            "hammer": [0.0, 100.0],
        },
    )

    result = candlestick_mod.detect_candlestick_patterns(
        symbol="EURUSD",
        timeframe="H1",
        limit=10,
        min_strength=0.0,
        min_gap=0,
        robust_only=False,
        whitelist="engulfing,hammer,unknown,doji",
        top_k=3,
    )

    assert result["success"] is True
    assert result["requested_detectors"] == [
        "engulfing",
        "hammer",
        "unknown",
        "doji",
    ]
    assert result["detectors_evaluated"] == ["doji", "engulfing", "hammer"]
    assert result["unsupported_detectors"] == ["unknown"]
    assert "UNKNOWN" in result["warnings"][-1]
    assert calls == [("cdl_pattern", ("doji", "engulfing", "hammer"))]


def test_aggregate_dispatcher_applies_robust_name_preset(monkeypatch):
    calls = _install_fake_aggregate_backend(
        monkeypatch,
        catalog=["doji", "engulfing"],
        signals={"doji": [0.0, 100.0], "engulfing": [0.0, 100.0]},
    )

    result = candlestick_mod.detect_candlestick_patterns(
        symbol="EURUSD",
        timeframe="H1",
        limit=10,
        min_strength=0.0,
        min_gap=0,
        robust_only=True,
        whitelist=None,
        top_k=3,
    )

    assert result["success"] is True
    assert result["detectors_evaluated"] == ["engulfing"]
    assert calls == [("cdl_pattern", ("engulfing",))]


def test_detect_candlestick_patterns_top_k_uses_semantic_strength(monkeypatch):
    class _FakeFrame(pd.DataFrame):
        @property
        def _constructor(self):
            return _FakeFrame

        @property
        def ta(self):
            frame = self

            class _Accessor:
                def cdl_doji(self, append=True):
                    _ = append
                    frame["cdl_doji"] = [0.0, 100.0]

                def cdl_engulfing(self, append=True):
                    _ = append
                    frame["cdl_engulfing"] = [0.0, 100.0]

            return _Accessor()

    monkeypatch.setattr(candlestick_mod, "_ensure_candlestick_runtime", lambda: None)
    monkeypatch.setattr(candlestick_mod, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(
        candlestick_mod,
        "_get_candlestick_pattern_methods",
        lambda _temp: ["cdl_doji", "cdl_engulfing"],
    )

    def _fake_rates_to_df(*_a, **_k):
        return _FakeFrame(
            {
                "time": [1_700_000_000.0, 1_700_003_600.0],
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
            }
        )

    monkeypatch.setattr(data_service_mod, "fetch_history_frame", _fake_rates_to_df)

    res = candlestick_mod.detect_candlestick_patterns(
        symbol="EURUSD",
        timeframe="H1",
        limit=10,
        min_strength=0.90,
        min_gap=0,
        robust_only=False,
        whitelist=None,
        top_k=1,
    )

    assert res["success"] is True
    assert res["data"][0]["pattern"] == "Bullish ENGULFING"
    assert res["data"][0]["confidence"] > 0.95


def test_detect_candlestick_patterns_dedupes_redundant_same_window_hits(monkeypatch):
    class _FakeFrame(pd.DataFrame):
        @property
        def _constructor(self):
            return _FakeFrame

        @property
        def ta(self):
            frame = self

            class _Accessor:
                def cdl_outside(self, append=True):
                    _ = append
                    frame["cdl_outside"] = [0.0, 100.0]

                def cdl_engulfing(self, append=True):
                    _ = append
                    frame["cdl_engulfing"] = [0.0, 100.0]

            return _Accessor()

    monkeypatch.setattr(candlestick_mod, "_ensure_candlestick_runtime", lambda: None)
    monkeypatch.setattr(candlestick_mod, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(
        candlestick_mod,
        "_get_candlestick_pattern_methods",
        lambda _temp: ["cdl_outside", "cdl_engulfing"],
    )

    def _fake_rates_to_df(*_a, **_k):
        return _FakeFrame(
            {
                "time": [1_700_000_000.0, 1_700_003_600.0],
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
            }
        )

    monkeypatch.setattr(data_service_mod, "fetch_history_frame", _fake_rates_to_df)

    res = candlestick_mod.detect_candlestick_patterns(
        symbol="EURUSD",
        timeframe="H1",
        limit=10,
        min_strength=0.70,
        min_gap=0,
        robust_only=False,
        whitelist=None,
        top_k=2,
    )

    assert res["success"] is True
    assert [row["pattern"] for row in res["data"]] == ["Bullish ENGULFING"]


def test_detect_candlestick_patterns_drops_still_forming_last_bar(monkeypatch):
    class _FakeFrame(pd.DataFrame):
        @property
        def _constructor(self):
            return _FakeFrame

        @property
        def ta(self):
            frame = self

            class _Accessor:
                def cdl_alpha(self, append=True):
                    _ = append
                    values = [0.0] * len(frame)
                    values[-1] = 200.0
                    frame["cdl_alpha"] = values

            return _Accessor()

    now_ts = float(datetime.now(timezone.utc).timestamp())

    monkeypatch.setattr(candlestick_mod, "_ensure_candlestick_runtime", lambda: None)
    monkeypatch.setattr(candlestick_mod, "TIMEFRAME_MAP", {"H1": 1})
    requested_counts = []

    def _fetch_history_frame(_symbol, _timeframe, count, **kwargs):
        requested_counts.append(count)
        assert kwargs.get("include_incomplete") is False
        return _FakeFrame(
            {
                "time": [now_ts - 7200.0, now_ts - 3600.0],
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
            }
        )

    monkeypatch.setattr(
        candlestick_mod, "_get_candlestick_pattern_methods", lambda _temp: ["cdl_alpha"]
    )
    monkeypatch.setattr(data_service_mod, "fetch_history_frame", _fetch_history_frame)

    res = candlestick_mod.detect_candlestick_patterns(
        symbol="EURUSD",
        timeframe="H1",
        limit=2,
        min_strength=0.70,
        min_gap=0,
        robust_only=False,
        whitelist=None,
        top_k=2,
    )

    assert res["success"] is True
    assert requested_counts == [
        2 + candlestick_mod._candlestick_volume_warmup_bars(None)
    ]
    assert res["candles"] == 2
    assert res["lookback_satisfied"] is True
    assert len(res["data"]) == 1
    assert res["data"][0]["end_index"] == 1


@pytest.mark.parametrize(
    ("end", "expected_hour"),
    [
        ("2026-08-14T12:30:00Z", 11),
        ("2026-08-14T13:00:00Z", 12),
    ],
)
def test_detect_candlestick_patterns_historical_end_uses_closed_bar_cutoff(
    monkeypatch, end, expected_hour
):
    class _FakeFrame(pd.DataFrame):
        @property
        def _constructor(self):
            return _FakeFrame

        @property
        def ta(self):
            frame = self

            class _Accessor:
                def cdl_alpha(self, append=True):
                    _ = append
                    values = [0.0] * len(frame)
                    values[-1] = 200.0
                    frame["cdl_alpha"] = values

            return _Accessor()

    class _FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            current = datetime(2026, 8, 14, 15, 0, tzinfo=timezone.utc)
            return current if tz is not None else current.replace(tzinfo=None)

    opens = [
        datetime(2026, 8, 14, hour, 0, tzinfo=timezone.utc).timestamp()
        for hour in range(8, 14)
    ]

    expected_last = datetime(
        2026, 8, 14, expected_hour, 0, tzinfo=timezone.utc
    )
    closed_opens = [ts for ts in opens if ts <= expected_last.timestamp()]

    def _fetch_history_frame(_symbol, _timeframe, _count, **kwargs):
        assert kwargs.get("end") == end
        assert kwargs.get("include_incomplete") is False
        return _FakeFrame(
            {
                "time": closed_opens,
                "open": [100.0 + index for index in range(len(closed_opens))],
                "high": [101.0 + index for index in range(len(closed_opens))],
                "low": [99.0 + index for index in range(len(closed_opens))],
                "close": [100.5 + index for index in range(len(closed_opens))],
            }
        )

    monkeypatch.setattr(candlestick_mod, "datetime", _FrozenDateTime)
    monkeypatch.setattr(candlestick_mod, "_ensure_candlestick_runtime", lambda: None)
    monkeypatch.setattr(candlestick_mod, "_use_client_tz", lambda: False)
    monkeypatch.setattr(candlestick_mod, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(
        candlestick_mod, "_get_candlestick_pattern_methods", lambda _temp: ["cdl_alpha"]
    )
    monkeypatch.setattr(data_service_mod, "fetch_history_frame", _fetch_history_frame)

    res = candlestick_mod.detect_candlestick_patterns(
        symbol="EURUSD",
        timeframe="H1",
        limit=10,
        min_strength=0.70,
        min_gap=0,
        robust_only=False,
        whitelist=None,
        top_k=2,
        end=end,
    )

    expected_last = datetime(
        2026, 8, 14, expected_hour, 0, tzinfo=timezone.utc
    )
    assert res["success"] is True
    assert res["data"][0]["time"] == expected_last.strftime("%Y-%m-%dT%H:%MZ")
    assert res["data"][0]["end_index"] == res["candles"] - 1


def test_detect_candlestick_patterns_requests_closed_history_bars(
    monkeypatch,
):
    class _FakeFrame(pd.DataFrame):
        @property
        def _constructor(self):
            return _FakeFrame

        @property
        def ta(self):
            frame = self

            class _Accessor:
                def cdl_alpha(self, append=True):
                    _ = append
                    values = [0.0] * len(frame)
                    values[-1] = 200.0
                    frame["cdl_alpha"] = values

            return _Accessor()

    now_ts = float(datetime.now(timezone.utc).timestamp())
    last_bar_open = now_ts - 3660.0

    monkeypatch.setattr(candlestick_mod, "_ensure_candlestick_runtime", lambda: None)
    monkeypatch.setattr(candlestick_mod, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(
        candlestick_mod, "_get_candlestick_pattern_methods", lambda _temp: ["cdl_alpha"]
    )

    def _fetch_history_frame(_symbol, _timeframe, _count, **kwargs):
        assert kwargs.get("include_incomplete") is False
        return _FakeFrame(
            {
                "time": [last_bar_open - 7200.0, last_bar_open - 3600.0],
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
            }
        )

    monkeypatch.setattr(data_service_mod, "fetch_history_frame", _fetch_history_frame)

    res = candlestick_mod.detect_candlestick_patterns(
        symbol="EURUSD",
        timeframe="H1",
        limit=10,
        min_strength=0.70,
        min_gap=0,
        robust_only=False,
        whitelist=None,
        top_k=2,
    )

    assert res["success"] is True
    assert len(res["data"]) == 1
    assert res["data"][0]["end_index"] == 1


def test_detect_candlestick_patterns_exposes_raw_signal_and_quality_warning(
    monkeypatch,
):
    class _FakeFrame(pd.DataFrame):
        @property
        def _constructor(self):
            return _FakeFrame

        @property
        def ta(self):
            frame = self

            class _Accessor:
                def cdl_alpha(self, append=True):
                    _ = append
                    frame["cdl_alpha"] = [0.0, 200.0, 0.0]

            return _Accessor()

    monkeypatch.setattr(candlestick_mod, "_ensure_candlestick_runtime", lambda: None)
    monkeypatch.setattr(candlestick_mod, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(
        candlestick_mod, "_get_candlestick_pattern_methods", lambda _temp: ["cdl_alpha"]
    )

    def _fake_rates_to_df(*_a, **_k):
        return _FakeFrame(
            {
                "time": [1_700_000_000.0, 1_700_003_600.0, 1_700_007_200.0],
                "open": [100.0, 100.0, 100.0],
                "high": [100.0, 100.0, 100.0],
                "low": [100.0, 100.0, 100.0],
                "close": [100.0, 100.0, 100.0],
            }
        )

    monkeypatch.setattr(data_service_mod, "fetch_history_frame", _fake_rates_to_df)

    res = candlestick_mod.detect_candlestick_patterns(
        symbol="EURUSD",
        timeframe="H1",
        limit=10,
        min_strength=0.75,
        min_gap=0,
        robust_only=False,
        whitelist=None,
        top_k=1,
    )

    assert res["success"] is True
    assert res["data"][0]["raw_signal"] == 200
    assert "warnings" in res
    assert any("repeated close prices" in warning for warning in res["warnings"])


def test_detect_candlestick_patterns_adds_volume_and_regime_enrichment(monkeypatch):
    class _FakeFrame(pd.DataFrame):
        @property
        def _constructor(self):
            return _FakeFrame

        @property
        def ta(self):
            frame = self

            class _Accessor:
                def cdl_alpha(self, append=True):
                    _ = append
                    frame["cdl_alpha"] = [0.0] * (len(frame) - 1) + [200.0]

            return _Accessor()

    monkeypatch.setattr(candlestick_mod, "_ensure_candlestick_runtime", lambda: None)
    monkeypatch.setattr(candlestick_mod, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(
        candlestick_mod, "_get_candlestick_pattern_methods", lambda _temp: ["cdl_alpha"]
    )

    def _fake_rates_to_df(*_a, **_k):
        rows = 25
        return _FakeFrame(
            {
                "time": [1_700_000_000.0 + (3600.0 * i) for i in range(rows)],
                "open": [100.0 + (0.3 * i) for i in range(rows)],
                "high": [100.5 + (0.3 * i) for i in range(rows)],
                "low": [99.8 + (0.3 * i) for i in range(rows)],
                "close": [100.0 + (0.35 * i) for i in range(rows)],
                "tick_volume": [100 + (3 * i) for i in range(rows - 1)] + [260],
            }
        )

    monkeypatch.setattr(data_service_mod, "fetch_history_frame", _fake_rates_to_df)

    res = candlestick_mod.detect_candlestick_patterns(
        symbol="EURUSD",
        timeframe="H1",
        limit=20,
        min_strength=0.90,
        min_gap=0,
        robust_only=False,
        whitelist=None,
        top_k=1,
        config={
            "use_volume_confirmation": True,
            "use_regime_context": True,
            "volume_confirm_min_ratio": 1.1,
        },
    )

    row = res["data"][0]
    assert row["volume_confirmation"]["status"] == "confirmed"
    assert row["volume_confirmation"]["baseline_sufficient"] is True
    assert row["regime_context"]["status"] == "aligned"
    assert row["end_index"] == 24
    assert res["candles"] == 20
    assert res["min_strength_stage"] == "post_confirmation"


def test_detect_candlestick_patterns_reapplies_min_strength_after_enrichment(
    monkeypatch,
):
    class _FakeFrame(pd.DataFrame):
        @property
        def _constructor(self):
            return _FakeFrame

        @property
        def ta(self):
            frame = self

            class _Accessor:
                def cdl_engulfing(self, append=True):
                    _ = append
                    frame["cdl_engulfing"] = [0.0] * (len(frame) - 1) + [100.0]

            return _Accessor()

    monkeypatch.setattr(candlestick_mod, "_ensure_candlestick_runtime", lambda: None)
    monkeypatch.setattr(candlestick_mod, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(
        candlestick_mod,
        "_get_candlestick_pattern_methods",
        lambda _temp: ["cdl_engulfing"],
    )

    def _fake_rates_to_df(*_a, **_k):
        rows = 25
        return _FakeFrame(
            {
                "time": [1_700_000_000.0 + (3600.0 * i) for i in range(rows)],
                "open": [100.0 - (0.2 * i) for i in range(rows)],
                "high": [100.4 - (0.2 * i) for i in range(rows)],
                "low": [99.8 - (0.2 * i) for i in range(rows)],
                "close": [100.0 - (0.25 * i) for i in range(rows)],
                "tick_volume": ([100.0] * (rows - 2)) + [50.0, 50.0],
            }
        )

    monkeypatch.setattr(data_service_mod, "fetch_history_frame", _fake_rates_to_df)

    res = candlestick_mod.detect_candlestick_patterns(
        symbol="EURUSD",
        timeframe="H1",
        limit=25,
        min_strength=0.95,
        min_gap=0,
        robust_only=False,
        whitelist=None,
        top_k=1,
        config={
            "use_volume_confirmation": True,
            "use_regime_context": True,
            "volume_confirm_min_ratio": 1.1,
            "volume_confirm_penalty": 0.06,
            "regime_countertrend_penalty": 0.05,
        },
    )

    assert res["success"] is True
    assert res["count"] == 0
    assert res["data"] == []


def test_attach_candlestick_volume_confirmation_uses_full_multibar_signal_window():
    row = {"start_index": 20, "end_index": 22, "confidence": 0.4}
    volume = np.full(30, 100.0, dtype=float)
    volume[20:23] = np.array([220.0, 90.0, 90.0], dtype=float)

    candlestick_mod._attach_candlestick_volume_confirmation(
        row,
        volume,
        "tick_volume",
        {"use_volume_confirmation": True, "volume_confirm_min_ratio": 1.1},
    )

    assert row["volume_confirmation"]["status"] == "confirmed"
    assert row["volume_confirmation"]["baseline_sufficient"] is True
    assert row["volume_confirmation"]["baseline_bars_used"] == 20
    assert row["volume_confirmation"]["baseline_bars_required"] == 20
    assert row["volume_confirmation"]["signal_avg_volume"] > 130.0
    assert row["volume_confirmation"]["signal_to_baseline_ratio"] > 1.1


def test_attach_candlestick_volume_confirmation_skips_partial_baseline():
    row = {"start_index": 2, "end_index": 2, "confidence": 0.5}
    volume = np.array([100.0, 100.0, 300.0], dtype=float)

    candlestick_mod._attach_candlestick_volume_confirmation(
        row,
        volume,
        "tick_volume",
        {"use_volume_confirmation": True, "volume_confirm_min_ratio": 1.1},
    )

    confirmation = row["volume_confirmation"]
    assert confirmation["status"] == "insufficient"
    assert confirmation["baseline_sufficient"] is False
    assert confirmation["baseline_bars_used"] < confirmation["baseline_bars_required"]
    assert "confidence_delta" not in confirmation
    assert row["confidence"] == 0.5


def test_attach_candlestick_volume_confirmation_omits_unused_saturated_delta():
    row = {"start_index": 20, "end_index": 21, "confidence": 1.0, "status": "closed"}
    volume = np.concatenate(
        [np.full(20, 100.0, dtype=float), np.array([300.0, 300.0], dtype=float)]
    )

    candlestick_mod._attach_candlestick_volume_confirmation(
        row,
        volume,
        "tick_volume",
        {
            "use_volume_confirmation": True,
            "volume_confirm_min_ratio": 1.1,
            "volume_confirm_bonus": 0.08,
            "volume_confirm_lookback_bars": 20,
            "volume_confirm_breakout_bars": 2,
        },
    )

    confirmation = row["volume_confirmation"]
    assert confirmation["status"] == "confirmed"
    assert "confidence_delta" not in confirmation
    assert row["confidence"] == 1.0


def test_extract_candlestick_rows_respects_start_index():
    df = pd.DataFrame({"time": ["T0", "T1", "T2"], "close": [100.0, 101.0, 102.0]})
    temp = pd.DataFrame({"cdl_engulfing": [100.0, 100.0, 100.0]})
    rows = _extract_candlestick_rows(
        df,
        temp,
        ["cdl_engulfing"],
        threshold=0.95,
        robust_only=True,
        robust_set={"engulfing"},
        whitelist_set=None,
        min_gap=0,
        top_k=1,
        deprioritize=set(),
        include_metrics=True,
        start_index=2,
    )
    assert len(rows) == 1
    assert rows[0][0] == "T2"


def test_candlestick_volume_confirmation_respects_dict_disable():
    row = {"start_index": 0, "end_index": 1, "confidence": 0.5}

    candlestick_mod._attach_candlestick_volume_confirmation(
        row,
        np.array([100.0, 110.0, 120.0], dtype=float),
        "volume",
        {"use_volume_confirmation": False},
    )

    assert row["volume_confirmation"]["status"] == "disabled"


def test_candlestick_bar_spans_keep_three_bar_patterns_aligned():
    assert candlestick_mod._CANDLESTICK_PATTERN_BAR_SPANS["2crows"] == 3
    assert candlestick_mod._CANDLESTICK_PATTERN_BAR_SPANS["hikkake"] == 3
    assert candlestick_mod._CANDLESTICK_PATTERN_BAR_SPANS["hikkakemod"] == 3


def test_candlestick_volume_confirmation_stable_with_declared_warmup(monkeypatch):
    visible = 25
    history = 80
    volume = np.full(history, 50.0, dtype=float)
    volume[-visible:] = 160.0
    volume[-1] = 320.0

    class _FakeFrame(pd.DataFrame):
        @property
        def _constructor(self):
            return _FakeFrame

        @property
        def ta(self):
            frame = self

            class _Accessor:
                def cdl_alpha(self, append=True):
                    _ = append
                    values = [0.0] * len(frame)
                    values[-visible] = 200.0
                    frame["cdl_alpha"] = values

            return _Accessor()

    monkeypatch.setattr(candlestick_mod, "_ensure_candlestick_runtime", lambda: None)
    monkeypatch.setattr(candlestick_mod, "TIMEFRAME_MAP", {"H1": 1})
    monkeypatch.setattr(
        candlestick_mod, "_get_candlestick_pattern_methods", lambda _temp: ["cdl_alpha"]
    )

    def _fake_rates_to_df(*args, **kwargs):
        count = int(args[2])
        rows = min(count, history)
        start = history - rows
        return _FakeFrame(
            {
                "time": [1_600_000_000.0 + (3600.0 * i) for i in range(start, history)],
                "open": [100.0] * rows,
                "high": [101.0] * rows,
                "low": [99.0] * rows,
                "close": [100.5] * rows,
                "tick_volume": volume[start:].tolist(),
            }
        )

    monkeypatch.setattr(data_service_mod, "fetch_history_frame", _fake_rates_to_df)

    config = {
        "use_volume_confirmation": True,
        "use_regime_context": False,
        "volume_confirm_min_ratio": 1.1,
    }
    short = candlestick_mod.detect_candlestick_patterns(
        symbol="EURUSD",
        timeframe="H1",
        limit=visible,
        min_strength=0.50,
        min_gap=0,
        robust_only=False,
        whitelist=None,
        top_k=1,
        config=config,
    )
    long = candlestick_mod.detect_candlestick_patterns(
        symbol="EURUSD",
        timeframe="H1",
        limit=history,
        min_strength=0.50,
        min_gap=0,
        robust_only=False,
        whitelist=None,
        top_k=1,
        last_n_bars=visible,
        config=config,
    )

    assert short["lookback_satisfied"] is True
    short_row = short["data"][0]
    long_row = long["data"][0]
    short_conf = short_row["volume_confirmation"]
    long_conf = long_row["volume_confirmation"]
    assert short_conf["status"] == long_conf["status"]
    assert short_conf["baseline_sufficient"] is True
    assert long_conf["baseline_sufficient"] is True
    assert short_conf.get("confidence_delta") == long_conf.get("confidence_delta")
    assert short_row["confidence"] == long_row["confidence"]
