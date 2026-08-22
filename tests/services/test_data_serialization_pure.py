from __future__ import annotations

import math

from mtdata.services.data_service import ticks as data_service


def test_tick_missing_normalization_skips_pandas_checks_for_plain_scalars(monkeypatch) -> None:
    calls = []

    def unexpected_isna(value):
        calls.append(value)
        return False

    monkeypatch.setattr(data_service.pd, "isna", unexpected_isna)
    payload = {
        "rows": [[1, 1.5, True, None, "value"], [2, math.inf, False, -math.inf]],
    }

    out = data_service._normalize_tick_missing_values(payload)

    assert out == {
        "rows": [[1, 1.5, True, None, "value"], [2, None, False, None]],
    }
    assert calls == []
