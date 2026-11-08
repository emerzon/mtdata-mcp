"""Presentation policy for market-status payloads."""

from typing import Any, Dict

from .output_contract import normalize_output_verbosity_detail


def normalize_market_status_output(
    result: Dict[str, Any],
    *,
    detail: Any = None,
) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return dict(result)

    detail_mode = normalize_output_verbosity_detail(detail)
    out = dict(result)
    if detail_mode == "full":
        return out

    markets = out.get("markets")
    if out.get("mode") == "global":
        out.pop("markets", None)
    elif isinstance(markets, list):
        compact_markets = []
        for market in markets:
            if isinstance(market, dict):
                market = {key: value for key, value in market.items() if key != "message"}
            compact_markets.append(market)
        out["markets"] = compact_markets
    out.pop("message", None)
    out.pop("upcoming_holidays", None)
    out.pop("upcoming_holidays_count", None)
    out.pop("upcoming_holidays_summary", None)
    return out
