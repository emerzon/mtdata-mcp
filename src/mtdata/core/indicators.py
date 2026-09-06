import logging
import re
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import Field

from ..shared.constants import DEFAULT_ROW_LIMIT
from ..shared.schema import (
    CategoryLiteral,
    DetailLiteral,
    IndicatorNameLiteral,
)
from ..utils.indicators import clean_help_text as _clean_help_text
from ..utils.indicators import indicator_engine_provenance
from ..utils.indicators import list_ta_indicators as _list_ta_indicators
from ..utils.utils import _table_from_rows
from ._mcp_instance import mcp
from .error_envelope import build_error_payload
from .execution_logging import run_logged_operation
from .output_contract import build_pagination_meta, normalize_output_detail

logger = logging.getLogger(__name__)

_DOC_SECTION_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 _/\-]{1,48})\s*:\s*$")
_DOC_PARAM_RE = re.compile(r"^[\-\*\u2022]?\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^)]*\))?\s*:\s*(.+)$")
_DOC_SIG_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\s*\(.*\)\s*(?:->\s*.+)?$")
_TRADING_STYLES = {"intraday", "swing", "position"}
_DEFAULT_INDICATOR_ORDER = (
    "rsi",
    "macd",
    "atr",
    "ema",
    "sma",
    "bbands",
    "adx",
    "stoch",
    "vwap",
    "obv",
    "ichimoku",
    "cci",
)
_DEFAULT_INDICATOR_RANK = {
    name: index for index, name in enumerate(_DEFAULT_INDICATOR_ORDER)
}
_CURATED_INDICATOR_CALCULATIONS = {
    "rsi": (
        "Input: close. Wilder/RMA average gains and losses over length "
        "(default 14); RSI = 100 - 100 / (1 + avg_gain/avg_loss). "
        "Output column: rsi_{length}."
    ),
    "macd": (
        "Input: close. MACD line = EMA(fast, default 12) - EMA(slow, default 26); "
        "signal = EMA(MACD, default 9); histogram = MACD - signal. "
        "Output columns: macd_{fast}_{slow}_{signal}, "
        "macd_h_{fast}_{slow}_{signal}, macd_s_{fast}_{slow}_{signal}."
    ),
    "ema": (
        "Input: close. Exponential moving average with smoothing 2/(length+1) "
        "(default length 10). Output column: ema_{length}."
    ),
    "sma": (
        "Input: close. Simple mean of the last length closes (default 10). "
        "Output column: sma_{length}."
    ),
    "atr": (
        "Inputs: high, low, close. True range is max(high-low, |high-prev_close|, "
        "|low-prev_close|); ATR is a Wilder/RMA of true range over length "
        "(default 14). Output column: atr_{length}."
    ),
    "bbands": (
        "Input: close. Middle band = SMA(length, backend default 5); upper/lower = "
        "middle ± stdev_multiplier * rolling standard deviation (default 2). "
        "A conventional 20-bar setting is available as bbands(20,2) and is not "
        "the omitted-parameter default. "
        "Output columns: bbl_{length}_{std}, bbm_{length}_{std}, bbu_{length}_{std}, "
        "bbb_{length}_{std}, bbp_{length}_{std}."
    ),
}


def _backend_calculation_reference(name: str) -> str:
    from importlib import metadata as importlib_metadata

    version = None
    try:
        version = importlib_metadata.version("pandas-ta-classic")
    except importlib_metadata.PackageNotFoundError:
        version = None
    library = "pandas-ta-classic"
    version_text = f" {version}" if version else ""
    return (
        f"See {library}{version_text} documentation for {name}. "
        "Indicator values can differ when TA-Lib is installed."
    )


_CATEGORY_TRADING_CONTEXT: Dict[str, Dict[str, Any]] = {
    "momentum": {
        "common_use": "momentum, reversal, and divergence checks",
        "typical_parameters": "use the library default first, then shorten for intraday or lengthen for position trades",
        "pairs_well_with": ["trend", "volume", "support_resistance"],
        "trading_styles": ["intraday", "swing", "position"],
    },
    "overlap": {
        "common_use": "trend direction, dynamic support/resistance, and pullback context",
        "typical_parameters": "20/50 for swing context; 100/200 for higher-timeframe trend",
        "pairs_well_with": ["momentum", "atr", "volume"],
        "trading_styles": ["intraday", "swing", "position"],
    },
    "trend": {
        "common_use": "trend strength, trend-following filters, and regime confirmation",
        "typical_parameters": "start with defaults; confirm against price structure and ATR",
        "pairs_well_with": ["moving_average", "momentum", "atr"],
        "trading_styles": ["swing", "position", "intraday"],
    },
    "volatility": {
        "common_use": "range expansion, stop distance, and breakout compression checks",
        "typical_parameters": "14 or 20 periods are common baselines",
        "pairs_well_with": ["trend", "support_resistance", "volume"],
        "trading_styles": ["intraday", "swing", "position"],
    },
    "volume": {
        "common_use": "participation, accumulation/distribution, and move confirmation",
        "typical_parameters": "compare to recent baseline; FX volume is usually tick volume",
        "pairs_well_with": ["price_action", "momentum", "volatility"],
        "trading_styles": ["intraday", "swing"],
    },
    "statistics": {
        "common_use": "normalization, mean reversion, and outlier checks",
        "typical_parameters": "20-period windows are common for rolling context",
        "pairs_well_with": ["volatility", "trend"],
        "trading_styles": ["intraday", "swing", "position"],
    },
    "performance": {
        "common_use": "return and drawdown context for research workflows",
        "typical_parameters": "align window to the holding period under test",
        "pairs_well_with": ["strategy_backtest", "risk"],
        "trading_styles": ["swing", "position"],
    },
    "candles": {
        "common_use": "price-action pattern context and candle-shape transforms",
        "typical_parameters": "use defaults, then validate on the traded timeframe",
        "pairs_well_with": ["support_resistance", "volume", "trend"],
        "trading_styles": ["intraday", "swing"],
    },
    "cycles": {
        "common_use": "cycle phase and turning-point research",
        "typical_parameters": "use as exploratory context, not a standalone trigger",
        "pairs_well_with": ["momentum", "trend"],
        "trading_styles": ["swing", "position"],
    },
}
_INDICATOR_TRADING_CONTEXT: Dict[str, Dict[str, Any]] = {
    "rsi": {
        "common_use": "overbought/oversold, momentum reversal, and divergence checks",
        "typical_parameters": "rsi(14); 70/30 bands are common, 80/20 is stricter",
        "pairs_well_with": ["macd", "atr", "support_resistance"],
        "trading_styles": ["intraday", "swing", "position"],
    },
    "macd": {
        "common_use": "trend-momentum shifts and pullback continuation confirmation",
        "typical_parameters": "macd(12,26,9) is the standard baseline",
        "pairs_well_with": ["ema", "rsi", "volume"],
        "trading_styles": ["intraday", "swing", "position"],
    },
    "bbands": {
        "common_use": "volatility contraction/expansion and mean-reversion context",
        "typical_parameters": "bbands(20,2); wider bands reduce false touches",
        "pairs_well_with": ["rsi", "atr", "volume"],
        "trading_styles": ["intraday", "swing"],
    },
    "atr": {
        "common_use": "stop distance, range expansion, and volatility-normalized sizing",
        "typical_parameters": "atr(14) is the standard baseline",
        "pairs_well_with": ["trend", "support_resistance", "barrier_tools"],
        "trading_styles": ["intraday", "swing", "position"],
    },
    "adx": {
        "common_use": "trend-strength filter before applying trend-following entries",
        "typical_parameters": "adx(14); readings above 20-25 often mark stronger trend",
        "pairs_well_with": ["ema", "macd", "atr"],
        "trading_styles": ["swing", "position", "intraday"],
    },
    "ema": {
        "common_use": "trend direction, pullback zones, and fast/slow cross context",
        "typical_parameters": "ema(20), ema(50), ema(200) are common baselines",
        "pairs_well_with": ["macd", "atr", "rsi"],
        "trading_styles": ["intraday", "swing", "position"],
    },
    "sma": {
        "common_use": "higher-timeframe trend and widely watched support/resistance",
        "typical_parameters": "sma(20), sma(50), sma(200) are common baselines",
        "pairs_well_with": ["rsi", "atr", "volume"],
        "trading_styles": ["swing", "position"],
    },
    "stoch": {
        "common_use": "range-bound momentum reversals and short-term exhaustion",
        "typical_parameters": "stoch(14,3,3) is a common baseline",
        "pairs_well_with": ["bbands", "support_resistance", "atr"],
        "trading_styles": ["intraday", "swing"],
    },
    "supertrend": {
        "common_use": "trend-following bias and trailing-stop context",
        "typical_parameters": "supertrend(7,3) is a common starting point",
        "pairs_well_with": ["atr", "adx", "ema"],
        "trading_styles": ["intraday", "swing"],
    },
    "vwap": {
        "common_use": "intraday fair-value reference and institutional participation context",
        "typical_parameters": "daily VWAP; reset at the configured broker-server day boundary",
        "pairs_well_with": ["volume", "rsi", "price_action"],
        "trading_styles": ["intraday"],
    },
    "obv": {
        "common_use": "volume confirmation and accumulation/distribution divergence",
        "typical_parameters": "use default OBV and compare slope/divergence to price",
        "pairs_well_with": ["macd", "support_resistance", "trend"],
        "trading_styles": ["swing", "position", "intraday"],
    },
}
_RELATED_INDICATORS: Dict[str, List[str]] = {
    "rsi": ["stochrsi", "tsi", "mfi"],
    "macd": ["ppo", "ema", "adx"],
    "bbands": ["kc", "donchian", "atr"],
    "atr": ["natr", "bbands", "kc"],
    "adx": ["aroon", "chop", "vortex"],
    "ema": ["sma", "hma", "macd"],
    "sma": ["ema", "wma", "vwma"],
    "stoch": ["stochrsi", "rsi", "willr"],
    "supertrend": ["atr", "adx", "ema"],
    "vwap": ["vwma", "obv", "pvt"],
    "obv": ["pvt", "ad", "mfi"],
}


def _canonical_doc_section(name: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", str(name or "").strip().lower()).strip("_")
    aliases = {
        "arg": "parameters",
        "args": "parameters",
        "argument": "parameters",
        "arguments": "parameters",
        "param": "parameters",
        "params": "parameters",
        "kwargs": "parameters",
        "keyword_arguments": "parameters",
        "sources": "sources",
        "source": "sources",
        "references": "sources",
        "reference": "sources",
        "calculation": "calculation",
        "calculations": "calculation",
        "formula": "calculation",
        "formulas": "calculation",
        "interpretation": "interpretation",
        "interpretations": "interpretation",
        "notes": "interpretation",
        "signals": "interpretation",
    }
    return aliases.get(key, key)


def _parse_doc_sections(text: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {"overview": []}
    current = "overview"
    for raw_line in str(text or "").splitlines():
        line = str(raw_line or "").strip()
        if not line:
            continue
        match = _DOC_SECTION_RE.match(line)
        if match:
            current = _canonical_doc_section(match.group(1))
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return sections


def _parse_parameter_docs(lines: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    current_param: Optional[str] = None
    for line in lines or []:
        text = str(line or "").strip()
        if not text:
            continue
        match = _DOC_PARAM_RE.match(text)
        if match:
            pname = str(match.group(1)).strip()
            pdesc = str(match.group(2)).strip()
            if not pname or not pdesc:
                current_param = None
                continue
            if pname in out:
                out[pname] = f"{out[pname]} {pdesc}".strip()
            else:
                out[pname] = pdesc
            current_param = pname
            continue
        if current_param and not _DOC_SECTION_RE.match(text):
            out[current_param] = f"{out[current_param]} {text}".strip()
            continue
        current_param = None
    return out


def _join_doc_lines(lines: List[str]) -> str:
    clean = [str(x).strip() for x in (lines or []) if str(x).strip()]
    return "\n".join(clean).strip()


def _clean_overview_lines(lines: List[str]) -> List[str]:
    cleaned: List[str] = []
    in_signature = False
    paren_depth = 0
    for raw in lines or []:
        text = str(raw or "").strip()
        if not text:
            continue
        if text.lower().startswith("python library documentation:"):
            continue
        if _DOC_SIG_RE.match(text):
            continue
        if not cleaned and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*\($", text):
            in_signature = True
            paren_depth = text.count("(") - text.count(")")
            continue
        if in_signature:
            paren_depth += text.count("(") - text.count(")")
            if paren_depth <= 0:
                in_signature = False
            continue
        cleaned.append(text)
    return cleaned


def _extract_interpretation(sections: Dict[str, List[str]]) -> Optional[str]:
    explicit = _join_doc_lines(sections.get("interpretation", []))
    if explicit:
        return explicit
    overview = _clean_overview_lines(sections.get("overview", []))
    return _join_doc_lines(overview) or None


def _extract_short_description(description: Optional[str]) -> Optional[str]:
    """Extract first line or first sentence from description for compact display."""
    if not description:
        return None
    text = str(description or "").strip()
    if not text:
        return None
    # Get first line
    lines = text.split('\n')
    first_line = lines[0].strip()
    if not first_line:
        return None
    # Truncate to reasonable length for compact display (around 80 chars)
    if len(first_line) > 80:
        # Try to break at a word boundary
        truncated = first_line[:77]
        last_space = truncated.rfind(' ')
        if last_space > 40:  # Only break if we have at least 40 chars
            return truncated[:last_space].strip() + "..."
        return truncated.strip() + "..."
    return first_line


def _indicator_trading_context(item: Dict[str, Any]) -> Dict[str, Any]:
    name = str(item.get("name") or "").strip().lower()
    category = str(item.get("category") or "").strip().lower()
    context = dict(_CATEGORY_TRADING_CONTEXT.get(category, {}))
    if context:
        context["trading_styles_basis"] = "category_heuristic"
        context["trading_styles_note"] = (
            "Broad workflow tag inherited from the indicator category; it is not "
            "an indicator-specific recommendation or performance claim."
        )
    indicator_context = _INDICATOR_TRADING_CONTEXT.get(name)
    if indicator_context:
        context.update(indicator_context)
        context["trading_styles_basis"] = "curated_indicator"
        context["trading_styles_note"] = (
            "Indicator-specific workflow tag curated for catalog discovery; validate "
            "parameters and performance on the intended timeframe."
        )
    return context


def _matches_trading_style(item: Dict[str, Any], style: str) -> bool:
    normalized = str(style or "").strip().lower()
    if not normalized:
        return True
    context = _indicator_trading_context(item)
    return normalized in {str(x).strip().lower() for x in context.get("trading_styles", [])}


def _default_indicator_sort_key(item: Dict[str, Any]) -> tuple[int, int, str, str]:
    """Put broadly used indicators ahead of the alphabetical catalog."""
    name = str(item.get("name") or "").strip().lower()
    rank = _DEFAULT_INDICATOR_RANK.get(name)
    if rank is not None:
        return 0, rank, "", name
    return 1, len(_DEFAULT_INDICATOR_ORDER), str(item.get("category") or ""), name


def _compact_indicator_spec_is_parser_valid(spec: str) -> bool:
    text = str(spec or "").strip()
    match = re.fullmatch(r"([A-Za-z0-9_]+)(?:\((.*)\))?", text)
    if not match:
        return False
    params_blob = match.group(2)
    if params_blob is None or not params_blob.strip():
        return True
    for raw_part in params_blob.split(","):
        part = raw_part.strip()
        if not part:
            continue
        value_text = part.split("=", 1)[-1].strip()
        try:
            float(value_text)
        except (TypeError, ValueError):
            return False
    return True


_PERIOD_PARAM_NAMES = {
    "length",
    "period",
    "timeperiod",
    "n",
    "fast",
    "slow",
    "signal",
}
_INDICATOR_PERIOD_FALLBACKS = {
    "cmo": 14,
}


def _indicator_preferred_spec(name: str, params: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
    lname = str(name or "").strip().lower()
    if lname == "vwap":
        return "vwap"
    if lname == "cdl_pattern":
        return "cdl_pattern"
    period_defaults: List[Any] = []
    non_period_defaults: List[Any] = []
    for raw in params or []:
        if not isinstance(raw, dict):
            continue
        default = raw.get("default")
        if isinstance(default, bool) or default is None:
            continue
        if not isinstance(default, (int, float)):
            continue
        param_name = str(raw.get("name") or "").strip().lower()
        if param_name in _PERIOD_PARAM_NAMES:
            period_defaults.append(default)
        else:
            non_period_defaults.append(default)
    typical = str(context.get("typical_parameters") or "")
    match = re.search(rf"\b{re.escape(lname)}\([^)]*\)", typical, flags=re.IGNORECASE)
    if match:
        spec = match.group(0).lower()
        numbers = [
            float(part.strip())
            for part in (match.group(0).split("(", 1)[-1].rstrip(")").split(","))
            if part.strip()
            and re.fullmatch(r"-?\d+(?:\.\d+)?", part.strip())
        ]
        first_number = numbers[0] if numbers else None
        uses_non_period_default = (
            first_number is not None
            and any(abs(float(first_number) - float(value)) < 1e-12 for value in non_period_defaults)
            and not any(abs(float(first_number) - float(value)) < 1e-12 for value in period_defaults)
        )
        if _compact_indicator_spec_is_parser_valid(spec) and not uses_non_period_default:
            return spec
    if period_defaults:
        spec = f"{lname}({period_defaults[0]})"
        if _compact_indicator_spec_is_parser_valid(spec):
            return spec
    fallback = _INDICATOR_PERIOD_FALLBACKS.get(lname)
    if fallback is not None:
        spec = f"{lname}({fallback})"
        if _compact_indicator_spec_is_parser_valid(spec):
            return spec
    return lname


def _suggest_indicator_catalog_name(name: Any, items: List[Dict[str, Any]]) -> Optional[str]:
    requested = str(name or "").strip().lower()
    if not requested:
        return None
    catalog = {
        str(item.get("name") or "").strip()
        for item in items
        if str(item.get("name") or "").strip()
    }
    catalog_lookup = {value.lower(): value for value in catalog}
    match = re.fullmatch(r"([a-z][a-z0-9_]*)[_\-]([0-9]{1,3})", requested)
    if not match:
        return None
    return catalog_lookup.get(match.group(1))


def _indicator_example_column(spec: str) -> str:
    text = str(spec or "").strip().lower()
    match = re.fullmatch(r"([a-z0-9_]+)(?:\(([^)]*)\))?", text)
    if not match:
        return text.replace("(", "_").replace(")", "").replace(",", "_")
    name = match.group(1)
    args = [
        re.sub(r"[^a-z0-9_.-]+", "", part.split("=", 1)[-1].strip().lower())
        for part in (match.group(2) or "").split(",")
        if part.strip()
    ]
    return "_".join([name, *args]) if args else name


def _indicator_usage_metadata(
    name: str,
    params: List[Dict[str, Any]],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    lname = str(name or "").strip().lower()
    if lname == "cdl_pattern":
        return {
            "compact_spec": None,
            "cli_supported": False,
            "note": (
                "cdl_pattern is not callable via compact CLI because pattern names "
                "are non-numeric. Use patterns_detect(mode='candlestick') to scan "
                "named candlesticks."
            ),
            "alternative": {
                "tool": "patterns_detect",
                "example": 'patterns_detect(symbol="EURUSD", mode="candlestick")',
            },
        }
    spec = _indicator_preferred_spec(name, params, context)
    return {
        "compact_spec": spec,
        "cli": f'--indicators "{spec}"',
        "python": f'indicators="{spec}"',
        "column_output": {
            "example": _indicator_example_column(spec),
            "note": "Exact column names are produced by the pandas-ta backend and may include additional suffixes for multi-output indicators.",
        },
    }


def _build_indicator_documentation(target: Dict[str, Any]) -> Dict[str, Any]:
    name = str(target.get("name") or "")
    raw_desc = str(target.get("description") or "")
    cleaned_desc = _clean_help_text(raw_desc, func_name=name) if raw_desc else ""
    sections = _parse_doc_sections(cleaned_desc)
    overview = _clean_overview_lines(sections.get("overview", []))
    param_docs = _parse_parameter_docs(sections.get("parameters", []))

    params_out: List[Dict[str, Any]] = []
    for raw in (target.get("params") or []):
        if not isinstance(raw, dict):
            continue
        p = dict(raw)
        pname = str(p.get("name") or "").strip()
        if pname and pname in param_docs and not p.get("description"):
            p["description"] = param_docs[pname]
        params_out.append(p)

    calc_text = _join_doc_lines(sections.get("calculation", [])) or None
    if not calc_text:
        calc_text = _CURATED_INDICATOR_CALCULATIONS.get(name.strip().lower())
    if not calc_text:
        calc_text = _backend_calculation_reference(name)
    interp_text = _extract_interpretation(sections)
    sources = []
    for item in sections.get("sources", []):
        src = re.sub(r"^[\-\*\u2022]\s*", "", str(item or "").strip())
        if src:
            sources.append(src)

    return {
        "description": _join_doc_lines(overview) or cleaned_desc,
        "calculation": calc_text,
        "parameters": params_out,
        "interpretation": interp_text,
        "sources": sources,
    }


def _indicator_search_rank(item: Dict[str, Any], query: str) -> tuple[int, str] | None:
    q = str(query or "").strip().lower()
    if not q:
        return None

    name = str(item.get("name") or "").strip().lower()
    category = str(item.get("category") or "").strip().lower()

    if name == q:
        return (0, name)
    if name.startswith(q):
        return (1, name)
    if q in name:
        return (2, name)
    if category == q:
        return (3, name)
    if category.startswith(q):
        return (4, name)
    if q in category:
        return (5, name)
    return None


def _format_indicator_param_summary(params: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for raw in params or []:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        if not name:
            continue
        if "default" in raw:
            parts.append(f"{name}={raw.get('default')}")
        else:
            parts.append(name)
        if len(parts) >= 4:
            break
    return ",".join(parts)


def _describe_indicator_params(
    params: List[Dict[str, Any]],
    *,
    include_descriptions: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw in params or []:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        if not name:
            continue
        row: Dict[str, Any] = {"name": name}
        if "default" in raw:
            row["default"] = raw.get("default")
        if include_descriptions and raw.get("description") not in (None, ""):
            row["description"] = raw.get("description")
        rows.append(row)
    return rows


def _normalize_indicator_list_limit(limit: Any) -> tuple[Optional[int], Optional[str]]:
    if limit is None:
        return None, None
    try:
        limit_value = int(float(limit))
    except (TypeError, ValueError, OverflowError):
        return None, f"Invalid limit: {limit}. Must be an integer >= 1."
    if limit_value <= 0:
        return None, f"Invalid limit: {limit_value}. Must be >= 1."
    return limit_value, None


@mcp.tool()
def indicators_list(  # noqa: C901
    search_term: Optional[str] = None,
    category: Optional[CategoryLiteral] = None,
    trading_style: Optional[Literal["intraday", "swing", "position"]] = None,
    limit: Annotated[int, Field(ge=1)] = DEFAULT_ROW_LIMIT,
    offset: Annotated[int, Field(ge=0)] = 0,
    detail: DetailLiteral = "compact",
) -> Dict[str, Any]:  # type: ignore
    """List indicators as a tabular result with optional search, category, and detail filters.

    Parameters: search_term?, category?, trading_style?, limit?, offset?, detail?
    """
    def _run() -> Dict[str, Any]:  # noqa: C901
        try:
            detail_mode = normalize_output_detail(detail, default="compact")
            detailed = detail_mode == "full"
            items = _list_ta_indicators(detailed=detailed)
            search_active = False
            if search_term:
                q = search_term.strip().lower()
                ranked = []
                for it in items:
                    rank = _indicator_search_rank(it, q)
                    if rank is not None:
                        ranked.append((rank, it))
                items = [it for _rank, it in sorted(ranked, key=lambda pair: pair[0])]
                search_active = True
            if category:
                cat_q = category.strip().lower()
                items = [it for it in items if (it.get('category') or '').lower() == cat_q]
            style_q = str(trading_style or "").strip().lower()
            if style_q and style_q not in _TRADING_STYLES:
                return {"error": f"Invalid trading_style: {trading_style}. Use intraday, swing, or position."}
            if style_q:
                items = [it for it in items if _matches_trading_style(it, style_q)]
            style_basis_counts: Dict[str, int] = {}
            if style_q:
                for item in items:
                    basis = str(
                        _indicator_trading_context(item).get("trading_styles_basis")
                        or "unknown"
                    )
                    style_basis_counts[basis] = style_basis_counts.get(basis, 0) + 1
            if not search_active and category is None and not style_q:
                items.sort(key=_default_indicator_sort_key)
            elif not search_active:
                items.sort(key=lambda x: (x.get('category') or '', x.get('name') or ''))
            total_matches = len(items)
            limit_value, limit_error = _normalize_indicator_list_limit(limit)
            if limit_error:
                return {"error": limit_error}
            try:
                offset_value = int(float(offset or 0))
            except Exception:
                return {"error": f"Invalid offset: {offset}. Must be a non-negative integer."}
            if offset_value < 0:
                return {"error": f"Invalid offset: {offset_value}. Must be >= 0."}
            if offset_value:
                items = items[offset_value:]
            if limit_value is not None:
                items = items[:limit_value]
            if detailed:
                rows = []
                for it in items:
                    docs = _build_indicator_documentation(it)
                    params = docs.get("parameters") or it.get("params") or []
                    rows.append(
                        [
                            it.get("name", ""),
                            it.get("category", ""),
                            _extract_short_description(docs.get("description") or it.get("description", "")),
                            len(params),
                            params,
                            _indicator_trading_context(it),
                            docs.get("description") or it.get("description", ""),
                        ]
                    )
                result = _table_from_rows(
                    [
                        "name",
                        "category",
                        "summary",
                        "params_count",
                        "params",
                        "trading_context",
                        "description",
                    ],
                    rows,
                )
            elif detail_mode == "standard":
                include_use = bool(search_active or style_q)
                headers = ["name", "category", "description", "params_count", "params"]
                if include_use:
                    headers.append("use")
                rows = []
                for it in items:
                    row = [
                        it.get('name',''),
                        it.get('category',''),
                        _extract_short_description(it.get('description', '')),
                        len(it.get("params") or []),
                        _format_indicator_param_summary(it.get("params") or []),
                    ]
                    if include_use:
                        row.append(_indicator_trading_context(it).get("common_use", ""))
                    rows.append(row)
                result = _table_from_rows(headers, rows)
            elif detail_mode == "summary":
                result = _table_from_rows(
                    ["name", "category", "description", "params_count"],
                    [
                        [
                            it.get("name", ""),
                            it.get("category", ""),
                            _extract_short_description(it.get("description", "")),
                            len(it.get("params") or []),
                        ]
                        for it in items
                    ],
                )
            else:
                result = _table_from_rows(
                    ["name", "category", "params_count"],
                    [
                        [
                            it.get("name", ""),
                            it.get("category", ""),
                            len(it.get("params") or []),
                        ]
                        for it in items
                    ],
                )
            result["detail"] = detail_mode
            if style_q:
                result["trading_style"] = style_q
                heuristic_matches = int(
                    style_basis_counts.get("category_heuristic", 0)
                )
                result["trading_style_filter"] = {
                    "requested": style_q,
                    "semantics": "broad_workflow_tag_not_performance_recommendation",
                    "curated_indicator_matches": int(
                        style_basis_counts.get("curated_indicator", 0)
                    ),
                    "category_heuristic_matches": heuristic_matches,
                    "unknown_basis_matches": int(style_basis_counts.get("unknown", 0)),
                }
                if heuristic_matches:
                    result["warnings"] = [
                        f"{heuristic_matches} match(es) inherit the {style_q} tag from "
                        "a broad indicator category. Inspect trading_context and the "
                        "indicator description before treating the tag as suitable."
                    ]
            more_available = max(0, total_matches - offset_value - len(items))
            result["pagination"] = build_pagination_meta(
                total=total_matches,
                returned=len(items),
                offset=offset_value,
                limit=limit_value,
            )
            if more_available > 0:
                result["search_hint"] = (
                    "Use search_term to match indicator names, "
                    "categories, or docs."
                )
            return result
        except Exception as exc:
            return {"error": f"Error listing indicators: {exc}"}

    return run_logged_operation(
        logger,
        operation="indicators_list",
        search_term=search_term,
        category=category,
        trading_style=trading_style,
        limit=limit,
        offset=offset,
        detail=detail,
        func=_run,
    )


# Note: category annotation is set at definition time above to be captured in the MCP schema

@mcp.tool()
def indicators_describe(
    name: IndicatorNameLiteral,
    detail: DetailLiteral = "compact",
) -> Dict[str, Any]:  # type: ignore
    """Return indicator information with compact, standard, or full detail.

    Parameters: name, detail
    """
    def _run() -> Dict[str, Any]:
        try:
            detail_mode = normalize_output_detail(detail, default="compact")
            items = _list_ta_indicators(detailed=True)
            target = next(
                (
                    it
                    for it in items
                    if it.get("name", "").lower() == str(name).lower()
                ),
                None,
            )
            if not target:
                suggested = _suggest_indicator_catalog_name(name, items)
                details: Dict[str, Any] = {"name": str(name)}
                if suggested:
                    details["suggested_name"] = suggested
                error = f"Indicator '{name}' not found"
                if suggested:
                    error = (
                        f"Indicator '{name}' not found. "
                        f"'{name}' looks like a fetch spec or output column; "
                        f"describe '{suggested}' instead."
                    )
                return build_error_payload(
                    error,
                    code="indicator_not_found",
                    operation="indicators_describe",
                    details=details,
                )
            indicator = dict(target)
            docs = _build_indicator_documentation(indicator)
            description = docs.get("description") or indicator.get("description") or ""
            params = docs.get("parameters") or indicator.get("params") or []
            trading_context = _indicator_trading_context(indicator)
            usage = _indicator_usage_metadata(str(indicator.get("name") or name), params, trading_context)
            interpretation = docs.get("interpretation") or trading_context.get("common_use")
            see_also = _RELATED_INDICATORS.get(str(indicator.get("name") or "").strip().lower(), [])
            compact_indicator: Dict[str, Any] = {
                "name": indicator.get("name"),
                "category": indicator.get("category"),
                "description": _extract_short_description(description) or description,
                "params": _describe_indicator_params(
                    params,
                    include_descriptions=False,
                ),
            }
            if trading_context:
                compact_indicator["trading_context"] = trading_context
            compact_indicator["usage"] = usage
            if interpretation:
                compact_indicator["interpretation"] = interpretation
            if see_also:
                compact_indicator["see_also"] = see_also
            if detail_mode in {"compact", "summary"}:
                return {
                    "success": True,
                    "detail": detail_mode,
                    "indicator": compact_indicator,
                }

            standard_indicator = dict(compact_indicator)
            standard_indicator["description"] = description
            standard_indicator["params"] = _describe_indicator_params(
                params,
                include_descriptions=True,
            )
            if detail_mode == "standard":
                return {
                    "success": True,
                    "detail": detail_mode,
                    "indicator": standard_indicator,
                }

            indicator["description"] = description
            indicator["params"] = params
            if trading_context:
                indicator["trading_context"] = trading_context
            indicator["usage"] = usage
            if interpretation:
                indicator["interpretation"] = interpretation
            if see_also:
                indicator["see_also"] = see_also
            indicator["documentation"] = {
                "calculation": docs.get("calculation"),
                "interpretation": docs.get("interpretation"),
                "sources": docs.get("sources") or [],
            }
            return {
                "success": True,
                "detail": "full",
                "indicator": indicator,
                "indicator_engine": indicator_engine_provenance(),
            }
        except Exception as exc:
            return {"error": f"Error getting indicator details: {exc}"}

    return run_logged_operation(
        logger,
        operation="indicators_describe",
        name=name,
        detail=detail,
        func=_run,
    )


