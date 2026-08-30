"""Configuration settings for MetaTrader5 MCP Server."""

import logging
import os
import threading
import warnings
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..utils.coercion import UNPARSED_BOOL, parse_bool_like
from .env import get_bool_env, get_csv_env, get_float_env, get_int_env

try:
    from dateutil import tz as dateutil_tz  # type: ignore
except Exception:
    dateutil_tz = None  # optional

try:
    import tzlocal  # type: ignore
except Exception:
    tzlocal = None  # optional

_WARNED_STATIC_OFFSET_OVERRIDE = False
_ENV_LOADED = False
_ENV_LOAD_LOCK = threading.Lock()
_LOGGER = logging.getLogger(__name__)


def _zoneinfo_from_name(name: Optional[str]):
    if not name:
        return None
    try:
        return ZoneInfo(str(name))
    except (ZoneInfoNotFoundError, Exception):
        return None


def _detect_local_client_tz():
    """Best-effort local timezone detection for unset client timezone config."""
    if tzlocal is not None:
        try:
            tz = tzlocal.get_localzone()
            if tz is not None:
                return tz
        except Exception:
            pass
    if dateutil_tz is not None:
        try:
            if os.name == "nt" and hasattr(dateutil_tz, "tzwinlocal"):
                tz = dateutil_tz.tzwinlocal()
                if tz is not None:
                    return tz
        except Exception:
            pass
        try:
            tz = dateutil_tz.tzlocal()
            if tz is not None:
                return tz
        except Exception:
            pass
    try:
        return datetime.now().astimezone().tzinfo
    except Exception:
        return None


def _suppress_noisy_third_party_logs() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    for logger_name, level in (
        ("numba.cuda.cudadrv.driver", logging.WARNING),
        ("absl", logging.ERROR),
        ("huggingface_hub", logging.WARNING),
        ("timesfm", logging.WARNING),
        ("timesfm_2p5_torch", logging.WARNING),
        ("torch", logging.WARNING),
        ("torchao", logging.ERROR),
        ("torch.distributed", logging.ERROR),
        ("torch.distributed.elastic.multiprocessing.redirects", logging.ERROR),
        ("torch._dynamo", logging.ERROR),
        ("transformers", logging.WARNING),
        ("lightgbm", logging.ERROR),
        ("lightning", logging.ERROR),
        ("pytorch_lightning", logging.ERROR),
    ):
        try:
            noisy_logger = logging.getLogger(logger_name)
            noisy_logger.setLevel(level)
            noisy_logger.propagate = False
        except Exception:
            pass
    try:
        from pydantic_settings.sources.utils import IncompleteFieldDefinitionWarning

        warnings.filterwarnings(
            "ignore",
            message=(
                r"Field 'lifespan' has an incomplete definition: its annotation "
                r"contains an unresolved forward reference.*"
            ),
            category=IncompleteFieldDefinitionWarning,
            module=r"pydantic_settings\.sources\.utils",
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*Redirects are currently not supported in Windows or MacOs.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            category=ImportWarning,
            module=r"umap(\..*)?$",
        )
        warnings.filterwarnings(
            "ignore",
            message=(
                r'Field name "json" in "[^"]+Arguments" shadows an attribute '
                r'in parent "ArgModelBase"'
            ),
            category=UserWarning,
            module=r"pydantic\._internal\._fields",
        )
    except Exception:
        pass


_suppress_noisy_third_party_logs()


def _env_optional_int(name: str) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return int(text)
    except (TypeError, ValueError):
        _LOGGER.warning("Invalid %s=%r; ignoring configured login.", name, raw)
        return None


def _env_optional_float(name: str) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        value = float(text)
    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid {name}={raw!r}; expected a number, not a silent fallback."
        ) from None
    if not value == value:
        raise ValueError(f"Invalid {name}={raw!r}; NaN is not allowed.")
    return value


def _env_strict_optional_int(name: str) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return int(text)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid {name}={raw!r}; expected an integer.") from None


def _env_strict_bool(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    parsed = parse_bool_like(raw)
    if parsed is not UNPARSED_BOOL:
        return bool(parsed)
    raise ValueError(
        f"Invalid boolean {name}={raw!r}. "
        "Accepted values are: 0, 1, false, n, no, off, on, true, y, yes."
    )


def _env_symbol_float_map(name: str) -> dict[str, float]:
    raw = os.getenv(name)
    if raw is None:
        return {}
    mapping: dict[str, float] = {}
    for item in str(raw).split(","):
        token = item.strip()
        if not token:
            continue
        separator = ":" if ":" in token else "=" if "=" in token else None
        if separator is None:
            raise ValueError(
                f"Invalid {name} entry {token!r}; expected SYMBOL:VALUE or SYMBOL=VALUE."
            )
        symbol_part, value_part = token.split(separator, 1)
        symbol = symbol_part.strip().upper()
        if not symbol:
            raise ValueError(f"Invalid {name} entry {token!r}; symbol is blank.")
        try:
            value = float(value_part.strip())
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid {name} entry {token!r}; volume cap must be numeric."
            ) from None
        if value <= 0:
            raise ValueError(
                f"Invalid {name} entry {token!r}; volume cap must be positive."
            )
        mapping[symbol] = float(value)
    return mapping


class _GuardrailSection:
    def model_dump(self) -> dict[str, object]:
        # Exclude private runtime fields (locks, version counters) that are not
        # part of the serializable policy surface and cannot be deep-copied.
        return {
            key: value
            for key, value in self.__dict__.items()
            if not str(key).startswith("_")
        }

    def has_configured_values(self) -> bool:
        for value in self.model_dump().values():
            if isinstance(value, bool):
                if value:
                    return True
                continue
            if isinstance(value, (dict, list)):
                if value:
                    return True
                continue
            if value is not None:
                return True
        return False


class TradeSafetyPolicyConfig(_GuardrailSection):
    def __init__(self) -> None:
        self.max_volume: Optional[float] = None
        self.require_stop_loss = False
        self.max_deviation: Optional[int] = None
        self.reduce_only = False


class AccountRiskLimitsConfig(_GuardrailSection):
    def __init__(self) -> None:
        self.min_margin_level_pct: Optional[float] = None
        self.max_floating_loss: Optional[float] = None
        self.max_total_exposure_lots: Optional[float] = None


class WalletRiskLimitsConfig(_GuardrailSection):
    def __init__(self) -> None:
        self.max_risk_pct_of_equity: Optional[float] = None
        self.max_risk_pct_of_balance: Optional[float] = None
        self.max_risk_pct_of_free_margin: Optional[float] = None


class TradeGuardrailsRuntimeConfig(_GuardrailSection):
    """Mutable env-backed trade guardrail configuration."""

    def __init__(self) -> None:
        self._reload_lock = threading.Lock()
        self._policy_version = 0
        self.enabled = False
        self.ignore_on_demo = False
        self.trading_enabled = True
        self.allowed_symbols: list[str] = []
        self.blocked_symbols: list[str] = []
        self.max_volume: Optional[float] = None
        self.max_volume_by_symbol: dict[str, float] = {}
        self.safety_policy = TradeSafetyPolicyConfig()
        self.account_risk_limits = AccountRiskLimitsConfig()
        self.wallet_risk_limits = WalletRiskLimitsConfig()
        self.reload_from_env()

    def is_enabled(self) -> bool:
        return bool(
            self.enabled
            or not self.trading_enabled
            or self.allowed_symbols
            or self.blocked_symbols
            or self.max_volume is not None
            or self.max_volume_by_symbol
            or self.safety_policy.has_configured_values()
            or self.account_risk_limits.has_configured_values()
            or self.wallet_risk_limits.has_configured_values()
        )

    def reload_from_env(self) -> None:
        # Atomic field swap under lock so concurrent evaluations never observe
        # a partially-updated multi-field policy.
        with self._reload_lock:
            self.enabled = _env_strict_bool(
                "MTDATA_TRADE_GUARDRAILS_ENABLED", default=False
            )
            self.ignore_on_demo = _env_strict_bool(
                "MTDATA_TRADE_GUARDRAILS_IGNORE_ON_DEMO",
                default=False,
            )
            self.trading_enabled = _env_strict_bool(
                "MTDATA_TRADING_ENABLED", default=True
            )
            self.allowed_symbols = [
                symbol.upper() for symbol in get_csv_env("MTDATA_TRADE_ALLOWED_SYMBOLS")
            ]
            self.blocked_symbols = [
                symbol.upper() for symbol in get_csv_env("MTDATA_TRADE_BLOCKED_SYMBOLS")
            ]
            self.max_volume = _env_optional_float("MTDATA_TRADE_MAX_VOLUME")
            self.max_volume_by_symbol = _env_symbol_float_map(
                "MTDATA_TRADE_MAX_VOLUME_BY_SYMBOL"
            )

            self.safety_policy.max_volume = _env_optional_float(
                "MTDATA_TRADE_SAFETY_MAX_VOLUME"
            )
            self.safety_policy.require_stop_loss = _env_strict_bool(
                "MTDATA_TRADE_SAFETY_REQUIRE_STOP_LOSS",
                default=False,
            )
            self.safety_policy.max_deviation = _env_strict_optional_int(
                "MTDATA_TRADE_SAFETY_MAX_DEVIATION"
            )
            self.safety_policy.reduce_only = _env_strict_bool(
                "MTDATA_TRADE_SAFETY_REDUCE_ONLY",
                default=False,
            )

            self.account_risk_limits.min_margin_level_pct = _env_optional_float(
                "MTDATA_TRADE_MIN_MARGIN_LEVEL_PCT"
            )
            self.account_risk_limits.max_floating_loss = _env_optional_float(
                "MTDATA_TRADE_MAX_FLOATING_LOSS"
            )
            self.account_risk_limits.max_total_exposure_lots = _env_optional_float(
                "MTDATA_TRADE_MAX_TOTAL_EXPOSURE_LOTS"
            )

            self.wallet_risk_limits.max_risk_pct_of_equity = _env_optional_float(
                "MTDATA_TRADE_MAX_RISK_PCT_OF_EQUITY"
            )
            self.wallet_risk_limits.max_risk_pct_of_balance = _env_optional_float(
                "MTDATA_TRADE_MAX_RISK_PCT_OF_BALANCE"
            )
            self.wallet_risk_limits.max_risk_pct_of_free_margin = _env_optional_float(
                "MTDATA_TRADE_MAX_RISK_PCT_OF_FREE_MARGIN"
            )
            self._policy_version += 1


def load_environment(*, force: bool = False) -> bool:
    """Load environment variables from `.env` once for application entrypoints."""
    global _ENV_LOADED
    with _ENV_LOAD_LOCK:
        if _ENV_LOADED and not force:
            return False

        loaded = False
        attempted = False
        try:
            from dotenv import find_dotenv, load_dotenv  # type: ignore

            env_path = find_dotenv()
            if env_path:
                loaded = bool(load_dotenv(env_path, override=force))
            else:
                loaded = bool(load_dotenv(override=force))
            attempted = True
        except Exception:
            loaded = False

        if attempted:
            _ENV_LOADED = True
        config_obj = globals().get("mt5_config")
        if config_obj is not None:
            try:
                config_obj.reload_from_env(warn_if_timezone_missing=True)
            except Exception as exc:
                _LOGGER.warning("Failed to reload MT5 configuration from environment: %s", exc)
        embeddings_obj = globals().get("news_embeddings_config")
        if embeddings_obj is not None:
            try:
                embeddings_obj.reload_from_env()
            except Exception as exc:
                _LOGGER.warning("Failed to reload news embeddings configuration from environment: %s", exc)
        guardrails_obj = globals().get("trade_guardrails_config")
        if guardrails_obj is not None:
            try:
                guardrails_obj.reload_from_env()
            except Exception as exc:
                _LOGGER.warning("Failed to reload trade guardrails configuration from environment: %s", exc)
        options_obj = globals().get("options_data_config")
        if options_obj is not None:
            try:
                options_obj.reload_from_env()
            except Exception as exc:
                _LOGGER.warning("Failed to reload options data configuration from environment: %s", exc)
        return loaded

class MT5Config:
    """MetaTrader5 connection configuration."""

    def __init__(self, *, warn_if_timezone_missing: bool = True):
        self.login: Optional[str] = None
        self._login_value: Optional[int] = None
        self.password: Optional[str] = None
        self.server: Optional[str] = None
        self.timeout = 30
        self.server_tz_name: Optional[str] = None
        self.client_tz_name: Optional[str] = None
        self.time_offset_minutes = 0
        self.broker_time_check_enabled = False
        self.broker_time_check_ttl_seconds = 60
        self.order_magic = 234000
        self.reload_from_env(warn_if_timezone_missing=warn_if_timezone_missing)

    def reload_from_env(self, *, warn_if_timezone_missing: bool = True) -> None:
        self.login = os.getenv("MT5_LOGIN")
        self._login_value = _env_strict_optional_int("MT5_LOGIN")
        self.password = os.getenv("MT5_PASSWORD")
        self.server = os.getenv("MT5_SERVER")
        timeout_raw = os.getenv("MT5_TIMEOUT")
        timeout = get_int_env("MT5_TIMEOUT", 30)
        if timeout <= 0 or timeout > 3600:
            _LOGGER.warning(
                "MT5_TIMEOUT=%r must be between 1 and 3600 seconds; using default 30.",
                timeout_raw,
            )
            timeout = 30
        self.timeout = timeout
        self.server_tz_name = os.getenv("MT5_SERVER_TZ")  # e.g., "Europe/Lisbon"
        self.client_tz_name = os.getenv("CLIENT_TZ") or os.getenv("MT5_CLIENT_TZ")  # e.g., "America/New_York"
        self.time_offset_minutes = get_int_env("MT5_TIME_OFFSET_MINUTES", 0)
        self.broker_time_check_enabled = get_bool_env("MTDATA_BROKER_TIME_CHECK", default=False)
        ttl_raw = os.getenv("MTDATA_BROKER_TIME_CHECK_TTL_SECONDS")
        ttl_seconds = get_int_env("MTDATA_BROKER_TIME_CHECK_TTL_SECONDS", 60)
        if ttl_seconds < 0:
            _LOGGER.warning(
                "MTDATA_BROKER_TIME_CHECK_TTL_SECONDS=%r is negative; clamping to 0.",
                ttl_raw,
            )
        self.broker_time_check_ttl_seconds = max(0, ttl_seconds)
        magic_raw = os.getenv("MTDATA_ORDER_MAGIC")
        magic_value = get_int_env("MTDATA_ORDER_MAGIC", 234000)
        if magic_value < 0 or magic_value > (1 << 64) - 1:
            _LOGGER.warning(
                "MTDATA_ORDER_MAGIC=%r is outside the MT5 ulong range; using default 234000.",
                magic_raw,
            )
            magic_value = 234000
        self.order_magic = int(magic_value)
    def get_login(self) -> Optional[int]:
        """Get login as integer if available"""
        return self._login_value
    
    def get_password(self) -> Optional[str]:
        """Get password"""
        return self.password
    
    def get_server(self) -> Optional[str]:
        """Get server name"""
        return self.server
    
    def has_credentials(self) -> bool:
        """Check if all credentials are available"""
        return self.credential_state() == "complete"

    def credential_state(self) -> str:
        """Return ``none``, ``complete``, or ``partial`` credential state."""
        configured = (
            self.get_login() is not None,
            bool(self.password),
            bool(str(self.server or "").strip()),
        )
        if all(configured):
            return "complete"
        if any(configured):
            return "partial"
        return "none"

    def get_timeout_milliseconds(self) -> int:
        """Return the documented seconds setting in MT5's millisecond unit."""
        return int(self.timeout) * 1000

    def get_time_offset_seconds(self, at_time: Optional[datetime] = None) -> int:
        """Return the broker session/calendar offset relative to UTC in seconds.

        Positive value means MT5 server time is ahead of UTC (UTC+X).
        The MT5 adapter also uses this offset at its boundary only when it has
        detected a terminal that exposes server-clock rather than UTC epochs.

        When ``at_time`` is provided and ``MT5_SERVER_TZ`` is configured, resolve the
        offset for that specific instant so historical DST transitions are respected.
        Naive datetimes are interpreted as UTC instants.
        """
        # 1. Prefer explicit offset in minutes (if set)
        if self.time_offset_minutes != 0:
            global _WARNED_STATIC_OFFSET_OVERRIDE
            if self.server_tz_name and not _WARNED_STATIC_OFFSET_OVERRIDE:
                _WARNED_STATIC_OFFSET_OVERRIDE = True
                _LOGGER.warning(
                    "MT5_TIME_OFFSET_MINUTES overrides MT5_SERVER_TZ; static offsets do not adjust for DST. "
                    "Prefer MT5_SERVER_TZ alone for DST-aware session/calendar calculations."
                )
            return int(self.time_offset_minutes) * 60
            
        # 2. Derive from MT5_SERVER_TZ if available
        tz = self.get_server_tz()
        if tz is not None:
            try:
                reference_time = at_time
                if reference_time is None:
                    reference_time = datetime.now(timezone.utc)
                elif reference_time.tzinfo is None or reference_time.utcoffset() is None:
                    reference_time = reference_time.replace(tzinfo=timezone.utc)
                else:
                    reference_time = reference_time.astimezone(timezone.utc)
                offset = reference_time.astimezone(tz).utcoffset()
                if offset is None:
                    return 0
                return int(offset.total_seconds())
            except Exception:
                pass

        return 0

    def get_server_tz(self):
        """Return a ZoneInfo timezone for the MT5 server, if configured."""
        return _zoneinfo_from_name(self.server_tz_name)

    def get_client_tz(self):
        """Return the client timezone from config or local autodetection."""
        if self.client_tz_name:
            return _zoneinfo_from_name(self.client_tz_name)
        return _detect_local_client_tz()


class NewsEmbeddingsConfig:
    """Optional embedding reranking configuration for unified news."""

    def __init__(self) -> None:
        self.enabled = False
        self.model_name = "Qwen/Qwen3-Embedding-0.6B"
        self.top_n = 8
        self.weight = 1.0
        self.truncate_dim: Optional[int] = None
        self.cache_size = 256
        self.hf_token_env_var = "HF_TOKEN"
        self.reload_from_env()

    def reload_from_env(self) -> None:
        self.enabled = get_bool_env("MTDATA_NEWS_EMBEDDINGS_ENABLED", default=False)
        self.model_name = (
            os.getenv("MTDATA_NEWS_EMBEDDINGS_MODEL", "Qwen/Qwen3-Embedding-0.6B").strip()
            or "Qwen/Qwen3-Embedding-0.6B"
        )
        self.top_n = max(1, get_int_env("MTDATA_NEWS_EMBEDDINGS_TOP_N", 8))
        self.weight = max(0.0, get_float_env("MTDATA_NEWS_EMBEDDINGS_WEIGHT", 1.0))
        self.truncate_dim = _env_optional_int("MTDATA_NEWS_EMBEDDINGS_TRUNCATE_DIM")
        if self.truncate_dim is not None and self.truncate_dim <= 0:
            _LOGGER.warning(
                "MTDATA_NEWS_EMBEDDINGS_TRUNCATE_DIM=%r is not positive; disabling truncation.",
                self.truncate_dim,
            )
            self.truncate_dim = None
        self.cache_size = max(0, get_int_env("MTDATA_NEWS_EMBEDDINGS_CACHE_SIZE", 256))
        self.hf_token_env_var = (
            os.getenv("MTDATA_NEWS_EMBEDDINGS_HF_TOKEN_ENV_VAR", "HF_TOKEN").strip() or "HF_TOKEN"
        )


class OptionsDataConfig:
    """Options data provider configuration."""

    def __init__(self) -> None:
        self.provider = "yahoo"
        self.api_key: Optional[str] = None
        self.base_url: Optional[str] = None
        self.reload_from_env()

    def reload_from_env(self) -> None:
        provider = os.getenv("MTDATA_OPTIONS_PROVIDER", "yahoo").strip().lower()
        if provider not in {"auto", "yahoo", "tradier"}:
            _LOGGER.warning(
                "Invalid MTDATA_OPTIONS_PROVIDER=%r; using yahoo.",
                provider,
            )
        api_key = (
            os.getenv("MTDATA_OPTIONS_API_KEY")
            or os.getenv("TRADIER_TOKEN")
            or os.getenv("TRADIER_API_KEY")
        )
        self.provider = provider
        self.api_key = api_key.strip() if isinstance(api_key, str) and api_key.strip() else None
        self.base_url = (
            os.getenv("MTDATA_OPTIONS_BASE_URL", "https://api.tradier.com/v1").strip()
            or "https://api.tradier.com/v1"
        )


# Global configuration instance. Entry points call `load_environment()` explicitly.
mt5_config = MT5Config(warn_if_timezone_missing=False)
news_embeddings_config = NewsEmbeddingsConfig()
trade_guardrails_config = TradeGuardrailsRuntimeConfig()
options_data_config = OptionsDataConfig()
