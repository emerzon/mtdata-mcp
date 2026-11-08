"""Transport-independent classification and warning for live trade mutations."""

LIVE_TRADE_MUTATION_TOOLS = frozenset({"trade_place", "trade_modify", "trade_close"})
LIVE_TRADE_MUTATION_WARNING = (
    "LIVE ORDER WARNING: this command can send real MT5 trade requests when "
    "--dry-run false. Preview mode is the default."
)
