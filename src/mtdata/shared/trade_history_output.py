"""Presentation column labels for trade-history rows."""

from typing import Any, List


def _trade_history_humanized_key(key: str) -> str:
    overrides = {
        "sl": "SL",
        "tp": "TP",
        "time": "Time",
        "fill_time": "Fill Time",
        "placed_time": "Placed Time",
        "done_time": "Done Time",
        "time_setup": "Setup Time",
        "time_done": "Done Time",
        "time_msc": "Time Msc",
        "ticket": "Ticket",
        "deal_ticket": "Deal Ticket",
        "order": "Order",
        "order_ticket": "Order Ticket",
        "deal": "Deal",
        "position_id": "Position ID",
        "position_by_id": "Position By ID",
        "symbol": "Symbol",
        "type": "Type",
        "type_code": "Type Code",
        "position_side": "Position Side",
        "entry": "Entry",
        "entry_code": "Entry Code",
        "deal_effect": "Deal Effect",
        "reason": "Reason",
        "reason_code": "Reason Code",
        "state": "State",
        "state_code": "State Code",
        "volume": "Volume",
        "volume_initial": "Initial Volume",
        "volume_current": "Current Volume",
        "price": "Price",
        "price_open": "Open Price",
        "price_current": "Current Price",
        "profit": "Profit",
        "commission": "Commission",
        "swap": "Swap",
        "fee": "Fee",
        "net_pnl": "Net P&L",
        "profit_basis": "Profit Basis",
        "comment": "Comments",
        "magic": "Magic",
        "deal_magic": "Deal Magic",
        "attributed_magic": "Attributed Magic",
        "attribution_method": "Attribution Method",
        "exit_trigger": "Exit Trigger",
        "exit_trigger_price": "Exit Trigger Price",
        "exit_trigger_source": "Exit Trigger Source",
    }
    return overrides.get(key, key.replace("_", " ").title())


def humanize_trade_history_items(items: List[Any]) -> List[Any]:
    """Return rows with the user-facing trade-history column labels."""
    styled: List[Any] = []
    for item in items:
        if not isinstance(item, dict):
            styled.append(item)
            continue
        styled.append(
            {_trade_history_humanized_key(str(key)): value for key, value in item.items()}
        )
    return styled
