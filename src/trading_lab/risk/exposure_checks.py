from __future__ import annotations

from trading_lab.execution.alpaca_account import AccountState
from trading_lab.execution.models import RiskDecision


def check_existing_exposure(
    *,
    symbol: str,
    account: AccountState,
) -> RiskDecision:
    held_symbols = {
        position["symbol"]
        for position in account.positions
    }

    open_order_symbols = {
        order["symbol"]
        for order in account.open_orders
    }

    if symbol in held_symbols:
        return RiskDecision(
            approved=False,
            reason=f"Existing position already held in {symbol}.",
        )

    if symbol in open_order_symbols:
        return RiskDecision(
            approved=False,
            reason=f"Open order already exists for {symbol}.",
        )

    return RiskDecision(
        approved=True,
        reason="No existing position or open order for symbol.",
    )