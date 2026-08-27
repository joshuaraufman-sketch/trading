from __future__ import annotations

import math


def calculate_position_size(
    account_equity: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
) -> int:
    """
    Calculate share quantity from fixed account risk.

    Example:
        $100,000 account
        0.5% risk
        $100 entry
        $98 stop

        account risk = $500
        risk per share = $2
        position size = 250 shares
    """

    if account_equity <= 0:
        raise ValueError("account_equity must be greater than zero")

    if not 0 < risk_pct <= 1:
        raise ValueError("risk_pct must be between 0 and 1")

    if entry_price <= 0 or stop_price <= 0:
        raise ValueError("prices must be greater than zero")

    risk_per_share = entry_price - stop_price

    if risk_per_share <= 0:
        raise ValueError(
            "stop_price must be below entry_price for a long trade"
        )

    account_risk = account_equity * risk_pct

    quantity = math.floor(account_risk / risk_per_share)

    return max(quantity, 0)