from __future__ import annotations

from trading_lab.backtest.position_size import (
    calculate_position_size,
)
from trading_lab.execution.models import OrderPlan


def build_long_order_plan(
    *,
    symbol: str,
    signal_time,
    reference_price: float,
    account_equity: float,
    stop_loss_pct: float = 0.02,
    risk_pct: float = 0.005,
) -> OrderPlan:

    if reference_price <= 0:
        raise ValueError(
            "reference_price must be greater than zero"
        )

    stop_price = (
        reference_price
        * (1 - stop_loss_pct)
    )

    quantity = calculate_position_size(
        account_equity=account_equity,
        risk_pct=risk_pct,
        entry_price=reference_price,
        stop_price=stop_price,
    )

    return OrderPlan(
        symbol=symbol,
        side="buy",
        quantity=quantity,
        signal_time=signal_time,
        reference_price=reference_price,
        stop_price=stop_price,
        risk_per_share=(
            reference_price - stop_price
        ),
        estimated_position_value=(
            reference_price * quantity
        ),
    )