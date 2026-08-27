from __future__ import annotations

from trading_lab.execution.models import (
    OrderPlan,
    RiskDecision,
)


def check_order_plan(
    order: OrderPlan,
    *,
    account_equity: float,
    available_cash: float,
    maximum_position_pct: float = 0.25,
    maximum_risk_pct: float = 0.005,
) -> RiskDecision:

    if account_equity <= 0:
        return RiskDecision(
            approved=False,
            reason="Invalid account equity.",
        )

    if available_cash < 0:
        return RiskDecision(
            approved=False,
            reason="Invalid available cash.",
        )

    if order.side != "buy":
        return RiskDecision(
            approved=False,
            reason="Stage 4 currently allows long entries only.",
        )

    if order.quantity <= 0:
        return RiskDecision(
            approved=False,
            reason="Quantity must be positive.",
        )

    if order.reference_price <= 0:
        return RiskDecision(
            approved=False,
            reason="Reference price must be positive.",
        )

    if order.stop_price <= 0:
        return RiskDecision(
            approved=False,
            reason="Stop price must be positive.",
        )

    if order.stop_price >= order.reference_price:
        return RiskDecision(
            approved=False,
            reason="Long-trade stop must be below entry price.",
        )

    maximum_position_value = (
        account_equity * maximum_position_pct
    )

    if order.estimated_position_value > maximum_position_value:
        return RiskDecision(
            approved=False,
            reason="Position exceeds maximum position size.",
        )

    if order.estimated_position_value > available_cash:
        return RiskDecision(
            approved=False,
            reason="Insufficient available cash.",
        )

    order_risk = (
        order.risk_per_share
        * order.quantity
    )

    maximum_risk = (
        account_equity * maximum_risk_pct
    )

    if order_risk > maximum_risk:
        return RiskDecision(
            approved=False,
            reason="Trade risk exceeds account risk limit.",
        )

    return RiskDecision(
        approved=True,
        reason="Order plan passed deterministic risk checks.",
    )