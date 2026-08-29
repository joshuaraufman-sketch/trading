"""
The single source of truth for turning target weights into orders.

Both the backtest and the live runner call ``compute_rebalance_orders``.
That is the entire point: parity between research and production is a
property of the architecture, not something asserted by a test that
drifts the moment either side changes. The sweep grid and
research_rules.yaml both drifted exactly that way before.

Why this is needed at all. The existing risk layer was built for
discrete, stop-managed entries: ``check_order_plan`` requires a buy
side, a positive stop price, and caps positions at 25% of equity. A
volatility-targeted weight schedule is a continuous exposure that
averages 72% and reduces itself by SELLING. Not one of its orders could
pass those rules. The strategy that survived validation cannot currently
be traded, and patching the caps would leave two divergent descriptions
of the same system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ExecutionPolicy:
    """
    Constraints applied identically in backtest and live.

    Anything that changes what gets traded belongs here. If a constraint
    lives only in the live path, the backtest is measuring a different
    strategy than the one that will run.
    """

    max_position_pct: float = 1.00
    max_gross_exposure: float = 1.00
    min_order_notional: float = 100.0
    rebalance_band: float = 0.05
    allow_short: bool = False
    allow_fractional_shares: bool = True
    max_orders_per_session: int = 10

    def __post_init__(self) -> None:
        if not 0 < self.max_position_pct <= 10:
            raise ValueError("max_position_pct must be in (0, 10]")

        if not 0 < self.max_gross_exposure <= 10:
            raise ValueError("max_gross_exposure must be in (0, 10]")

        if self.min_order_notional < 0:
            raise ValueError("min_order_notional cannot be negative")

        if self.rebalance_band < 0:
            raise ValueError("rebalance_band cannot be negative")

        if self.max_orders_per_session < 1:
            raise ValueError("max_orders_per_session must be at least 1")

    @property
    def uses_leverage(self) -> bool:
        return self.max_gross_exposure > 1.0

    @classmethod
    def from_yaml(cls, path: Path) -> "ExecutionPolicy":
        with Path(path).open("r", encoding="utf-8") as file:
            payload = yaml.safe_load(file) or {}

        known = {f for f in cls.__dataclass_fields__}
        unknown = set(payload) - known

        if unknown:
            raise ValueError(
                f"unknown execution policy keys: {sorted(unknown)}"
            )

        return cls(**payload)


@dataclass(frozen=True)
class RebalanceOrder:
    symbol: str
    side: str
    quantity: float
    reference_price: float
    current_weight: float
    target_weight: float

    @property
    def notional(self) -> float:
        return abs(self.quantity) * self.reference_price


@dataclass
class RebalancePlan:
    orders: list[RebalanceOrder] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    applied_weights: dict[str, float] = field(default_factory=dict)

    @property
    def turnover(self) -> float:
        return sum(order.notional for order in self.orders)


def clamp_weights(
    target_weights: dict[str, float],
    policy: ExecutionPolicy,
) -> dict[str, float]:
    """
    Apply position and gross-exposure caps to a set of target weights.

    Per-name caps are applied first, then the whole book is scaled down
    if it still breaches gross exposure. Scaling preserves the relative
    shape of the allocation, which matters: truncating instead would
    silently change which instruments dominate.
    """

    clamped: dict[str, float] = {}

    for symbol, weight in target_weights.items():
        value = float(weight)

        if value < 0 and not policy.allow_short:
            value = 0.0

        limit = policy.max_position_pct
        value = max(min(value, limit), -limit if policy.allow_short else 0.0)

        clamped[symbol] = value

    gross = sum(abs(w) for w in clamped.values())

    if gross > policy.max_gross_exposure and gross > 0:
        scale = policy.max_gross_exposure / gross
        clamped = {s: w * scale for s, w in clamped.items()}

    return clamped


def compute_rebalance_orders(
    *,
    target_weights: dict[str, float],
    current_quantities: dict[str, float],
    prices: dict[str, float],
    equity: float,
    policy: ExecutionPolicy,
) -> RebalancePlan:
    """
    Turn target weights into concrete orders.

    Called by the backtest once per session and by the live runner once
    per day, with identical semantics. ``current_quantities`` is shares
    held, so the live path can pass broker positions directly and the
    backtest can pass its simulated book.

    Orders below ``min_order_notional`` or inside ``rebalance_band`` are
    skipped with a recorded reason rather than dropped silently -- an
    unexplained gap between intended and actual exposure is how a live
    system diverges from its backtest without anyone noticing.

    When more orders qualify than ``max_orders_per_session`` allows, the
    largest by notional are kept. Truncating in arbitrary dictionary
    order would make the live result depend on iteration order, which is
    not reproducible in a backtest.
    """

    if equity <= 0:
        raise ValueError("equity must be greater than zero")

    plan = RebalancePlan()

    clamped = clamp_weights(target_weights, policy)
    plan.applied_weights = dict(clamped)

    candidates: list[RebalanceOrder] = []

    for symbol in sorted(set(clamped) | set(current_quantities)):
        target = clamped.get(symbol, 0.0)
        held = float(current_quantities.get(symbol, 0.0))
        price = float(prices.get(symbol, 0.0))

        if price <= 0:
            if held != 0 or target != 0:
                plan.skipped[symbol] = "no valid price"
            continue

        current_weight = held * price / equity
        drift = target - current_weight

        if abs(drift) <= policy.rebalance_band:
            plan.skipped[symbol] = (
                f"within rebalance band ({abs(drift):.4f} "
                f"<= {policy.rebalance_band:.4f})"
            )
            continue

        quantity = drift * equity / price

        if not policy.allow_fractional_shares:
            quantity = math.trunc(quantity)

            if quantity == 0:
                plan.skipped[symbol] = "rounds to zero whole shares"
                continue

        notional = abs(quantity) * price

        if notional < policy.min_order_notional:
            plan.skipped[symbol] = (
                f"below minimum notional (${notional:,.2f} "
                f"< ${policy.min_order_notional:,.2f})"
            )
            continue

        candidates.append(
            RebalanceOrder(
                symbol=symbol,
                side="buy" if quantity > 0 else "sell",
                quantity=abs(quantity),
                reference_price=price,
                current_weight=current_weight,
                target_weight=target,
            )
        )

    candidates.sort(key=lambda o: o.notional, reverse=True)

    plan.orders = candidates[:policy.max_orders_per_session]

    for order in candidates[policy.max_orders_per_session:]:
        plan.skipped[order.symbol] = (
            f"exceeded max_orders_per_session "
            f"({policy.max_orders_per_session})"
        )

    return plan
