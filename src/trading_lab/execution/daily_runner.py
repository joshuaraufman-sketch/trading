"""
Post-close daily run: decide what to trade tomorrow.

Runs after the close, computes the target weight from THAT session's
close, and produces orders for the next open. This is the faithful
implementation of what the backtest models: a weight knowable at the
close of t-1 is held into session t. Running pre-open instead would act
on a weight one session staler than the research assumed.

All decision logic lives here as pure functions over an account state
and a bar frame, so the ordering guarantees and refusal conditions are
testable without credentials, a network call, or a live market.

The runner REFUSES rather than guesses. Three conditions stop it:

  preflight failure   the account is not what was expected
  stale bars          the latest session is not the one being traded
  duplicate run       this session was already processed

Stale bars deserve particular attention. Acting on yesterday's close
while believing it is today's shifts the entire strategy by one session
and would be invisible in the logs -- the orders look perfectly
reasonable, they are just answering the wrong question.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from trading_lab.backtest.equity import normalize_session_dates
from trading_lab.execution.alpaca_account import AccountState
from trading_lab.execution.preflight import (
    AccountExpectations,
    PreflightResult,
    run_preflight,
)
from trading_lab.execution.rebalance import (
    ExecutionPolicy,
    RebalancePlan,
    compute_rebalance_orders,
)
from trading_lab.strategies.volatility_target import (
    volatility_target_weights,
)


@dataclass(frozen=True)
class StrategyConfig:
    """Parameters of the deployed strategy. Must match research."""

    symbol: str = "SPY"
    target_volatility: float = 0.10
    lookback: int = 20
    max_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.target_volatility <= 0:
            raise ValueError("target_volatility must be positive")

        if self.lookback < 2:
            raise ValueError("lookback must be at least 2")


@dataclass
class DailyRunPlan:
    session: date | None = None
    preflight: PreflightResult | None = None
    target_weights: dict[str, float] = field(default_factory=dict)
    rebalance: RebalancePlan | None = None
    blocked_reason: str | None = None
    diagnostics: dict = field(default_factory=dict)

    @property
    def should_trade(self) -> bool:
        return (
            self.blocked_reason is None
            and self.rebalance is not None
            and bool(self.rebalance.orders)
        )

    @property
    def blocked(self) -> bool:
        return self.blocked_reason is not None


def latest_session(bars: pd.DataFrame) -> date:
    """Most recent session present in a bar frame."""

    sessions = normalize_session_dates(bars["timestamp"])

    if sessions.empty:
        raise ValueError("bar frame contains no sessions")

    return sessions.max().date()


def build_daily_plan(
    *,
    account_state: AccountState,
    bars: pd.DataFrame,
    expected_session: date,
    expectations: AccountExpectations,
    policy: ExecutionPolicy,
    strategy: StrategyConfig,
    completed_sessions: set[date] | None = None,
) -> DailyRunPlan:
    """
    Decide what to do, without doing any of it.

    ``expected_session`` is the session whose close should drive the
    decision -- normally today, supplied by the caller from the market
    calendar. Passing it in rather than inferring it keeps this function
    free of clock and calendar dependencies, and makes the stale-bar
    check meaningful: the caller asserts which session it believes it is
    trading, and this refuses if the data disagrees.
    """

    plan = DailyRunPlan(session=expected_session)
    completed = completed_sessions or set()

    plan.preflight = run_preflight(account_state, expectations)

    if not plan.preflight.passed:
        failures = "; ".join(
            c.detail for c in plan.preflight.failures if c.blocking
        )
        plan.blocked_reason = f"preflight failed: {failures}"
        return plan

    if expected_session in completed:
        plan.blocked_reason = (
            f"session {expected_session} already processed; "
            f"refusing to submit duplicate orders"
        )
        return plan

    frame = bars[bars["symbol"] == strategy.symbol].copy()

    if frame.empty:
        plan.blocked_reason = (
            f"no bars supplied for {strategy.symbol}"
        )
        return plan

    observed = latest_session(frame)
    plan.diagnostics["latest_bar_session"] = observed

    if observed != expected_session:
        plan.blocked_reason = (
            f"stale or mismatched data: latest bar is {observed}, "
            f"expected {expected_session}. Refusing to act on the "
            f"wrong session's close."
        )
        return plan

    frame["session"] = normalize_session_dates(frame["timestamp"])
    closes = (
        frame.groupby("session")["close"].last().sort_index().astype(float)
    )

    if len(closes) < strategy.lookback + 1:
        plan.blocked_reason = (
            f"insufficient history: {len(closes)} sessions, need at "
            f"least {strategy.lookback + 1}"
        )
        return plan

    returns = closes.pct_change().fillna(0.0)

    weights = volatility_target_weights(
        returns,
        target_volatility=strategy.target_volatility,
        lookback=strategy.lookback,
        max_weight=strategy.max_weight,
    )

    target = float(weights.iloc[-1])
    reference_price = float(closes.iloc[-1])

    plan.target_weights = {strategy.symbol: target}
    plan.diagnostics.update(
        {
            "reference_price": reference_price,
            "realized_volatility": (
                strategy.target_volatility / target
                if target > 0
                else None
            ),
            "sessions_of_history": int(len(closes)),
        }
    )

    quantities = {
        position["symbol"]: float(position["qty"])
        for position in account_state.positions
    }

    plan.rebalance = compute_rebalance_orders(
        target_weights=plan.target_weights,
        current_quantities=quantities,
        prices={strategy.symbol: reference_price},
        equity=account_state.equity,
        policy=policy,
    )

    return plan


def summarize_plan(plan: DailyRunPlan) -> str:
    """Human-readable summary for the evening review."""

    lines: list[str] = []

    lines.append(f"session            {plan.session}")

    if plan.blocked:
        lines.append(f"status             BLOCKED")
        lines.append(f"reason             {plan.blocked_reason}")
        return "\n".join(lines)

    for symbol, weight in plan.target_weights.items():
        lines.append(f"target weight      {symbol} {weight:.2%}")

    price = plan.diagnostics.get("reference_price")

    if price:
        lines.append(f"reference close    ${price:,.2f}")

    vol = plan.diagnostics.get("realized_volatility")

    if vol:
        lines.append(f"implied vol        {vol:.2%}")

    if plan.rebalance is None or not plan.rebalance.orders:
        lines.append("status             NO ACTION (within band)")

        if plan.rebalance:
            for symbol, reason in plan.rebalance.skipped.items():
                lines.append(f"  {symbol:<8}{reason}")

        return "\n".join(lines)

    lines.append(f"status             {len(plan.rebalance.orders)} order(s)")

    for order in plan.rebalance.orders:
        lines.append(
            f"  {order.side.upper():<5}{order.symbol:<8}"
            f"{order.quantity:>12,.4f} sh  ${order.notional:>12,.2f}  "
            f"{order.current_weight:.2%} -> {order.target_weight:.2%}"
        )

    return "\n".join(lines)
