"""
Weight-based backtesting.

The existing runner expresses one shape of strategy: a signal fires, a
position is entered, a stop or a forced exit closes it. Many strategies
are not that shape. Volatility targeting is a continuous exposure level,
and every cross-sectional strategy -- rank a universe, assign weights,
rebalance -- is too. Neither can be expressed as discrete signalled
trades without distorting them.

This module runs a target-weight schedule instead: you supply what
fraction of equity should sit in each instrument on each session, and it
produces the equity curve after drift, rebalancing and costs.

Two properties are enforced here rather than left to callers, because
both are easy to get wrong in ways that silently manufacture edge:

1. Target weights are lagged one session. A weight computed from data
   through the close of t-1 earns the return from t-1 to t. Callers
   pass unlagged weights; this module does the shifting.
2. Turnover is charged. A strategy that rebalances daily and pays
   nothing for it is not a strategy, it is an accounting error.
"""

from __future__ import annotations

import pandas as pd

from trading_lab.execution.rebalance import (
    ExecutionPolicy,
    compute_rebalance_orders,
)


TRADING_DAYS_PER_YEAR = 252


def _daily_rate(annual_rate: float) -> float:
    return (1.0 + annual_rate) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0


def run_weight_backtest(
    returns: pd.DataFrame | pd.Series,
    target_weights: pd.DataFrame | pd.Series,
    *,
    starting_equity: float,
    cost_bps: float = 5.0,
    rebalance_band: float = 0.0,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Run a target-weight schedule and return an equity curve.

    ``returns`` and ``target_weights`` are session-indexed. Weights are
    expressed as a fraction of total equity; anything left over sits in
    cash earning ``risk_free_rate``. A total weight above 1.0 is
    leverage, and the caller is responsible for deciding whether that is
    permitted.

    ``cost_bps`` is charged on turnover, measured as the absolute change
    in weight actually traded. Round-trip costs should be reflected here
    -- a 5 bps figure means moving 100% of equity in or out costs 5 bps.

    ``rebalance_band`` suppresses trades smaller than the band, in weight
    terms. Real implementations use one because rebalancing to the exact
    target every session pays costs for trivial adjustments. Zero
    reproduces daily full rebalancing.

    The returned frame is compatible with ``calculate_performance`` and
    the benchmark comparison helpers.
    """

    if starting_equity <= 0:
        raise ValueError("starting_equity must be greater than zero")

    if cost_bps < 0:
        raise ValueError("cost_bps cannot be negative")

    if rebalance_band < 0:
        raise ValueError("rebalance_band cannot be negative")

    if isinstance(returns, pd.Series):
        returns = returns.to_frame(name=returns.name or "asset")

    if isinstance(target_weights, pd.Series):
        target_weights = target_weights.to_frame(
            name=target_weights.name or returns.columns[0]
        )

    missing = set(target_weights.columns) - set(returns.columns)

    if missing:
        raise ValueError(
            f"target weights supplied for instruments with no returns: "
            f"{sorted(missing)}"
        )

    sessions = returns.index

    if not sessions.is_monotonic_increasing:
        raise ValueError("returns must be sorted by session")

    if sessions.duplicated().any():
        raise ValueError("returns index contains duplicate sessions")

    weights = (
        target_weights
        .reindex(index=sessions, columns=returns.columns)
        .fillna(0.0)
        .astype(float)
    )

    # The lag that stops this from peeking. A weight derived from data
    # through the close of t-1 is what can be held into session t.
    intended = weights.shift(1).fillna(0.0)

    rf_daily = _daily_rate(risk_free_rate)
    cost_rate = cost_bps / 10_000.0

    equity = starting_equity
    held = pd.Series(0.0, index=returns.columns)

    records = []

    for session in sessions:
        target = intended.loc[session]
        change = target - held

        # Only trade legs that have drifted beyond the band.
        traded = change.where(change.abs() > rebalance_band, 0.0)
        held = held + traded

        turnover = float(traded.abs().sum())
        cost = turnover * cost_rate

        period_returns = returns.loc[session].fillna(0.0)
        invested = float(held.sum())

        gross = float((held * period_returns).sum())
        cash_return = (1.0 - invested) * rf_daily

        net_return = gross + cash_return - cost
        equity *= 1.0 + net_return

        # Positions drift with prices until the next rebalance.
        grown = held * (1.0 + period_returns)
        total = float(grown.sum()) + (1.0 - invested) * (1.0 + rf_daily)

        if total > 0:
            held = grown / total
        else:
            held = grown * 0.0

        records.append(
            {
                "session": session,
                "equity": equity,
                "daily_return": net_return,
                "exposure_pct": invested,
                "turnover": turnover,
                "cost": cost,
                "open_positions": int((target.abs() > 0).sum()),
            }
        )

    curve = pd.DataFrame(records).set_index("session")

    curve["running_peak"] = curve["equity"].cummax()
    curve["drawdown_pct"] = (
        curve["equity"] - curve["running_peak"]
    ) / curve["running_peak"]

    return curve


def turnover_summary(curve: pd.DataFrame) -> dict:
    """
    Annualized turnover and the total cost drag it produced.

    Worth reporting alongside any weight-based result: a strategy whose
    edge is smaller than its cost drag has no edge, and that comparison
    is invisible from the equity curve alone.
    """

    if "turnover" not in curve.columns:
        raise ValueError("curve must contain a 'turnover' column")

    sessions = len(curve)

    if sessions == 0:
        raise ValueError("curve is empty")

    total_turnover = float(curve["turnover"].sum())
    total_cost = float(curve["cost"].sum())

    years = sessions / TRADING_DAYS_PER_YEAR

    return {
        "total_turnover": total_turnover,
        "annualized_turnover": total_turnover / max(years, 1e-9),
        "total_cost_fraction": total_cost,
        "annualized_cost_drag": total_cost / max(years, 1e-9),
        "sessions_traded": int((curve["turnover"] > 0).sum()),
        "sessions": sessions,
    }

def run_policy_backtest(
    returns: pd.DataFrame | pd.Series,
    prices: pd.DataFrame | pd.Series,
    target_weights: pd.DataFrame | pd.Series,
    *,
    starting_equity: float,
    policy: ExecutionPolicy,
    cost_bps: float = 5.0,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Backtest a weight schedule through the LIVE order path.

    Identical to ``run_weight_backtest`` except that every session's
    trades are produced by ``compute_rebalance_orders`` -- the same
    function the live runner calls. Position caps, minimum notionals,
    the rebalance band, share rounding and the per-session order cap all
    apply exactly as they will in production.

    Use this, not ``run_weight_backtest``, for any result that is meant
    to predict live behaviour. ``run_weight_backtest`` remains useful
    for pure research questions where execution frictions are not the
    subject, but its numbers are an upper bound.

    Skipped orders are counted and returned so the gap between intended
    and achieved exposure is visible rather than inferred.
    """

    if isinstance(returns, pd.Series):
        returns = returns.to_frame(name=returns.name or "asset")

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=prices.name or returns.columns[0])

    if isinstance(target_weights, pd.Series):
        target_weights = target_weights.to_frame(
            name=target_weights.name or returns.columns[0]
        )

    sessions = returns.index

    if not sessions.is_monotonic_increasing:
        raise ValueError("returns must be sorted by session")

    weights = (
        target_weights
        .reindex(index=sessions, columns=returns.columns)
        .fillna(0.0)
        .astype(float)
    )

    # Same lag as run_weight_backtest: a weight known at the close of
    # t-1 is what can be held into session t.
    intended = weights.shift(1).fillna(0.0)
    prices = prices.reindex(index=sessions, columns=returns.columns).ffill()

    rf_daily = _daily_rate(risk_free_rate)
    cost_rate = cost_bps / 10_000.0

    equity = starting_equity
    quantities = {symbol: 0.0 for symbol in returns.columns}

    records = []

    for session in sessions:
        session_prices = {
            symbol: float(prices.loc[session, symbol])
            for symbol in returns.columns
            if pd.notna(prices.loc[session, symbol])
        }

        plan = compute_rebalance_orders(
            target_weights={
                symbol: float(intended.loc[session, symbol])
                for symbol in returns.columns
            },
            current_quantities=dict(quantities),
            prices=session_prices,
            equity=equity,
            policy=policy,
        )

        turnover_notional = 0.0

        for order in plan.orders:
            signed = (
                order.quantity if order.side == "buy" else -order.quantity
            )
            quantities[order.symbol] = (
                quantities.get(order.symbol, 0.0) + signed
            )
            turnover_notional += order.notional

        cost = turnover_notional / equity * cost_rate if equity > 0 else 0.0

        invested_value = sum(
            quantities.get(symbol, 0.0) * session_prices.get(symbol, 0.0)
            for symbol in returns.columns
        )
        cash = equity - invested_value

        gross = sum(
            quantities.get(symbol, 0.0)
            * session_prices.get(symbol, 0.0)
            * float(returns.loc[session, symbol] or 0.0)
            for symbol in returns.columns
        )

        pnl = gross + cash * rf_daily - cost * equity
        previous_equity = equity
        equity += pnl

        records.append(
            {
                "session": session,
                "equity": equity,
                "daily_return": (
                    pnl / previous_equity if previous_equity > 0 else 0.0
                ),
                "exposure_pct": (
                    invested_value / previous_equity
                    if previous_equity > 0
                    else 0.0
                ),
                "turnover": (
                    turnover_notional / previous_equity
                    if previous_equity > 0
                    else 0.0
                ),
                "cost": cost,
                "orders": len(plan.orders),
                "skipped": len(plan.skipped),
                "open_positions": int(
                    sum(1 for q in quantities.values() if abs(q) > 1e-9)
                ),
            }
        )

    curve = pd.DataFrame(records).set_index("session")
    curve["running_peak"] = curve["equity"].cummax()
    curve["drawdown_pct"] = (
        curve["equity"] - curve["running_peak"]
    ) / curve["running_peak"]

    return curve

