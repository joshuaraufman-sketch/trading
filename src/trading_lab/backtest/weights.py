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
