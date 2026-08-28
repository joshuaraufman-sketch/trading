from __future__ import annotations

import math

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252
DAYS_PER_YEAR = 365.25


def _daily_risk_free(
    returns: pd.Series,
    risk_free_rate: float | pd.Series,
) -> pd.Series:
    """
    Express an annual risk-free rate as a per-session rate.

    Accepts either a scalar annual rate or a session-indexed series of
    annual rates, which matters over 2017-2025 where short rates moved
    from roughly zero to roughly five percent. Leaving this at zero
    inflates Sharpe on any sample that spans the 2022-2024 rate cycle.
    """

    if isinstance(risk_free_rate, pd.Series):
        aligned = risk_free_rate.reindex(returns.index).ffill().bfill()
    else:
        aligned = pd.Series(
            float(risk_free_rate),
            index=returns.index,
        )

    return (1.0 + aligned) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0


def max_drawdown_duration_days(equity: pd.Series) -> int:
    """
    Longest stretch, in calendar days, spent below a prior equity peak.

    Depth alone understates how a drawdown is actually experienced. A
    twelve percent dip recovered in a week and a twelve percent dip that
    grinds for fourteen months are not the same risk.
    """

    if equity.empty:
        return 0

    peaks = equity.cummax()
    underwater = equity < peaks

    longest = 0
    start: pd.Timestamp | None = None

    for timestamp, is_under in underwater.items():
        if is_under and start is None:
            start = timestamp
        elif not is_under and start is not None:
            longest = max(longest, (timestamp - start).days)
            start = None

    if start is not None:
        longest = max(longest, (underwater.index[-1] - start).days)

    return int(longest)


def calculate_performance(
    curve: pd.DataFrame,
    *,
    risk_free_rate: float | pd.Series = 0.0,
) -> dict:
    """
    Compute calendar-based, risk-adjusted metrics from an equity curve.

    Expects the output of ``build_daily_equity_curve``. Every figure here
    is time-aware, which the trade-ordered metrics in ``metrics.py`` are
    not; use those for trade-level statistics such as expectancy and
    profit factor, and these for anything compared against a benchmark.
    """

    if "equity" not in curve.columns:
        raise ValueError("curve must contain an 'equity' column")

    if len(curve) < 2:
        raise ValueError(
            "curve must contain at least two sessions to measure "
            "time-based performance"
        )

    equity = curve["equity"].astype(float)
    returns = equity.pct_change().dropna()

    starting_equity = float(equity.iloc[0])
    ending_equity = float(equity.iloc[-1])

    if starting_equity <= 0:
        raise ValueError("starting equity must be greater than zero")

    elapsed_days = (equity.index[-1] - equity.index[0]).days
    years = max(elapsed_days / DAYS_PER_YEAR, 1e-9)

    total_return = ending_equity / starting_equity - 1.0

    if ending_equity > 0:
        cagr = (ending_equity / starting_equity) ** (1.0 / years) - 1.0
    else:
        cagr = -1.0

    daily_std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    annual_volatility = daily_std * math.sqrt(TRADING_DAYS_PER_YEAR)

    rf_daily = _daily_risk_free(returns, risk_free_rate)
    excess = returns - rf_daily

    if daily_std > 0:
        sharpe = (
            float(excess.mean())
            / daily_std
            * math.sqrt(TRADING_DAYS_PER_YEAR)
        )
    else:
        sharpe = 0.0

    downside = excess.clip(upper=0.0)
    downside_deviation = float(
        np.sqrt((downside**2).mean())
    ) * math.sqrt(TRADING_DAYS_PER_YEAR)

    if downside_deviation > 0:
        sortino = (
            float(excess.mean())
            * TRADING_DAYS_PER_YEAR
            / downside_deviation
        )
    else:
        sortino = 0.0

    peaks = equity.cummax()
    drawdown = (equity - peaks) / peaks
    max_drawdown = abs(float(drawdown.min()))

    calmar = cagr / max_drawdown if max_drawdown > 0 else 0.0

    if "exposure_pct" in curve.columns:
        exposure = curve["exposure_pct"].astype(float)
        average_exposure = float(exposure.mean())
    else:
        average_exposure = float("nan")

    if "open_positions" in curve.columns:
        time_in_market = float(
            (curve["open_positions"] > 0).mean()
        )
    else:
        time_in_market = float("nan")

    # What the strategy earns per unit of capital actually deployed.
    # A strategy that is flat eighty percent of the time and still
    # matches the index is doing something the index is not.
    if average_exposure and average_exposure > 0:
        exposure_adjusted_cagr = cagr / average_exposure
    else:
        exposure_adjusted_cagr = float("nan")

    return {
        "start_session": equity.index[0],
        "end_session": equity.index[-1],
        "sessions": int(len(equity)),
        "years": years,
        "starting_equity": starting_equity,
        "ending_equity": ending_equity,
        "total_return": total_return,
        "cagr": cagr,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_pct": max_drawdown,
        "max_drawdown_duration_days": max_drawdown_duration_days(equity),
        "calmar": calmar,
        "time_in_market": time_in_market,
        "average_exposure": average_exposure,
        "exposure_adjusted_cagr": exposure_adjusted_cagr,
    }
