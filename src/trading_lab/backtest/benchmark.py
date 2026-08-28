from __future__ import annotations

import math

import pandas as pd

from trading_lab.backtest.equity import normalize_session_dates
from trading_lab.backtest.performance import (
    TRADING_DAYS_PER_YEAR,
    _daily_risk_free,
    calculate_performance,
)


def calculate_buy_and_hold_return(
    df: pd.DataFrame,
    symbol: str,
) -> float:
    """
    Calculate simple buy-and-hold return over the available data.

    Retained for backwards compatibility. Prefer ``build_benchmark_curve``
    plus ``compare_to_benchmark``: a single total-return number cannot be
    risk adjusted and hides the fact that the strategy is only exposed
    part of the time.
    """

    required = {"symbol", "timestamp", "close"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    bars = (
        df[df["symbol"] == symbol]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if len(bars) < 2:
        raise ValueError(
            f"Not enough data to calculate benchmark for {symbol}"
        )

    first_close = float(bars.iloc[0]["close"])
    last_close = float(bars.iloc[-1]["close"])

    return (last_close / first_close) - 1


def build_benchmark_curve(
    bars: pd.DataFrame,
    symbol: str,
    *,
    starting_equity: float,
    sessions: pd.Index | None = None,
    allow_fractional_shares: bool = True,
) -> pd.DataFrame:
    """
    Build a buy-and-hold equity curve on the strategy's session calendar.

    Pass ``sessions`` (the strategy curve's index) so both curves cover
    exactly the same days. Comparing a strategy measured over one window
    against a benchmark measured over another is the most common way this
    comparison gets quietly rigged.
    """

    if starting_equity <= 0:
        raise ValueError("starting_equity must be greater than zero")

    required = {"symbol", "timestamp", "close"}
    missing = required - set(bars.columns)

    if missing:
        raise ValueError(
            f"bars is missing required columns: {sorted(missing)}"
        )

    frame = bars[bars["symbol"] == symbol].copy()

    if frame.empty:
        raise ValueError(f"No bars supplied for benchmark {symbol!r}")

    frame["session"] = normalize_session_dates(frame["timestamp"])

    closes = (
        frame
        .groupby("session")["close"]
        .last()
        .sort_index()
        .astype(float)
    )

    if sessions is not None:
        closes = closes.reindex(sessions).ffill()

        if closes.isna().any():
            raise ValueError(
                f"Benchmark {symbol!r} has no price on or before the "
                f"first strategy session"
            )

    if len(closes) < 2:
        raise ValueError(
            f"Not enough benchmark sessions for {symbol!r}"
        )

    entry_price = float(closes.iloc[0])

    if allow_fractional_shares:
        quantity = starting_equity / entry_price
    else:
        quantity = float(int(starting_equity // entry_price))

    if quantity <= 0:
        raise ValueError(
            "starting_equity is too small to buy one benchmark share"
        )

    cash = starting_equity - quantity * entry_price
    position_value = closes * quantity
    equity = position_value + cash

    curve = pd.DataFrame(
        {
            "cash": cash,
            "position_value": position_value,
            "equity": equity,
            "open_positions": 1,
        }
    )

    curve["exposure_pct"] = curve["position_value"] / curve["equity"]
    curve["daily_return"] = curve["equity"].pct_change().fillna(0.0)
    curve["running_peak"] = curve["equity"].cummax()
    curve["drawdown_pct"] = (
        curve["equity"] - curve["running_peak"]
    ) / curve["running_peak"]

    curve.index.name = "session"

    return curve


def compare_to_benchmark(
    strategy_curve: pd.DataFrame,
    benchmark_curve: pd.DataFrame,
    *,
    risk_free_rate: float | pd.Series = 0.0,
) -> dict:
    """
    Compare a strategy curve to a benchmark curve on a shared calendar.

    Beta answers the question the walk-forward results already raise: is
    this a strategy, or is it long index exposure with extra steps? A
    long-only trend filter on correlated index ETFs will show high beta
    and near-zero alpha, and that is the finding, not a bug.
    """

    shared = strategy_curve.index.intersection(benchmark_curve.index)

    if len(shared) < 3:
        raise ValueError(
            "strategy and benchmark curves share too few sessions "
            "to compare"
        )

    strategy = strategy_curve.loc[shared]
    benchmark = benchmark_curve.loc[shared]

    strategy_returns = strategy["equity"].pct_change().dropna()
    benchmark_returns = benchmark["equity"].pct_change().dropna()

    common = strategy_returns.index.intersection(
        benchmark_returns.index
    )
    strategy_returns = strategy_returns.loc[common]
    benchmark_returns = benchmark_returns.loc[common]

    rf_daily = _daily_risk_free(strategy_returns, risk_free_rate)

    strategy_excess = strategy_returns - rf_daily
    benchmark_excess = benchmark_returns - rf_daily

    benchmark_variance = float(benchmark_excess.var(ddof=1))

    if benchmark_variance > 0:
        covariance = float(
            strategy_excess.cov(benchmark_excess)
        )
        beta = covariance / benchmark_variance
    else:
        beta = float("nan")

    # Alpha without an error bar is not a finding. A small positive
    # number here is the single easiest way to talk yourself into a dead
    # strategy, so the t-statistic and interval travel with it.
    if beta == beta:  # not NaN
        daily_alpha = float(
            strategy_excess.mean() - beta * benchmark_excess.mean()
        )
        annual_alpha = daily_alpha * TRADING_DAYS_PER_YEAR

        residuals = strategy_excess - beta * benchmark_excess
        observations = int(len(residuals))
        residual_daily_std = float(residuals.std(ddof=1))

        residual_volatility = residual_daily_std * math.sqrt(
            TRADING_DAYS_PER_YEAR
        )

        if residual_daily_std > 0 and observations > 1:
            alpha_standard_error = (
                residual_daily_std
                / math.sqrt(observations)
                * TRADING_DAYS_PER_YEAR
            )
            alpha_t_stat = annual_alpha / alpha_standard_error
        else:
            alpha_standard_error = float("nan")
            alpha_t_stat = float("nan")
    else:
        annual_alpha = float("nan")
        residual_volatility = float("nan")
        alpha_standard_error = float("nan")
        alpha_t_stat = float("nan")

    correlation = float(strategy_returns.corr(benchmark_returns))

    active = strategy_returns - benchmark_returns
    tracking_error = float(
        active.std(ddof=1)
    ) * math.sqrt(TRADING_DAYS_PER_YEAR)

    if tracking_error > 0:
        information_ratio = (
            float(active.mean())
            * TRADING_DAYS_PER_YEAR
            / tracking_error
        )
    else:
        information_ratio = 0.0

    strategy_performance = calculate_performance(
        strategy,
        risk_free_rate=risk_free_rate,
    )
    benchmark_performance = calculate_performance(
        benchmark,
        risk_free_rate=risk_free_rate,
    )

    return {
        "sessions": int(len(shared)),
        "strategy": strategy_performance,
        "benchmark": benchmark_performance,
        "beta": beta,
        "annual_alpha": annual_alpha,
        "alpha_standard_error": alpha_standard_error,
        "alpha_t_stat": alpha_t_stat,
        "alpha_ci_low": annual_alpha - 1.96 * alpha_standard_error,
        "alpha_ci_high": annual_alpha + 1.96 * alpha_standard_error,
        "alpha_significant_at_05": bool(abs(alpha_t_stat) > 1.96)
        if alpha_t_stat == alpha_t_stat
        else False,
        "residual_volatility": residual_volatility,
        "correlation": correlation,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "excess_cagr": (
            strategy_performance["cagr"]
            - benchmark_performance["cagr"]
        ),
        "sharpe_difference": (
            strategy_performance["sharpe"]
            - benchmark_performance["sharpe"]
        ),
        "beats_benchmark_on_sharpe": (
            strategy_performance["sharpe"]
            > benchmark_performance["sharpe"]
        ),
    }

# ---------------------------------------------------------------------
# Null benchmarks: what you would have earned without any skill
# ---------------------------------------------------------------------


def _curve_from_returns(
    returns: pd.Series,
    *,
    starting_equity: float,
    exposure: pd.Series,
) -> pd.DataFrame:
    """
    Assemble a standard curve frame from a series of period returns.

    ``returns`` must already be lagged so that the exposure held at the
    close of t-1 earns the return from t-1 to t. The first session is
    seeded at ``starting_equity`` with a zero return.
    """

    equity = starting_equity * (1.0 + returns).cumprod()

    curve = pd.DataFrame(
        {
            "equity": equity,
            "exposure_pct": exposure.reindex(equity.index).fillna(0.0),
        }
    )

    curve["position_value"] = curve["equity"] * curve["exposure_pct"]
    curve["cash"] = curve["equity"] - curve["position_value"]
    curve["open_positions"] = (curve["exposure_pct"] > 0).astype(int)
    curve["daily_return"] = returns
    curve["running_peak"] = curve["equity"].cummax()
    curve["drawdown_pct"] = (
        curve["equity"] - curve["running_peak"]
    ) / curve["running_peak"]

    curve.index.name = "session"

    return curve


def _benchmark_returns(
    bars: pd.DataFrame,
    symbol: str,
    sessions: pd.Index,
) -> pd.Series:
    frame = bars[bars["symbol"] == symbol].copy()

    if frame.empty:
        raise ValueError(f"No bars supplied for benchmark {symbol!r}")

    frame["session"] = normalize_session_dates(frame["timestamp"])

    closes = (
        frame
        .groupby("session")["close"]
        .last()
        .sort_index()
        .astype(float)
        .reindex(sessions)
        .ffill()
    )

    if closes.isna().any():
        raise ValueError(
            f"Benchmark {symbol!r} has no price on or before the first "
            f"strategy session"
        )

    return closes.pct_change().fillna(0.0)


def build_static_blend_curve(
    bars: pd.DataFrame,
    symbol: str,
    *,
    starting_equity: float,
    weight: float,
    sessions: pd.Index,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    A fixed weight in the index, the remainder in cash, rebalanced daily.

    This is the null that matters most for a partially invested
    strategy. Comparing a strategy that averages forty percent exposure
    against a hundred percent buy-and-hold is too easy to wave away:
    "of course I underperform, I am in cash half the time." Matching the
    average exposure removes that excuse. If the strategy cannot beat a
    constant weight with no signals, no stops and no parameter sweep,
    the entire apparatus is subtracting value.

    Set ``weight`` to the strategy's average exposure.
    """

    if starting_equity <= 0:
        raise ValueError("starting_equity must be greater than zero")

    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be between 0 and 1")

    index_returns = _benchmark_returns(bars, symbol, sessions)
    rf_daily = (1.0 + risk_free_rate) ** (
        1.0 / TRADING_DAYS_PER_YEAR
    ) - 1.0

    blended = weight * index_returns + (1.0 - weight) * rf_daily
    blended.iloc[0] = 0.0

    exposure = pd.Series(weight, index=sessions)

    return _curve_from_returns(
        blended,
        starting_equity=starting_equity,
        exposure=exposure,
    )


def build_exposure_matched_curve(
    bars: pd.DataFrame,
    symbol: str,
    strategy_curve: pd.DataFrame,
    *,
    starting_equity: float,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Hold the index at the strategy's own daily exposure, in the index.

    Sharper than the static blend. It grants the strategy its entire
    exposure schedule for free -- every decision about when to be in and
    how large -- and asks only whether the choice of *which* instruments
    and *which* specific entries added anything on top. Underperforming
    this means the symbol selection and entry timing are actively
    destroying value, not merely failing to create it.

    Exposure is lagged one session: the weight held at the close of t-1
    earns the return from t-1 to t. Without the lag this would peek.
    """

    if starting_equity <= 0:
        raise ValueError("starting_equity must be greater than zero")

    if "exposure_pct" not in strategy_curve.columns:
        raise ValueError(
            "strategy_curve must contain an 'exposure_pct' column"
        )

    sessions = strategy_curve.index
    index_returns = _benchmark_returns(bars, symbol, sessions)

    rf_daily = (1.0 + risk_free_rate) ** (
        1.0 / TRADING_DAYS_PER_YEAR
    ) - 1.0

    exposure = (
        strategy_curve["exposure_pct"]
        .astype(float)
        .shift(1)
        .fillna(0.0)
        .clip(lower=0.0)
    )

    blended = exposure * index_returns + (1.0 - exposure) * rf_daily
    blended.iloc[0] = 0.0

    return _curve_from_returns(
        blended,
        starting_equity=starting_equity,
        exposure=exposure,
    )
