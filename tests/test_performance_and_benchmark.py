from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from trading_lab.backtest.benchmark import (
    build_benchmark_curve,
    compare_to_benchmark,
)
from trading_lab.backtest.equity import (
    build_daily_equity_curve,
    normalize_session_dates,
)
from trading_lab.backtest.models import Trade
from trading_lab.backtest.performance import (
    calculate_performance,
    max_drawdown_duration_days,
)


def _curve_from_equity(values, start="2024-01-02"):
    sessions = pd.bdate_range(start=start, periods=len(values))
    equity = pd.Series(values, index=sessions, dtype=float)

    curve = pd.DataFrame({"equity": equity})
    curve["open_positions"] = 1
    curve["exposure_pct"] = 1.0

    return curve


def test_cagr_on_a_clean_doubling_over_one_year():
    sessions = pd.to_datetime(["2024-01-01", "2025-01-01"])
    curve = pd.DataFrame(
        {"equity": [100_000.0, 200_000.0]},
        index=sessions,
    )

    result = calculate_performance(curve)

    assert result["total_return"] == pytest.approx(1.0)
    # 2024 is a leap year: 366 days annualized on a 365.25 basis.
    assert result["cagr"] == pytest.approx(1.0, abs=0.01)


def test_risk_free_rate_reduces_sharpe():
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0006, 0.01, 500)
    equity = 100_000 * np.cumprod(1 + returns)

    curve = _curve_from_equity(equity)

    zero_rf = calculate_performance(curve, risk_free_rate=0.0)
    five_pct = calculate_performance(curve, risk_free_rate=0.05)

    assert five_pct["sharpe"] < zero_rf["sharpe"]
    assert zero_rf["annual_volatility"] == pytest.approx(
        five_pct["annual_volatility"]
    )


def test_drawdown_duration_counts_calendar_days_underwater():
    equity = pd.Series(
        [100.0, 90.0, 95.0, 99.0, 101.0],
        index=pd.to_datetime(
            [
                "2024-01-01",
                "2024-01-02",
                "2024-02-01",
                "2024-03-01",
                "2024-04-01",
            ]
        ),
    )

    # Underwater from 2024-01-02 until recovery on 2024-04-01.
    assert max_drawdown_duration_days(equity) == 90


def test_flat_strategy_has_zero_sharpe_not_an_exception():
    curve = _curve_from_equity([100_000.0] * 30)
    result = calculate_performance(curve)

    assert result["sharpe"] == 0.0
    assert result["sortino"] == 0.0
    assert result["max_drawdown_pct"] == 0.0


def test_benchmark_curve_tracks_underlying_price():
    sessions = pd.bdate_range("2024-01-02", periods=4, tz="UTC")
    bars = pd.DataFrame(
        {
            "symbol": "SPY",
            "timestamp": sessions + pd.Timedelta(hours=5),
            "close": [100.0, 110.0, 105.0, 120.0],
        }
    )

    curve = build_benchmark_curve(
        bars,
        "SPY",
        starting_equity=100_000,
    )

    assert curve["equity"].iloc[0] == pytest.approx(100_000)
    assert curve["equity"].iloc[1] == pytest.approx(110_000)
    assert curve["equity"].iloc[-1] == pytest.approx(120_000)


def test_a_strategy_that_is_just_beta_shows_beta_one_and_no_alpha():
    """
    The diagnostic that matters for the current sma_crossover candidate.
    A position held continuously in the benchmark should report beta of
    one, alpha of zero, and correlation of one. If the real strategy
    lands near this, it is index exposure, not edge.
    """

    rng = np.random.default_rng(11)
    closes = 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, 260))
    sessions = pd.bdate_range("2023-01-02", periods=260, tz="UTC")

    bars = pd.DataFrame(
        {
            "symbol": "SPY",
            "timestamp": sessions + pd.Timedelta(hours=5),
            "close": closes,
        }
    )

    naive_sessions = normalize_session_dates(
        pd.Series(bars["timestamp"])
    )

    quantity = 500
    trade = Trade(
        symbol="SPY",
        entry_time=naive_sessions.iloc[0],
        exit_time=naive_sessions.iloc[-1],
        entry_price=float(closes[0]),
        exit_price=float(closes[-1]),
        quantity=quantity,
        fees=0.0,
    )

    strategy_curve = build_daily_equity_curve(
        [trade],
        bars,
        starting_equity=float(closes[0]) * quantity,
    )

    benchmark_curve = build_benchmark_curve(
        bars,
        "SPY",
        starting_equity=float(closes[0]) * quantity,
        sessions=strategy_curve.index,
    )

    comparison = compare_to_benchmark(
        strategy_curve,
        benchmark_curve,
    )

    assert comparison["beta"] == pytest.approx(1.0, abs=0.02)
    assert comparison["annual_alpha"] == pytest.approx(0.0, abs=0.01)
    assert comparison["correlation"] == pytest.approx(1.0, abs=0.01)
    assert comparison["tracking_error"] == pytest.approx(0.0, abs=0.01)


def test_half_exposure_strategy_shows_beta_near_half():
    rng = np.random.default_rng(23)
    closes = 100 * np.cumprod(1 + rng.normal(0.0003, 0.011, 260))
    sessions = pd.bdate_range("2023-01-02", periods=260, tz="UTC")

    bars = pd.DataFrame(
        {
            "symbol": "SPY",
            "timestamp": sessions + pd.Timedelta(hours=5),
            "close": closes,
        }
    )

    naive = normalize_session_dates(pd.Series(bars["timestamp"]))

    starting_equity = 100_000.0
    quantity = int(starting_equity * 0.5 / closes[0])

    trade = Trade(
        symbol="SPY",
        entry_time=naive.iloc[0],
        exit_time=naive.iloc[-1],
        entry_price=float(closes[0]),
        exit_price=float(closes[-1]),
        quantity=quantity,
        fees=0.0,
    )

    strategy_curve = build_daily_equity_curve(
        [trade],
        bars,
        starting_equity=starting_equity,
    )
    benchmark_curve = build_benchmark_curve(
        bars,
        "SPY",
        starting_equity=starting_equity,
        sessions=strategy_curve.index,
    )

    comparison = compare_to_benchmark(strategy_curve, benchmark_curve)

    assert comparison["beta"] == pytest.approx(0.5, abs=0.03)
    assert comparison["strategy"]["average_exposure"] == pytest.approx(
        0.5, abs=0.05
    )
