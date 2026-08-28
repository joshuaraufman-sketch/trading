"""
Tests for the end-to-end permutation pipeline.

The critical property: the pipeline must return "not significant" for a
schedule with no timing skill, and must NOT return that for a schedule
that genuinely predicts. A test harness that cannot distinguish those
two cases blesses everything it touches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_lab.backtest.benchmark import build_exposure_matched_curve
from trading_lab.backtest.equity import (
    build_daily_equity_curve,
    normalize_session_dates,
)
from trading_lab.backtest.performance import calculate_performance
from trading_lab.backtest.runner import run_long_signal_backtest
from trading_lab.validation.significance import (
    permute_signals,
    summarize_against_null,
)


EXECUTION = {
    "starting_equity": 100_000.0,
    "risk_pct": 0.005,
    "slippage_bps": 5.0,
    "fee_per_share": 0.005,
}
PARAMS = {"stop_loss_pct": 0.02, "holding_days": 10}


def _bars(seed: int, periods: int = 700, symbols=("SPY", "QQQ")):
    rng = np.random.default_rng(seed)
    stamps = (
        pd.bdate_range("2019-01-02", periods=periods, tz="UTC")
        + pd.Timedelta(hours=5)
    )

    frames = []

    for symbol in symbols:
        closes = 100 * np.cumprod(
            1 + rng.normal(0.0004, 0.012, periods)
        )
        frames.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "timestamp": stamps,
                    "open": closes,
                    "high": closes * 1.006,
                    "low": closes * 0.994,
                    "close": closes,
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def _evaluate(signal_df, bars):
    trades = run_long_signal_backtest(
        signal_df,
        starting_equity=EXECUTION["starting_equity"],
        risk_pct=EXECUTION["risk_pct"],
        stop_loss_pct=PARAMS["stop_loss_pct"],
        holding_days=PARAMS["holding_days"],
        slippage_bps=EXECUTION["slippage_bps"],
        fee_per_share=EXECUTION["fee_per_share"],
    )

    if not trades:
        return float("nan")

    curve = build_daily_equity_curve(
        trades,
        bars,
        starting_equity=EXECUTION["starting_equity"],
    )
    matched = build_exposure_matched_curve(
        bars,
        "SPY",
        curve,
        starting_equity=EXECUTION["starting_equity"],
    )

    return calculate_performance(matched)["sharpe"]


def _null(signal_df, bars, *, n=40, seed=0, method="circular_shift"):
    rng = np.random.default_rng(seed)
    return [
        _evaluate(permute_signals(signal_df, rng=rng, method=method), bars)
        for _ in range(n)
    ]


def test_random_schedule_is_not_significant():
    bars = _bars(seed=17)
    rng = np.random.default_rng(5)

    signal_df = bars.copy()
    signal_df["signal"] = rng.random(len(bars)) < 0.05

    observed = _evaluate(signal_df, bars)
    null = _null(signal_df, bars, n=40, seed=3)

    summary = summarize_against_null(observed, null)

    assert not summary["significant_at_05"]
    assert summary["null_samples"] > 30


def test_clairvoyant_schedule_is_significant():
    """
    A schedule that is reliably invested during large up-moves must
    clear the null. If the pipeline cannot detect this, it has no power
    and would reject real edge too.

    The boost is placed two bars after the signal, because that is where
    the exposure-matched metric can actually reach it:

        signal at bar i
          -> runner enters at the OPEN of bar i+1
          -> equity curve shows exposure from session i+1
          -> exposure is lagged one session to avoid peeking, so
             exposure[i+1] earns the return from close[i+1] to close[i+2]

    So the exposure-matched null measures return[i+2]. This is a real
    property of the measurement, not an artifact: it tests whether the
    schedule identifies good periods to be invested, not whether the
    signal predicts the very next bar.
    """

    rng = np.random.default_rng(21)
    periods = 700
    stamps = (
        pd.bdate_range("2019-01-02", periods=periods, tz="UTC")
        + pd.Timedelta(hours=5)
    )

    returns = rng.normal(0.0002, 0.010, periods)
    boost = rng.random(periods) < 0.10
    returns = returns + np.roll(boost.astype(float) * 0.018, 2)
    closes = 100 * np.cumprod(1 + returns)

    # Open at the prior close, so an entry does not already sit at the
    # post-move price. With open == close the strategy can never
    # capture the move it was built to catch.
    opens = np.concatenate([[100.0], closes[:-1]])

    bars = pd.DataFrame(
        {
            "symbol": "SPY",
            "timestamp": stamps,
            "open": opens,
            "high": np.maximum(opens, closes) * 1.004,
            "low": np.minimum(opens, closes) * 0.996,
            "close": closes,
        }
    )

    signal_df = bars.copy()
    signal_df["signal"] = boost

    observed = _evaluate(signal_df, bars)
    null = _null(signal_df, bars, n=40, seed=8)

    summary = summarize_against_null(observed, null)

    assert summary["significant_at_05"], (
        f"pipeline has no power: observed {observed:.3f} sits at "
        f"percentile {summary['percentile_of_null']:.1f}"
    )


def test_permutation_preserves_exposure_scale():
    """
    The null must hold a comparable amount of risk. If permuted
    schedules trade far less, the comparison is confounded and any
    Sharpe difference is about exposure, not timing.
    """

    bars = _bars(seed=31)
    rng = np.random.default_rng(2)

    signal_df = bars.copy()
    signal_df["signal"] = rng.random(len(bars)) < 0.05

    def exposure_of(frame):
        trades = run_long_signal_backtest(
            frame,
            starting_equity=EXECUTION["starting_equity"],
            risk_pct=EXECUTION["risk_pct"],
            stop_loss_pct=PARAMS["stop_loss_pct"],
            holding_days=PARAMS["holding_days"],
            slippage_bps=EXECUTION["slippage_bps"],
            fee_per_share=EXECUTION["fee_per_share"],
        )
        curve = build_daily_equity_curve(
            trades, bars, starting_equity=EXECUTION["starting_equity"]
        )
        return calculate_performance(curve)["average_exposure"]

    observed = exposure_of(signal_df)

    permuter = np.random.default_rng(11)
    permuted = [
        exposure_of(permute_signals(signal_df, rng=permuter))
        for _ in range(15)
    ]

    assert float(np.mean(permuted)) == pytest.approx(observed, rel=0.35)


def test_no_trades_yields_nan_not_zero():
    """
    A permutation producing no trades must not be scored as Sharpe 0.0.
    Silently counting empty runs as zero biases the null downward and
    makes everything look significant.
    """

    bars = _bars(seed=41, periods=120)
    signal_df = bars.copy()
    signal_df["signal"] = False

    assert np.isnan(_evaluate(signal_df, bars))


def test_summary_discards_nan_samples():
    summary = summarize_against_null(
        1.0,
        [0.1, 0.2, float("nan"), 0.3, float("nan")],
    )

    assert summary["null_samples"] == 3
    assert summary["discarded_samples"] == 2
