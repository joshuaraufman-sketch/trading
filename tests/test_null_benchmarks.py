from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_lab.backtest.benchmark import (
    build_benchmark_curve,
    build_exposure_matched_curve,
    build_static_blend_curve,
    compare_to_benchmark,
)
from trading_lab.backtest.equity import normalize_session_dates
from trading_lab.backtest.performance import calculate_performance


def _bars(seed=3, periods=520, symbol="SPY", drift=0.0004, vol=0.011):
    rng = np.random.default_rng(seed)
    closes = 100 * np.cumprod(1 + rng.normal(drift, vol, periods))
    stamps = pd.bdate_range("2022-01-03", periods=periods, tz="UTC")

    return pd.DataFrame(
        {
            "symbol": symbol,
            "timestamp": stamps + pd.Timedelta(hours=5),
            "close": closes,
        }
    )


def _sessions(bars):
    return pd.Index(
        normalize_session_dates(bars["timestamp"]).unique()
    ).sort_values()


def test_full_weight_blend_equals_buy_and_hold():
    bars = _bars()
    sessions = _sessions(bars)

    blend = build_static_blend_curve(
        bars,
        "SPY",
        starting_equity=100_000,
        weight=1.0,
        sessions=sessions,
        risk_free_rate=0.0,
    )
    hold = build_benchmark_curve(
        bars,
        "SPY",
        starting_equity=100_000,
        sessions=sessions,
    )

    assert blend["equity"].iloc[-1] == pytest.approx(
        hold["equity"].iloc[-1],
        rel=1e-9,
    )


def test_zero_weight_blend_earns_only_the_risk_free_rate():
    bars = _bars()
    sessions = _sessions(bars)

    blend = build_static_blend_curve(
        bars,
        "SPY",
        starting_equity=100_000,
        weight=0.0,
        sessions=sessions,
        risk_free_rate=0.05,
    )

    years = (sessions[-1] - sessions[0]).days / 365.25
    expected = 100_000 * (1.05 ** years)

    assert blend["equity"].iloc[-1] == pytest.approx(expected, rel=0.02)
    assert blend["drawdown_pct"].min() == pytest.approx(0.0)


def test_blend_volatility_scales_with_weight():
    bars = _bars()
    sessions = _sessions(bars)

    vols = []

    for weight in (0.25, 0.5, 1.0):
        curve = build_static_blend_curve(
            bars,
            "SPY",
            starting_equity=100_000,
            weight=weight,
            sessions=sessions,
        )
        vols.append(calculate_performance(curve)["annual_volatility"])

    assert vols[0] < vols[1] < vols[2]
    # Volatility should be close to linear in weight.
    assert vols[0] == pytest.approx(vols[2] * 0.25, rel=0.05)


def test_exposure_matched_reproduces_a_pure_index_holder():
    """
    A strategy fully invested in the index every day must be matched
    exactly by its own exposure-matched null: there is nothing left for
    selection or timing to explain.
    """

    bars = _bars()
    sessions = _sessions(bars)

    hold = build_benchmark_curve(
        bars,
        "SPY",
        starting_equity=100_000,
        sessions=sessions,
    )

    matched = build_exposure_matched_curve(
        bars,
        "SPY",
        hold,
        starting_equity=100_000,
        risk_free_rate=0.0,
    )

    # Exposure is lagged one session, so compare from the second bar on.
    assert matched["equity"].iloc[-1] == pytest.approx(
        hold["equity"].iloc[-1],
        rel=1e-6,
    )


def test_exposure_matched_does_not_peek():
    """
    The null must not benefit from knowing tomorrow's exposure. Build a
    strategy curve that is flat except for one perfectly timed day; the
    matched null should not capture that day's move.
    """

    bars = _bars(seed=11, periods=60)
    sessions = _sessions(bars)

    exposure = pd.Series(0.0, index=sessions)
    best_day = 30
    exposure.iloc[best_day] = 1.0

    strategy_curve = pd.DataFrame(
        {"equity": 100_000.0, "exposure_pct": exposure}
    )

    matched = build_exposure_matched_curve(
        bars,
        "SPY",
        strategy_curve,
        starting_equity=100_000,
        risk_free_rate=0.0,
    )

    returns = matched["daily_return"]

    # Only the session AFTER the exposed close should move.
    assert returns.iloc[best_day] == pytest.approx(0.0, abs=1e-12)
    assert returns.iloc[best_day + 1] != pytest.approx(0.0, abs=1e-12)


def test_alpha_carries_a_t_statistic_and_interval():
    bars = _bars()
    sessions = _sessions(bars)

    hold = build_benchmark_curve(
        bars, "SPY", starting_equity=100_000, sessions=sessions
    )
    blend = build_static_blend_curve(
        bars,
        "SPY",
        starting_equity=100_000,
        weight=0.4,
        sessions=sessions,
    )

    comparison = compare_to_benchmark(blend, hold)

    for key in (
        "alpha_t_stat",
        "alpha_standard_error",
        "alpha_ci_low",
        "alpha_ci_high",
        "residual_volatility",
        "alpha_significant_at_05",
    ):
        assert key in comparison

    # A constant-weight blend has no skill, so alpha must be noise.
    assert not comparison["alpha_significant_at_05"]
    assert abs(comparison["alpha_t_stat"]) < 1.96
    assert comparison["alpha_ci_low"] < 0 < comparison["alpha_ci_high"]


def test_invalid_weight_is_rejected():
    bars = _bars()
    sessions = _sessions(bars)

    with pytest.raises(ValueError, match="weight must be between"):
        build_static_blend_curve(
            bars,
            "SPY",
            starting_equity=100_000,
            weight=1.5,
            sessions=sessions,
        )
