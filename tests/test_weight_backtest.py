"""
Tests for weight-based backtesting and volatility targeting.

The look-ahead tests are the important ones. A weight schedule that can
see the return it is about to earn manufactures unlimited edge, and the
resulting curve looks entirely plausible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_lab.backtest.weights import (
    run_weight_backtest,
    turnover_summary,
)
from trading_lab.strategies.volatility_target import (
    realized_volatility,
    volatility_capture,
    volatility_forecast_skill,
    volatility_target_weights,
)


def _sessions(n):
    return pd.bdate_range("2020-01-02", periods=n)


def test_full_weight_reproduces_the_asset():
    sessions = _sessions(100)
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.normal(0.0004, 0.01, 100), index=sessions)

    curve = run_weight_backtest(
        returns,
        pd.Series(1.0, index=sessions),
        starting_equity=100_000,
        cost_bps=0.0,
    )

    # One session of lag: the first session is held flat.
    expected = 100_000 * (1 + returns.iloc[1:]).prod()

    assert curve["equity"].iloc[-1] == pytest.approx(expected, rel=1e-9)


def test_zero_weight_earns_only_cash():
    sessions = _sessions(252)
    rng = np.random.default_rng(4)
    returns = pd.Series(rng.normal(0.0, 0.02, 252), index=sessions)

    curve = run_weight_backtest(
        returns,
        pd.Series(0.0, index=sessions),
        starting_equity=100_000,
        cost_bps=0.0,
        risk_free_rate=0.05,
    )

    assert curve["equity"].iloc[-1] == pytest.approx(105_000, rel=0.01)
    assert curve["drawdown_pct"].min() == pytest.approx(0.0)


def test_weights_cannot_see_the_return_they_earn():
    """
    The test that matters. A schedule that is fully invested only on the
    single best session must NOT capture that session, because the
    weight is only knowable at the prior close.
    """

    sessions = _sessions(50)
    returns = pd.Series(0.0, index=sessions)
    returns.iloc[25] = 0.10

    perfect = pd.Series(0.0, index=sessions)
    perfect.iloc[25] = 1.0

    curve = run_weight_backtest(
        returns, perfect, starting_equity=100_000, cost_bps=0.0
    )

    assert curve["equity"].iloc[-1] == pytest.approx(100_000)
    assert curve["daily_return"].iloc[25] == pytest.approx(0.0)

    # Shifting the weight one session earlier DOES capture it.
    lagged = pd.Series(0.0, index=sessions)
    lagged.iloc[24] = 1.0

    captured = run_weight_backtest(
        returns, lagged, starting_equity=100_000, cost_bps=0.0
    )

    assert captured["equity"].iloc[-1] == pytest.approx(110_000)


def test_turnover_is_charged():
    sessions = _sessions(100)
    returns = pd.Series(0.0, index=sessions)

    # Alternate fully in and fully out every session.
    weights = pd.Series(
        [1.0 if i % 2 == 0 else 0.0 for i in range(100)],
        index=sessions,
    )

    free = run_weight_backtest(
        returns, weights, starting_equity=100_000, cost_bps=0.0
    )
    costly = run_weight_backtest(
        returns, weights, starting_equity=100_000, cost_bps=10.0
    )

    assert free["equity"].iloc[-1] == pytest.approx(100_000)
    assert costly["equity"].iloc[-1] < 99_000

    summary = turnover_summary(costly)
    assert summary["total_turnover"] > 90
    assert summary["annualized_cost_drag"] > 0


def test_rebalance_band_suppresses_small_trades():
    sessions = _sessions(200)
    rng = np.random.default_rng(9)
    returns = pd.Series(rng.normal(0.0, 0.01, 200), index=sessions)

    jittery = pd.Series(
        0.5 + rng.normal(0.0, 0.01, 200), index=sessions
    )

    tight = run_weight_backtest(
        returns, jittery, starting_equity=100_000,
        cost_bps=10.0, rebalance_band=0.0,
    )
    banded = run_weight_backtest(
        returns, jittery, starting_equity=100_000,
        cost_bps=10.0, rebalance_band=0.05,
    )

    assert (
        turnover_summary(banded)["total_turnover"]
        < turnover_summary(tight)["total_turnover"]
    )


def test_volatility_forecasts_volatility():
    """
    THE POSITIVE CONTROL. Volatility clusters, so trailing realized
    volatility must beat the unconditional mean at predicting forward
    volatility. If this fails on clustered data, the pipeline is broken
    and every negative result it has produced is suspect.
    """

    rng = np.random.default_rng(11)
    n = 1500

    # GARCH-like clustering: persistent conditional variance.
    vol = np.zeros(n)
    vol[0] = 0.01
    shocks = rng.normal(0, 1, n)

    for i in range(1, n):
        vol[i] = np.sqrt(
            0.00001 + 0.90 * vol[i - 1] ** 2
            + 0.08 * (vol[i - 1] * shocks[i - 1]) ** 2
        )

    returns = pd.Series(vol * shocks, index=_sessions(n))

    skill = volatility_forecast_skill(returns, lookback=20, horizon=20)

    assert skill["beats_unconditional_mean"], (
        f"mechanism test FAILED: shrunk R^2 = "
        f"{skill['shrunk_r_squared']:.3f}"
    )
    assert skill["correlation"] > 0.3
    assert skill["volatility_autocorrelation"] > 0.2
    assert skill["shrunk_r_squared"] > 0.10

    # The raw forecast is expected to look WORSE than the shrunk one,
    # and may well be negative. That is estimator noise, not absence of
    # signal, and conflating the two is how a real effect gets discarded.
    assert skill["raw_r_squared"] < skill["shrunk_r_squared"]
    assert 0.0 < skill["mean_shrinkage_slope"] < 1.0


def test_no_forecast_skill_on_constant_volatility():
    """
    Complement: on genuinely homoscedastic data there is nothing to
    forecast, so trailing volatility should NOT beat the unconditional
    mean by any meaningful margin.
    """

    rng = np.random.default_rng(13)
    returns = pd.Series(rng.normal(0, 0.01, 1500), index=_sessions(1500))

    skill = volatility_forecast_skill(returns, lookback=20, horizon=20)

    assert skill["shrunk_r_squared"] < 0.10
    assert abs(skill["correlation"]) < 0.25


def test_targeting_stabilises_realized_volatility():
    rng = np.random.default_rng(17)
    n = 1200

    regime = np.repeat([0.006, 0.025, 0.008, 0.030], n // 4)
    returns = pd.Series(
        rng.normal(0, 1, n) * regime, index=_sessions(n)
    )

    weights = volatility_target_weights(
        returns, target_volatility=0.15, lookback=20, max_weight=1.0
    )

    capture = volatility_capture(
        returns, weights, target_volatility=0.15, window=60
    )

    assert capture["dispersion_reduced"]
    assert (
        capture["targeted_volatility_dispersion"]
        < capture["raw_volatility_dispersion"]
    )


def test_weights_never_exceed_the_cap():
    rng = np.random.default_rng(19)
    returns = pd.Series(
        rng.normal(0, 0.0005, 600), index=_sessions(600)
    )

    weights = volatility_target_weights(
        returns, target_volatility=0.20, lookback=20, max_weight=1.0
    )

    assert weights.max() <= 1.0
    assert weights.min() >= 0.0


def test_insufficient_history_gives_zero_not_a_guess():
    rng = np.random.default_rng(23)
    returns = pd.Series(rng.normal(0, 0.01, 60), index=_sessions(60))

    weights = volatility_target_weights(returns, lookback=20)

    assert (weights.iloc[:19] == 0.0).all()
    assert weights.iloc[25] > 0.0


def test_duplicate_sessions_are_rejected():
    index = pd.DatetimeIndex(["2020-01-02", "2020-01-02", "2020-01-03"])
    returns = pd.Series([0.01, 0.02, 0.03], index=index)

    with pytest.raises(ValueError, match="duplicate sessions"):
        run_weight_backtest(
            returns,
            pd.Series(1.0, index=index),
            starting_equity=100_000,
        )
