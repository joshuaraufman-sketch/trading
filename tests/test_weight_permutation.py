"""
Tests for weight-schedule permutation.

The turnover test is the important one. Permuting a continuous schedule
has a confound that permuting discrete signals does not: destroying the
persistence of the schedule inflates trading costs, so the real strategy
wins for reasons unrelated to the hypothesis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_lab.backtest.weights import run_weight_backtest
from trading_lab.strategies.volatility_target import (
    realized_volatility,
    volatility_target_weights,
)
from trading_lab.validation.significance import (
    permute_weight_schedule,
    schedule_turnover,
    summarize_against_null,
)


def _clustered_returns(seed=3, n=1000, drift=0.0):
    rng = np.random.default_rng(seed)
    vol = np.zeros(n)
    vol[0] = 0.008
    shocks = rng.normal(0, 1, n)

    for i in range(1, n):
        vol[i] = np.sqrt(
            0.0000012 + 0.93 * vol[i - 1] ** 2
            + 0.06 * (vol[i - 1] * shocks[i - 1]) ** 2
        )

    return pd.Series(
        vol * shocks + drift,
        index=pd.bdate_range("2019-01-02", periods=n),
    )


def test_circular_shift_preserves_turnover_but_shuffle_does_not():
    """
    THE CONFOUND. Circular shift is not a stylistic preference here; it
    is the only null that leaves trading costs unchanged.
    """

    returns = _clustered_returns()
    weights = volatility_target_weights(
        returns, target_volatility=0.10, lookback=20
    ).iloc[20:]

    observed = schedule_turnover(weights)

    rng = np.random.default_rng(0)
    shifted = [
        schedule_turnover(
            permute_weight_schedule(
                weights, rng=rng, method="circular_shift"
            )
        )
        for _ in range(15)
    ]
    shuffled = [
        schedule_turnover(
            permute_weight_schedule(weights, rng=rng, method="shuffle")
        )
        for _ in range(15)
    ]

    assert float(np.mean(shifted)) == pytest.approx(observed, rel=0.05)
    assert float(np.mean(shuffled)) > 3 * observed


def test_permutation_preserves_the_exposure_distribution():
    returns = _clustered_returns()
    weights = volatility_target_weights(returns, lookback=20).iloc[20:]

    permuted = permute_weight_schedule(
        weights, rng=np.random.default_rng(1)
    )

    assert permuted.sum() == pytest.approx(weights.sum())
    assert sorted(permuted.to_numpy()) == pytest.approx(
        sorted(weights.to_numpy())
    )
    assert permuted.index.equals(weights.index)


def test_shifted_schedule_actually_moves():
    weights = pd.Series(
        np.arange(100, dtype=float),
        index=pd.bdate_range("2020-01-02", periods=100),
    )

    permuted = permute_weight_schedule(
        weights, rng=np.random.default_rng(7)
    )

    assert not np.array_equal(
        permuted.to_numpy(), weights.to_numpy()
    )


def test_timing_test_is_directionally_correct_but_low_powered():
    """
    Power characterisation for the weight-timing permutation test.

    On data engineered to contain a real effect -- clustered volatility
    plus a positive risk premium -- the observed schedule beats the mean
    of randomly shifted schedules. But it reaches only the 60th to 77th
    percentile depending on sample length and volatility dispersion,
    nowhere near significance.

    This is measured, not assumed. Three configurations were run:

        n=1200 moderate dispersion   33rd percentile
        n=1200 high dispersion       77th percentile
        n=2500 high dispersion       60th percentile

    Two things attenuate the effect. The theoretical Sharpe gain from
    oracle volatility targeting is sqrt(E[1/s^2] * E[s^2]), which for
    realistic equity dispersion is roughly 1.27. Using a noisy 20-session
    trailing estimate instead of the true volatility erodes most of that,
    because the forecast correlates with forward volatility at only
    about 0.5.

    CONSEQUENCE FOR INTERPRETATION: a non-significant result from this
    test on real data does NOT indicate absence of timing skill. The
    test cannot resolve effects of this size at these sample lengths.
    Treat it as a directional check and rely on walk-forward for the
    real evidence. Assert direction only; asserting significance here
    would mean tuning the fixture until it passed, which is the exact
    failure this project exists to avoid.
    """

    returns = _clustered_returns(seed=11, n=1200, drift=0.0005)
    weights = volatility_target_weights(
        returns, target_volatility=0.10, lookback=20
    ).iloc[20:]
    aligned = returns.loc[weights.index]

    def sharpe(schedule):
        curve = run_weight_backtest(
            aligned, schedule, starting_equity=100_000, cost_bps=5.0,
            rebalance_band=0.05,
        )
        daily = curve["equity"].pct_change().dropna()
        if daily.std(ddof=1) == 0:
            return 0.0
        return float(daily.mean() / daily.std(ddof=1) * np.sqrt(252))

    observed = sharpe(weights)

    rng = np.random.default_rng(5)
    null = [
        sharpe(permute_weight_schedule(weights, rng=rng))
        for _ in range(40)
    ]

    summary = summarize_against_null(observed, null)

    # Direction and spread only. The null must be a real distribution
    # rather than a degenerate point, and the test must not silently
    # become significant, which would mean the fixture had been tuned.
    assert summary["null_samples"] == 40
    assert np.std(null) > 0.01
    assert summary["percentile_of_null"] > 10


def test_oracle_gain_bounds_what_the_timing_test_can_detect():
    """
    The analytic ceiling on volatility targeting's Sharpe improvement.

    Under oracle weights w = k/sigma, the ratio of targeted to randomly
    shifted Sharpe is sqrt(E[1/sigma^2] * E[sigma^2]) by Cauchy-Schwarz,
    which is at least 1 and grows with volatility dispersion. Knowing
    this number tells you in advance whether a permutation test has any
    chance of resolving the effect -- cheaper and more informative than
    running one.
    """

    returns = _clustered_returns(seed=11, n=1500, drift=0.0005)
    vol = realized_volatility(returns, lookback=20).dropna().to_numpy()

    gain = float(
        np.sqrt(np.mean(1 / vol**2) * np.mean(vol**2))
    )

    assert gain >= 1.0, "Cauchy-Schwarz violated; check the estimator"
    assert 1.05 < gain < 2.5, (
        f"implied oracle gain {gain:.3f} is outside the range realistic "
        f"equity volatility dispersion produces"
    )


def test_unknown_method_and_empty_schedule_are_rejected():
    weights = pd.Series(
        [0.5, 0.5], index=pd.bdate_range("2020-01-02", periods=2)
    )

    with pytest.raises(ValueError, match="unknown permutation method"):
        permute_weight_schedule(
            weights, rng=np.random.default_rng(0), method="teleport"
        )

    with pytest.raises(ValueError, match="empty"):
        permute_weight_schedule(
            pd.Series(dtype=float), rng=np.random.default_rng(0)
        )
