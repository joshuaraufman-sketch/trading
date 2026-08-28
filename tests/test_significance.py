from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_lab.validation.significance import (
    build_null_distribution,
    deflated_sharpe_ratio,
    effective_sample_size,
    expected_maximum_sharpe,
    permutation_p_value,
    permute_signals,
    summarize_against_null,
)


def _signal_frame(seed: int = 3, symbols=("SPY", "QQQ"), periods=200):
    rng = np.random.default_rng(seed)
    rows = []

    sessions = pd.bdate_range("2023-01-02", periods=periods, tz="UTC")

    for symbol in symbols:
        closes = 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, periods))
        rows.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "timestamp": sessions,
                    "close": closes,
                    "signal": rng.random(periods) < 0.1,
                }
            )
        )

    return pd.concat(rows, ignore_index=True)


def test_permutation_preserves_signal_count_per_symbol():
    df = _signal_frame()
    rng = np.random.default_rng(0)

    for method in ("circular_shift", "shuffle"):
        permuted = permute_signals(df, rng=rng, method=method)

        original = df.groupby("symbol")["signal"].sum().sort_index()
        after = permuted.groupby("symbol")["signal"].sum().sort_index()

        pd.testing.assert_series_equal(original, after)


def test_permutation_actually_moves_signals():
    df = _signal_frame()
    rng = np.random.default_rng(1)

    permuted = permute_signals(df, rng=rng, method="shuffle")

    reference = df.sort_values(["symbol", "timestamp"])["signal"]
    result = permuted["signal"]

    assert not np.array_equal(
        reference.to_numpy(),
        result.to_numpy(),
    )


def test_unknown_permutation_method_is_rejected():
    df = _signal_frame()

    with pytest.raises(ValueError, match="unknown permutation method"):
        permute_signals(
            df,
            rng=np.random.default_rng(0),
            method="teleport",
        )


def test_worthless_strategy_is_not_significant():
    """
    A signal with no relationship to price must fail the permutation
    test. If this passes, the harness is broken and every result it
    blesses is worthless.
    """

    df = _signal_frame(seed=5)

    def evaluate(frame: pd.DataFrame) -> float:
        # Stand-in for a backtest: mean forward return after a signal.
        scored = frame.sort_values(["symbol", "timestamp"]).copy()
        scored["forward"] = (
            scored.groupby("symbol")["close"].pct_change().shift(-1)
        )
        selected = scored.loc[scored["signal"], "forward"].dropna()
        return float(selected.mean()) if len(selected) else 0.0

    observed = evaluate(df)

    null = build_null_distribution(
        df,
        evaluate=evaluate,
        n_permutations=120,
        seed=42,
    )

    summary = summarize_against_null(observed, null, label="mean_forward")

    assert summary["null_samples"] == 120
    assert not summary["significant_at_05"]
    assert 0.0 < summary["p_value"] <= 1.0


def test_genuinely_predictive_signal_is_significant():
    """
    The complement: a signal that actually predicts must clear the null,
    or the test has no power and would reject real edge too.
    """

    rng = np.random.default_rng(9)
    periods = 400
    sessions = pd.bdate_range("2022-01-03", periods=periods, tz="UTC")

    noise = rng.normal(0.0, 0.008, periods)
    signal = rng.random(periods) < 0.15

    # Forward return is boosted on the bar after a signal.
    returns = noise + np.roll(signal.astype(float) * 0.02, 1)
    closes = 100 * np.cumprod(1 + returns)

    df = pd.DataFrame(
        {
            "symbol": "SPY",
            "timestamp": sessions,
            "close": closes,
            "signal": signal,
        }
    )

    def evaluate(frame: pd.DataFrame) -> float:
        scored = frame.sort_values(["symbol", "timestamp"]).copy()
        scored["forward"] = (
            scored.groupby("symbol")["close"].pct_change().shift(-1)
        )
        selected = scored.loc[scored["signal"], "forward"].dropna()
        return float(selected.mean()) if len(selected) else 0.0

    observed = evaluate(df)
    null = build_null_distribution(
        df,
        evaluate=evaluate,
        n_permutations=150,
        seed=7,
    )

    summary = summarize_against_null(observed, null)

    assert summary["significant_at_05"]
    assert summary["percentile_of_null"] > 95


def test_p_value_never_reaches_zero():
    null = [0.0] * 50
    assert permutation_p_value(99.0, null) == pytest.approx(1 / 51)


def test_expected_max_sharpe_grows_with_trial_count():
    variance = 0.01

    few = expected_maximum_sharpe(2, trial_sharpe_variance=variance)
    many = expected_maximum_sharpe(36, trial_sharpe_variance=variance)
    lots = expected_maximum_sharpe(500, trial_sharpe_variance=variance)

    assert few < many < lots
    assert expected_maximum_sharpe(1, trial_sharpe_variance=variance) == 0.0


def test_deflated_sharpe_penalizes_wide_sweeps():
    """
    The same observed Sharpe should become less credible as the number
    of tried configurations rises. This is the correction the 36-point
    parameter sweep never had.
    """

    common = {
        "n_observations": 1500,
        "trial_sharpe_variance": 0.0009,
        "skew": -0.3,
        "kurtosis": 5.0,
    }

    single = deflated_sharpe_ratio(0.05, n_trials=1, **common)
    swept = deflated_sharpe_ratio(0.05, n_trials=36, **common)
    heavy = deflated_sharpe_ratio(0.05, n_trials=1000, **common)

    assert (
        single["deflated_sharpe_probability"]
        > swept["deflated_sharpe_probability"]
        > heavy["deflated_sharpe_probability"]
    )
    assert swept["expected_max_under_null"] > 0


def test_correlated_streams_shrink_effective_sample_size():
    """
    SPY/QQQ/IWM/DIA at high correlation are close to a single stream.
    200 trades across them is nowhere near 200 observations, which is
    what the minimum_trades gate currently assumes.
    """

    rng = np.random.default_rng(4)
    market = rng.normal(0, 0.01, 500)

    correlated = {
        name: pd.Series(market + rng.normal(0, 0.0025, 500))
        for name in ("SPY", "QQQ", "IWM", "DIA")
    }

    independent = {
        name: pd.Series(rng.normal(0, 0.01, 500))
        for name in ("A", "B", "C", "D")
    }

    corr_result = effective_sample_size(correlated)
    indep_result = effective_sample_size(independent)

    assert corr_result["mean_pairwise_correlation"] > 0.85
    assert corr_result["effective_sample_size"] < 0.4 * 2000
    assert indep_result["effective_sample_size"] > 0.8 * 2000
    assert (
        corr_result["inflation_factor"]
        > indep_result["inflation_factor"]
    )


def test_single_stream_is_unadjusted():
    result = effective_sample_size(
        {"SPY": pd.Series(np.random.default_rng(1).normal(0, 1, 100))}
    )

    assert result["effective_sample_size"] == 100.0
    assert result["inflation_factor"] == 1.0
