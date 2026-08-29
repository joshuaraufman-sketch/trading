"""
Tests for genuine walk-forward validation.

The properties that matter are all about ordering. A walk-forward that
lets training data touch a test window is not a weaker test -- it is the
same in-sample backtest wearing a different label, which is exactly the
failure the previous implementation had.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_lab.validation.walk_forward import (
    Fold,
    FoldResult,
    make_folds,
    parameter_stability,
    run_walk_forward,
    stitch_out_of_sample,
)


def _sessions(n=500):
    return pd.bdate_range("2020-01-02", periods=n)


def test_training_never_touches_the_test_window():
    """The single property that makes this validation rather than fitting."""

    folds = make_folds(_sessions(500), train_sessions=200, test_sessions=50)

    assert folds

    for fold in folds:
        assert fold.train_end < fold.test_start


def test_test_windows_are_disjoint_and_ordered():
    folds = make_folds(_sessions(500), train_sessions=200, test_sessions=50)

    for earlier, later in zip(folds, folds[1:]):
        assert earlier.test_end < later.test_start


def test_anchored_training_grows_rolling_stays_fixed():
    anchored = make_folds(
        _sessions(500), train_sessions=200, test_sessions=50,
        mode="anchored",
    )
    rolling = make_folds(
        _sessions(500), train_sessions=200, test_sessions=50,
        mode="rolling",
    )

    anchored_lengths = [f.train_sessions for f in anchored]
    rolling_lengths = [f.train_sessions for f in rolling]

    assert anchored_lengths == sorted(anchored_lengths)
    assert anchored_lengths[-1] > anchored_lengths[0]
    assert set(rolling_lengths) == {200}


def test_fold_count_matches_available_data():
    # 500 sessions, 200 train, 50 test -> boundaries at 200,250,...,450
    folds = make_folds(_sessions(500), train_sessions=200, test_sessions=50)
    assert len(folds) == 6


def test_insufficient_data_is_rejected():
    with pytest.raises(ValueError, match="need at least"):
        make_folds(_sessions(100), train_sessions=200, test_sessions=50)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown walk-forward mode"):
        make_folds(
            _sessions(500), train_sessions=200, test_sessions=50,
            mode="psychic",
        )


def test_selection_only_ever_sees_training_sessions():
    """
    Drive the loop with a select callable that records what it was
    given, then assert none of it fell inside a test window.
    """

    sessions = _sessions(400)
    seen: list[tuple] = []

    def select(fold):
        seen.append((fold.train_start, fold.train_end, fold.test_start))
        return {"param": fold.index}, 1.0, {}

    def evaluate(fold, config):
        window = pd.bdate_range(
            fold.test_start, fold.test_end, freq="C"
        )
        return pd.Series(0.001, index=window), 5, {}

    run_walk_forward(
        sessions,
        select=select,
        evaluate=evaluate,
        train_sessions=150,
        test_sessions=50,
    )

    assert seen

    for train_start, train_end, test_start in seen:
        assert train_start <= train_end < test_start


def test_stitched_curve_chains_across_folds():
    """
    Equity must carry across fold boundaries. Resetting to the starting
    balance each fold would hide compounding drawdowns.
    """

    first = pd.Series(
        0.01, index=pd.bdate_range("2021-01-04", periods=10)
    )
    second = pd.Series(
        0.01, index=pd.bdate_range("2021-01-18", periods=10)
    )

    results = [
        FoldResult(
            fold=Fold(0, *([pd.Timestamp("2021-01-01")] * 4), 10, 10),
            selected={"a": 1},
            selection_score=1.0,
            test_returns=first,
        ),
        FoldResult(
            fold=Fold(1, *([pd.Timestamp("2021-01-01")] * 4), 10, 10),
            selected={"a": 2},
            selection_score=1.0,
            test_returns=second,
        ),
    ]

    curve = stitch_out_of_sample(results, starting_equity=100_000)

    assert len(curve) == 20
    assert curve["equity"].iloc[-1] == pytest.approx(
        100_000 * (1.01 ** 20)
    )
    assert curve.index.is_monotonic_increasing


def test_overlapping_folds_are_rejected_when_stitching():
    """
    Overlapping out-of-sample windows would double count sessions and
    understate variance. Refuse rather than silently average.
    """

    window = pd.bdate_range("2021-01-04", periods=10)

    results = [
        FoldResult(
            fold=Fold(i, *([pd.Timestamp("2021-01-01")] * 4), 10, 10),
            selected={"a": i},
            selection_score=1.0,
            test_returns=pd.Series(0.001, index=window),
        )
        for i in range(2)
    ]

    with pytest.raises(ValueError, match="overlap"):
        stitch_out_of_sample(results, starting_equity=100_000)


def test_parameter_stability_flags_churn():
    """
    Parameters that jump every fold indicate overfitting even when the
    stitched curve looks acceptable.
    """

    def build(values):
        return [
            FoldResult(
                fold=Fold(i, *([pd.Timestamp("2021-01-01")] * 4), 10, 10),
                selected={"sma_window": v},
                selection_score=1.0,
            )
            for i, v in enumerate(values)
        ]

    stable = parameter_stability(build([10, 10, 10, 10]))
    churning = parameter_stability(build([10, 60, 20, 40]))

    assert stable["sma_window"]["distinct"] == 1
    assert stable["sma_window"]["changed_fraction"] == 0.0

    assert churning["sma_window"]["distinct"] == 4
    assert churning["sma_window"]["changed_fraction"] == 1.0
    assert (
        churning["sma_window"]["coefficient_of_variation"]
        > stable["sma_window"].get("coefficient_of_variation", 0.0)
    )


def test_folds_with_no_selection_are_skipped_not_zeroed():
    """
    A training window that produces no viable configuration must
    contribute nothing, not a flat zero-return stretch that would
    dampen measured volatility.
    """

    sessions = _sessions(400)

    def select(fold):
        if fold.index == 0:
            return None, float("nan"), {"reason": "no viable config"}
        return {"param": 1}, 1.0, {}

    def evaluate(fold, config):
        window = pd.bdate_range(fold.test_start, periods=5)
        return pd.Series(0.002, index=window), 3, {}

    results = run_walk_forward(
        sessions,
        select=select,
        evaluate=evaluate,
        train_sessions=150,
        test_sessions=50,
    )

    assert results[0].selected is None
    assert len(results[0].test_returns) == 0

    curve = stitch_out_of_sample(results, starting_equity=100_000)
    assert len(curve) == (len(results) - 1) * 5
