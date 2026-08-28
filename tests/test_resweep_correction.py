"""
Tests for the selection-corrected permutation path.

The property that matters: the corrected null must sit HIGHER than the
single-configuration null. If it does not, the correction is not
correcting anything and every p-value it produces is too small.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trading_lab.validation import permutation as permtest
from trading_lab.validation import sweep_grid
from trading_lab.validation.significance import permute_signals


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXECUTION = {
    "starting_equity": 100_000.0,
    "risk_pct": 0.005,
    "slippage_bps": 5.0,
    "fee_per_share": 0.005,
}


def _bars(seed=13, periods=420):
    rng = np.random.default_rng(seed)
    stamps = (
        pd.bdate_range("2020-01-02", periods=periods, tz="UTC")
        + pd.Timedelta(hours=5)
    )
    closes = 100 * np.cumprod(1 + rng.normal(0.0004, 0.012, periods))
    opens = np.concatenate([[100.0], closes[:-1]])

    return pd.DataFrame(
        {
            "symbol": "SPY",
            "timestamp": stamps,
            "open": opens,
            "high": np.maximum(opens, closes) * 1.004,
            "low": np.minimum(opens, closes) * 0.996,
            "close": closes,
        }
    )


def _signal_frames(bars, seed=4):
    """One signal set per window, as the real script builds them."""

    rng = np.random.default_rng(seed)
    frames = {}

    for window in sweep_grid.SMA_WINDOWS:
        frame = bars.copy()
        frame["signal"] = rng.random(len(bars)) < 0.06
        frames[window] = frame

    return frames


def test_grid_is_shared_not_duplicated():
    """
    The correction is only valid if it sweeps the same grid the real
    selection used. Both must come from one module.
    """

    sweep_source = (
        PROJECT_ROOT / "scripts" / "run_parameter_sweep.py"
    ).read_text()

    assert "sweep_grid" in sweep_source
    assert sweep_grid.grid_size() == 36
    assert len(sweep_grid.runner_combinations()) == 9


def test_evaluate_sweep_visits_every_configuration():
    bars = _bars()
    frames = _signal_frames(bars)

    result = permtest.evaluate_sweep(
        frames,
        bars,
        execution=EXECUTION,
        risk_free_rate=0.03,
    )

    assert result["configurations_evaluated"] == sweep_grid.grid_size()
    assert result["winning_config"] is not None
    assert result["winning_config"]["sma_window"] in sweep_grid.SMA_WINDOWS


def test_best_of_sweep_beats_any_single_configuration():
    """
    Taking the maximum over 36 configurations must be at least as good
    as any one of them. This is what raises the null and shrinks the
    p-value's overstatement.
    """

    bars = _bars()
    frames = _signal_frames(bars)

    swept = permtest.evaluate_sweep(
        frames,
        bars,
        execution=EXECUTION,
        risk_free_rate=0.03,
    )

    single = permtest.evaluate_schedule(
        frames[sweep_grid.SMA_WINDOWS[0]],
        bars,
        execution=EXECUTION,
        params={"holding_days": 10, "stop_loss_pct": 0.02},
        risk_free_rate=0.03,
    )

    assert swept["best_exposure_matched"] >= single["exposure_matched"]
    assert swept["best_strategy"] >= single["strategy"]


def test_corrected_null_sits_above_single_config_null():
    """
    The whole point of the correction. If the best-of-36 null were not
    higher than the fixed-parameter null, the correction would be
    cosmetic and its p-values would still be too small.
    """

    bars = _bars(seed=29)
    frames = _signal_frames(bars, seed=6)
    window = sweep_grid.SMA_WINDOWS[0]

    single_null = []
    rng = np.random.default_rng(1)

    for _ in range(12):
        permuted = permute_signals(frames[window], rng=rng)
        result = permtest.evaluate_schedule(
            permuted,
            bars,
            execution=EXECUTION,
            params={"holding_days": 10, "stop_loss_pct": 0.02},
            risk_free_rate=0.03,
        )
        single_null.append(result["exposure_matched"])

    swept_null = []
    rng = np.random.default_rng(1)

    for _ in range(12):
        result = permtest.evaluate_sweep(
            frames,
            bars,
            execution=EXECUTION,
            risk_free_rate=0.03,
            rng=rng,
        )
        swept_null.append(result["best_exposure_matched"])

    single_mean = float(np.nanmean(single_null))
    swept_mean = float(np.nanmean(swept_null))

    assert swept_mean > single_mean, (
        f"corrected null ({swept_mean:.3f}) must exceed single-config "
        f"null ({single_mean:.3f}) or the correction does nothing"
    )


def test_permutation_is_independent_per_window():
    """
    Each window must get its own permutation. Sharing one offset across
    all four would make the null less varied than the real procedure,
    which had four distinct signal generators to choose from.
    """

    bars = _bars()
    frames = _signal_frames(bars)

    rng = np.random.default_rng(0)
    permuted = {
        window: permute_signals(frame, rng=rng)
        for window, frame in frames.items()
    }

    windows = list(sweep_grid.SMA_WINDOWS)
    first = permuted[windows[0]]["signal"].to_numpy()
    second = permuted[windows[1]]["signal"].to_numpy()

    assert not np.array_equal(first, second)


def test_empty_sweep_returns_nan_not_negative_infinity():
    bars = _bars(periods=90)
    frames = {
        window: bars.assign(signal=False)
        for window in sweep_grid.SMA_WINDOWS
    }

    result = permtest.evaluate_sweep(
        frames,
        bars,
        execution=EXECUTION,
        risk_free_rate=0.03,
    )

    assert np.isnan(result["best_strategy"])
    assert np.isnan(result["best_exposure_matched"])
    assert result["winning_config"] is None
