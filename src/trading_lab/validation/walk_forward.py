"""
Genuine walk-forward validation.

The previous implementation applied one already-selected parameter set
to four fixed calendar windows, three of which were the same data those
parameters were fitted on. That measures nothing: it cannot detect
overfitting, because the parameters already saw every window.

Real walk-forward re-runs the *entire selection procedure* inside each
training window and applies the winner to the following window, which
the selection never saw. Stitching those out-of-sample windows together
produces one continuous equity curve made only of decisions that were
made in advance. That curve is the thing worth measuring.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    """One train/test split. Test always begins after train ends."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_sessions: int
    test_sessions: int


@dataclass
class FoldResult:
    fold: Fold
    selected: dict | None
    selection_score: float
    test_returns: pd.Series = field(default_factory=pd.Series)
    test_trades: int = 0
    diagnostics: dict = field(default_factory=dict)


def make_folds(
    sessions: Sequence[pd.Timestamp],
    *,
    train_sessions: int,
    test_sessions: int,
    mode: str = "anchored",
) -> list[Fold]:
    """
    Build non-overlapping, strictly forward-looking folds.

    ``anchored`` grows the training window from the start of the data,
    using every observation available at decision time. ``rolling``
    keeps it a fixed length, which adapts faster to regime change at the
    cost of throwing away history.

    Test windows are contiguous and never overlap each other, so the
    concatenated out-of-sample returns form a real timeline rather than
    a set of overlapping samples that would understate variance.
    """

    if mode not in {"anchored", "rolling"}:
        raise ValueError(f"unknown walk-forward mode: {mode!r}")

    if train_sessions < 2:
        raise ValueError("train_sessions must be at least 2")

    if test_sessions < 1:
        raise ValueError("test_sessions must be at least 1")

    index = pd.DatetimeIndex(sessions).sort_values()

    if len(index) < train_sessions + test_sessions:
        raise ValueError(
            f"need at least {train_sessions + test_sessions} sessions "
            f"for one fold; got {len(index)}"
        )

    folds: list[Fold] = []
    boundary = train_sessions
    number = 0

    while boundary + test_sessions <= len(index):
        if mode == "anchored":
            train_slice = index[:boundary]
        else:
            train_slice = index[boundary - train_sessions:boundary]

        test_slice = index[boundary:boundary + test_sessions]

        folds.append(
            Fold(
                index=number,
                train_start=train_slice[0],
                train_end=train_slice[-1],
                test_start=test_slice[0],
                test_end=test_slice[-1],
                train_sessions=len(train_slice),
                test_sessions=len(test_slice),
            )
        )

        boundary += test_sessions
        number += 1

    return folds


def run_walk_forward(
    sessions: Sequence[pd.Timestamp],
    *,
    select: Callable[[Fold], tuple[dict | None, float, dict]],
    evaluate: Callable[[Fold, dict], tuple[pd.Series, int, dict]],
    train_sessions: int,
    test_sessions: int,
    mode: str = "anchored",
    on_fold: Callable[[FoldResult], None] | None = None,
) -> list[FoldResult]:
    """
    Drive the walk-forward loop.

    ``select`` receives a fold and must run the whole selection
    procedure on the TRAINING window only, returning the winning
    configuration, its training score, and any diagnostics.

    ``evaluate`` receives the fold and that configuration and must
    return daily returns over the TEST window only.

    The split of responsibilities is deliberate: this module never
    touches price data, so the ordering guarantees it enforces can be
    tested without a data layer or a backtest engine.
    """

    folds = make_folds(
        sessions,
        train_sessions=train_sessions,
        test_sessions=test_sessions,
        mode=mode,
    )

    results: list[FoldResult] = []

    for fold in folds:
        selected, score, diagnostics = select(fold)

        if selected is None:
            result = FoldResult(
                fold=fold,
                selected=None,
                selection_score=float("nan"),
                test_returns=pd.Series(dtype=float),
                test_trades=0,
                diagnostics=diagnostics,
            )
        else:
            returns, trades, evaluation = evaluate(fold, selected)
            result = FoldResult(
                fold=fold,
                selected=selected,
                selection_score=score,
                test_returns=returns,
                test_trades=trades,
                diagnostics={**diagnostics, **evaluation},
            )

        results.append(result)

        if on_fold is not None:
            on_fold(result)

    return results


def stitch_out_of_sample(
    results: Sequence[FoldResult],
    *,
    starting_equity: float,
) -> pd.DataFrame:
    """
    Chain every test window into one continuous equity curve.

    This is the only curve in the project built entirely from decisions
    made before the data was seen. Folds are chained multiplicatively,
    so equity carries across boundaries instead of resetting.
    """

    if starting_equity <= 0:
        raise ValueError("starting_equity must be greater than zero")

    pieces = [
        result.test_returns
        for result in results
        if len(result.test_returns) > 0
    ]

    if not pieces:
        raise ValueError("no out-of-sample returns to stitch")

    returns = pd.concat(pieces).sort_index()

    duplicated = returns.index.duplicated()

    if duplicated.any():
        raise ValueError(
            f"out-of-sample windows overlap on "
            f"{int(duplicated.sum())} sessions; folds must be disjoint"
        )

    equity = starting_equity * (1.0 + returns).cumprod()

    curve = pd.DataFrame({"equity": equity, "daily_return": returns})
    curve["running_peak"] = curve["equity"].cummax()
    curve["drawdown_pct"] = (
        curve["equity"] - curve["running_peak"]
    ) / curve["running_peak"]
    curve.index.name = "session"

    return curve


def parameter_stability(
    results: Sequence[FoldResult],
) -> dict:
    """
    How much did the winning configuration move between folds?

    A strategy whose optimal parameters jump around every retraining
    window has not found a stable effect, even when its out-of-sample
    curve looks acceptable. High churn here is evidence of overfitting
    that aggregate performance can hide.
    """

    chosen = [r.selected for r in results if r.selected is not None]

    if not chosen:
        return {"folds_with_selection": 0}

    keys = sorted({key for config in chosen for key in config})
    summary: dict = {"folds_with_selection": len(chosen)}

    for key in keys:
        values = [config.get(key) for config in chosen]
        distinct = len({str(v) for v in values})

        entry = {
            "values": values,
            "distinct": distinct,
            "changed_fraction": (
                sum(
                    1
                    for a, b in zip(values, values[1:])
                    if a != b
                )
                / max(len(values) - 1, 1)
            ),
        }

        numeric = [v for v in values if isinstance(v, (int, float))]

        if len(numeric) == len(values) and len(numeric) > 1:
            mean = float(np.mean(numeric))
            entry["mean"] = mean
            entry["std"] = float(np.std(numeric, ddof=1))
            entry["coefficient_of_variation"] = (
                float(np.std(numeric, ddof=1) / abs(mean))
                if mean != 0
                else float("nan")
            )

        summary[key] = entry

    return summary
