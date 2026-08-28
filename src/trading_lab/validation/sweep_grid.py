"""
Single source of truth for the parameter sweep grid.

The selection-bias correction is only valid if the null re-runs the
*same* set of configurations the real selection chose from. Defining the
grid in two places guarantees they eventually drift and the correction
quietly stops matching reality, so both the sweep and the permutation
test import from here.

Note which stage each parameter affects: ``sma_window`` changes the
signals themselves, while ``holding_days`` and ``stop_loss_pct`` only
change how the backtest handles those signals. Anything permuting
signals must therefore permute per window, not once for the whole grid.
"""

from __future__ import annotations

from itertools import product


SMA_WINDOWS: tuple[int, ...] = (10, 20, 40, 60)
HOLDING_DAYS_OPTIONS: tuple[int, ...] = (3, 5, 10)
STOP_LOSS_OPTIONS: tuple[float, ...] = (0.015, 0.02, 0.03)


def sweep_combinations() -> list[dict]:
    """Every configuration the real selection procedure considered."""

    return [
        {
            "sma_window": window,
            "holding_days": holding_days,
            "stop_loss_pct": stop_loss_pct,
        }
        for window, holding_days, stop_loss_pct in product(
            SMA_WINDOWS,
            HOLDING_DAYS_OPTIONS,
            STOP_LOSS_OPTIONS,
        )
    ]


def runner_combinations() -> list[tuple[int, float]]:
    """
    The (holding_days, stop_loss_pct) pairs applied per signal set.

    Separated because signals only need regenerating when the window
    changes; these two can be swept against an already-built signal
    frame.
    """

    return list(product(HOLDING_DAYS_OPTIONS, STOP_LOSS_OPTIONS))


def grid_size() -> int:
    return (
        len(SMA_WINDOWS)
        * len(HOLDING_DAYS_OPTIONS)
        * len(STOP_LOSS_OPTIONS)
    )
