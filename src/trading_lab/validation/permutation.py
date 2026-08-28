"""
Reusable evaluation primitives for the permutation tests.

These live in the package rather than in the CLI script so they can be
tested without importing the whole data layer (and therefore alpaca-py).
The script is a thin wrapper around them.
"""

from __future__ import annotations

import math

import pandas as pd

from trading_lab.backtest.benchmark import build_exposure_matched_curve
from trading_lab.backtest.equity import build_daily_equity_curve
from trading_lab.backtest.metrics import calculate_metrics
from trading_lab.backtest.performance import calculate_performance
from trading_lab.backtest.runner import run_long_signal_backtest
from trading_lab.validation import sweep_grid
from trading_lab.validation.significance import permute_signals


BENCHMARK_SYMBOL = "SPY"


def evaluate_schedule(
    signal_df,
    bars,
    *,
    execution: dict,
    params: dict,
    risk_free_rate: float,
    want_profit_factor: bool = False,
) -> dict:
    """
    Run one backtest and return both Sharpe figures plus exposure.

    Returns NaN metrics when a permutation produces no trades, which
    happens occasionally and must not be silently counted as zero.
    """

    trades = run_long_signal_backtest(
        signal_df,
        starting_equity=float(execution["starting_equity"]),
        risk_pct=float(execution["risk_pct"]),
        stop_loss_pct=float(params["stop_loss_pct"]),
        holding_days=int(params["holding_days"]),
        slippage_bps=float(execution["slippage_bps"]),
        fee_per_share=float(execution["fee_per_share"]),
    )

    if not trades:
        return {
            "strategy": float("nan"),
            "exposure_matched": float("nan"),
            "average_exposure": float("nan"),
            "profit_factor": float("nan"),
            "trades": 0,
        }

    starting_equity = float(execution["starting_equity"])

    strategy_curve = build_daily_equity_curve(
        trades,
        bars,
        starting_equity=starting_equity,
    )

    matched_curve = build_exposure_matched_curve(
        bars,
        BENCHMARK_SYMBOL,
        strategy_curve,
        starting_equity=starting_equity,
        risk_free_rate=risk_free_rate,
    )

    strategy_performance = calculate_performance(
        strategy_curve,
        risk_free_rate=risk_free_rate,
    )
    matched_performance = calculate_performance(
        matched_curve,
        risk_free_rate=risk_free_rate,
    )

    profit_factor = float("nan")

    if want_profit_factor:
        profit_factor = calculate_metrics(
            trades,
            starting_equity=starting_equity,
        )["profit_factor"]

    return {
        "strategy": strategy_performance["sharpe"],
        "exposure_matched": matched_performance["sharpe"],
        "average_exposure": strategy_performance["average_exposure"],
        "profit_factor": profit_factor,
        "trades": len(trades),
    }



def evaluate_sweep(
    signal_frames: dict[int, "pd.DataFrame"],
    bars,
    *,
    execution: dict,
    risk_free_rate: float,
    rng=None,
    permute_method: str | None = None,
) -> dict:
    """
    Run the entire 36-configuration selection procedure and report the
    best result on each metric.

    This is the null that actually matters. Permuting signals while
    holding parameters fixed asks "given these parameters, is the timing
    special?" -- but the parameters were themselves chosen as the best of
    36 on this same data. Correcting for that requires re-running the
    whole selection inside every permutation.

    When ``rng`` is supplied each window's signal set is permuted
    independently. That mirrors the real procedure, which had four
    genuinely different signal generators to choose from: under the null
    it gets four independently worthless ones.

    Two selection rules are reported, because they answer different
    questions:

    ``best_<metric>``      the highest value any of the 36 could reach.
                           The strict data-snooping bound.
    ``<metric>_at_pf_best`` the metric of whichever configuration won on
                           profit factor. Faithful to how the candidate
                           was actually picked, but a weaker correction.
    """

    best_strategy = -math.inf
    best_matched = -math.inf

    best_profit_factor = -math.inf
    strategy_at_pf = float("nan")
    matched_at_pf = float("nan")
    winning_config: dict | None = None

    evaluated = 0

    for window, base_frame in signal_frames.items():
        if rng is not None:
            frame = permute_signals(
                base_frame,
                rng=rng,
                method=permute_method or "circular_shift",
            )
        else:
            frame = base_frame

        for holding_days, stop_loss_pct in (
            sweep_grid.runner_combinations()
        ):
            result = evaluate_schedule(
                frame,
                bars,
                execution=execution,
                params={
                    "holding_days": holding_days,
                    "stop_loss_pct": stop_loss_pct,
                },
                risk_free_rate=risk_free_rate,
                want_profit_factor=True,
            )

            evaluated += 1

            if result["trades"] == 0:
                continue

            if result["strategy"] == result["strategy"]:
                best_strategy = max(best_strategy, result["strategy"])

            if result["exposure_matched"] == result["exposure_matched"]:
                best_matched = max(best_matched, result["exposure_matched"])

            profit_factor = result["profit_factor"]

            if (
                profit_factor == profit_factor
                and math.isfinite(profit_factor)
                and profit_factor > best_profit_factor
            ):
                best_profit_factor = profit_factor
                strategy_at_pf = result["strategy"]
                matched_at_pf = result["exposure_matched"]
                winning_config = {
                    "sma_window": window,
                    "holding_days": holding_days,
                    "stop_loss_pct": stop_loss_pct,
                }

    return {
        "best_strategy": (
            best_strategy if math.isfinite(best_strategy) else float("nan")
        ),
        "best_exposure_matched": (
            best_matched if math.isfinite(best_matched) else float("nan")
        ),
        "strategy_at_pf_best": strategy_at_pf,
        "exposure_matched_at_pf_best": matched_at_pf,
        "best_profit_factor": (
            best_profit_factor
            if math.isfinite(best_profit_factor)
            else float("nan")
        ),
        "winning_config": winning_config,
        "configurations_evaluated": evaluated,
    }
