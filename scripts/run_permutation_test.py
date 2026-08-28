"""
Test whether a strategy's in/out schedule has real timing skill.

The performance report showed that holding plain SPY at sma_crossover's
own daily exposure produced a Sharpe of 0.83 against the index's 0.49.
That is either evidence that the trend filter picks good times to be
invested, or it is what any schedule with the same in/out frequency
would have produced over a window containing one bear market.

This script decides which. It permutes the signal timing -- preserving
the number of signals per symbol and, by default, their clustering --
re-runs the entire backtest, and rebuilds the exposure-matched curve
from each permuted schedule. The result is a null distribution for the
Sharpe of a randomly-timed schedule of the same shape.

Two metrics are tested per permutation:

  strategy          the strategy's own Sharpe, including symbol
                    selection, stops and forced exits
  exposure_matched  SPY held at the strategy's exposure schedule,
                    isolating timing skill from everything else

Note on what exposure_matched actually measures. A signal at bar i
produces an entry at the open of bar i+1, exposure from session i+1, and
-- because exposure is lagged one session to avoid peeking -- a return
contribution spanning close[i+1] to close[i+2]. So this metric asks
whether the schedule identifies good multi-day periods to be invested,
not whether the signal predicts the very next bar. A strategy with real
one-bar-ahead prediction and nothing else would score poorly here.

Usage:
    python scripts/run_permutation_test.py --split development \\
        --permutations 200 --risk-free-rate 0.03
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from trading_lab.backtest.benchmark import build_exposure_matched_curve
from trading_lab.backtest.equity import build_daily_equity_curve
from trading_lab.backtest.performance import calculate_performance
from trading_lab.backtest.runner import run_long_signal_backtest
from trading_lab.data.alpaca import get_daily_bars
from trading_lab.data.split import (
    get_development_data,
    get_holdout_data,
    get_validation_data,
)
from trading_lab.strategies.sma_crossover import SMACrossoverStrategy
from trading_lab.validation.significance import (
    permute_signals,
    summarize_against_null,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PATH = PROJECT_ROOT / "config" / "frozen_candidate.yaml"
REPORT_DIR = PROJECT_ROOT / "reports" / "significance"

BENCHMARK_SYMBOL = "SPY"


def load_candidate() -> dict:
    with CANDIDATE_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def select_split(df, split: str, candidate: dict):
    if split == "development":
        return get_development_data(df)

    if split == "validation":
        return get_validation_data(df)

    if split == "holdout":
        frozen = bool(
            candidate.get("holdout", {}).get("parameters_frozen", False)
        )
        return get_holdout_data(df, parameters_frozen=frozen)

    raise ValueError(f"Unknown split: {split!r}")


def evaluate_schedule(
    signal_df,
    bars,
    *,
    execution: dict,
    params: dict,
    risk_free_rate: float,
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

    return {
        "strategy": strategy_performance["sharpe"],
        "exposure_matched": matched_performance["sharpe"],
        "average_exposure": strategy_performance["average_exposure"],
        "trades": len(trades),
    }


def _report(summary: dict, *, title: str) -> None:
    print()
    print(title)
    print("-" * 64)
    print(f"{'observed Sharpe':<32}{summary['observed']:>10.3f}")
    print(f"{'null mean':<32}{summary['null_mean']:>10.3f}")
    print(f"{'null median':<32}{summary['null_median']:>10.3f}")
    print(f"{'null 95th percentile':<32}{summary['null_p95']:>10.3f}")
    print(f"{'null maximum':<32}{summary['null_max']:>10.3f}")
    print(
        f"{'observed sits at percentile':<32}"
        f"{summary['percentile_of_null']:>9.1f}%"
    )
    print(f"{'p-value':<32}{summary['p_value']:>10.4f}")
    print(
        f"{'significant at 0.05':<32}"
        f"{'YES' if summary['significant_at_05'] else 'NO':>10}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="development",
        choices=["development", "validation", "holdout"],
    )
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--method",
        default="circular_shift",
        choices=["circular_shift", "shuffle"],
        help=(
            "circular_shift preserves signal clustering and is the "
            "harder, more honest null. shuffle destroys clustering too, "
            "which weakens the null and flatters the strategy."
        ),
    )
    args = parser.parse_args()

    candidate = load_candidate()
    symbols = list(candidate["universe"]["symbols"])
    params = candidate["parameters"]
    execution = candidate["execution"]

    if BENCHMARK_SYMBOL not in symbols:
        symbols = [*symbols, BENCHMARK_SYMBOL]

    print(f"Loading bars for {symbols}...")

    df = get_daily_bars(
        symbols=symbols,
        start="2017-01-01",
        end="2025-12-31",
    )

    split_df = select_split(df, args.split, candidate)

    strategy = SMACrossoverStrategy(window=params["sma_window"])
    signal_df = strategy.generate_signals(split_df)

    observed = evaluate_schedule(
        signal_df,
        split_df,
        execution=execution,
        params=params,
        risk_free_rate=args.risk_free_rate,
    )

    print()
    print(f"PERMUTATION TEST - {args.split.upper()}")
    print("=" * 64)
    print(f"null method:        {args.method}")
    print(f"permutations:       {args.permutations}")
    print(f"risk-free rate:     {args.risk_free_rate:.2%}")
    print(f"observed trades:    {observed['trades']}")
    print(f"observed exposure:  {observed['average_exposure']:.2%}")

    rng = np.random.default_rng(args.seed)

    null_strategy: list[float] = []
    null_matched: list[float] = []
    null_exposure: list[float] = []

    started = time.time()

    for index in range(args.permutations):
        permuted = permute_signals(
            signal_df,
            rng=rng,
            method=args.method,
        )

        result = evaluate_schedule(
            permuted,
            split_df,
            execution=execution,
            params=params,
            risk_free_rate=args.risk_free_rate,
        )

        null_strategy.append(result["strategy"])
        null_matched.append(result["exposure_matched"])
        null_exposure.append(result["average_exposure"])

        done = index + 1

        if done % 25 == 0 or done == args.permutations:
            elapsed = time.time() - started
            rate = elapsed / done
            remaining = rate * (args.permutations - done)
            print(
                f"  {done}/{args.permutations} permutations"
                f"  ({elapsed:.0f}s elapsed, ~{remaining:.0f}s left)"
            )

    strategy_summary = summarize_against_null(
        observed["strategy"],
        null_strategy,
        label="strategy_sharpe",
    )
    matched_summary = summarize_against_null(
        observed["exposure_matched"],
        null_matched,
        label="exposure_matched_sharpe",
    )

    _report(
        strategy_summary,
        title="STRATEGY SHARPE vs randomly-timed schedules",
    )
    _report(
        matched_summary,
        title="EXPOSURE-MATCHED SHARPE vs randomly-timed schedules",
    )

    finite_exposure = [x for x in null_exposure if x == x]
    print()
    print("NULL EXPOSURE (confound check)")
    print("-" * 64)
    print(f"{'observed average exposure':<32}"
          f"{observed['average_exposure']:>9.2%}")
    if finite_exposure:
        print(f"{'null mean average exposure':<32}"
              f"{float(np.mean(finite_exposure)):>9.2%}")
        print(
            "If these differ materially the comparison is contaminated: "
            "the null is not holding the same amount of risk."
        )

    print()
    print("VERDICT")
    print("=" * 64)

    if matched_summary["significant_at_05"]:
        print(
            "The exposure schedule beats randomly-timed schedules of the "
            "same shape. Timing skill is not ruled out."
        )
        print(
            "Still not evidence: this schedule came from parameters "
            "selected on this same data. Correct for the 36-combination "
            "sweep before believing it."
        )
    else:
        print(
            "The exposure schedule is indistinguishable from randomly-"
            "timed schedules with the same in/out frequency."
        )
        print(
            "The apparent Sharpe advantage is a product of being out of "
            "the market, not of choosing when. There is no timing skill "
            "here to build on."
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = REPORT_DIR / f"{stamp}_{args.split}_permutation.json"

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "split": args.split,
                "generated_utc": stamp,
                "method": args.method,
                "permutations": args.permutations,
                "seed": args.seed,
                "risk_free_rate": args.risk_free_rate,
                "candidate": candidate,
                "observed": observed,
                "strategy_summary": strategy_summary,
                "exposure_matched_summary": matched_summary,
                "null_strategy_sharpe": null_strategy,
                "null_exposure_matched_sharpe": null_matched,
            },
            file,
            indent=2,
            default=str,
        )

    print()
    print(f"report saved: {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
