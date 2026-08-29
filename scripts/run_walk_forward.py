"""
Genuine walk-forward validation of the parameter sweep.

The previous version of this script applied one already-selected
parameter set to four fixed calendar windows, three of which were the
same data those parameters were fitted on. It could not detect
overfitting because there was nothing out-of-sample about it.

This version re-runs the entire sweep inside each training window and
applies the winner to the next window, which the selection never saw.
The stitched out-of-sample curve is then put through the same four gates
the performance report uses.

Usage:
    python scripts/run_walk_forward.py --split development \\
        --train-sessions 504 --test-sessions 126 --risk-free-rate 0.03
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from trading_lab.backtest.benchmark import (
    build_benchmark_curve,
    build_static_blend_curve,
    compare_to_benchmark,
)
from trading_lab.backtest.equity import (
    build_daily_equity_curve,
    normalize_session_dates,
)
from trading_lab.backtest.metrics import calculate_metrics
from trading_lab.backtest.performance import calculate_performance
from trading_lab.backtest.runner import run_long_signal_backtest
from trading_lab.data.alpaca import get_daily_bars
from trading_lab.data.split import (
    get_development_data,
    get_holdout_data,
    get_validation_data,
)
from trading_lab.strategies.sma_crossover import SMACrossoverStrategy
from trading_lab.validation import sweep_grid
from trading_lab.validation.walk_forward import (
    parameter_stability,
    run_walk_forward,
    stitch_out_of_sample,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PATH = PROJECT_ROOT / "config" / "frozen_candidate.yaml"
REPORT_DIR = PROJECT_ROOT / "reports" / "walk_forward"

BENCHMARK_SYMBOL = "SPY"

SELECTION_METRICS = {
    "sharpe": ("performance", "sharpe"),
    "profit_factor": ("trade", "profit_factor"),
    "average_r": ("trade", "average_r"),
    "expectancy": ("trade", "expectancy"),
}


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


def score_window(
    signal_frame,
    bars,
    *,
    holding_days: int,
    stop_loss_pct: float,
    execution: dict,
    risk_free_rate: float,
    metric: str,
):
    """Backtest one configuration over one window and score it."""

    starting_equity = float(execution["starting_equity"])

    trades = run_long_signal_backtest(
        signal_frame,
        starting_equity=starting_equity,
        risk_pct=float(execution["risk_pct"]),
        stop_loss_pct=stop_loss_pct,
        holding_days=holding_days,
        slippage_bps=float(execution["slippage_bps"]),
        fee_per_share=float(execution["fee_per_share"]),
    )

    if not trades:
        return float("-inf"), None, 0

    curve = build_daily_equity_curve(
        trades, bars, starting_equity=starting_equity
    )

    source, key = SELECTION_METRICS[metric]

    if source == "performance":
        score = calculate_performance(
            curve, risk_free_rate=risk_free_rate
        )[key]
    else:
        score = calculate_metrics(
            trades, starting_equity=starting_equity
        )[key]

    if score != score:  # NaN
        return float("-inf"), None, len(trades)

    return float(score), curve, len(trades)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", default="development",
        choices=["development", "validation", "holdout"],
    )
    parser.add_argument(
        "--train-sessions", type=int, default=504,
        help="Training window length in sessions (504 ~ 2 years).",
    )
    parser.add_argument(
        "--test-sessions", type=int, default=126,
        help="Out-of-sample window length in sessions (126 ~ 6 months).",
    )
    parser.add_argument(
        "--mode", default="anchored", choices=["anchored", "rolling"],
    )
    parser.add_argument(
        "--select-on", default="sharpe", choices=sorted(SELECTION_METRICS),
        help=(
            "Metric used to pick the winning configuration in each "
            "training window. Make this explicit: the original sweep "
            "ranked on profit factor but the candidate was chosen on "
            "net profit, an undocumented extra degree of freedom."
        ),
    )
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    args = parser.parse_args()

    candidate = load_candidate()
    symbols = list(candidate["universe"]["symbols"])
    execution = candidate["execution"]

    if BENCHMARK_SYMBOL not in symbols:
        symbols = [*symbols, BENCHMARK_SYMBOL]

    print(f"Loading bars for {symbols}...")

    df = get_daily_bars(
        symbols=symbols, start="2017-01-01", end="2025-12-31"
    )
    split_df = select_split(df, args.split, candidate)

    # Signals are generated once over the whole split, then sliced.
    # Generating inside each window would leave the first `window` bars
    # of every fold without an SMA and silently drop early signals.
    # This does not leak: the SMA is strictly backward looking.
    signal_frames = {
        window: SMACrossoverStrategy(window=window).generate_signals(
            split_df
        )
        for window in sweep_grid.SMA_WINDOWS
    }

    sessions = pd.Index(
        normalize_session_dates(split_df["timestamp"]).unique()
    ).sort_values()

    session_column = {
        window: normalize_session_dates(frame["timestamp"])
        for window, frame in signal_frames.items()
    }

    def slice_window(window, start, end):
        mask = (session_column[window] >= start) & (
            session_column[window] <= end
        )
        return signal_frames[window].loc[mask.to_numpy()].copy()

    def select(fold):
        best_score = float("-inf")
        best_config = None

        for window in sweep_grid.SMA_WINDOWS:
            frame = slice_window(window, fold.train_start, fold.train_end)

            if frame.empty:
                continue

            for holding_days, stop_loss_pct in (
                sweep_grid.runner_combinations()
            ):
                score, _, _ = score_window(
                    frame, frame,
                    holding_days=holding_days,
                    stop_loss_pct=stop_loss_pct,
                    execution=execution,
                    risk_free_rate=args.risk_free_rate,
                    metric=args.select_on,
                )

                if score > best_score:
                    best_score = score
                    best_config = {
                        "sma_window": window,
                        "holding_days": holding_days,
                        "stop_loss_pct": stop_loss_pct,
                    }

        if best_config is None:
            return None, float("nan"), {"reason": "no viable config"}

        return best_config, best_score, {}

    def evaluate(fold, config):
        frame = slice_window(
            config["sma_window"], fold.test_start, fold.test_end
        )

        _, curve, trades = score_window(
            frame, frame,
            holding_days=config["holding_days"],
            stop_loss_pct=config["stop_loss_pct"],
            execution=execution,
            risk_free_rate=args.risk_free_rate,
            metric=args.select_on,
        )

        if curve is None:
            index = sessions[
                (sessions >= fold.test_start) & (sessions <= fold.test_end)
            ]
            return pd.Series(0.0, index=index), 0, {"no_trades": True}

        returns = curve["equity"].pct_change().fillna(0.0)
        return returns, trades, {}

    print()
    print(f"WALK-FORWARD - {args.split.upper()}")
    print("=" * 78)
    print(f"mode:            {args.mode}")
    print(f"train window:    {args.train_sessions} sessions")
    print(f"test window:     {args.test_sessions} sessions")
    print(f"selection on:    {args.select_on}")
    print(f"grid:            {sweep_grid.grid_size()} configurations "
          f"re-swept per fold")
    print()
    print(f"{'fold':<6}{'train end':<13}{'test window':<26}"
          f"{'selected':<28}{'trades':>7}")
    print("-" * 78)

    def report(result):
        fold = result.fold

        if result.selected is None:
            chosen = "none viable"
        else:
            chosen = (
                f"w{result.selected['sma_window']} "
                f"h{result.selected['holding_days']} "
                f"s{result.selected['stop_loss_pct']}"
            )

        print(
            f"{fold.index:<6}{str(fold.train_end.date()):<13}"
            f"{str(fold.test_start.date()) + ' to ' + str(fold.test_end.date()):<26}"
            f"{chosen:<28}{result.test_trades:>7}"
        )

    results = run_walk_forward(
        sessions,
        select=select,
        evaluate=evaluate,
        train_sessions=args.train_sessions,
        test_sessions=args.test_sessions,
        mode=args.mode,
        on_fold=report,
    )

    starting_equity = float(execution["starting_equity"])
    oos = stitch_out_of_sample(results, starting_equity=starting_equity)

    oos_curve = oos.copy()
    oos_curve["open_positions"] = 1
    oos_curve["exposure_pct"] = float("nan")

    performance = calculate_performance(
        oos_curve, risk_free_rate=args.risk_free_rate
    )

    benchmark_curve = build_benchmark_curve(
        split_df, BENCHMARK_SYMBOL,
        starting_equity=starting_equity, sessions=oos.index,
    )
    comparison = compare_to_benchmark(
        oos_curve, benchmark_curve, risk_free_rate=args.risk_free_rate
    )

    print()
    print("OUT-OF-SAMPLE PERFORMANCE (stitched, never selected on)")
    print("=" * 78)
    print(f"{'':<26}{'WALK-FORWARD':>14}{'BUY & HOLD':>14}")
    print("-" * 78)

    bench = comparison["benchmark"]

    for label, key, pct in [
        ("total return", "total_return", True),
        ("CAGR", "cagr", True),
        ("annual volatility", "annual_volatility", True),
        ("Sharpe", "sharpe", False),
        ("Sortino", "sortino", False),
        ("max drawdown", "max_drawdown_pct", True),
        ("Calmar", "calmar", False),
    ]:
        fmt = (lambda v: f"{v:.2%}") if pct else (lambda v: f"{v:.3f}")
        print(f"{label:<26}{fmt(performance[key]):>14}"
              f"{fmt(bench[key]):>14}")

    print()
    print(f"{'beta':<26}{comparison['beta']:>14.3f}")
    print(f"{'annualized alpha':<26}"
          f"{comparison['annual_alpha']:>13.2%}")
    print(f"{'  t-statistic':<26}{comparison['alpha_t_stat']:>14.2f}")

    stability = parameter_stability(results)

    print()
    print("PARAMETER STABILITY")
    print("=" * 78)
    print("Configurations that jump every retraining window indicate an")
    print("unstable effect, even when the out-of-sample curve looks fine.")
    print()

    for key in ("sma_window", "holding_days", "stop_loss_pct"):
        entry = stability.get(key)

        if not entry:
            continue

        print(f"{key:<18}distinct {entry['distinct']}"
              f"   changed {entry['changed_fraction']:.0%} of folds"
              f"   values {entry['values']}")

    print()
    print("VERDICT")
    print("=" * 78)

    gates = {
        "out-of-sample Sharpe beats buy-and-hold":
            performance["sharpe"] > bench["sharpe"],
        "out-of-sample alpha significant":
            comparison["alpha_significant_at_05"],
        "parameters stable (sma_window changed < 50% of folds)":
            stability.get("sma_window", {}).get("changed_fraction", 1.0)
            < 0.5,
    }

    for label, passed in gates.items():
        print(f"{label:<56}{'PASS' if passed else 'FAIL'}")

    if not all(gates.values()):
        print()
        print(
            "This is the first honest out-of-sample measurement in the "
            "project. Every earlier 'walk-forward' number was in-sample."
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = REPORT_DIR / f"{stamp}_{args.split}_walk_forward.json"

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "split": args.split,
                "generated_utc": stamp,
                "mode": args.mode,
                "train_sessions": args.train_sessions,
                "test_sessions": args.test_sessions,
                "select_on": args.select_on,
                "risk_free_rate": args.risk_free_rate,
                "folds": [
                    {
                        "index": r.fold.index,
                        "train_start": r.fold.train_start,
                        "train_end": r.fold.train_end,
                        "test_start": r.fold.test_start,
                        "test_end": r.fold.test_end,
                        "selected": r.selected,
                        "selection_score": r.selection_score,
                        "test_trades": r.test_trades,
                    }
                    for r in results
                ],
                "out_of_sample": performance,
                "comparison": comparison,
                "parameter_stability": stability,
                "gates": gates,
            },
            file, indent=2, default=str,
        )

    curve_path = REPORT_DIR / f"{stamp}_{args.split}_oos_curve.csv"
    oos.to_csv(curve_path)

    print()
    print(f"report saved: {path.relative_to(PROJECT_ROOT)}")
    print(f"curve saved:  {curve_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
