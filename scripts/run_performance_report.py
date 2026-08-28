"""
Answer the only question that currently matters: does this strategy beat
buying and holding the index, per unit of risk?

Reads config/frozen_candidate.yaml rather than hardcoding parameters, so
the frozen candidate is a single source of truth instead of a comment.

Usage:
    python scripts/run_performance_report.py --split development
    python scripts/run_performance_report.py --split validation
    python scripts/run_performance_report.py --split holdout   # gated
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from trading_lab.backtest.benchmark import (
    build_benchmark_curve,
    compare_to_benchmark,
)
from trading_lab.backtest.equity import build_daily_equity_curve
from trading_lab.backtest.metrics import calculate_metrics
from trading_lab.backtest.runner import run_long_signal_backtest
from trading_lab.data.alpaca import get_daily_bars
from trading_lab.data.split import (
    get_development_data,
    get_holdout_data,
    get_validation_data,
)
from trading_lab.strategies.sma_crossover import SMACrossoverStrategy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PATH = PROJECT_ROOT / "config" / "frozen_candidate.yaml"
REPORT_DIR = PROJECT_ROOT / "reports" / "performance"

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
        # Raises PermissionError when parameters are not frozen.
        return get_holdout_data(df, parameters_frozen=frozen)

    raise ValueError(f"Unknown split: {split!r}")


def _pct(value: float) -> str:
    return f"{value * 100:>8.2f}%"


def _num(value: float) -> str:
    return f"{value:>9.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="development",
        choices=["development", "validation", "holdout"],
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.0,
        help=(
            "Annual risk-free rate. Leaving this at zero materially "
            "inflates Sharpe on any window covering 2022-2024."
        ),
    )
    args = parser.parse_args()

    candidate = load_candidate()

    symbols = candidate["universe"]["symbols"]
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

    starting_equity = float(execution["starting_equity"])

    trades = run_long_signal_backtest(
        signal_df,
        starting_equity=starting_equity,
        risk_pct=float(execution["risk_pct"]),
        stop_loss_pct=float(params["stop_loss_pct"]),
        holding_days=int(params["holding_days"]),
        slippage_bps=float(execution["slippage_bps"]),
        fee_per_share=float(execution["fee_per_share"]),
    )

    if not trades:
        print("No trades generated on this split. Nothing to report.")
        return

    trade_metrics = calculate_metrics(
        trades,
        starting_equity=starting_equity,
    )

    strategy_curve = build_daily_equity_curve(
        trades,
        split_df,
        starting_equity=starting_equity,
    )

    benchmark_curve = build_benchmark_curve(
        split_df,
        BENCHMARK_SYMBOL,
        starting_equity=starting_equity,
        sessions=strategy_curve.index,
    )

    comparison = compare_to_benchmark(
        strategy_curve,
        benchmark_curve,
        risk_free_rate=args.risk_free_rate,
    )

    strat = comparison["strategy"]
    bench = comparison["benchmark"]

    print()
    print(f"PERFORMANCE REPORT - {args.split.upper()}")
    print("=" * 58)
    print(
        f"{strat['start_session'].date()} to "
        f"{strat['end_session'].date()}  "
        f"({strat['sessions']} sessions, {strat['years']:.2f} years)"
    )
    print(f"risk-free rate assumed: {args.risk_free_rate:.2%}")
    print()

    print(f"{'':<28}{'STRATEGY':>13}{'BUY & HOLD':>15}")
    print("-" * 58)

    rows = [
        ("total return", "total_return", _pct),
        ("CAGR", "cagr", _pct),
        ("annual volatility", "annual_volatility", _pct),
        ("Sharpe", "sharpe", _num),
        ("Sortino", "sortino", _num),
        ("max drawdown", "max_drawdown_pct", _pct),
        ("max DD duration (days)", "max_drawdown_duration_days", _num),
        ("Calmar", "calmar", _num),
        ("time in market", "time_in_market", _pct),
        ("average exposure", "average_exposure", _pct),
        ("exposure-adj. CAGR", "exposure_adjusted_cagr", _pct),
    ]

    for label, key, fmt in rows:
        print(
            f"{label:<28}{fmt(strat[key]):>13}{fmt(bench[key]):>15}"
        )

    print()
    print("RELATIVE")
    print("-" * 58)
    print(f"{'beta to ' + BENCHMARK_SYMBOL:<28}{comparison['beta']:>13.3f}")
    print(f"{'annualized alpha':<28}{comparison['annual_alpha']:>12.2%}")
    print(f"{'correlation':<28}{comparison['correlation']:>13.3f}")
    print(f"{'tracking error':<28}{comparison['tracking_error']:>12.2%}")
    print(
        f"{'information ratio':<28}"
        f"{comparison['information_ratio']:>13.3f}"
    )
    print(
        f"{'excess CAGR':<28}{comparison['excess_cagr']:>12.2%}"
    )

    print()
    print("TRADE-LEVEL (for reference, not risk adjusted)")
    print("-" * 58)
    print(f"{'trades':<28}{trade_metrics['trade_count']:>13}")
    print(f"{'win rate':<28}{trade_metrics['win_rate']:>12.2%}")
    print(f"{'profit factor':<28}{trade_metrics['profit_factor']:>13.3f}")
    print(f"{'average R':<28}{trade_metrics['average_r']:>13.3f}")
    print(
        f"{'trade-ordered max DD':<28}"
        f"{trade_metrics['max_drawdown_pct']:>12.2%}"
    )
    print(
        f"{'calendar max DD':<28}"
        f"{strat['max_drawdown_pct']:>12.2%}"
        "   <-- the honest one"
    )

    print()
    print("VERDICT")
    print("-" * 58)

    beats_sharpe = comparison["beats_benchmark_on_sharpe"]
    positive_alpha = comparison["annual_alpha"] > 0

    print(
        f"beats buy-and-hold on Sharpe: "
        f"{'YES' if beats_sharpe else 'NO'}"
    )
    print(
        f"positive alpha after beta:    "
        f"{'YES' if positive_alpha else 'NO'}"
    )

    if comparison["beta"] > 0.6 and abs(comparison["annual_alpha"]) < 0.02:
        print()
        print(
            "High beta with negligible alpha. This is index exposure, "
            "not edge."
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report_path = REPORT_DIR / f"{stamp}_{args.split}_performance.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "split": args.split,
                "generated_utc": stamp,
                "risk_free_rate": args.risk_free_rate,
                "candidate": candidate,
                "trade_metrics": trade_metrics,
                "comparison": comparison,
            },
            file,
            indent=2,
            default=str,
        )

    curve_path = REPORT_DIR / f"{stamp}_{args.split}_equity_curve.csv"
    strategy_curve.join(
        benchmark_curve["equity"].rename("benchmark_equity")
    ).to_csv(curve_path)

    print()
    print(f"report saved: {report_path.relative_to(PROJECT_ROOT)}")
    print(f"curve saved:  {curve_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
