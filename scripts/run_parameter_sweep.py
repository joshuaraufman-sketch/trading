from __future__ import annotations

from itertools import product

import pandas as pd

from trading_lab.backtest.metrics import calculate_metrics
from trading_lab.backtest.runner import run_long_signal_backtest
from trading_lab.data.alpaca import get_daily_bars
from trading_lab.data.split import get_development_data
from trading_lab.strategies.sma_crossover import SMACrossoverStrategy
from trading_lab.validation.experiment_log import save_experiment
from trading_lab.validation.research_rules import evaluate_research_rules


SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]

STARTING_EQUITY = 100_000
RISK_PCT = 0.005
SLIPPAGE_BPS = 5
FEE_PER_SHARE = 0.005

SMA_WINDOWS = [10, 20, 40, 60]
HOLDING_DAYS_OPTIONS = [3, 5, 10]
STOP_LOSS_OPTIONS = [0.015, 0.02, 0.03]


def main():
    print("Loading development data...")

    df = get_daily_bars(
        symbols=SYMBOLS,
        start="2017-01-01",
        end="2025-12-31",
    )

    df = get_development_data(df)

    results = []

    combinations = list(
        product(
            SMA_WINDOWS,
            HOLDING_DAYS_OPTIONS,
            STOP_LOSS_OPTIONS,
        )
    )

    print(f"Running {len(combinations)} experiments...")

    for number, (
        sma_window,
        holding_days,
        stop_loss_pct,
    ) in enumerate(combinations, start=1):

        print(
            f"[{number}/{len(combinations)}] "
            f"SMA={sma_window}, "
            f"hold={holding_days}, "
            f"stop={stop_loss_pct:.3f}"
        )

        strategy = SMACrossoverStrategy(
            window=sma_window,
        )

        signal_df = strategy.generate_signals(df)

        trades = run_long_signal_backtest(
            signal_df,
            starting_equity=STARTING_EQUITY,
            risk_pct=RISK_PCT,
            stop_loss_pct=stop_loss_pct,
            holding_days=holding_days,
            slippage_bps=SLIPPAGE_BPS,
            fee_per_share=FEE_PER_SHARE,
        )

        metrics = calculate_metrics(
            trades,
            starting_equity=STARTING_EQUITY,
        )

        evaluation = evaluate_research_rules(
            metrics,
            minimum_trades=200,
            minimum_profit_factor=1.25,
            maximum_drawdown_pct=0.20,
        )

        parameters = {
            "symbols": SYMBOLS,
            "sma_window": sma_window,
            "holding_days": holding_days,
            "stop_loss_pct": stop_loss_pct,
            "starting_equity": STARTING_EQUITY,
            "risk_pct": RISK_PCT,
            "slippage_bps": SLIPPAGE_BPS,
            "fee_per_share": FEE_PER_SHARE,
        }

        path = save_experiment(
            strategy_name=strategy.name,
            parameters=parameters,
            metrics=metrics,
            evaluation=evaluation,
            dataset="development",
        )

        results.append(
            {
                "sma_window": sma_window,
                "holding_days": holding_days,
                "stop_loss_pct": stop_loss_pct,
                "trade_count": metrics["trade_count"],
                "profit_factor": metrics["profit_factor"],
                "expectancy": metrics["expectancy"],
                "average_r": metrics["average_r"],
                "max_drawdown_pct": metrics["max_drawdown_pct"],
                "net_profit": metrics["net_profit"],
                "passed": evaluation["passed"],
                "experiment_file": str(path),
            }
        )

    results_df = pd.DataFrame(results)

    ranked = results_df.sort_values(
        by=[
            "passed",
            "profit_factor",
            "average_r",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    print()
    print("TOP RESULTS")
    print("-----------")

    columns = [
        "sma_window",
        "holding_days",
        "stop_loss_pct",
        "trade_count",
        "profit_factor",
        "average_r",
        "max_drawdown_pct",
        "net_profit",
        "passed",
    ]

    print(
        ranked[columns]
        .head(10)
        .to_string(index=False)
    )

    output_path = (
        "experiments/parameter_sweep_summary.csv"
    )

    ranked.to_csv(
        output_path,
        index=False,
    )

    print()
    print("SUMMARY SAVED")
    print("-------------")
    print(output_path)


if __name__ == "__main__":
    main()