from __future__ import annotations

import pandas as pd

from trading_lab.backtest.metrics import calculate_metrics
from trading_lab.backtest.runner import run_long_signal_backtest
from trading_lab.data.alpaca import get_daily_bars
from trading_lab.strategies.sma_crossover import SMACrossoverStrategy


SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]

STARTING_EQUITY = 100_000
RISK_PCT = 0.005
SMA_WINDOW = 10
HOLDING_DAYS = 10
STOP_LOSS_PCT = 0.02
SLIPPAGE_BPS = 5
FEE_PER_SHARE = 0.005


PERIODS = [
    ("2017-2018", "2017-01-01", "2018-12-31"),
    ("2019-2020", "2019-01-01", "2020-12-31"),
    ("2021-2022", "2021-01-01", "2022-12-31"),
    ("2023-2024", "2023-01-01", "2024-12-31"),
]


def main():
    df = get_daily_bars(
        symbols=SYMBOLS,
        start="2017-01-01",
        end="2025-12-31",
    )

    strategy = SMACrossoverStrategy(
        window=SMA_WINDOW,
    )

    results = []

    for name, start, end in PERIODS:
        period_df = df[
            (
                pd.to_datetime(df["timestamp"], utc=True)
                >= pd.Timestamp(start, tz="UTC")
            )
            &
            (
                pd.to_datetime(df["timestamp"], utc=True)
                < pd.Timestamp(end, tz="UTC")
                + pd.Timedelta(days=1)
            )
        ].copy()

        signal_df = strategy.generate_signals(
            period_df
        )

        trades = run_long_signal_backtest(
            signal_df,
            starting_equity=STARTING_EQUITY,
            risk_pct=RISK_PCT,
            stop_loss_pct=STOP_LOSS_PCT,
            holding_days=HOLDING_DAYS,
            slippage_bps=SLIPPAGE_BPS,
            fee_per_share=FEE_PER_SHARE,
        )

        metrics = calculate_metrics(
            trades,
            starting_equity=STARTING_EQUITY,
        )

        results.append(
            {
                "period": name,
                "trade_count": metrics["trade_count"],
                "profit_factor": metrics["profit_factor"],
                "expectancy": metrics["expectancy"],
                "average_r": metrics["average_r"],
                "max_drawdown_pct": metrics["max_drawdown_pct"],
                "net_profit": metrics["net_profit"],
            }
        )

    results_df = pd.DataFrame(results)

    print("WALK-FORWARD RESULTS")
    print("--------------------")

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    output_path = (
        "experiments/walk_forward_summary.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print("SUMMARY SAVED")
    print("-------------")
    print(output_path)


if __name__ == "__main__":
    main()