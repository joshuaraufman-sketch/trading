from __future__ import annotations

import pandas as pd


INPUT_PATH = "experiments/parameter_sweep_summary.csv"


def main():
    df = pd.read_csv(INPUT_PATH)

    passed = df[df["passed"] == True].copy()

    print("PARAMETER SWEEP SUMMARY")
    print("-----------------------")
    print(f"total experiments: {len(df)}")
    print(f"passed research rules: {len(passed)}")

    print()
    print("BY SMA WINDOW")
    print("-------------")

    sma_summary = (
        df.groupby("sma_window")
        .agg(
            experiments=("profit_factor", "size"),
            average_profit_factor=("profit_factor", "mean"),
            median_profit_factor=("profit_factor", "median"),
            average_r=("average_r", "mean"),
            average_drawdown=("max_drawdown_pct", "mean"),
            pass_rate=("passed", "mean"),
        )
        .reset_index()
    )

    print(
        sma_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("BY HOLDING PERIOD")
    print("-----------------")

    holding_summary = (
        df.groupby("holding_days")
        .agg(
            experiments=("profit_factor", "size"),
            average_profit_factor=("profit_factor", "mean"),
            average_r=("average_r", "mean"),
            pass_rate=("passed", "mean"),
        )
        .reset_index()
    )

    print(
        holding_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("BY STOP LOSS")
    print("------------")

    stop_summary = (
        df.groupby("stop_loss_pct")
        .agg(
            experiments=("profit_factor", "size"),
            average_profit_factor=("profit_factor", "mean"),
            average_r=("average_r", "mean"),
            pass_rate=("passed", "mean"),
        )
        .reset_index()
    )

    print(
        stop_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("SMA=10 NEIGHBORHOOD")
    print("-------------------")

    neighborhood = (
        df[df["sma_window"] == 10]
        .sort_values(
            [
                "holding_days",
                "stop_loss_pct",
            ]
        )
    )

    columns = [
        "sma_window",
        "holding_days",
        "stop_loss_pct",
        "trade_count",
        "profit_factor",
        "average_r",
        "max_drawdown_pct",
        "passed",
    ]

    print(
        neighborhood[columns]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


if __name__ == "__main__":
    main()