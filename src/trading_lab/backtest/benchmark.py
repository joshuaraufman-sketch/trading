from __future__ import annotations

import pandas as pd


def calculate_buy_and_hold_return(
    df: pd.DataFrame,
    symbol: str,
) -> float:
    """
    Calculate simple buy-and-hold return over the available data.
    """

    required = {"symbol", "timestamp", "close"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    bars = (
        df[df["symbol"] == symbol]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if len(bars) < 2:
        raise ValueError(
            f"Not enough data to calculate benchmark for {symbol}"
        )

    first_close = float(bars.iloc[0]["close"])
    last_close = float(bars.iloc[-1]["close"])

    return (last_close / first_close) - 1