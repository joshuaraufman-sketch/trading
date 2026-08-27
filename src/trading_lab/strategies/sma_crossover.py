from __future__ import annotations

import pandas as pd

from trading_lab.strategies.base import Strategy


class SMACrossoverStrategy(Strategy):
    name = "sma_crossover"

    def __init__(
        self,
        window: int = 20,
    ) -> None:
        if window < 2:
            raise ValueError(
                "window must be at least 2"
            )

        self.window = window

    def generate_signals(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        result = (
            df
            .sort_values(
                ["symbol", "timestamp"]
            )
            .copy()
        )

        result["sma"] = (
            result
            .groupby("symbol")["close"]
            .transform(
                lambda x: x.rolling(
                    self.window
                ).mean()
            )
        )

        previous_close = (
            result
            .groupby("symbol")["close"]
            .shift(1)
        )

        previous_sma = (
            result
            .groupby("symbol")["sma"]
            .shift(1)
        )

        result["signal"] = (
            (result["close"] > result["sma"])
            & (previous_close <= previous_sma)
        )

        result["signal"] = (
            result["signal"]
            .fillna(False)
            .astype(bool)
        )

        return result