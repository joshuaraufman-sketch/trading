from __future__ import annotations

import pandas as pd

from trading_lab.backtest.costs import (
    calculate_fees,
    calculate_slippage,
)
from trading_lab.backtest.models import Trade
from trading_lab.backtest.position_size import calculate_position_size


def run_long_signal_backtest(
    df: pd.DataFrame,
    *,
    starting_equity: float = 100_000,
    risk_pct: float = 0.005,
    stop_loss_pct: float = 0.02,
    holding_days: int = 5,
    slippage_bps: float = 5,
    fee_per_share: float = 0.005,
) -> list[Trade]:
    """
    Simple long-only backtest runner.

    Expected input columns:
        symbol
        timestamp
        open
        close
        signal

    Rules:
        - signal == True creates a long entry
        - enter at next bar's open
        - stop is fixed percentage below entry
        - otherwise exit after holding_days bars

    This is infrastructure testing, not a production strategy.
    """

    required = {
        "symbol",
        "timestamp",
        "open",
        "close",
        "signal",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if holding_days < 1:
        raise ValueError("holding_days must be at least 1")

    if not 0 < stop_loss_pct < 1:
        raise ValueError(
            "stop_loss_pct must be between 0 and 1"
        )

    trades: list[Trade] = []
    equity = starting_equity

    for symbol, symbol_df in df.groupby("symbol"):
        bars = (
            symbol_df
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        i = 0

        while i < len(bars) - 1:
            row = bars.iloc[i]

            if not bool(row["signal"]):
                i += 1
                continue

            entry_index = i + 1

            if entry_index >= len(bars):
                break

            entry_bar = bars.iloc[entry_index]
            entry_price = float(entry_bar["open"])

            stop_price = entry_price * (1 - stop_loss_pct)

            quantity = calculate_position_size(
                account_equity=equity,
                risk_pct=risk_pct,
                entry_price=entry_price,
                stop_price=stop_price,
            )

            if quantity <= 0:
                i += 1
                continue

            planned_exit_index = min(
                entry_index + holding_days,
                len(bars) - 1,
            )

            exit_index = planned_exit_index
            exit_price = float(
                bars.iloc[planned_exit_index]["close"]
            )

            for j in range(
                entry_index,
                planned_exit_index + 1,
            ):
                bar = bars.iloc[j]

                if float(bar["low"]) <= stop_price:
                    exit_index = j
                    exit_price = stop_price
                    break

            exit_bar = bars.iloc[exit_index]

            entry_slippage = calculate_slippage(
                price=entry_price,
                quantity=quantity,
                slippage_bps=slippage_bps,
            )

            exit_slippage = calculate_slippage(
                price=exit_price,
                quantity=quantity,
                slippage_bps=slippage_bps,
            )

            entry_fees = calculate_fees(
                quantity=quantity,
                fee_per_share=fee_per_share,
            )

            exit_fees = calculate_fees(
                quantity=quantity,
                fee_per_share=fee_per_share,
            )

            trade = Trade(
                symbol=str(symbol),
                entry_time=entry_bar["timestamp"],
                exit_time=exit_bar["timestamp"],
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                fees=entry_fees + exit_fees,
                slippage=entry_slippage + exit_slippage,
            )

            trades.append(trade)

            equity += trade.net_pnl

            i = exit_index + 1

    return trades