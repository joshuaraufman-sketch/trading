from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading_lab.backtest.costs import (
    calculate_fees,
    calculate_slippage,
)
from trading_lab.backtest.models import Trade
from trading_lab.backtest.position_size import calculate_position_size


@dataclass
class TradeCandidate:
    symbol: str
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    raw_entry_price: float
    raw_exit_price: float
    stop_price: float


def _apply_buy_slippage(
    price: float,
    slippage_bps: float,
) -> float:
    return price * (1 + slippage_bps / 10_000)


def _apply_sell_slippage(
    price: float,
    slippage_bps: float,
) -> float:
    return price * (1 - slippage_bps / 10_000)


def _build_candidates(
    df: pd.DataFrame,
    *,
    stop_loss_pct: float,
    holding_days: int,
) -> list[TradeCandidate]:

    candidates: list[TradeCandidate] = []

    for symbol, symbol_df in df.groupby("symbol"):
        bars = (
            symbol_df
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        i = 0

        while i < len(bars) - 1:
            signal_bar = bars.iloc[i]

            if not bool(signal_bar["signal"]):
                i += 1
                continue

            entry_index = i + 1

            if entry_index >= len(bars):
                break

            entry_bar = bars.iloc[entry_index]
            raw_entry_price = float(entry_bar["open"])

            stop_price = raw_entry_price * (1 - stop_loss_pct)

            planned_exit_index = min(
                entry_index + holding_days,
                len(bars) - 1,
            )

            exit_index = planned_exit_index
            raw_exit_price = float(
                bars.iloc[planned_exit_index]["close"]
            )

            for j in range(
                entry_index,
                planned_exit_index + 1,
            ):
                bar = bars.iloc[j]

                bar_open = float(bar["open"])
                bar_low = float(bar["low"])

                # If market opens below the stop, assume the
                # stop fills at the worse opening price.
                if bar_open <= stop_price:
                    exit_index = j
                    raw_exit_price = bar_open
                    break

                # Otherwise assume stop fills at stop price
                # if the intraday low touches it.
                if bar_low <= stop_price:
                    exit_index = j
                    raw_exit_price = stop_price
                    break

            exit_bar = bars.iloc[exit_index]

            candidates.append(
                TradeCandidate(
                    symbol=str(symbol),
                    signal_time=signal_bar["timestamp"],
                    entry_time=entry_bar["timestamp"],
                    exit_time=exit_bar["timestamp"],
                    raw_entry_price=raw_entry_price,
                    raw_exit_price=raw_exit_price,
                    stop_price=stop_price,
                )
            )

            # Prevent overlapping trades in the same symbol.
            i = exit_index + 1

    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.entry_time,
            candidate.symbol,
        ),
    )


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
    Chronological long-only backtest runner.

    Rules:
        - signal occurs on completed bar
        - entry occurs at next bar open
        - stop is fixed percentage below raw entry
        - gap below stop fills at the worse opening price
        - slippage worsens actual entry and exit prices
        - otherwise exit after holding_days bars
        - no overlapping trades within the same symbol

    This is research infrastructure, not a production strategy.
    """

    required = {
        "symbol",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "signal",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if starting_equity <= 0:
        raise ValueError(
            "starting_equity must be greater than zero"
        )

    if holding_days < 1:
        raise ValueError(
            "holding_days must be at least 1"
        )

    if not 0 < stop_loss_pct < 1:
        raise ValueError(
            "stop_loss_pct must be between 0 and 1"
        )

    if slippage_bps < 0:
        raise ValueError(
            "slippage_bps cannot be negative"
        )

    candidates = _build_candidates(
        df,
        stop_loss_pct=stop_loss_pct,
        holding_days=holding_days,
    )

    trades: list[Trade] = []
    equity = starting_equity

    for candidate in candidates:
        entry_price = _apply_buy_slippage(
            candidate.raw_entry_price,
            slippage_bps,
        )

        exit_price = _apply_sell_slippage(
            candidate.raw_exit_price,
            slippage_bps,
        )

        quantity = calculate_position_size(
            account_equity=equity,
            risk_pct=risk_pct,
            entry_price=entry_price,
            stop_price=candidate.stop_price,
        )

        if quantity <= 0:
            continue

        entry_fees = calculate_fees(
            quantity=quantity,
            fee_per_share=fee_per_share,
        )

        exit_fees = calculate_fees(
            quantity=quantity,
            fee_per_share=fee_per_share,
        )

        # Slippage is already reflected in fill prices,
        # so we do not deduct it again from net P&L.
        trade = Trade(
            symbol=candidate.symbol,
            entry_time=candidate.entry_time,
            exit_time=candidate.exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            fees=entry_fees + exit_fees,
            slippage=0.0,
            initial_risk_per_share=(
                entry_price - candidate.stop_price
            ),
        )

        trades.append(trade)

        equity += trade.net_pnl

    return trades