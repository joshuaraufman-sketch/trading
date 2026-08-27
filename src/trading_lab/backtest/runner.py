from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading_lab.backtest.costs import calculate_fees
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


@dataclass
class OpenPosition:
    symbol: str
    exit_time: pd.Timestamp
    capital_committed: float


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

                if bar_open <= stop_price:
                    exit_index = j
                    raw_exit_price = bar_open
                    break

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

    Portfolio assumptions:
        - no leverage
        - no margin
        - overlapping positions may exist across symbols
        - new positions cannot exceed available cash
        - same-symbol trades cannot overlap
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

    realized_equity = starting_equity
    open_positions: list[OpenPosition] = []

    for candidate in candidates:

        # Release cash from positions that have already exited.
        open_positions = [
            position
            for position in open_positions
            if position.exit_time >= candidate.entry_time
        ]

        capital_in_use = sum(
            position.capital_committed
            for position in open_positions
        )

        available_cash = realized_equity - capital_in_use

        if available_cash <= 0:
            continue

        entry_price = _apply_buy_slippage(
            candidate.raw_entry_price,
            slippage_bps,
        )

        exit_price = _apply_sell_slippage(
            candidate.raw_exit_price,
            slippage_bps,
        )

        risk_quantity = calculate_position_size(
            account_equity=realized_equity,
            risk_pct=risk_pct,
            entry_price=entry_price,
            stop_price=candidate.stop_price,
        )

        cash_quantity = int(
            available_cash // entry_price
        )

        quantity = min(
            risk_quantity,
            cash_quantity,
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

        capital_committed = (
            entry_price * quantity
        )

        open_positions.append(
            OpenPosition(
                symbol=candidate.symbol,
                exit_time=candidate.exit_time,
                capital_committed=capital_committed,
            )
        )

        # For this simplified accounting model,
        # realized P&L is added when the trade is created.
        # Capital remains reserved until exit_time.
        realized_equity += trade.net_pnl

    return trades