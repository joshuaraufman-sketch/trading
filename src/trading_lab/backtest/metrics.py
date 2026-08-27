from __future__ import annotations

import math
from collections.abc import Iterable

import pandas as pd

from trading_lab.backtest.models import Trade


def trades_to_frame(trades: Iterable[Trade]) -> pd.DataFrame:
    rows = []

    for trade in trades:
        rows.append(
            {
                "symbol": trade.symbol,
                "entry_time": trade.entry_time,
                "exit_time": trade.exit_time,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "quantity": trade.quantity,
                "fees": trade.fees,
                "slippage": trade.slippage,
                "gross_pnl": trade.gross_pnl,
                "net_pnl": trade.net_pnl,
                "return_pct": trade.return_pct,
                "initial_risk": trade.initial_risk,
                "r_multiple": trade.r_multiple,
            }
        )

    return pd.DataFrame(rows)


def build_equity_curve(
    trades: Iterable[Trade],
    starting_equity: float,
) -> pd.DataFrame:
    trades = list(trades)

    if starting_equity <= 0:
        raise ValueError("starting_equity must be greater than zero")

    rows = [
        {
            "trade_number": 0,
            "equity": starting_equity,
        }
    ]

    equity = starting_equity

    for number, trade in enumerate(trades, start=1):
        equity += trade.net_pnl

        rows.append(
            {
                "trade_number": number,
                "equity": equity,
            }
        )

    curve = pd.DataFrame(rows)

    curve["running_peak"] = curve["equity"].cummax()

    curve["drawdown_pct"] = (
        curve["equity"] - curve["running_peak"]
    ) / curve["running_peak"]

    return curve


def calculate_metrics(
    trades: Iterable[Trade],
    starting_equity: float,
) -> dict:
    trades = list(trades)

    if starting_equity <= 0:
        raise ValueError("starting_equity must be greater than zero")

    if not trades:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "net_profit": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "average_r": 0.0,
            "max_drawdown_pct": 0.0,
            "ending_equity": starting_equity,
        }

    pnl = pd.Series(
        [trade.net_pnl for trade in trades],
        dtype=float,
    )

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = math.inf
    else:
        profit_factor = 0.0

    r_values = [
        trade.r_multiple
        for trade in trades
        if trade.r_multiple is not None
    ]

    average_r = (
        float(pd.Series(r_values).mean())
        if r_values
        else 0.0
    )

    curve = build_equity_curve(
        trades,
        starting_equity,
    )

    max_drawdown_pct = abs(
        float(curve["drawdown_pct"].min())
    )

    net_profit = float(pnl.sum())

    return {
        "trade_count": len(trades),
        "win_rate": float((pnl > 0).mean()),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_profit": net_profit,
        "profit_factor": profit_factor,
        "expectancy": float(pnl.mean()),
        "average_r": average_r,
        "max_drawdown_pct": max_drawdown_pct,
        "ending_equity": starting_equity + net_profit,
    }