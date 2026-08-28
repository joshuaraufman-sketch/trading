from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from trading_lab.backtest.models import Trade


MARKET_TIMEZONE = "America/New_York"


def normalize_session_dates(
    timestamps: pd.Series,
) -> pd.Series:
    """
    Convert bar timestamps to US market session dates.

    Alpaca returns daily bars stamped at the UTC equivalent of the
    market open, which shifts with daylight saving time. Converting to
    exchange-local time before taking the date keeps every bar on the
    session it actually belongs to.

    Timezone-naive input is assumed to already be a session date and is
    normalized without conversion. Treating it as UTC instead would map
    midnight back to 19:00 the previous day and silently shift every
    trade one session earlier.
    """

    parsed = pd.to_datetime(timestamps)

    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_convert(MARKET_TIMEZONE).dt.tz_localize(
            None
        )

    return parsed.dt.normalize()


@dataclass(frozen=True)
class _CashEvent:
    session: pd.Timestamp
    amount: float


def _build_close_panel(
    bars: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pivot bars into a session x symbol matrix of closing prices.

    Missing sessions for an individual symbol are forward filled so a
    halted or thinly traded name is carried at its last known price
    rather than silently dropping out of the equity curve.
    """

    required = {"symbol", "timestamp", "close"}
    missing = required - set(bars.columns)

    if missing:
        raise ValueError(
            f"bars is missing required columns: {sorted(missing)}"
        )

    frame = bars.loc[:, ["symbol", "timestamp", "close"]].copy()
    frame["session"] = normalize_session_dates(frame["timestamp"])

    panel = (
        frame
        .pivot_table(
            index="session",
            columns="symbol",
            values="close",
            aggfunc="last",
        )
        .sort_index()
    )

    return panel.ffill()


def build_daily_equity_curve(
    trades: Iterable[Trade],
    bars: pd.DataFrame,
    *,
    starting_equity: float,
    entry_fee_fraction: float = 0.5,
) -> pd.DataFrame:
    """
    Build a calendar-indexed, mark-to-market equity curve.

    The existing trade-ordered curve steps once per closed trade and
    therefore cannot see unrealized drawdown, cannot be compared to a
    benchmark on a common time axis, and understates risk whenever
    correlated positions are open at once. This walks every trading
    session instead, marking open positions at that session's close.

    ``Trade.fees`` stores the round-trip total. ``entry_fee_fraction``
    controls how much of it is charged at entry; the runner currently
    charges symmetric per-share fees, so 0.5 is correct for it.

    Returns a frame indexed by session date with columns:
        cash, position_value, equity, open_positions,
        exposure_pct, daily_return, running_peak, drawdown_pct
    """

    if starting_equity <= 0:
        raise ValueError("starting_equity must be greater than zero")

    if not 0.0 <= entry_fee_fraction <= 1.0:
        raise ValueError("entry_fee_fraction must be between 0 and 1")

    trades = list(trades)
    panel = _build_close_panel(bars)

    if panel.empty:
        raise ValueError("bars contained no usable sessions")

    sessions = panel.index

    # Per-session cash movements and per-session position deltas.
    cash_flows = pd.Series(0.0, index=sessions)
    holdings_delta = pd.DataFrame(
        0.0,
        index=sessions,
        columns=panel.columns,
    )

    for trade in trades:
        if trade.symbol not in panel.columns:
            raise ValueError(
                f"No price bars supplied for traded symbol "
                f"{trade.symbol!r}"
            )

        entry_session = normalize_session_dates(
            pd.Series([trade.entry_time])
        ).iloc[0]

        exit_session = normalize_session_dates(
            pd.Series([trade.exit_time])
        ).iloc[0]

        for label, session in (
            ("entry", entry_session),
            ("exit", exit_session),
        ):
            if session not in cash_flows.index:
                raise ValueError(
                    f"Trade {label} session {session.date()} for "
                    f"{trade.symbol} falls outside the supplied bars"
                )

        entry_fees = trade.fees * entry_fee_fraction
        exit_fees = trade.fees - entry_fees

        cash_flows.loc[entry_session] -= (
            trade.entry_price * trade.quantity + entry_fees
        )
        cash_flows.loc[exit_session] += (
            trade.exit_price * trade.quantity - exit_fees
        )

        holdings_delta.loc[entry_session, trade.symbol] += trade.quantity
        holdings_delta.loc[exit_session, trade.symbol] -= trade.quantity

    cash = starting_equity + cash_flows.cumsum()

    # Shares held at each session close, after that session's fills.
    holdings = holdings_delta.cumsum()

    marked = holdings * panel
    position_value = marked.sum(axis=1)
    open_positions = (holdings.abs() > 0).sum(axis=1)

    equity = cash + position_value

    curve = pd.DataFrame(
        {
            "cash": cash,
            "position_value": position_value,
            "equity": equity,
            "open_positions": open_positions.astype(int),
        }
    )

    curve["exposure_pct"] = (
        curve["position_value"] / curve["equity"]
    ).where(curve["equity"] > 0, 0.0)

    curve["daily_return"] = curve["equity"].pct_change().fillna(0.0)
    curve["running_peak"] = curve["equity"].cummax()
    curve["drawdown_pct"] = (
        curve["equity"] - curve["running_peak"]
    ) / curve["running_peak"]

    curve.index.name = "session"

    return curve
