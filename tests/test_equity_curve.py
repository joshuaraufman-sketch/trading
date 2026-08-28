from __future__ import annotations

import pandas as pd
import pytest

from trading_lab.backtest.equity import build_daily_equity_curve
from trading_lab.backtest.models import Trade


def _bars(symbol: str, closes: list[float], start: str = "2024-01-02"):
    sessions = pd.bdate_range(start=start, periods=len(closes), tz="UTC")

    # Alpaca stamps daily bars at the UTC equivalent of the open.
    sessions = sessions + pd.Timedelta(hours=5)

    return pd.DataFrame(
        {
            "symbol": symbol,
            "timestamp": sessions,
            "close": closes,
        }
    )


def test_flat_curve_with_no_trades():
    bars = _bars("SPY", [100.0, 101.0, 102.0, 103.0])

    curve = build_daily_equity_curve(
        [],
        bars,
        starting_equity=100_000,
    )

    assert len(curve) == 4
    assert (curve["equity"] == 100_000).all()
    assert (curve["open_positions"] == 0).all()
    assert curve["drawdown_pct"].min() == 0.0


def test_open_position_is_marked_to_market():
    """
    The core reason this module exists: a trade that dips and recovers
    must produce a drawdown. The trade-ordered curve in metrics.py
    reports zero here because it only steps on close.
    """

    bars = _bars("SPY", [100.0, 90.0, 100.0, 110.0])
    sessions = pd.to_datetime(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    )

    trade = Trade(
        symbol="SPY",
        entry_time=sessions[0],
        exit_time=sessions[3],
        entry_price=100.0,
        exit_price=110.0,
        quantity=100,
        fees=0.0,
    )

    curve = build_daily_equity_curve(
        [trade],
        bars,
        starting_equity=100_000,
    )

    # 100 shares at 100 -> 10,000 deployed, marked at 90 on day two.
    assert curve.loc[sessions[1], "equity"] == pytest.approx(99_000)
    assert curve["drawdown_pct"].min() == pytest.approx(-0.01, abs=1e-9)
    assert curve.loc[sessions[3], "equity"] == pytest.approx(101_000)

    # And the position is cash again on the exit session.
    assert curve.loc[sessions[3], "open_positions"] == 0
    assert curve.loc[sessions[1], "open_positions"] == 1


def test_correlated_positions_compound_drawdown():
    """
    Four correlated ETFs stopping out together is one bet, not four.
    A trade-sequence curve smears that across four steps; a calendar
    curve shows the single-day hit.
    """

    sessions = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])

    bars = pd.concat(
        [
            _bars("SPY", [100.0, 95.0, 95.0]),
            _bars("QQQ", [100.0, 95.0, 95.0]),
            _bars("IWM", [100.0, 95.0, 95.0]),
            _bars("DIA", [100.0, 95.0, 95.0]),
        ],
        ignore_index=True,
    )

    trades = [
        Trade(
            symbol=symbol,
            entry_time=sessions[0],
            exit_time=sessions[2],
            entry_price=100.0,
            exit_price=95.0,
            quantity=100,
            fees=0.0,
        )
        for symbol in ("SPY", "QQQ", "IWM", "DIA")
    ]

    curve = build_daily_equity_curve(
        trades,
        bars,
        starting_equity=100_000,
    )

    assert curve.loc[sessions[0], "open_positions"] == 4
    # 40,000 deployed, down 5% on day two -> -2,000 on the account.
    assert curve.loc[sessions[1], "equity"] == pytest.approx(98_000)
    assert curve["drawdown_pct"].min() == pytest.approx(-0.02, abs=1e-9)


def test_fees_are_split_across_entry_and_exit():
    bars = _bars("SPY", [100.0, 100.0])
    sessions = pd.to_datetime(["2024-01-02", "2024-01-03"])

    trade = Trade(
        symbol="SPY",
        entry_time=sessions[0],
        exit_time=sessions[1],
        entry_price=100.0,
        exit_price=100.0,
        quantity=10,
        fees=10.0,
    )

    curve = build_daily_equity_curve(
        [trade],
        bars,
        starting_equity=100_000,
        entry_fee_fraction=0.5,
    )

    assert curve.loc[sessions[0], "cash"] == pytest.approx(98_995.0)
    assert curve.loc[sessions[1], "equity"] == pytest.approx(99_990.0)


def test_unknown_symbol_is_rejected():
    bars = _bars("SPY", [100.0, 101.0])

    trade = Trade(
        symbol="QQQ",
        entry_time=pd.Timestamp("2024-01-02"),
        exit_time=pd.Timestamp("2024-01-03"),
        entry_price=100.0,
        exit_price=101.0,
        quantity=10,
    )

    with pytest.raises(ValueError, match="No price bars"):
        build_daily_equity_curve(
            [trade],
            bars,
            starting_equity=100_000,
        )


def test_trade_outside_bar_range_is_rejected():
    bars = _bars("SPY", [100.0, 101.0])

    trade = Trade(
        symbol="SPY",
        entry_time=pd.Timestamp("2024-01-02"),
        exit_time=pd.Timestamp("2030-01-03"),
        entry_price=100.0,
        exit_price=101.0,
        quantity=10,
    )

    with pytest.raises(ValueError, match="outside the supplied bars"):
        build_daily_equity_curve(
            [trade],
            bars,
            starting_equity=100_000,
        )
