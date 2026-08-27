import pandas as pd
import pytest

from trading_lab.backtest.runner import (
    run_long_signal_backtest,
)


def make_gap_data():
    return pd.DataFrame(
        {
            "symbol": ["TEST"] * 4,
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-04",
                ],
                utc=True,
            ),
            "open": [
                100,
                100,
                95,
                96,
            ],
            "high": [
                101,
                101,
                97,
                98,
            ],
            "low": [
                99,
                99,
                94,
                95,
            ],
            "close": [
                100,
                100,
                96,
                97,
            ],
            "signal": [
                True,
                False,
                False,
                False,
            ],
        }
    )


def test_gap_below_stop_fills_at_open():
    df = make_gap_data()

    trades = run_long_signal_backtest(
        df,
        starting_equity=100_000,
        risk_pct=0.005,
        stop_loss_pct=0.02,
        holding_days=3,
        slippage_bps=0,
        fee_per_share=0,
    )

    assert len(trades) == 1

    trade = trades[0]

    # Entry = 100, stop = 98.
    # Next bar opens at 95, so stop cannot fill at 98.
    assert trade.exit_price == pytest.approx(95)


def test_slippage_changes_actual_fill_prices():
    df = make_gap_data()

    trades = run_long_signal_backtest(
        df,
        starting_equity=100_000,
        risk_pct=0.005,
        stop_loss_pct=0.02,
        holding_days=1,
        slippage_bps=10,
        fee_per_share=0,
    )

    trade = trades[0]

    assert trade.entry_price == pytest.approx(100.10)

    # Raw exit is 95 because of gap stop.
    # Selling slippage worsens it by 10 bps.
    assert trade.exit_price == pytest.approx(
        95 * 0.999
    )

    # Slippage is represented in prices, not charged twice.
    assert trade.slippage == pytest.approx(0)


def test_trades_returned_in_entry_time_order():
    df = pd.DataFrame(
        {
            "symbol": [
                "ZZZ",
                "ZZZ",
                "ZZZ",
                "AAA",
                "AAA",
                "AAA",
            ],
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                ],
                utc=True,
            ),
            "open": [
                100,
                101,
                102,
                50,
                51,
                52,
            ],
            "high": [
                101,
                102,
                103,
                51,
                52,
                53,
            ],
            "low": [
                99,
                100,
                101,
                49,
                50,
                51,
            ],
            "close": [
                100,
                101,
                102,
                50,
                51,
                52,
            ],
            "signal": [
                True,
                False,
                False,
                True,
                False,
                False,
            ],
        }
    )

    trades = run_long_signal_backtest(
        df,
        starting_equity=100_000,
        risk_pct=0.005,
        stop_loss_pct=0.10,
        holding_days=1,
        slippage_bps=0,
        fee_per_share=0,
    )

    assert len(trades) == 2

    assert trades[0].entry_time <= trades[1].entry_time
    assert trades[0].symbol == "AAA"
    assert trades[1].symbol == "ZZZ"