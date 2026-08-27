import pandas as pd
import pytest

from trading_lab.backtest.runner import run_long_signal_backtest


def make_test_data():
    return pd.DataFrame(
        {
            "symbol": ["TEST"] * 6,
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-04",
                    "2025-01-05",
                    "2025-01-06",
                ],
                utc=True,
            ),
            "open": [
                100,
                100,
                102,
                103,
                104,
                105,
            ],
            "high": [
                101,
                103,
                104,
                105,
                106,
                107,
            ],
            "low": [
                99,
                99,
                101,
                102,
                103,
                104,
            ],
            "close": [
                100,
                102,
                103,
                104,
                105,
                106,
            ],
            "signal": [
                True,
                False,
                False,
                False,
                False,
                False,
            ],
        }
    )


def test_entry_occurs_on_next_bar():
    df = make_test_data()

    trades = run_long_signal_backtest(
        df,
        starting_equity=100_000,
        risk_pct=0.005,
        stop_loss_pct=0.02,
        holding_days=2,
        slippage_bps=0,
        fee_per_share=0,
    )

    assert len(trades) == 1

    trade = trades[0]

    assert trade.entry_price == pytest.approx(100)
    assert trade.entry_time == df.iloc[1]["timestamp"]


def test_holding_period_exit():
    df = make_test_data()

    trades = run_long_signal_backtest(
        df,
        starting_equity=100_000,
        risk_pct=0.005,
        stop_loss_pct=0.02,
        holding_days=2,
        slippage_bps=0,
        fee_per_share=0,
    )

    trade = trades[0]

    assert trade.exit_time == df.iloc[3]["timestamp"]
    assert trade.exit_price == pytest.approx(104)


def test_stop_loss_exit():
    df = make_test_data()

    df.loc[2, "low"] = 97

    trades = run_long_signal_backtest(
        df,
        starting_equity=100_000,
        risk_pct=0.005,
        stop_loss_pct=0.02,
        holding_days=4,
        slippage_bps=0,
        fee_per_share=0,
    )

    trade = trades[0]

    assert trade.exit_time == df.iloc[2]["timestamp"]
    assert trade.exit_price == pytest.approx(98)


def test_no_overlapping_trade_from_same_symbol():
    df = make_test_data()

    df.loc[1, "signal"] = True
    df.loc[2, "signal"] = True

    trades = run_long_signal_backtest(
        df,
        starting_equity=100_000,
        risk_pct=0.005,
        stop_loss_pct=0.02,
        holding_days=2,
        slippage_bps=0,
        fee_per_share=0,
    )

    assert len(trades) == 1