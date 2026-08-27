import pandas as pd

from trading_lab.backtest.runner import (
    run_long_signal_backtest,
)


def test_portfolio_does_not_use_more_cash_than_available():
    df = pd.DataFrame(
        {
            "symbol": [
                "AAA",
                "AAA",
                "AAA",
                "BBB",
                "BBB",
                "BBB",
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
                100,
                100,
                100,
                100,
                100,
            ],
            "high": [
                101,
                101,
                101,
                101,
                101,
                101,
            ],
            "low": [
                99,
                99,
                99,
                99,
                99,
                99,
            ],
            "close": [
                100,
                100,
                100,
                100,
                100,
                100,
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
        starting_equity=10_000,
        risk_pct=0.50,
        stop_loss_pct=0.50,
        holding_days=1,
        slippage_bps=0,
        fee_per_share=0,
    )

    total_entry_notional = sum(
        trade.entry_price * trade.quantity
        for trade in trades
    )

    assert total_entry_notional <= 10_000


def test_second_position_is_reduced_when_cash_is_limited():
    df = pd.DataFrame(
        {
            "symbol": [
                "AAA",
                "AAA",
                "AAA",
                "BBB",
                "BBB",
                "BBB",
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
                100,
                100,
                100,
                100,
                100,
            ],
            "high": [
                101,
                101,
                101,
                101,
                101,
                101,
            ],
            "low": [
                99,
                99,
                99,
                99,
                99,
                99,
            ],
            "close": [
                100,
                100,
                100,
                100,
                100,
                100,
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
        starting_equity=10_000,
        risk_pct=0.30,
        stop_loss_pct=0.50,
        holding_days=1,
        slippage_bps=0,
        fee_per_share=0,
    )

    assert len(trades) == 2

    first_notional = (
        trades[0].entry_price * trades[0].quantity
    )

    second_notional = (
        trades[1].entry_price * trades[1].quantity
    )

    assert first_notional + second_notional <= 10_000


def test_cash_is_released_after_position_exit():
    df = pd.DataFrame(
        {
            "symbol": [
                "AAA",
                "AAA",
                "AAA",
                "BBB",
                "BBB",
                "BBB",
                "BBB",
            ],
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
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
                100,
                100,
                100,
                100,
                100,
            ],
            "high": [
                101,
                101,
                101,
                101,
                101,
                101,
                101,
            ],
            "low": [
                99,
                99,
                99,
                99,
                99,
                99,
                99,
            ],
            "close": [
                100,
                100,
                100,
                100,
                100,
                100,
                100,
            ],
            "signal": [
                True,
                False,
                False,
                True,
                False,
                False,
                False,
            ],
        }
    )

    trades = run_long_signal_backtest(
        df,
        starting_equity=10_000,
        risk_pct=0.50,
        stop_loss_pct=0.50,
        holding_days=1,
        slippage_bps=0,
        fee_per_share=0,
    )

    assert len(trades) == 2