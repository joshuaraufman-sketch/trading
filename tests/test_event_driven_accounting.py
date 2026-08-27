import pandas as pd

from trading_lab.backtest.runner import (
    run_long_signal_backtest,
)


def test_unrealized_profit_does_not_increase_next_position_size():
    df = pd.DataFrame(
        {
            "symbol": [
                "AAA",
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
                    "2025-01-04",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-04",
                    "2025-01-05",
                ],
                utc=True,
            ),
            "open": [
                100,
                100,
                150,
                150,
                100,
                100,
                100,
                100,
            ],
            "high": [
                101,
                151,
                151,
                151,
                101,
                101,
                101,
                101,
            ],
            "low": [
                99,
                99,
                149,
                149,
                99,
                99,
                99,
                99,
            ],
            "close": [
                100,
                150,
                150,
                150,
                100,
                100,
                100,
                100,
            ],
            "signal": [
                True,
                False,
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
        risk_pct=0.10,
        stop_loss_pct=0.20,
        holding_days=2,
        slippage_bps=0,
        fee_per_share=0,
    )

    assert len(trades) == 2

    aaa = next(
        trade for trade in trades
        if trade.symbol == "AAA"
    )

    bbb = next(
        trade for trade in trades
        if trade.symbol == "BBB"
    )

    # AAA has a large unrealized gain while BBB enters.
    # BBB sizing must still use the original realized equity,
    # not AAA's future profit.
    assert aaa.exit_time > bbb.entry_time

    expected_bbb_quantity = 50

    assert bbb.quantity == expected_bbb_quantity