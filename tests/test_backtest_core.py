from datetime import datetime, timezone

import pytest

from trading_lab.backtest.costs import (
    calculate_fees,
    calculate_slippage,
)
from trading_lab.backtest.models import Trade
from trading_lab.backtest.position_size import calculate_position_size


def test_position_size():
    quantity = calculate_position_size(
        account_equity=100_000,
        risk_pct=0.005,
        entry_price=100,
        stop_price=98,
    )

    assert quantity == 250


def test_position_size_rejects_invalid_stop():
    with pytest.raises(ValueError):
        calculate_position_size(
            account_equity=100_000,
            risk_pct=0.005,
            entry_price=100,
            stop_price=101,
        )


def test_slippage():
    result = calculate_slippage(
        price=100,
        quantity=250,
        slippage_bps=5,
    )

    assert result == pytest.approx(12.50)


def test_fees():
    result = calculate_fees(
        quantity=250,
        fee_per_share=0.005,
    )

    assert result == pytest.approx(1.25)


def test_trade_net_pnl():
    trade = Trade(
        symbol="SPY",
        entry_time=datetime(
            2025,
            1,
            2,
            tzinfo=timezone.utc,
        ),
        exit_time=datetime(
            2025,
            1,
            3,
            tzinfo=timezone.utc,
        ),
        entry_price=100,
        exit_price=104,
        quantity=250,
        fees=1.25,
        slippage=12.50,
    )

    assert trade.gross_pnl == pytest.approx(1000)
    assert trade.net_pnl == pytest.approx(986.25)