import pytest

from trading_lab.execution.reconciliation import (
    reconcile_fill,
)


def test_reconcile_worse_buy_fill():
    result = reconcile_fill(
        order_id="test-order",
        symbol="SPY",
        reference_price=100.00,
        status="filled",
        filled_avg_price=100.10,
        filled_qty=50,
    )

    assert result.slippage_per_share == pytest.approx(0.10)
    assert result.slippage_bps == pytest.approx(10.0)
    assert result.total_slippage_dollars == pytest.approx(5.0)


def test_reconcile_better_buy_fill():
    result = reconcile_fill(
        order_id="test-order",
        symbol="SPY",
        reference_price=100.00,
        status="filled",
        filled_avg_price=99.95,
        filled_qty=20,
    )

    assert result.slippage_per_share == pytest.approx(-0.05)
    assert result.slippage_bps == pytest.approx(-5.0)
    assert result.total_slippage_dollars == pytest.approx(-1.0)


def test_unfilled_order_has_no_slippage():
    result = reconcile_fill(
        order_id="test-order",
        symbol="SPY",
        reference_price=100.00,
        status="new",
        filled_avg_price=None,
        filled_qty=0,
    )

    assert result.slippage_per_share is None
    assert result.slippage_bps is None
    assert result.total_slippage_dollars is None