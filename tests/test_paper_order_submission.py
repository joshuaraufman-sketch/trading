from dataclasses import replace

import pytest

from trading_lab.execution.alpaca_orders import (
    submit_paper_market_order,
)
from trading_lab.execution.models import OrderPlan


def make_order():
    return OrderPlan(
        symbol="SPY",
        side="buy",
        quantity=10,
        signal_time=None,
        reference_price=100.00,
        stop_price=98.00,
        risk_per_share=2.00,
        estimated_position_value=1000.00,
    )


def test_rejects_nonpositive_quantity():
    order = replace(
        make_order(),
        quantity=0,
    )

    with pytest.raises(
        ValueError,
        match="quantity",
    ):
        submit_paper_market_order(order)


def test_rejects_nonpositive_stop():
    order = replace(
        make_order(),
        stop_price=0,
    )

    with pytest.raises(
        ValueError,
        match="Stop price",
    ):
        submit_paper_market_order(order)


def test_rejects_stop_above_reference():
    order = replace(
        make_order(),
        stop_price=101.00,
    )

    with pytest.raises(
        ValueError,
        match="below reference price",
    ):
        submit_paper_market_order(order)