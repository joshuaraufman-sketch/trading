from datetime import datetime, timezone

from trading_lab.execution.planner import (
    build_long_order_plan,
)
from trading_lab.risk.order_checks import (
    check_order_plan,
)


def test_valid_order_plan_passes_risk_checks():

    order = build_long_order_plan(
        symbol="SPY",
        signal_time=datetime(
            2026,
            8,
            27,
            tzinfo=timezone.utc,
        ),
        reference_price=100,
        account_equity=100_000,
        stop_loss_pct=0.02,
        risk_pct=0.005,
    )

    decision = check_order_plan(
        order,
        account_equity=100_000,
        available_cash=100_000,
        maximum_position_pct=0.50,
        maximum_risk_pct=0.005,
    )

    assert decision.approved is True


def test_position_limit_can_block_order():

    order = build_long_order_plan(
        symbol="SPY",
        signal_time=datetime(
            2026,
            8,
            27,
            tzinfo=timezone.utc,
        ),
        reference_price=100,
        account_equity=100_000,
        stop_loss_pct=0.02,
        risk_pct=0.005,
    )

    decision = check_order_plan(
        order,
        account_equity=100_000,
        available_cash=100_000,
        maximum_position_pct=0.10,
        maximum_risk_pct=0.005,
    )

    assert decision.approved is False
    assert "maximum position size" in decision.reason