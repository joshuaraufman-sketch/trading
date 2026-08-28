from datetime import date

import pytest

from trading_lab.execution.exit_schedule import (
    calculate_planned_exit_date,
    holding_period_is_due,
)
from trading_lab.risk.market_calendar import (
    advance_trading_days,
)


def test_advance_ten_trading_sessions():
    result = advance_trading_days(
        date(2026, 8, 28),
        10,
    )

    assert result == date(
        2026,
        9,
        14,
    )


def test_exit_date_skips_labor_day():
    result = calculate_planned_exit_date(
        filled_at_utc=(
            "2026-08-28T13:31:04+00:00"
        ),
        holding_days=10,
    )

    assert result == date(
        2026,
        9,
        14,
    )


def test_holding_period_not_due_early():
    assert not holding_period_is_due(
        planned_exit_date=date(
            2026,
            9,
            14,
        ),
        current_date=date(
            2026,
            9,
            11,
        ),
    )


def test_holding_period_due_on_exit_date():
    assert holding_period_is_due(
        planned_exit_date=date(
            2026,
            9,
            14,
        ),
        current_date=date(
            2026,
            9,
            14,
        ),
    )


def test_holding_period_stays_due_if_missed():
    assert holding_period_is_due(
        planned_exit_date=date(
            2026,
            9,
            14,
        ),
        current_date=date(
            2026,
            9,
            15,
        ),
    )


def test_invalid_holding_period():
    with pytest.raises(ValueError):
        calculate_planned_exit_date(
            filled_at_utc=(
                "2026-08-28T13:31:04+00:00"
            ),
            holding_days=0,
        )
