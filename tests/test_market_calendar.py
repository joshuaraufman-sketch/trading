from datetime import date

from trading_lab.risk.market_calendar import (
    check_is_next_trading_session,
    is_trading_day,
    next_trading_day,
)


def test_weekday_is_trading_day():
    assert is_trading_day(
        date(2026, 8, 27)
    )


def test_weekend_is_not_trading_day():
    assert not is_trading_day(
        date(2026, 8, 29)
    )


def test_friday_advances_to_monday():
    assert next_trading_day(
        date(2026, 8, 28)
    ) == date(
        2026,
        8,
        31,
    )


def test_holiday_is_skipped():
    assert next_trading_day(
        date(2026, 9, 4)
    ) == date(
        2026,
        9,
        8,
    )


def test_next_session_passes():
    decision = check_is_next_trading_session(
        signal_date="2026-08-27",
        current_date=date(
            2026,
            8,
            28,
        ),
    )

    assert decision.approved is True


def test_wrong_session_is_blocked():
    decision = check_is_next_trading_session(
        signal_date="2026-08-27",
        current_date=date(
            2026,
            8,
            31,
        ),
    )

    assert decision.approved is False