from datetime import date

from trading_lab.risk.pending_signal_age import (
    check_pending_signal_age,
)


def test_yesterday_signal_passes():
    decision = check_pending_signal_age(
        signal_date="2026-08-27",
        current_date=date(
            2026,
            8,
            28,
        ),
    )

    assert decision.approved is True


def test_friday_signal_can_pass_monday():
    decision = check_pending_signal_age(
        signal_date="2026-08-28",
        current_date=date(
            2026,
            8,
            31,
        ),
    )

    assert decision.approved is True


def test_same_day_signal_is_blocked():
    decision = check_pending_signal_age(
        signal_date="2026-08-27",
        current_date=date(
            2026,
            8,
            27,
        ),
    )

    assert decision.approved is False


def test_old_signal_expires():
    decision = check_pending_signal_age(
        signal_date="2026-08-20",
        current_date=date(
            2026,
            8,
            27,
        ),
    )

    assert decision.approved is False


def test_invalid_date_is_blocked():
    decision = check_pending_signal_age(
        signal_date="not-a-date",
        current_date=date(
            2026,
            8,
            27,
        ),
    )

    assert decision.approved is False