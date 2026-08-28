from __future__ import annotations

from datetime import date, timedelta

from trading_lab.execution.models import RiskDecision


US_MARKET_HOLIDAYS_2026 = {
    date(2026, 1, 1),   # New Year's Day
    date(2026, 1, 19),  # Martin Luther King Jr. Day
    date(2026, 2, 16),  # Presidents' Day
    date(2026, 4, 3),   # Good Friday
    date(2026, 5, 25),  # Memorial Day
    date(2026, 6, 19),  # Juneteenth
    date(2026, 7, 3),   # Independence Day observed
    date(2026, 9, 7),   # Labor Day
    date(2026, 11, 26), # Thanksgiving
    date(2026, 12, 25), # Christmas
}


def is_trading_day(
    day: date,
) -> bool:
    if day.weekday() >= 5:
        return False

    if day in US_MARKET_HOLIDAYS_2026:
        return False

    return True


def next_trading_day(
    day: date,
) -> date:
    candidate = day + timedelta(days=1)

    while not is_trading_day(candidate):
        candidate += timedelta(days=1)

    return candidate


def check_is_next_trading_session(
    *,
    signal_date: str,
    current_date: date,
) -> RiskDecision:
    """
    Require execution to occur on the immediately
    following U.S. trading session.
    """

    try:
        parsed_signal_date = date.fromisoformat(
            signal_date
        )
    except ValueError:
        return RiskDecision(
            approved=False,
            reason="Pending signal date is invalid.",
        )

    expected_execution_date = next_trading_day(
        parsed_signal_date
    )

    if current_date != expected_execution_date:
        return RiskDecision(
            approved=False,
            reason=(
                "Current date is not the next "
                "eligible trading session."
            ),
        )

    return RiskDecision(
        approved=True,
        reason=(
            "Next trading session check passed."
        ),
    )

def advance_trading_days(
    day: date,
    sessions: int,
) -> date:
    """
    Advance by a number of subsequent U.S.
    trading sessions.

    The starting day is not counted.
    """

    if sessions < 0:
        raise ValueError(
            "sessions cannot be negative"
        )

    result = day

    for _ in range(sessions):
        result = next_trading_day(result)

    return result
