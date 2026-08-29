from __future__ import annotations

from datetime import date, timedelta

from trading_lab.execution.models import RiskDecision


# Hardcoded holidays are a dated asset. The original set covered 2026
# only, which means that on 1 January 2027 every 2027 holiday would have
# been treated as a normal trading day -- silently, with no error.
#
# Two defences now exist. Coverage is declared explicitly and
# `holidays_cover` reports when a date falls outside it, so a scheduler
# can refuse rather than guess. And the daily runner's stale-bar check
# is the backstop: on a holiday no new bar exists, so it refuses anyway.
#
# Dates for 2027-2028 are computed from the standard NYSE rules. Verify
# against the exchange calendar before relying on them, and extend well
# before the coverage window lapses.
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

US_MARKET_HOLIDAYS_2027 = {
    date(2027, 1, 1),   # New Year's Day
    date(2027, 1, 18),  # Martin Luther King Jr. Day
    date(2027, 2, 15),  # Presidents' Day
    date(2027, 3, 26),  # Good Friday
    date(2027, 5, 31),  # Memorial Day
    date(2027, 6, 18),  # Juneteenth observed (19th is a Saturday)
    date(2027, 7, 5),   # Independence Day observed (4th is a Sunday)
    date(2027, 9, 6),   # Labor Day
    date(2027, 11, 25), # Thanksgiving
    date(2027, 12, 24), # Christmas observed (25th is a Saturday)
}

US_MARKET_HOLIDAYS_2028 = {
    date(2028, 1, 17),  # Martin Luther King Jr. Day
    date(2028, 2, 21),  # Presidents' Day
    date(2028, 4, 14),  # Good Friday
    date(2028, 5, 29),  # Memorial Day
    date(2028, 6, 19),  # Juneteenth
    date(2028, 7, 4),   # Independence Day
    date(2028, 9, 4),   # Labor Day
    date(2028, 11, 23), # Thanksgiving
    date(2028, 12, 25), # Christmas
}

US_MARKET_HOLIDAYS = (
    US_MARKET_HOLIDAYS_2026
    | US_MARKET_HOLIDAYS_2027
    | US_MARKET_HOLIDAYS_2028
)

COVERED_YEARS = frozenset({2026, 2027, 2028})


def holidays_cover(day: date) -> bool:
    """
    Whether the holiday table actually covers this date.

    Callers that automate anything should check this. Outside the
    covered years `is_trading_day` cannot distinguish a holiday from a
    normal session and will answer confidently and wrongly.
    """

    return day.year in COVERED_YEARS


def is_trading_day(
    day: date,
) -> bool:
    if day.weekday() >= 5:
        return False

    if day in US_MARKET_HOLIDAYS:
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
