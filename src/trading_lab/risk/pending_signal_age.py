from __future__ import annotations

from datetime import date

from trading_lab.execution.models import RiskDecision


def check_pending_signal_age(
    *,
    signal_date: str,
    current_date: date,
    maximum_calendar_age_days: int = 3,
) -> RiskDecision:
    """
    Validate that a pending signal is recent enough
    to be considered for next-session execution.

    The calendar-day allowance is intentionally a
    little wider than one day so Friday signals can
    remain eligible on Monday.
    """

    if maximum_calendar_age_days < 1:
        return RiskDecision(
            approved=False,
            reason=(
                "Maximum pending signal age "
                "must be at least one day."
            ),
        )

    try:
        parsed_signal_date = date.fromisoformat(
            signal_date
        )
    except ValueError:
        return RiskDecision(
            approved=False,
            reason="Pending signal date is invalid.",
        )

    age_days = (
        current_date - parsed_signal_date
    ).days

    if age_days < 1:
        return RiskDecision(
            approved=False,
            reason=(
                "Pending signal is not from a "
                "prior session."
            ),
        )

    if age_days > maximum_calendar_age_days:
        return RiskDecision(
            approved=False,
            reason="Pending signal has expired.",
        )

    return RiskDecision(
        approved=True,
        reason=(
            f"Pending signal age check passed. "
            f"Age: {age_days} calendar day(s)."
        ),
    )