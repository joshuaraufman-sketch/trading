from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from trading_lab.execution.models import RiskDecision


NEW_YORK = ZoneInfo("America/New_York")


def check_next_open_entry_window(
    *,
    start_hour: int = 9,
    start_minute: int = 30,
    end_hour: int = 9,
    end_minute: int = 40,
) -> RiskDecision:
    """
    Allow entry only during a narrow window just after
    the regular U.S. equity market opens.
    """

    now_et = datetime.now(NEW_YORK)

    if now_et.weekday() >= 5:
        return RiskDecision(
            approved=False,
            reason="Market is closed for the weekend.",
        )

    current_minutes = (
        now_et.hour * 60
        + now_et.minute
    )

    start_minutes = (
        start_hour * 60
        + start_minute
    )

    end_minutes = (
        end_hour * 60
        + end_minute
    )

    if not (
        start_minutes
        <= current_minutes
        < end_minutes
    ):
        return RiskDecision(
            approved=False,
            reason=(
                "Outside next-session entry window."
            ),
        )

    return RiskDecision(
        approved=True,
        reason=(
            "Next-session entry window check passed."
        ),
    )