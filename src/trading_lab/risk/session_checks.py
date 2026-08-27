from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from trading_lab.execution.models import RiskDecision


NEW_YORK = ZoneInfo("America/New_York")


def check_signal_freshness(
    *,
    signal_time,
    maximum_age_days: int = 1,
) -> RiskDecision:
    """
    Reject signals that are older than the allowed age.
    """

    if signal_time is None:
        return RiskDecision(
            approved=False,
            reason="Signal timestamp is missing.",
        )

    now = datetime.now(timezone.utc)

    age = now - signal_time.to_pydatetime()

    if age.days > maximum_age_days:
        return RiskDecision(
            approved=False,
            reason="Signal is stale.",
        )

    return RiskDecision(
        approved=True,
        reason="Signal freshness check passed.",
    )


def check_execution_window() -> RiskDecision:
    """
    Allow paper entry only during regular U.S. equity market hours.

    This is intentionally conservative.
    """

    now_et = datetime.now(NEW_YORK)

    if now_et.weekday() >= 5:
        return RiskDecision(
            approved=False,
            reason="Market is closed for the weekend.",
        )

    minutes = (
        now_et.hour * 60
        + now_et.minute
    )

    market_open = 9 * 60 + 30
    market_close = 16 * 60

    if not market_open <= minutes < market_close:
        return RiskDecision(
            approved=False,
            reason="Outside regular market hours.",
        )

    return RiskDecision(
        approved=True,
        reason="Execution window check passed.",
    )