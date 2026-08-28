from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from trading_lab.risk.market_calendar import (
    advance_trading_days,
)


NEW_YORK = ZoneInfo("America/New_York")


def calculate_planned_exit_date(
    *,
    filled_at_utc: str,
    holding_days: int,
) -> date:
    if holding_days < 1:
        raise ValueError(
            "holding_days must be at least 1"
        )

    try:
        filled_at = datetime.fromisoformat(
            filled_at_utc.replace(
                "Z",
                "+00:00",
            )
        )
    except ValueError as exc:
        raise ValueError(
            "filled_at_utc is invalid"
        ) from exc

    if filled_at.tzinfo is None:
        raise ValueError(
            "filled_at_utc must include "
            "timezone information"
        )

    entry_date = (
        filled_at
        .astimezone(NEW_YORK)
        .date()
    )

    return advance_trading_days(
        entry_date,
        holding_days,
    )


def holding_period_is_due(
    *,
    planned_exit_date: date,
    current_date: date,
) -> bool:
    return current_date >= planned_exit_date
