"""
The evening glance. One command that says whether anything needs you.

Catches the quiet failure specifically: a scheduled task that stopped
running produces no logs, no errors and no alerts. Everything looks
calm. Only the age of the last successful run reveals it.

    python scripts/check_health.py

Exit 0 healthy, 1 needs attention.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from trading_lab.execution.health import assess_health, load_health
from trading_lab.risk.market_calendar import (
    COVERED_YEARS,
    holidays_cover,
    is_trading_day,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEALTH_PATH = PROJECT_ROOT / "state" / "health.json"

NEW_YORK = ZoneInfo("America/New_York")


def main() -> None:
    today = datetime.now(NEW_YORK).date()
    state = load_health(HEALTH_PATH)
    healthy, explanation = assess_health(
        state, today=today, is_trading_day=is_trading_day
    )

    print("SYSTEM HEALTH")
    print("=" * 66)
    print(f"last run (UTC)     {state.last_run_utc or 'never'}")
    print(f"last session       {state.last_session or 'never'}")
    print(f"last status        {state.last_status}")

    if state.last_detail:
        print(f"detail             {state.last_detail}")

    print(f"consecutive fails  {state.consecutive_failures}")
    print()
    print(f"{'HEALTHY' if healthy else 'NEEDS ATTENTION'}: {explanation}")

    if not holidays_cover(today):
        print()
        print(
            f"WARNING: the holiday table covers "
            f"{sorted(COVERED_YEARS)} and today is {today.year}. "
            f"Trading-day detection is unreliable."
        )
        healthy = False

    sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    main()
