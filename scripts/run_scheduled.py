"""
Scheduler entry point. One command, run once per weekday evening.

Gates on whether it should act, runs the daily decision, reconciles,
records health, and exits with a code a scheduler can alert on.

    python scripts/run_scheduled.py            # dry run
    python scripts/run_scheduled.py --submit   # live

Exit codes:
    0  healthy - acted or correctly declined to act
    1  a run executed and something is wrong
    2  refused to run (not a trading day, before the close, calendar
       coverage lapsed)

Code 2 is deliberately distinct. "Nothing to do" and "something broke"
must not look the same to a scheduler, or every weekend becomes an
alert and the alerts get muted.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from trading_lab.execution.health import (
    assess_health,
    load_health,
    record_run,
)
from trading_lab.risk.market_calendar import (
    holidays_cover,
    is_trading_day,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEALTH_PATH = PROJECT_ROOT / "state" / "health.json"

NEW_YORK = ZoneInfo("America/New_York")

# Bars are usually available shortly after the 16:00 close. The stale-bar
# check inside the daily runner is the real guard; this only avoids
# pointless attempts.
EARLIEST_HOUR_ET = 16
EARLIEST_MINUTE_ET = 20


def _run(script: str, *args: str) -> tuple[int, str]:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / script), *args],
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout + result.stderr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    parser.add_argument(
        "--force", action="store_true",
        help="Skip the trading-day and time gates. For testing.",
    )
    args = parser.parse_args()

    now = datetime.now(NEW_YORK)
    today = now.date()

    print(f"SCHEDULED RUN  {now:%Y-%m-%d %H:%M %Z}")
    print("=" * 66)

    if not args.force:
        if not holidays_cover(today):
            print(
                f"REFUSING: holiday table does not cover {today.year}. "
                f"Extend market_calendar.py before automating further."
            )
            sys.exit(2)

        if not is_trading_day(today):
            print(f"Not a trading day ({today}). Nothing to do.")
            sys.exit(2)

        too_early = (
            now.hour < EARLIEST_HOUR_ET
            or (
                now.hour == EARLIEST_HOUR_ET
                and now.minute < EARLIEST_MINUTE_ET
            )
        )

        if too_early:
            print(
                f"REFUSING: before {EARLIEST_HOUR_ET}:"
                f"{EARLIEST_MINUTE_ET:02d} ET. Today's bar is incomplete."
            )
            sys.exit(2)

    daily_args = ["--submit"] if args.submit else []
    daily_code, daily_output = _run("run_daily.py", *daily_args)

    print()
    print(daily_output.rstrip())

    if daily_code != 0:
        record_run(
            HEALTH_PATH, session=today, status="daily_run_blocked",
            detail=daily_output.strip().splitlines()[-1][:200]
            if daily_output.strip() else "no output",
        )
        print()
        print("DAILY RUN BLOCKED - see above")
        sys.exit(1)

    recon_code, recon_output = _run("run_reconciliation.py")

    print()
    print(recon_output.rstrip())

    if recon_code != 0:
        record_run(
            HEALTH_PATH, session=today, status="reconciliation_failed",
            detail="intended and actual exposure disagree",
        )
        print()
        print("RECONCILIATION FAILED - see above")
        sys.exit(1)

    state = record_run(HEALTH_PATH, session=today, status="ok")
    healthy, explanation = assess_health(
        state, today=today, is_trading_day=is_trading_day
    )

    print()
    print(f"HEALTH: {'OK' if healthy else 'DEGRADED'} - {explanation}")
    sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    main()
