"""
Health state for an unattended system.

The point of this project is a system that runs while its owner is at
work. That only works if a failure is visible without watching it
happen. A run that crashes, or silently stops being scheduled, must be
noticeable from a single evening glance.

Two failure shapes matter and they are different:

  LOUD    a run executed and reported a problem
  QUIET   no run executed at all

The second is more dangerous and easier to miss. A crashed scheduler
task produces no output, no log, and no alert -- everything simply looks
calm. Staleness detection is the only thing that catches it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class HealthState:
    last_run_utc: str = ""
    last_session: str = ""
    last_status: str = "unknown"
    last_detail: str = ""
    consecutive_failures: int = 0

    @property
    def last_session_date(self) -> date | None:
        if not self.last_session:
            return None

        try:
            return date.fromisoformat(self.last_session)
        except ValueError:
            return None


def load_health(path: Path) -> HealthState:
    path = Path(path)

    if not path.exists():
        return HealthState()

    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError):
        return HealthState(
            last_status="corrupt",
            last_detail="health file could not be read",
        )

    return HealthState(
        last_run_utc=str(payload.get("last_run_utc", "")),
        last_session=str(payload.get("last_session", "")),
        last_status=str(payload.get("last_status", "unknown")),
        last_detail=str(payload.get("last_detail", "")),
        consecutive_failures=int(
            payload.get("consecutive_failures", 0) or 0
        ),
    )


def record_run(
    path: Path,
    *,
    session: date,
    status: str,
    detail: str = "",
) -> HealthState:
    """
    Record the outcome of a run.

    ``consecutive_failures`` accumulates so a single transient blip
    reads differently from a persistent breakage. One failed run is
    noise; four in a row is a system that has stopped working.
    """

    previous = load_health(path)

    failures = (
        0 if status == "ok" else previous.consecutive_failures + 1
    )

    state = HealthState(
        last_run_utc=datetime.now(timezone.utc).isoformat(),
        last_session=session.isoformat(),
        last_status=status,
        last_detail=detail,
        consecutive_failures=failures,
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "last_run_utc": state.last_run_utc,
                "last_session": state.last_session,
                "last_status": state.last_status,
                "last_detail": state.last_detail,
                "consecutive_failures": state.consecutive_failures,
            },
            file,
            indent=2,
        )

    return state


def sessions_since(
    last: date | None,
    today: date,
    *,
    is_trading_day,
) -> int:
    """
    Trading sessions elapsed since the last recorded run.

    Counting sessions rather than calendar days is what makes a weekend
    or a holiday indistinguishable from normal operation, so the check
    does not cry wolf every Monday morning.
    """

    if last is None:
        return -1

    count = 0
    cursor = last

    while cursor < today:
        cursor = date.fromordinal(cursor.toordinal() + 1)

        if is_trading_day(cursor):
            count += 1

    return count


def assess_health(
    state: HealthState,
    *,
    today: date,
    is_trading_day,
    stale_after_sessions: int = 2,
) -> tuple[bool, str]:
    """
    Is the system healthy? Returns (healthy, explanation).

    Catches the quiet failure: a scheduler task that stopped running
    produces no logs and no alerts, so only the age of the last
    successful run reveals it.
    """

    if state.last_status == "corrupt":
        return False, "health file is unreadable"

    if not state.last_run_utc:
        return False, "no run has ever been recorded"

    elapsed = sessions_since(
        state.last_session_date, today, is_trading_day=is_trading_day
    )

    if elapsed < 0:
        return False, "last session could not be parsed"

    if elapsed > stale_after_sessions:
        return False, (
            f"no run in {elapsed} trading sessions - the scheduled task "
            f"may have stopped"
        )

    if state.consecutive_failures >= 3:
        return False, (
            f"{state.consecutive_failures} consecutive failures: "
            f"{state.last_detail}"
        )

    if state.last_status != "ok":
        return False, (
            f"last run reported {state.last_status}: {state.last_detail}"
        )

    return True, f"last run {elapsed} session(s) ago, status ok"
