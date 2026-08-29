"""
Tests for unattended-system health tracking.

The staleness tests matter most. A run that fails is loud. A scheduled
task that stopped running is silent -- no logs, no errors, no alerts,
everything looks calm. Only the age of the last run reveals it.
"""

from __future__ import annotations

from datetime import date

import pytest

from trading_lab.execution.health import (
    assess_health,
    load_health,
    record_run,
    sessions_since,
)
from trading_lab.risk.market_calendar import (
    COVERED_YEARS,
    holidays_cover,
    is_trading_day,
)


def test_first_run_records_ok(tmp_path):
    path = tmp_path / "health.json"
    state = record_run(path, session=date(2026, 8, 28), status="ok")

    assert state.last_status == "ok"
    assert state.consecutive_failures == 0
    assert load_health(path).last_session == "2026-08-28"


def test_failures_accumulate_then_reset(tmp_path):
    """
    One failed run is noise. Four in a row is a system that has stopped
    working, and the two should not read the same.
    """

    path = tmp_path / "health.json"

    for i in range(3):
        state = record_run(
            path, session=date(2026, 8, 24 + i),
            status="daily_run_blocked", detail="stale bars",
        )

    assert state.consecutive_failures == 3

    state = record_run(path, session=date(2026, 8, 28), status="ok")
    assert state.consecutive_failures == 0


def test_missing_health_file_is_not_healthy(tmp_path):
    healthy, why = assess_health(
        load_health(tmp_path / "nope.json"),
        today=date(2026, 8, 28),
        is_trading_day=is_trading_day,
    )

    assert not healthy
    assert "no run has ever been recorded" in why


def test_corrupt_health_file_is_not_trusted(tmp_path):
    path = tmp_path / "health.json"
    path.write_text("{ garbled")

    healthy, why = assess_health(
        load_health(path),
        today=date(2026, 8, 28),
        is_trading_day=is_trading_day,
    )

    assert not healthy
    assert "unreadable" in why


def test_weekend_gap_is_not_stale(tmp_path):
    """
    A Friday run checked on Monday must read healthy, or the check
    cries wolf every weekend and gets ignored.
    """

    path = tmp_path / "health.json"
    record_run(path, session=date(2026, 8, 28), status="ok")  # Friday

    healthy, _ = assess_health(
        load_health(path),
        today=date(2026, 8, 31),                              # Monday
        is_trading_day=is_trading_day,
    )

    assert healthy


def test_silent_stoppage_is_caught(tmp_path):
    """
    THE QUIET FAILURE. The task stopped running a week ago. No errors
    were produced because nothing ran.
    """

    path = tmp_path / "health.json"
    record_run(path, session=date(2026, 8, 17), status="ok")

    healthy, why = assess_health(
        load_health(path),
        today=date(2026, 8, 28),
        is_trading_day=is_trading_day,
    )

    assert not healthy
    assert "may have stopped" in why


def test_sessions_since_skips_holidays():
    # 2026-11-26 is Thanksgiving, so Wed 25th to Fri 27th is one session.
    assert sessions_since(
        date(2026, 11, 25), date(2026, 11, 27),
        is_trading_day=is_trading_day,
    ) == 1


def test_repeated_failures_are_unhealthy_even_if_recent(tmp_path):
    path = tmp_path / "health.json"

    for i in range(3):
        record_run(
            path, session=date(2026, 8, 26 + i),
            status="reconciliation_failed", detail="exposure gap",
        )

    healthy, why = assess_health(
        load_health(path),
        today=date(2026, 8, 28),
        is_trading_day=is_trading_day,
    )

    assert not healthy
    assert "consecutive failures" in why


# ---------------------------------------------------------------------
# Calendar coverage
# ---------------------------------------------------------------------


def test_holiday_coverage_is_declared():
    """
    The original table covered 2026 only, so on 1 January 2027 every
    2027 holiday would have been treated as a normal trading day --
    silently, with no error.
    """

    assert holidays_cover(date(2026, 6, 1))
    assert holidays_cover(date(2027, 6, 1))
    assert not holidays_cover(date(2035, 6, 1))
    assert 2026 in COVERED_YEARS


def test_known_holidays_are_not_trading_days():
    for day in (
        date(2026, 12, 25),
        date(2027, 1, 1),
        date(2027, 11, 25),
        date(2028, 7, 4),
    ):
        assert not is_trading_day(day), f"{day} should be a holiday"


def test_ordinary_weekdays_are_trading_days():
    assert is_trading_day(date(2027, 3, 3))
    assert is_trading_day(date(2028, 3, 2))


def test_weekends_are_never_trading_days():
    assert not is_trading_day(date(2026, 8, 29))   # Saturday
    assert not is_trading_day(date(2026, 8, 30))   # Sunday
