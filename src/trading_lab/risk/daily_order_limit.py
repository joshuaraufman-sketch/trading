from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from trading_lab.execution.models import RiskDecision


NEW_YORK = ZoneInfo("America/New_York")


def count_submitted_orders_today(
    log_dir: Path,
) -> int:
    """
    Count paper orders recorded as submitted today
    across all forward-test log files.
    """

    today_et = datetime.now(
        NEW_YORK
    ).date()

    if not log_dir.exists():
        return 0

    submitted = 0

    for path in log_dir.glob(
        "*_forward_run.json"
    ):
        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                record = json.load(file)
        except (
            OSError,
            json.JSONDecodeError,
        ):
            continue

        run_time = record.get(
            "run_time_utc"
        )

        if not run_time:
            continue

        try:
            run_datetime = (
                datetime.fromisoformat(
                    run_time
                )
            )
        except ValueError:
            continue

        run_date_et = (
            run_datetime
            .astimezone(NEW_YORK)
            .date()
        )

        if run_date_et != today_et:
            continue

        summary = record.get(
            "summary",
            {},
        )

        submitted += int(
            summary.get(
                "paper_orders_submitted",
                0,
            )
        )

    return submitted


def check_daily_order_limit(
    *,
    log_dir: Path,
    maximum_orders_per_day: int,
) -> RiskDecision:
    if maximum_orders_per_day <= 0:
        return RiskDecision(
            approved=False,
            reason=(
                "Maximum orders per day "
                "must be positive."
            ),
        )

    already_submitted = (
        count_submitted_orders_today(
            log_dir
        )
    )

    if (
        already_submitted
        >= maximum_orders_per_day
    ):
        return RiskDecision(
            approved=False,
            reason=(
                "Daily paper order limit "
                "already reached."
            ),
        )

    return RiskDecision(
        approved=True,
        reason=(
            f"Daily order limit check passed. "
            f"{already_submitted}/"
            f"{maximum_orders_per_day} "
            f"orders submitted today."
        ),
    )