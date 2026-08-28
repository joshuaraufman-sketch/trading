from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
from pathlib import Path


@dataclass(frozen=True)
class ForwardTestStatus:
    log_files: int
    valid_logs: int
    invalid_logs: int
    first_run_date: date | None
    latest_run_date: date | None
    calendar_days_elapsed: int
    unique_run_days: int
    signal_evaluations: int
    unique_signals: int
    approved_candidates: int
    paper_orders_submitted: int
    action_counts: dict[str, int]


def _parse_run_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None

    try:
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except ValueError:
        return None


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool):
        return 0

    try:
        result = int(value)
    except (TypeError, ValueError):
        return 0

    return max(result, 0)


def summarize_forward_logs(
    log_dir: str | Path,
    *,
    as_of: date | None = None,
) -> ForwardTestStatus:
    log_dir = Path(log_dir)

    if as_of is None:
        as_of = datetime.now(
            timezone.utc
        ).date()

    paths = sorted(
        log_dir.glob("*_forward_run.json")
    )

    run_dates: list[date] = []
    signal_evaluations = 0
    unique_signal_keys: set[tuple[str, str]] = set()
    approved_candidates = 0
    paper_orders_submitted = 0
    invalid_logs = 0
    actions: Counter[str] = Counter()

    for path in paths:
        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                record = json.load(file)
        except (OSError, json.JSONDecodeError):
            invalid_logs += 1
            continue

        if not isinstance(record, dict):
            invalid_logs += 1
            continue

        run_time = _parse_run_time(
            record.get("run_time_utc")
        )

        if run_time is None:
            invalid_logs += 1
            continue

        run_dates.append(run_time.date())

        symbols = record.get("symbols", [])

        if isinstance(symbols, list):
            for item in symbols:
                if not isinstance(item, dict):
                    continue

                if item.get("signal") is True:
                    signal_evaluations += 1

                    symbol = item.get("symbol")
                    signal_time = item.get("signal_time")

                    if (
                        isinstance(symbol, str)
                        and isinstance(signal_time, str)
                    ):
                        unique_signal_keys.add(
                            (symbol, signal_time)
                        )

                action = item.get("action")

                if isinstance(action, str):
                    actions[action] += 1

        summary = record.get("summary", {})

        if isinstance(summary, dict):
            approved_candidates += _nonnegative_int(
                summary.get(
                    "approved_candidates"
                )
            )
            paper_orders_submitted += (
                _nonnegative_int(
                    summary.get(
                        "paper_orders_submitted"
                    )
                )
            )

    if run_dates:
        first_run = min(run_dates)
        latest_run = max(run_dates)

        calendar_days_elapsed = max(
            (as_of - first_run).days + 1,
            0,
        )
    else:
        first_run = None
        latest_run = None
        calendar_days_elapsed = 0

    return ForwardTestStatus(
        log_files=len(paths),
        valid_logs=len(run_dates),
        invalid_logs=invalid_logs,
        first_run_date=first_run,
        latest_run_date=latest_run,
        calendar_days_elapsed=calendar_days_elapsed,
        unique_run_days=len(set(run_dates)),
        signal_evaluations=signal_evaluations,
        unique_signals=len(unique_signal_keys),
        approved_candidates=approved_candidates,
        paper_orders_submitted=(
            paper_orders_submitted
        ),
        action_counts=dict(actions),
    )
