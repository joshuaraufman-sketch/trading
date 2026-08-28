import json
from datetime import date

from trading_lab.validation.forward_status import (
    summarize_forward_logs,
)


def _write_log(path, record):
    path.write_text(
        json.dumps(record),
        encoding="utf-8",
    )


def test_forward_status_summary(tmp_path):
    _write_log(
        tmp_path / "a_forward_run.json",
        {
            "run_time_utc": (
                "2026-08-27T22:00:00+00:00"
            ),
            "symbols": [
                {
                    "symbol": "SPY",
                    "signal_time": "2026-08-27 04:00:00+00:00",
                    "signal": True,
                    "action": "dry_run",
                },
                {
                    "signal": False,
                    "action": "no_signal",
                },
            ],
            "summary": {
                "approved_candidates": 1,
                "paper_orders_submitted": 0,
            },
        },
    )

    _write_log(
        tmp_path / "b_forward_run.json",
        {
            "run_time_utc": (
                "2026-08-28T14:00:00+00:00"
            ),
            "symbols": [
                {
                    "symbol": "SPY",
                    "signal_time": "2026-08-27 04:00:00+00:00",
                    "signal": True,
                    "action": (
                        "paper_order_submitted"
                    ),
                },
                {
                    "symbol": "QQQ",
                    "signal_time": "2026-08-28 04:00:00+00:00",
                    "signal": True,
                    "action": "dry_run",
                },
            ],
            "summary": {
                "approved_candidates": 1,
                "paper_orders_submitted": 1,
            },
        },
    )

    (
        tmp_path / "bad_forward_run.json"
    ).write_text(
        "not json",
        encoding="utf-8",
    )

    result = summarize_forward_logs(
        tmp_path,
        as_of=date(2026, 8, 29),
    )

    assert result.log_files == 3
    assert result.valid_logs == 2
    assert result.invalid_logs == 1
    assert result.first_run_date == date(
        2026, 8, 27
    )
    assert result.latest_run_date == date(
        2026, 8, 28
    )
    assert result.calendar_days_elapsed == 3
    assert result.unique_run_days == 2
    assert result.signal_evaluations == 3
    assert result.unique_signals == 2
    assert result.approved_candidates == 2
    assert result.paper_orders_submitted == 1
    assert result.action_counts["dry_run"] == 2


def test_forward_status_empty_directory(
    tmp_path,
):
    result = summarize_forward_logs(
        tmp_path,
        as_of=date(2026, 8, 29),
    )

    assert result.log_files == 0
    assert result.valid_logs == 0
    assert result.calendar_days_elapsed == 0
    assert result.paper_orders_submitted == 0
