import json
from datetime import datetime, timezone

from trading_lab.risk.daily_order_limit import (
    check_daily_order_limit,
    count_submitted_orders_today,
)


def write_log(
    directory,
    *,
    submitted,
):
    record = {
        "run_time_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "summary": {
            "paper_orders_submitted": (
                submitted
            ),
        },
    }

    path = (
        directory
        / f"log_{submitted}.json"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            record,
            file,
        )


def test_no_logs_means_zero_orders(
    tmp_path,
):
    assert (
        count_submitted_orders_today(
            tmp_path
        )
        == 0
    )


def test_submitted_order_is_counted(
    tmp_path,
):
    path = (
        tmp_path
        / "test_forward_run.json"
    )

    record = {
        "run_time_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "summary": {
            "paper_orders_submitted": 1,
        },
    }

    path.write_text(
        json.dumps(record),
        encoding="utf-8",
    )

    assert (
        count_submitted_orders_today(
            tmp_path
        )
        == 1
    )


def test_daily_limit_blocks_after_one(
    tmp_path,
):
    path = (
        tmp_path
        / "test_forward_run.json"
    )

    record = {
        "run_time_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "summary": {
            "paper_orders_submitted": 1,
        },
    }

    path.write_text(
        json.dumps(record),
        encoding="utf-8",
    )

    decision = check_daily_order_limit(
        log_dir=tmp_path,
        maximum_orders_per_day=1,
    )

    assert decision.approved is False


def test_daily_limit_passes_when_unused(
    tmp_path,
):
    decision = check_daily_order_limit(
        log_dir=tmp_path,
        maximum_orders_per_day=1,
    )

    assert decision.approved is True