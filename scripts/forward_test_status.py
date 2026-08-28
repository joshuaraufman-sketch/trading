from __future__ import annotations

from pathlib import Path

import yaml

from trading_lab.validation.forward_status import (
    summarize_forward_logs,
    summarize_forward_trades,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "reports" / "forward_test"
TRADE_DIR = PROJECT_ROOT / "reports" / "forward_trades"
RULES_PATH = PROJECT_ROOT / "config" / "research_rules.yaml"


def main():
    with RULES_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        rules = yaml.safe_load(file)

    paper_rules = rules["paper_trading"]

    minimum_trades = int(
        paper_rules["minimum_forward_test_trades"]
    )
    minimum_days = int(
        paper_rules["minimum_forward_test_days"]
    )

    status = summarize_forward_logs(LOG_DIR)
    trades = summarize_forward_trades(
        TRADE_DIR
    )

    print("FORWARD TEST STATUS")
    print("-------------------")
    print(f"log files: {status.log_files}")
    print(f"valid logs: {status.valid_logs}")
    print(f"invalid logs: {status.invalid_logs}")
    print(
        f"first recorded run: "
        f"{status.first_run_date or 'none'}"
    )
    print(
        f"latest recorded run: "
        f"{status.latest_run_date or 'none'}"
    )
    print(
        f"unique run days: "
        f"{status.unique_run_days}"
    )

    day_pct = (
        min(
            status.calendar_days_elapsed
            / minimum_days
            * 100,
            100,
        )
        if minimum_days
        else 0
    )

    print()
    print("60-DAY REQUIREMENT")
    print("------------------")
    print(
        f"calendar days elapsed: "
        f"{status.calendar_days_elapsed}"
        f"/{minimum_days} "
        f"({day_pct:.1f}%)"
    )

    print()
    print("FORWARD ACTIVITY")
    print("----------------")
    print(
        f"positive signal evaluations: "
        f"{status.signal_evaluations}"
    )
    print(
        f"unique signals: "
        f"{status.unique_signals}"
    )
    print(
        f"approved candidates: "
        f"{status.approved_candidates}"
    )
    print(
        f"paper entries submitted: "
        f"{status.paper_orders_submitted}"
    )

    print()
    print("30-TRADE REQUIREMENT")
    print("--------------------")

    trade_pct = (
        min(
            trades.completed_trades
            / minimum_trades
            * 100,
            100,
        )
        if minimum_trades
        else 0
    )

    print(
        f"completed round-trip trades: "
        f"{trades.completed_trades}"
        f"/{minimum_trades} "
        f"({trade_pct:.1f}%)"
    )
    print(
        f"open trades: "
        f"{trades.open_trades}"
    )
    print(
        f"submitted awaiting fill: "
        f"{trades.submitted_trades}"
    )
    print(
        f"invalid lifecycle records: "
        f"{trades.invalid_records}"
    )
    print(
        f"realized P&L (closed trades): "
        f"${trades.total_realized_pnl:,.2f}"
    )

    trade_requirement_met = (
        trades.completed_trades
        >= minimum_trades
    )

    print(
        "status: "
        + (
            "REQUIREMENT MET"
            if trade_requirement_met
            else "IN PROGRESS"
        )
    )

    if status.action_counts:
        print()
        print("RECORDED ACTIONS")
        print("----------------")

        for action, count in sorted(
            status.action_counts.items()
        ):
            print(f"{action}: {count}")


if __name__ == "__main__":
    main()
