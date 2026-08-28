from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from alpaca.data.enums import DataFeed

from trading_lab.data.alpaca import get_daily_bars
from trading_lab.execution.alpaca_account import get_account_state
from trading_lab.execution.alpaca_orders import submit_paper_market_order
from trading_lab.execution.pending_signals import (
    load_pending_signals,
    mark_signal_processed,
)
from trading_lab.execution.planner import build_long_order_plan
from trading_lab.execution.trade_lifecycle import create_trade_record
from trading_lab.risk.daily_order_limit import check_daily_order_limit
from trading_lab.risk.entry_window import check_next_open_entry_window
from trading_lab.risk.exposure_checks import check_existing_exposure
from trading_lab.risk.order_checks import check_order_plan
from trading_lab.risk.pending_signal_age import check_pending_signal_age
from trading_lab.risk.market_calendar import check_is_next_trading_session


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORWARD_LOG_DIR = PROJECT_ROOT / "reports" / "forward_test"

MAXIMUM_POSITION_PCT = 0.25
MAXIMUM_RISK_PCT = 0.005
MAX_NEW_ORDERS_PER_DAY = 1
MAXIMUM_PENDING_SIGNAL_AGE_DAYS = 3


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--submit-paper",
        action="store_true",
        help="Submit approved orders to the Alpaca paper account.",
    )

    return parser.parse_args()


def get_current_reference_price(
    symbol: str,
) -> float:
    """
    Fetch the latest available IEX daily bar and use
    its current close as the execution reference.

    During the opening window this is only a sizing
    reference. The actual market fill may differ.
    """

    today = datetime.now(
        timezone.utc
    ).date()

    tomorrow = today + timedelta(days=1)

    df = get_daily_bars(
        symbols=[symbol],
        start=today.isoformat(),
        end=tomorrow.isoformat(),
        use_cache=False,
        feed=DataFeed.IEX,
    )

    if df.empty:
        raise RuntimeError(
            f"No current market data available for {symbol}."
        )

    latest = (
        df.sort_values("timestamp")
        .iloc[-1]
    )

    price = float(
        latest["close"]
    )

    if price <= 0:
        raise RuntimeError(
            f"Invalid current price for {symbol}."
        )

    return price


def main():
    args = parse_args()

    print("PENDING SIGNAL EXECUTION")
    print("------------------------")

    mode = (
        "paper_submission"
        if args.submit_paper
        else "dry_run"
    )

    print(f"mode: {mode}")

    entry_window = (
        check_next_open_entry_window()
    )

    print(
        f"entry-window approved: "
        f"{entry_window.approved}"
    )
    print(
        f"entry-window reason: "
        f"{entry_window.reason}"
    )

    pending_signals = (
        load_pending_signals()
    )

    print(
        f"pending signals found: "
        f"{len(pending_signals)}"
    )

    if not pending_signals:
        return

    account = get_account_state()

    print(
        f"equity=${account.equity:,.2f}, "
        f"cash=${account.cash:,.2f}"
    )

    daily_limit = (
        check_daily_order_limit(
            log_dir=FORWARD_LOG_DIR,
            maximum_orders_per_day=(
                MAX_NEW_ORDERS_PER_DAY
            ),
        )
    )

    print(
        f"daily-order-limit approved: "
        f"{daily_limit.approved}"
    )
    print(
        f"daily-order-limit reason: "
        f"{daily_limit.reason}"
    )

    current_date = datetime.now(
        timezone.utc
    ).date()

    submitted = 0

    for signal in pending_signals:
        symbol = signal["symbol"]
        signal_path = signal["_path"]

        print()
        print(symbol)
        print(
            f"original signal date: "
            f"{signal['signal_date']}"
        )
        print(
            f"signal reference close: "
            f"${signal['signal_reference_price']:,.2f}"
        )

        age_check = check_pending_signal_age(
            signal_date=signal[
                "signal_date"
            ],
            current_date=current_date,
            maximum_calendar_age_days=(
                MAXIMUM_PENDING_SIGNAL_AGE_DAYS
            ),
        )

        print(
            f"signal-age approved: "
            f"{age_check.approved}"
        )
        print(
            f"signal-age reason: "
            f"{age_check.reason}"
        )

        if not age_check.approved:
            if (
                age_check.reason
                == "Pending signal has expired."
            ):
                mark_signal_processed(
                    signal_path,
                    status="expired",
                )

                print(
                    "action: EXPIRED — "
                    "signal marked expired"
                )
            else:
                print(
                    "action: BLOCKED — "
                    "signal age"
                )

            continue

        session_check = check_is_next_trading_session(
            signal_date=signal["signal_date"],
            current_date=current_date,
        )

        print(
            f"next-session approved: "
            f"{session_check.approved}"
        )
        print(
            f"next-session reason: "
            f"{session_check.reason}"
        )

        if not session_check.approved:
            print(
                "action: BLOCKED — "
                "not next trading session"
            )
            continue

        if not entry_window.approved:
            print(
                "action: BLOCKED — entry window"
            )
            continue

        if not daily_limit.approved:
            print(
                "action: BLOCKED — daily order limit"
            )
            continue

        exposure = check_existing_exposure(
            symbol=symbol,
            account=account,
        )

        if not exposure.approved:
            print(
                f"action: BLOCKED — "
                f"{exposure.reason}"
            )
            continue

        current_price = (
            get_current_reference_price(
                symbol
            )
        )

        print(
            f"current execution reference: "
            f"${current_price:,.2f}"
        )

        parameters = signal[
            "parameters"
        ]

        stop_loss_pct = float(
            parameters["stop_loss_pct"]
        )

        risk_pct = float(
            parameters["risk_pct"]
        )

        holding_days = int(
            parameters["holding_days"]
        )

        order = build_long_order_plan(
            symbol=symbol,
            signal_time=signal[
                "signal_time"
            ],
            reference_price=current_price,
            account_equity=account.equity,
            stop_loss_pct=stop_loss_pct,
            risk_pct=risk_pct,
        )

        decision = check_order_plan(
            order,
            account_equity=account.equity,
            available_cash=account.cash,
            maximum_position_pct=(
                MAXIMUM_POSITION_PCT
            ),
            maximum_risk_pct=(
                MAXIMUM_RISK_PCT
            ),
        )

        print(
            f"quantity: "
            f"{order.quantity}"
        )
        print(
            f"planned stop: "
            f"${order.stop_price:,.2f}"
        )
        print(
            f"estimated value: "
            f"${order.estimated_position_value:,.2f}"
        )
        print(
            f"risk approved: "
            f"{decision.approved}"
        )
        print(
            f"risk reason: "
            f"{decision.reason}"
        )

        if not decision.approved:
            print(
                "action: BLOCKED — order risk"
            )
            continue

        if not args.submit_paper:
            print(
                "action: DRY RUN — "
                "pending signal remains pending"
            )
            continue

        response = (
            submit_paper_market_order(
                order
            )
        )

        submitted += 1

        mark_signal_processed(
            signal_path,
            status="submitted",
            order_id=str(
                response.id
            ),
        )

        trade_path = create_trade_record(
            entry_order_id=str(response.id),
            symbol=symbol,
            strategy_name=signal[
                "strategy_name"
            ],
            signal_date=signal[
                "signal_date"
            ],
            signal_time=signal[
                "signal_time"
            ],
            reference_price=current_price,
            quantity=float(order.quantity),
            planned_stop_price=float(
                order.stop_price
            ),
            holding_days=holding_days,
        )

        print(
            "action: PAPER ORDER SUBMITTED"
        )
        print(
            f"trade lifecycle record: "
            f"{trade_path}"
        )
        print(
            f"order id: "
            f"{response.id}"
        )
        print(
            f"status: "
            f"{response.status}"
        )

        break

    print()
    print("SUMMARY")
    print("-------")
    print(
        f"paper orders submitted: "
        f"{submitted}"
    )


if __name__ == "__main__":
    main()