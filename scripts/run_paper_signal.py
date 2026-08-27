from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from alpaca.data.enums import DataFeed

from trading_lab.data.alpaca import get_daily_bars
from trading_lab.execution.alpaca_account import get_account_state
from trading_lab.execution.alpaca_orders import submit_paper_market_order
from trading_lab.execution.forward_log import save_forward_run
from trading_lab.execution.planner import build_long_order_plan
from trading_lab.risk.daily_order_limit import check_daily_order_limit
from trading_lab.risk.exposure_checks import check_existing_exposure
from trading_lab.risk.order_checks import check_order_plan
from trading_lab.risk.session_checks import (
    check_execution_window,
    check_signal_freshness,
)
from trading_lab.strategies.sma_crossover import SMACrossoverStrategy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORWARD_LOG_DIR = PROJECT_ROOT / "reports" / "forward_test"

SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]

SMA_WINDOW = 10
STOP_LOSS_PCT = 0.02
RISK_PCT = 0.005

MAXIMUM_POSITION_PCT = 0.25
MAXIMUM_RISK_PCT = 0.005
MAX_NEW_ORDERS_PER_RUN = 1
MAX_NEW_ORDERS_PER_DAY = 1
MAXIMUM_SIGNAL_AGE_DAYS = 1


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--submit-paper",
        action="store_true",
        help="Actually submit approved orders to the Alpaca paper account.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("PAPER TRADING SIGNAL RUN")
    print("------------------------")

    mode = (
        "paper_submission"
        if args.submit_paper
        else "dry_run"
    )

    print(f"mode: {mode}")
    print(
        f"maximum new orders this run: "
        f"{MAX_NEW_ORDERS_PER_RUN}"
    )
    print(
        f"maximum new orders per day: "
        f"{MAX_NEW_ORDERS_PER_DAY}"
    )

    session = check_execution_window()

    daily_limit = check_daily_order_limit(
        log_dir=FORWARD_LOG_DIR,
        maximum_orders_per_day=MAX_NEW_ORDERS_PER_DAY,
    )

    print(
        f"market-session approved: "
        f"{session.approved}"
    )
    print(
        f"market-session reason: "
        f"{session.reason}"
    )
    print(
        f"daily-order-limit approved: "
        f"{daily_limit.approved}"
    )
    print(
        f"daily-order-limit reason: "
        f"{daily_limit.reason}"
    )

    account = get_account_state()

    print(
        f"equity=${account.equity:,.2f}, "
        f"cash=${account.cash:,.2f}"
    )

    run_record = {
        "run_time_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "mode": mode,
        "strategy": {
            "name": "sma_crossover",
            "sma_window": SMA_WINDOW,
            "holding_days": 10,
            "stop_loss_pct": STOP_LOSS_PCT,
            "risk_pct": RISK_PCT,
            "market_data_feed": "IEX",
        },
        "account": {
            "equity": account.equity,
            "cash": account.cash,
            "buying_power": account.buying_power,
            "positions": account.positions,
            "open_orders": account.open_orders,
        },
        "session_check": {
            "approved": session.approved,
            "reason": session.reason,
        },
        "daily_order_limit_check": {
            "approved": daily_limit.approved,
            "reason": daily_limit.reason,
        },
        "symbols": [],
    }

    end_date = datetime.now(
        timezone.utc
    ).date()

    start_date = (
        end_date - timedelta(days=45)
    )

    df = get_daily_bars(
        symbols=SYMBOLS,
        start=start_date.isoformat(),
        end=(end_date + timedelta(days=1)).isoformat(),
        use_cache=False,
        feed=DataFeed.IEX,
    )

    strategy = SMACrossoverStrategy(
        window=SMA_WINDOW,
    )

    signal_df = strategy.generate_signals(df)

    submitted = 0
    approved_candidates = 0

    for symbol in SYMBOLS:
        symbol_df = (
            signal_df[
                signal_df["symbol"] == symbol
            ]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        if symbol_df.empty:
            continue

        latest = symbol_df.iloc[-1]

        latest_time = latest["timestamp"]
        latest_close = float(
            latest["close"]
        )
        signal = bool(
            latest["signal"]
        )

        symbol_record = {
            "symbol": symbol,
            "signal": signal,
            "signal_time": latest_time,
            "reference_price": latest_close,
            "action": None,
        }

        print()
        print(symbol)
        print(f"signal: {signal}")
        print(f"signal bar: {latest_time}")

        if not signal:
            symbol_record["action"] = "no_signal"
            run_record["symbols"].append(
                symbol_record
            )
            continue

        freshness = check_signal_freshness(
            signal_time=latest_time,
            maximum_age_days=MAXIMUM_SIGNAL_AGE_DAYS,
        )

        symbol_record["freshness_check"] = {
            "approved": freshness.approved,
            "reason": freshness.reason,
        }

        if not freshness.approved:
            print("action: BLOCKED")
            symbol_record["action"] = (
                "blocked_stale_signal"
            )
            run_record["symbols"].append(
                symbol_record
            )
            continue

        if not session.approved:
            print("action: BLOCKED")
            symbol_record["action"] = (
                "blocked_market_session"
            )
            run_record["symbols"].append(
                symbol_record
            )
            continue

        if not daily_limit.approved:
            print("action: BLOCKED")
            symbol_record["action"] = (
                "blocked_daily_order_limit"
            )
            run_record["symbols"].append(
                symbol_record
            )
            continue

        exposure = check_existing_exposure(
            symbol=symbol,
            account=account,
        )

        symbol_record["exposure_check"] = {
            "approved": exposure.approved,
            "reason": exposure.reason,
        }

        if not exposure.approved:
            print("action: BLOCKED")
            symbol_record["action"] = (
                "blocked_existing_exposure"
            )
            run_record["symbols"].append(
                symbol_record
            )
            continue

        order = build_long_order_plan(
            symbol=symbol,
            signal_time=latest_time,
            reference_price=latest_close,
            account_equity=account.equity,
            stop_loss_pct=STOP_LOSS_PCT,
            risk_pct=RISK_PCT,
        )

        decision = check_order_plan(
            order,
            account_equity=account.equity,
            available_cash=account.cash,
            maximum_position_pct=MAXIMUM_POSITION_PCT,
            maximum_risk_pct=MAXIMUM_RISK_PCT,
        )

        symbol_record["order_plan"] = {
            "quantity": order.quantity,
            "reference_price": order.reference_price,
            "stop_price": order.stop_price,
            "estimated_position_value": (
                order.estimated_position_value
            ),
        }

        symbol_record["order_risk_check"] = {
            "approved": decision.approved,
            "reason": decision.reason,
        }

        if not decision.approved:
            print("action: BLOCKED")
            symbol_record["action"] = (
                "blocked_order_risk"
            )
            run_record["symbols"].append(
                symbol_record
            )
            continue

        approved_candidates += 1

        if (
            approved_candidates
            > MAX_NEW_ORDERS_PER_RUN
        ):
            print("action: BLOCKED — order limit")
            symbol_record["action"] = (
                "blocked_order_limit"
            )
            run_record["symbols"].append(
                symbol_record
            )
            continue

        if not args.submit_paper:
            print(
                "action: DRY RUN — no order submitted"
            )
            symbol_record["action"] = "dry_run"
            run_record["symbols"].append(
                symbol_record
            )
            continue

        response = submit_paper_market_order(
            order
        )

        submitted += 1

        symbol_record["action"] = (
            "paper_order_submitted"
        )

        symbol_record["alpaca_order"] = {
            "id": str(response.id),
            "status": str(response.status),
        }

        run_record["symbols"].append(
            symbol_record
        )

        print("action: PAPER ORDER SUBMITTED")
        print(f"order id: {response.id}")
        print(f"status: {response.status}")

    run_record["summary"] = {
        "approved_candidates": (
            approved_candidates
        ),
        "paper_orders_submitted": submitted,
    }

    log_path = save_forward_run(
        run_record
    )

    print()
    print("SUMMARY")
    print("-------")
    print(
        f"approved candidates: "
        f"{approved_candidates}"
    )
    print(
        f"paper orders submitted: "
        f"{submitted}"
    )

    print()
    print("FORWARD TEST LOG")
    print("----------------")
    print(log_path)


if __name__ == "__main__":
    main()