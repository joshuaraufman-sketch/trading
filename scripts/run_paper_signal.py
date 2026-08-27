from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

from alpaca.data.enums import DataFeed

from trading_lab.data.alpaca import get_daily_bars
from trading_lab.execution.alpaca_account import get_account_state
from trading_lab.execution.alpaca_orders import submit_paper_market_order
from trading_lab.execution.planner import build_long_order_plan
from trading_lab.risk.exposure_checks import check_existing_exposure
from trading_lab.risk.order_checks import check_order_plan
from trading_lab.strategies.sma_crossover import SMACrossoverStrategy


SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]

SMA_WINDOW = 10
STOP_LOSS_PCT = 0.02
RISK_PCT = 0.005

MAXIMUM_POSITION_PCT = 0.25
MAXIMUM_RISK_PCT = 0.005

MAX_NEW_ORDERS_PER_RUN = 1


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

    if args.submit_paper:
        print("mode: PAPER SUBMISSION ENABLED")
    else:
        print("mode: DRY RUN")

    print(
        f"maximum new orders this run: "
        f"{MAX_NEW_ORDERS_PER_RUN}"
    )

    account = get_account_state()

    print(
        f"equity=${account.equity:,.2f}, "
        f"cash=${account.cash:,.2f}"
    )

    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=45)

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
        signal = bool(latest["signal"])

        print()
        print(symbol)
        print(f"signal: {signal}")

        if not signal:
            continue

        exposure = check_existing_exposure(
            symbol=symbol,
            account=account,
        )

        if not exposure.approved:
            print("risk approved: False")
            print(f"risk reason: {exposure.reason}")
            print("action: BLOCKED")
            continue

        latest_time = latest["timestamp"]
        latest_close = float(latest["close"])

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

        print(f"quantity: {order.quantity}")
        print(
            f"estimated value: "
            f"${order.estimated_position_value:,.2f}"
        )
        print(f"risk approved: {decision.approved}")
        print(f"risk reason: {decision.reason}")

        if not decision.approved:
            print("action: BLOCKED")
            continue

        approved_candidates += 1

        if approved_candidates > MAX_NEW_ORDERS_PER_RUN:
            print(
                "action: BLOCKED — maximum new orders "
                "per run reached"
            )
            continue

        if not args.submit_paper:
            print("action: DRY RUN — no order submitted")
            continue

        response = submit_paper_market_order(
            order
        )

        submitted += 1

        print("action: PAPER ORDER SUBMITTED")
        print(f"order id: {response.id}")
        print(f"status: {response.status}")

        if submitted >= MAX_NEW_ORDERS_PER_RUN:
            print(
                "submission limit reached; "
                "no additional orders will be sent"
            )

    print()
    print("SUMMARY")
    print("-------")
    print(f"approved candidates: {approved_candidates}")
    print(f"paper orders submitted: {submitted}")


if __name__ == "__main__":
    main()