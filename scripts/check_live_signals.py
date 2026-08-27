from __future__ import annotations

from datetime import datetime, timedelta, timezone

from alpaca.data.enums import DataFeed

from trading_lab.data.alpaca import get_daily_bars
from trading_lab.execution.alpaca_account import get_account_state
from trading_lab.execution.planner import build_long_order_plan
from trading_lab.risk.order_checks import check_order_plan
from trading_lab.strategies.sma_crossover import SMACrossoverStrategy


SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]

SMA_WINDOW = 10
STOP_LOSS_PCT = 0.02
RISK_PCT = 0.005

MAXIMUM_POSITION_PCT = 0.25
MAXIMUM_RISK_PCT = 0.005


def main():
    print("Reading paper account...")

    account = get_account_state()

    print(
        f"equity=${account.equity:,.2f}, "
        f"cash=${account.cash:,.2f}"
    )

    print()
    print("Loading recent IEX market data...")

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

    print()
    print("CURRENT SIGNAL CHECK")
    print("--------------------")

    signals_found = 0

    for symbol in SYMBOLS:
        symbol_df = (
            signal_df[
                signal_df["symbol"] == symbol
            ]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        if symbol_df.empty:
            print(f"{symbol}: no data")
            continue

        latest = symbol_df.iloc[-1]

        latest_time = latest["timestamp"]
        latest_close = float(latest["close"])
        latest_sma = float(latest["sma"])
        signal = bool(latest["signal"])

        print()
        print(symbol)
        print(f"  latest bar: {latest_time}")
        print(f"  close: {latest_close:.2f}")
        print(f"  SMA({SMA_WINDOW}): {latest_sma:.2f}")
        print(f"  signal: {signal}")

        if not signal:
            continue

        signals_found += 1

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

        print()
        print("  PROPOSED ORDER")
        print(f"  quantity: {order.quantity}")
        print(
            f"  reference price: "
            f"${order.reference_price:.2f}"
        )
        print(
            f"  stop price: "
            f"${order.stop_price:.2f}"
        )
        print(
            f"  estimated value: "
            f"${order.estimated_position_value:,.2f}"
        )

        print()
        print("  RISK DECISION")
        print(f"  approved: {decision.approved}")
        print(f"  reason: {decision.reason}")

    print()
    print("SUMMARY")
    print("-------")
    print(f"signals found: {signals_found}")
    print("orders submitted: 0")


if __name__ == "__main__":
    main()