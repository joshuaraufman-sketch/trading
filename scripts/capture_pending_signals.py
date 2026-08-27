from __future__ import annotations

from datetime import datetime, timedelta, timezone

from alpaca.data.enums import DataFeed

from trading_lab.data.alpaca import get_daily_bars
from trading_lab.execution.pending_signals import (
    save_pending_signal,
)
from trading_lab.strategies.sma_crossover import (
    SMACrossoverStrategy,
)


SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]

SMA_WINDOW = 10
HOLDING_DAYS = 10
STOP_LOSS_PCT = 0.02
RISK_PCT = 0.005


def main():
    print("AFTER-CLOSE SIGNAL CAPTURE")
    print("--------------------------")

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

    saved = 0

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

        signal = bool(
            latest["signal"]
        )

        signal_time = (
            latest["timestamp"]
        )

        close = float(
            latest["close"]
        )

        print()
        print(symbol)
        print(f"signal: {signal}")
        print(f"signal bar: {signal_time}")
        print(f"signal close: ${close:,.2f}")

        if not signal:
            continue

        path = save_pending_signal(
            symbol=symbol,
            signal_time=signal_time,
            signal_reference_price=close,
            strategy_name="sma_crossover",
            parameters={
                "sma_window": SMA_WINDOW,
                "holding_days": HOLDING_DAYS,
                "stop_loss_pct": STOP_LOSS_PCT,
                "risk_pct": RISK_PCT,
            },
        )

        saved += 1

        print("action: PENDING SIGNAL SAVED")
        print(f"path: {path}")

    print()
    print("SUMMARY")
    print("-------")
    print(f"pending signals saved: {saved}")


if __name__ == "__main__":
    main()