from trading_lab.backtest.benchmark import (
    calculate_buy_and_hold_return,
)
from trading_lab.backtest.metrics import (
    build_equity_curve,
    calculate_metrics,
)
from trading_lab.backtest.runner import (
    run_long_signal_backtest,
)
from trading_lab.data.alpaca import get_daily_bars
from trading_lab.data.split import get_development_data


SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]


def main():
    df = get_daily_bars(
        symbols=SYMBOLS,
        start="2017-01-01",
        end="2025-12-31",
    )

    df = get_development_data(df)

    df = (
        df
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )

    df["sma_20"] = (
        df.groupby("symbol")["close"]
        .transform(lambda x: x.rolling(20).mean())
    )

    previous_close = (
        df.groupby("symbol")["close"]
        .shift(1)
    )

    previous_sma = (
        df.groupby("symbol")["sma_20"]
        .shift(1)
    )

    df["signal"] = (
        (df["close"] > df["sma_20"])
        & (previous_close <= previous_sma)
    )

    trades = run_long_signal_backtest(
        df,
        starting_equity=100_000,
        risk_pct=0.005,
        stop_loss_pct=0.02,
        holding_days=5,
        slippage_bps=5,
        fee_per_share=0.005,
    )

    metrics = calculate_metrics(
        trades,
        starting_equity=100_000,
    )

    print("STRATEGY METRICS")
    print("----------------")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print()
    print("BENCHMARK RETURNS")
    print("-----------------")

    for symbol in SYMBOLS:
        result = calculate_buy_and_hold_return(
            df,
            symbol,
        )
        print(f"{symbol}: {result:.2%}")

    print()
    print("EQUITY CURVE SAMPLE")
    print("-------------------")

    curve = build_equity_curve(
        trades,
        starting_equity=100_000,
    )

    print(curve.tail())


if __name__ == "__main__":
    main()