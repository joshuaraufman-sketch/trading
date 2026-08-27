from trading_lab.data.alpaca import get_daily_bars
from trading_lab.data.split import (
    get_development_data,
    get_holdout_data,
    get_validation_data,
)


SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]


def summarize(name, df):
    print()
    print(name.upper())
    print("-" * len(name))
    print(f"rows: {len(df)}")
    print(f"start: {df['timestamp'].min()}")
    print(f"end: {df['timestamp'].max()}")
    print(f"symbols: {sorted(df['symbol'].unique())}")


def main():
    df = get_daily_bars(
        symbols=SYMBOLS,
        start="2017-01-01",
        end="2025-12-31",
    )

    development = get_development_data(df)
    validation = get_validation_data(df)

    summarize("development", development)
    summarize("validation", validation)

    print()
    print("HOLDOUT ACCESS TEST")
    print("-------------------")

    try:
        get_holdout_data(df)
        print("FAILED: holdout was accessible without permission")
    except PermissionError as exc:
        print("PASSED: holdout correctly blocked")
        print(exc)

    holdout = get_holdout_data(
        df,
        parameters_frozen=True,
    )

    summarize("holdout", holdout)


if __name__ == "__main__":
    main()