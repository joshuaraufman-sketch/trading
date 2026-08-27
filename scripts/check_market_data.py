from trading_lab.data.alpaca import get_daily_bars
from trading_lab.data.quality import validate_bars


SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]


def main():
    print("Downloading market data...")

    df = get_daily_bars(
        symbols=SYMBOLS,
        start="2017-01-01",
        end="2025-12-31",
    )

    print()
    print(df.head())
    print()
    print(df.tail())

    report = validate_bars(df)

    print()
    print("DATA QUALITY REPORT")
    print("-------------------")

    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()