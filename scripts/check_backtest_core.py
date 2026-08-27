from datetime import datetime, timezone

from trading_lab.backtest.costs import (
    calculate_fees,
    calculate_slippage,
)
from trading_lab.backtest.models import Trade
from trading_lab.backtest.position_size import calculate_position_size


def main():
    print("POSITION SIZE TEST")
    print("------------------")

    quantity = calculate_position_size(
        account_equity=100_000,
        risk_pct=0.005,
        entry_price=100,
        stop_price=98,
    )

    print(f"quantity: {quantity}")

    print()
    print("COST TEST")
    print("---------")

    fees = calculate_fees(
        quantity=quantity,
        fee_per_share=0.005,
    )

    slippage = calculate_slippage(
        price=100,
        quantity=quantity,
        slippage_bps=5,
    )

    print(f"fees: ${fees:.2f}")
    print(f"slippage: ${slippage:.2f}")

    print()
    print("TRADE TEST")
    print("----------")

    trade = Trade(
        symbol="SPY",
        entry_time=datetime(
            2025,
            1,
            2,
            tzinfo=timezone.utc,
        ),
        exit_time=datetime(
            2025,
            1,
            3,
            tzinfo=timezone.utc,
        ),
        entry_price=100,
        exit_price=104,
        quantity=quantity,
        fees=fees,
        slippage=slippage,
    )

    print(f"gross pnl: ${trade.gross_pnl:.2f}")
    print(f"net pnl: ${trade.net_pnl:.2f}")
    print(f"return: {trade.return_pct:.4%}")


if __name__ == "__main__":
    main()