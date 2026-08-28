from __future__ import annotations

import argparse

from trading_lab.execution.alpaca_orders import get_paper_order
from trading_lab.execution.reconciliation import reconcile_fill
from trading_lab.execution.trade_lifecycle import update_entry_fill


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve an Alpaca paper order and compare "
            "its fill with the strategy reference price."
        )
    )

    parser.add_argument(
        "order_id",
        help="Alpaca paper order ID.",
    )

    parser.add_argument(
        "--reference-price",
        type=float,
        required=True,
        help=(
            "Strategy reference price used when "
            "the order was planned."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    order = get_paper_order(
        args.order_id
    )

    filled_avg_price = (
        float(order.filled_avg_price)
        if order.filled_avg_price is not None
        else None
    )

    filled_qty = (
        float(order.filled_qty)
        if order.filled_qty is not None
        else 0.0
    )

    result = reconcile_fill(
        order_id=str(order.id),
        symbol=str(order.symbol),
        reference_price=args.reference_price,
        status=str(order.status),
        filled_avg_price=filled_avg_price,
        filled_qty=filled_qty,
    )

    try:
        lifecycle_path = update_entry_fill(
            entry_order_id=str(order.id),
            status=str(order.status),
            filled_qty=filled_qty,
            filled_avg_price=filled_avg_price,
        )
    except FileNotFoundError:
        lifecycle_path = None

    print("PAPER ORDER RECONCILIATION")
    print("--------------------------")
    print(f"order id: {result.order_id}")
    print(f"symbol: {result.symbol}")
    print(f"status: {result.status}")

    if lifecycle_path is not None:
        print(
            f"lifecycle record updated: "
            f"{lifecycle_path}"
        )
    else:
        print(
            "lifecycle record: not found "
            "(legacy/untracked order)"
        )
    print(
        f"reference price: "
        f"${result.reference_price:,.4f}"
    )
    print(
        f"filled quantity: "
        f"{result.filled_qty:g}"
    )

    if result.filled_avg_price is None:
        print("filled average price: not filled")
        print("slippage: not available")
        return

    print(
        f"filled average price: "
        f"${result.filled_avg_price:,.4f}"
    )
    print(
        f"slippage/share: "
        f"${result.slippage_per_share:,.4f}"
    )
    print(
        f"slippage: "
        f"{result.slippage_bps:,.2f} bps"
    )
    print(
        f"total slippage: "
        f"${result.total_slippage_dollars:,.2f}"
    )


if __name__ == "__main__":
    main()