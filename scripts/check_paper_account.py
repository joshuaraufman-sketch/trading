from trading_lab.execution.alpaca_account import (
    get_account_state,
)


def main():
    print("Reading Alpaca paper account...")
    print()

    state = get_account_state()

    print("ACCOUNT")
    print("-------")
    print(f"equity: ${state.equity:,.2f}")
    print(f"cash: ${state.cash:,.2f}")
    print(f"buying_power: ${state.buying_power:,.2f}")

    print()
    print("POSITIONS")
    print("---------")

    if not state.positions:
        print("None")
    else:
        for position in state.positions:
            print(position)

    print()
    print("OPEN ORDERS")
    print("-----------")

    if not state.open_orders:
        print("None")
    else:
        for order in state.open_orders:
            print(order)


if __name__ == "__main__":
    main()