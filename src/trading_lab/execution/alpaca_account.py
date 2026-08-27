from __future__ import annotations

import os
from dataclasses import dataclass

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest
from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class AccountState:
    equity: float
    cash: float
    buying_power: float
    positions: list[dict]
    open_orders: list[dict]


def _get_trading_client() -> TradingClient:
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise RuntimeError(
            "Alpaca credentials were not found."
        )

    return TradingClient(
        api_key,
        secret_key,
        paper=True,
    )


def get_account_state() -> AccountState:
    """
    Read Alpaca paper-account state.

    This function performs no trading actions.
    """

    client = _get_trading_client()

    account = client.get_account()

    positions_raw = client.get_all_positions()

    orders_raw = client.get_orders(
        filter=GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
        )
    )

    positions = [
        {
            "symbol": position.symbol,
            "qty": float(position.qty),
            "market_value": float(position.market_value),
            "avg_entry_price": float(position.avg_entry_price),
            "unrealized_pl": float(position.unrealized_pl),
        }
        for position in positions_raw
    ]

    open_orders = [
        {
            "id": str(order.id),
            "symbol": order.symbol,
            "side": str(order.side),
            "qty": (
                float(order.qty)
                if order.qty is not None
                else None
            ),
            "status": str(order.status),
            "type": str(order.type),
        }
        for order in orders_raw
    ]

    return AccountState(
        equity=float(account.equity),
        cash=float(account.cash),
        buying_power=float(account.buying_power),
        positions=positions,
        open_orders=open_orders,
    )