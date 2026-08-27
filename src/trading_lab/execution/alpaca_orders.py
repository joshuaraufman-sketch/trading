from __future__ import annotations

import os

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    OrderClass,
    OrderSide,
    TimeInForce,
)
from alpaca.trading.requests import (
    MarketOrderRequest,
    StopLossRequest,
)
from dotenv import load_dotenv

from trading_lab.execution.models import OrderPlan


load_dotenv()


def _get_paper_trading_client() -> TradingClient:
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


def submit_paper_market_order(
    order: OrderPlan,
):
    """
    Submit a PAPER market buy with an attached
    protective stop-loss order.

    The parent order is a market buy.
    Once the parent fills, Alpaca activates the
    stop-loss child order.
    """

    if order.side != "buy":
        raise ValueError(
            "Stage 4 currently supports buy orders only."
        )

    if order.quantity <= 0:
        raise ValueError(
            "Order quantity must be positive."
        )

    if order.stop_price <= 0:
        raise ValueError(
            "Stop price must be positive."
        )

    if order.stop_price >= order.reference_price:
        raise ValueError(
            "Stop price must be below reference price."
        )

    client = _get_paper_trading_client()

    stop_loss = StopLossRequest(
        stop_price=round(
            order.stop_price,
            2,
        ),
    )

    request = MarketOrderRequest(
        symbol=order.symbol,
        qty=order.quantity,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.OTO,
        stop_loss=stop_loss,
    )

    return client.submit_order(
        order_data=request,
    )


def get_paper_order(
    order_id: str,
):
    """
    Retrieve one order from the Alpaca paper account.
    """

    if not order_id:
        raise ValueError(
            "order_id is required."
        )

    client = _get_paper_trading_client()

    return client.get_order_by_id(
        order_id
    )