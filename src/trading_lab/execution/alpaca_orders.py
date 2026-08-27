from __future__ import annotations

import os

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
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
    Submit a market order to the Alpaca PAPER account only.
    """

    if order.side != "buy":
        raise ValueError(
            "Stage 4 currently supports buy orders only."
        )

    if order.quantity <= 0:
        raise ValueError(
            "Order quantity must be positive."
        )

    client = _get_paper_trading_client()

    request = MarketOrderRequest(
        symbol=order.symbol,
        qty=order.quantity,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
    )

    return client.submit_order(
        order_data=request,
    )