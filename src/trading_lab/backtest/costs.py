from __future__ import annotations


def calculate_slippage(
    price: float,
    quantity: int,
    slippage_bps: float,
) -> float:
    """
    Estimate total slippage cost.

    1 basis point = 0.01%.
    """

    if price <= 0:
        raise ValueError("price must be greater than zero")

    if quantity <= 0:
        raise ValueError("quantity must be greater than zero")

    if slippage_bps < 0:
        raise ValueError("slippage_bps cannot be negative")

    notional = price * quantity

    return notional * (slippage_bps / 10_000)


def calculate_fees(
    quantity: int,
    fee_per_share: float = 0.0,
) -> float:
    """
    Estimate transaction fees using a simple per-share model.
    """

    if quantity <= 0:
        raise ValueError("quantity must be greater than zero")

    if fee_per_share < 0:
        raise ValueError("fee_per_share cannot be negative")

    return quantity * fee_per_share