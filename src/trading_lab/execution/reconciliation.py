from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FillReconciliation:
    order_id: str
    symbol: str
    status: str
    reference_price: float
    filled_avg_price: float | None
    filled_qty: float
    slippage_per_share: float | None
    slippage_bps: float | None
    total_slippage_dollars: float | None


def reconcile_fill(
    *,
    order_id: str,
    symbol: str,
    reference_price: float,
    status: str,
    filled_avg_price: float | None,
    filled_qty: float,
) -> FillReconciliation:
    """
    Compare an actual Alpaca fill with the strategy reference price.

    Positive slippage means the buy filled worse than expected.
    Negative slippage means the buy filled better than expected.
    """

    if reference_price <= 0:
        raise ValueError(
            "reference_price must be greater than zero"
        )

    if filled_qty < 0:
        raise ValueError(
            "filled_qty cannot be negative"
        )

    if filled_avg_price is None or filled_qty == 0:
        return FillReconciliation(
            order_id=order_id,
            symbol=symbol,
            status=status,
            reference_price=reference_price,
            filled_avg_price=None,
            filled_qty=filled_qty,
            slippage_per_share=None,
            slippage_bps=None,
            total_slippage_dollars=None,
        )

    slippage_per_share = (
        filled_avg_price - reference_price
    )

    slippage_bps = (
        slippage_per_share
        / reference_price
        * 10_000
    )

    total_slippage_dollars = (
        slippage_per_share
        * filled_qty
    )

    return FillReconciliation(
        order_id=order_id,
        symbol=symbol,
        status=status,
        reference_price=reference_price,
        filled_avg_price=filled_avg_price,
        filled_qty=filled_qty,
        slippage_per_share=slippage_per_share,
        slippage_bps=slippage_bps,
        total_slippage_dollars=total_slippage_dollars,
    )