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
    side: str = "buy",
) -> FillReconciliation:
    """
    Compare an actual Alpaca fill with the strategy reference price.

    Slippage is signed so that POSITIVE always means WORSE, on either
    side. A buy filling above the reference price is worse; a sell
    filling above it is better. The original version assumed buys only,
    which was safe while the strategy could only enter long positions
    and is wrong now that volatility targeting reduces exposure by
    selling -- it would have reported every good sell as a loss.
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

    if side not in {"buy", "sell"}:
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

    direction = 1.0 if side == "buy" else -1.0

    slippage_per_share = direction * (
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