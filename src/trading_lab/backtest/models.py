from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Trade:
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    fees: float = 0.0
    slippage: float = 0.0

    @property
    def gross_pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.quantity

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.fees - self.slippage

    @property
    def return_pct(self) -> float:
        capital_used = self.entry_price * self.quantity

        if capital_used <= 0:
            return 0.0

        return self.net_pnl / capital_used