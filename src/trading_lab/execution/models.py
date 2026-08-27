from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class OrderPlan:
    symbol: str
    side: str
    quantity: int
    signal_time: datetime
    reference_price: float
    stop_price: float
    risk_per_share: float
    estimated_position_value: float


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    reason: str