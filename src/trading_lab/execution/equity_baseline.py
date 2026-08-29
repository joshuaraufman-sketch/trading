"""
Normalized equity reporting.

Run logs are the forward-test audit trail: they record what the system
actually decided and submitted, and unlike backtest output they can
never be regenerated. So they belong in version control -- which for
this repository means they are PUBLIC.

That makes raw dollar figures a problem. On a paper account an equity
line is harmless, but the same code pointed at a live account would
publish a real balance to GitHub every trading day. Order notionals leak
the same information: 900 shares of SPY at $600 states the account size
as plainly as the equity field does.

Everything is therefore logged as a ratio. This is not only safer, it is
the more useful unit: the paper account holds $500,000 while the
backtests assume $100,000, and only percentages compare across them.

The baseline itself lives in a gitignored local file, so the one place a
dollar figure exists is never committed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def account_fingerprint(account_id: str) -> str:
    """
    Stable, non-identifying tag for an account.

    Detects a credential swap exactly as well as the raw id -- a
    different account yields a different fingerprint -- without putting
    the identifier itself in a public log.
    """

    if not account_id:
        return ""

    return hashlib.sha256(
        account_id.encode("utf-8")
    ).hexdigest()[:12]


@dataclass(frozen=True)
class EquityBaseline:
    account_fingerprint: str
    baseline_equity: float
    recorded_utc: str

    def ratio(self, equity: float) -> float:
        """Equity as a multiple of the baseline. 1.0 means flat."""

        if self.baseline_equity <= 0:
            raise ValueError("baseline equity must be positive")

        return equity / self.baseline_equity


def load_or_create_baseline(
    path: Path,
    *,
    account_id: str,
    equity: float,
) -> tuple[EquityBaseline, bool]:
    """
    Read the stored baseline, or establish one on first run.

    Returns the baseline and whether it was just created or reset.

    A baseline recorded against a different account is RESET rather than
    reused. Carrying one across an account swap would silently report a
    fictional cumulative return -- which is precisely the class of
    error that the paper-account replacement already caused once.
    """

    if equity <= 0:
        raise ValueError("equity must be positive")

    fingerprint = account_fingerprint(account_id)
    path = Path(path)

    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as file:
                payload = json.load(file)

            existing = EquityBaseline(
                account_fingerprint=str(
                    payload["account_fingerprint"]
                ),
                baseline_equity=float(payload["baseline_equity"]),
                recorded_utc=str(payload["recorded_utc"]),
            )

            if existing.account_fingerprint == fingerprint:
                return existing, False

        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            # A corrupt baseline is reset rather than trusted. Reporting
            # returns against a garbled figure is worse than starting
            # the series again.
            pass

    baseline = EquityBaseline(
        account_fingerprint=fingerprint,
        baseline_equity=float(equity),
        recorded_utc=datetime.now(timezone.utc).isoformat(),
    )

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "account_fingerprint": baseline.account_fingerprint,
                "baseline_equity": baseline.baseline_equity,
                "recorded_utc": baseline.recorded_utc,
            },
            file,
            indent=2,
        )

    return baseline, True


def normalize_order(order, equity: float) -> dict:
    """
    Describe an order without revealing account size.

    Weights and the notional-as-a-fraction-of-equity carry everything
    needed to audit the decision. Share counts and dollar notionals are
    omitted; Alpaca's own order history remains the source of truth for
    fills, and it is not public.
    """

    if equity <= 0:
        raise ValueError("equity must be positive")

    return {
        "symbol": order.symbol,
        "side": order.side,
        "notional_pct": order.notional / equity,
        "current_weight": order.current_weight,
        "target_weight": order.target_weight,
        "weight_delta": order.target_weight - order.current_weight,
    }
