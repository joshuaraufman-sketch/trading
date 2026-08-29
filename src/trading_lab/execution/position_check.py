"""
Did the account end up where the plan intended?

Slippage is a diagnostic. This is the safety question. An order that was
rejected, partially filled, or never submitted leaves the account
holding something other than what the strategy decided, and nothing
currently notices. For a system that runs while its owner is at work,
silent divergence is the failure mode that matters.

Reconciliation compares INTENDED weights against ACTUAL weights and
classifies any gap. It runs in the evening against the day's run log and
the broker's own record of what happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Overnight price movement alone will shift a held weight slightly
# between the decision close and the next session. That is expected
# drift, not a failure, so the tolerance sits above typical daily
# movement but well below the rebalance band.
DEFAULT_TOLERANCE = 0.02


@dataclass(frozen=True)
class WeightDiscrepancy:
    symbol: str
    intended_weight: float
    actual_weight: float
    drift: float
    severity: str
    likely_cause: str

    @property
    def within_tolerance(self) -> bool:
        return self.severity == "ok"


@dataclass
class ReconciliationReport:
    discrepancies: list[WeightDiscrepancy] = field(default_factory=list)
    order_issues: list[dict] = field(default_factory=list)
    checked_symbols: int = 0

    @property
    def clean(self) -> bool:
        return not self.problems and not self.order_issues

    @property
    def problems(self) -> list[WeightDiscrepancy]:
        return [d for d in self.discrepancies if not d.within_tolerance]


def classify_orders(
    planned: list[dict],
    broker_orders: list[dict],
) -> list[dict]:
    """
    Match planned orders against what the broker actually recorded.

    An order that was planned but has no broker record is the most
    serious case: the system believed it acted and did not.
    """

    issues: list[dict] = []

    by_symbol: dict[str, list[dict]] = {}

    for order in broker_orders:
        by_symbol.setdefault(order.get("symbol", ""), []).append(order)

    for plan in planned:
        symbol = plan.get("symbol", "")
        matches = by_symbol.get(symbol, [])

        if not matches:
            issues.append(
                {
                    "symbol": symbol,
                    "issue": "planned order has no broker record",
                    "severity": "critical",
                    "detail": (
                        "the system believed it submitted this order "
                        "and the broker has no trace of it"
                    ),
                }
            )
            continue

        for match in matches:
            status = str(match.get("status", "")).lower()

            if "rejected" in status or "canceled" in status:
                issues.append(
                    {
                        "symbol": symbol,
                        "issue": f"order {status}",
                        "severity": "critical",
                        "detail": (
                            f"broker reports status {status}; intended "
                            f"exposure was not achieved"
                        ),
                    }
                )
            elif "partial" in status:
                issues.append(
                    {
                        "symbol": symbol,
                        "issue": "partial fill",
                        "severity": "warning",
                        "detail": (
                            f"filled {match.get('filled_qty', '?')} of "
                            f"{match.get('qty', '?')}"
                        ),
                    }
                )
            elif status and "filled" not in status:
                issues.append(
                    {
                        "symbol": symbol,
                        "issue": f"unresolved status: {status}",
                        "severity": "warning",
                        "detail": "order neither filled nor rejected",
                    }
                )

    return issues


def reconcile_positions(
    *,
    intended_weights: dict[str, float],
    actual_quantities: dict[str, float],
    prices: dict[str, float],
    equity: float,
    submitted: bool,
    order_issues: list[dict] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> ReconciliationReport:
    """
    Compare intended against actual exposure and explain any gap.

    ``submitted`` distinguishes a dry run from a real one. A dry run
    that shows a large gap is working correctly; a submitted run that
    shows the same gap is a failure. Conflating the two would either
    cry wolf every evening or hide real breakage.
    """

    if equity <= 0:
        raise ValueError("equity must be positive")

    report = ReconciliationReport()
    report.order_issues = list(order_issues or [])

    issue_symbols = {i["symbol"] for i in report.order_issues}

    symbols = sorted(set(intended_weights) | set(actual_quantities))
    report.checked_symbols = len(symbols)

    for symbol in symbols:
        intended = float(intended_weights.get(symbol, 0.0))
        quantity = float(actual_quantities.get(symbol, 0.0))
        price = float(prices.get(symbol, 0.0))

        if price <= 0:
            report.discrepancies.append(
                WeightDiscrepancy(
                    symbol=symbol,
                    intended_weight=intended,
                    actual_weight=float("nan"),
                    drift=float("nan"),
                    severity="unknown",
                    likely_cause="no price available to value position",
                )
            )
            continue

        actual = quantity * price / equity
        drift = actual - intended

        if abs(drift) <= tolerance:
            severity, cause = "ok", "within tolerance"
        elif not submitted:
            severity, cause = "ok", (
                "dry run: no orders were submitted, so a gap is expected"
            )
        elif symbol in issue_symbols:
            severity, cause = "major", (
                "explained by an order issue above"
            )
        elif abs(drift) > 3 * tolerance:
            severity, cause = "major", (
                "large unexplained gap between intended and actual "
                "exposure"
            )
        else:
            severity, cause = "minor", (
                "small unexplained gap; check fill prices and timing"
            )

        report.discrepancies.append(
            WeightDiscrepancy(
                symbol=symbol,
                intended_weight=intended,
                actual_weight=actual,
                drift=drift,
                severity=severity,
                likely_cause=cause,
            )
        )

    return report


def summarize_reconciliation(report: ReconciliationReport) -> str:
    lines: list[str] = []

    if report.clean:
        lines.append(
            f"RECONCILED CLEAN - {report.checked_symbols} symbol(s) "
            f"match intended exposure"
        )
    else:
        lines.append("RECONCILIATION FOUND ISSUES")

    if report.order_issues:
        lines.append("")
        lines.append("ORDER ISSUES")
        lines.append("-" * 66)

        for issue in report.order_issues:
            lines.append(
                f"  [{issue['severity'].upper()}] {issue['symbol']}: "
                f"{issue['issue']}"
            )
            lines.append(f"      {issue['detail']}")

    lines.append("")
    lines.append("POSITIONS")
    lines.append("-" * 66)
    lines.append(
        f"  {'symbol':<8}{'intended':>10}{'actual':>10}"
        f"{'drift':>10}  status"
    )

    for item in report.discrepancies:
        lines.append(
            f"  {item.symbol:<8}{item.intended_weight:>9.2%}"
            f"{item.actual_weight:>10.2%}{item.drift:>+10.2%}  "
            f"{item.severity}"
        )

        if not item.within_tolerance:
            lines.append(f"      {item.likely_cause}")

    return "\n".join(lines)
