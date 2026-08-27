from __future__ import annotations


def evaluate_research_rules(
    metrics: dict,
    *,
    minimum_trades: int = 200,
    minimum_profit_factor: float = 1.25,
    maximum_drawdown_pct: float = 0.20,
) -> dict:

    checks = {
        "minimum_trades": (
            metrics["trade_count"]
            >= minimum_trades
        ),
        "minimum_profit_factor": (
            metrics["profit_factor"]
            >= minimum_profit_factor
        ),
        "maximum_drawdown": (
            metrics["max_drawdown_pct"]
            <= maximum_drawdown_pct
        ),
        "positive_expectancy": (
            metrics["expectancy"] > 0
        ),
    }

    return {
        "passed": all(checks.values()),
        "checks": checks,
    }