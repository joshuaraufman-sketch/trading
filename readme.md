# AI Trading Lab

A research-first trading system for developing, validating, and paper-trading systematic strategies before any live deployment.

## Stage 1 goals

1. Define a reproducible project structure.
2. Standardize strategy definitions.
3. Establish market-data access.
4. Build a trustworthy backtesting core.
5. Record every experiment and its assumptions.
6. Keep paper/live trading credentials out of the repository.

## Planned architecture

```text
trading/
├── config/
├── data/
├── experiments/
├── notebooks/
├── reports/
├── src/
│   └── trading_lab/
│       ├── data/
│       ├── strategies/
│       ├── backtest/
│       ├── validation/
│       ├── execution/
│       └── risk/
└── tests/
## Safety rules

- Research and paper trading first.
- No live brokerage credentials in GitHub.
- No strategy advances because of one attractive backtest.
- Fees and slippage must be included.
- Holdout data remains untouched until strategy parameters are frozen.
- Risk controls are deterministic; AI may analyze but should not bypass them.
