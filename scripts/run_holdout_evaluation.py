from __future__ import annotations

from trading_lab.backtest.metrics import calculate_metrics
from trading_lab.backtest.runner import run_long_signal_backtest
from trading_lab.data.alpaca import get_daily_bars
from trading_lab.data.split import get_holdout_data
from trading_lab.strategies.sma_crossover import SMACrossoverStrategy
from trading_lab.validation.experiment_log import save_experiment
from trading_lab.validation.research_rules import evaluate_research_rules


SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]

STARTING_EQUITY = 100_000
RISK_PCT = 0.005
SMA_WINDOW = 10
HOLDING_DAYS = 10
STOP_LOSS_PCT = 0.02
SLIPPAGE_BPS = 5
FEE_PER_SHARE = 0.005


def main():
    print("Loading sealed holdout data...")

    df = get_daily_bars(
        symbols=SYMBOLS,
        start="2017-01-01",
        end="2025-12-31",
    )

    df = get_holdout_data(
        df,
        parameters_frozen=True,
    )

    strategy = SMACrossoverStrategy(
        window=SMA_WINDOW,
    )

    signal_df = strategy.generate_signals(df)

    trades = run_long_signal_backtest(
        signal_df,
        starting_equity=STARTING_EQUITY,
        risk_pct=RISK_PCT,
        stop_loss_pct=STOP_LOSS_PCT,
        holding_days=HOLDING_DAYS,
        slippage_bps=SLIPPAGE_BPS,
        fee_per_share=FEE_PER_SHARE,
    )

    metrics = calculate_metrics(
        trades,
        starting_equity=STARTING_EQUITY,
    )

    evaluation = evaluate_research_rules(
        metrics,
        minimum_trades=25,
        minimum_profit_factor=1.10,
        maximum_drawdown_pct=0.20,
    )

    parameters = {
        "symbols": SYMBOLS,
        "sma_window": SMA_WINDOW,
        "holding_days": HOLDING_DAYS,
        "stop_loss_pct": STOP_LOSS_PCT,
        "starting_equity": STARTING_EQUITY,
        "risk_pct": RISK_PCT,
        "slippage_bps": SLIPPAGE_BPS,
        "fee_per_share": FEE_PER_SHARE,
        "parameters_frozen_before_holdout": True,
        "holdout_access": "one_time_final_evaluation",
    }

    path = save_experiment(
        strategy_name=strategy.name,
        parameters=parameters,
        metrics=metrics,
        evaluation=evaluation,
        dataset="holdout_2025",
    )

    print()
    print("FROZEN PARAMETERS")
    print("-----------------")

    for key, value in parameters.items():
        print(f"{key}: {value}")

    print()
    print("HOLDOUT METRICS")
    print("---------------")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print()
    print("HOLDOUT GATE")
    print("------------")

    for key, value in evaluation["checks"].items():
        print(f"{key}: {value}")

    print()
    print(f"overall_passed: {evaluation['passed']}")

    print()
    print("HOLDOUT EXPERIMENT SAVED")
    print("------------------------")
    print(path)


if __name__ == "__main__":
    main()