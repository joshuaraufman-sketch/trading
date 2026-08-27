from trading_lab.backtest.metrics import (
    calculate_metrics,
)
from trading_lab.backtest.runner import (
    run_long_signal_backtest,
)
from trading_lab.data.alpaca import (
    get_daily_bars,
)
from trading_lab.data.split import (
    get_development_data,
)
from trading_lab.strategies.sma_crossover import (
    SMACrossoverStrategy,
)
from trading_lab.validation.experiment_log import (
    save_experiment,
)
from trading_lab.validation.research_rules import (
    evaluate_research_rules,
)


SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]

STARTING_EQUITY = 100_000
RISK_PCT = 0.005
STOP_LOSS_PCT = 0.02
HOLDING_DAYS = 5
SLIPPAGE_BPS = 5
FEE_PER_SHARE = 0.005
SMA_WINDOW = 20


def main():

    print("Loading development data...")

    df = get_daily_bars(
        symbols=SYMBOLS,
        start="2017-01-01",
        end="2025-12-31",
    )

    df = get_development_data(df)

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
        minimum_trades=200,
        minimum_profit_factor=1.25,
        maximum_drawdown_pct=0.20,
    )

    parameters = {
        "symbols": SYMBOLS,
        "sma_window": SMA_WINDOW,
        "starting_equity": STARTING_EQUITY,
        "risk_pct": RISK_PCT,
        "stop_loss_pct": STOP_LOSS_PCT,
        "holding_days": HOLDING_DAYS,
        "slippage_bps": SLIPPAGE_BPS,
        "fee_per_share": FEE_PER_SHARE,
    }

    path = save_experiment(
        strategy_name=strategy.name,
        parameters=parameters,
        metrics=metrics,
        evaluation=evaluation,
        dataset="development",
    )

    print()
    print("STRATEGY")
    print("--------")
    print(strategy.name)

    print()
    print("METRICS")
    print("-------")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print()
    print("RESEARCH RULES")
    print("--------------")

    for key, value in evaluation["checks"].items():
        print(f"{key}: {value}")

    print()
    print(
        f"overall_passed: "
        f"{evaluation['passed']}"
    )

    print()
    print("EXPERIMENT SAVED")
    print("----------------")
    print(path)


if __name__ == "__main__":
    main()