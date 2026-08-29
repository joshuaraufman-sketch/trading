"""
Volatility targeting, run as the project's positive control.

The harness has only ever been pointed at strategies that fail. It has
never confirmed a true effect end to end, which makes every negative
result it produces ambiguous: dead strategy, or pipeline eating signal?

This runs a documented, mechanism-driven effect through the full
pipeline in three layers, weakest claim last:

  LAYER 1  Mechanism. Does trailing volatility predict forward
           volatility? High statistical power, should be clearly true.
           THIS is the positive control. If it fails, the pipeline is
           broken and prior negative results are suspect.

  LAYER 2  Implementation. Did targeting actually stabilise realized
           risk? Nearly mechanical when layer 1 holds. Failure here
           points at a coding fault, not a market fact.

  LAYER 3  Economics. Does it improve Sharpe net of turnover costs?
           Weak, path dependent, sample dependent.

Prediction recorded before running (see DECISIONS.md): layers 1 and 2
pass clearly; layer 3 may FAIL on 2017-2022 because March 2020 is the
worst case for volatility targeting -- volatility explodes, the model
de-risks, and the V-recovery arrives while exposure is still reduced.
Layers 1-2 passing with layer 3 failing is a correct result, not a
malfunction.

Usage:
    python scripts/run_volatility_target.py --split development \\
        --target-vol 0.10 --lookback 20 --risk-free-rate 0.03
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from trading_lab.backtest.benchmark import (
    build_benchmark_curve,
    build_static_blend_curve,
    compare_to_benchmark,
)
from trading_lab.backtest.equity import normalize_session_dates
from trading_lab.backtest.performance import calculate_performance
from trading_lab.backtest.weights import (
    run_weight_backtest,
    turnover_summary,
)
from trading_lab.data.alpaca import get_daily_bars
from trading_lab.data.split import (
    get_development_data,
    get_holdout_data,
    get_validation_data,
)
from trading_lab.validation.significance import (
    permute_weight_schedule,
    schedule_turnover,
    summarize_against_null,
)
from trading_lab.validation.walk_forward import (
    parameter_stability,
    run_walk_forward,
    stitch_out_of_sample,
)
from trading_lab.strategies.volatility_target import (
    volatility_capture,
    volatility_forecast_skill,
    volatility_target_weights,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "volatility_target"

SYMBOL = "SPY"


def select_split(df, split: str):
    if split == "development":
        return get_development_data(df)
    if split == "validation":
        return get_validation_data(df)
    if split == "holdout":
        # Sealed unless parameters are frozen; this is a control run and
        # has no business touching it.
        return get_holdout_data(df, parameters_frozen=False)
    raise ValueError(f"Unknown split: {split!r}")


def _pct(value: float) -> str:
    return f"{value:>9.2%}"


def _num(value: float) -> str:
    return f"{value:>9.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", default="development",
        choices=["development", "validation", "holdout"],
    )
    parser.add_argument("--target-vol", type=float, default=0.10)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument(
        "--max-weight", type=float, default=1.0,
        help="1.0 means de-risk only, never lever up. The honest "
             "default for an account without margin.",
    )
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument(
        "--rebalance-band", type=float, default=0.05,
        help="Suppress weight changes smaller than this. Zero "
             "reproduces full daily rebalancing.",
    )
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument(
        "--permutations", type=int, default=0,
        help=(
            "Layer 4: circular-shift the weight schedule to test whether "
            "its TIMING matters. Low statistical power by construction -- "
            "see tests/test_weight_permutation.py. A non-significant "
            "result here does not indicate absence of effect."
        ),
    )
    parser.add_argument(
        "--walk-forward", action="store_true",
        help=(
            "Layer 5: re-select the lookback in each training window and "
            "apply it to the next. The better-powered stability test."
        ),
    )
    parser.add_argument("--train-sessions", type=int, default=504)
    parser.add_argument("--test-sessions", type=int, default=126)
    args = parser.parse_args()

    print(f"Loading {SYMBOL} bars...")

    df = get_daily_bars(
        symbols=[SYMBOL], start="2017-01-01", end="2025-12-31"
    )
    split_df = select_split(df, args.split)

    frame = split_df[split_df["symbol"] == SYMBOL].copy()
    frame["session"] = normalize_session_dates(frame["timestamp"])

    closes = (
        frame.groupby("session")["close"].last().sort_index().astype(float)
    )
    returns = closes.pct_change().fillna(0.0)

    starting_equity = 100_000.0

    print()
    print(f"VOLATILITY TARGETING - {args.split.upper()}")
    print("=" * 72)
    print(f"symbol           {SYMBOL}")
    print(f"sessions         {len(returns)} "
          f"({closes.index[0].date()} to {closes.index[-1].date()})")
    print(f"target vol       {args.target_vol:.1%}")
    print(f"lookback         {args.lookback} sessions")
    print(f"max weight       {args.max_weight:.2f}")
    print(f"cost             {args.cost_bps:.1f} bps on turnover")
    print(f"rebalance band   {args.rebalance_band:.2f}")

    # ---------------- LAYER 1: mechanism ----------------
    skill = volatility_forecast_skill(
        returns, lookback=args.lookback, horizon=args.horizon,
    )

    print()
    print("LAYER 1 - MECHANISM (the positive control)")
    print("=" * 72)
    print("Does trailing volatility predict forward volatility?")
    print()
    print(f"{'correlation':<38}{skill['correlation']:>10.3f}")
    print(f"{'volatility autocorrelation':<38}"
          f"{skill['volatility_autocorrelation']:>10.3f}")
    print(f"{'raw R-squared (uncalibrated)':<38}"
          f"{skill['raw_r_squared']:>10.3f}")
    print(f"{'shrunk R-squared (the verdict)':<38}"
          f"{skill['shrunk_r_squared']:>10.3f}")
    print(f"{'mean shrinkage slope':<38}"
          f"{skill['mean_shrinkage_slope']:>10.3f}")
    print(f"{'out-of-sample points evaluated':<38}"
          f"{skill['evaluated_out_of_sample']:>10}")
    print()
    print("raw is full-sample and uncalibrated; shrunk is out-of-sample "
          "after burn-in.")
    print("They are different estimands -- shrunk is the honest one.")

    if skill["raw_r_squared"] < 0:
        print()
        print("The negative raw R-squared is expected, not a problem. "
              "Trailing volatility")
        print("is unbiased but noisy, so uncalibrated it loses to the "
              "unconditional mean")
        print("even while the correlation is clearly positive.")

    mechanism_ok = skill["beats_unconditional_mean"]
    print()
    print(f"MECHANISM: {'PASS' if mechanism_ok else 'FAIL'}")

    if not mechanism_ok:
        print()
        print("!! The positive control failed. Volatility clustering is "
              "one of the most")
        print("!! robust facts in finance. Suspect the pipeline, not the "
              "market, and")
        print("!! treat every prior negative result as unverified.")

    # ---------------- LAYER 2: implementation ----------------
    weights = volatility_target_weights(
        returns,
        target_volatility=args.target_vol,
        lookback=args.lookback,
        max_weight=args.max_weight,
    )

    capture = volatility_capture(
        returns, weights, target_volatility=args.target_vol, window=60,
    )

    print()
    print("LAYER 2 - IMPLEMENTATION")
    print("=" * 72)
    print("Did targeting actually stabilise realized risk?")
    print()
    print(f"{'':<34}{'UNTARGETED':>12}{'TARGETED':>12}")
    print("-" * 72)
    print(f"{'mean realized volatility':<34}"
          f"{capture['raw_mean_volatility']:>11.2%}"
          f"{capture['targeted_mean_volatility']:>12.2%}")
    print(f"{'dispersion of realized vol':<34}"
          f"{capture['raw_volatility_dispersion']:>11.2%}"
          f"{capture['targeted_volatility_dispersion']:>12.2%}")
    print(f"{'mean absolute miss vs target':<34}"
          f"{capture['raw_mean_absolute_miss']:>11.2%}"
          f"{capture['targeted_mean_absolute_miss']:>12.2%}")
    print(f"{'mean weight held':<34}{'':<12}{weights.mean():>11.2%}")

    implementation_ok = capture["dispersion_reduced"]
    print()
    print(f"IMPLEMENTATION: {'PASS' if implementation_ok else 'FAIL'}")

    # ---------------- LAYER 3: economics ----------------
    curve = run_weight_backtest(
        returns, weights,
        starting_equity=starting_equity,
        cost_bps=args.cost_bps,
        rebalance_band=args.rebalance_band,
        risk_free_rate=args.risk_free_rate,
    )

    benchmark_curve = build_benchmark_curve(
        split_df, SYMBOL,
        starting_equity=starting_equity, sessions=curve.index,
    )
    comparison = compare_to_benchmark(
        curve, benchmark_curve, risk_free_rate=args.risk_free_rate,
    )

    strat = comparison["strategy"]
    bench = comparison["benchmark"]

    blend = calculate_performance(
        build_static_blend_curve(
            split_df, SYMBOL,
            starting_equity=starting_equity,
            weight=min(max(float(strat["average_exposure"]), 0.0), 1.0),
            sessions=curve.index,
            risk_free_rate=args.risk_free_rate,
        ),
        risk_free_rate=args.risk_free_rate,
    )

    turnover = turnover_summary(curve)

    print()
    print("LAYER 3 - ECONOMICS")
    print("=" * 72)
    print(
        f"{'':<28}{'VOL TARGET':>11}{'BUY & HOLD':>12}"
        f"{'STATIC BLEND':>14}"
    )
    print("-" * 72)

    for label, key, fmt in [
        ("CAGR", "cagr", _pct),
        ("annual volatility", "annual_volatility", _pct),
        ("Sharpe", "sharpe", _num),
        ("Sortino", "sortino", _num),
        ("max drawdown", "max_drawdown_pct", _pct),
        ("Calmar", "calmar", _num),
        ("average exposure", "average_exposure", _pct),
    ]:
        print(f"{label:<28}{fmt(strat[key]):>11}{fmt(bench[key]):>12}"
              f"{fmt(blend[key]):>14}")

    print()
    print(f"{'beta':<28}{comparison['beta']:>11.3f}")
    print(f"{'annualized alpha':<28}"
          f"{comparison['annual_alpha']:>10.2%}")
    print(f"{'  t-statistic':<28}{comparison['alpha_t_stat']:>11.2f}")
    print()
    print(f"{'annualized turnover':<28}"
          f"{turnover['annualized_turnover']:>11.2f}x")
    print(f"{'annualized cost drag':<28}"
          f"{turnover['annualized_cost_drag']:>10.2%}")
    print(f"{'sessions traded':<28}"
          f"{turnover['sessions_traded']:>7}/{turnover['sessions']}")

    economics_ok = strat["sharpe"] > bench["sharpe"]

    print()
    print(f"ECONOMICS: {'PASS' if economics_ok else 'FAIL'}")

    print()
    print("VERDICT")
    print("=" * 72)

    if mechanism_ok and implementation_ok and economics_ok:
        print("All three layers pass. The harness confirms a known "
              "effect end to end.")
        print("It can now be trusted to reject as well as accept.")
    elif mechanism_ok and implementation_ok:
        print("Mechanism and implementation pass; economics fail.")
        print()
        print("This is the predicted outcome, and it is a CORRECT "
              "result, not a fault.")
        print("Volatility forecasts, targeting stabilises risk, and the "
              "Sharpe gain")
        print("does not survive this particular sample. Check whether "
              "the shortfall")
        print("is concentrated in the 2020 de-risk-then-miss-the-"
              "recovery episode.")
        print()
        print("The harness is validated: it confirmed a real mechanism "
              "rather than")
        print("rejecting everything put in front of it.")
    elif not mechanism_ok:
        print("MECHANISM FAILED. This is the outcome that indicates a "
              "broken pipeline.")
        print("Do not proceed to new strategies. Every negative result "
              "produced so")
        print("far should be treated as unverified until this is "
              "understood.")
    else:
        print("Mechanism passes but implementation fails. That points "
              "at a coding")
        print("fault in the weighting or backtest path, not a market "
              "fact.")


    # ---------------- LAYER 4: does the timing matter? ----------------
    if args.permutations > 0:
        valid = weights.iloc[args.lookback:]
        aligned = returns.loc[valid.index]

        def schedule_sharpe(schedule):
            c = run_weight_backtest(
                aligned, schedule, starting_equity=starting_equity,
                cost_bps=args.cost_bps,
                rebalance_band=args.rebalance_band,
                risk_free_rate=args.risk_free_rate,
            )
            return calculate_performance(
                c, risk_free_rate=args.risk_free_rate
            )["sharpe"]

        observed_sharpe = schedule_sharpe(valid)
        rng = np.random.default_rng(0)

        null, null_turnover = [], []

        for _ in range(args.permutations):
            shifted = permute_weight_schedule(
                valid, rng=rng, method="circular_shift"
            )
            null.append(schedule_sharpe(shifted))
            null_turnover.append(schedule_turnover(shifted))

        summary = summarize_against_null(observed_sharpe, null)

        print()
        print("LAYER 4 - DOES THE TIMING MATTER?")
        print("=" * 72)
        print("Circular-shifted schedules keep the identical path shape,")
        print("and therefore identical turnover. Only their placement in")
        print("time changes.")
        print()
        print(f"{'observed Sharpe':<38}{summary['observed']:>10.3f}")
        print(f"{'null mean':<38}{summary['null_mean']:>10.3f}")
        print(f"{'null 95th percentile':<38}{summary['null_p95']:>10.3f}")
        print(f"{'observed percentile':<38}"
              f"{summary['percentile_of_null']:>9.1f}%")
        print(f"{'p-value':<38}{summary['p_value']:>10.4f}")
        print()
        print(f"{'observed turnover':<38}"
              f"{schedule_turnover(valid):>10.1f}")
        print(f"{'null mean turnover':<38}"
              f"{float(np.mean(null_turnover)):>10.1f}")
        print("Turnover must match, or this measures cost drag, not timing.")
        print()
        print("LOW POWER BY CONSTRUCTION. On simulated data containing a")
        print("known effect this test reaches only the 60th-77th")
        print("percentile. A non-significant result here is uninformative;")
        print("treat it as directional and rely on layer 5.")

    # ---------------- LAYER 5: walk-forward the lookback ----------------
    if args.walk_forward:
        lookbacks = (10, 20, 40, 60)

        def select(fold):
            best, chosen = float("-inf"), None

            for lookback in lookbacks:
                w = volatility_target_weights(
                    returns, target_volatility=args.target_vol,
                    lookback=lookback, max_weight=args.max_weight,
                )
                window = (returns.index >= fold.train_start) & (
                    returns.index <= fold.train_end
                )
                c = run_weight_backtest(
                    returns.loc[window], w.loc[window],
                    starting_equity=starting_equity,
                    cost_bps=args.cost_bps,
                    rebalance_band=args.rebalance_band,
                    risk_free_rate=args.risk_free_rate,
                )
                score = calculate_performance(
                    c, risk_free_rate=args.risk_free_rate
                )["sharpe"]

                if score > best:
                    best, chosen = score, {"lookback": lookback}

            return (chosen, best, {}) if chosen else (
                None, float("nan"), {}
            )

        def evaluate(fold, config):
            w = volatility_target_weights(
                returns, target_volatility=args.target_vol,
                lookback=config["lookback"], max_weight=args.max_weight,
            )
            window = (returns.index >= fold.test_start) & (
                returns.index <= fold.test_end
            )
            c = run_weight_backtest(
                returns.loc[window], w.loc[window],
                starting_equity=starting_equity,
                cost_bps=args.cost_bps,
                rebalance_band=args.rebalance_band,
                risk_free_rate=args.risk_free_rate,
            )
            return c["equity"].pct_change().fillna(0.0), 0, {}

        results = run_walk_forward(
            returns.index, select=select, evaluate=evaluate,
            train_sessions=args.train_sessions,
            test_sessions=args.test_sessions,
        )

        oos = stitch_out_of_sample(results, starting_equity=starting_equity)
        oos_curve = oos.copy()
        oos_curve["open_positions"] = 1
        oos_curve["exposure_pct"] = float("nan")

        oos_performance = calculate_performance(
            oos_curve, risk_free_rate=args.risk_free_rate
        )
        oos_benchmark = calculate_performance(
            build_benchmark_curve(
                split_df, SYMBOL, starting_equity=starting_equity,
                sessions=oos.index,
            ),
            risk_free_rate=args.risk_free_rate,
        )
        stability = parameter_stability(results)

        print()
        print("LAYER 5 - WALK-FORWARD (lookback re-selected per fold)")
        print("=" * 72)
        print(f"{'folds':<30}{len(results):>10}")
        print(f"{'lookbacks chosen':<30}"
              f"{str(stability['lookback']['values']):>40}")
        print(f"{'changed':<30}"
              f"{stability['lookback']['changed_fraction']:>9.0%}")
        print()
        print(f"{'':<30}{'WALK-FWD':>11}{'BUY & HOLD':>13}")
        print("-" * 72)
        for label, key, fmt in [
            ("CAGR", "cagr", _pct), ("Sharpe", "sharpe", _num),
            ("max drawdown", "max_drawdown_pct", _pct),
            ("Calmar", "calmar", _num),
        ]:
            print(f"{label:<30}{fmt(oos_performance[key]):>11}"
                  f"{fmt(oos_benchmark[key]):>13}")
        print()
        print("This is out-of-sample: the lookback used in each window was")
        print("chosen before that window was seen.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = REPORT_DIR / f"{stamp}_{args.split}_volatility_target.json"

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "split": args.split,
                "generated_utc": stamp,
                "parameters": vars(args),
                "layer_1_mechanism": skill,
                "layer_2_implementation": capture,
                "layer_3_comparison": comparison,
                "static_blend": blend,
                "turnover": turnover,
                "gates": {
                    "mechanism": mechanism_ok,
                    "implementation": implementation_ok,
                    "economics": economics_ok,
                },
            },
            file, indent=2, default=str,
        )

    curve_path = REPORT_DIR / f"{stamp}_{args.split}_vol_target_curve.csv"
    curve.join(
        benchmark_curve["equity"].rename("benchmark_equity")
    ).join(weights.rename("target_weight")).to_csv(curve_path)

    print()
    print(f"report saved: {path.relative_to(PROJECT_ROOT)}")
    print(f"curve saved:  {curve_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
