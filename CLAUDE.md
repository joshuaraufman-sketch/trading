# AI Trading Lab — working agreement

Read this before doing anything. Then read `DECISIONS.md` for what has
already been settled and what is still open.

## What this project is

A research-first systematic trading lab. Two phases, sequenced, not
parallel:

1. **Research.** Find whether any strategy here has edge over
   buy-and-hold on a risk-adjusted basis.
2. **Execution.** Only after phase one produces a candidate that
   survives honest out-of-sample validation.

Paper trading only. No live capital, on any timeline currently planned.

## Current honest state

- One candidate exists: `sma_crossover` on SPY/QQQ/IWM/DIA, daily bars.
- Its own walk-forward loses money in 2021-2022 (PF 0.75) and makes
  money in three bull windows. That is the signature of long equity beta,
  not edge.
- No benchmark comparison had ever been run until 2026-08-28.
- The execution layer (orders, reconciliation, lifecycle, promotion
  gates) is more built out than the research that would justify it.

## Hard rules

1. **The 2025 holdout is sealed.** Do not read it, sample it, plot it,
   or "just check" it. It is the last clean data in the project and it
   is already too small. `get_holdout_data` enforces this — do not work
   around the `PermissionError`.
2. **Execution work is frozen.** No changes to `execution/` or `risk/`
   until a candidate passes validation. If a task seems to require it,
   stop and ask.
3. **No live trading credentials, ever.** `ALPACA_PAPER_TRADE` stays
   true. Never commit `.env`.
4. **Never ask the user to paste API keys into a conversation.**
5. **Deterministic risk controls are not negotiable.** Analyze them,
   propose changes to them, never bypass them.

## Working agreements

- **Append to `DECISIONS.md` whenever something is decided or rejected,
  including the reasoning.** Undocumented decisions do not survive to the
  next session. This is not optional bookkeeping; it is the memory.
- **Config over constants.** New scripts read `config/*.yaml`. Do not
  hardcode parameters that already live in YAML. Several existing
  scripts violate this and should be migrated when touched.
- **Tests pass before anything is called done.** `python -m pytest -q`.
- **A backtest result is a hypothesis, not a finding.** Report it with
  its sample size and its out-of-sample status attached.
- **Say when something doesn't work.** A negative result that kills a
  candidate is more valuable here than a positive one that survives on
  a technicality. Do not soften it.

## Do not trust these artifacts

- `experiments/*.json` — written before the filename-collision fix;
  roughly 27 of 36 sweep records were silently overwritten.
- `experiments/parameter_sweep_summary.csv` — rows point at experiment
  files that contain different parameters.
- `experiments/walk_forward_summary.csv` — produced by a script that is
  not actually walk-forward (fixed parameters, three in-sample windows).

Re-run these before citing them as evidence.

## Metrics: which module to use

- `backtest/metrics.py` — trade-level only. Expectancy, profit factor,
  win rate, average R. Its `build_equity_curve` is trade-ordered and
  **understates drawdown**; do not use it for risk or benchmarking.
- `backtest/equity.py` — calendar, mark-to-market equity curve. Use this
  for anything involving drawdown, exposure, or time.
- `backtest/performance.py` — CAGR, Sharpe, Sortino, Calmar, drawdown
  duration. Always pass a real `risk_free_rate`; zero inflates Sharpe
  across the 2022-2024 rate cycle.
- `backtest/benchmark.py` — `build_benchmark_curve` +
  `compare_to_benchmark`. Always pass `sessions=strategy_curve.index`
  so both curves cover identical days.

## Commands

```bash
python -m pytest -q                                  # full suite
python scripts/run_performance_report.py --split development --risk-free-rate 0.03
python scripts/run_parameter_sweep.py                # re-run post-fix
python scripts/forward_test_status.py                # paper progress
```

## Starting a session

1. Read `DECISIONS.md`, newest entries first.
2. Check the ranked open questions at the bottom of it.
3. Confirm the task at hand isn't blocked by a frozen area or a sealed
   dataset.
4. Do the work. Append what was decided.
