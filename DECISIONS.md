# Decisions and Open Questions

Append-only log of what was decided, what was rejected, and why. This
file is the project's memory. Anything not written here does not exist
between sessions.

Newest entries at the top.

---

## 2026-08-28 — Project objective set: the harness is the product

**Decision:** Success is defined as systematic learning, not profit.
The deliverable of this project is a validation harness that reliably
kills bad strategies. Strategies are inputs to it, not the goal.

**Consequence:** Methodology work outranks strategy work indefinitely.
A negative result that correctly kills a candidate is a success.

## 2026-08-28 — Timeframe deferred, but parameterized

**Decision:** Horizon stays undecided. In exchange, bar size, holding
period, and the cost model must be explicit parameters everywhere —
never implicit in the code. Default to daily bars.

**Reasoning:** Horizon is an architectural constraint, not an empirical
finding. Intraday requires minute bars, spread-dominated cost models,
different infrastructure, and drags in PDT rules under $25k equity.
Deferring the decision is fine; discovering it via rewrite is not.

## 2026-08-28 — Selection bias is now measurable

**Decision:** Added `validation/significance.py`: signal permutation
nulls, permutation p-values, the Bailey/Lopez de Prado deflated Sharpe
ratio, and correlation-adjusted effective sample size.

**Reasoning:** 36 parameter combinations were swept and the best kept,
with no accounting for selection. Under plausible dispersion, the
expected best-of-36 annualized Sharpe from strategies with *no edge*
falls somewhere around 0.7 to 1.4. A long-only trend filter on index
ETFs is unlikely to clear that. The observed profit factor of 1.46 was
never compared against anything, so it carried no information.

**Note on nulls:** `circular_shift` is the default because it preserves
signal clustering while breaking price alignment. `shuffle` destroys
clustering too, which makes the null weaker and flatters the strategy.
Report which was used.

**Open:** `evaluate_research_rules()` still does not require a
permutation test or a deflated Sharpe. Wiring these into the promotion
gate is the next governance task, and it should be done before any new
strategy is written.

## 2026-08-28 — Research/execution sequencing

**Decision:** The execution layer is frozen. No new work on order
routing, reconciliation, lifecycle, or promotion gates until a strategy
candidate survives honest validation.

**Reasoning:** Execution is currently ahead of research. We have a
promotion pipeline for a strategy whose own walk-forward shows a loss in
2021-2022. Further execution work only makes it easier to deploy
something that does not work.

## 2026-08-28 — Calendar equity curves replace trade-ordered curves

**Decision:** `backtest/equity.py` builds a session-indexed,
mark-to-market equity curve. `backtest/performance.py` computes
time-based metrics from it. `metrics.py` is retained for trade-level
statistics only (expectancy, profit factor, win rate, average R).

**Reasoning:** `build_equity_curve` in `metrics.py` steps once per
closed trade. It cannot see unrealized drawdown, cannot share a time
axis with a benchmark, and understates risk when correlated positions
are open simultaneously. Reported drawdown was therefore not the
quantity the promotion gate thought it was gating on.

**Assumption to revisit:** `Trade.fees` stores a round-trip total, so
the curve splits it 50/50 across entry and exit
(`entry_fee_fraction=0.5`). Correct for the current runner, which
charges symmetric per-share fees. If the fee model changes, change this.

## 2026-08-28 — Benchmark comparison is now enforceable

**Decision:** `benchmark.py` gains `build_benchmark_curve` and
`compare_to_benchmark`. `calculate_buy_and_hold_return` is retained but
deprecated in favour of them.

**Reasoning:** `require_benchmark_comparison: true` has been in
`research_rules.yaml` since the start and was enforced nowhere. The old
single-number total return could not be risk adjusted and hid the fact
that the strategy is only exposed part of the time.

**Open:** `evaluate_research_rules()` still does not check beta, alpha,
or Sharpe-vs-benchmark. Adding those gates is the next governance task.

## 2026-08-28 — Experiment log collisions fixed

**Decision:** Experiment filenames now include a 10-character digest of
strategy, dataset, and parameters.

**Reasoning:** Second-resolution timestamps collided. The 36-combination
sweep produced 9 files; roughly 27 records were silently overwritten,
and `parameter_sweep_summary.csv` still pointed multiple rows at the
same file. The audit trail was actively wrong.

**Action required:** Existing `experiments/*.json` and the two summary
CSVs are unreliable. Re-run the sweep before treating any of it as
evidence.

## 2026-08-28 — Timezone handling for session dates

**Decision:** `normalize_session_dates` converts tz-aware timestamps to
America/New_York before taking the date, and normalizes tz-naive input
without conversion.

**Reasoning:** Alpaca stamps daily bars at the UTC equivalent of the
open, which moves with DST. Naive midnight timestamps treated as UTC map
back to 19:00 the previous day, shifting every trade one session early.
Caught by `test_open_position_is_marked_to_market`.

---

# Open questions, ranked

1. **Does sma_crossover have any edge over buy-and-hold?** Run
   `scripts/run_performance_report.py --split development`. Expect high
   beta and near-zero alpha. If so, the candidate is dead and no amount
   of execution polish saves it.
2. **Walk-forward is not walk-forward.** `run_walk_forward.py` applies
   already-selected parameters to fixed windows, three of which are
   in-sample. Needs rebuilding as rolling re-optimization.
3. **Validation is spent.** 2023-2024 was consumed by the walk-forward
   run. Only the 2025 holdout is clean, and at ~30 correlated trades it
   is too small to clear the project's own 200-trade bar.
4. **Config is decorative.** Only `forward_test_status.py` and the new
   `run_performance_report.py` read YAML. Everything else hardcodes.
5. **No aggregate risk limit.** Per-trade risk is capped at 0.5%;
   nothing caps total open risk across four ~0.9-correlated ETFs.
6. **Backtest and live paths differ.** Live enforces a 1-order/day cap,
   a 25% position cap, session checks, and uses the IEX feed against
   unadjusted prices; the backtest does none of that and uses adjusted
   SIP. The forward test cannot currently validate the backtest.
7. **Paper account is contaminated** with manual BUD/QQQ/QQQM positions.
   QQQ signals are permanently blocked by `check_existing_exposure`.
   Reset before the forward-test clock counts.
