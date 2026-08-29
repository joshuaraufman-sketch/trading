# Decisions and Open Questions

Append-only log of what was decided, what was rejected, and why. This
file is the project's memory. Anything not written here does not exist
between sessions.

Newest entries at the top.

---

## 2026-08-28 — Selection correction run: nothing survives

**Result, development split, 100 permutations, full 36-config re-sweep
inside every permutation (3,600 backtests, 31 minutes):**

```
metric                          obs     null    pct        p    p x4
best-of-sweep strategy        0.360    0.076    94%   0.0693   0.277
best-of-sweep exposure-match  1.025    0.909    75%   0.2574   1.030
strategy @ PF winner          0.327   -0.145    93%   0.0792   0.317
exposure-match @ PF winner    1.025    0.750    86%   0.1485   0.594
```

**Nothing is significant.** Smallest p is 0.0693; Bonferroni across the
four metrics gives 0.277. The earlier uncorrected result of p = 0.0199
on strategy Sharpe degrades to 0.0792 once selection is accounted for.

**The number worth keeping — selection bias, measured:**

```
exposure-matched Sharpe, null mean
  fixed single config     0.458
  best of 36 configs      0.909
```

**Choosing the best of 36 configurations adds +0.451 Sharpe out of pure
noise.** No edge, no information, just picking the winner of a search.
That gap is the entire reason the uncorrected test read p = 0.0498: the
observed value was being raced against a null that was half a Sharpe
point too low.

**Practical discovery — the analytic bound works.** The closed-form
`expected_maximum_sharpe` predicted a best-of-36 null of 0.936; the
31-minute simulation measured 0.909. Three percent error. Screen with
the analytic version in milliseconds and spend compute only when a
result lands near the bound.

**Independent confirmation of the undocumented selection.** The script's
profit-factor winner is `{10, 10, 0.03}`, the top row of the original
sweep ranking — not the `{10, 10, 0.02}` recorded in
`frozen_candidate.yaml`. Selecting by the criterion the code actually
ranks on yields a different configuration than was frozen.

**Direction worth noting, not acting on.** All four observed values sit
at the 75th percentile or above. That is what a small real effect looks
like in an underpowered sample, and equally what noise looks like. Six
years of daily data on four correlated ETFs cannot separate them. Do not
attempt to resolve this on this split.

## 2026-08-28 — Walk-forward rebuilt as actual walk-forward

**Decision:** `scripts/run_walk_forward.py` was rewritten. Fold logic
lives in `validation/walk_forward.py`, testable without price data or a
backtest engine.

**What was wrong:** the old script applied one already-selected
parameter set to four fixed calendar windows, three of which were the
same data those parameters were fitted on, and the fourth was the
validation split. It could not detect overfitting because nothing about
it was out-of-sample. Every number it ever produced was in-sample.

**What it does now:** re-runs the entire 36-configuration sweep inside
each training window, applies the winner to the following window that
the selection never saw, and stitches those test windows into one
continuous equity curve. That curve is the only one in the project built
solely from decisions made before the data was seen. It is then put
through the same benchmark gates the performance report uses.

**Design choices worth remembering:**

- Signals are generated once over the whole split and then sliced.
  Generating inside each fold would leave the first `window` bars of
  every fold without an SMA and silently drop early signals. This does
  not leak: the SMA is strictly backward looking.
- `--select-on` is explicit and defaults to Sharpe. The original sweep
  ranked on profit factor while the candidate was chosen on net profit;
  leaving the criterion implicit is how that extra degree of freedom
  went unrecorded.
- Anchored mode grows the training window; rolling keeps it fixed.
  Anchored is the default.
- Folds with no viable configuration contribute nothing rather than a
  flat zero-return stretch, which would dampen measured volatility.
- Overlapping test windows raise rather than being silently averaged.

**Parameter stability is now reported.** Configurations that jump every
retraining window indicate an unstable effect even when the aggregate
out-of-sample curve looks acceptable. It is a gate, not a footnote.

**Caveat on sample size:** at 504 training and 126 test sessions the
development split yields roughly 7 folds of about 25 trades each. That
is thin. Treat fold-level results as directional and only the stitched
curve as measurable.

## 2026-08-28 — The signal beats random entry, and still loses to cash

**Finding, development split, 200 circular-shift permutations:**

```
                          observed   null mean   percentile   p-value
strategy Sharpe              0.360      -0.184        98.5%    0.0199
exposure-matched Sharpe      0.827       0.458        95.5%    0.0498
```

**What is real:** randomly-timed schedules with identical signal
frequency produce a *negative* mean Sharpe (-0.184). The real signal
produces +0.360. The SMA crossover carries genuine information relative
to entering at random, and the effect is large.

**What that does not mean:** the strategy still fails all four gates in
the performance report. It loses to buy-and-hold, to a static 40% index
blend, and to its own exposure-matched null. Both facts hold at once —
**the signal has information, but not enough to pay for its own trading
costs and cash drag.** Better than random; worse than doing nothing.

**Why the 0.827 is not evidence:**

1. p = 0.0498 means exactly 9 of 200 permutations beat it. Ten would
   give 0.0547 and a NO. One draw flips the verdict.
2. Two metrics were tested. Bonferroni puts exposure-matched at 0.0996;
   only the strategy-Sharpe result survives at 0.0398.
3. The test held parameters fixed at the values selected as best-of-36
   on this same data, so it does not correct for selection at all.
   Estimating with `expected_maximum_sharpe`: the single-config null has
   mean 0.458 and implied sd 0.223, giving an expected best-of-36 around
   0.936 — **above the observed 0.827**. (Upper bound; the 36 configs
   are correlated, so the true expectation is lower.)

**Exposure confound checked and dismissed:** observed exposure 40.49% vs
null 34.53%. Scaling exposure by a constant scales mean and standard
deviation identically, so Sharpe is invariant to the level. The gap
shows the null trades a different pattern, not that the result is
inflated.

**Do not use this to revive sma_crossover.** It remains rejected.

## 2026-08-28 — Selection-corrected permutation testing added

**Decision:** `run_permutation_test.py --resweep` re-runs the entire
36-configuration sweep inside every permutation and takes the best. The
grid moved to `validation/sweep_grid.py` so the sweep and the correction
cannot drift apart. Evaluation primitives moved to
`validation/permutation.py` so they are testable without importing the
data layer.

**Reasoning:** the fixed-parameter permutation test asks "given these
parameters, is the timing special?" when the parameters were themselves
chosen as the best of 36 on this data. `build_null_distribution`'s
docstring always said the evaluate callable should run the entire
selection procedure; the first version did not. That biased toward
finding significance.

**Two selection rules are reported**, because they answer different
questions:

- `best_*` — the highest value any of the 36 could reach. Strict
  data-snooping bound.
- `*_at_pf_best` — the metric of whichever configuration won on profit
  factor. Faithful to how the candidate was actually chosen, weaker as
  a correction.

**Cost:** ~36x. Roughly 35 minutes at 100 permutations, 70 at 200.

**Open:** the correction still assumes the 36-point grid was the whole
search. Every strategy variant considered and discarded by hand is an
additional untracked trial. Future work should log those too.

## 2026-08-28 — sma_crossover is DEAD. Do not resurrect it.

**Decision:** The `sma_crossover` candidate (SMA 10 / hold 10 / stop 2%
on SPY, QQQ, IWM, DIA) is rejected. `config/frozen_candidate.yaml` no
longer describes a live candidate. Any future session proposing to tune,
extend, or re-test it must read this entry first.

**Evidence — development split, 2017-01-03 to 2022-12-30, rf 3%:**

```
                       STRATEGY   BUY & HOLD   40% SPY / 60% CASH
CAGR                      5.91%       11.10%                6.28%
volatility                8.83%       19.46%                7.88%
Sharpe                     0.36         0.49                 0.42
max drawdown             14.37%       33.79%               13.68%
Calmar                     0.41         0.33                 0.46
beta to SPY               0.243
annualized alpha          0.87%  (t = 0.29, 95% CI -5.10% to +6.84%)
information ratio        -0.381
```

**Three independent reasons, any one sufficient:**

1. Alpha t-statistic 0.29. The confidence interval spans zero by a wide
   margin. There is no measurable edge.
2. It is dominated by a constant 40% index / 60% cash allocation on
   every axis: higher return, lower volatility, smaller drawdown,
   better Calmar. The signals, stops and sweep subtract value.
3. Levered 2.20x to match index volatility it returns 9.41% against the
   index's 11.10%, at a 31.67% drawdown against 33.79%. Same risk, less
   return.

All of this is *before* correcting for having swept 36 parameter
combinations and kept the winner.

**Predictions that were wrong, recorded so the reasoning improves:**

- Beta was predicted above 0.6; it came in at 0.243. The error was
  conflating "trades index ETFs" with "has index-like beta." Beta 0.243
  against average exposure 0.405 means the low beta is a low-exposure
  artifact. **Low beta is not evidence of edge; it is evidence of being
  out of the market.**
- Calendar drawdown was predicted to be materially worse than the
  trade-ordered 14.08%; it was 14.37%. With small positions and tight
  stops there is little unrealized swing to capture. The mark-to-market
  machinery remains correct and will matter for higher-exposure
  strategies, but it changed nothing here.

## 2026-08-28 — Exposure-matched nulls are now standing benchmarks

**Decision:** Every candidate must clear four gates, not one:
buy-and-hold Sharpe, static-blend Sharpe, exposure-matched Sharpe, and
alpha significance at |t| > 1.96. `run_performance_report.py` enforces
all four and reports which failed.

**Reasoning:** 100% buy-and-hold is the wrong comparator for a partially
invested strategy — it is trivially dismissed with "of course I
underperform, I am in cash half the time." Two nulls remove the excuse:

- **Static blend.** A constant weight equal to the strategy's average
  exposure, rebalanced daily, cash earning the risk-free rate. Asks
  whether the apparatus beats sitting still.
- **Exposure-matched.** Holds the index at the strategy's *own* daily
  exposure, lagged one session. Grants the entire exposure schedule for
  free and asks only whether symbol selection and entry timing added
  anything. Losing to this means they are actively destructive.

**Also added:** alpha now travels with a standard error, t-statistic and
95% interval. A bare alpha of 0.87% is the single easiest way to talk
yourself into a dead strategy.

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
