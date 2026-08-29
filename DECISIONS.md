# Decisions and Open Questions

Append-only log of what was decided, what was rejected, and why. This
file is the project's memory. Anything not written here does not exist
between sessions.

Newest entries at the top.

---

## 2026-08-28 — Paper account replaced; identity now pinned

**What happened:** Alpaca offers no paper-account reset, so a new paper
account was created and credentials swapped in `.env`. This was the
right call — it gives a clean audit boundary rather than a murky one.

**What nothing noticed.** No code recorded which account a run touched.
`AccountState` captured equity, cash and positions but no identity, so
the swap was invisible to the codebase. Two consequences:

1. `count_submitted_orders_today` globs every `*_forward_run.json`, so
   order counts now span two accounts.
2. The four forward runs tracked in git (2026-08-27) belong to the
   RETIRED account, with different starting equity and unrelated manual
   positions. The forward-test clock must not count them.

**Action required, not yet done:** move
`reports/forward_test/*.json` from before the swap into an archive
directory, or the promotion gate will count a dead account's history.

**Fix shipped:** `AccountState` now records `account_id`,
`account_number`, `status`, `pattern_day_trader` and `trading_blocked`.
`config/account.yaml` pins the expected account id;
`execution/preflight.py` refuses to proceed on a mismatch. While the id
is blank, identity is a warning rather than a block, and
`scripts/preflight_check.py` prints the observed id to paste in.

**The check that would have caught the original contamination:** the
retired account held manual BUD, QQQ and QQQM positions for weeks. QQQ
was in the strategy universe, so `check_existing_exposure` silently
blocked one of four symbols, and forward P&L mixed strategy trades with
unrelated holdings. Preflight now fails on any position outside the
declared universe. There is a test using those exact symbols.

**Note on safety already present:** `_get_trading_client` hardcodes
`paper=True`, so live credentials in `.env` would fail rather than trade
real money. Keep it that way. Going live must be a deliberate code
change, never an environment variable.

**PDT is a warning, not a block.** Under $25k a margin account is capped
at three day trades per five business days. A daily-rebalancing strategy
can trip it unnoticed. Surfaced, not enforced, since the IRA path may
make it irrelevant.

## 2026-08-28 — Goal clarified, and the execution layer was blocking it

**Goal on record:** semi-passive income. A system that runs during
market hours without CONSTANT monitoring — evening checks and failure
alerts are expected and fine. Path C chosen: ship the volatility-target
baseline, research edge behind it.

**Uncomfortable arithmetic that must not be forgotten.** Volatility
targeting made LESS money than buy-and-hold out of sample: 9.77% CAGR
against 14.55%, with a third of the drawdown. De-risk-only targeting
cannot outperform on absolute return by construction. The product is
most of the return with a third of the pain, plus the behavioural
benefit that an 11% drawdown does not make people sell at the bottom
while a 34% one does. **It is not alpha and must never be described as
such.**

**Tax interaction, unresolved.** 2.69x annual turnover realizes
short-term gains taxed as ordinary income. That drag is plausibly the
same magnitude as the entire (statistically unestablished) advantage.
Both taxable and IRA accounts are available. The IRA removes the drag
entirely but prohibits margin. Add after-tax comparison to the
performance report before deciding.

**Leverage: not yet, and the reason is specific.** The out-of-sample
Sharpe gap was 0.047 against a standard error near 0.59. If the true
edge is zero, leverage amplifies identical returns and identical risk
while paying margin interest — it costs money precisely when you are
wrong. Revisit only after (1) a quarter of unattended paper running
clean, (2) live unlevered through a real drawdown, (3) the advantage
holding on data that is not 2017-2022. Start at 1.2-1.3x if ever, not
1.95x. Vol targeting's self-de-levering is a genuine argument in its
favour; it is not sufficient.

## 2026-08-28 — Backtest/live parity fixed architecturally

**What was found:** the live risk layer structurally PROHIBITED the
strategy that passed validation. `check_order_plan` requires
`side == "buy"`, requires a positive `stop_price`, and caps positions at
25% of equity. Volatility targeting averages 72% exposure and reduces
itself by SELLING. Not one of its orders could have been placed. The
risk layer was built for discrete stop-managed trades; a continuous
weight is a different shape entirely.

**Rejected fix:** make the backtest imitate the live constraints. Two
descriptions of one system drift the moment either changes — exactly how
the sweep grid and `research_rules.yaml` drifted before.

**Adopted fix:** one function, `execution/rebalance.py`
`compute_rebalance_orders`, answering "given target weights, current
positions, prices and equity, what orders?" The backtest calls it every
session via `run_policy_backtest`; the live runner will call it once a
day. **Parity is a property of the architecture, not an assertion.**

`config/execution_policy.yaml` is the single source of truth for
constraints, and a test asserts its keys match the dataclass exactly —
guarding against the `research_rules.yaml` failure where a config file's
keys were read by nothing.

**Design choices worth remembering:**

- Gross exposure is enforced by SCALING the book, not truncating.
  Truncation would silently change which instrument dominates.
- Skipped orders record a reason. An unexplained gap between intended
  and achieved exposure is how live diverges from backtest unnoticed.
- When more orders qualify than the session cap allows, the largest by
  notional are kept. Dictionary-order truncation would make live results
  depend on iteration order and be unreproducible in backtest.
- A symbol dropped from the target set is SOLD, not forgotten.
- `max_gross_exposure: 1.00` is where leverage would be switched on. It
  is one line, deliberately, so enabling it is an explicit recorded act.

**Use `run_policy_backtest` for anything meant to predict live
behaviour.** `run_weight_backtest` remains valid for research questions
where execution friction is not the subject, but its numbers are an
upper bound.

## 2026-08-28 — Volatility targeting survives walk-forward. Baseline established.

**Layers 4 and 5, development split:**

```
LAYER 4  timing permutation   88th percentile, p = 0.1244
                              turnover 21.3 observed vs 21.8 null (matched)
LAYER 5  walk-forward         7 folds, lookback [20,20,10,10,10,20,20], 33% churn

                        walk-fwd   buy & hold      gap
Sharpe                     0.646        0.599    0.047
CAGR                       9.77%       14.55%   -4.78%
max drawdown              11.33%       33.79%  -22.46%
Calmar                     0.862        0.431    0.431
```

**The comparison that settles the harness question:**

```
                    in-sample   out-of-sample   retained
sma_crossover           0.360           0.112        31%
volatility target       0.651           0.646        99%
```

`sma_crossover` lost 69% of its Sharpe out of sample -- fitted noise.
Volatility targeting lost 1%. Two strategies, one pipeline, opposite
verdicts, both correct. The harness is validated in both directions.

**What is NOT established.** The out-of-sample Sharpe gap of 0.047
against a standard error near 0.59 is nothing. It also shrank to a fifth
of the in-sample gap (0.164), because 2019-2022 was a better window for
simply holding than 2017-2022 was. Do not claim a Sharpe edge.

**What IS durable.** A third of the drawdown and double the Calmar.
That follows from the validated layer 2 -- if the volatility forecast
works, drawdown reduction is nearly mechanical -- rather than from the
return path cooperating. Most of the return with a third of the pain,
and no demonstrable Sharpe edge. Not alpha. Calling it alpha would be
the first dishonest thing in this project.

**Layer 4 in context.** 88th percentile exceeds the 60th-77th that
simulated data containing a KNOWN effect reached. Consistent with a real
effect, cannot establish one. The power limit was measured before the
run, which is the only reason the number is interpretable.

**Stability gate lesson.** The lookback churned 33%, alternating between
adjacent grid values 10 and 20. Mild. Meanwhile `sma_crossover` had 0%
churn and was worthless. **Parameter stability is necessary but never
sufficient**, and this gate would have passed a dead strategy while
flagging a live one. Weight it accordingly.

**Decision:** volatility targeting becomes a standing baseline. Any
future candidate must beat it, not just buy-and-hold. It should be added
as a column in the performance report.

## 2026-08-28 — Universe expansion: survivorship bias must be designed for

**Decision:** before any cross-sectional work, the universe layer must
handle survivorship bias explicitly. This is recorded in advance because
it silently inflates every cross-sectional backtest and is far easier to
design around than to detect afterwards.

**Why it has not mattered so far.** SPY, QQQ, IWM and DIA are ETFs that
did not delist during the sample. There is no survivorship problem in
the current universe, which is exactly why the trap is invisible today
and will not be tomorrow.

**The failure mode.** Asking a data provider for "S&P 500 constituents"
returns TODAY's constituents. Backtesting those over 2017-2022 means
trading a basket selected for having survived and performed well enough
to remain in the index. Companies that went bankrupt, got acquired at a
discount, or were removed for underperformance are simply absent. The
effect is large, systematically positive, and completely invisible in
the results -- the equity curve looks clean.

**Requirement:** the universe must be point-in-time. On any given
session the tradeable set is what was actually listed and liquid on that
session, including names that later delisted. If point-in-time
membership data is unavailable, the honest alternatives are to use a
liquidity screen applied to the full listed universe as of each date, or
to state plainly in every report that results carry survivorship bias
and are therefore upper bounds.

**Second trap: look-ahead in the liquidity screen.** Filtering on
average dollar volume computed over the whole sample selects names that
became liquid later. Screens must use trailing data only, on the same
lag discipline already enforced in `weights.py` and the exposure-matched
null.

**Third trap: cross-sectional effective sample size.** Adding 300
symbols does not give 300 independent bets. Equities share a dominant
market factor. `effective_sample_size` in `significance.py` already
measures this and must be reported for any cross-sectional result. The
gain over the current universe is real but far smaller than the symbol
count suggests.

## 2026-08-28 — Positive control PASSED. Harness validated end to end.

**Volatility targeting, development split, SPY, target 10%, lookback 20,
5 bps costs, 0.05 rebalance band, rf 3%:**

```
LAYER 1  MECHANISM        PASS   correlation 0.559, shrunk R2 0.194
LAYER 2  IMPLEMENTATION   PASS   vol dispersion 10.29% -> 2.01%
                                 mean miss vs target 8.04% -> 1.56%
LAYER 3  ECONOMICS        PASS   Sharpe 0.651 vs 0.487

                    VOL TARGET   BUY & HOLD   STATIC BLEND
CAGR                     9.38%       11.10%          9.16%
volatility              10.00%       19.46%         13.93%
Sharpe                   0.651        0.487          0.487
max drawdown            11.73%       33.79%         25.10%
Calmar                   0.800        0.328          0.365
beta 0.447, alpha 2.28% (t = 1.13), turnover 2.69x, cost drag 0.13%
```

**The harness is validated.** It correctly killed a fake effect four
ways and now correctly confirms a real one. Every future negative result
means something because the pipeline demonstrably does not reject
everything by construction. That is the entire finding.

**Scrutiny applied equally, as it must be:**

- The Sharpe improvement is NOT statistically significant. 0.651 +/-
  0.450 against 0.487 +/- 0.432; alpha t = 1.13. Six years cannot
  establish a 0.16 Sharpe difference.
- What IS clean: the static blend at 71.6% weight has Sharpe 0.487,
  identical to buy-and-hold to three decimals, confirming scale
  invariance. So the entire gain comes from WHEN exposure was held, not
  how much. The mechanism is isolated structurally, not statistically.
- Drawdown 11.73% vs 33.79% and Calmar 0.800 vs 0.328 are the more
  robust results, because they follow from the validated layer 2 rather
  than from the return path cooperating.

**Why this differs from sma_crossover, and it is the whole point:**

```
                       sma_crossover   vol target
configurations tried              36            1
parameters from             the data   convention
mechanism stated first            no          yes
prediction pre-registered         no          yes
```

No selection correction is required because there was no selection.
A marginal result costs less credibility when it was not bought with 36
attempts.

**Precondition discovered while testing:** volatility targeting improves
Sharpe only when the asset has a positive expected return. With zero
drift the gross mean is zero, net return is minus the cost drag, and
Sharpe reduces to -cost/volatility -- so LOWERING volatility makes it
worse. This is not alpha; it is a more efficient way to hold beta, and
it needs beta to be worth holding. Caught by a fixture with no drift
producing the real schedule at the 5th percentile.

## 2026-08-28 — Weight-schedule permutation, and its power limits

**Added:** `permute_weight_schedule` and `schedule_turnover` in
`significance.py`; layers 4 and 5 in `run_volatility_target.py`.

**The confound, measured.** Shuffling a weight schedule inflates
turnover roughly sevenfold (22.0 -> 153.5 in testing), because
volatility-target weights are highly persistent and shuffling makes them
flip every session. The real strategy then wins on cost drag alone.
`circular_shift` preserves the path shape and therefore turnover
exactly, moving only placement in time. It is not a preference here, it
is the only valid null. The runner prints observed and null turnover so
the confound stays visible.

**Power characterised, not assumed.** On data engineered to contain a
real effect, the timing test reaches only:

```
n=1200 moderate dispersion    33rd percentile
n=1200 high dispersion        77th percentile
n=2500 high dispersion        60th percentile
```

Two attenuations. The analytic ceiling on the gain is
sqrt(E[1/s^2] * E[s^2]), about 1.27 for realistic equity dispersion.
Using a noisy 20-session trailing estimate instead of true volatility
erodes most of that, since the forecast correlates with forward
volatility at only about 0.5.

**CONSEQUENCE: a non-significant result from layer 4 does not indicate
absence of timing skill.** The test cannot resolve effects of this size
at these sample lengths. It is a directional check. Layer 5
(walk-forward) carries the real evidence.

**Deliberately not fixed by tuning.** The obvious move was to adjust
fixture parameters until the assertion passed. That is precisely the
failure this project exists to prevent, so the test asserts direction
only and documents the measured power instead.

**Also worth keeping:** compute the analytic ceiling before running a
permutation test. If the expected effect is small relative to sampling
noise, the test cannot resolve it and the compute is wasted.

## 2026-08-28 — Hypothesis framework and first real hypothesis

**The test a hypothesis must pass before any code is written:** who is
on the other side, and why do they keep losing? Every dollar of edge is
someone else's. If the counterparty cannot be named and their persistent
willingness to lose explained, it is a pattern, not a hypothesis.

`sma_crossover` failed this before it was ever run. "Price crossed a
line" names no counterparty and describes no mechanism.

Durable sources for a retail participant, in rough order of reliability:
forced traders (index inclusion, close auctions, margin calls, quarter
end), risk transfer (volatility premium, carry), mandate constraints,
and behavioural effects (weakest, most arbitraged).

**Structural decision — the universe is the binding constraint.**
SPY/QQQ/IWM/DIA correlate around 0.9, giving an effective sample size
near one instrument. That means every expressible strategy is a
market-timing strategy, the hardest category in the field. The universe
choice, made before any hypothesis existed, forced the worst odds
available. Broadening to a few hundred liquid names is agreed and is
prerequisite for any cross-sectional work.

**First hypothesis: volatility targeting, run as a positive control.**

The harness has only ever been pointed at things that fail. It has never
confirmed a true effect end to end, so a future negative result is
ambiguous: dead strategy, or pipeline eating signal? Volatility
targeting is documented, robust, and mechanism-driven, which makes it
the right calibration.

Split into two claims with very different power, tested separately:

- **Mechanism:** trailing volatility predicts forward volatility. High
  power, should be clearly true. This is the actual positive control.
- **Economic:** targeting constant risk improves Sharpe. Weak, path
  dependent, sample dependent.

**Prediction recorded in advance:** mechanism passes clearly; Sharpe may
FAIL on 2017-2022. March 2020 is the worst case for vol targeting —
volatility explodes, the model de-risks, and the V-recovery arrives
while exposure is still reduced. A Sharpe failure would be a real
property of the strategy on this sample, not a harness fault. A
*mechanism* failure would indicate a broken pipeline.

## 2026-08-28 — Weight-based backtesting added

**Decision:** `backtest/weights.py` runs target-weight schedules with
drift, rebalancing bands and turnover costs, alongside the existing
discrete-trade runner.

**Reasoning:** the runner expresses exactly one shape — signal, entry,
stop, forced exit. Volatility targeting is a continuous exposure level,
and every cross-sectional strategy (rank, weight, rebalance) is
weight-based too. Neither fits the trade-based runner without distortion.

**Enforced inside the module, not left to callers:** weights are lagged
one session, and turnover is charged. Both are easy to get wrong in ways
that silently manufacture edge, and a look-ahead weight schedule
produces an entirely plausible-looking curve. There is a test asserting
a schedule invested only on the single best session captures nothing.

## 2026-08-28 — The positive control failed, and the failure was the lesson

**What happened:** the mechanism test initially reported R-squared of
-0.046 on simulated GARCH data with obvious volatility clustering. The
control existed to catch exactly this, and it did.

**Diagnosis:** trailing realized volatility is an unbiased but NOISY
estimator. Its dispersion (sd 0.119) slightly exceeds forward realized
volatility's (0.117), so used as a raw point forecast the estimator
noise inflates mean squared error past the unconditional mean —
producing a negative R-squared while the correlation was 0.484 and the
true conditional-volatility autocorrelation was 0.561.

**An uncalibrated forecast can make a genuine effect look like nothing.**
Had this run on a novel strategy instead of a known-positive control,
the conclusion would have been "no effect" and the finding discarded.

**Fix:** `volatility_forecast_skill` now reports three numbers, and only
the third is the verdict — correlation (information content),
`raw_r_squared` (calibration diagnostic, often negative), and
`shrunk_r_squared` (expanding-window shrinkage fit, the honest measure).
On the same data the shrunk R-squared is 0.239 with a shrinkage slope of
0.478.

**Look-ahead guard:** the expanding fit at observation i uses only
observations whose forward window closed before i. Overlapping forward
windows would otherwise leak future data into the slope.

**Carry this forward:** before concluding a hypothesis has no effect,
check whether the forecast is calibrated. Correlation and R-squared can
disagree, and R-squared alone will discard real signal.

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
