"""
Volatility targeting, and the mechanism test that justifies it.

The hypothesis has two claims with very different statistical power, and
they must be tested separately:

**Mechanism claim.** Trailing realized volatility predicts forward
realized volatility. This should be strongly true -- volatility clusters
-- and it is the reason the strategy has any basis at all. It is the
project's positive control: a pipeline that cannot confirm this is
broken, and every negative result it has produced is suspect.

**Economic claim.** Holding constant risk rather than constant dollars
improves Sharpe. Far weaker, path dependent, and sample dependent. On a
window containing March 2020 it may well fail: volatility explodes, the
model de-risks, and the recovery arrives while exposure is still
reduced. That failure would be a real property of the strategy, not a
harness fault.

Note what this is not. Volatility targeting does not predict returns and
makes no claim to alpha. It is a risk-management transformation, so the
honest expectation is beta near one, alpha near zero, and a Sharpe that
may or may not improve.

**Precondition: the asset must have a positive expected return.** The
strategy harvests a risk premium more efficiently by holding constant
risk rather than constant dollars. With zero drift there is no premium
to harvest, and targeting actively HURTS Sharpe: the numerator becomes
the negative cost drag while targeting shrinks the denominator, so lower
volatility makes the ratio worse. Answering "who is on the other side"
for this strategy: nobody. It is a better way to hold beta, and it needs
beta to be worth holding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def realized_volatility(
    returns: pd.Series,
    *,
    lookback: int = 20,
    annualize: bool = True,
) -> pd.Series:
    """
    Trailing realized volatility through each session's close.

    The value at session t uses returns up to and including t, so it is
    knowable at that close. Anything acting on it must wait for the next
    session; ``run_weight_backtest`` enforces that lag.
    """

    if lookback < 2:
        raise ValueError("lookback must be at least 2")

    vol = returns.rolling(lookback).std(ddof=1)

    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS_PER_YEAR)

    return vol


def volatility_target_weights(
    returns: pd.Series,
    *,
    target_volatility: float = 0.10,
    lookback: int = 20,
    max_weight: float = 1.0,
    min_weight: float = 0.0,
) -> pd.Series:
    """
    Weight that would have held risk at ``target_volatility``.

    ``max_weight`` of 1.0 makes this de-risk only, never levering up,
    which is the honest default for an account without margin. Raising
    it above 1.0 permits leverage when volatility is low and changes the
    risk profile materially.

    Sessions without enough history to estimate volatility get a weight
    of zero rather than a guess.
    """

    if target_volatility <= 0:
        raise ValueError("target_volatility must be greater than zero")

    if max_weight < min_weight:
        raise ValueError("max_weight cannot be below min_weight")

    if min_weight < 0:
        raise ValueError("min_weight cannot be negative")

    vol = realized_volatility(returns, lookback=lookback)

    weights = target_volatility / vol.replace(0.0, np.nan)
    weights = weights.clip(lower=min_weight, upper=max_weight)

    return weights.fillna(0.0)


def volatility_forecast_skill(
    returns: pd.Series,
    *,
    lookback: int = 20,
    horizon: int = 20,
    burn_in: int = 250,
) -> dict:
    """
    Does trailing volatility predict forward volatility?

    This is the mechanism test, and it has a subtlety that cost this
    project a failed positive control before it was understood.

    Trailing realized volatility is an unbiased but NOISY estimate of
    current volatility. Its dispersion exceeds that of forward realized
    volatility, so used as a raw point forecast the estimator noise
    inflates mean squared error past the unconditional mean -- producing
    a NEGATIVE R-squared even when the correlation is strongly positive
    and the underlying clustering is real. An uncalibrated forecast can
    make a genuine effect look like nothing.

    So three numbers are reported, and only the third is the verdict:

    ``correlation``
        Information content. Is there any relationship at all?
    ``raw_r_squared``
        Skill of trailing volatility used directly as a forecast.
        Frequently negative. Not a verdict, a calibration diagnostic.
    ``shrunk_r_squared``
        Skill after shrinking toward the mean, with the shrinkage slope
        fitted on an expanding window. This is the honest measure.

    The expanding fit at observation ``i`` uses only observations whose
    forward window closed before ``i``, so neither the slope nor the
    baseline ever sees data from the period being predicted. Overlapping
    forward windows would otherwise leak future information into the
    fit.
    """

    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    if burn_in < 30:
        raise ValueError("burn_in must be at least 30 observations")

    trailing = realized_volatility(returns, lookback=lookback)

    forward = (
        returns
        .rolling(horizon)
        .std(ddof=1)
        .shift(-horizon)
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )

    frame = pd.DataFrame(
        {"trailing": trailing, "forward": forward}
    ).dropna()

    if len(frame) < burn_in + horizon + 30:
        raise ValueError(
            "not enough overlapping observations to measure forecast "
            f"skill: need {burn_in + horizon + 30}, have {len(frame)}"
        )

    trailing_values = frame["trailing"].to_numpy()
    forward_values = frame["forward"].to_numpy()

    unconditional = float(forward_values.mean())

    raw_sse = float(((forward_values - trailing_values) ** 2).sum())
    baseline_sse = float(((forward_values - unconditional) ** 2).sum())
    raw_r_squared = (
        1.0 - raw_sse / baseline_sse if baseline_sse > 0 else 0.0
    )

    predictions = []
    actuals = []
    baselines = []
    slopes = []

    for index in range(burn_in, len(frame)):
        # Only observations whose forward window has already closed.
        usable = index - horizon

        if usable < 30:
            continue

        x = trailing_values[:usable]
        y = forward_values[:usable]

        variance = float(np.var(x, ddof=1))

        if variance <= 0:
            continue

        slope = float(np.cov(x, y, ddof=1)[0, 1] / variance)
        intercept = float(y.mean() - slope * x.mean())

        predictions.append(intercept + slope * trailing_values[index])
        actuals.append(forward_values[index])
        baselines.append(float(y.mean()))
        slopes.append(slope)

    if not predictions:
        raise ValueError("expanding-window fit produced no predictions")

    predictions = np.asarray(predictions)
    actuals = np.asarray(actuals)
    baselines = np.asarray(baselines)

    shrunk_sse = float(((actuals - predictions) ** 2).sum())
    shrunk_baseline_sse = float(((actuals - baselines) ** 2).sum())
    shrunk_r_squared = (
        1.0 - shrunk_sse / shrunk_baseline_sse
        if shrunk_baseline_sse > 0
        else 0.0
    )

    return {
        "observations": int(len(frame)),
        "evaluated_out_of_sample": int(len(predictions)),
        "lookback": lookback,
        "horizon": horizon,
        "correlation": float(frame["trailing"].corr(frame["forward"])),
        "raw_r_squared": raw_r_squared,
        "shrunk_r_squared": shrunk_r_squared,
        "mean_shrinkage_slope": float(np.mean(slopes)),
        "volatility_autocorrelation": float(
            trailing.dropna().autocorr(lag=horizon)
        ),
        "mean_forward_volatility": unconditional,
        "beats_unconditional_mean": bool(shrunk_r_squared > 0),
    }


def volatility_capture(
    returns: pd.Series,
    weights: pd.Series,
    *,
    target_volatility: float,
    window: int = 60,
) -> dict:
    """
    Did targeting actually stabilise realized risk?

    The second-order mechanism check. If the forecast has skill, the
    volatility of the targeted series should sit closer to the target
    and vary less than the volatility of the untargeted series. This is
    nearly mechanical when the forecast works, and failing it points at
    an implementation fault rather than a market fact.
    """

    held = weights.shift(1).fillna(0.0)
    targeted = held * returns

    raw_vol = realized_volatility(returns, lookback=window).dropna()
    targeted_vol = realized_volatility(targeted, lookback=window).dropna()

    shared = raw_vol.index.intersection(targeted_vol.index)
    raw_vol = raw_vol.loc[shared]
    targeted_vol = targeted_vol.loc[shared]

    return {
        "raw_mean_volatility": float(raw_vol.mean()),
        "targeted_mean_volatility": float(targeted_vol.mean()),
        "raw_volatility_dispersion": float(raw_vol.std(ddof=1)),
        "targeted_volatility_dispersion": float(
            targeted_vol.std(ddof=1)
        ),
        "raw_mean_absolute_miss": float(
            (raw_vol - target_volatility).abs().mean()
        ),
        "targeted_mean_absolute_miss": float(
            (targeted_vol - target_volatility).abs().mean()
        ),
        "dispersion_reduced": bool(
            targeted_vol.std(ddof=1) < raw_vol.std(ddof=1)
        ),
    }
