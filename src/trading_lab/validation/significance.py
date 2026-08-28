"""
Selection-bias and significance tooling.

The central question this module answers: given that N parameter
combinations were tried and the best one was kept, how good would the
best of N *worthless* strategies have looked on the same data?

Without that number, a profit factor of 1.46 selected from a 36-point
sweep is uninterpretable. It might be edge. It might be the expected
maximum of 36 draws from a distribution centred on nothing.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import pandas as pd


EULER_MASCHERONI = 0.5772156649015329


# ---------------------------------------------------------------------
# Signal permutation: building the null
# ---------------------------------------------------------------------


def permute_signals(
    df: pd.DataFrame,
    *,
    rng: np.random.Generator,
    method: str = "circular_shift",
) -> pd.DataFrame:
    """
    Produce a null version of a signal column, per symbol.

    Both methods preserve the number of signals per symbol, so the null
    strategy trades at the same frequency as the real one and incurs the
    same costs. Only the timing is destroyed.

    ``circular_shift`` rotates each symbol's signal series by a random
    offset. This preserves the clustering structure of signals (trend
    strategies fire in bursts) while breaking their alignment with
    price. It is the more conservative and generally more honest null.

    ``shuffle`` fully randomizes signal positions. It destroys clustering
    as well as alignment, which makes the null easier to beat and
    therefore flatters the strategy. Use it only as a cross-check.
    """

    if "signal" not in df.columns:
        raise ValueError("df must contain a 'signal' column")

    if method not in {"circular_shift", "shuffle"}:
        raise ValueError(f"unknown permutation method: {method!r}")

    result = df.sort_values(["symbol", "timestamp"]).copy()
    permuted = np.empty(len(result), dtype=bool)

    position = 0

    for _, group in result.groupby("symbol", sort=True):
        signals = group["signal"].to_numpy(dtype=bool)
        size = len(signals)

        if size == 0:
            continue

        if method == "circular_shift":
            offset = int(rng.integers(0, size))
            block = np.roll(signals, offset)
        else:
            block = rng.permutation(signals)

        permuted[position:position + size] = block
        position += size

    result["signal"] = permuted

    return result


def build_null_distribution(
    df: pd.DataFrame,
    *,
    evaluate: Callable[[pd.DataFrame], float],
    n_permutations: int = 200,
    method: str = "circular_shift",
    seed: int = 0,
) -> np.ndarray:
    """
    Run ``evaluate`` against ``n_permutations`` permuted signal sets.

    ``evaluate`` must take a signal frame and return the single number
    being tested. To account for selection bias, it should perform the
    *entire* selection procedure and return the best result — sweeping
    36 combinations and returning the winner. Evaluating one fixed
    parameter set understates the bias you are trying to measure.
    """

    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1")

    rng = np.random.default_rng(seed)
    results = np.empty(n_permutations, dtype=float)

    for index in range(n_permutations):
        permuted = permute_signals(df, rng=rng, method=method)
        results[index] = float(evaluate(permuted))

    return results


def permutation_p_value(
    observed: float,
    null_distribution: Sequence[float],
    *,
    higher_is_better: bool = True,
) -> float:
    """
    One-sided p-value with the standard +1 correction.

    The correction keeps the p-value from ever reaching exactly zero,
    which is the honest treatment: a finite number of permutations can
    never prove impossibility.
    """

    null = np.asarray(list(null_distribution), dtype=float)
    null = null[np.isfinite(null)]

    if null.size == 0:
        raise ValueError("null distribution is empty after filtering")

    if higher_is_better:
        at_least_as_extreme = int((null >= observed).sum())
    else:
        at_least_as_extreme = int((null <= observed).sum())

    return (at_least_as_extreme + 1) / (null.size + 1)


def summarize_against_null(
    observed: float,
    null_distribution: Sequence[float],
    *,
    higher_is_better: bool = True,
    label: str = "metric",
) -> dict:
    """
    Package an observed result against its null with a plain verdict.
    """

    null = np.asarray(list(null_distribution), dtype=float)
    finite = null[np.isfinite(null)]

    p_value = permutation_p_value(
        observed,
        finite,
        higher_is_better=higher_is_better,
    )

    percentile = float((finite < observed).mean() * 100)

    return {
        "label": label,
        "observed": float(observed),
        "null_mean": float(finite.mean()),
        "null_median": float(np.median(finite)),
        "null_p95": float(np.percentile(finite, 95)),
        "null_max": float(finite.max()),
        "null_samples": int(finite.size),
        "discarded_samples": int(null.size - finite.size),
        "percentile_of_null": percentile,
        "p_value": p_value,
        "significant_at_05": bool(p_value < 0.05),
    }


# ---------------------------------------------------------------------
# Analytic selection-bias correction
# ---------------------------------------------------------------------


def _standard_normal_ppf(probability: float) -> float:
    """Inverse standard normal CDF via the error function."""

    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be strictly between 0 and 1")

    # ppf(p) = sqrt(2) * erfinv(2p - 1); erfinv via bisection on erf.
    target = 2.0 * probability - 1.0

    low, high = -10.0, 10.0

    for _ in range(200):
        middle = (low + high) / 2.0

        if math.erf(middle) < target:
            low = middle
        else:
            high = middle

    return math.sqrt(2.0) * (low + high) / 2.0


def _standard_normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def expected_maximum_sharpe(
    n_trials: int,
    *,
    trial_sharpe_variance: float,
) -> float:
    """
    Expected best Sharpe from ``n_trials`` strategies with no real edge.

    This is the Bailey and Lopez de Prado result. ``trial_sharpe_variance``
    is the variance of the (non-annualized, per-observation) Sharpe ratios
    actually observed across your sweep — compute it from the sweep
    itself, do not guess.

    The practical use: if your best result is below this number, you have
    found nothing, no matter how good the profit factor looks.
    """

    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")

    if trial_sharpe_variance < 0:
        raise ValueError("trial_sharpe_variance cannot be negative")

    if n_trials == 1 or trial_sharpe_variance == 0:
        return 0.0

    scale = math.sqrt(trial_sharpe_variance)

    first = _standard_normal_ppf(1.0 - 1.0 / n_trials)
    second = _standard_normal_ppf(1.0 - 1.0 / (n_trials * math.e))

    return scale * (
        (1.0 - EULER_MASCHERONI) * first
        + EULER_MASCHERONI * second
    )


def deflated_sharpe_ratio(
    observed_sharpe: float,
    *,
    n_observations: int,
    n_trials: int,
    trial_sharpe_variance: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> dict:
    """
    Probability the observed Sharpe is real given how many were tried.

    ``observed_sharpe`` must be per-observation, not annualized: divide
    an annualized figure by sqrt(252) for daily returns. Returns the
    deflated Sharpe probability, where values below roughly 0.95 mean the
    result is not distinguishable from the best of N lucky draws.
    """

    if n_observations < 2:
        raise ValueError("n_observations must be at least 2")

    benchmark = expected_maximum_sharpe(
        n_trials,
        trial_sharpe_variance=trial_sharpe_variance,
    )

    denominator = (
        1.0
        - skew * observed_sharpe
        + (kurtosis - 1.0) / 4.0 * observed_sharpe**2
    )

    if denominator <= 0:
        raise ValueError(
            "degenerate higher moments: cannot deflate this Sharpe"
        )

    statistic = (
        (observed_sharpe - benchmark)
        * math.sqrt(n_observations - 1)
        / math.sqrt(denominator)
    )

    probability = _standard_normal_cdf(statistic)

    return {
        "observed_sharpe": float(observed_sharpe),
        "expected_max_under_null": float(benchmark),
        "deflated_sharpe_probability": float(probability),
        "n_trials": int(n_trials),
        "n_observations": int(n_observations),
        "passes_at_95": bool(probability >= 0.95),
    }


# ---------------------------------------------------------------------
# Effective sample size
# ---------------------------------------------------------------------


def effective_sample_size(
    returns_by_stream: dict[str, pd.Series],
) -> dict:
    """
    Correlation-adjusted count of independent observations.

    Four ETFs correlated at 0.9 do not give four independent streams,
    and 200 trades across them is not 200 observations. The current
    ``minimum_trades: 200`` gate treats them as if it were, which is the
    quiet reason a 36-point sweep looked well powered when it was not.

    Uses the standard variance-inflation adjustment:
        n_eff = n * k / (1 + (k - 1) * mean_pairwise_correlation)
    """

    if len(returns_by_stream) == 0:
        raise ValueError("no return streams supplied")

    frame = pd.DataFrame(returns_by_stream).dropna()

    if frame.empty:
        raise ValueError("return streams share no overlapping periods")

    streams = frame.shape[1]
    observations = frame.shape[0]
    nominal = observations * streams

    if streams == 1:
        return {
            "streams": 1,
            "observations_per_stream": observations,
            "nominal_sample_size": nominal,
            "mean_pairwise_correlation": 0.0,
            "effective_sample_size": float(nominal),
            "inflation_factor": 1.0,
        }

    correlations = frame.corr().to_numpy()
    upper = correlations[np.triu_indices(streams, k=1)]
    mean_correlation = float(np.nanmean(upper))

    divisor = 1.0 + (streams - 1) * mean_correlation

    if divisor <= 0:
        effective = float(nominal)
    else:
        effective = nominal / divisor

    return {
        "streams": int(streams),
        "observations_per_stream": int(observations),
        "nominal_sample_size": int(nominal),
        "mean_pairwise_correlation": mean_correlation,
        "effective_sample_size": float(effective),
        "inflation_factor": float(nominal / effective)
        if effective > 0
        else float("inf"),
    }
