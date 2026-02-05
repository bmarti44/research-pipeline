"""
Statistical functions for format friction analysis.

Provides Wilson confidence intervals, sign tests, Cohen's kappa,
bootstrap CIs, and distribution metrics including Hartigan's dip test.
"""

import math
from typing import Optional

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.proportion import proportion_confint


def wilson_ci(
    successes: int, total: int, confidence: float = 0.95
) -> tuple[float, float]:
    """
    Wilson score interval for binomial proportions.

    Args:
        successes: Number of successes
        total: Total number of trials
        confidence: Confidence level (default 0.95)

    Returns:
        Tuple of (lower bound, upper bound)
    """
    if total == 0:
        return (0.0, 1.0)
    return proportion_confint(successes, total, alpha=1 - confidence, method="wilson")


def sign_test(
    n_positive: int, n_total: int, alternative: str = "greater"
) -> float:
    """
    Sign test for paired binary data.

    Args:
        n_positive: Number of positive differences
        n_total: Total number of non-tied pairs
        alternative: 'greater', 'less', or 'two-sided'

    Returns:
        p-value from binomial test
    """
    if n_total == 0:
        return 1.0
    result = binomtest(n_positive, n_total, 0.5, alternative=alternative)
    return result.pvalue


def cohens_kappa(labels1: list, labels2: list) -> float:
    """
    Cohen's kappa for inter-rater agreement.

    Args:
        labels1: First rater's labels
        labels2: Second rater's labels

    Returns:
        Cohen's kappa coefficient
    """
    if len(labels1) != len(labels2) or len(labels1) == 0:
        return 0.0
    return cohen_kappa_score(labels1, labels2)


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 10000,
    seed: int = 42,
    ci: float = 0.95,
) -> tuple[float, float]:
    """
    Bootstrap confidence interval with fixed seed for reproducibility.

    Args:
        values: List of values to bootstrap
        n_bootstrap: Number of bootstrap replicates
        seed: Random seed for reproducibility
        ci: Confidence interval level (default 0.95)

    Returns:
        Tuple of (lower bound, upper bound)
    """
    if not values:
        return (0.0, 0.0)

    rng = np.random.default_rng(seed)
    values_arr = np.array(values)
    n = len(values_arr)

    means = np.array(
        [np.mean(rng.choice(values_arr, size=n, replace=True)) for _ in range(n_bootstrap)]
    )

    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return (float(lower), float(upper))


def dip_test(values: list[float]) -> tuple[float, float]:
    """
    Hartigan's dip test for bimodality.

    Args:
        values: List of values to test

    Returns:
        Tuple of (dip statistic, p-value)
    """
    try:
        from diptest import diptest as _diptest

        arr = np.array(values)
        return _diptest(arr)
    except ImportError:
        # Fallback if diptest not installed
        return (0.0, 1.0)


def compute_distribution_metrics(friction_values: list[float]) -> dict:
    """
    Compute full distribution metrics to expose bimodality.

    Per REVIEW.md, reporting only the mean hides bimodal distribution
    where 41% show zero friction and 41% show severe friction (>30pp).

    Args:
        friction_values: List of friction values (percentage points)

    Returns:
        Dictionary with comprehensive distribution metrics
    """
    if not friction_values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "iqr": 0.0,
            "min": 0.0,
            "max": 0.0,
            "percentiles": {},
            "n_zero": 0,
            "n_severe": 0,
            "pct_zero": 0.0,
            "pct_severe": 0.0,
            "dip_statistic": 0.0,
            "dip_pvalue": 1.0,
        }

    arr = np.array(friction_values)
    dip_stat, dip_p = dip_test(friction_values)

    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "percentiles": {
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
        },
        "n_zero": int(np.sum(arr == 0)),
        "n_severe": int(np.sum(arr > 30)),
        "pct_zero": float(np.sum(arr == 0) / len(arr) * 100),
        "pct_severe": float(np.sum(arr > 30) / len(arr) * 100),
        "dip_statistic": float(dip_stat),
        "dip_pvalue": float(dip_p),
    }


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """
    Benjamini-Hochberg procedure for multiple comparison correction.

    Args:
        p_values: List of p-values to correct
        alpha: Family-wise error rate (default 0.05)

    Returns:
        List of booleans indicating which hypotheses are rejected
    """
    n = len(p_values)
    if n == 0:
        return []

    # Sort p-values with original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    # Find BH threshold
    rejected = [False] * n
    max_k = 0

    for k, (orig_idx, p) in enumerate(indexed, 1):
        if p <= k * alpha / n:
            max_k = k

    # Reject all hypotheses with rank <= max_k
    for k, (orig_idx, p) in enumerate(indexed, 1):
        if k <= max_k:
            rejected[orig_idx] = True

    return rejected


def mcnemar_test(b: int, c: int) -> tuple[float, float]:
    """
    McNemar's test for paired binary data.

    Args:
        b: Count of pairs where first is 1 and second is 0
        c: Count of pairs where first is 0 and second is 1

    Returns:
        Tuple of (chi-squared statistic, p-value)
    """
    from scipy.stats import chi2

    if b + c == 0:
        return (0.0, 1.0)

    chi_sq = (b - c) ** 2 / (b + c)
    p_value = 1 - chi2.cdf(chi_sq, df=1)
    return (chi_sq, p_value)
