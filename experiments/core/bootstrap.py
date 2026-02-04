"""
Bootstrap confidence interval utilities.

Provides specialized bootstrap functions for mean and difference CIs
with deterministic seeding for reproducibility.
"""

from typing import Optional

import numpy as np


def bootstrap_mean_ci(
    values: list[float],
    n_bootstrap: int = 10000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict:
    """
    Bootstrap confidence interval for the mean.

    Args:
        values: List of values to bootstrap
        n_bootstrap: Number of bootstrap replicates
        seed: Random seed for reproducibility
        ci: Confidence interval level (default 0.95)

    Returns:
        Dictionary with mean, lower, upper, and bootstrap distribution
    """
    if not values:
        return {
            "mean": 0.0,
            "lower": 0.0,
            "upper": 0.0,
            "n_bootstrap": n_bootstrap,
            "seed": seed,
        }

    np.random.seed(seed)
    values_arr = np.array(values)
    n = len(values_arr)

    bootstrap_means = np.array(
        [np.mean(np.random.choice(values_arr, size=n, replace=True)) for _ in range(n_bootstrap)]
    )

    alpha = 1 - ci
    lower = float(np.percentile(bootstrap_means, alpha / 2 * 100))
    upper = float(np.percentile(bootstrap_means, (1 - alpha / 2) * 100))

    return {
        "mean": float(np.mean(values_arr)),
        "lower": lower,
        "upper": upper,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }


def bootstrap_difference_ci(
    values1: list[float],
    values2: list[float],
    n_bootstrap: int = 10000,
    seed: int = 42,
    ci: float = 0.95,
    paired: bool = True,
) -> dict:
    """
    Bootstrap confidence interval for the difference between two means.

    Args:
        values1: First list of values
        values2: Second list of values
        n_bootstrap: Number of bootstrap replicates
        seed: Random seed for reproducibility
        ci: Confidence interval level (default 0.95)
        paired: Whether to use paired bootstrap (default True)

    Returns:
        Dictionary with difference, lower, upper, and significance
    """
    if not values1 or not values2:
        return {
            "difference": 0.0,
            "lower": 0.0,
            "upper": 0.0,
            "significant": False,
            "n_bootstrap": n_bootstrap,
            "seed": seed,
        }

    np.random.seed(seed)
    arr1 = np.array(values1)
    arr2 = np.array(values2)

    if paired:
        # Paired bootstrap: resample indices
        if len(arr1) != len(arr2):
            raise ValueError("Paired bootstrap requires equal length arrays")
        n = len(arr1)
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(n, size=n, replace=True)
            diff = np.mean(arr1[indices]) - np.mean(arr2[indices])
            bootstrap_diffs.append(diff)
    else:
        # Unpaired bootstrap: resample each independently
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            mean1 = np.mean(np.random.choice(arr1, size=len(arr1), replace=True))
            mean2 = np.mean(np.random.choice(arr2, size=len(arr2), replace=True))
            bootstrap_diffs.append(mean1 - mean2)

    bootstrap_diffs = np.array(bootstrap_diffs)
    alpha = 1 - ci
    lower = float(np.percentile(bootstrap_diffs, alpha / 2 * 100))
    upper = float(np.percentile(bootstrap_diffs, (1 - alpha / 2) * 100))

    # Significant if CI doesn't include zero
    significant = lower > 0 or upper < 0

    return {
        "difference": float(np.mean(arr1) - np.mean(arr2)),
        "lower": lower,
        "upper": upper,
        "significant": significant,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }


def bootstrap_proportion_ci(
    successes: int,
    total: int,
    n_bootstrap: int = 10000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict:
    """
    Bootstrap confidence interval for a proportion.

    Args:
        successes: Number of successes
        total: Total number of trials
        n_bootstrap: Number of bootstrap replicates
        seed: Random seed for reproducibility
        ci: Confidence interval level (default 0.95)

    Returns:
        Dictionary with proportion, lower, upper
    """
    if total == 0:
        return {
            "proportion": 0.0,
            "lower": 0.0,
            "upper": 0.0,
            "n_bootstrap": n_bootstrap,
            "seed": seed,
        }

    np.random.seed(seed)

    # Create binary array
    values = np.array([1] * successes + [0] * (total - successes))

    bootstrap_props = np.array(
        [np.mean(np.random.choice(values, size=total, replace=True)) for _ in range(n_bootstrap)]
    )

    alpha = 1 - ci
    lower = float(np.percentile(bootstrap_props, alpha / 2 * 100))
    upper = float(np.percentile(bootstrap_props, (1 - alpha / 2) * 100))

    return {
        "proportion": successes / total,
        "lower": lower,
        "upper": upper,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }
