"""
Statistical tests registry for research studies.

Pre-built statistical tests that can be selected via configuration.
All tests are registered by name and can be dynamically invoked.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import numpy as np
from scipy import stats
from functools import wraps


# =============================================================================
# Test Result Container
# =============================================================================

@dataclass
class TestResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    effect_size_name: Optional[str] = None
    confidence_interval: Optional[tuple[float, float]] = None
    confidence_level: float = 0.95

    # Additional details
    n_observations: int = 0
    degrees_of_freedom: Optional[float] = None
    power: Optional[float] = None

    # Interpretation helpers
    significant: bool = False
    alpha: float = 0.05
    interpretation: str = ""

    # Raw data for reproducibility
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "effect_size_name": self.effect_size_name,
            "confidence_interval": self.confidence_interval,
            "confidence_level": self.confidence_level,
            "n_observations": self.n_observations,
            "degrees_of_freedom": self.degrees_of_freedom,
            "power": self.power,
            "significant": self.significant,
            "alpha": self.alpha,
            "interpretation": self.interpretation,
            "details": self.details,
        }


# =============================================================================
# Test Registry
# =============================================================================

# Global registry of statistical tests
_TEST_REGISTRY: dict[str, Callable] = {}


def register_test(name: str):
    """Decorator to register a statistical test."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        _TEST_REGISTRY[name] = wrapper
        wrapper.test_name = name
        return wrapper
    return decorator


def get_test(name: str) -> Callable:
    """Get a test by name from the registry."""
    if name not in _TEST_REGISTRY:
        available = list(_TEST_REGISTRY.keys())
        raise ValueError(f"Unknown test: {name}. Available: {available}")
    return _TEST_REGISTRY[name]


def list_tests() -> list[str]:
    """List all registered tests."""
    return list(_TEST_REGISTRY.keys())


def run_test(name: str, *args, **kwargs) -> TestResult:
    """Run a test by name."""
    test_fn = get_test(name)
    return test_fn(*args, **kwargs)


# =============================================================================
# Pre-built Statistical Tests
# =============================================================================

# -----------------------------------------------------------------------------
# Proportion Tests
# -----------------------------------------------------------------------------

@register_test("two_proportion_z")
def two_proportion_z_test(
    successes1: int,
    n1: int,
    successes2: int,
    n2: int,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TestResult:
    """
    Two-proportion z-test for comparing two independent proportions.

    Args:
        successes1: Number of successes in group 1
        n1: Total observations in group 1
        successes2: Number of successes in group 2
        n2: Total observations in group 2
        alpha: Significance level
        alternative: 'two-sided', 'less', or 'greater'
    """
    p1 = successes1 / n1 if n1 > 0 else 0
    p2 = successes2 / n2 if n2 > 0 else 0

    # Pooled proportion
    p_pooled = (successes1 + successes2) / (n1 + n2)

    # Standard error
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))

    # Z statistic
    if se > 0:
        z = (p1 - p2) / se
    else:
        z = 0.0

    # P-value
    if alternative == "two-sided":
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    elif alternative == "less":
        p_value = stats.norm.cdf(z)
    else:  # greater
        p_value = 1 - stats.norm.cdf(z)

    # Effect size: Cohen's h
    h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

    # Confidence interval for difference
    se_diff = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    z_crit = stats.norm.ppf(1 - alpha/2)
    ci = (p1 - p2 - z_crit * se_diff, p1 - p2 + z_crit * se_diff)

    return TestResult(
        test_name="two_proportion_z",
        statistic=z,
        p_value=p_value,
        effect_size=h,
        effect_size_name="Cohen's h",
        confidence_interval=ci,
        n_observations=n1 + n2,
        significant=p_value < alpha,
        alpha=alpha,
        interpretation=f"p1={p1:.3f}, p2={p2:.3f}, diff={p1-p2:.3f}",
        details={
            "p1": p1,
            "p2": p2,
            "difference": p1 - p2,
            "pooled_proportion": p_pooled,
            "alternative": alternative,
        },
    )


@register_test("chi_square")
def chi_square_test(
    observed: list[list[int]],
    alpha: float = 0.05,
) -> TestResult:
    """
    Chi-square test of independence.

    Args:
        observed: 2D contingency table (list of lists)
        alpha: Significance level
    """
    observed = np.array(observed)
    chi2, p_value, dof, expected = stats.chi2_contingency(observed)

    # Effect size: Cramér's V
    n = observed.sum()
    min_dim = min(observed.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

    return TestResult(
        test_name="chi_square",
        statistic=chi2,
        p_value=p_value,
        effect_size=cramers_v,
        effect_size_name="Cramér's V",
        n_observations=int(n),
        degrees_of_freedom=dof,
        significant=p_value < alpha,
        alpha=alpha,
        interpretation=f"χ²({dof})={chi2:.2f}, V={cramers_v:.3f}",
        details={
            "observed": observed.tolist(),
            "expected": expected.tolist(),
        },
    )


@register_test("mcnemar")
def mcnemar_test(
    b: int,  # Success->Failure count
    c: int,  # Failure->Success count
    alpha: float = 0.05,
    exact: bool = False,
) -> TestResult:
    """
    McNemar's test for paired nominal data.

    For paired before/after or matched pairs designs.

    Args:
        b: Discordant pairs (+ then -)
        c: Discordant pairs (- then +)
        alpha: Significance level
        exact: Use exact binomial test (for small samples)
    """
    if exact or (b + c) < 25:
        # Exact binomial test
        p_value = stats.binom_test(b, b + c, 0.5, alternative='two-sided')
        statistic = b  # Just report the count
    else:
        # Chi-square approximation with continuity correction
        statistic = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0
        p_value = 1 - stats.chi2.cdf(statistic, df=1)

    # Effect size: odds ratio
    odds_ratio = b / c if c > 0 else float('inf')

    return TestResult(
        test_name="mcnemar",
        statistic=statistic,
        p_value=p_value,
        effect_size=odds_ratio,
        effect_size_name="Odds Ratio (b/c)",
        n_observations=b + c,
        degrees_of_freedom=1,
        significant=p_value < alpha,
        alpha=alpha,
        interpretation=f"b={b}, c={c}, OR={odds_ratio:.2f}",
        details={
            "b": b,
            "c": c,
            "exact": exact or (b + c) < 25,
        },
    )


@register_test("fisher_exact")
def fisher_exact_test(
    table: list[list[int]],
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TestResult:
    """
    Fisher's exact test for 2x2 contingency tables.

    Better than chi-square for small samples.

    Args:
        table: 2x2 contingency table [[a, b], [c, d]]
        alpha: Significance level
        alternative: 'two-sided', 'less', or 'greater'
    """
    table = np.array(table)
    odds_ratio, p_value = stats.fisher_exact(table, alternative=alternative)

    n = table.sum()

    return TestResult(
        test_name="fisher_exact",
        statistic=odds_ratio,
        p_value=p_value,
        effect_size=odds_ratio,
        effect_size_name="Odds Ratio",
        n_observations=int(n),
        significant=p_value < alpha,
        alpha=alpha,
        interpretation=f"OR={odds_ratio:.2f}",
        details={
            "table": table.tolist(),
            "alternative": alternative,
        },
    )


# -----------------------------------------------------------------------------
# Continuous Tests
# -----------------------------------------------------------------------------

@register_test("independent_t")
def independent_t_test(
    group1: list[float],
    group2: list[float],
    alpha: float = 0.05,
    equal_var: bool = True,
    alternative: str = "two-sided",
) -> TestResult:
    """
    Independent samples t-test.

    Args:
        group1: Values for group 1
        group2: Values for group 2
        alpha: Significance level
        equal_var: Assume equal variances (Student's t) or not (Welch's t)
        alternative: 'two-sided', 'less', or 'greater'
    """
    group1 = np.array(group1)
    group2 = np.array(group2)

    t_stat, p_value = stats.ttest_ind(
        group1, group2,
        equal_var=equal_var,
        alternative=alternative,
    )

    # Effect size: Cohen's d
    pooled_std = np.sqrt(
        ((len(group1)-1)*group1.std(ddof=1)**2 + (len(group2)-1)*group2.std(ddof=1)**2) /
        (len(group1) + len(group2) - 2)
    )
    cohens_d = (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0

    # Degrees of freedom
    if equal_var:
        dof = len(group1) + len(group2) - 2
    else:
        # Welch-Satterthwaite
        v1, v2 = group1.var(ddof=1), group2.var(ddof=1)
        n1, n2 = len(group1), len(group2)
        dof = (v1/n1 + v2/n2)**2 / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))

    # Confidence interval for difference
    se = np.sqrt(group1.var(ddof=1)/len(group1) + group2.var(ddof=1)/len(group2))
    t_crit = stats.t.ppf(1 - alpha/2, dof)
    diff = group1.mean() - group2.mean()
    ci = (diff - t_crit * se, diff + t_crit * se)

    return TestResult(
        test_name="independent_t",
        statistic=t_stat,
        p_value=p_value,
        effect_size=cohens_d,
        effect_size_name="Cohen's d",
        confidence_interval=ci,
        n_observations=len(group1) + len(group2),
        degrees_of_freedom=dof,
        significant=p_value < alpha,
        alpha=alpha,
        interpretation=f"M1={group1.mean():.3f}, M2={group2.mean():.3f}, d={cohens_d:.3f}",
        details={
            "mean1": float(group1.mean()),
            "mean2": float(group2.mean()),
            "std1": float(group1.std(ddof=1)),
            "std2": float(group2.std(ddof=1)),
            "equal_var": equal_var,
            "alternative": alternative,
        },
    )


@register_test("paired_t")
def paired_t_test(
    before: list[float],
    after: list[float],
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TestResult:
    """
    Paired samples t-test.

    Args:
        before: Values before treatment
        after: Values after treatment
        alpha: Significance level
        alternative: 'two-sided', 'less', or 'greater'
    """
    before = np.array(before)
    after = np.array(after)
    differences = after - before

    t_stat, p_value = stats.ttest_rel(before, after, alternative=alternative)

    # Effect size: Cohen's d for paired samples
    cohens_d = differences.mean() / differences.std(ddof=1) if differences.std(ddof=1) > 0 else 0

    # Degrees of freedom
    dof = len(differences) - 1

    # Confidence interval for mean difference
    se = differences.std(ddof=1) / np.sqrt(len(differences))
    t_crit = stats.t.ppf(1 - alpha/2, dof)
    ci = (differences.mean() - t_crit * se, differences.mean() + t_crit * se)

    return TestResult(
        test_name="paired_t",
        statistic=t_stat,
        p_value=p_value,
        effect_size=cohens_d,
        effect_size_name="Cohen's d (paired)",
        confidence_interval=ci,
        n_observations=len(differences),
        degrees_of_freedom=dof,
        significant=p_value < alpha,
        alpha=alpha,
        interpretation=f"Mean diff={differences.mean():.3f}, d={cohens_d:.3f}",
        details={
            "mean_before": float(before.mean()),
            "mean_after": float(after.mean()),
            "mean_difference": float(differences.mean()),
            "std_difference": float(differences.std(ddof=1)),
            "alternative": alternative,
        },
    )


@register_test("mann_whitney")
def mann_whitney_test(
    group1: list[float],
    group2: list[float],
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TestResult:
    """
    Mann-Whitney U test (non-parametric alternative to independent t-test).

    Args:
        group1: Values for group 1
        group2: Values for group 2
        alpha: Significance level
        alternative: 'two-sided', 'less', or 'greater'
    """
    group1 = np.array(group1)
    group2 = np.array(group2)

    u_stat, p_value = stats.mannwhitneyu(
        group1, group2,
        alternative=alternative,
    )

    # Effect size: rank-biserial correlation
    n1, n2 = len(group1), len(group2)
    r = 1 - (2 * u_stat) / (n1 * n2)

    return TestResult(
        test_name="mann_whitney",
        statistic=u_stat,
        p_value=p_value,
        effect_size=r,
        effect_size_name="Rank-biserial r",
        n_observations=n1 + n2,
        significant=p_value < alpha,
        alpha=alpha,
        interpretation=f"U={u_stat:.1f}, r={r:.3f}",
        details={
            "n1": n1,
            "n2": n2,
            "median1": float(np.median(group1)),
            "median2": float(np.median(group2)),
            "alternative": alternative,
        },
    )


@register_test("wilcoxon")
def wilcoxon_test(
    before: list[float],
    after: list[float],
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TestResult:
    """
    Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

    Args:
        before: Values before treatment
        after: Values after treatment
        alpha: Significance level
        alternative: 'two-sided', 'less', or 'greater'
    """
    before = np.array(before)
    after = np.array(after)

    stat, p_value = stats.wilcoxon(
        before, after,
        alternative=alternative,
    )

    # Effect size: matched-pairs rank-biserial correlation
    differences = after - before
    n = len(differences[differences != 0])
    r = 1 - (2 * stat) / (n * (n + 1) / 2) if n > 0 else 0

    return TestResult(
        test_name="wilcoxon",
        statistic=stat,
        p_value=p_value,
        effect_size=r,
        effect_size_name="Matched-pairs r",
        n_observations=len(before),
        significant=p_value < alpha,
        alpha=alpha,
        interpretation=f"W={stat:.1f}, r={r:.3f}",
        details={
            "n_pairs": len(before),
            "n_nonzero": n,
            "median_difference": float(np.median(differences)),
            "alternative": alternative,
        },
    )


# -----------------------------------------------------------------------------
# ANOVA and Related
# -----------------------------------------------------------------------------

@register_test("one_way_anova")
def one_way_anova(
    *groups: list[float],
    alpha: float = 0.05,
) -> TestResult:
    """
    One-way ANOVA for comparing multiple groups.

    Args:
        *groups: Variable number of groups to compare
        alpha: Significance level
    """
    groups = [np.array(g) for g in groups]

    f_stat, p_value = stats.f_oneway(*groups)

    # Effect size: eta-squared
    all_data = np.concatenate(groups)
    grand_mean = all_data.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
    ss_total = sum((x - grand_mean)**2 for x in all_data)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0

    # Degrees of freedom
    df_between = len(groups) - 1
    df_within = len(all_data) - len(groups)

    return TestResult(
        test_name="one_way_anova",
        statistic=f_stat,
        p_value=p_value,
        effect_size=eta_squared,
        effect_size_name="η² (eta-squared)",
        n_observations=len(all_data),
        degrees_of_freedom=df_between,
        significant=p_value < alpha,
        alpha=alpha,
        interpretation=f"F({df_between},{df_within})={f_stat:.2f}, η²={eta_squared:.3f}",
        details={
            "df_between": df_between,
            "df_within": df_within,
            "group_means": [float(g.mean()) for g in groups],
            "group_stds": [float(g.std(ddof=1)) for g in groups],
            "group_ns": [len(g) for g in groups],
        },
    )


@register_test("kruskal_wallis")
def kruskal_wallis_test(
    *groups: list[float],
    alpha: float = 0.05,
) -> TestResult:
    """
    Kruskal-Wallis H-test (non-parametric alternative to one-way ANOVA).

    Args:
        *groups: Variable number of groups to compare
        alpha: Significance level
    """
    groups = [np.array(g) for g in groups]

    h_stat, p_value = stats.kruskal(*groups)

    # Effect size: epsilon-squared
    n = sum(len(g) for g in groups)
    epsilon_squared = h_stat / (n - 1) if n > 1 else 0

    return TestResult(
        test_name="kruskal_wallis",
        statistic=h_stat,
        p_value=p_value,
        effect_size=epsilon_squared,
        effect_size_name="ε² (epsilon-squared)",
        n_observations=n,
        degrees_of_freedom=len(groups) - 1,
        significant=p_value < alpha,
        alpha=alpha,
        interpretation=f"H({len(groups)-1})={h_stat:.2f}, ε²={epsilon_squared:.3f}",
        details={
            "group_medians": [float(np.median(g)) for g in groups],
            "group_ns": [len(g) for g in groups],
        },
    )


# -----------------------------------------------------------------------------
# Correlation Tests
# -----------------------------------------------------------------------------

@register_test("pearson_correlation")
def pearson_correlation(
    x: list[float],
    y: list[float],
    alpha: float = 0.05,
) -> TestResult:
    """
    Pearson correlation coefficient.

    Args:
        x: First variable
        y: Second variable
        alpha: Significance level
    """
    x, y = np.array(x), np.array(y)

    r, p_value = stats.pearsonr(x, y)

    # Confidence interval using Fisher's z-transformation
    n = len(x)
    z = 0.5 * np.log((1 + r) / (1 - r)) if abs(r) < 1 else 0
    se_z = 1 / np.sqrt(n - 3) if n > 3 else 0
    z_crit = stats.norm.ppf(1 - alpha/2)
    z_lower, z_upper = z - z_crit * se_z, z + z_crit * se_z
    ci = (np.tanh(z_lower), np.tanh(z_upper))

    return TestResult(
        test_name="pearson_correlation",
        statistic=r,
        p_value=p_value,
        effect_size=r,
        effect_size_name="Pearson r",
        confidence_interval=ci,
        n_observations=n,
        degrees_of_freedom=n - 2,
        significant=p_value < alpha,
        alpha=alpha,
        interpretation=f"r={r:.3f}, r²={r**2:.3f}",
        details={
            "r_squared": r**2,
        },
    )


@register_test("spearman_correlation")
def spearman_correlation(
    x: list[float],
    y: list[float],
    alpha: float = 0.05,
) -> TestResult:
    """
    Spearman rank correlation coefficient.

    Args:
        x: First variable
        y: Second variable
        alpha: Significance level
    """
    x, y = np.array(x), np.array(y)

    rho, p_value = stats.spearmanr(x, y)

    return TestResult(
        test_name="spearman_correlation",
        statistic=rho,
        p_value=p_value,
        effect_size=rho,
        effect_size_name="Spearman ρ",
        n_observations=len(x),
        significant=p_value < alpha,
        alpha=alpha,
        interpretation=f"ρ={rho:.3f}",
        details={},
    )


# -----------------------------------------------------------------------------
# Bootstrap-based Tests
# -----------------------------------------------------------------------------

@register_test("bootstrap_proportion_diff")
def bootstrap_proportion_diff(
    successes1: int,
    n1: int,
    successes2: int,
    n2: int,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> TestResult:
    """
    Bootstrap confidence interval for difference in proportions.

    Args:
        successes1: Number of successes in group 1
        n1: Total observations in group 1
        successes2: Number of successes in group 2
        n2: Total observations in group 2
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level
        seed: Random seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    # Observed proportions
    p1 = successes1 / n1 if n1 > 0 else 0
    p2 = successes2 / n2 if n2 > 0 else 0
    observed_diff = p1 - p2

    # Bootstrap
    diffs = []
    for _ in range(n_bootstrap):
        # Resample from binomial
        boot_successes1 = rng.binomial(n1, p1)
        boot_successes2 = rng.binomial(n2, p2)
        boot_p1 = boot_successes1 / n1
        boot_p2 = boot_successes2 / n2
        diffs.append(boot_p1 - boot_p2)

    diffs = np.array(diffs)

    # Percentile CI
    ci = (np.percentile(diffs, 100 * alpha/2), np.percentile(diffs, 100 * (1 - alpha/2)))

    # P-value (proportion of bootstrap samples with opposite sign or zero)
    if observed_diff > 0:
        p_value = np.mean(diffs <= 0) * 2
    elif observed_diff < 0:
        p_value = np.mean(diffs >= 0) * 2
    else:
        p_value = 1.0
    p_value = min(p_value, 1.0)

    return TestResult(
        test_name="bootstrap_proportion_diff",
        statistic=observed_diff,
        p_value=p_value,
        effect_size=observed_diff,
        effect_size_name="Difference in proportions",
        confidence_interval=ci,
        n_observations=n1 + n2,
        significant=not (ci[0] <= 0 <= ci[1]),
        alpha=alpha,
        interpretation=f"Δp={observed_diff:.3f}, 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]",
        details={
            "p1": p1,
            "p2": p2,
            "n_bootstrap": n_bootstrap,
            "seed": seed,
            "bootstrap_mean": float(diffs.mean()),
            "bootstrap_std": float(diffs.std()),
        },
    )


@register_test("bootstrap_mean_diff")
def bootstrap_mean_diff(
    group1: list[float],
    group2: list[float],
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> TestResult:
    """
    Bootstrap confidence interval for difference in means.

    Args:
        group1: Values for group 1
        group2: Values for group 2
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level
        seed: Random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    group1, group2 = np.array(group1), np.array(group2)

    observed_diff = group1.mean() - group2.mean()

    # Bootstrap
    diffs = []
    for _ in range(n_bootstrap):
        boot1 = rng.choice(group1, size=len(group1), replace=True)
        boot2 = rng.choice(group2, size=len(group2), replace=True)
        diffs.append(boot1.mean() - boot2.mean())

    diffs = np.array(diffs)

    # Percentile CI
    ci = (np.percentile(diffs, 100 * alpha/2), np.percentile(diffs, 100 * (1 - alpha/2)))

    # P-value
    if observed_diff > 0:
        p_value = np.mean(diffs <= 0) * 2
    elif observed_diff < 0:
        p_value = np.mean(diffs >= 0) * 2
    else:
        p_value = 1.0
    p_value = min(p_value, 1.0)

    return TestResult(
        test_name="bootstrap_mean_diff",
        statistic=observed_diff,
        p_value=p_value,
        effect_size=observed_diff,
        effect_size_name="Difference in means",
        confidence_interval=ci,
        n_observations=len(group1) + len(group2),
        significant=not (ci[0] <= 0 <= ci[1]),
        alpha=alpha,
        interpretation=f"ΔM={observed_diff:.3f}, 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]",
        details={
            "mean1": float(group1.mean()),
            "mean2": float(group2.mean()),
            "n_bootstrap": n_bootstrap,
            "seed": seed,
        },
    )


# -----------------------------------------------------------------------------
# Reliability Tests
# -----------------------------------------------------------------------------

@register_test("cohens_kappa")
def cohens_kappa(
    rater1: list[Any],
    rater2: list[Any],
    alpha: float = 0.05,
) -> TestResult:
    """
    Cohen's kappa for inter-rater reliability.

    Args:
        rater1: Ratings from rater 1
        rater2: Ratings from rater 2
        alpha: Significance level
    """
    from sklearn.metrics import cohen_kappa_score

    kappa = cohen_kappa_score(rater1, rater2)

    # Agreement
    agreement = sum(r1 == r2 for r1, r2 in zip(rater1, rater2)) / len(rater1)

    return TestResult(
        test_name="cohens_kappa",
        statistic=kappa,
        p_value=0.0,  # Not typically used for kappa
        effect_size=kappa,
        effect_size_name="Cohen's κ",
        n_observations=len(rater1),
        significant=kappa >= 0.70,  # Convention: κ ≥ 0.70 is acceptable
        alpha=alpha,
        interpretation=f"κ={kappa:.3f}, agreement={agreement:.1%}",
        details={
            "agreement": agreement,
            "interpretation": (
                "Poor" if kappa < 0.20 else
                "Fair" if kappa < 0.40 else
                "Moderate" if kappa < 0.60 else
                "Good" if kappa < 0.80 else
                "Very Good"
            ),
        },
    )


@register_test("icc")
def intraclass_correlation(
    ratings: list[list[float]],
    icc_type: str = "ICC(2,1)",
    alpha: float = 0.05,
) -> TestResult:
    """
    Intraclass Correlation Coefficient.

    Args:
        ratings: Matrix of ratings [subjects × raters]
        icc_type: Type of ICC (ICC(1,1), ICC(2,1), ICC(3,1), etc.)
        alpha: Significance level
    """
    import pingouin as pg
    import pandas as pd

    # Convert to long format for pingouin
    ratings = np.array(ratings)
    n_subjects, n_raters = ratings.shape

    data = []
    for i in range(n_subjects):
        for j in range(n_raters):
            data.append({"subject": i, "rater": j, "rating": ratings[i, j]})

    df = pd.DataFrame(data)

    # Compute ICC
    icc_result = pg.intraclass_corr(
        data=df,
        targets="subject",
        raters="rater",
        ratings="rating",
    )

    # Find the requested ICC type
    type_map = {
        "ICC(1,1)": "ICC1",
        "ICC(2,1)": "ICC2",
        "ICC(3,1)": "ICC3",
        "ICC(1,k)": "ICC1k",
        "ICC(2,k)": "ICC2k",
        "ICC(3,k)": "ICC3k",
    }
    icc_name = type_map.get(icc_type, "ICC2")
    row = icc_result[icc_result["Type"] == icc_name].iloc[0]

    icc_value = row["ICC"]
    ci = (row["CI95%"][0], row["CI95%"][1])
    p_value = row["pval"]

    return TestResult(
        test_name="icc",
        statistic=icc_value,
        p_value=p_value,
        effect_size=icc_value,
        effect_size_name=icc_type,
        confidence_interval=ci,
        n_observations=n_subjects,
        significant=p_value < alpha,
        alpha=alpha,
        interpretation=f"{icc_type}={icc_value:.3f}, 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]",
        details={
            "icc_type": icc_type,
            "n_subjects": n_subjects,
            "n_raters": n_raters,
            "f_value": row["F"],
            "df1": row["df1"],
            "df2": row["df2"],
        },
    )


# =============================================================================
# Power Analysis
# =============================================================================

def compute_power(
    test_name: str,
    effect_size: float,
    n: int,
    alpha: float = 0.05,
    ratio: float = 1.0,
    alternative: str = "two-sided",
    k: int = 2,
) -> dict[str, Any]:
    """
    Compute statistical power for a given test, effect size, and sample size.

    Also computes the required sample size for 80% and 90% power targets.

    Args:
        test_name: Type of test. One of:
            "two_proportion_z", "chi_square" — uses Cohen's h
            "independent_t", "paired_t" — uses Cohen's d
            "one_way_anova" — uses Cohen's f
        effect_size: Expected effect size (Cohen's h, d, or f depending on test)
        n: Sample size per group
        alpha: Significance level
        ratio: Ratio of group sizes (n2/n1). Only used for two-sample tests.
        alternative: "two-sided" or "one-sided"
        k: Number of groups (only used for one_way_anova, default 2)

    Returns:
        Dictionary with keys: power, n_for_80, n_for_90, test_name, effect_size, n, alpha
    """
    tail_factor = 1 if alternative == "one-sided" else 2
    z_alpha = stats.norm.ppf(1 - alpha / tail_factor)

    if test_name in ("two_proportion_z", "chi_square"):
        # Power for two-proportion z-test using Cohen's h
        # SE under H1 ≈ sqrt((1/n1 + 1/n2)) for arcsine-transformed proportions
        n1 = n
        n2 = int(n * ratio)
        se = np.sqrt(1.0 / n1 + 1.0 / n2)
        noncentrality = abs(effect_size) / se
        power = 1 - stats.norm.cdf(z_alpha - noncentrality)

        def _required_n(target_power: float) -> int:
            z_beta = stats.norm.ppf(target_power)
            # n per group: ((z_alpha + z_beta) / effect_size)^2 * (1 + 1/ratio)
            n_per = ((z_alpha + z_beta) / abs(effect_size)) ** 2 * (1 + 1.0 / ratio)
            return max(int(np.ceil(n_per)), 2)

    elif test_name in ("independent_t",):
        n1 = n
        n2 = int(n * ratio)
        se = np.sqrt(1.0 / n1 + 1.0 / n2)
        noncentrality = abs(effect_size) / se
        df = n1 + n2 - 2
        if df < 1:
            return {"power": 0.0, "n_for_80": None, "n_for_90": None,
                    "test_name": test_name, "effect_size": effect_size,
                    "n": n, "alpha": alpha}
        crit = stats.t.ppf(1 - alpha / tail_factor, df)
        power = 1 - stats.nct.cdf(crit, df, noncentrality)

        def _required_n(target_power: float) -> int:
            z_beta = stats.norm.ppf(target_power)
            n_per = ((z_alpha + z_beta) / abs(effect_size)) ** 2 * (1 + 1.0 / ratio)
            return max(int(np.ceil(n_per)), 2)

    elif test_name in ("paired_t",):
        noncentrality = abs(effect_size) * np.sqrt(n)
        df = n - 1
        if df < 1:
            return {"power": 0.0, "n_for_80": None, "n_for_90": None,
                    "test_name": test_name, "effect_size": effect_size,
                    "n": n, "alpha": alpha}
        crit = stats.t.ppf(1 - alpha / tail_factor, df)
        power = 1 - stats.nct.cdf(crit, df, noncentrality)

        def _required_n(target_power: float) -> int:
            z_beta = stats.norm.ppf(target_power)
            return max(int(np.ceil(((z_alpha + z_beta) / abs(effect_size)) ** 2)), 2)

    elif test_name in ("one_way_anova",):
        # Cohen's f, approximate power via noncentral F
        # noncentrality parameter lambda = n * k * f^2
        lam = n * k * effect_size ** 2
        df1 = k - 1
        df2 = k * (n - 1)
        if df2 < 1:
            return {"power": 0.0, "n_for_80": None, "n_for_90": None,
                    "test_name": test_name, "effect_size": effect_size,
                    "n": n, "alpha": alpha}
        crit = stats.f.ppf(1 - alpha, df1, df2)
        power = 1 - stats.ncf.cdf(crit, df1, df2, lam)

        def _required_n(target_power: float) -> int:
            z_beta = stats.norm.ppf(target_power)
            return max(int(np.ceil(((z_alpha + z_beta) / abs(effect_size)) ** 2 / k)), 2)

    else:
        # Fallback: normal approximation for unknown test types
        noncentrality = abs(effect_size) * np.sqrt(n)
        power = 1 - stats.norm.cdf(z_alpha - noncentrality)

        def _required_n(target_power: float) -> int:
            z_beta = stats.norm.ppf(target_power)
            return max(int(np.ceil(((z_alpha + z_beta) / abs(effect_size)) ** 2)), 2)

    return {
        "power": float(power),
        "n_for_80": _required_n(0.80),
        "n_for_90": _required_n(0.90),
        "test_name": test_name,
        "effect_size": effect_size,
        "n": n,
        "alpha": alpha,
    }


# =============================================================================
# Convenience Functions
# =============================================================================

def run_analysis(
    results: dict[str, Any],
    tests: list[str],
    alpha: float = 0.05,
) -> list[TestResult]:
    """
    Run multiple statistical tests on results data.

    Args:
        results: Dictionary containing analysis data
        tests: List of test names to run
        alpha: Significance level

    Returns:
        List of TestResult objects
    """
    outputs = []

    for test_name in tests:
        test_fn = get_test(test_name)

        # Each test expects specific data - this is a dispatcher
        # Real implementation would need to map results to test args
        try:
            if test_name in ["two_proportion_z", "bootstrap_proportion_diff"]:
                result = test_fn(
                    successes1=results.get("successes1", 0),
                    n1=results.get("n1", 1),
                    successes2=results.get("successes2", 0),
                    n2=results.get("n2", 1),
                    alpha=alpha,
                )
            elif test_name in ["chi_square", "fisher_exact"]:
                result = test_fn(
                    observed=results.get("contingency_table", [[0, 0], [0, 0]]),
                    alpha=alpha,
                )
            elif test_name in ["independent_t", "mann_whitney", "bootstrap_mean_diff"]:
                result = test_fn(
                    group1=results.get("group1", []),
                    group2=results.get("group2", []),
                    alpha=alpha,
                )
            else:
                # Generic call - test must handle its own args
                result = test_fn(alpha=alpha, **results)

            outputs.append(result)
        except Exception as e:
            # Create error result
            outputs.append(TestResult(
                test_name=test_name,
                statistic=0.0,
                p_value=1.0,
                interpretation=f"Error: {e}",
                alpha=alpha,
            ))

    return outputs
