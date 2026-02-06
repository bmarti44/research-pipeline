"""
Adaptive stopping rules for the research pipeline.

Implements sequential analysis with preregistered stopping criteria.
Allows early stopping for futility or efficacy while controlling Type I error.
"""

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Literal
import numpy as np
from scipy import stats


@dataclass
class StoppingRule:
    """A preregistered stopping rule."""
    name: str
    type: Literal["efficacy", "futility", "harm"]
    check_at_n: list[int]  # Sample sizes at which to check
    threshold: float  # p-value or effect size threshold
    alpha_spent: float = 0.0  # Alpha spending for this interim


@dataclass
class InterimAnalysis:
    """Result of an interim analysis."""
    n_completed: int
    n_total: int
    check_point: int
    effect_size: Optional[float]
    p_value: Optional[float]
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    decision: Literal["continue", "stop_efficacy", "stop_futility", "stop_harm"]
    reason: str
    alpha_spent_cumulative: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive stopping."""
    enabled: bool = False
    interim_looks: list[float] = field(default_factory=lambda: [0.5, 0.75])  # Proportions of total N
    alpha_total: float = 0.05
    spending_function: str = "obrien-fleming"  # or "pocock", "linear"
    efficacy_threshold: float = 0.025  # One-sided
    futility_threshold: float = 0.20  # Conditional power below this = futile
    min_effect_size: float = 0.10  # Minimum effect of interest


def load_adaptive_config(config: dict) -> AdaptiveConfig:
    """Load adaptive stopping configuration."""
    adaptive = config.get("adaptive", {})

    if not adaptive.get("enabled", False):
        return AdaptiveConfig(enabled=False)

    return AdaptiveConfig(
        enabled=True,
        interim_looks=adaptive.get("interim_looks", [0.5, 0.75]),
        alpha_total=adaptive.get("alpha", 0.05),
        spending_function=adaptive.get("spending_function", "obrien-fleming"),
        efficacy_threshold=adaptive.get("efficacy_threshold", 0.025),
        futility_threshold=adaptive.get("futility_threshold", 0.20),
        min_effect_size=adaptive.get("min_effect_size", 0.10),
    )


def compute_alpha_spending(
    information_fraction: float,
    alpha_total: float,
    spending_function: str,
) -> float:
    """
    Compute cumulative alpha spent at given information fraction.

    Implements O'Brien-Fleming, Pocock, and linear spending functions.
    """
    t = information_fraction

    if spending_function == "obrien-fleming":
        # O'Brien-Fleming: very conservative early, aggressive late
        # α*(t) = 2[1 - Φ(z_{α/2} / √t)]
        z = stats.norm.ppf(1 - alpha_total / 2)
        return 2 * (1 - stats.norm.cdf(z / math.sqrt(t)))

    elif spending_function == "pocock":
        # Pocock: uniform spending
        # α*(t) = α * log(1 + (e-1)*t)
        return alpha_total * math.log(1 + (math.e - 1) * t)

    elif spending_function == "linear":
        # Linear: simple proportional spending
        return alpha_total * t

    else:
        raise ValueError(f"Unknown spending function: {spending_function}")


def compute_stopping_boundaries(
    n_looks: int,
    information_fractions: list[float],
    alpha_total: float,
    spending_function: str,
) -> list[float]:
    """
    Compute stopping boundaries (critical z-values) for each interim look.
    """
    boundaries = []
    alpha_spent_prev = 0

    for i, t in enumerate(information_fractions):
        alpha_spent = compute_alpha_spending(t, alpha_total, spending_function)
        alpha_increment = alpha_spent - alpha_spent_prev

        # Convert to critical z-value
        z_crit = stats.norm.ppf(1 - alpha_increment / 2)
        boundaries.append(z_crit)

        alpha_spent_prev = alpha_spent

    return boundaries


def compute_conditional_power(
    effect_observed: float,
    effect_target: float,
    se_current: float,
    n_current: int,
    n_final: int,
    alpha: float = 0.05,
) -> float:
    """
    Compute conditional power: probability of rejecting H0 at final analysis
    given current data.
    """
    if n_current >= n_final:
        return 1.0 if effect_observed / se_current > stats.norm.ppf(1 - alpha / 2) else 0.0

    # Variance at final
    var_final = (se_current ** 2) * (n_current / n_final)
    se_final = math.sqrt(var_final)

    # Critical value at final
    z_crit = stats.norm.ppf(1 - alpha / 2)

    # Conditional power assuming effect is effect_target
    # P(Z_final > z_crit | current data)
    # Z_final ~ N(effect_target/se_final, 1) approximately

    # Project current effect to final
    effect_projected = effect_observed  # Assumes effect stays same

    z_projected = effect_projected / se_final
    conditional_power = 1 - stats.norm.cdf(z_crit - z_projected)

    return conditional_power


def perform_interim_analysis(
    study_path: Path,
    config: dict,
    check_point: int,
) -> InterimAnalysis:
    """
    Perform interim analysis at specified checkpoint.

    Loads current scores, computes statistics, and makes stopping decision.
    """
    adaptive_config = load_adaptive_config(config)

    if not adaptive_config.enabled:
        return InterimAnalysis(
            n_completed=0,
            n_total=0,
            check_point=check_point,
            effect_size=None,
            p_value=None,
            ci_lower=None,
            ci_upper=None,
            decision="continue",
            reason="Adaptive stopping not enabled",
            alpha_spent_cumulative=0,
        )

    # Load trials
    trials_path = study_path / "stages" / "2_generate" / "trials.json"
    with open(trials_path) as f:
        trials = json.load(f)
    n_total = len(trials)

    # Load completed scores
    scores_path = study_path / "stages" / "4_evaluate" / "scores"
    if not scores_path.exists():
        return InterimAnalysis(
            n_completed=0,
            n_total=n_total,
            check_point=check_point,
            effect_size=None,
            p_value=None,
            ci_lower=None,
            ci_upper=None,
            decision="continue",
            reason="No scores available yet",
            alpha_spent_cumulative=0,
        )

    score_files = list(scores_path.glob("trial_*.json"))
    n_completed = len(score_files)

    if n_completed < check_point:
        return InterimAnalysis(
            n_completed=n_completed,
            n_total=n_total,
            check_point=check_point,
            effect_size=None,
            p_value=None,
            ci_lower=None,
            ci_upper=None,
            decision="continue",
            reason=f"Not yet at checkpoint: {n_completed}/{check_point}",
            alpha_spent_cumulative=0,
        )

    # Load scores and compute statistics
    scores_by_condition = {}
    for sf in score_files:
        with open(sf) as f:
            score = json.load(f)

        # Get condition from trial
        trial_id = score["trial_id"]
        trial = next((t for t in trials if t["trial_id"] == trial_id), None)
        if trial:
            condition = trial.get("condition", "unknown")
            if condition not in scores_by_condition:
                scores_by_condition[condition] = []

            # Use strict mode as primary
            correct = score.get("modes", {}).get("strict", {}).get("correct", False)
            scores_by_condition[condition].append(1 if correct else 0)

    # Need at least 2 conditions to compare
    conditions = list(scores_by_condition.keys())
    if len(conditions) < 2:
        return InterimAnalysis(
            n_completed=n_completed,
            n_total=n_total,
            check_point=check_point,
            effect_size=None,
            p_value=None,
            ci_lower=None,
            ci_upper=None,
            decision="continue",
            reason="Need at least 2 conditions for comparison",
            alpha_spent_cumulative=0,
        )

    # Compare first two conditions (e.g., NL vs JSON)
    scores_a = np.array(scores_by_condition[conditions[0]])
    scores_b = np.array(scores_by_condition[conditions[1]])

    # Compute effect size (difference in proportions)
    prop_a = np.mean(scores_a) if len(scores_a) > 0 else 0
    prop_b = np.mean(scores_b) if len(scores_b) > 0 else 0
    effect_size = prop_a - prop_b

    # Standard error
    n_a, n_b = len(scores_a), len(scores_b)
    if n_a > 0 and n_b > 0:
        pooled_prop = (np.sum(scores_a) + np.sum(scores_b)) / (n_a + n_b)
        se = math.sqrt(pooled_prop * (1 - pooled_prop) * (1/n_a + 1/n_b)) if pooled_prop > 0 else 0.01
    else:
        se = 0.01

    # Z-test
    z_stat = effect_size / se if se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # Confidence interval
    ci_lower = effect_size - 1.96 * se
    ci_upper = effect_size + 1.96 * se

    # Information fraction
    info_fraction = n_completed / n_total

    # Alpha spent so far
    alpha_spent = compute_alpha_spending(
        info_fraction,
        adaptive_config.alpha_total,
        adaptive_config.spending_function,
    )

    # Get boundary for this look
    boundaries = compute_stopping_boundaries(
        len(adaptive_config.interim_looks),
        adaptive_config.interim_looks + [1.0],
        adaptive_config.alpha_total,
        adaptive_config.spending_function,
    )

    # Find which interim we're at
    look_idx = 0
    for i, fraction in enumerate(adaptive_config.interim_looks):
        if info_fraction >= fraction:
            look_idx = i

    z_boundary = boundaries[look_idx] if look_idx < len(boundaries) else 1.96

    # Decision logic
    decision = "continue"
    reason = ""

    # Efficacy stopping
    if abs(z_stat) > z_boundary:
        decision = "stop_efficacy"
        reason = f"Effect crossed efficacy boundary: |z|={abs(z_stat):.2f} > {z_boundary:.2f}"

    # Futility stopping
    elif info_fraction >= adaptive_config.interim_looks[0]:  # Only after first look
        cond_power = compute_conditional_power(
            effect_observed=effect_size,
            effect_target=adaptive_config.min_effect_size,
            se_current=se,
            n_current=n_completed,
            n_final=n_total,
            alpha=adaptive_config.alpha_total,
        )

        if cond_power < adaptive_config.futility_threshold:
            decision = "stop_futility"
            reason = f"Conditional power too low: {cond_power:.2%} < {adaptive_config.futility_threshold:.2%}"
        else:
            reason = f"Continuing: CP={cond_power:.2%}, |z|={abs(z_stat):.2f}"

    else:
        reason = f"Continuing: info fraction {info_fraction:.2%}, |z|={abs(z_stat):.2f}"

    return InterimAnalysis(
        n_completed=n_completed,
        n_total=n_total,
        check_point=check_point,
        effect_size=effect_size,
        p_value=p_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        decision=decision,
        reason=reason,
        alpha_spent_cumulative=alpha_spent,
    )


def should_stop(study_path: Path, config: dict) -> tuple[bool, str, Optional[InterimAnalysis]]:
    """
    Check if study should stop early based on adaptive rules.

    Returns (should_stop, reason, interim_analysis).
    """
    adaptive_config = load_adaptive_config(config)

    if not adaptive_config.enabled:
        return False, "Adaptive stopping not enabled", None

    # Load trial count
    trials_path = study_path / "stages" / "2_generate" / "trials.json"
    if not trials_path.exists():
        return False, "No trials generated", None

    with open(trials_path) as f:
        trials = json.load(f)
    n_total = len(trials)

    # Determine current N
    scores_path = study_path / "stages" / "4_evaluate" / "scores"
    if not scores_path.exists():
        return False, "No scores yet", None

    n_completed = len(list(scores_path.glob("trial_*.json")))

    # Check if we're at an interim look
    for fraction in adaptive_config.interim_looks:
        check_point = int(n_total * fraction)
        if n_completed >= check_point:
            interim = perform_interim_analysis(study_path, config, check_point)

            if interim.decision in ["stop_efficacy", "stop_futility", "stop_harm"]:
                return True, interim.reason, interim

    return False, "Continue", None


def save_interim_analysis(study_path: Path, analysis: InterimAnalysis) -> None:
    """Save interim analysis result."""
    interim_path = study_path / "stages" / "interim_analyses"
    interim_path.mkdir(parents=True, exist_ok=True)

    filename = f"interim_{analysis.check_point:05d}.json"
    with open(interim_path / filename, "w") as f:
        json.dump({
            "n_completed": analysis.n_completed,
            "n_total": analysis.n_total,
            "check_point": analysis.check_point,
            "effect_size": analysis.effect_size,
            "p_value": analysis.p_value,
            "ci_lower": analysis.ci_lower,
            "ci_upper": analysis.ci_upper,
            "decision": analysis.decision,
            "reason": analysis.reason,
            "alpha_spent_cumulative": analysis.alpha_spent_cumulative,
            "timestamp": analysis.timestamp,
        }, f, indent=2)


def get_adjusted_alpha(study_path: Path, config: dict) -> float:
    """
    Get alpha adjusted for interim analyses already performed.

    If no interim analyses were done, returns the full alpha.
    """
    adaptive_config = load_adaptive_config(config)

    if not adaptive_config.enabled:
        return adaptive_config.alpha_total

    # Load interim analyses
    interim_path = study_path / "stages" / "interim_analyses"
    if not interim_path.exists():
        return adaptive_config.alpha_total

    interim_files = list(interim_path.glob("interim_*.json"))
    if not interim_files:
        return adaptive_config.alpha_total

    # Get final alpha spent
    latest = sorted(interim_files)[-1]
    with open(latest) as f:
        analysis = json.load(f)

    return adaptive_config.alpha_total - analysis.get("alpha_spent_cumulative", 0)
