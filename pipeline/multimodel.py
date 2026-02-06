"""
Multi-model support for the research pipeline.

Treats model as an independent variable, enabling cross-model comparisons.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class ModelConfig:
    """Configuration for a model in the study."""
    alias: str  # User-friendly name (e.g., "claude-sonnet")
    model_id: str  # Actual API model ID (e.g., "claude-sonnet-4-20250514")
    provider: str  # anthropic, openai, google
    temperature: float = 0.0
    max_tokens: int = 4096
    family: Optional[str] = None  # claude, gpt, gemini

    def to_dict(self) -> dict:
        return {
            "alias": self.alias,
            "model_id": self.model_id,
            "provider": self.provider,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "family": self.family,
        }


@dataclass
class ModelAsIV:
    """Model treated as an independent variable."""
    variable_name: str = "model"
    levels: list[ModelConfig] = field(default_factory=list)
    is_between_subjects: bool = True  # Usually between-subjects for models
    cross_with_conditions: bool = True  # Full factorial with other IVs


def load_models_config(config: dict) -> list[ModelConfig]:
    """Load model configurations from study config."""
    models = []

    for m in config.get("models", []):
        models.append(ModelConfig(
            alias=m.get("alias", m.get("model_id", "unknown")),
            model_id=m.get("model_id", ""),
            provider=m.get("provider", ""),
            temperature=m.get("temperature", 0.0),
            max_tokens=m.get("max_tokens", 4096),
            family=m.get("family"),
        ))

    return models


def create_model_iv(config: dict) -> Optional[ModelAsIV]:
    """Create model as IV from config if multiple models specified."""
    models = load_models_config(config)

    if len(models) <= 1:
        return None

    return ModelAsIV(
        variable_name="model",
        levels=models,
        is_between_subjects=config.get("design", {}).get("model_between_subjects", True),
        cross_with_conditions=config.get("design", {}).get("model_cross_conditions", True),
    )


def generate_model_trials(
    base_trials: list[dict],
    model_iv: ModelAsIV,
) -> list[dict]:
    """
    Expand trials to include model as a condition.

    If model is crossed with other conditions, creates full factorial.
    """
    if not model_iv.cross_with_conditions:
        # Each trial gets randomly assigned to a model
        # (handled during execution, not here)
        return base_trials

    # Full factorial: each trial × each model
    expanded_trials = []
    trial_id = 0

    for trial in base_trials:
        for model in model_iv.levels:
            new_trial = trial.copy()
            new_trial["trial_id"] = f"trial_{trial_id:04d}"
            new_trial["model"] = model.alias
            new_trial["model_config"] = model.to_dict()
            trial_id += 1
            expanded_trials.append(new_trial)

    return expanded_trials


def analyze_model_effects(
    scores: list[dict],
    trials: list[dict],
    primary_dv: str = "correct",
) -> dict:
    """
    Analyze effects with model as a factor.

    Returns aggregates by model and model × condition interactions.
    """
    import numpy as np
    from scipy import stats

    # Map trial_id to model
    trial_to_model = {}
    for trial in trials:
        trial_to_model[trial["trial_id"]] = trial.get("model", "unknown")

    # Organize scores by model and condition
    by_model = {}
    by_model_condition = {}

    for score in scores:
        trial_id = score.get("trial_id")
        model = trial_to_model.get(trial_id, "unknown")

        # Get primary DV value
        value = score.get("modes", {}).get("strict", {}).get(primary_dv, 0)
        if isinstance(value, bool):
            value = 1 if value else 0

        # Get condition from trial
        trial = next((t for t in trials if t["trial_id"] == trial_id), {})
        condition = trial.get("condition", "unknown")

        # Aggregate
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(value)

        key = (model, condition)
        if key not in by_model_condition:
            by_model_condition[key] = []
        by_model_condition[key].append(value)

    # Compute stats by model
    model_stats = {}
    for model, values in by_model.items():
        arr = np.array(values)
        model_stats[model] = {
            "n": len(arr),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0,
            "ci_lower": float(np.mean(arr) - 1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else float(np.mean(arr)),
            "ci_upper": float(np.mean(arr) + 1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else float(np.mean(arr)),
        }

    # Compute stats by model × condition
    interaction_stats = {}
    for (model, condition), values in by_model_condition.items():
        arr = np.array(values)
        interaction_stats[f"{model}_{condition}"] = {
            "model": model,
            "condition": condition,
            "n": len(arr),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0,
        }

    # Test for model main effect (chi-square or ANOVA)
    models = list(by_model.keys())
    if len(models) >= 2:
        # Chi-square test for proportions
        observed = [sum(by_model[m]) for m in models]
        totals = [len(by_model[m]) for m in models]

        # Expected under null (equal proportions)
        overall_rate = sum(observed) / sum(totals)
        expected = [overall_rate * n for n in totals]

        # Avoid division by zero
        if all(e > 0 for e in expected):
            chi2 = sum((o - e) ** 2 / e for o, e in zip(observed, expected))
            df = len(models) - 1
            p_value = 1 - stats.chi2.cdf(chi2, df)
        else:
            chi2 = 0
            p_value = 1.0

        model_test = {
            "test": "chi_square",
            "statistic": chi2,
            "df": df,
            "p_value": p_value,
            "significant": p_value < 0.05,
        }
    else:
        model_test = None

    return {
        "by_model": model_stats,
        "by_model_condition": interaction_stats,
        "model_main_effect": model_test,
    }


def verify_model_versions(study_path: Path) -> tuple[bool, list[str]]:
    """
    Verify that model versions used in execution match configured versions.
    """
    issues = []

    # Load resolved config
    config_path = study_path / "stages" / "1_configure" / "config_resolved.yaml"
    if not config_path.exists():
        return False, ["No resolved config found"]

    with open(config_path) as f:
        config = yaml.safe_load(f)

    expected_models = {m["alias"]: m["model_id"] for m in config.get("models", [])}

    # Check execution log
    log_path = study_path / "stages" / "3_execute" / "execution_log.json"
    if not log_path.exists():
        return False, ["No execution log found"]

    with open(log_path) as f:
        log = json.load(f)

    # Check responses
    responses_path = study_path / "stages" / "3_execute" / "responses"
    if responses_path.exists():
        for rf in list(responses_path.glob("trial_*.json"))[:10]:  # Sample
            with open(rf) as f:
                resp = json.load(f)

            used_model = resp.get("model", "")

            # Check if it matches expected
            for alias, expected_id in expected_models.items():
                if alias in used_model or expected_id in used_model:
                    # Model ID should match exactly, not just contain alias
                    if expected_id not in used_model and used_model != expected_id:
                        issues.append(
                            f"Model mismatch in {rf.name}: expected {expected_id}, got {used_model}"
                        )
                    break

    return len(issues) == 0, issues
