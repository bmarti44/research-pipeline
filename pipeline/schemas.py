"""
Schema validation for the research pipeline.

Provides validation for:
- Study configurations
- Task definitions
- Trial data
- Response data
- Evaluation results
- Analysis outputs
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Union
from pathlib import Path
import re


# =============================================================================
# Validation Results
# =============================================================================

@dataclass
class ValidationError:
    """A single validation error."""
    path: str  # JSON path to error (e.g., "conditions[0].name")
    message: str
    severity: str = "error"  # error, warning, info

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.path}: {self.message}"


@dataclass
class ValidationResult:
    """Result of validating a schema."""
    valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)

    def add_error(self, path: str, message: str) -> None:
        """Add an error."""
        self.errors.append(ValidationError(path, message, "error"))
        self.valid = False

    def add_warning(self, path: str, message: str) -> None:
        """Add a warning."""
        self.warnings.append(ValidationError(path, message, "warning"))

    def merge(self, other: "ValidationResult", prefix: str = "") -> None:
        """Merge another validation result into this one."""
        for error in other.errors:
            path = f"{prefix}.{error.path}" if prefix else error.path
            self.errors.append(ValidationError(path, error.message, error.severity))
        for warning in other.warnings:
            path = f"{prefix}.{warning.path}" if prefix else warning.path
            self.warnings.append(ValidationError(path, warning.message, warning.severity))
        if not other.valid:
            self.valid = False

    def __str__(self) -> str:
        lines = []
        if self.valid:
            lines.append("Validation PASSED")
        else:
            lines.append("Validation FAILED")

        if self.errors:
            lines.append(f"\nErrors ({len(self.errors)}):")
            for error in self.errors:
                lines.append(f"  {error}")

        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                lines.append(f"  {warning}")

        return "\n".join(lines)


# =============================================================================
# Schema Definitions
# =============================================================================

# Valid identifier pattern
IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

# Valid evaluation modes
VALID_EVAL_MODES = {"strict", "intent", "functional", "semantic"}

# Valid design types
VALID_DESIGN_TYPES = {"between", "within", "mixed"}

# Valid providers
VALID_PROVIDERS = {"anthropic", "openai", "google"}


def validate_identifier(value: str, name: str = "identifier") -> ValidationResult:
    """Validate an identifier (lowercase, alphanumeric, underscores)."""
    result = ValidationResult(valid=True)

    if not isinstance(value, str):
        result.add_error(name, f"Must be a string, got {type(value).__name__}")
        return result

    if not IDENTIFIER_PATTERN.match(value):
        result.add_error(
            name,
            f"Must start with lowercase letter and contain only lowercase letters, numbers, underscores. Got: '{value}'"
        )

    return result


# =============================================================================
# Config Schema Validation
# =============================================================================

def validate_study_config(config: dict) -> ValidationResult:
    """
    Validate a study configuration dictionary.

    Args:
        config: Study configuration dict

    Returns:
        ValidationResult with errors and warnings
    """
    result = ValidationResult(valid=True)

    # Required top-level sections
    required_sections = ["study", "conditions", "trials"]
    for section in required_sections:
        if section not in config:
            result.add_error(section, f"Required section '{section}' is missing")

    # Validate study section
    if "study" in config:
        study = config["study"]
        if not isinstance(study, dict):
            result.add_error("study", "Must be a dictionary")
        else:
            if "name" not in study:
                result.add_error("study.name", "Study name is required")
            else:
                result.merge(validate_identifier(study["name"], "study.name"))

            if "version" not in study:
                result.add_warning("study.version", "Version not specified")

            if "hypothesis" not in study:
                result.add_warning("study.hypothesis", "Hypothesis not specified")

    # Validate conditions
    if "conditions" in config:
        conditions = config["conditions"]
        if not isinstance(conditions, list):
            result.add_error("conditions", "Must be a list")
        elif len(conditions) < 2:
            result.add_error("conditions", "At least 2 conditions required")
        else:
            condition_names = set()
            for i, condition in enumerate(conditions):
                cond_result = validate_condition(condition)
                result.merge(cond_result, f"conditions[{i}]")

                # Check for duplicate names
                name = condition.get("name", "")
                if name in condition_names:
                    result.add_error(f"conditions[{i}].name", f"Duplicate condition name: '{name}'")
                condition_names.add(name)

    # Validate trials section
    if "trials" in config:
        trials = config["trials"]
        if not isinstance(trials, dict):
            result.add_error("trials", "Must be a dictionary")
        else:
            if "repetitions" in trials:
                reps = trials["repetitions"]
                if not isinstance(reps, int) or reps < 1:
                    result.add_error("trials.repetitions", "Must be a positive integer")

            if "seed" in trials:
                seed = trials["seed"]
                if not isinstance(seed, int):
                    result.add_error("trials.seed", "Must be an integer")

    # Validate models section
    if "models" in config:
        models = config["models"]
        if not isinstance(models, list):
            result.add_error("models", "Must be a list")
        elif len(models) == 0:
            result.add_error("models", "At least one model required")
        else:
            for i, model in enumerate(models):
                model_result = validate_model_config(model)
                result.merge(model_result, f"models[{i}]")

    # Validate evaluation section
    if "evaluation" in config:
        eval_config = config["evaluation"]
        if not isinstance(eval_config, dict):
            result.add_error("evaluation", "Must be a dictionary")
        else:
            if "modes" in eval_config:
                modes = eval_config["modes"]
                if not isinstance(modes, list):
                    result.add_error("evaluation.modes", "Must be a list")
                else:
                    for i, mode in enumerate(modes):
                        if mode not in VALID_EVAL_MODES:
                            result.add_warning(
                                f"evaluation.modes[{i}]",
                                f"Unknown evaluation mode: '{mode}'. Valid: {VALID_EVAL_MODES}"
                            )

    # Validate analysis section
    if "analysis" in config:
        analysis = config["analysis"]
        if not isinstance(analysis, dict):
            result.add_error("analysis", "Must be a dictionary")
        else:
            if "alpha" in analysis:
                alpha = analysis["alpha"]
                if not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
                    result.add_error("analysis.alpha", "Must be between 0 and 1")

            if "tests" in analysis:
                tests = analysis["tests"]
                if not isinstance(tests, list):
                    result.add_error("analysis.tests", "Must be a list")

    # Validate pilot section
    if "pilot" in config:
        pilot = config["pilot"]
        if not isinstance(pilot, dict):
            result.add_error("pilot", "Must be a dictionary")
        else:
            if "fraction" in pilot:
                frac = pilot["fraction"]
                if not isinstance(frac, (int, float)) or not 0 < frac < 1:
                    result.add_error("pilot.fraction", "Must be between 0 and 1")

    return result


def validate_condition(condition: dict) -> ValidationResult:
    """Validate a condition definition."""
    result = ValidationResult(valid=True)

    if not isinstance(condition, dict):
        result.add_error("", "Condition must be a dictionary")
        return result

    # Required fields
    if "name" not in condition:
        result.add_error("name", "Condition name is required")
    else:
        result.merge(validate_identifier(condition["name"], "name"))

    # Optional fields
    if "description" in condition and not isinstance(condition["description"], str):
        result.add_error("description", "Must be a string")

    return result


def validate_model_config(model: dict) -> ValidationResult:
    """Validate a model configuration."""
    result = ValidationResult(valid=True)

    if not isinstance(model, dict):
        result.add_error("", "Model config must be a dictionary")
        return result

    # Must have alias or model_id
    if "alias" not in model and "model_id" not in model:
        result.add_error("", "Model must have 'alias' or 'model_id'")

    # Validate provider if specified
    if "provider" in model:
        provider = model["provider"]
        if provider not in VALID_PROVIDERS:
            result.add_warning("provider", f"Unknown provider: '{provider}'. Valid: {VALID_PROVIDERS}")

    # Validate temperature if specified
    if "temperature" in model:
        temp = model["temperature"]
        if not isinstance(temp, (int, float)) or not 0 <= temp <= 2:
            result.add_error("temperature", "Must be between 0 and 2")

    return result


# =============================================================================
# Task Schema Validation
# =============================================================================

def validate_task(task: dict) -> ValidationResult:
    """Validate a task definition."""
    result = ValidationResult(valid=True)

    if not isinstance(task, dict):
        result.add_error("", "Task must be a dictionary")
        return result

    # Required fields
    required = ["id", "prompt"]
    for field in required:
        if field not in task:
            result.add_error(field, f"Required field '{field}' is missing")

    # Validate id
    if "id" in task:
        result.merge(validate_identifier(task["id"], "id"))

    # Validate prompt
    if "prompt" in task:
        if not isinstance(task["prompt"], str):
            result.add_error("prompt", "Must be a string")
        elif len(task["prompt"]) < 5:
            result.add_warning("prompt", "Prompt seems too short")

    # Validate expected if present
    if "expected" in task:
        expected = task["expected"]
        if not isinstance(expected, dict):
            result.add_error("expected", "Must be a dictionary")

    return result


def validate_tasks_file(tasks: dict) -> ValidationResult:
    """Validate a tasks.yaml file."""
    result = ValidationResult(valid=True)

    if not isinstance(tasks, dict):
        result.add_error("", "Tasks file must be a dictionary")
        return result

    # Validate tools if present
    if "tools" in tasks:
        tools = tasks["tools"]
        if not isinstance(tools, list):
            result.add_error("tools", "Must be a list")
        else:
            for i, tool in enumerate(tools):
                tool_result = validate_tool(tool)
                result.merge(tool_result, f"tools[{i}]")

    # Validate tasks
    if "tasks" not in tasks:
        result.add_error("tasks", "Required section 'tasks' is missing")
    else:
        task_list = tasks["tasks"]
        if not isinstance(task_list, list):
            result.add_error("tasks", "Must be a list")
        elif len(task_list) == 0:
            result.add_error("tasks", "At least one task required")
        else:
            task_ids = set()
            for i, task in enumerate(task_list):
                task_result = validate_task(task)
                result.merge(task_result, f"tasks[{i}]")

                # Check for duplicate IDs
                task_id = task.get("id", "")
                if task_id in task_ids:
                    result.add_error(f"tasks[{i}].id", f"Duplicate task ID: '{task_id}'")
                task_ids.add(task_id)

    return result


def validate_tool(tool: dict) -> ValidationResult:
    """Validate a tool definition."""
    result = ValidationResult(valid=True)

    if not isinstance(tool, dict):
        result.add_error("", "Tool must be a dictionary")
        return result

    # Required fields
    if "name" not in tool:
        result.add_error("name", "Tool name is required")
    else:
        result.merge(validate_identifier(tool["name"], "name"))

    if "description" not in tool:
        result.add_warning("description", "Tool description recommended")

    # Validate parameters if present
    if "parameters" in tool:
        params = tool["parameters"]
        if not isinstance(params, dict):
            result.add_error("parameters", "Must be a dictionary")
        else:
            for param_name, param_def in params.items():
                if not isinstance(param_def, dict):
                    result.add_error(f"parameters.{param_name}", "Must be a dictionary")
                elif "type" not in param_def:
                    result.add_warning(f"parameters.{param_name}.type", "Type not specified")

    return result


# =============================================================================
# Trial and Response Validation
# =============================================================================

def validate_trial(trial: dict) -> ValidationResult:
    """Validate a trial definition."""
    result = ValidationResult(valid=True)

    if not isinstance(trial, dict):
        result.add_error("", "Trial must be a dictionary")
        return result

    required = ["trial_id", "task_id", "condition"]
    for field in required:
        if field not in trial:
            result.add_error(field, f"Required field '{field}' is missing")

    if "repetition" in trial:
        rep = trial["repetition"]
        if not isinstance(rep, int) or rep < 0:
            result.add_error("repetition", "Must be a non-negative integer")

    return result


def validate_response(response: dict) -> ValidationResult:
    """Validate a response record."""
    result = ValidationResult(valid=True)

    if not isinstance(response, dict):
        result.add_error("", "Response must be a dictionary")
        return result

    required = ["trial_id", "response", "success"]
    for field in required:
        if field not in response:
            result.add_error(field, f"Required field '{field}' is missing")

    if "timestamp" not in response:
        result.add_warning("timestamp", "Timestamp not recorded")

    return result


# =============================================================================
# Study Directory Validation
# =============================================================================

def validate_study_directory(study_path: Path) -> ValidationResult:
    """
    Validate a study directory structure.

    Args:
        study_path: Path to study directory

    Returns:
        ValidationResult with errors and warnings
    """
    result = ValidationResult(valid=True)

    if not study_path.exists():
        result.add_error("", f"Study directory does not exist: {study_path}")
        return result

    if not study_path.is_dir():
        result.add_error("", f"Path is not a directory: {study_path}")
        return result

    # Check required files
    required_files = ["config.yaml"]
    for filename in required_files:
        if not (study_path / filename).exists():
            result.add_error(filename, f"Required file '{filename}' is missing")

    # Check for tasks definition (either .yaml or .py)
    has_tasks = (study_path / "tasks.yaml").exists() or (study_path / "tasks.py").exists()
    if not has_tasks:
        result.add_error("tasks", "No tasks.yaml or tasks.py found")

    # Validate config.yaml if present
    config_path = study_path / "config.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            config_result = validate_study_config(config)
            result.merge(config_result, "config.yaml")
        except Exception as e:
            result.add_error("config.yaml", f"Failed to parse: {e}")

    # Validate tasks.yaml if present
    tasks_path = study_path / "tasks.yaml"
    if tasks_path.exists():
        try:
            import yaml
            with open(tasks_path) as f:
                tasks = yaml.safe_load(f)
            tasks_result = validate_tasks_file(tasks)
            result.merge(tasks_result, "tasks.yaml")
        except Exception as e:
            result.add_error("tasks.yaml", f"Failed to parse: {e}")

    # Check stages directory structure
    stages_path = study_path / "stages"
    if stages_path.exists():
        expected_stages = [
            "1_configure", "2_generate", "3_execute",
            "4_evaluate", "5_analyze", "6_report"
        ]
        for stage in expected_stages:
            if not (stages_path / stage).exists():
                result.add_warning(f"stages/{stage}", "Stage directory not created yet")

    return result


# =============================================================================
# High-Level Validation Functions
# =============================================================================

def validate_config_file(path: Path) -> ValidationResult:
    """
    Validate a config.yaml file.

    Args:
        path: Path to config.yaml

    Returns:
        ValidationResult
    """
    result = ValidationResult(valid=True)

    if not path.exists():
        result.add_error("", f"File does not exist: {path}")
        return result

    try:
        import yaml
        with open(path) as f:
            config = yaml.safe_load(f)
        return validate_study_config(config)
    except Exception as e:
        result.add_error("", f"Failed to parse config: {e}")
        return result


def validate_all(study_path: Path) -> ValidationResult:
    """
    Run all validations on a study.

    Args:
        study_path: Path to study directory

    Returns:
        Combined ValidationResult
    """
    return validate_study_directory(study_path)
