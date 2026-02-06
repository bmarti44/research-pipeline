"""
Generic model classes for research studies.

These are the building blocks that can be instantiated via configuration.
All domain-specific details come from config, not from class definitions.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Callable
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path


# =============================================================================
# Conditions
# =============================================================================

@dataclass
class Condition:
    """
    An experimental condition.

    Conditions are treatments/levels of the independent variable.
    They are defined generically and instantiated with specific parameters.

    Example config:
        conditions:
          - name: treatment_a
            description: "First treatment"
            params:
              prompt_modifier: "You must respond in JSON"
              temperature: 0.0
    """
    name: str
    description: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    # Optional: condition-specific prompt template
    prompt_template: Optional[str] = None

    # Optional: condition-specific system instructions
    system_instructions: Optional[str] = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("Condition must have a name")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "params": self.params,
            "prompt_template": self.prompt_template,
            "system_instructions": self.system_instructions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Condition":
        return cls(**data)

    @classmethod
    def from_config(cls, config: dict) -> "Condition":
        """Create condition from YAML config entry."""
        return cls(
            name=config["name"],
            description=config.get("description", ""),
            params=config.get("params", {}),
            prompt_template=config.get("prompt_template"),
            system_instructions=config.get("system_instructions"),
        )


# =============================================================================
# Tasks
# =============================================================================

@dataclass
class Task:
    """
    A single experimental task/stimulus.

    Tasks are presented to the model and responses are evaluated.
    They are defined generically and can represent any type of task.

    Example config:
        tasks:
          - id: task_001
            category: arithmetic
            prompt: "What is 2 + 2?"
            expected:
              answer: "4"
            metadata:
              difficulty: easy
    """
    id: str
    prompt: str
    category: str = "default"
    expected: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Optional: task-specific context or setup
    context: Optional[str] = None

    # Optional: tools available for this task
    tools: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            raise ValueError("Task must have an id")
        if not self.prompt:
            raise ValueError("Task must have a prompt")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "category": self.category,
            "expected": self.expected,
            "metadata": self.metadata,
            "context": self.context,
            "tools": self.tools,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        return cls(**data)

    @classmethod
    def from_config(cls, config: dict) -> "Task":
        """Create task from YAML config entry."""
        return cls(
            id=config["id"],
            prompt=config["prompt"],
            category=config.get("category", "default"),
            expected=config.get("expected", {}),
            metadata=config.get("metadata", {}),
            context=config.get("context"),
            tools=config.get("tools", []),
        )


# =============================================================================
# Trials
# =============================================================================

@dataclass
class Trial:
    """
    A single experimental trial.

    A trial is a unique combination of task × condition × repetition.
    Trials are generated from the study design, not defined manually.
    """
    trial_id: str
    task_id: str
    condition_name: str
    repetition: int

    # Links to full objects (populated at runtime)
    task: Optional[Task] = None
    condition: Optional[Condition] = None

    # Execution metadata
    seed: Optional[int] = None
    order: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "trial_id": self.trial_id,
            "task_id": self.task_id,
            "condition_name": self.condition_name,
            "repetition": self.repetition,
            "seed": self.seed,
            "order": self.order,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Trial":
        return cls(
            trial_id=data["trial_id"],
            task_id=data["task_id"],
            condition_name=data["condition_name"],
            repetition=data["repetition"],
            seed=data.get("seed"),
            order=data.get("order"),
        )


# =============================================================================
# Responses
# =============================================================================

@dataclass
class Response:
    """
    A model's response to a trial.

    Captures everything needed for evaluation and analysis.
    """
    trial_id: str
    model: str
    raw_output: str

    # Timing
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    latency_ms: Optional[float] = None

    # Token usage
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    # API metadata
    request_id: Optional[str] = None
    api_version: Optional[str] = None

    # Error handling
    error: Optional[str] = None
    retry_count: int = 0

    # Parsed output (populated by extractor)
    parsed: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "trial_id": self.trial_id,
            "model": self.model,
            "raw_output": self.raw_output,
            "timestamp": self.timestamp,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "request_id": self.request_id,
            "api_version": self.api_version,
            "error": self.error,
            "retry_count": self.retry_count,
            "parsed": self.parsed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Response":
        return cls(**data)


# =============================================================================
# Evaluations
# =============================================================================

class EvaluationMode(Enum):
    """Standard evaluation modes."""
    STRICT = "strict"      # Exact match required
    INTENT = "intent"      # Correct intent, flexible format
    FUNCTIONAL = "functional"  # Functionally equivalent acceptable


@dataclass
class Evaluation:
    """
    Evaluation of a single response.

    Supports multiple evaluation modes for the same response.
    """
    trial_id: str
    evaluator: str  # Which evaluator produced this

    # Results by mode
    results: dict[str, dict] = field(default_factory=dict)
    # e.g., {"strict": {"correct": False, "reason": "..."}, "intent": {"correct": True}}

    # Timing
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Determinism check
    is_deterministic: bool = True
    determinism_checks: int = 1

    def to_dict(self) -> dict:
        return {
            "trial_id": self.trial_id,
            "evaluator": self.evaluator,
            "results": self.results,
            "timestamp": self.timestamp,
            "is_deterministic": self.is_deterministic,
            "determinism_checks": self.determinism_checks,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Evaluation":
        return cls(**data)

    def is_correct(self, mode: EvaluationMode = EvaluationMode.STRICT) -> bool:
        """Check if response was correct under given mode."""
        mode_key = mode.value if isinstance(mode, EvaluationMode) else mode
        return self.results.get(mode_key, {}).get("correct", False)


# =============================================================================
# Scenarios (Collections of related tasks)
# =============================================================================

@dataclass
class Scenario:
    """
    A scenario is a coherent collection of related tasks.

    Scenarios group tasks that share context or test related behaviors.
    They can define shared setup, tools, or evaluation criteria.

    Example config:
        scenarios:
          - id: file_operations
            description: "Tasks involving file manipulation"
            shared_context: "You have access to a filesystem."
            shared_tools: [read_file, write_file, edit_file]
            tasks:
              - id: task_001
                prompt: "Read config.yaml"
                ...
    """
    id: str
    description: str = ""

    # Shared context for all tasks in scenario
    shared_context: Optional[str] = None

    # Shared tools for all tasks in scenario
    shared_tools: list[dict] = field(default_factory=list)

    # Tasks in this scenario
    task_ids: list[str] = field(default_factory=list)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "shared_context": self.shared_context,
            "shared_tools": self.shared_tools,
            "task_ids": self.task_ids,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Scenario":
        return cls(**data)

    @classmethod
    def from_config(cls, config: dict) -> "Scenario":
        """Create scenario from YAML config entry."""
        return cls(
            id=config["id"],
            description=config.get("description", ""),
            shared_context=config.get("shared_context"),
            shared_tools=config.get("shared_tools", []),
            task_ids=config.get("task_ids", []),
            metadata=config.get("metadata", {}),
        )


# =============================================================================
# Study Design
# =============================================================================

@dataclass
class StudyDesign:
    """
    Complete study design specification.

    This is the top-level container that holds all conditions, tasks,
    and design parameters. It's built from config and used to generate trials.
    """
    name: str
    version: str = "1.0.0"
    hypothesis: str = ""

    # Design components
    conditions: list[Condition] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    scenarios: list[Scenario] = field(default_factory=list)

    # Design parameters
    design_type: str = "between"  # between, within, mixed
    repetitions: int = 1
    seed: int = 42
    randomize_order: bool = True

    # Evaluation settings
    evaluation_modes: list[str] = field(default_factory=lambda: ["strict", "intent", "functional"])

    # Statistical tests to run (by name, looked up in registry)
    statistical_tests: list[str] = field(default_factory=list)
    alpha: float = 0.05

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "hypothesis": self.hypothesis,
            "conditions": [c.to_dict() for c in self.conditions],
            "tasks": [t.to_dict() for t in self.tasks],
            "scenarios": [s.to_dict() for s in self.scenarios],
            "design_type": self.design_type,
            "repetitions": self.repetitions,
            "seed": self.seed,
            "randomize_order": self.randomize_order,
            "evaluation_modes": self.evaluation_modes,
            "statistical_tests": self.statistical_tests,
            "alpha": self.alpha,
        }

    @classmethod
    def from_config(cls, config: dict) -> "StudyDesign":
        """Build study design from YAML config."""
        # Parse conditions
        conditions = [
            Condition.from_config(c)
            for c in config.get("conditions", [])
        ]

        # Parse tasks
        tasks = [
            Task.from_config(t)
            for t in config.get("tasks", {}).get("items", config.get("tasks", []))
            if isinstance(t, dict)
        ]

        # Parse scenarios
        scenarios = [
            Scenario.from_config(s)
            for s in config.get("scenarios", [])
        ]

        # Get trial settings
        trials_config = config.get("trials", {})

        # Get analysis settings
        analysis_config = config.get("analysis", {})

        return cls(
            name=config.get("study", {}).get("name", "unnamed"),
            version=config.get("study", {}).get("version", "1.0.0"),
            hypothesis=config.get("study", {}).get("hypothesis", ""),
            conditions=conditions,
            tasks=tasks,
            scenarios=scenarios,
            design_type=config.get("design", {}).get("type", "between"),
            repetitions=trials_config.get("repetitions", 1),
            seed=trials_config.get("seed", 42),
            randomize_order=trials_config.get("randomize_order", True),
            evaluation_modes=config.get("evaluation", {}).get("modes", ["strict", "intent", "functional"]),
            statistical_tests=[t["name"] for t in analysis_config.get("tests", [])],
            alpha=analysis_config.get("alpha", 0.05),
        )

    def generate_trials(self) -> list[Trial]:
        """Generate all trials from the study design."""
        import numpy as np

        rng = np.random.default_rng(self.seed)
        trials = []
        trial_num = 0

        for task in self.tasks:
            for condition in self.conditions:
                for rep in range(self.repetitions):
                    trial_num += 1
                    trial = Trial(
                        trial_id=f"trial_{trial_num:04d}",
                        task_id=task.id,
                        condition_name=condition.name,
                        repetition=rep + 1,
                        task=task,
                        condition=condition,
                        seed=int(rng.integers(0, 2**31)),
                    )
                    trials.append(trial)

        # Randomize order if requested
        if self.randomize_order:
            rng.shuffle(trials)

        # Assign order
        for i, trial in enumerate(trials):
            trial.order = i + 1

        return trials

    def get_task(self, task_id: str) -> Optional[Task]:
        """Look up task by ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def get_condition(self, name: str) -> Optional[Condition]:
        """Look up condition by name."""
        for condition in self.conditions:
            if condition.name == name:
                return condition
        return None


# =============================================================================
# Tool Definitions (Generic)
# =============================================================================

@dataclass
class ToolParameter:
    """A parameter for a tool."""
    name: str
    type: str  # string, number, boolean, array, object
    description: str = ""
    required: bool = True
    default: Any = None
    enum: list[Any] = field(default_factory=list)

    def to_schema(self) -> dict:
        """Convert to JSON Schema format."""
        schema = {
            "type": self.type,
            "description": self.description,
        }
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class Tool:
    """
    A tool definition.

    Tools are defined generically and can be used by any study.
    """
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)

    def to_schema(self) -> dict:
        """Convert to JSON Schema format for API calls."""
        required = [p.name for p in self.parameters if p.required]
        properties = {p.name: p.to_schema() for p in self.parameters}

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    @classmethod
    def from_config(cls, config: dict) -> "Tool":
        """Create tool from config."""
        params = []
        for name, spec in config.get("parameters", {}).items():
            if isinstance(spec, dict):
                params.append(ToolParameter(
                    name=name,
                    type=spec.get("type", "string"),
                    description=spec.get("description", ""),
                    required=spec.get("required", True),
                    default=spec.get("default"),
                    enum=spec.get("enum", []),
                ))
            else:
                params.append(ToolParameter(name=name, type="string"))

        return cls(
            name=config["name"],
            description=config.get("description", ""),
            parameters=params,
        )


# =============================================================================
# Factory Functions
# =============================================================================

def load_study_design(config_path: Path) -> StudyDesign:
    """Load study design from config file."""
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return StudyDesign.from_config(config)


def load_tasks_from_file(tasks_path: Path) -> list[Task]:
    """Load tasks from a Python or YAML file."""
    import importlib.util
    import yaml

    if tasks_path.suffix == ".py":
        spec = importlib.util.spec_from_file_location("tasks", tasks_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, "get_tasks"):
            task_dicts = module.get_tasks()
            return [Task.from_config(t) for t in task_dicts]
        elif hasattr(module, "TASKS"):
            return [Task.from_config(t) for t in module.TASKS]
        else:
            raise ValueError(f"Tasks file must define get_tasks() or TASKS: {tasks_path}")

    elif tasks_path.suffix in [".yaml", ".yml"]:
        with open(tasks_path) as f:
            data = yaml.safe_load(f)
        return [Task.from_config(t) for t in data.get("tasks", data)]

    else:
        raise ValueError(f"Unsupported tasks file format: {tasks_path.suffix}")


def load_tools_from_file(tools_path: Path) -> list[Tool]:
    """Load tool definitions from a Python or YAML file."""
    import importlib.util
    import yaml

    if tools_path.suffix == ".py":
        spec = importlib.util.spec_from_file_location("tools", tools_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, "get_tools"):
            tool_dicts = module.get_tools()
            return [Tool.from_config(t) for t in tool_dicts]
        elif hasattr(module, "TOOLS"):
            return [Tool.from_config(t) for t in module.TOOLS]
        else:
            raise ValueError(f"Tools file must define get_tools() or TOOLS: {tools_path}")

    elif tools_path.suffix in [".yaml", ".yml"]:
        with open(tools_path) as f:
            data = yaml.safe_load(f)
        return [Tool.from_config(t) for t in data.get("tools", data)]

    else:
        raise ValueError(f"Unsupported tools file format: {tools_path.suffix}")
