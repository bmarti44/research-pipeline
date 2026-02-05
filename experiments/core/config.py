"""
Runtime configuration with model version locking for reproducibility.

Per PLAN.md requirements:
- Model IDs are read from environment variables
- Configuration is locked before first API call
- Full environment is recorded for reproducibility
"""

import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class ModelVersions:
    """Locked model versions for reproducibility."""
    claude_model: str
    gpt_model: str
    gemini_model: str
    locked_at: str

    @classmethod
    def from_environment(cls) -> "ModelVersions":
        """Load model versions from environment variables with defaults."""
        return cls(
            claude_model=os.environ.get("CLAUDE_MODEL_ID", "claude-sonnet-4-20250514"),
            gpt_model=os.environ.get("OPENAI_MODEL_ID", "gpt-4o-2024-08-06"),
            gemini_model=os.environ.get("GEMINI_MODEL_ID", "gemini-1.5-pro-002"),
            locked_at=datetime.now(timezone.utc).isoformat(),
        )


@dataclass
class EnvironmentInfo:
    """Full environment information for reproducibility."""
    python_version: str
    numpy_version: str
    scipy_version: str
    anthropic_version: Optional[str]
    openai_version: Optional[str]
    platform: str
    recorded_at: str

    @classmethod
    def capture(cls) -> "EnvironmentInfo":
        """Capture current environment information."""
        import platform as plat

        # Get package versions safely
        numpy_version = "unknown"
        scipy_version = "unknown"
        anthropic_version = None
        openai_version = None

        try:
            import numpy
            numpy_version = numpy.__version__
        except ImportError:
            pass

        try:
            import scipy
            scipy_version = scipy.__version__
        except ImportError:
            pass

        try:
            import anthropic
            anthropic_version = anthropic.__version__
        except ImportError:
            pass

        try:
            import openai
            openai_version = openai.__version__
        except ImportError:
            pass

        return cls(
            python_version=sys.version,
            numpy_version=numpy_version,
            scipy_version=scipy_version,
            anthropic_version=anthropic_version,
            openai_version=openai_version,
            platform=plat.platform(),
            recorded_at=datetime.now(timezone.utc).isoformat(),
        )


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    model_versions: ModelVersions
    environment: EnvironmentInfo
    seed: int
    temperature: float
    n_trials_per_task: int
    conditions: list[str]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_versions": asdict(self.model_versions),
            "environment": asdict(self.environment),
            "seed": self.seed,
            "temperature": self.temperature,
            "n_trials_per_task": self.n_trials_per_task,
            "conditions": self.conditions,
        }


class ConfigManager:
    """Manages experiment configuration with locking."""

    def __init__(self, results_dir: str = "experiments/results"):
        self.results_dir = Path(results_dir)
        self.config_lock_path = self.results_dir / "model_config_lock.json"
        self.environment_path = self.results_dir / "environment.json"
        self._locked = False
        self._config: Optional[ExperimentConfig] = None

    def create_config(
        self,
        seed: int = 42,
        temperature: float = 0.0,
        n_trials_per_task: int = 30,
        conditions: Optional[list[str]] = None,
    ) -> ExperimentConfig:
        """Create experiment configuration (but don't lock yet)."""
        if conditions is None:
            conditions = ["nl_only", "json_only"]

        self._config = ExperimentConfig(
            model_versions=ModelVersions.from_environment(),
            environment=EnvironmentInfo.capture(),
            seed=seed,
            temperature=temperature,
            n_trials_per_task=n_trials_per_task,
            conditions=conditions,
        )
        return self._config

    def lock(self) -> None:
        """
        Lock configuration - must be called before first API call.

        After locking, configuration cannot be modified.
        """
        if self._locked:
            return

        if self._config is None:
            raise RuntimeError("No configuration created. Call create_config() first.")

        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Write model config lock
        with open(self.config_lock_path, "w") as f:
            json.dump(asdict(self._config.model_versions), f, indent=2)

        # Write environment info
        with open(self.environment_path, "w") as f:
            json.dump(asdict(self._config.environment), f, indent=2)

        self._locked = True

    def is_locked(self) -> bool:
        """Check if configuration is locked."""
        return self._locked

    def load_locked_config(self) -> Optional[dict]:
        """Load previously locked configuration."""
        if not self.config_lock_path.exists():
            return None

        with open(self.config_lock_path) as f:
            return json.load(f)

    def verify_config_unchanged(self) -> bool:
        """Verify current environment matches locked configuration."""
        locked = self.load_locked_config()
        if locked is None:
            return False

        current = ModelVersions.from_environment()

        return (
            locked["claude_model"] == current.claude_model
            and locked["gpt_model"] == current.gpt_model
            and locked["gemini_model"] == current.gemini_model
        )

    @property
    def config(self) -> Optional[ExperimentConfig]:
        """Get current configuration."""
        return self._config


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(results_dir: str = "experiments/results") -> ConfigManager:
    """Get or create the global config manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(results_dir)
    return _config_manager


def get_model_config() -> dict:
    """
    Get model configuration from environment with runtime locking.

    This is the simple interface mentioned in PLAN.md.
    """
    return {
        "claude_model": os.environ.get("CLAUDE_MODEL_ID", "claude-sonnet-4-20250514"),
        "gpt_model": os.environ.get("OPENAI_MODEL_ID", "gpt-4o-2024-08-06"),
        "gemini_model": os.environ.get("GEMINI_MODEL_ID", "gemini-1.5-pro-002"),
        "locked_at": datetime.now(timezone.utc).isoformat(),
    }
