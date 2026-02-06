"""
Utility functions for the research pipeline.

Provides centralized utilities for:
- Hashing and checksums
- Timestamps and date handling
- File operations
- JSON serialization helpers
- Logging configuration
"""

import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np


# =============================================================================
# Logging Configuration
# =============================================================================

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Configure logging for the pipeline.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional file to write logs to
        format_string: Custom format string

    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logger = logging.getLogger("pipeline")
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "pipeline") -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


# =============================================================================
# Hashing and Checksums
# =============================================================================

def hash_string(s: str, algorithm: str = "sha256") -> str:
    """
    Compute hash of a string.

    Args:
        s: String to hash
        algorithm: Hash algorithm (default: sha256)

    Returns:
        Hash string with algorithm prefix (e.g., "sha256:abc123...")
    """
    hasher = hashlib.new(algorithm)
    hasher.update(s.encode("utf-8"))
    return f"{algorithm}:{hasher.hexdigest()}"


def hash_file(path: Path, algorithm: str = "sha256") -> str:
    """
    Compute hash of a file.

    Args:
        path: Path to file
        algorithm: Hash algorithm (default: sha256)

    Returns:
        Hash string with algorithm prefix
    """
    hasher = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return f"{algorithm}:{hasher.hexdigest()}"


def hash_directory(
    path: Path,
    algorithm: str = "sha256",
    exclude: Optional[list[str]] = None,
    include_structure: bool = True,
) -> str:
    """
    Compute deterministic hash of directory contents.

    Files are sorted by path for determinism.

    Args:
        path: Path to directory
        algorithm: Hash algorithm (default: sha256)
        exclude: Patterns to exclude
        include_structure: Whether to include file paths in hash

    Returns:
        Hash string with algorithm prefix
    """
    exclude = exclude or []
    hasher = hashlib.new(algorithm)

    for file in sorted(path.rglob("*")):
        if file.is_file():
            # Skip excluded patterns
            if any(ex in str(file) for ex in exclude):
                continue

            if include_structure:
                # Include relative path in hash for structure
                rel_path = file.relative_to(path)
                hasher.update(str(rel_path).encode())

            hasher.update(file.read_bytes())

    return f"{algorithm}:{hasher.hexdigest()}"


def hash_dict(d: dict, algorithm: str = "sha256") -> str:
    """
    Compute hash of a dictionary (JSON-serialized).

    Args:
        d: Dictionary to hash
        algorithm: Hash algorithm

    Returns:
        Hash string with algorithm prefix
    """
    # Sort keys for determinism
    json_str = json.dumps(d, sort_keys=True, default=str)
    return hash_string(json_str, algorithm)


def verify_hash(content: Union[str, bytes, Path], expected_hash: str) -> bool:
    """
    Verify content matches expected hash.

    Args:
        content: String, bytes, or file path to verify
        expected_hash: Expected hash (with algorithm prefix)

    Returns:
        True if hash matches
    """
    algorithm, expected = expected_hash.split(":", 1)

    if isinstance(content, Path):
        actual = hash_file(content, algorithm)
    elif isinstance(content, bytes):
        hasher = hashlib.new(algorithm)
        hasher.update(content)
        actual = f"{algorithm}:{hasher.hexdigest()}"
    else:
        actual = hash_string(content, algorithm)

    return actual == expected_hash


# =============================================================================
# Timestamps and Dates
# =============================================================================

def now_utc() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


def now_iso() -> str:
    """Get current UTC datetime as ISO string."""
    return now_utc().isoformat()


def parse_iso(iso_string: str) -> datetime:
    """Parse ISO format datetime string."""
    return datetime.fromisoformat(iso_string)


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable form.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "2h 30m 15s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


# =============================================================================
# JSON Serialization Helpers
# =============================================================================

class PipelineJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types, datetimes, Paths, etc."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, "to_dict"):
            return obj.to_dict()
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)


def json_dumps(obj: Any, indent: int = 2, **kwargs) -> str:
    """
    Serialize object to JSON string with pipeline-aware encoding.

    Args:
        obj: Object to serialize
        indent: Indentation level
        **kwargs: Additional arguments to json.dumps

    Returns:
        JSON string
    """
    return json.dumps(obj, cls=PipelineJSONEncoder, indent=indent, **kwargs)


def json_dump(obj: Any, path: Path, indent: int = 2, **kwargs) -> None:
    """
    Serialize object to JSON file with pipeline-aware encoding.

    Args:
        obj: Object to serialize
        path: Path to write to
        indent: Indentation level
        **kwargs: Additional arguments to json.dump
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, cls=PipelineJSONEncoder, indent=indent, **kwargs)


def json_load(path: Path) -> Any:
    """
    Load JSON from file.

    Args:
        path: Path to JSON file

    Returns:
        Parsed JSON object
    """
    with open(path) as f:
        return json.load(f)


# =============================================================================
# File Operations
# =============================================================================

def ensure_dir(path: Path) -> Path:
    """
    Ensure directory exists, creating if necessary.

    Args:
        path: Directory path

    Returns:
        The path (for chaining)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_write(path: Path, content: str, backup: bool = True) -> None:
    """
    Safely write content to file with optional backup.

    Args:
        path: Path to write to
        content: Content to write
        backup: Whether to create backup of existing file
    """
    if backup and path.exists():
        backup_path = path.with_suffix(path.suffix + ".bak")
        backup_path.write_text(path.read_text())

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def read_yaml(path: Path) -> dict:
    """
    Read YAML file.

    Args:
        path: Path to YAML file

    Returns:
        Parsed YAML as dict
    """
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict) -> None:
    """
    Write dict to YAML file.

    Args:
        path: Path to write to
        data: Data to write
    """
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# =============================================================================
# Environment and Configuration
# =============================================================================

def get_env(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """
    Get environment variable with optional default and required check.

    Args:
        key: Environment variable name
        default: Default value if not set
        required: Raise error if not set and no default

    Returns:
        Environment variable value

    Raises:
        ValueError: If required and not set
    """
    value = os.environ.get(key, default)
    if required and value is None:
        raise ValueError(f"Required environment variable {key} is not set")
    return value


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path to project root (parent of pipeline/)
    """
    return Path(__file__).parent.parent


def get_papers_dir() -> Path:
    """Get the papers directory."""
    return get_project_root() / "papers"


def get_templates_dir() -> Path:
    """Get the templates directory."""
    return get_project_root() / "templates"


# =============================================================================
# Data Validation Helpers
# =============================================================================

def is_valid_study_name(name: str) -> bool:
    """
    Check if study name is valid.

    Args:
        name: Study name to validate

    Returns:
        True if valid
    """
    import re
    return bool(re.match(r"^[a-z][a-z0-9_]*$", name))


def is_valid_condition_name(name: str) -> bool:
    """
    Check if condition name is valid.

    Args:
        name: Condition name to validate

    Returns:
        True if valid
    """
    import re
    return bool(re.match(r"^[a-z][a-z0-9_]*$", name))


def validate_probability(value: float, name: str = "probability") -> float:
    """
    Validate that value is a valid probability [0, 1].

    Args:
        value: Value to validate
        name: Name for error message

    Returns:
        Validated value

    Raises:
        ValueError: If not valid probability
    """
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")
    return value


def validate_positive_int(value: int, name: str = "value") -> int:
    """
    Validate that value is a positive integer.

    Args:
        value: Value to validate
        name: Name for error message

    Returns:
        Validated value

    Raises:
        ValueError: If not positive integer
    """
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value
