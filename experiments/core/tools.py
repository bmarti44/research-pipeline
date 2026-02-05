"""
Tool schemas and definitions for the format friction experiment.

Per PLAN.md:
- Complexity tiers: Control, Simple, Medium, Complex
- Factorial adversarial categories: Adv-JSON, Adv-Escape, Adv-Unicode, Adv-Combined

Each tool is defined with a JSON Schema for its parameters.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional
import json


class ToolComplexity(Enum):
    """Tool complexity tiers as defined in PLAN.md."""
    CONTROL = "control"
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    ADV_JSON = "adv_json"
    ADV_ESCAPE = "adv_escape"
    ADV_UNICODE = "adv_unicode"
    ADV_COMBINED = "adv_combined"


@dataclass
class ToolDefinition:
    """Definition of a tool with its schema."""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    complexity: ToolComplexity
    required_params: list[str]

    def to_json_schema(self) -> dict:
        """Convert to JSON Schema format for API calls."""
        return {
            "type": "object",
            "properties": self.parameters,
            "required": self.required_params,
        }

    def to_anthropic_format(self) -> dict:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.to_json_schema(),
        }

    def to_openai_format(self) -> dict:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.to_json_schema(),
            },
        }


# ==============================================================================
# CONTROL TIER - Minimal tools for baseline
# ==============================================================================

NOOP_TOOL = ToolDefinition(
    name="noop",
    description="A no-operation tool that does nothing. Used as a control.",
    parameters={},
    complexity=ToolComplexity.CONTROL,
    required_params=[],
)

# ==============================================================================
# SIMPLE TIER - Single required parameter
# ==============================================================================

READ_FILE_TOOL = ToolDefinition(
    name="read_file",
    description="Read the contents of a file at the specified path.",
    parameters={
        "path": {
            "type": "string",
            "description": "The absolute path to the file to read",
        },
    },
    complexity=ToolComplexity.SIMPLE,
    required_params=["path"],
)

DELETE_FILE_TOOL = ToolDefinition(
    name="delete_file",
    description="Delete a file at the specified path.",
    parameters={
        "path": {
            "type": "string",
            "description": "The absolute path to the file to delete",
        },
    },
    complexity=ToolComplexity.SIMPLE,
    required_params=["path"],
)

LIST_DIRECTORY_TOOL = ToolDefinition(
    name="list_directory",
    description="List the contents of a directory.",
    parameters={
        "path": {
            "type": "string",
            "description": "The absolute path to the directory to list",
        },
    },
    complexity=ToolComplexity.SIMPLE,
    required_params=["path"],
)

# ==============================================================================
# MEDIUM TIER - Multiple parameters, some optional
# ==============================================================================

WRITE_FILE_TOOL = ToolDefinition(
    name="write_file",
    description="Write content to a file at the specified path.",
    parameters={
        "path": {
            "type": "string",
            "description": "The absolute path to the file to write",
        },
        "content": {
            "type": "string",
            "description": "The content to write to the file",
        },
        "overwrite": {
            "type": "boolean",
            "description": "Whether to overwrite existing file (default: false)",
            "default": False,
        },
    },
    complexity=ToolComplexity.MEDIUM,
    required_params=["path", "content"],
)

SEARCH_FILES_TOOL = ToolDefinition(
    name="search_files",
    description="Search for files matching a pattern.",
    parameters={
        "pattern": {
            "type": "string",
            "description": "The glob pattern to match files (e.g., '*.py')",
        },
        "path": {
            "type": "string",
            "description": "The directory to search in",
        },
        "recursive": {
            "type": "boolean",
            "description": "Whether to search recursively (default: true)",
            "default": True,
        },
    },
    complexity=ToolComplexity.MEDIUM,
    required_params=["pattern", "path"],
)

RUN_COMMAND_TOOL = ToolDefinition(
    name="run_command",
    description="Run a shell command.",
    parameters={
        "command": {
            "type": "string",
            "description": "The command to execute",
        },
        "working_dir": {
            "type": "string",
            "description": "The working directory for the command",
        },
        "timeout": {
            "type": "integer",
            "description": "Timeout in seconds (default: 60)",
            "default": 60,
        },
    },
    complexity=ToolComplexity.MEDIUM,
    required_params=["command"],
)

# ==============================================================================
# COMPLEX TIER - Nested objects, arrays
# ==============================================================================

EDIT_FILE_TOOL = ToolDefinition(
    name="edit_file",
    description="Make line-based edits to a file. Each edit replaces exact string matches.",
    parameters={
        "path": {
            "type": "string",
            "description": "The absolute path to the file to edit",
        },
        "edits": {
            "type": "array",
            "description": "List of edits to apply",
            "items": {
                "type": "object",
                "properties": {
                    "old_string": {
                        "type": "string",
                        "description": "The text to search for",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The text to replace with",
                    },
                },
                "required": ["old_string", "new_string"],
            },
        },
    },
    complexity=ToolComplexity.COMPLEX,
    required_params=["path", "edits"],
)

CREATE_FILES_TOOL = ToolDefinition(
    name="create_files",
    description="Create multiple files at once.",
    parameters={
        "files": {
            "type": "array",
            "description": "List of files to create",
            "items": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path for the file",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content for the file",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    complexity=ToolComplexity.COMPLEX,
    required_params=["files"],
)

HTTP_REQUEST_TOOL = ToolDefinition(
    name="http_request",
    description="Make an HTTP request.",
    parameters={
        "method": {
            "type": "string",
            "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
            "description": "The HTTP method",
        },
        "url": {
            "type": "string",
            "description": "The URL to request",
        },
        "headers": {
            "type": "object",
            "description": "HTTP headers as key-value pairs",
            "additionalProperties": {"type": "string"},
        },
        "body": {
            "type": "string",
            "description": "Request body (for POST, PUT, PATCH)",
        },
    },
    complexity=ToolComplexity.COMPLEX,
    required_params=["method", "url"],
)

# ==============================================================================
# ADVERSARIAL TIER - Factorial design per PLAN.md
# ==============================================================================

# Adv-JSON: Content that IS JSON (nested JSON in args)
WRITE_JSON_FILE_TOOL = ToolDefinition(
    name="write_json_file",
    description="Write JSON content to a file. The content must be valid JSON.",
    parameters={
        "path": {
            "type": "string",
            "description": "The path to write the JSON file",
        },
        "json_content": {
            "type": "object",
            "description": "The JSON object to write to the file",
            "additionalProperties": True,
        },
    },
    complexity=ToolComplexity.ADV_JSON,
    required_params=["path", "json_content"],
)

# Adv-Escape: Content requiring escaping (quotes, backslashes)
WRITE_ESCAPED_TOOL = ToolDefinition(
    name="write_escaped_content",
    description="Write content that may contain special characters requiring escaping.",
    parameters={
        "path": {
            "type": "string",
            "description": "The path to write the file",
        },
        "content": {
            "type": "string",
            "description": "The content to write (may contain quotes, backslashes, etc.)",
        },
    },
    complexity=ToolComplexity.ADV_ESCAPE,
    required_params=["path", "content"],
)

# Adv-Unicode: Content with unicode characters
WRITE_UNICODE_TOOL = ToolDefinition(
    name="write_unicode_content",
    description="Write content that may contain unicode characters.",
    parameters={
        "path": {
            "type": "string",
            "description": "The path to write the file",
        },
        "content": {
            "type": "string",
            "description": "The content to write (may contain emoji, non-ASCII, etc.)",
        },
    },
    complexity=ToolComplexity.ADV_UNICODE,
    required_params=["path", "content"],
)

# Adv-Combined: All adversarial factors combined
WRITE_COMPLEX_CONFIG_TOOL = ToolDefinition(
    name="write_complex_config",
    description="Write a complex configuration file with JSON, escaping, and unicode support.",
    parameters={
        "path": {
            "type": "string",
            "description": "The path to write the config file",
        },
        "config": {
            "type": "object",
            "description": "The configuration object (may contain nested JSON, special chars, unicode)",
            "additionalProperties": True,
        },
        "format": {
            "type": "string",
            "enum": ["json", "yaml", "toml"],
            "description": "Output format for the config",
        },
    },
    complexity=ToolComplexity.ADV_COMBINED,
    required_params=["path", "config", "format"],
)


# ==============================================================================
# Tool Collections
# ==============================================================================

ALL_TOOLS: list[ToolDefinition] = [
    # Control
    NOOP_TOOL,
    # Simple
    READ_FILE_TOOL,
    DELETE_FILE_TOOL,
    LIST_DIRECTORY_TOOL,
    # Medium
    WRITE_FILE_TOOL,
    SEARCH_FILES_TOOL,
    RUN_COMMAND_TOOL,
    # Complex
    EDIT_FILE_TOOL,
    CREATE_FILES_TOOL,
    HTTP_REQUEST_TOOL,
    # Adversarial
    WRITE_JSON_FILE_TOOL,
    WRITE_ESCAPED_TOOL,
    WRITE_UNICODE_TOOL,
    WRITE_COMPLEX_CONFIG_TOOL,
]

TOOLS_BY_COMPLEXITY: dict[ToolComplexity, list[ToolDefinition]] = {
    ToolComplexity.CONTROL: [NOOP_TOOL],
    ToolComplexity.SIMPLE: [READ_FILE_TOOL, DELETE_FILE_TOOL, LIST_DIRECTORY_TOOL],
    ToolComplexity.MEDIUM: [WRITE_FILE_TOOL, SEARCH_FILES_TOOL, RUN_COMMAND_TOOL],
    ToolComplexity.COMPLEX: [EDIT_FILE_TOOL, CREATE_FILES_TOOL, HTTP_REQUEST_TOOL],
    ToolComplexity.ADV_JSON: [WRITE_JSON_FILE_TOOL],
    ToolComplexity.ADV_ESCAPE: [WRITE_ESCAPED_TOOL],
    ToolComplexity.ADV_UNICODE: [WRITE_UNICODE_TOOL],
    ToolComplexity.ADV_COMBINED: [WRITE_COMPLEX_CONFIG_TOOL],
}


def get_tool_by_name(name: str) -> Optional[ToolDefinition]:
    """Look up a tool by name."""
    for tool in ALL_TOOLS:
        if tool.name == name:
            return tool
    return None


def get_tools_for_experiment(
    include_adversarial: bool = True,
    complexities: Optional[list[ToolComplexity]] = None,
) -> list[ToolDefinition]:
    """Get tools for an experiment run."""
    if complexities is not None:
        return [t for t in ALL_TOOLS if t.complexity in complexities]

    if include_adversarial:
        return ALL_TOOLS

    # Exclude adversarial tools
    adversarial = {
        ToolComplexity.ADV_JSON,
        ToolComplexity.ADV_ESCAPE,
        ToolComplexity.ADV_UNICODE,
        ToolComplexity.ADV_COMBINED,
    }
    return [t for t in ALL_TOOLS if t.complexity not in adversarial]


def format_tools_for_anthropic(tools: list[ToolDefinition]) -> list[dict]:
    """Format tools for Anthropic API."""
    return [t.to_anthropic_format() for t in tools]


def format_tools_for_openai(tools: list[ToolDefinition]) -> list[dict]:
    """Format tools for OpenAI API."""
    return [t.to_openai_format() for t in tools]


def validate_tool_call(
    tool_name: str,
    args: dict[str, Any],
    tool: Optional[ToolDefinition] = None,
) -> tuple[bool, list[str]]:
    """
    Validate a tool call against its schema.

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    if tool is None:
        tool = get_tool_by_name(tool_name)

    if tool is None:
        return False, [f"Unknown tool: {tool_name}"]

    errors = []

    # Check required parameters
    for param in tool.required_params:
        if param not in args:
            errors.append(f"Missing required parameter: {param}")

    # Basic type validation
    for param_name, param_value in args.items():
        if param_name not in tool.parameters:
            errors.append(f"Unknown parameter: {param_name}")
            continue

        param_schema = tool.parameters[param_name]
        expected_type = param_schema.get("type")

        if expected_type == "string" and not isinstance(param_value, str):
            errors.append(f"Parameter {param_name} should be string, got {type(param_value).__name__}")
        elif expected_type == "integer" and not isinstance(param_value, int):
            errors.append(f"Parameter {param_name} should be integer, got {type(param_value).__name__}")
        elif expected_type == "boolean" and not isinstance(param_value, bool):
            errors.append(f"Parameter {param_name} should be boolean, got {type(param_value).__name__}")
        elif expected_type == "array" and not isinstance(param_value, list):
            errors.append(f"Parameter {param_name} should be array, got {type(param_value).__name__}")
        elif expected_type == "object" and not isinstance(param_value, dict):
            errors.append(f"Parameter {param_name} should be object, got {type(param_value).__name__}")

    return len(errors) == 0, errors
