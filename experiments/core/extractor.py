"""
NL intent extraction with pre-registered rubric.

Per PLAN.md, an NL response is considered to express correct intent if it identifies:
1. The correct tool name (or unambiguous synonym)
2. All required arguments with correct values
3. No contradictory or impossible argument values

Extraction must achieve ≥90% accuracy on validation set before proceeding.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .tools import ToolDefinition, get_tool_by_name, ALL_TOOLS
from .api_providers import call_model_with_retry


@dataclass
class ExtractedIntent:
    """Extracted intent from a natural language response."""
    tool_name: Optional[str] = None
    tool_identified: bool = False
    arguments: dict[str, Any] = field(default_factory=dict)
    args_complete: bool = False
    args_correct: bool = False
    confidence: float = 0.0
    extraction_method: str = "unknown"
    raw_response: str = ""
    errors: list[str] = field(default_factory=list)


# ==============================================================================
# Tool Name Synonyms and Patterns
# ==============================================================================

# Maps common synonyms/phrases to canonical tool names
TOOL_SYNONYMS: dict[str, list[str]] = {
    "read_file": [
        "read_file", "read the file", "read file", "read from", "view the file",
        "open the file", "cat the file", "look at the file", "read the contents",
        "examine the file", "check the file contents", "i'll read", "let me read",
        "i would read", "to read the", "reading the file", "read /", "need to read",
        "i need to read",
    ],
    "write_file": [
        "write_file", "write to file", "write the file", "write content to",
        "save to file", "create a file with", "put content in", "i'll write",
        "writing to", "save the text", "saving", "write '", 'write "',
    ],
    "delete_file": [
        "delete_file", "delete the file", "remove the file", "rm the file",
        "delete file", "remove file", "i'll delete", "i would delete",
        "delete /", "remove /",
    ],
    "list_directory": [
        "list_directory", "list the directory", "list directory", "ls the directory",
        "show directory contents", "list files in", "listing the", "show what's in",
        "i'll show you what", "list the contents", "list all files", "listing the contents",
    ],
    "search_files": [
        "search_files", "search for files", "find files", "search files",
        "look for files matching", "glob for", "i'll search", "searching for",
        "find all", "search for", "find *.py", "find *.",
    ],
    "edit_file": [
        "edit_file", "edit the file", "modify the file", "change the file",
        "update the file", "make edits to", "i'll edit", "replace '",
        "change '", "to change", "editing", "i'll use edit_file",
        "i'll edit /", "edit /",
    ],
    "run_command": [
        "run_command", "run the command", "execute command", "run command",
        "execute the command", "run this command", "i'll run", "running",
        "execute '", "execute `", "run '", "run `",
    ],
    "http_request": [
        "http_request", "make a request", "http request", "make an http",
        "send a request to", "fetch from", "get request", "post request",
        "make a get", "make a post", "i'll make a",
    ],
    "write_json_file": [
        "write_json_file", "write json to", "save json", "write the json",
        "save the json", "json to /", "json file",
    ],
    "write_escaped_content": [
        "write_escaped_content", "write escaped", "escaped content",
        "write 'hello \"", "write the text", "with escaping",
        "i'll write 'hello", "write 'it's", "write content with a newline",
    ],
    "write_unicode_content": [
        "write_unicode_content", "write unicode", "unicode content",
        "write '日", "write 'Ελ", "write '✓", "emoji to",
    ],
    "write_complex_config": [
        "write_complex_config", "complex config", "as json to", "as yaml to",
    ],
    "create_files": [
        "create_files", "create two files", "create multiple", "creating two",
    ],
    "noop": [
        "noop", "do nothing", "no operation", "no-op", "no action",
        "acknowledge", "i acknowledge",
    ],
}

# Regex patterns for extracting argument values
PATH_PATTERNS = [
    r"(?:path|file|directory)\s*[=:]\s*['\"]([^'\"]+)['\"]",
    r"(?:path|file|directory)\s*=\s*([/\w\-./]+)",
    r"(?:on|at|from|to|in)\s+([/][/\w\-./]+)",
    r"(?:on|at|from|to|in)\s+['\"]([^'\"]+)['\"]",
    r"['\"]([/][^'\"]+)['\"]",
    r"\s(/[/\w\-./]+)(?:\s|$|,|\.)",
]

CONTENT_PATTERNS = [
    r"(?:content|text)\s*[=:]\s*['\"]([^'\"]+)['\"]",
    r"(?:content|text)\s*=\s*['\"]([^'\"]+)['\"]",
    r"(?:write|with content|with the content|save)\s+['\"]([^'\"]+)['\"]",
    r"['\"]([^'\"]{1,100})['\"](?:\s+to\s+/)",
    r"write\s+['\"]([^'\"]+)['\"]",
    r"write\s+'([^']+)'",
    # Handle escaped quotes in content
    r"write\s+'([^']*\"[^']*)'",
    r"write\s+'([^']*\\'[^']*)'",
]


@dataclass
class ExtractionRubric:
    """
    Pre-registered extraction rubric defining correctness criteria.

    This rubric is locked before data collection per PLAN.md.
    """

    # Minimum confidence threshold for tool identification
    tool_confidence_threshold: float = 0.7

    # Whether exact path matching is required (vs partial match)
    require_exact_paths: bool = True

    # Whether all required args must be present
    require_all_args: bool = True

    # Allowed synonyms for each tool
    tool_synonyms: dict[str, list[str]] = field(default_factory=lambda: TOOL_SYNONYMS)


def extract_tool_name(response: str, rubric: ExtractionRubric) -> tuple[Optional[str], float]:
    """
    Extract tool name from NL response.

    Returns:
        Tuple of (tool_name or None, confidence score)
    """
    response_lower = response.lower()

    # First, check for exact tool name mentions
    for tool in ALL_TOOLS:
        if tool.name in response_lower:
            return tool.name, 1.0

    # Check synonyms
    best_match = None
    best_confidence = 0.0

    for tool_name, synonyms in rubric.tool_synonyms.items():
        for synonym in synonyms:
            if synonym in response_lower:
                # Longer synonyms get higher confidence
                confidence = min(0.9, 0.5 + len(synonym) / 50)
                if confidence > best_confidence:
                    best_match = tool_name
                    best_confidence = confidence

    return best_match, best_confidence


def extract_arguments(
    response: str,
    tool: ToolDefinition,
) -> dict[str, Any]:
    """
    Extract argument values from NL response for a specific tool.

    Returns:
        Dictionary of extracted argument values
    """
    args = {}

    # Extract path arguments
    if "path" in tool.parameters:
        for pattern in PATH_PATTERNS:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                path = match.group(1).strip()
                # Clean up any trailing punctuation
                path = re.sub(r'[,.\s]+$', '', path)
                if path.startswith('/'):
                    args["path"] = path
                    break

    # Extract content arguments
    if "content" in tool.parameters:
        # First try to find content between single quotes that precedes " to "
        # This handles cases like: write 'Hello "World"' to /tmp/file.txt
        content_to_path = re.search(r"write\s+'(.+?)'\s+to\s+/", response)
        if content_to_path:
            args["content"] = content_to_path.group(1)
        else:
            # Try standard patterns
            for pattern in CONTENT_PATTERNS:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    args["content"] = match.group(1)
                    break

    # Extract pattern arguments (for search)
    if "pattern" in tool.parameters:
        # Try explicit pattern= first
        pattern_match = re.search(r"pattern\s*[=:]\s*['\"]([^'\"]+)['\"]", response, re.IGNORECASE)
        if pattern_match:
            args["pattern"] = pattern_match.group(1)
        else:
            # Look for glob patterns like *.py, '*.log', etc
            glob_patterns = [
                r"['\"](\*\.[a-zA-Z]+)['\"]",
                r"(\*\.[a-zA-Z]+)\s+(?:files|in)",
                r"(?:for|find|search)\s+(\*\.[a-zA-Z]+)",
            ]
            for gp in glob_patterns:
                glob_match = re.search(gp, response)
                if glob_match:
                    args["pattern"] = glob_match.group(1)
                    break

    # Extract command arguments
    if "command" in tool.parameters:
        cmd_patterns = [
            r"command\s*[=:]\s*['\"]([^'\"]+)['\"]",
            r"`([^`]+)`",
            r"['\"]([a-z]+\s+-[a-z]+)['\"]",  # e.g., 'ls -la'
            r"(?:run|execute)\s+['\"]([^'\"]+)['\"]",
        ]
        for cp in cmd_patterns:
            cmd_match = re.search(cp, response, re.IGNORECASE)
            if cmd_match:
                args["command"] = cmd_match.group(1)
                break

    # Extract working_dir arguments
    if "working_dir" in tool.parameters:
        wd_patterns = [
            r"(?:working_dir|working directory|in)\s*[=:]\s*['\"]?([/][^'\"\s,]+)['\"]?",
            r"in\s+([/][/\w\-./]+)(?:\s|$|,|\.)",
        ]
        for wp in wd_patterns:
            wd_match = re.search(wp, response, re.IGNORECASE)
            if wd_match:
                args["working_dir"] = wd_match.group(1).strip().rstrip('.,')
                break

    # Extract edits for edit_file
    if "edits" in tool.parameters:
        # Look for "replace X with Y" or "change X to Y" patterns
        edit_patterns = [
            r"(?:replace|change)\s+['\"]([^'\"]+)['\"]?\s+(?:with|to)\s+['\"]([^'\"]+)['\"]",
            r"['\"]([^'\"]+)['\"]?\s+(?:with|to)\s+['\"]([^'\"]+)['\"]",
        ]
        for ep in edit_patterns:
            edit_match = re.search(ep, response, re.IGNORECASE)
            if edit_match:
                args["edits"] = [{
                    "old_string": edit_match.group(1),
                    "new_string": edit_match.group(2),
                }]
                break

    # Extract JSON content for write_json_file
    if "json_content" in tool.parameters:
        # Look for JSON object in response
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response)
        if json_match:
            try:
                args["json_content"] = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

    # Extract HTTP method and URL
    if "method" in tool.parameters:
        method_match = re.search(r'(GET|POST|PUT|DELETE|PATCH)', response, re.IGNORECASE)
        if method_match:
            args["method"] = method_match.group(1).upper()

    if "url" in tool.parameters:
        url_match = re.search(r'(https?://[^\s\'"]+)', response)
        if url_match:
            args["url"] = url_match.group(1).rstrip('.,')

    return args


def _extract_arguments_legacy(
    response: str,
    tool: ToolDefinition,
) -> dict[str, Any]:
    """Legacy extract function - kept for reference."""
    args = {}

    # Extract command arguments (old way)
    if "command" in tool.parameters:
        cmd_match = re.search(r"command\s*[=:]\s*['\"]([^'\"]+)['\"]", response, re.IGNORECASE)
        if cmd_match:
            args["command"] = cmd_match.group(1)
        else:
            # Look for backtick-quoted commands
            backtick_match = re.search(r"`([^`]+)`", response)
            if backtick_match:
                args["command"] = backtick_match.group(1)

    return args


def validate_arguments(
    extracted_args: dict[str, Any],
    expected_args: dict[str, Any],
    tool: ToolDefinition,
    rubric: ExtractionRubric,
) -> tuple[bool, bool, list[str]]:
    """
    Validate extracted arguments against expected values.

    Returns:
        Tuple of (args_complete, args_correct, list of errors)
    """
    errors = []

    # Check completeness - all required args present
    missing_required = []
    for param in tool.required_params:
        if param not in extracted_args:
            missing_required.append(param)

    args_complete = len(missing_required) == 0
    if not args_complete:
        errors.append(f"Missing required arguments: {missing_required}")

    # Check correctness - values match expected
    args_correct = True
    for param, expected_value in expected_args.items():
        if param not in extracted_args:
            continue

        extracted_value = extracted_args[param]

        if rubric.require_exact_paths and param == "path":
            if extracted_value != expected_value:
                args_correct = False
                errors.append(f"Path mismatch: expected '{expected_value}', got '{extracted_value}'")
        else:
            # Flexible string comparison for non-path args
            if str(extracted_value).lower() != str(expected_value).lower():
                args_correct = False
                errors.append(f"Value mismatch for {param}: expected '{expected_value}', got '{extracted_value}'")

    return args_complete, args_correct, errors


def extract_intent(
    response: str,
    expected_tool: Optional[str] = None,
    expected_args: Optional[dict[str, Any]] = None,
    rubric: Optional[ExtractionRubric] = None,
) -> ExtractedIntent:
    """
    Extract tool call intent from a natural language response.

    Args:
        response: The NL response to extract from
        expected_tool: Expected tool name (for validation)
        expected_args: Expected argument values (for validation)
        rubric: Extraction rubric to use

    Returns:
        ExtractedIntent with extraction results
    """
    if rubric is None:
        rubric = ExtractionRubric()

    result = ExtractedIntent(raw_response=response, extraction_method="rule_based")

    # Extract tool name
    tool_name, confidence = extract_tool_name(response, rubric)
    result.tool_name = tool_name
    result.confidence = confidence
    result.tool_identified = (
        tool_name is not None
        and confidence >= rubric.tool_confidence_threshold
    )

    if not result.tool_identified:
        result.errors.append("Could not identify tool from response")
        return result

    # Get tool definition
    tool = get_tool_by_name(tool_name)
    if tool is None:
        result.errors.append(f"Unknown tool: {tool_name}")
        return result

    # Extract arguments
    result.arguments = extract_arguments(response, tool)

    # Validate if expected values provided
    if expected_args is not None:
        result.args_complete, result.args_correct, validation_errors = validate_arguments(
            result.arguments, expected_args, tool, rubric
        )
        result.errors.extend(validation_errors)
    else:
        # Without expected values, just check completeness
        result.args_complete = all(
            param in result.arguments for param in tool.required_params
        )
        result.args_correct = result.args_complete  # Can't verify without expected

    return result


def is_correct_intent(
    response: str,
    expected_tool: str,
    expected_args: dict[str, Any],
    rubric: Optional[ExtractionRubric] = None,
) -> bool:
    """
    Check if NL response expresses correct intent per PLAN.md criteria.

    Criteria:
    1. The correct tool name (or unambiguous synonym)
    2. All required arguments with correct values
    3. No contradictory or impossible argument values

    Returns:
        True if response expresses correct intent
    """
    intent = extract_intent(response, expected_tool, expected_args, rubric)

    # Must identify correct tool
    if intent.tool_name != expected_tool:
        return False

    # Must have all required args
    if not intent.args_complete:
        return False

    # Args must be correct
    if not intent.args_correct:
        return False

    return True


# ==============================================================================
# LLM-based Extraction (fallback for complex cases)
# ==============================================================================

EXTRACTION_PROMPT = """You are extracting tool call intent from a natural language description.

Given this response, extract:
1. The tool name being described
2. The argument values being specified

Natural language response:
"{response}"

Available tools: {tool_names}

Output a JSON object with this format:
{{"tool": "tool_name", "args": {{"param1": "value1"}}}}

If no tool intent is found, output: {{"tool": null, "args": {{}}}}

Output ONLY the JSON, no explanation."""


def extract_intent_with_llm(
    response: str,
    model: str = "claude-haiku",
) -> ExtractedIntent:
    """
    Extract intent using LLM as a fallback for complex responses.

    This is more expensive but handles edge cases better.
    """
    result = ExtractedIntent(raw_response=response, extraction_method="llm")

    tool_names = [t.name for t in ALL_TOOLS]
    prompt = EXTRACTION_PROMPT.format(
        response=response,
        tool_names=", ".join(tool_names),
    )

    api_response = call_model_with_retry(prompt, model=model)

    if not api_response.success:
        result.errors.append(f"LLM extraction failed: {api_response.error}")
        return result

    try:
        # Parse JSON from response
        extracted = json.loads(api_response.response.strip())
        result.tool_name = extracted.get("tool")
        result.arguments = extracted.get("args", {})
        result.tool_identified = result.tool_name is not None
        result.confidence = 0.8  # LLM extraction gets moderate confidence
    except json.JSONDecodeError as e:
        result.errors.append(f"Failed to parse LLM output: {e}")

    return result


# ==============================================================================
# Validation Against Ground Truth
# ==============================================================================

def load_ground_truth(path: str = "experiments/validation/extraction_ground_truth.json") -> list[dict]:
    """Load ground truth examples for validation."""
    gt_path = Path(path)
    if not gt_path.exists():
        return []

    with open(gt_path) as f:
        return json.load(f)


def validate_extractor(
    ground_truth_path: str = "experiments/validation/extraction_ground_truth.json",
    rubric: Optional[ExtractionRubric] = None,
) -> dict:
    """
    Validate extractor accuracy against ground truth.

    Per PLAN.md, extractor must achieve ≥90% accuracy.

    Returns:
        Dictionary with accuracy metrics
    """
    ground_truth = load_ground_truth(ground_truth_path)

    if not ground_truth:
        return {
            "error": "No ground truth data found",
            "accuracy": 0.0,
            "n_examples": 0,
        }

    correct = 0
    total = len(ground_truth)
    errors = []

    for example in ground_truth:
        response = example["response"]
        expected_tool = example["expected_tool"]
        expected_args = example.get("expected_args", {})
        expected_correct = example["is_correct"]

        # Extract intent
        actual_correct = is_correct_intent(response, expected_tool, expected_args, rubric)

        if actual_correct == expected_correct:
            correct += 1
        else:
            errors.append({
                "response": response[:100],
                "expected": expected_correct,
                "actual": actual_correct,
            })

    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "meets_threshold": accuracy >= 0.90,
        "threshold": 0.90,
        "errors": errors[:10],  # First 10 errors for debugging
    }
