"""
Evaluation framework for format friction experiments.

Provides multiple evaluation modes with equivalent standards for both conditions:

1. STRICT: Exact tool name + exact argument values
   - JSON: Parsed tool/args must match exactly
   - NL: Extracted tool/args must match exactly

2. INTENT: Correct tool family + key values referenced
   - JSON: Tool in same family + required values present
   - NL: Tool mentioned/implied + required values mentioned

3. FUNCTIONAL: Would the call achieve the intended goal?
   - JSON: Tool could accomplish the task + args are valid
   - NL: Description indicates correct action would be taken

Academic rigor requires:
- Same evaluation standard applied to both conditions
- Clear operationalization of "correct"
- Reporting results under multiple evaluation modes
- Transparency about what each mode measures
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .tools import ToolDefinition, get_tool_by_name


class EvaluationMode(Enum):
    """Evaluation strictness modes."""
    STRICT = "strict"          # Exact match required
    INTENT = "intent"          # Intent signals sufficient
    FUNCTIONAL = "functional"  # Functional equivalence accepted


# Tool equivalence classes - tools that can accomplish the same tasks
TOOL_EQUIVALENCE = {
    # File writing tools
    "write_file": {"write_file", "write_escaped_content", "write_unicode_content"},
    "write_escaped_content": {"write_file", "write_escaped_content", "write_unicode_content"},
    "write_unicode_content": {"write_file", "write_escaped_content", "write_unicode_content"},

    # JSON/config writing tools
    "write_json_file": {"write_json_file", "write_file", "write_complex_config"},
    "write_complex_config": {"write_complex_config", "write_json_file", "write_file"},

    # These have no equivalents
    "read_file": {"read_file"},
    "edit_file": {"edit_file"},
    "search_files": {"search_files"},
    "list_directory": {"list_directory"},
    "run_command": {"run_command"},
    "http_request": {"http_request"},
    "noop": {"noop"},
}


@dataclass
class EvaluationResult:
    """Result of evaluating a response."""
    mode: str
    is_correct: bool

    # Detailed checks
    tool_correct: bool = False
    tool_equivalent: bool = False  # For functional mode
    args_present: bool = False
    args_correct: bool = False

    # Signal detection (same for NL and JSON) - NEW
    signal_detected: bool = False      # Did model signal intent in text?
    signal_tool_correct: bool = False  # Was the signaled tool correct?
    signal_args_present: bool = False  # Were signaled args present?

    # JSON-specific checks - NEW
    json_attempted: bool = False       # Did response contain JSON-like structure?
    json_valid: bool = False           # Was JSON syntactically valid?
    json_schema_correct: bool = False  # Did JSON match expected schema?

    # Format friction flag - NEW
    format_friction: bool = False      # Signal correct but JSON failed

    # Metadata
    expected_tool: str = ""
    actual_tool: Optional[str] = None
    confidence: float = 0.0
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def normalize_for_comparison(value: Any) -> str:
    """Normalize a value for fuzzy comparison."""
    if isinstance(value, str):
        # Normalize escape sequences
        s = value.replace("\\'", "'").replace('\\"', '"')
        s = s.replace("\\\\", "\\")
        s = s.replace("\\n", "\n").replace("\\t", "\t")
        return s.lower().strip()
    elif isinstance(value, dict):
        return json.dumps(value, sort_keys=True).lower()
    elif isinstance(value, (list, tuple)):
        return json.dumps(list(value), sort_keys=True).lower()
    elif isinstance(value, bool):
        return "true" if value else "false"
    else:
        return str(value).lower()


def values_equivalent(expected: Any, actual: Any) -> bool:
    """Check if two values are functionally equivalent."""
    if expected == actual:
        return True

    # Normalize and compare
    norm_expected = normalize_for_comparison(expected)
    norm_actual = normalize_for_comparison(actual)

    if norm_expected == norm_actual:
        return True

    # For strings, check if one contains the other (for partial matches)
    if isinstance(expected, str) and isinstance(actual, str):
        if norm_expected in norm_actual or norm_actual in norm_expected:
            return True

    return False


def value_mentioned_in_text(value: Any, text: str) -> bool:
    """Check if a value is mentioned in natural language text."""
    text_lower = text.lower()
    text_normalized = normalize_for_comparison(text)

    if isinstance(value, str):
        value_normalized = normalize_for_comparison(value)

        # Direct match
        if value.lower() in text_lower:
            return True

        # Normalized match
        if value_normalized in text_normalized:
            return True

        # Quoted match (various styles)
        for quote in ["'", '"', "`"]:
            if f"{quote}{value}{quote}" in text:
                return True
            # Escaped version
            escaped = value.replace("'", "\\'").replace('"', '\\"')
            if f"{quote}{escaped}{quote}" in text:
                return True

        # Path filename
        if "/" in value:
            filename = value.split("/")[-1]
            if filename.lower() in text_lower:
                return True

        # Doubled backslashes
        if "\\" in value:
            doubled = value.replace("\\", "\\\\")
            if doubled.lower() in text_lower:
                return True

        return False

    elif isinstance(value, dict):
        # Check if key-value pairs are mentioned
        matches = 0
        for k, v in value.items():
            if k.lower() in text_lower and value_mentioned_in_text(v, text):
                matches += 1
        return matches >= len(value) * 0.5

    elif isinstance(value, (list, tuple)):
        matches = sum(1 for v in value if value_mentioned_in_text(v, text))
        return matches >= len(value) * 0.5

    elif isinstance(value, bool):
        return ("true" if value else "false") in text_lower

    else:
        return str(value) in text


def tools_are_equivalent(expected: str, actual: str) -> bool:
    """Check if two tools are functionally equivalent."""
    if expected == actual:
        return True

    equivalents = TOOL_EQUIVALENCE.get(expected, {expected})
    return actual in equivalents


# Tool name patterns for NL detection
TOOL_PATTERNS = {
    "read_file": [r"\bread[_\s]?file\b", r"\breading\b.*file", r"\bopen\b.*file", r"\bcat\b", r"\bview\b.*file"],
    "write_file": [r"\bwrite[_\s]?file\b", r"\bwriting\b.*file", r"\bcreate\b.*file", r"\bsave\b.*file"],
    "write_escaped_content": [r"\bwrite[_\s]?escaped", r"\bescaped[_\s]?content\b"],
    "write_unicode_content": [r"\bwrite[_\s]?unicode", r"\bunicode[_\s]?content\b"],
    "write_json_file": [r"\bwrite[_\s]?json", r"\bjson[_\s]?file\b", r"\bsave\b.*json"],
    "write_complex_config": [r"\bwrite[_\s]?complex", r"\bcomplex[_\s]?config\b"],
    "edit_file": [r"\bedit[_\s]?file\b", r"\bediting\b.*file", r"\bmodify\b.*file"],
    "search_files": [r"\bsearch[_\s]?files?\b", r"\bfind\b.*files?", r"\bgrep\b"],
    "list_directory": [r"\blist[_\s]?dir", r"\bls\b", r"\bdir\b", r"\bdirectory\b.*list"],
    "run_command": [r"\brun[_\s]?command\b", r"\bexecute\b", r"\bshell\b", r"\bbash\b"],
    "http_request": [r"\bhttp[_\s]?request\b", r"\bfetch\b", r"\bapi\b.*call", r"\brequest\b.*url"],
    "noop": [r"\bnoop\b", r"\bno[_\s-]?op\b", r"\bno\s+action\b", r"\backnowledge\b", r"\bnothing\b"],
}


def detect_tool_in_text(text: str, expected_tool: str) -> tuple[bool, bool, Optional[str]]:
    """
    Detect if a tool is mentioned in text.

    Returns:
        (exact_match, equivalent_match, detected_tool_name)
    """
    text_lower = text.lower()

    # Check for exact tool name
    if expected_tool.lower() in text_lower:
        return True, True, expected_tool

    # Check for tool name with underscores as spaces
    tool_spaced = expected_tool.replace("_", " ").lower()
    if tool_spaced in text_lower:
        return True, True, expected_tool

    # Check patterns for expected tool
    patterns = TOOL_PATTERNS.get(expected_tool, [])
    for pattern in patterns:
        if re.search(pattern, text_lower):
            return True, True, expected_tool

    # Check for equivalent tools
    equivalents = TOOL_EQUIVALENCE.get(expected_tool, set())
    for equiv_tool in equivalents:
        if equiv_tool == expected_tool:
            continue
        if equiv_tool.lower() in text_lower:
            return False, True, equiv_tool
        equiv_spaced = equiv_tool.replace("_", " ").lower()
        if equiv_spaced in text_lower:
            return False, True, equiv_tool
        # Check patterns
        equiv_patterns = TOOL_PATTERNS.get(equiv_tool, [])
        for pattern in equiv_patterns:
            if re.search(pattern, text_lower):
                return False, True, equiv_tool

    return False, False, None


def parse_json_tool_call(response: str) -> tuple[Optional[str], Optional[dict], list[str]]:
    """
    Parse a JSON tool call from response.

    Returns:
        (tool_name, args_dict, errors)
    """
    errors = []
    response = response.strip()

    # Try direct JSON parse
    parsed = None
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try code block
    if parsed is None:
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1).strip())
            except json.JSONDecodeError as e:
                errors.append(f"JSON in code block invalid: {e}")

    # Try embedded JSON
    if parsed is None:
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError as e:
                errors.append(f"Embedded JSON invalid: {e}")

    if parsed is None:
        errors.append("Could not parse JSON from response")
        return None, None, errors

    if not isinstance(parsed, dict):
        errors.append("JSON is not an object")
        return None, None, errors

    tool_name = parsed.get("tool")
    args = parsed.get("args", {})

    if tool_name is None:
        errors.append("Missing 'tool' key")

    return tool_name, args, errors


def evaluate_json_response(
    response: str,
    expected_tool: str,
    expected_args: dict[str, Any],
    mode: EvaluationMode,
) -> EvaluationResult:
    """
    Evaluate a JSON-formatted response.

    This function checks BOTH:
    1. Signal detection: Did the model indicate correct intent (in text OR JSON)?
    2. JSON compliance: Did the model produce valid, correctly-structured JSON?

    Format Friction = signal_detected AND NOT json_schema_correct
    """
    result = EvaluationResult(
        mode=mode.value,
        is_correct=False,
        expected_tool=expected_tool,
    )

    # STEP 1: Check for signal in text (same logic as NL evaluation)
    # This detects if the model INDICATED the correct intent, regardless of JSON
    exact_match, equiv_match, detected_tool = detect_tool_in_text(response, expected_tool)
    result.signal_tool_correct = exact_match or equiv_match

    # Check if args are mentioned in text
    args_mentioned = 0
    for param, expected_value in expected_args.items():
        if value_mentioned_in_text(expected_value, response):
            args_mentioned += 1
    result.signal_args_present = args_mentioned >= len(expected_args) * 0.5 if expected_args else True

    # Overall signal detection
    result.signal_detected = result.signal_tool_correct and result.signal_args_present

    # STEP 2: Parse JSON
    actual_tool, actual_args, parse_errors = parse_json_tool_call(response)

    # Track JSON-specific results
    result.json_attempted = bool(re.search(r'\{.*\}', response, re.DOTALL))
    result.json_valid = actual_tool is not None  # Parsed successfully

    if not result.json_valid:
        result.errors.extend(parse_errors)
        # Check for format friction: signal present but JSON failed
        result.format_friction = result.signal_detected
        if result.format_friction:
            result.notes.append("FORMAT FRICTION: Model signaled correct intent but failed to produce valid JSON")
        return result

    result.actual_tool = actual_tool

    # STEP 3: Check JSON tool correctness
    result.tool_correct = (actual_tool == expected_tool)
    result.tool_equivalent = tools_are_equivalent(expected_tool, actual_tool)

    if not result.tool_correct:
        if result.tool_equivalent:
            result.notes.append(f"Used equivalent tool '{actual_tool}' instead of '{expected_tool}'")
        else:
            result.errors.append(f"Wrong tool: expected '{expected_tool}', got '{actual_tool}'")

    # STEP 4: Check JSON args
    if actual_args is None:
        actual_args = {}

    tool_def = get_tool_by_name(expected_tool)
    required_params = tool_def.required_params if tool_def else list(expected_args.keys())

    # Args presence
    missing = [p for p in required_params if p not in actual_args]
    result.args_present = len(missing) == 0
    if missing:
        result.errors.append(f"Missing required args: {missing}")

    # Args correctness
    result.args_correct = True
    for param, expected_value in expected_args.items():
        actual_value = actual_args.get(param)

        if mode == EvaluationMode.STRICT:
            if actual_value != expected_value:
                result.args_correct = False
                result.errors.append(f"Arg '{param}': expected {repr(expected_value)}, got {repr(actual_value)}")
        else:
            # INTENT or FUNCTIONAL: allow equivalent values
            if not values_equivalent(expected_value, actual_value):
                result.args_correct = False
                result.errors.append(f"Arg '{param}': expected {repr(expected_value)}, got {repr(actual_value)}")

    # STEP 5: Determine JSON schema correctness
    if mode == EvaluationMode.STRICT:
        result.json_schema_correct = result.tool_correct and result.args_present and result.args_correct
    elif mode == EvaluationMode.INTENT:
        result.json_schema_correct = result.tool_correct and result.args_present and result.args_correct
    else:  # FUNCTIONAL
        result.json_schema_correct = result.tool_equivalent and result.args_present and result.args_correct

    # STEP 6: Overall correctness = valid JSON with correct schema
    result.is_correct = result.json_valid and result.json_schema_correct

    # STEP 7: Check for format friction
    # Format friction = model signaled correct intent BUT JSON schema is wrong
    result.format_friction = result.signal_detected and not result.json_schema_correct
    if result.format_friction:
        result.notes.append("FORMAT FRICTION: Model signaled correct intent but JSON schema incorrect")

    result.confidence = 1.0 if result.is_correct else 0.0
    return result


def evaluate_nl_response(
    response: str,
    expected_tool: str,
    expected_args: dict[str, Any],
    mode: EvaluationMode,
) -> EvaluationResult:
    """
    Evaluate a natural language response.

    For NL condition, we only check signal detection (did model indicate correct intent).
    There's no JSON to validate, so format_friction is always False.
    """
    result = EvaluationResult(
        mode=mode.value,
        is_correct=False,
        expected_tool=expected_tool,
    )

    if not response.strip():
        result.errors.append("Empty response")
        return result

    # Check for refusal
    refusal_phrases = ["i cannot", "i can't", "i won't", "unable to", "not possible", "refuse"]
    if any(phrase in response.lower() for phrase in refusal_phrases):
        result.errors.append("Response contains refusal")
        return result

    # Detect tool (same logic used for both NL and JSON signal detection)
    exact_match, equiv_match, detected_tool = detect_tool_in_text(response, expected_tool)
    result.tool_correct = exact_match
    result.tool_equivalent = equiv_match
    result.actual_tool = detected_tool

    # Set signal detection fields (same as JSON condition)
    result.signal_tool_correct = exact_match or equiv_match

    if not exact_match:
        if equiv_match:
            result.notes.append(f"Mentioned equivalent tool '{detected_tool}' instead of '{expected_tool}'")
        else:
            result.errors.append(f"Tool '{expected_tool}' not mentioned")

    # Check args mentioned
    tool_def = get_tool_by_name(expected_tool)
    required_params = tool_def.required_params if tool_def else list(expected_args.keys())

    args_found = 0
    args_correct_count = 0

    for param, expected_value in expected_args.items():
        is_required = param in required_params

        if value_mentioned_in_text(expected_value, response):
            args_found += 1
            args_correct_count += 1
        elif is_required:
            result.errors.append(f"Required arg '{param}' value not found")

    total_required = len([p for p in expected_args if p in required_params])
    result.args_present = args_found >= total_required if total_required > 0 else True
    result.args_correct = args_correct_count >= len(expected_args) * 0.8

    # Set signal detection fields
    result.signal_args_present = result.args_present
    result.signal_detected = result.signal_tool_correct and result.signal_args_present

    # NL has no JSON requirement, so these are N/A
    result.json_attempted = False
    result.json_valid = False  # N/A for NL
    result.json_schema_correct = False  # N/A for NL
    result.format_friction = False  # No format friction possible in NL condition

    # Overall correctness based on mode (same as before)
    if mode == EvaluationMode.STRICT:
        result.is_correct = result.tool_correct and result.args_present and result.args_correct
    elif mode == EvaluationMode.INTENT:
        result.is_correct = result.tool_correct and result.args_present
    else:  # FUNCTIONAL
        result.is_correct = result.tool_equivalent and result.args_present

    result.confidence = (args_correct_count / len(expected_args)) if expected_args else 1.0
    return result


def evaluate_response(
    response: str,
    expected_tool: str,
    expected_args: dict[str, Any],
    is_json_condition: bool,
    mode: EvaluationMode = EvaluationMode.INTENT,
) -> EvaluationResult:
    """
    Unified evaluation interface.

    Args:
        response: Model response to evaluate
        expected_tool: Expected tool name
        expected_args: Expected argument values
        is_json_condition: True for JSON-only, False for NL-only
        mode: Evaluation strictness mode

    Returns:
        EvaluationResult with detailed judgment
    """
    if is_json_condition:
        return evaluate_json_response(response, expected_tool, expected_args, mode)
    else:
        return evaluate_nl_response(response, expected_tool, expected_args, mode)


def evaluate_trial_all_modes(
    response: str,
    expected_tool: str,
    expected_args: dict[str, Any],
    is_json_condition: bool,
) -> dict[str, EvaluationResult]:
    """
    Evaluate a trial under all evaluation modes.

    Returns dict mapping mode name to result.
    """
    return {
        mode.value: evaluate_response(
            response, expected_tool, expected_args, is_json_condition, mode
        )
        for mode in EvaluationMode
    }
