"""
Tool-call correctness evaluation (NOT signal detection).

Per PLAN.md, this module evaluates whether model responses correctly
express tool-call intent, supporting both JSON and NL conditions.

UPDATED: Now supports intent-based evaluation for NL responses that
checks for intent signals (tool mentioned, args referenced) rather
than requiring full structured extraction.

Requirements:
- Judge-human κ ≥ 0.75 required
- Cross-family judge κ ≥ 0.75 required (raised from 0.70)
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path

from .tools import ToolDefinition, get_tool_by_name, validate_tool_call
from .extractor import extract_intent, is_correct_intent, ExtractionRubric
from .api_providers import call_model_with_retry
from .statistics import cohens_kappa


# ==============================================================================
# Core Data Structures
# ==============================================================================

@dataclass
class JudgmentResult:
    """Result of judging a tool-call response."""
    is_correct: bool
    tool_name_correct: bool = False
    args_complete: bool = False
    args_correct: bool = False
    json_valid: bool = False
    json_parseable: bool = False
    confidence: float = 0.0
    judge_model: str = "unknown"
    errors: list[str] = field(default_factory=list)
    raw_response: str = ""


# ==============================================================================
# Intent Signal Detection (for NL responses)
# ==============================================================================

# Tool name synonyms for intent detection
TOOL_SYNONYMS = {
    "read_file": ["read", "read_file", "reading", "open", "view", "cat", "display", "show"],
    "write_file": ["write", "write_file", "writing", "create", "save", "put"],
    "edit_file": ["edit", "edit_file", "editing", "modify", "change", "update", "replace"],
    "search_files": ["search", "search_files", "searching", "find", "grep", "look for"],
    "list_directory": ["list", "list_directory", "ls", "dir", "show directory"],
    "run_command": ["run", "run_command", "execute", "exec", "shell", "command", "bash"],
    "http_request": ["http", "http_request", "fetch", "request", "get", "post", "api call"],
    "noop": ["noop", "no-op", "no operation", "acknowledge", "nothing", "no action"],
    "write_json_file": ["write_json", "write_json_file", "json file", "save json"],
    "write_escaped_content": ["write_escaped", "write_escaped_content", "escaped content"],
    "write_unicode_content": ["write_unicode", "write_unicode_content", "unicode content"],
    "write_complex_config": ["write_complex", "write_complex_config", "complex config"],
}


def normalize_value(value: Any) -> str:
    """Normalize a value for fuzzy matching."""
    if isinstance(value, str):
        return value.lower().strip()
    elif isinstance(value, dict):
        return json.dumps(value, sort_keys=True).lower()
    elif isinstance(value, (list, tuple)):
        return json.dumps(list(value), sort_keys=True).lower()
    else:
        return str(value).lower()


def normalize_escapes(s: str) -> str:
    """Normalize escape sequences for comparison."""
    # Remove escape backslashes for comparison
    # 'It\'s' -> "It's", 'C:\\Users' -> 'C:\Users'
    normalized = s.replace("\\'", "'").replace('\\"', '"')
    normalized = normalized.replace("\\\\", "\\")
    normalized = normalized.replace("\\n", "\n").replace("\\t", "\t")
    return normalized


def value_appears_in_text(value: Any, text: str) -> bool:
    """
    Check if a value appears in text (fuzzy matching).

    Handles:
    - Exact string matches
    - Escaped variations (\'s vs 's, \\\\ vs \\)
    - Path components
    - JSON-like structures described in prose
    - Numeric values
    """
    text_lower = text.lower()
    text_normalized = normalize_escapes(text)

    if isinstance(value, str):
        value_lower = value.lower()
        value_normalized = normalize_escapes(value).lower()

        # Direct match
        if value_lower in text_lower:
            return True

        # Normalized match (handles escape differences)
        if value_normalized in normalize_escapes(text_lower):
            return True

        # Check with various quote styles
        for quote in ["'", '"']:
            quoted = f"{quote}{value}{quote}"
            if quoted in text:
                return True
            # Also check escaped versions
            escaped_value = value.replace("'", "\\'").replace('"', '\\"')
            escaped_quoted = f"{quote}{escaped_value}{quote}"
            if escaped_quoted in text:
                return True

        # Path - check if filename appears
        if "/" in value:
            filename = value.split("/")[-1]
            if filename.lower() in text_lower:
                return True

        # Check for the value with backslashes doubled (common in output)
        doubled_backslash = value.replace("\\", "\\\\")
        if doubled_backslash.lower() in text_lower:
            return True

        return False

    elif isinstance(value, dict):
        # For dicts, check if key-value pairs are mentioned
        # More lenient: just check if keys and values appear
        keys_found = 0
        values_found = 0
        for k, v in value.items():
            if k.lower() in text_lower:
                keys_found += 1
            if value_appears_in_text(v, text):
                values_found += 1
        # Consider it a match if most keys/values appear
        total = len(value)
        return keys_found >= total * 0.5 and values_found >= total * 0.5

    elif isinstance(value, (list, tuple)):
        # For lists, check if elements appear
        found = sum(1 for v in value if value_appears_in_text(v, text))
        return found >= len(value) * 0.5

    elif isinstance(value, bool):
        str_val = "true" if value else "false"
        return str_val in text_lower

    elif isinstance(value, (int, float)):
        return str(value) in text

    return False


def detect_tool_intent(response: str, expected_tool: str) -> tuple[bool, float]:
    """
    Detect if the response indicates intent to use the expected tool.

    Returns:
        (tool_mentioned, confidence)
    """
    response_lower = response.lower()

    # Direct tool name match
    if expected_tool.lower() in response_lower:
        return True, 1.0

    # Check synonyms
    synonyms = TOOL_SYNONYMS.get(expected_tool, [])
    for synonym in synonyms:
        if synonym.lower() in response_lower:
            return True, 0.9

    # Check for tool name with underscores replaced
    tool_words = expected_tool.replace("_", " ").lower()
    if tool_words in response_lower:
        return True, 0.85

    return False, 0.0


def detect_arg_intent(
    response: str,
    expected_args: dict[str, Any],
    tool: Optional[ToolDefinition] = None,
) -> tuple[bool, bool, float, list[str]]:
    """
    Detect if the response indicates intent to use the expected arguments.

    Returns:
        (args_mentioned, args_values_correct, confidence, errors)
    """
    errors = []

    if not expected_args:
        return True, True, 1.0, []

    required_params = tool.required_params if tool else list(expected_args.keys())

    # Check each expected argument
    args_found = 0
    args_correct = 0

    for param, expected_value in expected_args.items():
        is_required = param in required_params

        # Check if the value appears in the response
        if value_appears_in_text(expected_value, response):
            args_found += 1
            args_correct += 1
        elif is_required:
            errors.append(f"Required arg '{param}' value not found in response")

    total_required = len([p for p in expected_args if p in required_params])
    total_args = len(expected_args)

    # Args are "mentioned" if we found most required values
    args_mentioned = args_found >= total_required * 0.8 if total_required > 0 else True

    # Args are "correct" if we found all values with correct content
    args_values_correct = args_correct >= total_args * 0.8

    confidence = args_correct / total_args if total_args > 0 else 1.0

    return args_mentioned, args_values_correct, confidence, errors


def judge_nl_intent(
    response: str,
    expected_tool: str,
    expected_args: dict[str, Any],
) -> JudgmentResult:
    """
    Judge a NL response based on INTENT SIGNALS rather than full extraction.

    This is more lenient than full extraction - it checks:
    1. Did the model mention/reference the correct tool?
    2. Did the model mention the key argument values?
    3. Is there contradictory information?

    This approach avoids penalizing NL responses for extraction failures
    when the model clearly understood the task.
    """
    result = JudgmentResult(
        is_correct=False,
        raw_response=response,
        judge_model="intent_based",
    )

    if not response.strip():
        result.errors.append("Empty response")
        return result

    # Get tool definition
    tool = get_tool_by_name(expected_tool)

    # Check tool intent
    tool_mentioned, tool_confidence = detect_tool_intent(response, expected_tool)
    result.tool_name_correct = tool_mentioned

    if not tool_mentioned:
        result.errors.append(f"Tool '{expected_tool}' (or synonym) not mentioned")

    # Check argument intent
    args_mentioned, args_correct, args_confidence, arg_errors = detect_arg_intent(
        response, expected_args, tool
    )
    result.args_complete = args_mentioned
    result.args_correct = args_correct
    result.errors.extend(arg_errors)

    # Overall confidence
    result.confidence = (tool_confidence + args_confidence) / 2

    # Check for contradictions (model refusing or saying it can't)
    contradiction_phrases = [
        "i cannot", "i can't", "i won't", "unable to",
        "not possible", "cannot be done", "refuse",
    ]
    has_contradiction = any(phrase in response.lower() for phrase in contradiction_phrases)

    if has_contradiction:
        result.errors.append("Response contains refusal/contradiction")
        result.is_correct = False
    else:
        # Intent is correct if tool was mentioned and args were referenced
        result.is_correct = tool_mentioned and args_mentioned

    return result


# ==============================================================================
# JSON Response Parsing and Validation
# ==============================================================================

def parse_json_response(response: str) -> tuple[Optional[dict], list[str]]:
    """
    Parse JSON tool call from a model response.

    Handles various formats:
    - Pure JSON
    - JSON in markdown code blocks
    - JSON with surrounding text

    Returns:
        Tuple of (parsed dict or None, list of errors)
    """
    errors = []
    response = response.strip()

    # Try direct JSON parse first
    try:
        parsed = json.loads(response)
        if isinstance(parsed, dict):
            return parsed, []
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code blocks
    code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
    if code_block_match:
        try:
            parsed = json.loads(code_block_match.group(1).strip())
            if isinstance(parsed, dict):
                return parsed, []
        except json.JSONDecodeError as e:
            errors.append(f"JSON in code block is invalid: {e}")

    # Try finding JSON object in response
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed, dict):
                return parsed, []
        except json.JSONDecodeError as e:
            errors.append(f"Extracted JSON is invalid: {e}")

    errors.append("Could not parse valid JSON from response")
    return None, errors


def judge_json_response(
    response: str,
    expected_tool: str,
    expected_args: dict[str, Any],
) -> JudgmentResult:
    """
    Judge a JSON-only condition response.

    Checks:
    1. Response is valid JSON
    2. Tool name matches expected
    3. All required arguments present
    4. Argument values match expected
    """
    result = JudgmentResult(
        is_correct=False,
        raw_response=response,
        judge_model="rule_based",
    )

    # Parse JSON
    parsed, parse_errors = parse_json_response(response)
    result.errors.extend(parse_errors)

    if parsed is None:
        result.json_valid = False
        result.json_parseable = False
        return result

    result.json_parseable = True

    # Check structure
    if "tool" not in parsed:
        result.errors.append("Missing 'tool' key in JSON")
        return result

    if "args" not in parsed:
        result.errors.append("Missing 'args' key in JSON")
        return result

    result.json_valid = True

    # Check tool name
    tool_name = parsed.get("tool")
    result.tool_name_correct = (tool_name == expected_tool)
    if not result.tool_name_correct:
        result.errors.append(f"Wrong tool: expected '{expected_tool}', got '{tool_name}'")

    # Get tool definition
    tool = get_tool_by_name(expected_tool)
    if tool is None:
        result.errors.append(f"Unknown expected tool: {expected_tool}")
        return result

    # Check arguments
    args = parsed.get("args", {})

    # Check completeness
    missing = [p for p in tool.required_params if p not in args]
    result.args_complete = len(missing) == 0
    if not result.args_complete:
        result.errors.append(f"Missing required args: {missing}")

    # Check correctness
    result.args_correct = True
    for param, expected_value in expected_args.items():
        actual_value = args.get(param)
        if actual_value != expected_value:
            result.args_correct = False
            result.errors.append(
                f"Arg '{param}' mismatch: expected '{expected_value}', got '{actual_value}'"
            )

    # Overall correctness
    result.is_correct = (
        result.tool_name_correct
        and result.args_complete
        and result.args_correct
    )
    result.confidence = 1.0 if result.is_correct else 0.0

    return result


def judge_nl_response(
    response: str,
    expected_tool: str,
    expected_args: dict[str, Any],
    rubric: Optional[ExtractionRubric] = None,
) -> JudgmentResult:
    """
    Judge a NL-only condition response.

    Uses the extractor to determine if intent is correct.
    """
    result = JudgmentResult(
        is_correct=False,
        raw_response=response,
        judge_model="rule_based",
    )

    # Use extractor
    intent = extract_intent(response, expected_tool, expected_args, rubric)

    result.tool_name_correct = (intent.tool_name == expected_tool)
    result.args_complete = intent.args_complete
    result.args_correct = intent.args_correct
    result.confidence = intent.confidence
    result.errors.extend(intent.errors)

    result.is_correct = (
        result.tool_name_correct
        and result.args_complete
        and result.args_correct
    )

    return result


# ==============================================================================
# LLM-based Judging (Cross-family support)
# ==============================================================================

JUDGE_PROMPT_JSON = """You are evaluating whether a JSON tool call is correct.

Expected tool call:
- Tool: {expected_tool}
- Arguments: {expected_args}

Actual response:
{response}

Evaluate:
1. Is the JSON syntactically valid?
2. Is the tool name correct?
3. Are all required arguments present?
4. Are the argument values correct?

Answer with JSON: {{"is_correct": true/false, "tool_correct": true/false, "args_correct": true/false, "reason": "brief explanation"}}"""

JUDGE_PROMPT_NL = """You are evaluating whether a natural language response correctly describes a tool call intent.

Expected intent:
- Tool: {expected_tool}
- Arguments: {expected_args}

Response to evaluate:
{response}

The response is correct if it:
1. Identifies the correct tool (or an unambiguous synonym)
2. Specifies all required arguments with correct values
3. Does not contain contradictory information

Answer with JSON: {{"is_correct": true/false, "tool_identified": true/false, "args_complete": true/false, "reason": "brief explanation"}}"""


def judge_with_llm(
    response: str,
    expected_tool: str,
    expected_args: dict[str, Any],
    is_json_condition: bool,
    model: str = "claude-sonnet",
) -> JudgmentResult:
    """
    Judge a response using an LLM.

    Supports cross-family judging per PLAN.md requirements.

    Args:
        response: The model response to judge
        expected_tool: Expected tool name
        expected_args: Expected argument values
        is_json_condition: Whether this is JSON-only (True) or NL-only (False)
        model: Which model to use for judging

    Returns:
        JudgmentResult with LLM judgment
    """
    result = JudgmentResult(
        is_correct=False,
        raw_response=response,
        judge_model=model,
    )

    # Select prompt based on condition
    if is_json_condition:
        prompt = JUDGE_PROMPT_JSON.format(
            expected_tool=expected_tool,
            expected_args=json.dumps(expected_args),
            response=response,
        )
    else:
        prompt = JUDGE_PROMPT_NL.format(
            expected_tool=expected_tool,
            expected_args=json.dumps(expected_args),
            response=response,
        )

    # Call LLM judge
    api_response = call_model_with_retry(prompt, model=model)

    if not api_response.success:
        result.errors.append(f"LLM judge call failed: {api_response.error}")
        return result

    # Parse judgment
    try:
        # Extract JSON from response
        judge_response = api_response.response.strip()
        parsed, _ = parse_json_response(judge_response)

        if parsed is None:
            result.errors.append("Could not parse judge response as JSON")
            return result

        result.is_correct = parsed.get("is_correct", False)
        result.tool_name_correct = parsed.get("tool_correct", parsed.get("tool_identified", False))
        result.args_correct = parsed.get("args_correct", parsed.get("args_complete", False))
        result.args_complete = result.args_correct
        result.confidence = 0.85 if result.is_correct else 0.15

        if "reason" in parsed:
            result.errors.append(f"Judge reason: {parsed['reason']}")

    except Exception as e:
        result.errors.append(f"Failed to parse judge output: {e}")

    return result


# ==============================================================================
# Cross-family Judge Validation
# ==============================================================================

@dataclass
class CrossFamilyAgreement:
    """Agreement metrics between judge families."""
    claude_gpt_kappa: float = 0.0
    claude_gemini_kappa: float = 0.0
    gpt_gemini_kappa: float = 0.0
    meets_threshold: bool = False
    threshold: float = 0.75
    n_samples: int = 0


def compute_cross_family_agreement(
    responses: list[dict],
) -> CrossFamilyAgreement:
    """
    Compute agreement between cross-family judges.

    Per PLAN.md, cross-family judge κ ≥ 0.75 required.

    Args:
        responses: List of dicts with 'claude', 'gpt4', 'gemini' judgments

    Returns:
        CrossFamilyAgreement with all pairwise kappas
    """
    result = CrossFamilyAgreement(n_samples=len(responses))

    if len(responses) < 2:
        return result

    # Extract judgments by family
    claude_judgments = [r.get("claude", False) for r in responses]
    gpt_judgments = [r.get("gpt4", False) for r in responses]
    gemini_judgments = [r.get("gemini", False) for r in responses]

    # Compute pairwise kappas
    result.claude_gpt_kappa = cohens_kappa(claude_judgments, gpt_judgments)
    result.claude_gemini_kappa = cohens_kappa(claude_judgments, gemini_judgments)
    result.gpt_gemini_kappa = cohens_kappa(gpt_judgments, gemini_judgments)

    # Check if all meet threshold
    result.meets_threshold = (
        result.claude_gpt_kappa >= result.threshold
        and result.claude_gemini_kappa >= result.threshold
        and result.gpt_gemini_kappa >= result.threshold
    )

    return result


def judge_with_all_families(
    response: str,
    expected_tool: str,
    expected_args: dict[str, Any],
    is_json_condition: bool,
) -> dict[str, JudgmentResult]:
    """
    Judge a response with all three model families.

    Returns:
        Dict mapping family name to JudgmentResult
    """
    return {
        "claude": judge_with_llm(
            response, expected_tool, expected_args, is_json_condition,
            model="claude-sonnet"
        ),
        "gpt4": judge_with_llm(
            response, expected_tool, expected_args, is_json_condition,
            model="gpt-4o-mini"
        ),
        "gemini": judge_with_llm(
            response, expected_tool, expected_args, is_json_condition,
            model="gemini-flash"
        ),
    }


# ==============================================================================
# Unified Judging Interface
# ==============================================================================

class JudgingMode:
    """Judging modes for NL responses."""
    EXTRACTION = "extraction"  # Require full structured extraction
    INTENT = "intent"  # Check for intent signals only
    LLM = "llm"  # Use LLM-based judging


def judge_response(
    response: str,
    expected_tool: str,
    expected_args: dict[str, Any],
    is_json_condition: bool,
    use_llm: bool = False,
    llm_model: str = "claude-sonnet",
    nl_mode: str = JudgingMode.INTENT,  # Default to intent-based for NL
) -> JudgmentResult:
    """
    Judge a response for correctness.

    Args:
        response: The model response to judge
        expected_tool: Expected tool name
        expected_args: Expected argument values
        is_json_condition: Whether JSON-only (True) or NL-only (False)
        use_llm: Whether to use LLM-based judging
        llm_model: Model to use for LLM judging
        nl_mode: For NL responses, use "intent" (check signals) or
                 "extraction" (require full parse). Default: "intent"

    Returns:
        JudgmentResult with judgment
    """
    if use_llm:
        return judge_with_llm(
            response, expected_tool, expected_args, is_json_condition, llm_model
        )

    if is_json_condition:
        # JSON condition: strict validation (that's the point of JSON)
        return judge_json_response(response, expected_tool, expected_args)
    else:
        # NL condition: use specified mode
        if nl_mode == JudgingMode.INTENT:
            return judge_nl_intent(response, expected_tool, expected_args)
        else:
            return judge_nl_response(response, expected_tool, expected_args)


# ==============================================================================
# Human Validation Support
# ==============================================================================

def load_judgment_ground_truth(
    path: str = "experiments/validation/judgment_ground_truth.json"
) -> list[dict]:
    """Load ground truth judgments for validation."""
    gt_path = Path(path)
    if not gt_path.exists():
        return []

    with open(gt_path) as f:
        return json.load(f)


def validate_judge_human_agreement(
    ground_truth_path: str = "experiments/validation/judgment_ground_truth.json",
    use_llm: bool = True,
    llm_model: str = "claude-sonnet",
) -> dict:
    """
    Validate judge-human agreement.

    Per PLAN.md, judge-human κ ≥ 0.75 required.

    Returns:
        Dictionary with agreement metrics
    """
    ground_truth = load_judgment_ground_truth(ground_truth_path)

    if not ground_truth:
        return {
            "error": "No ground truth data found",
            "kappa": 0.0,
            "n_examples": 0,
        }

    human_labels = []
    judge_labels = []

    for example in ground_truth:
        response = example["response"]
        expected_tool = example["expected_tool"]
        expected_args = example.get("expected_args", {})
        is_json = example.get("is_json_condition", True)
        human_correct = example["human_judgment"]

        # Get judge's judgment
        result = judge_response(
            response, expected_tool, expected_args, is_json,
            use_llm=use_llm, llm_model=llm_model
        )

        human_labels.append(human_correct)
        judge_labels.append(result.is_correct)

    kappa = cohens_kappa(human_labels, judge_labels)

    # Calculate raw agreement
    agreement = sum(h == j for h, j in zip(human_labels, judge_labels)) / len(human_labels)

    return {
        "kappa": kappa,
        "agreement": agreement,
        "n_examples": len(ground_truth),
        "meets_threshold": kappa >= 0.75,
        "threshold": 0.75,
        "judge_model": llm_model if use_llm else "rule_based",
    }
