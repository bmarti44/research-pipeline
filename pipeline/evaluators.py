"""
Evaluator registry for research studies.

Pre-built evaluators that can be selected via configuration.
Evaluators score model responses against expected outputs.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from functools import wraps
import json
import re

from .models import Task, Response, Evaluation, EvaluationMode


# =============================================================================
# Evaluator Registry
# =============================================================================

_EVALUATOR_REGISTRY: dict[str, Callable] = {}


def register_evaluator(name: str):
    """Decorator to register an evaluator."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        _EVALUATOR_REGISTRY[name] = wrapper
        wrapper.evaluator_name = name
        return wrapper
    return decorator


def get_evaluator(name: str) -> Callable:
    """Get an evaluator by name."""
    if name not in _EVALUATOR_REGISTRY:
        available = list(_EVALUATOR_REGISTRY.keys())
        raise ValueError(f"Unknown evaluator: {name}. Available: {available}")
    return _EVALUATOR_REGISTRY[name]


def list_evaluators() -> list[str]:
    """List all registered evaluators."""
    return list(_EVALUATOR_REGISTRY.keys())


# =============================================================================
# Evaluation Result Helpers
# =============================================================================

@dataclass
class EvalResult:
    """Result from a single evaluation mode."""
    correct: bool
    reason: str = ""
    score: float = 0.0  # For non-binary evaluations
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "correct": self.correct,
            "reason": self.reason,
            "score": self.score,
            "details": self.details,
        }


def create_evaluation(
    trial_id: str,
    evaluator: str,
    strict_result: EvalResult,
    intent_result: Optional[EvalResult] = None,
    functional_result: Optional[EvalResult] = None,
) -> Evaluation:
    """Helper to create Evaluation with multiple modes."""
    results = {
        "strict": strict_result.to_dict(),
    }
    if intent_result:
        results["intent"] = intent_result.to_dict()
    if functional_result:
        results["functional"] = functional_result.to_dict()

    return Evaluation(
        trial_id=trial_id,
        evaluator=evaluator,
        results=results,
    )


# =============================================================================
# Pre-built Evaluators
# =============================================================================

# -----------------------------------------------------------------------------
# Exact Match Evaluators
# -----------------------------------------------------------------------------

@register_evaluator("exact_match")
def exact_match_evaluator(
    response: Response,
    task: Task,
    **kwargs,
) -> Evaluation:
    """
    Simple exact match evaluation.

    Compares response output to expected answer exactly.
    """
    expected = task.expected.get("answer", "")
    actual = response.raw_output.strip()

    is_correct = actual == expected

    strict_result = EvalResult(
        correct=is_correct,
        reason="Exact match" if is_correct else f"Expected '{expected}', got '{actual}'",
    )

    # Intent: case-insensitive, whitespace-normalized
    normalized_expected = " ".join(expected.lower().split())
    normalized_actual = " ".join(actual.lower().split())
    intent_correct = normalized_expected == normalized_actual

    intent_result = EvalResult(
        correct=intent_correct,
        reason="Normalized match" if intent_correct else "Mismatch after normalization",
    )

    return create_evaluation(
        trial_id=response.trial_id,
        evaluator="exact_match",
        strict_result=strict_result,
        intent_result=intent_result,
        functional_result=intent_result,  # Same as intent for exact match
    )


@register_evaluator("contains")
def contains_evaluator(
    response: Response,
    task: Task,
    **kwargs,
) -> Evaluation:
    """
    Check if response contains expected substring(s).

    task.expected should have 'contains' (string or list of strings).
    """
    expected = task.expected.get("contains", [])
    if isinstance(expected, str):
        expected = [expected]

    actual = response.raw_output

    missing = [e for e in expected if e not in actual]
    is_correct = len(missing) == 0

    strict_result = EvalResult(
        correct=is_correct,
        reason="All expected strings found" if is_correct else f"Missing: {missing}",
        details={"missing": missing, "expected": expected},
    )

    # Intent: case-insensitive
    actual_lower = actual.lower()
    missing_intent = [e for e in expected if e.lower() not in actual_lower]
    intent_correct = len(missing_intent) == 0

    intent_result = EvalResult(
        correct=intent_correct,
        reason="All expected strings found (case-insensitive)" if intent_correct else f"Missing: {missing_intent}",
    )

    return create_evaluation(
        trial_id=response.trial_id,
        evaluator="contains",
        strict_result=strict_result,
        intent_result=intent_result,
        functional_result=intent_result,
    )


@register_evaluator("regex")
def regex_evaluator(
    response: Response,
    task: Task,
    **kwargs,
) -> Evaluation:
    """
    Evaluate response against regex pattern(s).

    task.expected should have 'pattern' or 'patterns' (string or list).
    """
    patterns = task.expected.get("patterns", task.expected.get("pattern", []))
    if isinstance(patterns, str):
        patterns = [patterns]

    actual = response.raw_output

    results = []
    for pattern in patterns:
        match = re.search(pattern, actual)
        results.append({
            "pattern": pattern,
            "matched": match is not None,
            "match": match.group() if match else None,
        })

    all_matched = all(r["matched"] for r in results)
    failed = [r["pattern"] for r in results if not r["matched"]]

    strict_result = EvalResult(
        correct=all_matched,
        reason="All patterns matched" if all_matched else f"Failed patterns: {failed}",
        details={"results": results},
    )

    # Intent: case-insensitive regex
    results_ci = []
    for pattern in patterns:
        match = re.search(pattern, actual, re.IGNORECASE)
        results_ci.append({"pattern": pattern, "matched": match is not None})

    all_matched_ci = all(r["matched"] for r in results_ci)

    intent_result = EvalResult(
        correct=all_matched_ci,
        reason="All patterns matched (case-insensitive)" if all_matched_ci else "Pattern mismatch",
    )

    return create_evaluation(
        trial_id=response.trial_id,
        evaluator="regex",
        strict_result=strict_result,
        intent_result=intent_result,
        functional_result=intent_result,
    )


# -----------------------------------------------------------------------------
# Tool Call Evaluators
# -----------------------------------------------------------------------------

@register_evaluator("tool_call")
def tool_call_evaluator(
    response: Response,
    task: Task,
    **kwargs,
) -> Evaluation:
    """
    Evaluate tool call correctness.

    task.expected should have:
      - tool: expected tool name
      - args: expected arguments (dict)
    """
    expected_tool = task.expected.get("tool")
    expected_args = task.expected.get("args", {})

    # Get parsed tool call from response
    parsed = response.parsed
    actual_tool = parsed.get("tool")
    actual_args = parsed.get("args", {})

    # Strict: exact tool and exact args
    tool_match = actual_tool == expected_tool
    args_match = actual_args == expected_args

    strict_correct = tool_match and args_match
    strict_reason = []
    if not tool_match:
        strict_reason.append(f"Wrong tool: expected '{expected_tool}', got '{actual_tool}'")
    if not args_match:
        strict_reason.append(f"Wrong args: expected {expected_args}, got {actual_args}")

    strict_result = EvalResult(
        correct=strict_correct,
        reason="; ".join(strict_reason) if strict_reason else "Correct tool call",
        details={
            "tool_match": tool_match,
            "args_match": args_match,
            "expected_tool": expected_tool,
            "actual_tool": actual_tool,
            "expected_args": expected_args,
            "actual_args": actual_args,
        },
    )

    # Intent: correct tool, args values referenced (not exact)
    def values_referenced(expected: dict, actual: dict, raw_output: str) -> bool:
        """Check if expected values appear somewhere in actual args or raw output."""
        for key, value in expected.items():
            str_value = str(value)
            # Check if value is in actual args or raw output
            value_found = (
                str_value in str(actual.get(key, "")) or
                str_value in raw_output
            )
            if not value_found:
                return False
        return True

    intent_correct = tool_match and values_referenced(expected_args, actual_args, response.raw_output)

    intent_result = EvalResult(
        correct=intent_correct,
        reason="Correct intent" if intent_correct else "Intent mismatch",
    )

    # Functional: equivalent tool acceptable
    # Define tool equivalence classes
    tool_equivalences = kwargs.get("tool_equivalences", {})
    equivalent_tools = tool_equivalences.get(expected_tool, [expected_tool])

    functional_tool_match = actual_tool in equivalent_tools
    functional_correct = functional_tool_match and values_referenced(expected_args, actual_args, response.raw_output)

    functional_result = EvalResult(
        correct=functional_correct,
        reason="Functionally correct" if functional_correct else "Functional mismatch",
        details={"equivalent_tools": equivalent_tools},
    )

    return create_evaluation(
        trial_id=response.trial_id,
        evaluator="tool_call",
        strict_result=strict_result,
        intent_result=intent_result,
        functional_result=functional_result,
    )


@register_evaluator("tool_call_nl")
def tool_call_nl_evaluator(
    response: Response,
    task: Task,
    **kwargs,
) -> Evaluation:
    """
    Evaluate natural language description of a tool call.

    For NL condition where model describes intent rather than producing JSON.
    """
    expected_tool = task.expected.get("tool")
    expected_args = task.expected.get("args", {})

    raw_output = response.raw_output.lower()

    # Check if tool name or synonyms are mentioned
    tool_synonyms = kwargs.get("tool_synonyms", {})
    tool_names = [expected_tool.lower()] + [s.lower() for s in tool_synonyms.get(expected_tool, [])]

    tool_mentioned = any(name in raw_output for name in tool_names)

    # Check if arg values are referenced
    args_referenced = all(
        str(value).lower() in raw_output
        for value in expected_args.values()
    )

    # For NL, strict = intent = functional (all semantic)
    is_correct = tool_mentioned and args_referenced

    reason_parts = []
    if not tool_mentioned:
        reason_parts.append(f"Tool '{expected_tool}' not mentioned")
    if not args_referenced:
        missing = [k for k, v in expected_args.items() if str(v).lower() not in raw_output]
        reason_parts.append(f"Args not referenced: {missing}")

    result = EvalResult(
        correct=is_correct,
        reason="Correct NL description" if is_correct else "; ".join(reason_parts),
        details={
            "tool_mentioned": tool_mentioned,
            "args_referenced": args_referenced,
            "expected_tool": expected_tool,
            "expected_args": expected_args,
        },
    )

    return create_evaluation(
        trial_id=response.trial_id,
        evaluator="tool_call_nl",
        strict_result=result,
        intent_result=result,
        functional_result=result,
    )


# -----------------------------------------------------------------------------
# JSON Structure Evaluators
# -----------------------------------------------------------------------------

@register_evaluator("json_valid")
def json_valid_evaluator(
    response: Response,
    task: Task,
    **kwargs,
) -> Evaluation:
    """
    Check if response is valid JSON.
    """
    raw_output = response.raw_output.strip()

    try:
        parsed = json.loads(raw_output)
        is_valid = True
        parse_error = None
    except json.JSONDecodeError as e:
        is_valid = False
        parse_error = str(e)
        parsed = None

    strict_result = EvalResult(
        correct=is_valid,
        reason="Valid JSON" if is_valid else f"Invalid JSON: {parse_error}",
        details={"parsed": parsed, "error": parse_error},
    )

    return create_evaluation(
        trial_id=response.trial_id,
        evaluator="json_valid",
        strict_result=strict_result,
        intent_result=strict_result,
        functional_result=strict_result,
    )


@register_evaluator("json_schema")
def json_schema_evaluator(
    response: Response,
    task: Task,
    **kwargs,
) -> Evaluation:
    """
    Validate response against JSON schema.

    task.expected should have 'schema' with JSON Schema definition.
    """
    import jsonschema

    raw_output = response.raw_output.strip()
    schema = task.expected.get("schema", {})

    try:
        parsed = json.loads(raw_output)
        jsonschema.validate(parsed, schema)
        is_valid = True
        error = None
    except json.JSONDecodeError as e:
        is_valid = False
        error = f"Invalid JSON: {e}"
        parsed = None
    except jsonschema.ValidationError as e:
        is_valid = False
        error = f"Schema validation failed: {e.message}"
        parsed = None

    strict_result = EvalResult(
        correct=is_valid,
        reason="Valid and matches schema" if is_valid else error,
        details={"parsed": parsed, "error": error},
    )

    # Intent: valid JSON with required fields present (not exact schema)
    intent_correct = False
    if parsed is not None:
        required = schema.get("required", [])
        has_required = all(k in parsed for k in required)
        intent_correct = has_required

    intent_result = EvalResult(
        correct=intent_correct,
        reason="Has required fields" if intent_correct else "Missing required fields",
    )

    return create_evaluation(
        trial_id=response.trial_id,
        evaluator="json_schema",
        strict_result=strict_result,
        intent_result=intent_result,
        functional_result=intent_result,
    )


# -----------------------------------------------------------------------------
# Numeric Evaluators
# -----------------------------------------------------------------------------

@register_evaluator("numeric")
def numeric_evaluator(
    response: Response,
    task: Task,
    tolerance: float = 0.0001,
    **kwargs,
) -> Evaluation:
    """
    Evaluate numeric answer.

    task.expected should have 'value' (the expected number).
    """
    expected = task.expected.get("value")
    raw_output = response.raw_output.strip()

    # Try to extract number from response
    numbers = re.findall(r'-?\d+\.?\d*', raw_output)

    if not numbers:
        strict_result = EvalResult(
            correct=False,
            reason="No number found in response",
        )
    else:
        # Take the first number found
        actual = float(numbers[0])
        is_correct = abs(actual - expected) <= tolerance

        strict_result = EvalResult(
            correct=is_correct,
            reason=f"Expected {expected}, got {actual}" if not is_correct else "Correct",
            details={"expected": expected, "actual": actual, "tolerance": tolerance},
        )

    return create_evaluation(
        trial_id=response.trial_id,
        evaluator="numeric",
        strict_result=strict_result,
        intent_result=strict_result,
        functional_result=strict_result,
    )


# -----------------------------------------------------------------------------
# Classification Evaluators
# -----------------------------------------------------------------------------

@register_evaluator("classification")
def classification_evaluator(
    response: Response,
    task: Task,
    **kwargs,
) -> Evaluation:
    """
    Evaluate classification response.

    task.expected should have 'label' (the expected class).
    """
    expected = task.expected.get("label", "").lower()
    raw_output = response.raw_output.strip().lower()

    # Strict: exact match
    strict_correct = raw_output == expected

    strict_result = EvalResult(
        correct=strict_correct,
        reason="Correct class" if strict_correct else f"Expected '{expected}', got '{raw_output}'",
    )

    # Intent: expected label appears in response
    intent_correct = expected in raw_output

    intent_result = EvalResult(
        correct=intent_correct,
        reason="Label found in response" if intent_correct else "Label not found",
    )

    return create_evaluation(
        trial_id=response.trial_id,
        evaluator="classification",
        strict_result=strict_result,
        intent_result=intent_result,
        functional_result=intent_result,
    )


# -----------------------------------------------------------------------------
# LLM-as-Judge Evaluators
# -----------------------------------------------------------------------------

@register_evaluator("llm_judge")
def llm_judge_evaluator(
    response: Response,
    task: Task,
    judge_model: str = "claude-3-haiku-20240307",
    judge_prompt: Optional[str] = None,
    **kwargs,
) -> Evaluation:
    """
    Use an LLM to evaluate the response.

    This is a placeholder - actual implementation requires API calls.
    """
    # Default judge prompt
    if judge_prompt is None:
        judge_prompt = """Evaluate whether this response correctly addresses the task.

Task: {task_prompt}
Expected: {expected}
Response: {response}

Is the response correct? Answer with just 'correct' or 'incorrect' followed by a brief reason."""

    # This would need to make an actual API call
    # For now, return placeholder
    strict_result = EvalResult(
        correct=False,
        reason="LLM judge evaluation not implemented - requires API call",
        details={
            "judge_model": judge_model,
            "task_prompt": task.prompt,
            "expected": task.expected,
        },
    )

    return create_evaluation(
        trial_id=response.trial_id,
        evaluator="llm_judge",
        strict_result=strict_result,
        intent_result=strict_result,
        functional_result=strict_result,
    )


# =============================================================================
# Evaluation Runner
# =============================================================================

def evaluate_response(
    response: Response,
    task: Task,
    evaluator_name: str,
    **kwargs,
) -> Evaluation:
    """
    Evaluate a single response using the specified evaluator.
    """
    evaluator = get_evaluator(evaluator_name)
    return evaluator(response, task, **kwargs)


def evaluate_batch(
    responses: list[tuple[Response, Task]],
    evaluator_name: str,
    **kwargs,
) -> list[Evaluation]:
    """
    Evaluate a batch of responses.
    """
    evaluator = get_evaluator(evaluator_name)
    return [evaluator(response, task, **kwargs) for response, task in responses]


def run_all_evaluators(
    response: Response,
    task: Task,
    evaluator_names: list[str],
    **kwargs,
) -> dict[str, Evaluation]:
    """
    Run multiple evaluators on a single response.
    """
    results = {}
    for name in evaluator_names:
        evaluator = get_evaluator(name)
        results[name] = evaluator(response, task, **kwargs)
    return results
