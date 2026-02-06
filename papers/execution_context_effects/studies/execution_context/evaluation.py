"""
Evaluation logic for tool-calling studies.

Evaluates model responses for correct tool selection and argument values.
"""

import json
import re


def normalize_value(value: str) -> str:
    """Normalize a value for comparison."""
    if not isinstance(value, str):
        return str(value).lower()
    # Handle escape sequences
    value = value.replace("\\'", "'").replace('\\"', '"')
    value = value.replace("\\\\", "\\")
    return value.lower().strip()


def value_in_text(value, text: str) -> bool:
    """Check if value appears in text with normalization."""
    if not isinstance(value, str):
        return str(value) in text

    text_lower = text.lower()
    value_lower = value.lower()
    value_norm = normalize_value(value)

    # Direct match
    if value_lower in text_lower:
        return True

    # Normalized match
    if value_norm in normalize_value(text):
        return True

    return False


def evaluate_nl_response(response: str, task: dict) -> dict:
    """
    Evaluate NL response for signal detection.

    Checks if the model mentioned the correct tool and argument values.
    """
    expected_tool = task.get("expected_tool")
    expected_args = task.get("expected_args", {})

    # No tool expected (control task)
    if expected_tool is None:
        return {
            "correct": True,
            "tool_mentioned": True,
            "args_mentioned": True,
            "notes": "Control task - no tool expected",
        }

    response_lower = response.lower()

    # Check tool mentioned
    tool_mentioned = (
        expected_tool.lower() in response_lower or
        expected_tool.replace("_", " ").lower() in response_lower
    )

    # Check args mentioned
    args_found = 0
    for key, value in expected_args.items():
        if value_in_text(value, response):
            args_found += 1

    args_mentioned = args_found >= len(expected_args) * 0.5 if expected_args else True

    return {
        "correct": tool_mentioned and args_mentioned,
        "tool_mentioned": tool_mentioned,
        "args_mentioned": args_mentioned,
        "args_found": args_found,
        "args_expected": len(expected_args),
    }


def evaluate_json_response(response: str, task: dict) -> dict:
    """
    Evaluate JSON response for correct structure and content.
    """
    expected_tool = task.get("expected_tool")
    expected_args = task.get("expected_args", {})

    result = {
        "correct": False,
        "json_valid": False,
        "json_structure": False,
        "tool_correct": False,
        "args_present": False,
        "actual_tool": None,
        "actual_args": None,
    }

    # No tool expected (control task)
    if expected_tool is None:
        result["correct"] = True
        result["notes"] = "Control task - no tool expected"
        return result

    # Try to parse JSON
    try:
        # Strip markdown code blocks if present
        clean_response = response.strip()
        if clean_response.startswith("```"):
            clean_response = re.sub(r"```\w*\n?", "", clean_response)
            clean_response = clean_response.strip()

        parsed = json.loads(clean_response)
        result["json_valid"] = True
    except json.JSONDecodeError:
        return result

    if not isinstance(parsed, dict):
        return result

    # Check structure
    if "tool" in parsed:
        result["json_structure"] = True
        result["actual_tool"] = parsed.get("tool")

        # Check tool name
        result["tool_correct"] = (parsed.get("tool") == expected_tool)

        # Check args
        args = parsed.get("args", parsed)  # args might be at top level
        if isinstance(args, dict):
            result["actual_args"] = args
            result["args_present"] = all(
                key in args for key in expected_args.keys()
            )

    result["correct"] = (
        result["json_valid"] and
        result["json_structure"] and
        result["tool_correct"] and
        result["args_present"]
    )

    return result


def evaluate_response(
    response: str,
    task: dict,
    condition: str,
) -> dict:
    """
    Evaluate a model response.

    Args:
        response: The model's response text
        task: The task dictionary
        condition: The condition name ('nl_only' or 'json_only')

    Returns:
        Dictionary with evaluation results for each mode
    """
    expected_tool = task.get("expected_tool")
    expected_args = task.get("expected_args", {})

    # Evaluate based on condition
    if condition == "nl_only":
        nl_eval = evaluate_nl_response(response, task)

        return {
            "strict": {
                "correct": nl_eval["correct"] and nl_eval["tool_mentioned"],
                "details": nl_eval,
            },
            "intent": {
                "correct": nl_eval["correct"],
                "details": nl_eval,
            },
            "functional": {
                "correct": nl_eval["tool_mentioned"],  # Just needs to mention tool
                "details": nl_eval,
            },
        }

    elif condition == "json_only":
        json_eval = evaluate_json_response(response, task)
        nl_eval = evaluate_nl_response(response, task)  # Also check signal

        # Format friction: signaled correctly but JSON failed
        format_friction = nl_eval["tool_mentioned"] and not json_eval["correct"]

        return {
            "strict": {
                "correct": json_eval["correct"],
                "details": json_eval,
                "format_friction": format_friction,
            },
            "intent": {
                "correct": json_eval["tool_correct"] if json_eval["json_valid"] else False,
                "details": json_eval,
                "format_friction": format_friction,
            },
            "functional": {
                "correct": json_eval["json_valid"] and json_eval["tool_correct"],
                "details": json_eval,
                "format_friction": format_friction,
            },
        }

    else:
        return {
            "strict": {"correct": False, "error": f"Unknown condition: {condition}"},
            "intent": {"correct": False, "error": f"Unknown condition: {condition}"},
            "functional": {"correct": False, "error": f"Unknown condition: {condition}"},
        }
