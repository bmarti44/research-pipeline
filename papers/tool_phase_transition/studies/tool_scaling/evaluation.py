"""
Evaluation for Tool Phase Transition Study.

Evaluates tool selection accuracy and parameter correctness.
"""

import json
import re
from typing import Any


def extract_tool_call(response: str) -> dict[str, Any] | None:
    """
    Extract tool call from model response.

    Args:
        response: Model response text

    Returns:
        Dict with 'tool' and 'args' or None
    """
    # Try to find JSON tool call
    patterns = [
        r'\{[^{}]*"tool"[^{}]*\}',
        r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\}',
        r'\{[^{}]*"function"[^{}]*\}',
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                tool_name = data.get("tool") or data.get("name") or data.get("function")
                args = data.get("args") or data.get("arguments") or data.get("parameters", {})
                if tool_name:
                    return {"tool": tool_name, "args": args}
            except json.JSONDecodeError:
                continue

    # Try to extract tool name from text
    tool_patterns = [
        r'I would (?:use|call|invoke) (?:the )?[`"\']?(\w+)[`"\']?',
        r'(?:call|using|invoke)[:\s]+[`"\']?(\w+)[`"\']?',
        r'tool[:\s]+[`"\']?(\w+)[`"\']?',
    ]

    for pattern in tool_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return {"tool": match.group(1), "args": {}}

    return None


def check_tool_correct(
    selected_tool: str | None,
    correct_tool: str,
    available_tools: list[dict]
) -> dict[str, Any]:
    """
    Check if selected tool is correct.

    Args:
        selected_tool: Tool name selected by model
        correct_tool: Expected correct tool
        available_tools: List of available tool definitions

    Returns:
        Dict with correctness metrics
    """
    available_names = {t["name"] for t in available_tools}

    if selected_tool is None:
        return {
            "correct_tool": False,
            "hallucinated": False,
            "no_tool_selected": True,
        }

    correct = selected_tool.lower() == correct_tool.lower()
    hallucinated = selected_tool.lower() not in {n.lower() for n in available_names}

    return {
        "correct_tool": correct,
        "hallucinated": hallucinated,
        "no_tool_selected": False,
        "selected_tool": selected_tool,
    }


def check_parameters_correct(
    selected_args: dict,
    correct_args: dict
) -> dict[str, Any]:
    """
    Check if parameters are correct.

    Args:
        selected_args: Args from model
        correct_args: Expected correct args

    Returns:
        Dict with parameter correctness metrics
    """
    if not correct_args:
        return {"correct_params": True, "missing_params": [], "wrong_params": []}

    missing = []
    wrong = []

    for key, expected_value in correct_args.items():
        if key not in selected_args:
            missing.append(key)
        elif str(selected_args[key]).lower() != str(expected_value).lower():
            # Allow partial matches for strings
            if isinstance(expected_value, str):
                if expected_value.lower() not in str(selected_args[key]).lower():
                    wrong.append(key)
            else:
                wrong.append(key)

    correct = len(missing) == 0 and len(wrong) == 0

    return {
        "correct_params": correct,
        "missing_params": missing,
        "wrong_params": wrong,
        "param_accuracy": (
            (len(correct_args) - len(missing) - len(wrong)) / len(correct_args)
            if correct_args else 1.0
        ),
    }


def evaluate_response(task: dict[str, Any], response: str) -> dict[str, Any]:
    """
    Evaluate a tool selection response.

    Args:
        task: Task configuration
        response: Model response

    Returns:
        Evaluation dict
    """
    # Extract tool call
    tool_call = extract_tool_call(response)

    selected_tool = tool_call["tool"] if tool_call else None
    selected_args = tool_call["args"] if tool_call else {}

    # Check tool correctness
    tool_result = check_tool_correct(
        selected_tool,
        task["correct_tool"],
        task["available_tools"]
    )

    # Check parameter correctness
    param_result = check_parameters_correct(
        selected_args,
        task["correct_args"]
    )

    # Overall correctness
    correct = tool_result["correct_tool"] and param_result["correct_params"]

    return {
        "task_id": task.get("task_id"),
        "tool_count": task.get("tool_count"),
        "distractor_type": task.get("distractor_type"),
        "category": task.get("category"),

        "correct": correct,
        "correct_tool": tool_result["correct_tool"],
        "correct_params": param_result["correct_params"],
        "hallucinated_tool": tool_result.get("hallucinated", False),
        "no_tool_selected": tool_result.get("no_tool_selected", False),

        "selected_tool": selected_tool,
        "expected_tool": task["correct_tool"],
        "param_accuracy": param_result.get("param_accuracy", 0),
    }


def aggregate_by_tool_count(evaluations: list[dict]) -> dict[int, dict]:
    """Aggregate results by tool count to find phase transition."""
    from collections import defaultdict

    by_count = defaultdict(list)
    for ev in evaluations:
        count = ev.get("tool_count", 0)
        by_count[count].append(ev)

    aggregates = {}
    for count in sorted(by_count.keys()):
        evals = by_count[count]
        n = len(evals)

        correct = sum(1 for e in evals if e.get("correct"))
        tool_correct = sum(1 for e in evals if e.get("correct_tool"))
        hallucinated = sum(1 for e in evals if e.get("hallucinated_tool"))

        aggregates[count] = {
            "n": n,
            "accuracy": round(correct / n, 4) if n > 0 else 0,
            "tool_accuracy": round(tool_correct / n, 4) if n > 0 else 0,
            "hallucination_rate": round(hallucinated / n, 4) if n > 0 else 0,
        }

    return aggregates


def find_phase_transition(aggregates: dict[int, dict]) -> dict[str, Any]:
    """
    Find the phase transition point in accuracy.

    Args:
        aggregates: Tool count -> metrics dict

    Returns:
        Phase transition analysis
    """
    counts = sorted(aggregates.keys())
    accuracies = [aggregates[c]["accuracy"] for c in counts]

    if len(counts) < 2:
        return {"error": "Insufficient data"}

    # Find largest single-step drop
    drops = []
    for i in range(len(accuracies) - 1):
        drop = accuracies[i] - accuracies[i + 1]
        drops.append({
            "from_count": counts[i],
            "to_count": counts[i + 1],
            "drop": drop,
            "from_accuracy": accuracies[i],
            "to_accuracy": accuracies[i + 1],
        })

    # Sort by drop magnitude
    drops.sort(key=lambda x: x["drop"], reverse=True)

    cliff_point = drops[0] if drops else None

    # Check if it's a true phase transition (>15% drop)
    is_phase_transition = cliff_point and cliff_point["drop"] > 0.15

    return {
        "cliff_point": cliff_point,
        "is_phase_transition": is_phase_transition,
        "all_drops": drops[:3],  # Top 3 drops
        "recommended_max_tools": cliff_point["from_count"] if cliff_point else max(counts),
    }


if __name__ == "__main__":
    # Test tool extraction
    test_responses = [
        '{"tool": "get_weather", "args": {"location": "NYC"}}',
        'I would call the send_email tool with the recipient address.',
        'Using: query_database with table=users',
    ]

    for resp in test_responses:
        result = extract_tool_call(resp)
        print(f"Response: {resp[:50]}... -> {result}")
