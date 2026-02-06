"""
Evaluation for Multi-Pass Diminishing Returns Study.

Tracks error detection across multiple passes and fits detection curves.
"""

import re
from typing import Any
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


def check_error_detected(
    response: str,
    error_location: str,
    error_description: str
) -> dict[str, Any]:
    """
    Check if the seeded error was detected in the response.

    Args:
        response: Model's code review response
        error_location: Expected location of error (e.g., "line 3")
        error_description: Description of the error

    Returns:
        Detection result dict
    """
    response_lower = response.lower()

    # Check for "no issues found" response
    no_issues_patterns = [
        "no issues found",
        "no bugs found",
        "no errors found",
        "code looks correct",
        "no problems",
        "looks good",
    ]

    if any(p in response_lower for p in no_issues_patterns):
        return {
            "detected": False,
            "localized": False,
            "fix_proposed": False,
            "false_negative": True,
        }

    # Extract key terms from error description for matching
    key_terms = []

    # Common error indicators
    error_indicators = {
        "off-by-one": ["off by one", "off-by-one", ">= instead of >", "> instead of >=", "boundary"],
        "sql_injection": ["sql injection", "parameterized", "sanitize", "escape"],
        "xss": ["xss", "cross-site", "sanitize", "escape", "innerhtml"],
        "race_condition": ["race condition", "thread safe", "synchroniz", "lock", "atomic"],
        "null": ["null", "none", "undefined", "empty", "missing check"],
        "division": ["division by zero", "divide by zero", "empty list"],
        "syntax": ["syntax", "typo", "missing", "extra", "bracket", "parenthesis"],
        "type": ["type error", "type mismatch", "cannot concatenate", "string.*int"],
        "n+1": ["n+1", "n plus 1", "query in loop", "multiple queries"],
        "deadlock": ["deadlock", "lock order", "circular"],
        "path_traversal": ["path traversal", "directory traversal", "../", "sanitize path"],
    }

    # Check if any error indicators match
    detected = False
    matched_indicators = []

    for indicator_type, patterns in error_indicators.items():
        for pattern in patterns:
            if pattern in response_lower or re.search(pattern, response_lower):
                detected = True
                matched_indicators.append(indicator_type)
                break

    # Also check for generic bug mentions with location
    if not detected:
        bug_patterns = [
            r"bug\s*(on|at|in)?\s*(line)?\s*\d+",
            r"error\s*(on|at|in)?\s*(line)?\s*\d+",
            r"issue\s*(on|at|in)?\s*(line)?\s*\d+",
            r"problem\s*(on|at|in)?\s*(line)?\s*\d+",
            r"line\s*\d+.*(?:bug|error|issue|wrong|incorrect)",
        ]
        for pattern in bug_patterns:
            if re.search(pattern, response_lower):
                detected = True
                break

    # Check if location was correctly identified
    localized = False
    if detected:
        # Extract line number from expected location
        line_match = re.search(r'line\s*(\d+)', error_location.lower())
        if line_match:
            expected_line = line_match.group(1)
            localized = expected_line in response

    # Check if fix was proposed
    fix_proposed = any(p in response_lower for p in [
        "fix:", "fixed:", "should be", "change to", "replace with",
        "instead of", "correction", "suggested fix", "to fix"
    ])

    return {
        "detected": detected,
        "localized": localized,
        "fix_proposed": fix_proposed,
        "matched_indicators": matched_indicators,
        "false_negative": not detected,
    }


def count_false_positives(
    response: str,
    actual_error_count: int = 1
) -> int:
    """
    Count likely false positives in the response.

    Args:
        response: Model's code review response
        actual_error_count: Number of actual errors in the code

    Returns:
        Estimated false positive count
    """
    # Count distinct issue mentions
    issue_patterns = [
        r'^\d+\.',  # Numbered list
        r'^-\s+',   # Bullet point
        r'issue\s*\d+',
        r'bug\s*\d+',
        r'error\s*\d+',
    ]

    reported_issues = 0
    for pattern in issue_patterns:
        matches = re.findall(pattern, response, re.MULTILINE)
        reported_issues = max(reported_issues, len(matches))

    # If no structured list found, estimate from "and" conjunctions in bug descriptions
    if reported_issues == 0:
        # Simple heuristic: count major bug-related sentences
        sentences = response.split('.')
        bug_sentences = [s for s in sentences if any(
            word in s.lower() for word in ['bug', 'error', 'issue', 'problem', 'wrong']
        )]
        reported_issues = len(bug_sentences)

    false_positives = max(0, reported_issues - actual_error_count)
    return false_positives


def evaluate_single_pass(
    task: dict[str, Any],
    response: str
) -> dict[str, Any]:
    """
    Evaluate a single review pass.

    Args:
        task: Task configuration
        response: Model's review response

    Returns:
        Evaluation dict
    """
    detection = check_error_detected(
        response,
        task["error_location"],
        task["error_description"]
    )

    false_positives = count_false_positives(response)

    return {
        "task_id": task.get("task_id"),
        "error_type": task.get("error_type"),
        "complexity": task.get("complexity"),
        "expected_detectability": task.get("expected_detectability"),

        "detected": detection["detected"],
        "localized": detection["localized"],
        "fix_proposed": detection["fix_proposed"],
        "false_positives": false_positives,

        "response_length": len(response),
    }


def evaluate_multipass(
    task: dict[str, Any],
    responses: list[str]
) -> dict[str, Any]:
    """
    Evaluate multiple passes over the same code.

    Args:
        task: Task configuration
        responses: List of review responses (one per pass)

    Returns:
        Multi-pass evaluation with detection timeline
    """
    pass_results = []
    first_detection_pass = None
    cumulative_detected = False

    for i, response in enumerate(responses):
        pass_num = i + 1
        result = evaluate_single_pass(task, response)
        result["pass_number"] = pass_num

        if result["detected"] and not cumulative_detected:
            first_detection_pass = pass_num
            cumulative_detected = True

        result["cumulative_detected"] = cumulative_detected
        pass_results.append(result)

    return {
        "task_id": task.get("task_id"),
        "error_type": task.get("error_type"),
        "complexity": task.get("complexity"),
        "expected_detectability": task.get("expected_detectability"),

        "num_passes": len(responses),
        "first_detection_pass": first_detection_pass,
        "ever_detected": cumulative_detected,

        "pass_results": pass_results,
    }


# =============================================================================
# Curve Fitting for Detection Models
# =============================================================================

def geometric_model(n: np.ndarray, p: float) -> np.ndarray:
    """
    Geometric decay model: P(detect by pass n) = 1 - (1-p)^n

    Args:
        n: Pass numbers
        p: Per-pass detection probability

    Returns:
        Cumulative detection probability
    """
    return 1 - (1 - p) ** n


def fit_detection_curve(
    pass_numbers: list[int],
    detection_rates: list[float]
) -> dict[str, Any]:
    """
    Fit the geometric model to observed detection rates.

    Args:
        pass_numbers: List of pass numbers
        detection_rates: Observed cumulative detection rates

    Returns:
        Fitted parameters and goodness of fit
    """
    try:
        popt, pcov = curve_fit(
            geometric_model,
            np.array(pass_numbers),
            np.array(detection_rates),
            p0=[0.5],
            bounds=(0, 1)
        )

        p_fitted = popt[0]

        # Calculate predicted values
        predicted = geometric_model(np.array(pass_numbers), p_fitted)

        # Calculate R²
        ss_res = np.sum((np.array(detection_rates) - predicted) ** 2)
        ss_tot = np.sum((np.array(detection_rates) - np.mean(detection_rates)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {
            "p_per_pass": float(p_fitted),
            "r_squared": float(r_squared),
            "predicted_rates": predicted.tolist(),
            "fit_successful": True,
        }

    except Exception as e:
        return {
            "fit_successful": False,
            "error": str(e),
        }


def aggregate_by_error_type_and_pass(
    evaluations: list[dict]
) -> dict[str, dict[int, float]]:
    """
    Aggregate detection rates by error type and pass number.

    Args:
        evaluations: List of multi-pass evaluation results

    Returns:
        Dict mapping error_type -> {pass_number -> detection_rate}
    """
    from collections import defaultdict

    # Organize by error type
    by_type = defaultdict(lambda: defaultdict(list))

    for ev in evaluations:
        error_type = ev.get("error_type", "unknown")
        for pass_result in ev.get("pass_results", []):
            pass_num = pass_result.get("pass_number", 0)
            detected = pass_result.get("cumulative_detected", False)
            by_type[error_type][pass_num].append(1 if detected else 0)

    # Calculate rates
    result = {}
    for error_type, by_pass in by_type.items():
        result[error_type] = {}
        for pass_num in sorted(by_pass.keys()):
            detections = by_pass[pass_num]
            result[error_type][pass_num] = sum(detections) / len(detections)

    return result


def find_knee_point(pass_numbers: list[int], rates: list[float]) -> int | None:
    """
    Find the "knee" point where diminishing returns begin.

    Uses the method of finding maximum curvature.

    Args:
        pass_numbers: Pass numbers
        rates: Detection rates

    Returns:
        Pass number of knee point, or None
    """
    if len(rates) < 3:
        return None

    # Calculate second derivative (curvature proxy)
    rates_arr = np.array(rates)
    first_deriv = np.diff(rates_arr)
    second_deriv = np.diff(first_deriv)

    # Knee is where second derivative is most negative (rate of improvement drops)
    if len(second_deriv) > 0:
        knee_idx = np.argmin(second_deriv)
        return pass_numbers[knee_idx + 1]  # +1 because of diff

    return None


if __name__ == "__main__":
    # Test detection
    test_task = {
        "task_id": "test",
        "error_type": "logic_error",
        "error_location": "line 6",
        "error_description": "Wrong comparison operator",
        "complexity": "moderate",
        "expected_detectability": "moderate",
    }

    test_response = """
    I found an issue in the code:

    1. Line 6: The comparison operator is wrong. It uses `<` when it should use `>`.
       This causes the function to find the minimum instead of maximum.

       Fix: Change `if n < max_val:` to `if n > max_val:`
    """

    result = evaluate_single_pass(test_task, test_response)
    print("Test result:", result)
