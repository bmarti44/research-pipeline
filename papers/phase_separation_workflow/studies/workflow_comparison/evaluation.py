"""
Evaluation for Phase Separation Workflow Study.

Evaluates:
1. Task completion and code quality
2. Error detection and propagation
3. Critique quality (for mechanism analysis)
"""

import re
import ast
from typing import Any

import numpy as np


def check_code_syntax(code: str, language: str = "python") -> dict[str, Any]:
    """
    Check if code has valid syntax.

    Args:
        code: Code string to check
        language: Programming language

    Returns:
        Syntax check result
    """
    if language != "python":
        return {"valid": True, "error": None}  # Only check Python for now

    # Extract code from markdown if present
    code_match = re.search(r'```python\n(.*?)```', code, re.DOTALL)
    if code_match:
        code = code_match.group(1)

    try:
        ast.parse(code)
        return {"valid": True, "error": None}
    except SyntaxError as e:
        return {"valid": False, "error": str(e)}


def count_errors_in_plan(plan_content: str) -> dict[str, Any]:
    """
    Heuristically identify potential issues in a plan.

    Args:
        plan_content: The plan markdown content

    Returns:
        Error analysis
    """
    issues = []
    plan_lower = plan_content.lower()

    # Check for missing considerations
    missing_checks = [
        ("edge case", "May not handle edge cases"),
        ("error handling", "Missing error handling discussion"),
        ("validation", "No input validation mentioned"),
        ("test", "No testing strategy"),
        ("security", "No security considerations"),
    ]

    for keyword, issue in missing_checks:
        if keyword not in plan_lower:
            issues.append({"type": "missing", "description": issue, "severity": "minor"})

    # Check for vague language (potential issues)
    vague_patterns = [
        (r'\bsomehow\b', "Vague: 'somehow'"),
        (r'\bmaybe\b', "Uncertain: 'maybe'"),
        (r'\bprobably\b', "Uncertain: 'probably'"),
        (r'\betc\.?\b', "Incomplete: 'etc'"),
        (r'\band so on\b', "Incomplete: 'and so on'"),
    ]

    for pattern, issue in vague_patterns:
        if re.search(pattern, plan_lower):
            issues.append({"type": "vague", "description": issue, "severity": "minor"})

    return {
        "issue_count": len(issues),
        "issues": issues,
        "has_critical": any(i["severity"] == "critical" for i in issues),
    }


def evaluate_critique_quality(
    review_content: str,
    plan_content: str
) -> dict[str, Any]:
    """
    Evaluate the quality of a critique/review.

    Key metrics for mechanism analysis:
    - Does it find real issues (not just praise)?
    - Is it specific or vague?
    - Does it defend vs challenge the plan?

    Args:
        review_content: The review content
        plan_content: The original plan being reviewed

    Returns:
        Critique quality metrics
    """
    review_lower = review_content.lower()

    # Check for substantive critique vs defense
    defense_indicators = [
        "looks good", "well thought out", "comprehensive",
        "no major issues", "solid plan", "well structured",
        "i agree with", "this is correct"
    ]

    critique_indicators = [
        "missing", "should", "could be improved", "potential issue",
        "doesn't handle", "fails to", "overlooked", "concern",
        "bug", "error", "problem", "flaw", "weakness"
    ]

    defense_count = sum(1 for d in defense_indicators if d in review_lower)
    critique_count = sum(1 for c in critique_indicators if c in review_lower)

    # Check for specific vs vague feedback
    specific_patterns = [
        r'line \d+',
        r'step \d+',
        r'in the \w+ function',
        r'when \w+ is',
        r'if \w+ is (null|none|empty|negative)',
    ]
    specificity_count = sum(
        1 for p in specific_patterns if re.search(p, review_lower)
    )

    # Count actionable items
    actionable_patterns = [
        r'should (add|include|check|handle|consider)',
        r'need to',
        r'must (add|include|check)',
        r'recommend',
        r'suggest',
    ]
    actionable_count = sum(
        1 for p in actionable_patterns if re.search(p, review_lower)
    )

    # Calculate scores
    critique_score = min(critique_count / 5.0, 1.0)  # Normalize
    defense_score = min(defense_count / 3.0, 1.0)
    specificity_score = min(specificity_count / 3.0, 1.0)
    actionability_score = min(actionable_count / 3.0, 1.0)

    # Overall quality: high critique, high specificity, high actionability, low defense
    quality_score = (
        critique_score * 0.3 +
        specificity_score * 0.3 +
        actionability_score * 0.3 -
        defense_score * 0.1
    )

    return {
        "quality_score": round(max(0, min(1, quality_score)), 3),
        "critique_indicators": critique_count,
        "defense_indicators": defense_count,
        "specificity": round(specificity_score, 3),
        "actionability": round(actionability_score, 3),
        "is_substantive": critique_count > defense_count,
        "is_defensive": defense_count > critique_count,
    }


def evaluate_context_collapse(
    phase_outputs: dict[str, str],
    task: dict[str, Any]
) -> dict[str, Any]:
    """
    Evaluate context collapse indicators.

    Measures whether details are lost across iterations.

    Args:
        phase_outputs: Dict mapping phase name to output
        task: Task configuration

    Returns:
        Context collapse metrics
    """
    # Get original requirements from task
    description = task.get("description", "")
    requirement_keywords = set(
        word.lower() for word in re.findall(r'\b\w{4,}\b', description)
        if word.lower() not in {'this', 'that', 'with', 'from', 'should', 'must', 'will'}
    )

    # Track detail preservation across iterations
    phase_scores = []
    for phase_name, output in phase_outputs.items():
        if not output:
            continue
        output_lower = output.lower()
        # Count how many requirement keywords appear
        preserved = sum(1 for kw in requirement_keywords if kw in output_lower)
        preservation_rate = preserved / len(requirement_keywords) if requirement_keywords else 1.0
        phase_scores.append({
            "phase": phase_name,
            "preservation_rate": preservation_rate,
            "keywords_preserved": preserved,
        })

    # Check for degradation pattern (later phases have lower preservation)
    if len(phase_scores) >= 2:
        first_half_avg = np.mean([p["preservation_rate"] for p in phase_scores[:len(phase_scores)//2]])
        second_half_avg = np.mean([p["preservation_rate"] for p in phase_scores[len(phase_scores)//2:]])
        degradation = first_half_avg - second_half_avg
    else:
        degradation = 0

    # Final output coverage
    final_output = phase_outputs.get("execute") or phase_outputs.get("plan_and_execute", "")
    final_preserved = sum(1 for kw in requirement_keywords if kw in final_output.lower())
    final_coverage = final_preserved / len(requirement_keywords) if requirement_keywords else 1.0

    return {
        "detail_preservation": round(final_coverage, 4),
        "iteration_degradation": round(degradation, 4),
        "shows_collapse_pattern": degradation > 0.1,  # 10% degradation threshold
        "phase_scores": phase_scores,
        "requirement_coverage": round(final_coverage, 4),
    }


def evaluate_code_quality(code: str, task: dict[str, Any]) -> dict[str, Any]:
    """
    Evaluate the quality of generated code.

    Args:
        code: Generated code
        task: Task configuration

    Returns:
        Code quality metrics
    """
    # Syntax check
    syntax = check_code_syntax(code, task.get("language", "python"))

    if not syntax["valid"]:
        return {
            "syntax_valid": False,
            "syntax_error": syntax["error"],
            "quality_score": 0,
        }

    # Extract code from markdown
    code_match = re.search(r'```python\n(.*?)```', code, re.DOTALL)
    clean_code = code_match.group(1) if code_match else code

    # Quality indicators
    has_docstring = '"""' in clean_code or "'''" in clean_code
    has_type_hints = "->" in clean_code or ": " in clean_code
    has_error_handling = "try:" in clean_code or "except" in clean_code or "raise" in clean_code

    # Complexity indicators (rough)
    line_count = len(clean_code.strip().split('\n'))
    function_count = len(re.findall(r'\ndef \w+', clean_code))

    # Calculate quality score
    quality_components = [
        (has_docstring, 0.2),
        (has_type_hints, 0.2),
        (has_error_handling, 0.3),
        (function_count > 0, 0.2),
        (10 < line_count < 200, 0.1),
    ]
    quality_score = sum(weight for present, weight in quality_components if present)

    return {
        "syntax_valid": True,
        "has_docstring": has_docstring,
        "has_type_hints": has_type_hints,
        "has_error_handling": has_error_handling,
        "line_count": line_count,
        "function_count": function_count,
        "quality_score": round(quality_score, 3),
    }


def evaluate_workflow_execution(
    task: dict[str, Any],
    phase_outputs: dict[str, str]
) -> dict[str, Any]:
    """
    Evaluate a complete workflow execution.

    Args:
        task: Task configuration
        phase_outputs: Dict mapping phase name to output content

    Returns:
        Complete evaluation
    """
    workflow = task.get("workflow", "unknown")
    workflow_config = task.get("workflow_config", {})

    # Get final code (from execute or plan_and_execute phase)
    final_code = phase_outputs.get("execute") or phase_outputs.get("plan_and_execute", "")

    # Evaluate code quality
    code_quality = evaluate_code_quality(final_code, task)

    # Evaluate plan if present
    plan_analysis = None
    if "plan" in phase_outputs:
        plan_analysis = count_errors_in_plan(phase_outputs["plan"])

    # Evaluate review if present
    review_analysis = None
    if "review" in phase_outputs and "plan" in phase_outputs:
        review_analysis = evaluate_critique_quality(
            phase_outputs["review"],
            phase_outputs["plan"]
        )

    # Count error propagation
    errors_in_plan = plan_analysis["issue_count"] if plan_analysis else 0
    errors_caught = 0
    if review_analysis and review_analysis.get("critique_indicators"):
        errors_caught = review_analysis["critique_indicators"]

    # Evaluate context collapse
    collapse_analysis = evaluate_context_collapse(phase_outputs, task)

    # Task completion heuristic
    task_completed = (
        code_quality["syntax_valid"] and
        code_quality["quality_score"] > 0.3 and
        code_quality["function_count"] > 0
    )

    return {
        "task_id": task.get("task_id"),
        "workflow": workflow,
        "complexity": task.get("complexity"),
        "task_type": task.get("task_type"),

        "task_completed": task_completed,
        "code_quality": code_quality,

        "plan_analysis": plan_analysis,
        "review_analysis": review_analysis,

        "errors_in_plan": errors_in_plan,
        "errors_caught_in_review": errors_caught,
        "error_propagation": max(0, errors_in_plan - errors_caught),

        "collapse_analysis": collapse_analysis,
        "detail_preservation": collapse_analysis.get("detail_preservation", 1.0),
        "shows_collapse_pattern": collapse_analysis.get("shows_collapse_pattern", False),

        "phases_completed": list(phase_outputs.keys()),
        "context_clearing_used": workflow_config.get("context_clearing", False),
        "external_memory_used": workflow_config.get("external_memory", False),
    }


def aggregate_by_workflow(evaluations: list[dict]) -> dict[str, dict]:
    """Aggregate results by workflow type."""
    from collections import defaultdict

    by_workflow = defaultdict(list)
    for ev in evaluations:
        workflow = ev.get("workflow", "unknown")
        by_workflow[workflow].append(ev)

    aggregates = {}
    for workflow, evals in by_workflow.items():
        n = len(evals)
        if n == 0:
            continue

        completed = sum(1 for e in evals if e.get("task_completed"))
        avg_quality = sum(
            e.get("code_quality", {}).get("quality_score", 0)
            for e in evals
        ) / n

        # Review metrics (if applicable)
        with_reviews = [e for e in evals if e.get("review_analysis")]
        avg_critique_quality = 0
        if with_reviews:
            avg_critique_quality = sum(
                e["review_analysis"].get("quality_score", 0)
                for e in with_reviews
            ) / len(with_reviews)

        # Error metrics
        avg_errors_caught = sum(
            e.get("errors_caught_in_review", 0) for e in evals
        ) / n

        aggregates[workflow] = {
            "n": n,
            "completion_rate": round(completed / n, 4),
            "avg_code_quality": round(avg_quality, 4),
            "avg_critique_quality": round(avg_critique_quality, 4),
            "avg_errors_caught": round(avg_errors_caught, 2),
        }

    return aggregates


if __name__ == "__main__":
    # Test evaluation
    test_code = '''
```python
def factorial(n: int) -> int:
    """Calculate factorial of n."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 1
    return n * factorial(n - 1)
```
'''

    test_task = {
        "task_id": "test",
        "language": "python",
        "complexity": "simple",
    }

    result = evaluate_code_quality(test_code, test_task)
    print("Code quality:", result)

    test_review = """
    The plan looks generally good but has some issues:

    1. Missing edge case: doesn't handle when n is None
    2. Should add input validation at the start
    3. Consider using iteration instead of recursion for large n

    Recommend adding a check for n being negative.
    """

    test_plan = "Basic factorial implementation using recursion."

    critique = evaluate_critique_quality(test_review, test_plan)
    print("Critique quality:", critique)
