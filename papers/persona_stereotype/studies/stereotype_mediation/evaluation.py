"""
Evaluation for Persona Stereotype Mediation Study.

Evaluates reasoning accuracy on MMLU-style questions.
"""

import re
from typing import Any


def extract_answer(response: str) -> str | None:
    """
    Extract the selected answer (A, B, C, or D) from response.

    Args:
        response: Model response text

    Returns:
        Answer letter or None
    """
    # Direct answer patterns
    patterns = [
        r'\b([ABCD])\)',  # A)
        r'\bAnswer[:\s]+([ABCD])\b',
        r'\bselect[:\s]+([ABCD])\b',
        r'\bcorrect answer is[:\s]+([ABCD])\b',
        r'\bThe answer is[:\s]+([ABCD])\b',
        r'^([ABCD])[\.\)\s]',
        r'\b([ABCD])\s+is (?:the )?correct',
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()

    # Look for standalone letter at start of response
    first_line = response.strip().split('\n')[0]
    if first_line.strip().upper() in ['A', 'B', 'C', 'D']:
        return first_line.strip().upper()

    # Check for letter followed by description
    for letter in ['A', 'B', 'C', 'D']:
        if response.strip().upper().startswith(letter):
            return letter

    return None


def check_persona_maintained(response: str, persona_prompt: str) -> bool:
    """
    Check if the response maintains the assigned persona.

    Args:
        response: Model response
        persona_prompt: The persona system prompt

    Returns:
        True if persona seems maintained
    """
    response_lower = response.lower()

    # Check for persona-breaking phrases
    breaking_phrases = [
        "as an ai",
        "i'm an ai",
        "i am an ai",
        "i don't have personal",
        "i cannot roleplay",
        "i should clarify",
    ]

    for phrase in breaking_phrases:
        if phrase in response_lower:
            return False

    # If no persona was assigned, consider it maintained
    if "no persona" in persona_prompt.lower() or "helpful ai assistant" in persona_prompt.lower():
        return True

    # For personas, check for any persona-consistent language
    # (This is a rough heuristic)
    return True


def evaluate_response(task: dict[str, Any], response: str) -> dict[str, Any]:
    """
    Evaluate a reasoning response.

    Args:
        task: Task configuration with question and correct answer
        response: Model response

    Returns:
        Evaluation dict
    """
    # Extract answer
    selected_answer = extract_answer(response)

    # Check correctness
    correct = (
        selected_answer is not None and
        selected_answer.upper() == task["correct_answer"].upper()
    )

    # Check persona maintenance
    persona_maintained = check_persona_maintained(
        response,
        task.get("persona_prompt", "")
    )

    # Response quality indicators
    response_length = len(response)
    has_reasoning = any(word in response.lower() for word in [
        "because", "therefore", "since", "thus", "so ", "means"
    ])

    return {
        "task_id": task.get("task_id"),
        "demographic": task.get("demographic"),
        "condition": task.get("condition"),
        "domain": task.get("domain"),

        "correct": correct,
        "selected_answer": selected_answer,
        "expected_answer": task["correct_answer"],
        "answer_extracted": selected_answer is not None,

        "persona_maintained": persona_maintained,
        "has_reasoning": has_reasoning,
        "response_length": response_length,
    }


def aggregate_by_condition(evaluations: list[dict]) -> dict[str, dict]:
    """Aggregate results by persona condition."""
    from collections import defaultdict

    by_condition = defaultdict(list)
    for ev in evaluations:
        cond = ev.get("condition", "unknown")
        by_condition[cond].append(ev)

    aggregates = {}
    for cond, evals in by_condition.items():
        n = len(evals)
        if n == 0:
            continue

        correct = sum(1 for e in evals if e.get("correct"))
        extracted = sum(1 for e in evals if e.get("answer_extracted"))
        persona_kept = sum(1 for e in evals if e.get("persona_maintained"))

        aggregates[cond] = {
            "n": n,
            "accuracy": round(correct / n, 4),
            "extraction_rate": round(extracted / n, 4),
            "persona_maintenance_rate": round(persona_kept / n, 4),
        }

    return aggregates


def aggregate_by_demographic(evaluations: list[dict]) -> dict[str, dict]:
    """Aggregate results by demographic."""
    from collections import defaultdict

    by_demo = defaultdict(list)
    for ev in evaluations:
        demo = ev.get("demographic", "unknown")
        by_demo[demo].append(ev)

    aggregates = {}
    for demo, evals in by_demo.items():
        n = len(evals)
        if n == 0:
            continue

        correct = sum(1 for e in evals if e.get("correct"))

        # Calculate accuracy by condition within this demographic
        by_cond = defaultdict(list)
        for ev in evals:
            by_cond[ev.get("condition", "unknown")].append(ev)

        cond_accuracies = {}
        for cond, cond_evals in by_cond.items():
            cond_correct = sum(1 for e in cond_evals if e.get("correct"))
            cond_accuracies[cond] = round(cond_correct / len(cond_evals), 4) if cond_evals else 0

        aggregates[demo] = {
            "n": n,
            "overall_accuracy": round(correct / n, 4),
            "by_condition": cond_accuracies,
        }

    return aggregates


def calculate_degradation_effect(aggregates: dict[str, dict]) -> dict[str, Any]:
    """
    Calculate degradation effect sizes.

    Args:
        aggregates: Condition-level aggregates

    Returns:
        Effect size analysis
    """
    baseline = aggregates.get("no_persona", {}).get("accuracy", 0)

    effects = {}
    for cond, data in aggregates.items():
        if cond == "no_persona":
            continue

        accuracy = data.get("accuracy", 0)
        degradation = baseline - accuracy

        effects[cond] = {
            "accuracy": accuracy,
            "degradation_from_baseline": round(degradation, 4),
            "degradation_percent": round(degradation / baseline * 100, 2) if baseline > 0 else 0,
        }

    # Key comparisons for mechanism test
    if "demographic_only" in effects and "counter_stereotypical" in effects:
        rescue_effect = (
            effects["counter_stereotypical"]["degradation_from_baseline"] -
            effects["demographic_only"]["degradation_from_baseline"]
        )
        effects["counter_stereotypical_rescue"] = round(rescue_effect, 4)

    if "nonsense_control" in effects:
        cognitive_load_effect = effects["nonsense_control"]["degradation_from_baseline"]
        effects["cognitive_load_only"] = round(cognitive_load_effect, 4)
        # If cognitive load effect is near zero but demographic effect is large,
        # this supports stereotype activation mechanism
        effects["supports_stereotype_mechanism"] = (
            abs(cognitive_load_effect) < 0.05 and
            effects.get("demographic_only", {}).get("degradation_from_baseline", 0) > 0.10
        )

    return effects


if __name__ == "__main__":
    # Test answer extraction
    test_responses = [
        "The answer is B) 5",
        "A) This is correct because...",
        "I believe the correct answer is C",
        "After analysis, D is the right choice.",
    ]

    for resp in test_responses:
        answer = extract_answer(resp)
        print(f"Response: {resp[:40]}... -> Answer: {answer}")
