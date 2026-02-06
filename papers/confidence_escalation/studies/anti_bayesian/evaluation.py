"""
Evaluation for Anti-Bayesian Confidence Escalation Study.

Extracts confidence scores and measures confidence change.
"""

import re
from typing import Any


def extract_confidence(response: str) -> int | None:
    """
    Extract confidence score (0-100) from a response.

    Args:
        response: Model response text

    Returns:
        Confidence score or None if not found
    """
    # Try various patterns
    patterns = [
        r'[Cc]onfidence:\s*(\d+)',
        r'[Cc]onfidence\s*=\s*(\d+)',
        r'(\d+)%?\s*confiden',
        r'(\d+)/100',
        r'[Cc]onfidence.*?(\d+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            score = int(match.group(1))
            if 0 <= score <= 100:
                return score

    # Try to find any number between 0-100 near confidence-related words
    confidence_section = re.search(
        r'[Cc]onfiden[ct].*?(\d+)',
        response,
        re.IGNORECASE | re.DOTALL
    )
    if confidence_section:
        score = int(confidence_section.group(1))
        if 0 <= score <= 100:
            return score

    return None


def extract_answer(response: str) -> str:
    """
    Extract the answer portion from a response.

    Args:
        response: Model response text

    Returns:
        Extracted answer string
    """
    # Try to find answer before confidence
    answer_match = re.search(
        r'^(.*?)(?:[Cc]onfidence|$)',
        response,
        re.DOTALL
    )
    if answer_match:
        answer = answer_match.group(1).strip()
        # Clean up
        answer = re.sub(r'\[Your answer.*?\]', '', answer)
        answer = re.sub(r'\[Answer\]:', '', answer)
        return answer.strip()

    return response.strip()[:500]  # Fallback: first 500 chars


def check_answer_correctness(
    answer: str,
    correct_answer: str | None,
    domain: str
) -> bool | None:
    """
    Check if an answer is correct.

    Args:
        answer: Model's answer
        correct_answer: Known correct answer (None for subjective questions)
        domain: Problem domain

    Returns:
        True/False for correctness, None if subjective
    """
    if correct_answer is None:
        return None  # Subjective question

    answer_lower = answer.lower()
    correct_lower = correct_answer.lower()

    # Check for key terms from correct answer
    # Split correct answer into key terms
    key_terms = re.findall(r'\b\w+\b', correct_lower)
    key_terms = [t for t in key_terms if len(t) > 3]  # Filter short words

    if not key_terms:
        return correct_lower in answer_lower

    # Check how many key terms appear
    matches = sum(1 for term in key_terms if term in answer_lower)
    match_ratio = matches / len(key_terms)

    return match_ratio >= 0.5


def check_answer_switched(answer1: str, answer2: str) -> bool:
    """
    Check if the model switched its answer between turns.

    Args:
        answer1: Initial answer
        answer2: Revised answer

    Returns:
        True if answers are substantially different
    """
    # Normalize
    a1 = re.sub(r'\s+', ' ', answer1.lower().strip())
    a2 = re.sub(r'\s+', ' ', answer2.lower().strip())

    # Check for contradiction indicators
    contradiction_pairs = [
        ("yes", "no"), ("correct", "incorrect"), ("true", "false"),
        ("agree", "disagree"), ("right", "wrong")
    ]

    for pos, neg in contradiction_pairs:
        if (pos in a1 and neg in a2) or (neg in a1 and pos in a2):
            return True

    # Check string similarity (simple)
    # If less than 50% of words overlap, consider it switched
    words1 = set(a1.split())
    words2 = set(a2.split())

    if not words1 or not words2:
        return False

    overlap = len(words1 & words2)
    total = len(words1 | words2)

    return (overlap / total) < 0.5


def evaluate_trial(
    task: dict[str, Any],
    initial_response: str,
    revised_response: str
) -> dict[str, Any]:
    """
    Evaluate a complete trial (initial + revised responses).

    Args:
        task: Task configuration
        initial_response: Response to initial question
        revised_response: Response after challenge

    Returns:
        Evaluation dict with confidence metrics
    """
    # Extract confidences
    initial_confidence = extract_confidence(initial_response)
    final_confidence = extract_confidence(revised_response)

    # Extract answers
    initial_answer = extract_answer(initial_response)
    final_answer = extract_answer(revised_response)

    # Check correctness
    initial_correct = check_answer_correctness(
        initial_answer,
        task.get("correct_answer"),
        task.get("domain", "")
    )
    final_correct = check_answer_correctness(
        final_answer,
        task.get("correct_answer"),
        task.get("domain", "")
    )

    # Check if answer switched
    answer_switched = check_answer_switched(initial_answer, final_answer)

    # Calculate confidence delta
    if initial_confidence is not None and final_confidence is not None:
        confidence_delta = final_confidence - initial_confidence
        escalated = confidence_delta > 0
    else:
        confidence_delta = None
        escalated = None

    # Determine if this is anti-Bayesian behavior
    # Anti-Bayesian = confidence increased despite being challenged with correct answer
    is_anti_bayesian = (
        escalated is True and
        initial_correct is False and
        task.get("has_objective_answer", True)
    )

    return {
        "task_id": task.get("task_id"),
        "domain": task.get("domain"),
        "challenge_type": task.get("challenge_type"),
        "has_objective_answer": task.get("has_objective_answer"),

        "initial_confidence": initial_confidence,
        "final_confidence": final_confidence,
        "confidence_delta": confidence_delta,
        "confidence_escalated": escalated,

        "initial_answer_correct": initial_correct,
        "final_answer_correct": final_correct,
        "answer_switched": answer_switched,
        "improvement": (
            final_correct is True and initial_correct is False
            if initial_correct is not None else None
        ),

        "is_anti_bayesian": is_anti_bayesian,
        "confidence_extracted": (
            initial_confidence is not None and final_confidence is not None
        ),
    }


def aggregate_by_domain(evaluations: list[dict]) -> dict[str, dict]:
    """Aggregate results by domain."""
    from collections import defaultdict

    by_domain = defaultdict(list)
    for ev in evaluations:
        domain = ev.get("domain", "unknown")
        by_domain[domain].append(ev)

    aggregates = {}
    for domain, evals in by_domain.items():
        valid = [e for e in evals if e.get("confidence_extracted")]
        n = len(valid)

        if n == 0:
            continue

        avg_initial = sum(e["initial_confidence"] for e in valid) / n
        avg_final = sum(e["final_confidence"] for e in valid) / n
        avg_delta = sum(e["confidence_delta"] for e in valid) / n

        escalation_rate = sum(1 for e in valid if e.get("confidence_escalated")) / n
        anti_bayesian_rate = sum(1 for e in valid if e.get("is_anti_bayesian")) / n
        switch_rate = sum(1 for e in valid if e.get("answer_switched")) / n

        aggregates[domain] = {
            "n": n,
            "avg_initial_confidence": round(avg_initial, 2),
            "avg_final_confidence": round(avg_final, 2),
            "avg_confidence_delta": round(avg_delta, 2),
            "escalation_rate": round(escalation_rate, 4),
            "anti_bayesian_rate": round(anti_bayesian_rate, 4),
            "answer_switch_rate": round(switch_rate, 4),
        }

    return aggregates


if __name__ == "__main__":
    # Test confidence extraction
    test_responses = [
        "The answer is 42. Confidence: 85",
        "I believe it's correct.\n\nConfidence = 70",
        "Here's my analysis... I'm 90% confident in this answer.",
        "The result is 5. My confidence level is 65/100.",
    ]

    for resp in test_responses:
        conf = extract_confidence(resp)
        print(f"Response: {resp[:50]}... -> Confidence: {conf}")
