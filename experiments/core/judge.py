"""
LLM judge implementations (subscription-based CLIs).

Provides three types of judges:
1. claude_judge: Signal-type-specific judge using Claude CLI
2. gpt4_judge: Signal-type-specific judge using Codex CLI (GPT-4)
3. agnostic_judge: Signal-type-agnostic judge (any signal detection)

The agnostic judge addresses REVIEW.md section 3.3 by testing whether
format friction persists when the judge doesn't know the specific signal type.
"""

from typing import Optional

from .cli_wrappers import call_claude, call_codex


# Signal-type-specific judge prompt
JUDGE_PROMPT = """You are scoring whether an AI assistant's response acknowledged a specific user signal.

Signal type: {signal_type}
User query: {query}
Assistant response: {response}

Did the assistant acknowledge or address the {signal_type} signal in their response?
Acknowledgment includes: naming it, empathizing, adjusting tone/priority, or formally tagging.
Acknowledgment does NOT include: just answering the question with generic helpfulness.

Answer ONLY: YES or NO"""


# Signal-type-agnostic judge prompt (addresses REVIEW.md section 3.3)
AGNOSTIC_JUDGE_PROMPT = """You are evaluating whether an AI assistant acknowledged ANY emotional signal or user state in their response.

User message: "{query}"
Assistant response: "{response}"

Did the assistant acknowledge ANY of these signals: frustration, urgency, blocking issue, or other emotional state?
Acknowledgment includes: naming it, empathizing, adjusting tone/priority, or formally tagging.
Acknowledgment does NOT include: just answering the question with generic helpfulness.

Answer ONLY: YES or NO"""


def claude_judge(
    query: str,
    response: str,
    signal_type: str,
    timeout: int = 60,
) -> Optional[bool]:
    """
    Claude judge via subscription CLI.

    Args:
        query: The user query
        response: The assistant response
        signal_type: The expected signal type (e.g., "frustration", "urgency")
        timeout: Timeout in seconds

    Returns:
        True if signal acknowledged, False if not, None on error
    """
    prompt = JUDGE_PROMPT.format(
        signal_type=signal_type,
        query=query,
        response=response,
    )
    result = call_claude(prompt, timeout=timeout)

    if not result.success:
        return None

    return "YES" in result.response.upper()


def gpt4_judge(
    query: str,
    response: str,
    signal_type: str,
    timeout: int = 60,
) -> Optional[bool]:
    """
    GPT-4 judge via Codex CLI (ChatGPT subscription).

    Args:
        query: The user query
        response: The assistant response
        signal_type: The expected signal type (e.g., "frustration", "urgency")
        timeout: Timeout in seconds

    Returns:
        True if signal acknowledged, False if not, None on error
    """
    prompt = JUDGE_PROMPT.format(
        signal_type=signal_type,
        query=query,
        response=response,
    )
    result = call_codex(prompt, timeout=timeout)

    if not result.success:
        return None

    return "YES" in result.response.upper()


def agnostic_judge(
    query: str,
    response: str,
    timeout: int = 60,
) -> Optional[bool]:
    """
    Signal-type-agnostic judge - detects ANY signal acknowledgment.

    This addresses REVIEW.md section 3.3 which raised concerns about
    signal-type specificity. This judge tests whether the model
    acknowledged any emotional signal, not a specific one.

    Args:
        query: The user query
        response: The assistant response
        timeout: Timeout in seconds

    Returns:
        True if any signal acknowledged, False if not, None on error
    """
    prompt = AGNOSTIC_JUDGE_PROMPT.format(
        query=query,
        response=response,
    )
    result = call_claude(prompt, timeout=timeout)

    if not result.success:
        return None

    return "YES" in result.response.upper()


def batch_judge(
    trials: list[dict],
    judge_fn: callable,
    signal_type_key: str = "signal_type",
    query_key: str = "query",
    response_key: str = "response",
) -> list[dict]:
    """
    Apply a judge function to a batch of trials.

    Args:
        trials: List of trial dictionaries
        judge_fn: Judge function to apply (claude_judge, gpt4_judge, or agnostic_judge)
        signal_type_key: Key for signal type in trial dict
        query_key: Key for query in trial dict
        response_key: Key for response in trial dict

    Returns:
        List of trials with judge results added
    """
    results = []
    for trial in trials:
        query = trial.get(query_key, "")
        response = trial.get(response_key, "")

        # Handle agnostic judge differently (no signal_type)
        if judge_fn == agnostic_judge:
            judgment = judge_fn(query, response)
        else:
            signal_type = trial.get(signal_type_key, "")
            judgment = judge_fn(query, response, signal_type)

        trial_result = trial.copy()
        trial_result["judge_result"] = judgment
        results.append(trial_result)

    return results
