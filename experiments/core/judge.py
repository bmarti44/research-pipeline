"""
LLM judge implementations.

Provides judges using either:
- API access (anthropic, openai, google-generativeai SDKs)
- CLI wrappers (fallback for subscription-only access)

Judge types:
1. claude_judge: Signal-type-specific judge using Claude
2. gpt4_judge: Signal-type-specific judge using GPT-4
3. gemini_judge: Signal-type-specific judge using Gemini
4. agnostic_judge: Signal-type-agnostic judge (any signal detection)

The agnostic judge addresses REVIEW.md section 3.3 by testing whether
format friction persists when the judge doesn't know the specific signal type.
"""

import os

from .api_providers import call_model as _call_model_api


class APIKeyError(Exception):
    """Raised when required API key is missing."""
    pass


class APICallError(Exception):
    """Raised when API call fails."""
    pass


def _call_judge(prompt: str, model: str = "claude-sonnet", timeout: int = 60) -> str:
    """
    Call a model for judging using API access.

    Args:
        prompt: The judge prompt
        model: Model to use (claude-sonnet, gpt-4o-mini, gemini-flash)
        timeout: Timeout in seconds

    Returns:
        Response text

    Raises:
        APIKeyError: If required API key is not set
        APICallError: If API call fails
    """
    # Check if relevant API key exists
    key_map = {
        "claude": "ANTHROPIC_API_KEY",
        "gpt": "OPENAI_API_KEY",
        "gemini": "GOOGLE_API_KEY",
    }

    required_key = None
    for prefix, env_var in key_map.items():
        if model.startswith(prefix):
            required_key = env_var
            break

    if required_key and not os.environ.get(required_key):
        raise APIKeyError(f"API key {required_key} not set. Set it in environment variables.")

    result = _call_model_api(prompt, system_prompt="", model=model, timeout=timeout)

    if not result.success:
        raise APICallError(f"API call failed: {result.error}")

    return result.response


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
) -> bool:
    """
    Claude judge via API.

    Args:
        query: The user query
        response: The assistant response
        signal_type: The expected signal type (e.g., "frustration", "urgency")
        timeout: Timeout in seconds

    Returns:
        True if signal acknowledged, False if not

    Raises:
        APIKeyError: If ANTHROPIC_API_KEY not set
        APICallError: If API call fails
    """
    prompt = JUDGE_PROMPT.format(
        signal_type=signal_type,
        query=query,
        response=response,
    )
    result = _call_judge(prompt, model="claude-sonnet", timeout=timeout)
    return "YES" in result.upper()


def gpt4_judge(
    query: str,
    response: str,
    signal_type: str,
    timeout: int = 60,
) -> bool:
    """
    GPT-4 judge via API.

    Args:
        query: The user query
        response: The assistant response
        signal_type: The expected signal type (e.g., "frustration", "urgency")
        timeout: Timeout in seconds

    Returns:
        True if signal acknowledged, False if not

    Raises:
        APIKeyError: If OPENAI_API_KEY not set
        APICallError: If API call fails
    """
    prompt = JUDGE_PROMPT.format(
        signal_type=signal_type,
        query=query,
        response=response,
    )
    result = _call_judge(prompt, model="gpt-4o-mini", timeout=timeout)
    return "YES" in result.upper()


def gemini_judge(
    query: str,
    response: str,
    signal_type: str,
    timeout: int = 60,
) -> bool:
    """
    Gemini judge via API.

    Args:
        query: The user query
        response: The assistant response
        signal_type: The expected signal type (e.g., "frustration", "urgency")
        timeout: Timeout in seconds

    Returns:
        True if signal acknowledged, False if not

    Raises:
        APIKeyError: If GOOGLE_API_KEY not set
        APICallError: If API call fails
    """
    prompt = JUDGE_PROMPT.format(
        signal_type=signal_type,
        query=query,
        response=response,
    )
    result = _call_judge(prompt, model="gemini-flash", timeout=timeout)
    return "YES" in result.upper()


def agnostic_judge(
    query: str,
    response: str,
    model: str = "claude-sonnet",
    timeout: int = 60,
) -> bool:
    """
    Signal-type-agnostic judge - detects ANY signal acknowledgment.

    This addresses REVIEW.md section 3.3 which raised concerns about
    signal-type specificity. This judge tests whether the model
    acknowledged any emotional signal, not a specific one.

    Args:
        query: The user query
        response: The assistant response
        model: Model to use for judging
        timeout: Timeout in seconds

    Returns:
        True if any signal acknowledged, False if not

    Raises:
        APIKeyError: If required API key not set
        APICallError: If API call fails
    """
    prompt = AGNOSTIC_JUDGE_PROMPT.format(
        query=query,
        response=response,
    )
    result = _call_judge(prompt, model=model, timeout=timeout)
    return "YES" in result.upper()


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
