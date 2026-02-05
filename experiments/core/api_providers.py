"""
Unified API interface for multiple LLM providers.

Supports:
- Anthropic Claude (via anthropic SDK)
- OpenAI GPT (via openai SDK)
- Google Gemini (via google-generativeai SDK)

Environment variables:
- ANTHROPIC_API_KEY: Anthropic API key
- OPENAI_API_KEY: OpenAI API key
- GOOGLE_API_KEY: Google AI API key
"""

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Callable, Any
from enum import Enum


class Provider(Enum):
    CLAUDE = "claude"
    OPENAI = "openai"
    GEMINI = "gemini"


@dataclass
class RetryConfig:
    """Configuration for API retry logic with exponential backoff."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    provider: Provider
    model_id: str
    max_tokens: int = 2048
    temperature: float = 1.0


# Default model configurations
MODELS = {
    "claude-sonnet": ModelConfig(Provider.CLAUDE, "claude-sonnet-4-20250514"),
    "claude-haiku": ModelConfig(Provider.CLAUDE, "claude-3-5-haiku-latest"),
    "gpt-4o": ModelConfig(Provider.OPENAI, "gpt-4o"),
    "gpt-4o-mini": ModelConfig(Provider.OPENAI, "gpt-4o-mini"),
    "gemini-flash": ModelConfig(Provider.GEMINI, "gemini-2.0-flash"),
    "gemini-pro": ModelConfig(Provider.GEMINI, "gemini-1.5-pro"),
}


@dataclass
class APIResponse:
    """Standardized response from any provider with full metadata for reproducibility."""
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    model: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    # Metadata for reproducibility (PLAN.md requirement)
    request_id: Optional[str] = None
    timestamp: Optional[str] = None
    latency_ms: Optional[int] = None
    retry_count: int = 0
    finish_reason: Optional[str] = None
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset_at: Optional[str] = None


def _call_claude(
    prompt: str,
    system_prompt: str,
    config: ModelConfig,
    timeout: int = 60,
) -> APIResponse:
    """Call Claude via Anthropic API with full metadata capture."""
    timestamp = datetime.now(timezone.utc).isoformat()
    start_time = time.perf_counter()

    try:
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return APIResponse(success=False, error="ANTHROPIC_API_KEY not set", timestamp=timestamp)

        client = anthropic.Anthropic(api_key=api_key, timeout=timeout)

        message = client.messages.create(
            model=config.model_id,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        response_text = message.content[0].text if message.content else ""

        return APIResponse(
            success=True,
            response=response_text,
            model=config.model_id,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            request_id=getattr(message, 'id', None),
            timestamp=timestamp,
            latency_ms=latency_ms,
            finish_reason=message.stop_reason if hasattr(message, 'stop_reason') else None,
        )

    except Exception as e:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        return APIResponse(success=False, error=str(e), timestamp=timestamp, latency_ms=latency_ms)


def _call_openai(
    prompt: str,
    system_prompt: str,
    config: ModelConfig,
    timeout: int = 60,
) -> APIResponse:
    """Call OpenAI via API with full metadata capture."""
    timestamp = datetime.now(timezone.utc).isoformat()
    start_time = time.perf_counter()

    try:
        import openai

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return APIResponse(success=False, error="OPENAI_API_KEY not set", timestamp=timestamp)

        client = openai.OpenAI(api_key=api_key, timeout=timeout)

        response = client.chat.completions.create(
            model=config.model_id,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        response_text = response.choices[0].message.content if response.choices else ""
        finish_reason = response.choices[0].finish_reason if response.choices else None

        return APIResponse(
            success=True,
            response=response_text,
            model=config.model_id,
            input_tokens=response.usage.prompt_tokens if response.usage else None,
            output_tokens=response.usage.completion_tokens if response.usage else None,
            request_id=response.id if hasattr(response, 'id') else None,
            timestamp=timestamp,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
        )

    except Exception as e:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        return APIResponse(success=False, error=str(e), timestamp=timestamp, latency_ms=latency_ms)


def _call_gemini(
    prompt: str,
    system_prompt: str,
    config: ModelConfig,
    timeout: int = 60,
) -> APIResponse:
    """Call Gemini via Google AI API with full metadata capture."""
    timestamp = datetime.now(timezone.utc).isoformat()
    start_time = time.perf_counter()

    try:
        import google.generativeai as genai

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return APIResponse(success=False, error="GOOGLE_API_KEY not set", timestamp=timestamp)

        genai.configure(api_key=api_key)

        model = genai.GenerativeModel(
            model_name=config.model_id,
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=config.max_tokens,
                temperature=config.temperature,
            ),
        )

        response = model.generate_content(prompt)
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        response_text = response.text if response.text else ""

        # Token counts from Gemini
        input_tokens = None
        output_tokens = None
        if hasattr(response, 'usage_metadata'):
            input_tokens = getattr(response.usage_metadata, 'prompt_token_count', None)
            output_tokens = getattr(response.usage_metadata, 'candidates_token_count', None)

        # Get finish reason from candidates
        finish_reason = None
        if hasattr(response, 'candidates') and response.candidates:
            finish_reason = getattr(response.candidates[0], 'finish_reason', None)
            if finish_reason:
                finish_reason = str(finish_reason)

        return APIResponse(
            success=True,
            response=response_text,
            model=config.model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            timestamp=timestamp,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
        )

    except Exception as e:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        return APIResponse(success=False, error=str(e), timestamp=timestamp, latency_ms=latency_ms)


def call_model(
    prompt: str,
    system_prompt: str = "",
    model: str = "claude-sonnet",
    timeout: int = 60,
) -> APIResponse:
    """
    Call any supported model with a unified interface.

    Args:
        prompt: User message/prompt
        system_prompt: System instructions
        model: Model name (see MODELS dict for options)
        timeout: Request timeout in seconds

    Returns:
        APIResponse with success status and response text
    """
    if model not in MODELS:
        return APIResponse(
            success=False,
            error=f"Unknown model: {model}. Available: {list(MODELS.keys())}"
        )

    config = MODELS[model]

    if config.provider == Provider.CLAUDE:
        return _call_claude(prompt, system_prompt, config, timeout)
    elif config.provider == Provider.OPENAI:
        return _call_openai(prompt, system_prompt, config, timeout)
    elif config.provider == Provider.GEMINI:
        return _call_gemini(prompt, system_prompt, config, timeout)
    else:
        return APIResponse(success=False, error=f"Unsupported provider: {config.provider}")


def call_model_with_retry(
    prompt: str,
    system_prompt: str = "",
    model: str = "claude-sonnet",
    timeout: int = 60,
    retry_config: Optional[RetryConfig] = None,
) -> APIResponse:
    """
    Call model with exponential backoff retry on rate limit errors.

    Args:
        prompt: User message/prompt
        system_prompt: System instructions
        model: Model name (see MODELS dict for options)
        timeout: Request timeout in seconds
        retry_config: Configuration for retry behavior

    Returns:
        APIResponse with success status, response text, and retry_count
    """
    if retry_config is None:
        retry_config = RetryConfig()

    last_response: Optional[APIResponse] = None

    for attempt in range(retry_config.max_retries + 1):
        response = call_model(prompt, system_prompt, model, timeout)

        if response.success:
            response.retry_count = attempt
            return response

        last_response = response

        # Check if error is rate-limit related
        error_lower = (response.error or "").lower()
        is_rate_limit = any(term in error_lower for term in [
            "rate limit", "rate_limit", "ratelimit",
            "429", "too many requests", "quota exceeded"
        ])

        if not is_rate_limit:
            # Not a rate limit error, don't retry
            response.retry_count = attempt
            return response

        if attempt < retry_config.max_retries:
            delay = min(
                retry_config.base_delay * (retry_config.exponential_base ** attempt),
                retry_config.max_delay
            )
            time.sleep(delay)

    # Max retries exceeded
    if last_response:
        last_response.retry_count = retry_config.max_retries
        last_response.error = f"Max retries exceeded: {last_response.error}"
        return last_response

    return APIResponse(
        success=False,
        error="Max retries exceeded",
        retry_count=retry_config.max_retries,
    )


def check_api_keys() -> dict[str, bool]:
    """Check which API keys are configured."""
    return {
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "google": bool(os.environ.get("GOOGLE_API_KEY")),
    }


def list_available_models() -> list[str]:
    """List models with available API keys."""
    keys = check_api_keys()
    available = []

    for name, config in MODELS.items():
        if config.provider == Provider.CLAUDE and keys["anthropic"]:
            available.append(name)
        elif config.provider == Provider.OPENAI and keys["openai"]:
            available.append(name)
        elif config.provider == Provider.GEMINI and keys["google"]:
            available.append(name)

    return available
