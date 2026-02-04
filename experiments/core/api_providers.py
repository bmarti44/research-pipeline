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
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class Provider(Enum):
    CLAUDE = "claude"
    OPENAI = "openai"
    GEMINI = "gemini"


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
    """Standardized response from any provider."""
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    model: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


def _call_claude(
    prompt: str,
    system_prompt: str,
    config: ModelConfig,
    timeout: int = 60,
) -> APIResponse:
    """Call Claude via Anthropic API."""
    try:
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return APIResponse(success=False, error="ANTHROPIC_API_KEY not set")

        client = anthropic.Anthropic(api_key=api_key, timeout=timeout)

        message = client.messages.create(
            model=config.model_id,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = message.content[0].text if message.content else ""

        return APIResponse(
            success=True,
            response=response_text,
            model=config.model_id,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )

    except Exception as e:
        return APIResponse(success=False, error=str(e))


def _call_openai(
    prompt: str,
    system_prompt: str,
    config: ModelConfig,
    timeout: int = 60,
) -> APIResponse:
    """Call OpenAI via API."""
    try:
        import openai

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return APIResponse(success=False, error="OPENAI_API_KEY not set")

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

        response_text = response.choices[0].message.content if response.choices else ""

        return APIResponse(
            success=True,
            response=response_text,
            model=config.model_id,
            input_tokens=response.usage.prompt_tokens if response.usage else None,
            output_tokens=response.usage.completion_tokens if response.usage else None,
        )

    except Exception as e:
        return APIResponse(success=False, error=str(e))


def _call_gemini(
    prompt: str,
    system_prompt: str,
    config: ModelConfig,
    timeout: int = 60,
) -> APIResponse:
    """Call Gemini via Google AI API."""
    try:
        import google.generativeai as genai

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return APIResponse(success=False, error="GOOGLE_API_KEY not set")

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
        response_text = response.text if response.text else ""

        # Token counts from Gemini
        input_tokens = None
        output_tokens = None
        if hasattr(response, 'usage_metadata'):
            input_tokens = getattr(response.usage_metadata, 'prompt_token_count', None)
            output_tokens = getattr(response.usage_metadata, 'candidates_token_count', None)

        return APIResponse(
            success=True,
            response=response_text,
            model=config.model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    except Exception as e:
        return APIResponse(success=False, error=str(e))


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
