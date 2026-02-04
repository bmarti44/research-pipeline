"""
CLI wrappers for subscription-based model access (NO API KEYS).

This module provides wrappers for calling LLMs via subscription-based CLIs:
- claude: Claude Code CLI (Anthropic subscription)
- codex: Codex CLI (ChatGPT subscription)
- gemini: Gemini CLI (Google subscription)

IMPORTANT: Do NOT use SDK libraries (anthropic, openai, google) as they
require API keys. These wrappers use the CLIs which authenticate via
browser/OAuth through the user's subscription.
"""

import os
import subprocess
import shutil
from dataclasses import dataclass
from typing import Optional


@dataclass
class CLIResponse:
    """Response from a CLI model call."""

    model: str
    response: str
    success: bool
    error: Optional[str] = None
    exit_code: Optional[int] = None


def _check_api_keys_unset() -> None:
    """Verify API keys are unset to prevent accidental SDK usage."""
    problematic_keys = []
    for key in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"]:
        if os.environ.get(key):
            problematic_keys.append(key)

    if problematic_keys:
        import warnings

        warnings.warn(
            f"API keys detected: {problematic_keys}. "
            "This project uses subscription-based CLIs only. "
            "Consider unsetting these to avoid confusion.",
            UserWarning,
        )


def is_cli_available(cli_name: str) -> bool:
    """Check if a CLI tool is available on the system."""
    return shutil.which(cli_name) is not None


def call_claude(prompt: str, timeout: int = 120) -> CLIResponse:
    """
    Call Claude via Claude Code CLI (subscription).

    Args:
        prompt: The prompt to send to Claude
        timeout: Timeout in seconds (default 120)

    Returns:
        CLIResponse with model response or error
    """
    _check_api_keys_unset()

    if not is_cli_available("claude"):
        return CLIResponse(
            model="claude",
            response="",
            success=False,
            error="Claude CLI not found. Install via: npm install -g @anthropic-ai/claude-code",
        )

    try:
        result = subprocess.run(
            ["claude", "--print", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return CLIResponse(
            model="claude",
            response=result.stdout.strip(),
            success=result.returncode == 0,
            error=result.stderr.strip() if result.returncode != 0 else None,
            exit_code=result.returncode,
        )
    except subprocess.TimeoutExpired:
        return CLIResponse(
            model="claude",
            response="",
            success=False,
            error=f"Timeout after {timeout} seconds",
        )
    except Exception as e:
        return CLIResponse(
            model="claude",
            response="",
            success=False,
            error=str(e),
        )


def call_codex(prompt: str, timeout: int = 120) -> CLIResponse:
    """
    Call Codex via Codex CLI (ChatGPT subscription).

    Args:
        prompt: The prompt to send to Codex
        timeout: Timeout in seconds (default 120)

    Returns:
        CLIResponse with model response or error
    """
    _check_api_keys_unset()

    if not is_cli_available("codex"):
        return CLIResponse(
            model="codex",
            response="",
            success=False,
            error="Codex CLI not found",
        )

    try:
        result = subprocess.run(
            ["codex", "exec"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return CLIResponse(
            model="codex",
            response=result.stdout.strip(),
            success=result.returncode == 0,
            error=result.stderr.strip() if result.returncode != 0 else None,
            exit_code=result.returncode,
        )
    except subprocess.TimeoutExpired:
        return CLIResponse(
            model="codex",
            response="",
            success=False,
            error=f"Timeout after {timeout} seconds",
        )
    except Exception as e:
        return CLIResponse(
            model="codex",
            response="",
            success=False,
            error=str(e),
        )


def call_gemini(prompt: str, timeout: int = 120) -> CLIResponse:
    """
    Call Gemini via Gemini CLI (Google subscription).

    Args:
        prompt: The prompt to send to Gemini
        timeout: Timeout in seconds (default 120)

    Returns:
        CLIResponse with model response or error
    """
    _check_api_keys_unset()

    if not is_cli_available("gemini"):
        return CLIResponse(
            model="gemini",
            response="",
            success=False,
            error="Gemini CLI not found",
        )

    try:
        result = subprocess.run(
            ["gemini", "--prompt", prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return CLIResponse(
            model="gemini",
            response=result.stdout.strip(),
            success=result.returncode == 0,
            error=result.stderr.strip() if result.returncode != 0 else None,
            exit_code=result.returncode,
        )
    except subprocess.TimeoutExpired:
        return CLIResponse(
            model="gemini",
            response="",
            success=False,
            error=f"Timeout after {timeout} seconds",
        )
    except Exception as e:
        return CLIResponse(
            model="gemini",
            response="",
            success=False,
            error=str(e),
        )


def get_available_clis() -> dict[str, bool]:
    """
    Check which CLIs are available on the system.

    Returns:
        Dictionary mapping CLI names to availability status
    """
    return {
        "claude": is_cli_available("claude"),
        "codex": is_cli_available("codex"),
        "gemini": is_cli_available("gemini"),
    }
