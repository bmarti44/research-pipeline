"""
PreToolUse and PostToolUse hooks for the Claude Agent SDK.

Uses ClaudeSDKClient (not bare query()) for hook support.
"""

from typing import Any, Optional
from dataclasses import dataclass, field

from .rules import RuleValidator, ValidationContext, Decision
from .convergence import (
    ConvergenceState,
    FeedbackLevel,
    get_feedback_message,
    FORCE_DIRECT_ANSWER_MESSAGE,
)


@dataclass
class HookState:
    """Shared state across hook invocations in a session."""
    validator: RuleValidator
    context: ValidationContext
    convergence: ConvergenceState
    rejections: list[dict] = field(default_factory=list)
    feedback_messages: list[str] = field(default_factory=list)

    @classmethod
    def create(cls, user_query: str, validator: Optional[RuleValidator] = None) -> "HookState":
        return cls(
            validator=validator or RuleValidator(),
            context=ValidationContext(user_query=user_query),
            convergence=ConvergenceState(),
        )

    def get_pending_feedback(self) -> str | None:
        """Get and clear pending feedback messages."""
        if self.convergence.should_force_direct_answer():
            return FORCE_DIRECT_ANSWER_MESSAGE

        if self.feedback_messages:
            msg = "\n\n---\n\n".join(self.feedback_messages)
            self.feedback_messages.clear()
            return msg
        return None


def create_pre_tool_use_hook(state: HookState):
    """
    Create a PreToolUse hook that validates tool calls.

    Returns:
    - {}: Allow the tool call
    - {"hookSpecificOutput": {...}}: Deny with reason
    """

    async def pre_tool_use_hook(
        input_data: dict[str, Any],
        tool_use_id: str | None,
        context: Any,
    ) -> dict[str, Any]:
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})

        # Check if we should force direct answer
        if state.convergence.should_force_direct_answer():
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": FORCE_DIRECT_ANSWER_MESSAGE,
                }
            }

        # Validate
        result = state.validator.validate(
            tool_name,
            tool_input,
            state.context,
        )

        if result.decision == Decision.DENY:
            # Record rejection and get feedback level
            feedback_level = state.convergence.record_rejection(result.rule_id)

            # Get appropriate feedback message
            feedback_msg = get_feedback_message(
                result.rule_id, feedback_level, result.reason
            )
            state.feedback_messages.append(feedback_msg)

            # Track rejection
            state.rejections.append({
                "rule_id": result.rule_id,
                "tool_name": tool_name,
                "reason": result.reason,
                "feedback_level": feedback_level.name,
            })

            # Return denial
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": feedback_msg,
                }
            }

        # Track allowed tool for context
        state.context.tool_history.append(f"{tool_name}:{str(tool_input)[:50]}")

        # Track search queries for duplicate detection
        if tool_name in ("WebSearch", "web_search"):
            query = tool_input.get("query", "")
            state.context.search_queries.append(query)

        return {}  # Allow

    return pre_tool_use_hook


def create_post_tool_use_hook(state: HookState):
    """
    Create a PostToolUse hook to track discovered paths.

    This is critical for F13 (hallucinated path) to work correctly.
    """

    async def post_tool_use_hook(
        input_data: dict[str, Any],
        tool_use_id: str | None,
        context: Any,
    ) -> dict[str, Any]:
        tool_name = input_data.get("tool_name", "")
        tool_result = input_data.get("tool_result", {})

        # Track paths discovered via Glob/LS
        if tool_name in ("Glob", "LS", "glob", "ls", "list_directory"):
            # Extract paths from result
            # Result format varies; try common patterns
            paths = []

            if isinstance(tool_result, list):
                paths = [str(p) for p in tool_result if isinstance(p, str)]
            elif isinstance(tool_result, dict):
                # Some tools return {"files": [...]} or {"paths": [...]}
                for key in ("files", "paths", "entries", "results"):
                    if key in tool_result and isinstance(tool_result[key], list):
                        paths = [str(p) for p in tool_result[key] if isinstance(p, str)]
                        break
            elif isinstance(tool_result, str):
                # Might be newline-separated paths
                paths = [p.strip() for p in tool_result.split("\n") if p.strip()]

            if paths:
                state.context.add_known_paths(paths)

        return {}  # Don't modify result

    return post_tool_use_hook


def create_logging_hooks():
    """Create hooks that just log tool calls (for baseline comparison)."""

    calls: list[dict] = []

    async def log_pre_tool_use(
        input_data: dict[str, Any],
        tool_use_id: str | None,
        context: Any,
    ) -> dict[str, Any]:
        calls.append({
            "event": "pre",
            "tool_name": input_data.get("tool_name"),
            "tool_input": input_data.get("tool_input"),
        })
        return {}  # Always allow

    async def log_post_tool_use(
        input_data: dict[str, Any],
        tool_use_id: str | None,
        context: Any,
    ) -> dict[str, Any]:
        calls.append({
            "event": "post",
            "tool_name": input_data.get("tool_name"),
        })
        return {}

    return log_pre_tool_use, log_post_tool_use, calls
