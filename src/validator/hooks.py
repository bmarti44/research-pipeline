"""
PreToolUse and PostToolUse hooks for the Claude Agent SDK.

Uses ClaudeSDKClient (not bare query()) for hook support.
"""

from typing import Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

from .rules import RuleValidator, ValidationContext, Decision
from .convergence import (
    ConvergenceState,
    FeedbackLevel,
    get_feedback_message,
    FORCE_DIRECT_ANSWER_MESSAGE,
)


@dataclass
class ValidationEvent:
    """A single validation decision for audit logging."""
    timestamp: str
    tool_name: str
    tool_input: dict
    decision: str  # "allow" or "deny"
    rule_id: str | None
    reason: str
    feedback_level: str | None
    semantic_scores: dict  # Classifier scores if applicable


@dataclass
class HookState:
    """Shared state across hook invocations in a session."""
    validator: RuleValidator
    context: ValidationContext
    convergence: ConvergenceState
    rejections: list[dict] = field(default_factory=list)
    feedback_messages: list[str] = field(default_factory=list)
    # Enhanced audit logging
    validation_log: list[ValidationEvent] = field(default_factory=list)
    tool_call_sequence: list[dict] = field(default_factory=list)

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
        timestamp = datetime.now(timezone.utc).isoformat()
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})

        # Collect semantic scores for audit logging
        semantic_scores = {}
        if tool_name in ("WebSearch", "web_search"):
            _, static_score = state.validator.semantic.is_static_knowledge_query(state.context.user_query)
            _, memory_score = state.validator.semantic.is_memory_reference_query(state.context.user_query)
            query = tool_input.get("query", "")
            _, dup_score = state.validator.semantic.is_duplicate_search(query, state.context.search_queries)
            semantic_scores = {
                "static_knowledge": round(static_score, 4),
                "memory_reference": round(memory_score, 4),
                "duplicate_search": round(dup_score, 4),
            }

        # Check if we should force direct answer
        if state.convergence.should_force_direct_answer():
            event = ValidationEvent(
                timestamp=timestamp,
                tool_name=tool_name,
                tool_input=tool_input,
                decision="deny",
                rule_id="CONVERGENCE",
                reason="Forced direct answer after max rejections",
                feedback_level="FORCED",
                semantic_scores=semantic_scores,
            )
            state.validation_log.append(event)
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
                "tool_input": tool_input,
                "reason": result.reason,
                "feedback_level": feedback_level.name,
                "timestamp": timestamp,
            })

            # Log validation event
            event = ValidationEvent(
                timestamp=timestamp,
                tool_name=tool_name,
                tool_input=tool_input,
                decision="deny",
                rule_id=result.rule_id,
                reason=result.reason,
                feedback_level=feedback_level.name,
                semantic_scores=semantic_scores,
            )
            state.validation_log.append(event)

            # Return denial
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": feedback_msg,
                }
            }

        # Log allowed validation event
        event = ValidationEvent(
            timestamp=timestamp,
            tool_name=tool_name,
            tool_input=tool_input,
            decision="allow",
            rule_id=None,
            reason="All rules passed",
            feedback_level=None,
            semantic_scores=semantic_scores,
        )
        state.validation_log.append(event)

        # Track allowed tool for context
        state.context.tool_history.append(f"{tool_name}:{str(tool_input)[:50]}")

        # Track search queries for duplicate detection (F10, F20)
        if tool_name in ("WebSearch", "web_search"):
            query = tool_input.get("query", "")
            state.context.search_queries.append(query)

        # Track glob patterns for F17
        if tool_name in ("Glob", "glob", "find_files", "list_files"):
            pattern = tool_input.get("pattern", "") or tool_input.get("glob", "")
            if pattern:
                state.context.add_glob_pattern(pattern)

        # Log tool call sequence
        state.tool_call_sequence.append({
            "timestamp": timestamp,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "status": "started",
        })

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
        timestamp = datetime.now(timezone.utc).isoformat()
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {})
        tool_result = input_data.get("tool_result", {})

        # Log tool completion in sequence
        state.tool_call_sequence.append({
            "timestamp": timestamp,
            "tool_name": tool_name,
            "status": "completed",
            "result_type": type(tool_result).__name__,
            "result_length": len(str(tool_result)) if tool_result else 0,
        })

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
                # Log discovered paths
                state.tool_call_sequence[-1]["discovered_paths"] = paths[:20]  # Cap for size

        # F16: Track file reads
        if tool_name in ("Read", "View", "view_file", "read_file"):
            file_path = tool_input.get("file_path", "") or tool_input.get("path", "")
            if file_path:
                state.context.add_file_read(file_path)

        # F18: Track tool outputs for reverification detection
        output_summary = _summarize_tool_output(tool_name, tool_result)
        state.context.add_tool_output(tool_name, tool_input, output_summary)

        return {}  # Don't modify result

    return post_tool_use_hook


def _summarize_tool_output(tool_name: str, tool_result: Any) -> str:
    """Create a brief summary of tool output for F18 tracking."""
    if tool_result is None:
        return "empty"

    result_str = str(tool_result)
    if len(result_str) > 200:
        result_str = result_str[:200] + "..."

    # Detect success/failure indicators
    success_indicators = ["success", "created", "added", "committed", "written", "installed"]
    failure_indicators = ["error", "failed", "not found", "exception"]

    result_lower = result_str.lower()
    for indicator in success_indicators:
        if indicator in result_lower:
            return f"success:{indicator}"
    for indicator in failure_indicators:
        if indicator in result_lower:
            return f"failure:{indicator}"

    return f"completed:{len(result_str)}chars"


@dataclass
class BaselineToolOutput:
    """Record of a baseline tool call output."""
    tool_name: str
    tool_input: dict
    output_summary: str


@dataclass
class BaselineContext:
    """Tracking context for baseline trials (no validation, but tracks state for fair comparison)."""
    tool_history: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    known_paths: set[str] = field(default_factory=set)
    prior_searches: list[str] = field(default_factory=list)  # Pre-populated for F10 scenarios

    # F16: Track file reads
    read_files: list[str] = field(default_factory=list)

    # F17: Track glob patterns
    glob_patterns: list[str] = field(default_factory=list)

    # F18: Track recent tool outputs
    recent_outputs: list[BaselineToolOutput] = field(default_factory=list)

    def add_known_paths(self, paths: list[str]) -> None:
        self.known_paths.update(paths)

    def add_file_read(self, file_path: str) -> None:
        self.read_files.append(file_path)

    def add_glob_pattern(self, pattern: str) -> None:
        self.glob_patterns.append(pattern)

    def add_tool_output(self, tool_name: str, tool_input: dict, output_summary: str) -> None:
        self.recent_outputs.append(BaselineToolOutput(tool_name, tool_input, output_summary))
        if len(self.recent_outputs) > 5:
            self.recent_outputs = self.recent_outputs[-5:]

    def to_dict(self) -> dict:
        return {
            "tool_history": self.tool_history,
            "search_queries": self.search_queries,
            "known_paths": list(self.known_paths),
            "prior_searches": self.prior_searches,
            "read_files": self.read_files,
            "glob_patterns": self.glob_patterns,
        }


def create_logging_hooks(prior_searches: Optional[list[str]] = None):
    """Create hooks that log tool calls and track state (for baseline comparison).

    Args:
        prior_searches: Optional list of prior search queries for F10 scenario testing.
                       These are tracked to enable fair comparison with validated trials.
    """

    calls: list[dict] = []
    context = BaselineContext(prior_searches=prior_searches or [])

    # Initialize search_queries with prior_searches for F10 comparison
    if prior_searches:
        context.search_queries.extend(prior_searches)

    async def log_pre_tool_use(
        input_data: dict[str, Any],
        tool_use_id: str | None,
        hook_context: Any,
    ) -> dict[str, Any]:
        tool_name = input_data.get("tool_name")
        tool_input = input_data.get("tool_input", {})

        calls.append({
            "event": "pre",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool_name": tool_name,
            "tool_input": tool_input,
        })

        # Track tool in history
        context.tool_history.append(f"{tool_name}:{str(tool_input)[:50]}")

        # Track search queries for F10/F20 comparison
        if tool_name in ("WebSearch", "web_search"):
            query = tool_input.get("query", "")
            if query:
                context.search_queries.append(query)

        # Track glob patterns for F17 comparison
        if tool_name in ("Glob", "glob", "find_files", "list_files"):
            pattern = tool_input.get("pattern", "") or tool_input.get("glob", "")
            if pattern:
                context.add_glob_pattern(pattern)

        return {}  # Always allow

    async def log_post_tool_use(
        input_data: dict[str, Any],
        tool_use_id: str | None,
        hook_context: Any,
    ) -> dict[str, Any]:
        tool_name = input_data.get("tool_name")
        tool_input = input_data.get("tool_input", {})
        tool_result = input_data.get("tool_result")

        calls.append({
            "event": "post",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool_name": tool_name,
            "tool_result_type": type(tool_result).__name__,
        })

        # Track discovered paths for F13 comparison
        if tool_name in ("Glob", "LS", "glob", "ls", "list_directory"):
            paths = []
            if isinstance(tool_result, list):
                paths = [str(p) for p in tool_result if isinstance(p, str)]
            elif isinstance(tool_result, dict):
                for key in ("files", "paths", "entries", "results"):
                    if key in tool_result and isinstance(tool_result[key], list):
                        paths = [str(p) for p in tool_result[key] if isinstance(p, str)]
                        break
            elif isinstance(tool_result, str):
                paths = [p.strip() for p in tool_result.split("\n") if p.strip()]

            if paths:
                context.add_known_paths(paths)

        # F16: Track file reads
        if tool_name in ("Read", "View", "view_file", "read_file"):
            file_path = tool_input.get("file_path", "") or tool_input.get("path", "")
            if file_path:
                context.add_file_read(file_path)

        # F18: Track tool outputs for comparison
        output_summary = _summarize_tool_output(tool_name, tool_result)
        context.add_tool_output(tool_name, tool_input, output_summary)

        return {}

    return log_pre_tool_use, log_post_tool_use, calls, context
