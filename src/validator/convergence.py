"""
Convergence control for validator feedback loops.

Prevents infinite loops by:
1. Tracking rejections per rule
2. Escalating feedback specificity
3. Forcing direct answer after threshold
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FeedbackLevel(Enum):
    """Escalating feedback specificity."""
    GENTLE = 1      # "Consider answering directly"
    DIRECT = 2      # "Do not use WebSearch for this"
    FORCEFUL = 3    # "You MUST answer without tools"


@dataclass
class RuleViolation:
    """Track violations for a specific rule."""
    rule_id: str
    count: int = 0
    last_feedback_level: FeedbackLevel = FeedbackLevel.GENTLE


@dataclass
class ConvergenceState:
    """Tracks convergence across a session."""
    max_rejections_per_rule: int = 2
    max_total_rejections: int = 5

    violations: dict[str, RuleViolation] = field(default_factory=dict)
    total_rejections: int = 0
    forced_direct_answer: bool = False
    termination_reason: Optional[str] = None

    def record_rejection(self, rule_id: str) -> FeedbackLevel:
        """Record a rejection and return appropriate feedback level."""
        self.total_rejections += 1

        if rule_id not in self.violations:
            self.violations[rule_id] = RuleViolation(rule_id=rule_id)

        violation = self.violations[rule_id]
        violation.count += 1

        # Escalate feedback based on count
        if violation.count == 1:
            violation.last_feedback_level = FeedbackLevel.GENTLE
        elif violation.count == 2:
            violation.last_feedback_level = FeedbackLevel.DIRECT
        else:
            violation.last_feedback_level = FeedbackLevel.FORCEFUL

        # Check termination conditions
        if violation.count > self.max_rejections_per_rule:
            self.forced_direct_answer = True
            self.termination_reason = f"Rule {rule_id} violated {violation.count} times"

        if self.total_rejections > self.max_total_rejections:
            self.forced_direct_answer = True
            self.termination_reason = f"Total rejections ({self.total_rejections}) exceeded limit"

        return violation.last_feedback_level

    def should_force_direct_answer(self) -> bool:
        """Check if we should force a direct answer."""
        return self.forced_direct_answer


FEEDBACK_TEMPLATES = {
    FeedbackLevel.GENTLE: {
        "F1": "This appears to be a factual question you can answer from your knowledge. Consider responding directly.",
        "F4": "This seems to reference our past conversations. Consider using memory search instead of web search.",
        "F8": "This query mentions location. Consider getting the user's location first.",
        "F10": "You've already searched for something similar. Consider using the previous results.",
        "F13": "That file path wasn't mentioned by the user or discovered via listing. Consider listing the directory first.",
        "F15": "That appears to be a binary file. Consider explaining you cannot read binary files.",
    },
    FeedbackLevel.DIRECT: {
        "F1": "Do NOT use WebSearch. Answer this factual question directly from your knowledge.",
        "F4": "Do NOT use WebSearch. Use memory/conversation search for questions about our past discussions.",
        "F8": "Do NOT search yet. First get the user's location for this location-dependent query.",
        "F10": "Do NOT repeat the search. Use the results you already have.",
        "F13": "Do NOT read that path. List the directory first to confirm it exists.",
        "F15": "Do NOT attempt to read binary files. Explain this limitation to the user.",
    },
    FeedbackLevel.FORCEFUL: {
        "F1": "STOP. You MUST answer this question directly WITHOUT any tools. This is static knowledge.",
        "F4": "STOP. You MUST NOT use web search for memory questions. Answer based on conversation context.",
        "F8": "STOP. You MUST get location before searching. Use get_location tool first.",
        "F10": "STOP. You MUST use existing search results. Do not search again.",
        "F13": "STOP. You MUST list the directory before reading unknown paths.",
        "F15": "STOP. Binary files cannot be read. Explain this to the user.",
    },
}


def get_feedback_message(rule_id: str, level: FeedbackLevel, reason: str) -> str:
    """Get the appropriate feedback message for a rule violation."""
    template = FEEDBACK_TEMPLATES.get(level, {}).get(rule_id)
    if template:
        return f"{template}\n\nDetails: {reason}"
    return f"Validation failed: {reason}"


FORCE_DIRECT_ANSWER_MESSAGE = """
CRITICAL: You have repeatedly attempted invalid tool calls. You MUST now respond
directly to the user's question using ONLY your existing knowledge. Do NOT attempt
any tool calls. Provide your best answer based on what you know.
"""
