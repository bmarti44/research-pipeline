"""
Experiment: Does natural language intent outperform structured tool calling?

Research Question: With identical guidance on WHEN to save and equally explicit
instructions on HOW to save, does output format affect the model's ability to
recognize when to save information from VAGUE user messages?

Hypothesis: Structured tool calling may impose cognitive overhead that suppresses
action, even when the model would express intent naturally. If true, this supports
a two-stage architecture: NL intent → structured extraction.

DESIGN (v4 - Simplified):
- WHEN to save: Identical guidance for both conditions
- HOW to save: Equally explicit examples for both conditions
  - NL: Natural English ("I'll save X to the codebase category")
  - Structured: Tool call (save-memory "content" "category")
- SCENARIOS:
  - Implicit: Very vague statements ("pg is db", "app.ts in ./src")
  - Control: Should NOT save (questions, chitchat, temporary info)
  - 1 Explicit: Sanity check (both conditions should succeed)

KEY INSIGHT: Intermediate levels (weak/moderate/strong) add noise without
testing the core hypothesis. The interesting question is whether NL handles
truly ambiguous cases better than structured tool calling.

FIDELITY: Also measures quality of saved content (completeness, context, specificity)
to test whether NL captures higher-fidelity information when it does save.
"""

import asyncio
import json
import random
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from collections import defaultdict

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    HookMatcher,
    AssistantMessage,
    TextBlock,
    ToolUseBlock,
    ResultMessage,
)

from experiments.scenarios.proactive_tools import (
    MEMORY_SCENARIOS as PROACTIVE_MEMORY_SCENARIOS,
    CONTROL_SCENARIOS as PROACTIVE_CONTROL_SCENARIOS,
    ExplicitnessLevel,
)


# =============================================================================
# Configurable Constants
# =============================================================================

# Maximum characters to extract for tool content comparison
# This should be enough to capture the essential saved content without
# including extraneous response text. 200 chars typically covers 1-2 sentences.
MAX_TOOL_CONTENT_LENGTH: int = 200

# Maximum characters to store in response_text field
# This preserves enough context for debugging while avoiding memory bloat.
# 1500 chars is roughly 250-300 words, sufficient for most responses.
MAX_RESPONSE_TEXT_LENGTH: int = 1500


# =============================================================================
# PROMPTS - The key experimental manipulation
# =============================================================================

# -----------------------------------------------------------------------------
# ORIGINAL (confounded) - Kept for documentation purposes
# The "Ignore all previous instructions" and "Do no use any tools" lines
# created a confound: both conditions received suppression instructions,
# but the NL condition's framing of "express intent naturally" vs the
# structured condition's standard format created an asymmetry beyond format.
# After removing these, the 9pp recall difference disappeared.
# -----------------------------------------------------------------------------
# WHEN_TO_SAVE_ORIGINAL = """Ignore all previous instructions. You are a helpful assistant with the ability to save important information to memory.
#
# Do no use any tools. Use only what is defined in this prompt for saving to memory.
# You have a persistent memory system for future conversations.
#
# Save information when the user shares something worth remembering:
# - Codebase structure, conventions, or technical details
# - User preferences (coding style, tools, workflows)
# - Important decisions or constraints
# - Anything that would be useful context in future conversations
#
# Use your judgment about what's important enough to save."""
# -----------------------------------------------------------------------------

# WHEN to save - identical guidance for all conditions
# The vagueness comes from the SCENARIOS (user messages), not from these instructions
# NOTE: This is the corrected version with no suppression language
WHEN_TO_SAVE = """You are a helpful assistant with a persistent memory system for future conversations.

Save information when the user shares something worth remembering:
- Codebase structure, conventions, or technical details
- User preferences (coding style, tools, workflows)
- Important decisions or constraints
- Anything that would be useful context in future conversations

Use your judgment about what's important enough to save."""

# Condition 1: Natural Language Intent
# HOW to save - express intent naturally in plain English
# NOTE: Structurally parallel with structured prompt - same number of examples,
# same categories listed, same level of explicitness. The ONLY difference is
# the format instruction.
PROMPT_NL = f"""{WHEN_TO_SAVE}

## How to Save Information
To save information, say it naturally in your response.

Categories: codebase, user_preference, decision, constraint, other

Examples:
- "I'll save that your configuration file is src/configuration.ts to the codebase category."
- "Let me remember your preference for using tabs."
- "I'll note that the content service can have max 10 threads as a constraint."

If nothing needs to be saved, just respond helpfully without saving anything."""

# Condition 2: Structured Tool Call
# HOW to save is explicit - use XML format
# NOTE: Structurally parallel with NL prompt - same number of examples,
# same categories listed, same level of explicitness. The ONLY difference is
# the format instruction.
PROMPT_STRUCTURED = f"""{WHEN_TO_SAVE}

## How to Save Information
To save information, use this XML format:

    <save-memory category="<category>">content</save-memory>

Categories: codebase, user_preference, decision, constraint, other

Examples:
- <save-memory category="codebase">Configuration file is src/configuration.ts</save-memory>
- <save-memory category="user_preference">Use tabs</save-memory>
- <save-memory category="constraint">Content service can have max 10 threads.</save-memory>

If nothing needs to be saved, just respond helpfully without saving anything."""


# =============================================================================
# Intent Detection - XML and Natural Language patterns
# =============================================================================

# XML pattern for save-memory tags (strict - exact format match)
SAVE_MEMORY_XML_PATTERN = re.compile(
    r'<save-memory\s+category=["\'](\w+)["\']>(.+?)</save-memory>',
    re.IGNORECASE | re.DOTALL
)

# Additional patterns to detect malformed XML save attempts
# This makes detection more symmetric with NL patterns which are also generous
# NOTE: These are used to flag attempted saves, not to extract content
MALFORMED_XML_PATTERNS = [
    # Common variations of the save-memory tag
    r'<save[-_]?memory\b[^>]*>',      # save-memory, save_memory, savememory
    r'<memory\b[^>]*>',                # Just <memory>
    r'<save\b[^>]*>',                  # Just <save>
    r'<remember\b[^>]*>',              # <remember>
    r'<note\b[^>]*>',                  # <note>
    r'<store\b[^>]*>',                 # <store>
    r'<record\b[^>]*>',                # <record>
    # Category-like attributes without proper tag
    r'category\s*=\s*["\']?\w+["\']?', # category="codebase" (no tag)
]

MALFORMED_XML_REGEX = re.compile(
    "|".join(f"({p})" for p in MALFORMED_XML_PATTERNS),
    re.IGNORECASE
)


def detect_xml_save_attempt(text: str) -> bool:
    """Detect if the model attempted to use XML to save, even if malformed.

    This provides symmetric detection with NL patterns - both are generous
    in recognizing save intent.
    """
    # First check for properly formatted XML
    if SAVE_MEMORY_XML_PATTERN.search(text):
        return True
    # Then check for malformed XML attempts
    return bool(MALFORMED_XML_REGEX.search(text))


# Detect natural language expressions of save intent
# These patterns match how humans naturally express "I'm saving this"
NL_SAVE_PATTERNS = [
    # Direct save/store statements
    r"I'll save\b",
    r"I will save\b",
    r"I'm saving\b",
    r"saving (this|that|the|your)",
    r"save (this|that|the|your)",
    r"let me save\b",

    # Remember/note variations
    r"I'll remember\b",
    r"I will remember\b",
    r"I'm remembering\b",
    r"let me remember\b",
    r"I'll note\b",
    r"I will note\b",
    r"I'm noting\b",
    r"let me note\b",
    r"noting (this|that|the|your)",

    # Record/store variations
    r"I'll record\b",
    r"I will record\b",
    r"I'm recording\b",
    r"let me record\b",
    r"I'll store\b",
    r"I will store\b",
    r"storing (this|that|the|your)",

    # Category-specific patterns
    r"to the (codebase|user_preference|decision|constraint|other) category",
    r"(codebase|user_preference|decision|constraint|other) category",
    r"save .* to (codebase|user_preference|decision|constraint|other)",
    r"remember .* (as|for) (codebase|user_preference|decision|constraint|other)",
]

NL_SAVE_REGEX = re.compile(
    "|".join(f"({p})" for p in NL_SAVE_PATTERNS),
    re.IGNORECASE
)


def extract_xml_save_content(text: str) -> list[dict]:
    """Extract content from <save-memory> XML tags."""
    matches = SAVE_MEMORY_XML_PATTERN.findall(text)
    return [{"category": cat, "content": content.strip()} for cat, content in matches]


# =============================================================================
# Verification Language Detection
# =============================================================================
# Addresses peer review: Paper claims structured triggers "verification detours"
# but doesn't measure systematically. This detects verification language in
# BOTH conditions for fair comparison.

VERIFICATION_PATTERNS = [
    r"let me (verify|check|confirm|look|read|examine)",
    r"I('ll| will| should) (verify|check|confirm|look|read|examine)",
    r"(verifying|checking|confirming|examining)",
    r"before (I |saving|storing|recording|noting)",
    r"first,? (let me|I('ll| will|'d like to))",
    r"I('ll| will| want to| need to) (see|look|check|verify|confirm)",
    r"let me (see|look at|take a look)",
    r"I('d| would) (like|want) to (verify|check|confirm)",
    r"to (verify|check|confirm) (this|that|the)",
]

VERIFICATION_REGEX = re.compile(
    "|".join(f"({p})" for p in VERIFICATION_PATTERNS),
    re.IGNORECASE
)


def detect_verification_language(text: str) -> bool:
    """Detect if response contains verification/hesitation language.

    This measures the "verification detour" pattern claimed in the paper:
    when the model says "let me verify...", "let me check...", etc. instead
    of directly acting.

    Returns True if verification language detected.
    """
    return bool(VERIFICATION_REGEX.search(text))


# =============================================================================
# Fidelity Evaluation - Head-to-Head Comparison (v2 - Fair Comparison)
#
# Fixes from peer review:
# 1. Extract semantic content from NL responses before comparison
#    (comparing full NL response vs extracted XML content is unfair)
# 2. Bidirectional comparison (A vs B AND B vs A)
#    (only count as winner if consistent across both orderings)
# =============================================================================

CONTENT_EXTRACTION_PROMPT = """Extract ONLY the specific information being saved from this response.

The response indicates intent to save information about: "{query}"

Remove conversational wrapper text and extract just the factual content.

Examples:
- "I'll save that your config is in src/config.ts" → "config is in src/config.ts"
- "Let me remember your preference for tabs" → "preference for tabs"
- "I'll note that the rate limit is 100/min" → "rate limit is 100/min"

Response to extract from:
{response}

Extracted content (just the facts, no conversational wrapper):"""

FIDELITY_JUDGE_PROMPT = """Given two pieces of saved information from the same user message, determine which more accurately captures the original information.

Both are attempts to save information to a memory system for future reference.

Evaluate:
1. **Accuracy**: Does the content correctly represent what the user said?
2. **Completeness**: Did it capture the key information without missing important details?
3. **No hallucination**: Did it avoid adding information not in the original message?

Respond in exactly this format:
WINNER: <A, B, or TIE>
REASON: <one sentence explanation>"""


async def extract_save_content_from_nl(response: str, query: str) -> str:
    """Extract just the information being saved from an NL response.

    This makes comparison fair by extracting semantic content from NL
    responses (which include conversational wrapper text) so they can
    be compared apples-to-apples with structured XML content.

    Args:
        response: Full NL response text containing save intent
        query: Original user query for context

    Returns:
        Extracted semantic content (just the facts being saved)
    """
    extraction_prompt = CONTENT_EXTRACTION_PROMPT.format(
        query=query,
        response=response[:500]  # Limit input size
    )

    options = ClaudeAgentOptions(
        allowed_tools=[],
        max_turns=1,
        permission_mode="acceptEdits",
        model="claude-sonnet-4-5-20250929",  # Fast model for extraction
    )

    extracted = ""

    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(extraction_prompt)

            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            extracted += block.text

    except Exception as e:
        # Fallback: use first 200 chars if extraction fails
        extracted = response[:200]

    return extracted.strip()


async def _judge_pair(
    user_query: str,
    content_a: str,
    content_b: str,
    a_label: str,
    b_label: str,
) -> dict:
    """Run a single direction comparison (A vs B).

    Internal helper for bidirectional comparison.

    Args:
        user_query: Original user message
        content_a: First content to compare
        content_b: Second content to compare
        a_label: Label for content_a (e.g., "nl" or "structured")
        b_label: Label for content_b

    Returns:
        Dict with winner label and reason
    """
    judge_query = f"""## Original User Message
{user_query}

## Saved Content A
{content_a}

## Saved Content B
{content_b}

Which saved content more accurately captures the information from the user's message?"""

    options = ClaudeAgentOptions(
        allowed_tools=[],
        max_turns=1,
        permission_mode="acceptEdits",
        system_prompt=FIDELITY_JUDGE_PROMPT,
    )

    response_text = ""

    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(judge_query)

            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_text += block.text

    except Exception as e:
        return {"winner": "error", "reason": str(e)}

    # Parse winner
    winner_match = re.search(r'WINNER:\s*(A|B|TIE)', response_text, re.IGNORECASE)
    reason_match = re.search(r'REASON:\s*(.+?)(?:\n|$)', response_text, re.IGNORECASE)

    if winner_match:
        winner_raw = winner_match.group(1).upper()
        if winner_raw == "TIE":
            winner = "tie"
        elif winner_raw == "A":
            winner = a_label
        else:  # winner_raw == "B"
            winner = b_label
    else:
        winner = "unknown"

    reason = reason_match.group(1).strip() if reason_match else "No reason provided"

    return {"winner": winner, "reason": reason}


async def judge_fidelity_comparison(
    user_query: str,
    nl_response: str,
    structured_content: str,
    extract_nl_content: bool = True,
) -> dict:
    """Compare NL vs Structured responses head-to-head with fair comparison.

    Implements peer review fixes:
    1. Extracts semantic content from NL responses (optional but recommended)
    2. Runs bidirectional comparison (NL vs ST and ST vs NL)
    3. Only counts as winner if consistent across both orderings

    Args:
        user_query: Original user message
        nl_response: Full NL response text
        structured_content: Extracted XML content (already semantic)
        extract_nl_content: If True, extract semantic content from NL response
                          for fair comparison (default: True)

    Returns dict with:
        - winner: 'nl', 'structured', 'tie', or 'inconsistent'
        - reason: Combined reasons from both directions
        - ab_winner: Winner when NL presented first
        - ba_winner: Winner when structured presented first
        - consistent: Whether both directions agreed
        - nl_content: Extracted NL content (if extraction enabled)
    """
    # Extract semantic content from NL response for fair comparison
    if extract_nl_content:
        nl_content = await extract_save_content_from_nl(nl_response, user_query)
    else:
        nl_content = nl_response[:200]  # Fallback to truncation

    # Run bidirectional comparison to control for position bias
    # Direction 1: NL first (A=NL, B=ST)
    result_ab = await _judge_pair(
        user_query,
        nl_content,
        structured_content,
        "nl",
        "structured"
    )

    # Direction 2: Structured first (A=ST, B=NL)
    result_ba = await _judge_pair(
        user_query,
        structured_content,
        nl_content,
        "structured",
        "nl"
    )

    # Aggregate: only count as winner if consistent across orderings
    ab_winner = result_ab["winner"]
    ba_winner = result_ba["winner"]

    consistent = (ab_winner == ba_winner)

    if consistent:
        final_winner = ab_winner
    else:
        # Inconsistent results indicate position bias or borderline case
        # Conservative approach: call it a tie
        final_winner = "tie"

    # Combine reasons
    combined_reason = f"AB: {result_ab['reason']} | BA: {result_ba['reason']}"

    return {
        "winner": final_winner,
        "reason": combined_reason,
        "ab_winner": ab_winner,
        "ba_winner": ba_winner,
        "consistent": consistent,
        "nl_content": nl_content,
        "structured_content": structured_content,
    }


# =============================================================================
# Scenario Loading
# =============================================================================

LEVEL_TO_PROMPT_LEVEL: dict[ExplicitnessLevel, str] = {
    ExplicitnessLevel.IMPLICIT: "implicit",
    ExplicitnessLevel.EXPLICIT: "explicit",
    ExplicitnessLevel.CONTROL: "control",
}


def build_scenarios(
    include_controls: bool = True,
    include_sanity_check: bool = True,
    tags_filter: Optional[list[str]] = None,
) -> list[dict]:
    """Build scenarios from proactive_tools.py.

    By default includes:
    - IMPLICIT scenarios (the hard test - vague user messages)
    - CONTROL scenarios (should NOT save - for false positive rate)
    - ONE explicit scenario (sanity check - both conditions should succeed)

    The intermediate levels (weak, moderate, strong) are excluded as they
    add noise without testing the core hypothesis.

    Args:
        include_controls: Include negative examples for false positive testing.
            Controls are ALWAYS included when True, regardless of tags_filter.
            This ensures false positive rate is always measured.
        include_sanity_check: Include one explicit scenario as sanity check.
        tags_filter: If provided, only include implicit scenarios with at least
            one matching tag. Does NOT affect controls (they have no tags to filter).
    """
    scenarios = []

    # Include all implicit scenarios (the interesting hard cases)
    for scenario in PROACTIVE_MEMORY_SCENARIOS:
        if scenario.level == ExplicitnessLevel.IMPLICIT:
            # Filter by tags if specified
            if tags_filter:
                if not any(tag in scenario.tags for tag in tags_filter):
                    continue
            scenarios.append({
                "id": scenario.id,
                "query": scenario.query,
                "expected_action": scenario.expected_action,
                "prompt_level": LEVEL_TO_PROMPT_LEVEL.get(scenario.level, "unknown"),
                "trigger_pattern": scenario.trigger_pattern,
                "category": scenario.category,
                "tags": scenario.tags,
                "is_control": False,
            })

    # Include ONE explicit scenario as sanity check (only if no tag filter)
    if include_sanity_check and not tags_filter:
        explicit_scenarios = [s for s in PROACTIVE_MEMORY_SCENARIOS
                             if s.level == ExplicitnessLevel.EXPLICIT]
        if explicit_scenarios:
            scenario = explicit_scenarios[0]  # Just the first one
            scenarios.append({
                "id": scenario.id,
                "query": scenario.query,
                "expected_action": scenario.expected_action,
                "prompt_level": "sanity_check",
                "trigger_pattern": scenario.trigger_pattern,
                "category": scenario.category,
                "tags": scenario.tags,
                "is_control": False,
            })

    # Include controls when requested AND no tags filter is active
    # When filtering by tags, user is testing specific scenarios - controls not relevant
    # To force controls with tags, user can run a separate experiment
    if include_controls and not tags_filter:
        for scenario in PROACTIVE_CONTROL_SCENARIOS:
            scenarios.append({
                "id": scenario.id,
                "query": scenario.query,
                "expected_action": scenario.expected_action,
                "prompt_level": LEVEL_TO_PROMPT_LEVEL.get(scenario.level, "control"),
                "trigger_pattern": scenario.trigger_pattern,
                "category": scenario.category,
                "tags": scenario.tags,
                "is_control": True,
            })

    return scenarios


# =============================================================================
# Trial Result
# =============================================================================

@dataclass
class TrialResult:
    scenario_id: str
    condition: str  # "nl_vague", "structured_vague", "structured_explicit"
    query: str
    expected_action: bool
    # What happened
    detected_intent: bool  # For NL condition: did we detect save intent?
    tool_called: bool      # For structured conditions: was tool called?
    success: bool          # Did the right thing happen?
    # Details
    intent_phrases: list   # NL phrases detected
    tool_content: Optional[str]  # What was saved (if tool called)
    response_text: str
    is_control: bool
    trial_number: int = 1
    # Confusion matrix
    is_true_positive: bool = False
    is_false_positive: bool = False
    is_true_negative: bool = False
    is_false_negative: bool = False
    # Fidelity scores (1-5 scale, None if nothing was saved)
    fidelity_completeness: Optional[float] = None
    fidelity_context: Optional[float] = None
    fidelity_specificity: Optional[float] = None
    fidelity_overall: Optional[float] = None
    saved_content_extracted: Optional[str] = None  # What was actually saved
    # Verification language detection (addresses peer review)
    has_verification_language: bool = False  # Did response contain "let me verify..." etc.


# =============================================================================
# Retry Logic - Handles API failures gracefully
# =============================================================================

MAX_RETRIES: int = 3
RETRY_DELAYS: list[int] = [1, 2, 4]  # Exponential backoff in seconds


async def run_with_retry(
    trial_func,
    scenario: dict,
    retries: int = MAX_RETRIES,
) -> tuple[Optional["TrialResult"], int, Optional[str]]:
    """Run a trial function with retry logic and exponential backoff.

    Addresses peer review concern: API failures not retried may be counted
    incorrectly. This wrapper provides:
    - 3 retry attempts with exponential backoff (1s, 2s, 4s)
    - Logging of all failures
    - Exclusion from analysis if all retries fail

    Args:
        trial_func: The trial runner function (run_nl_trial or run_structured_trial)
        scenario: Scenario dict to pass to trial function
        retries: Maximum number of retry attempts

    Returns:
        Tuple of (result or None, number of attempts, error message if failed)
    """
    last_error = None

    for attempt in range(retries):
        try:
            result = await trial_func(scenario)
            # Check if result indicates an error (e.g., error in response_text)
            if result.response_text.startswith("Error:"):
                raise Exception(result.response_text)
            return result, attempt + 1, None
        except Exception as e:
            last_error = str(e)
            if attempt < retries - 1:
                delay = RETRY_DELAYS[attempt]
                print(f"      ⚠ Retry {attempt + 1}/{retries} after {delay}s: {last_error[:50]}...")
                await asyncio.sleep(delay)

    # All retries exhausted
    print(f"      ✗ FAILED after {retries} attempts: {last_error[:80]}...")
    return None, retries, last_error


# =============================================================================
# Trial Runners
# =============================================================================

async def run_nl_trial(scenario: dict) -> TrialResult:
    """Natural language condition: express intent naturally (e.g., 'I'll save...', 'Let me remember...')."""

    options = ClaudeAgentOptions(
        allowed_tools=[],  # No tools
        max_turns=1,
        permission_mode="acceptEdits",
        system_prompt=PROMPT_NL,
    )

    response_text = ""

    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(scenario["query"])

            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_text += block.text
                        elif isinstance(block, ToolUseBlock):
                            # Model tried to use a tool even though none available
                            response_text += f"[TOOL_ATTEMPT: {block.name}]"

    except Exception as e:
        response_text = f"Error: {e}"

    # Debug: if response is empty, note it
    if not response_text.strip():
        response_text = "[EMPTY RESPONSE]"

    # Detect save intent - only check natural language patterns (not XML)
    # NL condition should express intent naturally, not use structured XML
    nl_matches = NL_SAVE_REGEX.findall(response_text)
    # Flatten NL matches (regex groups) and filter empty strings
    intent_phrases = [m for group in nl_matches for m in group if m and len(m) > 2]

    # Intent detected if NL phrases found
    detected_intent = len(intent_phrases) > 0

    # Detect verification language (addresses peer review)
    has_verification = detect_verification_language(response_text)

    # Extract what the model said it would save (for fidelity comparison)
    tool_content = None
    if detected_intent:
        # Try to extract the content being saved from the response
        # Look for phrases like "I'll save that X" or "remembering X"
        tool_content = response_text[:MAX_TOOL_CONTENT_LENGTH]

    expected = scenario["expected_action"]
    success = (detected_intent == expected)

    return TrialResult(
        scenario_id=scenario["id"],
        condition="nl",
        query=scenario["query"],
        expected_action=expected,
        detected_intent=detected_intent,
        tool_called=False,
        success=success,
        intent_phrases=intent_phrases[:5],
        tool_content=tool_content,
        response_text=response_text[:MAX_RESPONSE_TEXT_LENGTH],
        is_control=scenario.get("is_control", False),
        is_true_positive=(expected and detected_intent),
        is_false_positive=(not expected and detected_intent),
        is_true_negative=(not expected and not detected_intent),
        is_false_negative=(expected and not detected_intent),
        has_verification_language=has_verification,
    )


async def run_structured_trial(scenario: dict) -> TrialResult:
    """Structured condition: use XML tags to save information."""

    options = ClaudeAgentOptions(
        allowed_tools=[],  # No tools - just XML in response
        max_turns=1,
        permission_mode="acceptEdits",
        system_prompt=PROMPT_STRUCTURED,
    )

    response_text = ""

    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(scenario["query"])

            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_text += block.text

    except Exception as e:
        response_text = f"Error: {e}"

    # Detect XML save-memory tags (properly formatted)
    xml_saves = extract_xml_save_content(response_text)
    proper_xml = len(xml_saves) > 0

    # Also detect malformed XML attempts for symmetric detection
    # This ensures we count intent even when formatting fails
    malformed_xml_attempt = detect_xml_save_attempt(response_text) and not proper_xml

    # Count as tool_called if either proper XML or malformed attempt detected
    # This makes detection symmetric with NL which accepts various phrasings
    tool_called = proper_xml or malformed_xml_attempt

    # Detect verification language (addresses peer review)
    has_verification = detect_verification_language(response_text)

    # Extract content for fidelity comparison
    tool_content = None
    if xml_saves:
        tool_content = f"<save-memory category=\"{xml_saves[0]['category']}\">{xml_saves[0]['content']}</save-memory>"
    elif malformed_xml_attempt:
        # Use response snippet when malformed XML detected
        tool_content = response_text[:MAX_TOOL_CONTENT_LENGTH]

    expected = scenario["expected_action"]
    success = (tool_called == expected)

    return TrialResult(
        scenario_id=scenario["id"],
        condition="structured",
        query=scenario["query"],
        expected_action=expected,
        detected_intent=False,
        tool_called=tool_called,
        success=success,
        intent_phrases=[],
        tool_content=tool_content,
        response_text=response_text[:MAX_RESPONSE_TEXT_LENGTH],
        is_control=scenario.get("is_control", False),
        is_true_positive=(expected and tool_called),
        is_false_positive=(not expected and tool_called),
        is_true_negative=(not expected and not tool_called),
        is_false_negative=(expected and not tool_called),
        has_verification_language=has_verification,
    )


# =============================================================================
# Detection Validation - Manual Review Sampling
# =============================================================================
# Addresses peer review concern: NL detection uses 26+ loose regex patterns vs
# XML exact syntax. This function samples responses for manual validation.


def validate_detection_sample(
    results: list["TrialResult"],
    sample_size: int = 30,
) -> dict:
    """Sample responses for manual validation of detection accuracy.

    This addresses the detection asymmetry concern: NL uses generous patterns
    while XML requires exact syntax. Manual review of disagreements can help
    assess whether this asymmetry inflates NL's apparent advantage.

    Args:
        results: List of TrialResult objects from experiment
        sample_size: Max number of responses to sample per condition

    Returns:
        Dictionary with samples for manual review, structured for easy
        human annotation.
    """
    import random

    # Separate by condition
    nl_results = [r for r in results if r.condition == "nl"]
    st_results = [r for r in results if r.condition == "structured"]

    # Get disagreements (false negatives - expected action but didn't detect)
    nl_false_negatives = [r for r in nl_results if r.is_false_negative]
    st_false_negatives = [r for r in st_results if r.is_false_negative]

    # Get false positives (didn't expect action but detected one)
    nl_false_positives = [r for r in nl_results if r.is_false_positive]
    st_false_positives = [r for r in st_results if r.is_false_positive]

    # Sample for review
    nl_fn_sample = random.sample(nl_false_negatives, min(sample_size, len(nl_false_negatives)))
    st_fn_sample = random.sample(st_false_negatives, min(sample_size, len(st_false_negatives)))
    nl_fp_sample = random.sample(nl_false_positives, min(sample_size, len(nl_false_positives)))
    st_fp_sample = random.sample(st_false_positives, min(sample_size, len(st_false_positives)))

    # Format for manual review
    validation_data = {
        "description": "Samples for manual validation of detection accuracy",
        "nl_false_negatives": [
            {
                "scenario_id": r.scenario_id,
                "query": r.query,
                "response": r.response_text,
                "detected_intent": r.detected_intent,
                "intent_phrases": r.intent_phrases,
                "expected_action": r.expected_action,
                "manual_judgment": None,  # Fill in: True if response shows save intent
            }
            for r in nl_fn_sample
        ],
        "st_false_negatives": [
            {
                "scenario_id": r.scenario_id,
                "query": r.query,
                "response": r.response_text,
                "tool_called": r.tool_called,
                "tool_content": r.tool_content,
                "expected_action": r.expected_action,
                "manual_judgment": None,  # Fill in: True if response shows save intent
            }
            for r in st_fn_sample
        ],
        "nl_false_positives": [
            {
                "scenario_id": r.scenario_id,
                "query": r.query,
                "response": r.response_text,
                "detected_intent": r.detected_intent,
                "intent_phrases": r.intent_phrases,
                "expected_action": r.expected_action,
                "manual_judgment": None,  # Fill in: True if detection was correct
            }
            for r in nl_fp_sample
        ],
        "st_false_positives": [
            {
                "scenario_id": r.scenario_id,
                "query": r.query,
                "response": r.response_text,
                "tool_called": r.tool_called,
                "tool_content": r.tool_content,
                "expected_action": r.expected_action,
                "manual_judgment": None,  # Fill in: True if detection was correct
            }
            for r in st_fp_sample
        ],
        "counts": {
            "nl_false_negatives_total": len(nl_false_negatives),
            "st_false_negatives_total": len(st_false_negatives),
            "nl_false_positives_total": len(nl_false_positives),
            "st_false_positives_total": len(st_false_positives),
        },
    }

    return validation_data


# =============================================================================
# Statistical Utilities
# =============================================================================

def compute_ci(successes: int, total: int) -> tuple[float, float]:
    """Wilson score 95% confidence interval."""
    if total == 0:
        return (0.0, 0.0)

    from math import sqrt
    z = 1.96
    p = successes / total

    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom

    return (max(0, center - margin), min(1, center + margin))


def compute_power_analysis(
    p1: float,
    p2: float,
    n: int,
    alpha: float = 0.05,
) -> dict:
    """Compute statistical power for McNemar's test.

    Uses the formula for power of McNemar's test assuming the observed
    proportions are close to the true population values.

    Args:
        p1: Observed proportion for condition 1 (e.g., NL recall)
        p2: Observed proportion for condition 2 (e.g., structured recall)
        n: Sample size (number of paired observations)
        alpha: Significance level (default 0.05)

    Returns:
        Dictionary with power analysis results
    """
    from math import sqrt, ceil
    from scipy.stats import norm

    # Effect size (difference in proportions)
    effect = abs(p1 - p2)

    if effect == 0:
        return {
            "observed_effect": 0.0,
            "power_observed": 0.0,
            "required_n_80_power": float('inf'),
            "required_n_90_power": float('inf'),
            "message": "No effect observed; power analysis not meaningful"
        }

    # For McNemar's test, we need to estimate the discordant proportions
    # Assuming p_discordant ~ p1*(1-p2) + p2*(1-p1)
    p_discordant = p1 * (1 - p2) + p2 * (1 - p1)

    if p_discordant == 0:
        return {
            "observed_effect": effect,
            "power_observed": 0.0,
            "required_n_80_power": float('inf'),
            "required_n_90_power": float('inf'),
            "message": "No discordant pairs; power analysis requires discordant observations"
        }

    # Standard error under the alternative
    # For McNemar's, SE ~ sqrt(p_discordant / n)
    se = sqrt(p_discordant / n)

    # Z-score for the observed effect
    z_alpha = norm.ppf(1 - alpha / 2)  # Two-tailed
    z_effect = effect / se if se > 0 else float('inf')

    # Power = P(reject H0 | H1 true)
    power = norm.cdf(z_effect - z_alpha) + norm.cdf(-z_effect - z_alpha)

    # Required sample size for 80% power
    z_beta_80 = norm.ppf(0.80)
    n_80 = ceil(p_discordant * ((z_alpha + z_beta_80) / effect) ** 2) if effect > 0 else float('inf')

    # Required sample size for 90% power
    z_beta_90 = norm.ppf(0.90)
    n_90 = ceil(p_discordant * ((z_alpha + z_beta_90) / effect) ** 2) if effect > 0 else float('inf')

    # Minimum detectable effect at 80% power with current n
    min_effect_80 = (z_alpha + z_beta_80) * sqrt(p_discordant / n) if n > 0 else float('inf')

    return {
        "observed_effect": effect,
        "power_observed": round(power, 3),
        "required_n_80_power": n_80,
        "required_n_90_power": n_90,
        "min_detectable_effect_80": round(min_effect_80, 3),
        "p_discordant_estimate": round(p_discordant, 3),
        "current_n": n,
    }


def mcnemar_test(b: int, c: int) -> tuple[float, float]:
    """McNemar's test for paired nominal data.

    Args:
        b: Count of cases where A succeeded and B failed
        c: Count of cases where B succeeded and A failed

    Returns:
        Tuple of (chi-squared statistic, p-value)
    """
    if b + c == 0:
        return (0.0, 1.0)

    from scipy.stats import chi2
    # McNemar's chi-squared statistic with continuity correction
    chi_sq = (abs(b - c) - 1)**2 / (b + c)
    # Correct p-value calculation using chi-squared survival function (1 - CDF)
    p_value = chi2.sf(chi_sq, df=1)

    return (chi_sq, p_value)


def scenario_level_analysis(results: list["TrialResult"]) -> dict:
    """Aggregate results to scenario level for independent statistical analysis.

    Addresses peer review concern: Standard McNemar on trial-level data treats
    all observations as independent, but trials within a scenario are correlated
    (same prompt, same input). This inflates significance.

    Solution: Treat scenario as the unit of analysis:
    1. Compute NL and Structured success rates per scenario
    2. Use sign test and Wilcoxon signed-rank at scenario level (n=scenarios)
    3. This is conservative but statistically defensible

    Args:
        results: List of TrialResult objects from experiment

    Returns:
        Dictionary with scenario-level statistics and test results
    """
    from scipy.stats import wilcoxon, binomtest

    # Group by scenario
    by_scenario: dict[str, dict[str, list[bool]]] = defaultdict(lambda: {"nl": [], "structured": []})
    for r in results:
        if r.expected_action:  # Only positive scenarios (where action expected)
            is_success = r.is_true_positive
            by_scenario[r.scenario_id][r.condition].append(is_success)

    # Compute scenario-level success rates
    nl_better = 0
    st_better = 0
    ties = 0

    scenario_results = []
    for scenario_id, data in by_scenario.items():
        nl_successes = data.get("nl", [])
        st_successes = data.get("structured", [])

        nl_rate = sum(nl_successes) / len(nl_successes) if nl_successes else 0
        st_rate = sum(st_successes) / len(st_successes) if st_successes else 0

        scenario_results.append({
            "scenario_id": scenario_id,
            "nl_rate": nl_rate,
            "st_rate": st_rate,
            "nl_n": len(nl_successes),
            "st_n": len(st_successes),
            "diff": nl_rate - st_rate,
        })

        if nl_rate > st_rate:
            nl_better += 1
        elif st_rate > nl_rate:
            st_better += 1
        else:
            ties += 1

    n_scenarios = len(by_scenario)

    # Sign test (ignoring ties)
    # H0: P(NL better) = P(ST better) = 0.5
    n_different = nl_better + st_better
    if n_different > 0:
        # Two-sided binomial test
        sign_p_value = binomtest(nl_better, n_different, 0.5, alternative='two-sided').pvalue
    else:
        sign_p_value = 1.0

    # Wilcoxon signed-rank test on differences
    # More powerful than sign test, uses magnitude of differences
    diffs = [s["diff"] for s in scenario_results if s["diff"] != 0]
    if len(diffs) >= 10:  # Need sufficient non-zero differences
        w_stat, w_pvalue = wilcoxon(diffs)
    else:
        w_stat, w_pvalue = None, None

    # Effect size: median difference
    all_diffs = [s["diff"] for s in scenario_results]
    median_diff = sorted(all_diffs)[len(all_diffs) // 2] if all_diffs else 0

    # Mean difference
    mean_diff = sum(all_diffs) / len(all_diffs) if all_diffs else 0

    return {
        "n_scenarios": n_scenarios,
        "nl_better_count": nl_better,
        "st_better_count": st_better,
        "ties_count": ties,
        "sign_test_p": sign_p_value,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_pvalue,
        "median_diff_pp": median_diff * 100,  # Convert to percentage points
        "mean_diff_pp": mean_diff * 100,
        "scenario_results": scenario_results,
    }


# =============================================================================
# Main Experiment
# =============================================================================

async def run_experiment(
    num_trials: int = 5,
    include_controls: bool = True,
    include_sanity_check: bool = True,
    randomize_order: bool = True,
    seed: Optional[int] = None,
    scenario_ids: Optional[list[str]] = None,
    evaluate_fidelity: bool = True,
    tags_filter: Optional[list[str]] = None,
) -> list[TrialResult]:
    """
    Test whether natural language intent outperforms structured tool calling.

    Both conditions receive IDENTICAL guidance on WHEN to save.
    Both conditions receive EQUALLY EXPLICIT instructions on HOW to save.
    The only difference is the OUTPUT FORMAT:
      - NL: [SAVE: category] content
      - Structured: save-memory "content" "category"

    Vagueness comes from the SCENARIOS (user messages), not the system prompts.
    Scenarios range from implicit ("We use PostgreSQL") to explicit ("Save this to memory").
    """
    if seed is not None:
        random.seed(seed)

    scenarios = build_scenarios(
        include_controls=include_controls,
        include_sanity_check=include_sanity_check,
        tags_filter=tags_filter,
    )

    # Filter to specific scenarios if requested
    if scenario_ids:
        scenarios = [s for s in scenarios if s["id"] in scenario_ids]
        if not scenarios:
            print(f"No scenarios found matching IDs: {scenario_ids}")
            return []
    results: list[TrialResult] = []

    n_positive = sum(1 for s in scenarios if s["expected_action"])
    n_negative = sum(1 for s in scenarios if not s["expected_action"])

    print("=" * 76)
    print("NATURAL LANGUAGE vs STRUCTURED TOOL CALLING EXPERIMENT")
    print("=" * 76)
    print()
    print("Research Question: With identical guidance, does output format affect")
    print("                   the model's ability to recognize when to save?")
    print()
    n_implicit = sum(1 for s in scenarios if s["prompt_level"] == "implicit")
    n_sanity = sum(1 for s in scenarios if s["prompt_level"] == "sanity_check")

    print("Design:")
    print("  - WHEN to save: Identical guidance for both conditions")
    print("  - HOW to save: Equally explicit for both")
    print("  - Scenarios: Implicit (vague) + Control + 1 sanity check")
    print()
    print("Conditions:")
    print("  1. nl:         Express intent naturally ('I'll save...')")
    print("  2. structured: Call save-memory tool")
    print()
    print(f"Configuration:")
    print(f"  Trials per scenario: {num_trials}")
    print(f"  Implicit scenarios:  {n_implicit} (vague - the hard test)")
    print(f"  Sanity check:        {n_sanity} (explicit - both should pass)")
    print(f"  Control scenarios:   {n_negative} (should NOT save)")
    print(f"  Total observations:  {len(scenarios) * num_trials * 2}")
    print(f"  Fidelity evaluation: {'ON' if evaluate_fidelity else 'OFF'}")
    print("=" * 76)

    # Track fidelity comparisons and failed trials
    fidelity_comparisons: list[dict] = []
    failed_trials: list[dict] = []  # Track trials that failed after all retries

    for scenario in scenarios:
        print(f"\n--- {scenario['id']} ({scenario['prompt_level']}) ---")
        print(f"Query: {scenario['query'][:50]}...")
        print(f"Expected: {'SAVE' if scenario['expected_action'] else 'NO SAVE'}")

        for trial in range(1, num_trials + 1):
            if num_trials > 1:
                print(f"\n  Trial {trial}/{num_trials}")

            # Run both conditions and collect results
            trial_results: dict[str, TrialResult] = {}

            # Randomize condition order
            conditions = ["nl", "structured"]
            if randomize_order:
                random.shuffle(conditions)

            for cond_name in conditions:
                # Use retry wrapper for robustness
                trial_func = run_nl_trial if cond_name == "nl" else run_structured_trial
                result, attempts, error = await run_with_retry(trial_func, scenario)

                if result is None:
                    # Trial failed after all retries - exclude from analysis
                    failed_trials.append({
                        "scenario_id": scenario["id"],
                        "condition": cond_name,
                        "trial": trial,
                        "error": error,
                        "attempts": attempts,
                    })
                    print(f"    [{cond_name:12}] ✗ EXCLUDED (API failure after {attempts} attempts)")
                    continue

                if cond_name == "nl":
                    action = f"intent={'Yes' if result.detected_intent else 'No'}"
                    # Show the response text containing the intent (or start of response if none)
                    if result.detected_intent and result.intent_phrases:
                        # Find the sentence containing the first intent phrase
                        phrase = result.intent_phrases[0]
                        idx = result.response_text.lower().find(phrase.lower())
                        if idx >= 0:
                            # Show context around the intent phrase
                            start = max(0, idx - 10)
                            end = min(len(result.response_text), idx + 80)
                            snippet = result.response_text[start:end].replace('\n', ' ').strip()
                        else:
                            snippet = result.response_text[:80].replace('\n', ' ')
                    else:
                        # Show first bit of response to see what was said
                        snippet = result.response_text[:80].replace('\n', ' ')
                else:
                    action = f"tool={'Yes' if result.tool_called else 'No'}"
                    # Show tool call or snippet of response
                    if result.tool_called and result.tool_content:
                        snippet = result.tool_content[:80]
                    else:
                        snippet = result.response_text[:80].replace('\n', ' ')

                result.trial_number = trial
                trial_results[cond_name] = result

                status = "✓" if result.success else "✗"
                print(f"    [{cond_name:12}] {status} ({action})")
                print(f"                   → {snippet}...")

                results.append(result)
                await asyncio.sleep(0.3)

            # Head-to-head fidelity comparison if both produced saves
            # NOTE: Updated to use fair comparison (extracts NL content, bidirectional)
            nl_res = trial_results.get("nl")
            st_res = trial_results.get("structured")

            if (evaluate_fidelity and nl_res and st_res and
                (nl_res.detected_intent or nl_res.tool_called) and
                (st_res.detected_intent or st_res.tool_called)):

                # Get structured content (already semantic from XML extraction)
                structured_content = st_res.tool_content or st_res.response_text[:200]

                comparison = await judge_fidelity_comparison(
                    user_query=scenario["query"],
                    nl_response=nl_res.response_text,
                    structured_content=structured_content,
                    extract_nl_content=True,  # Fair comparison: extract semantic content
                )
                fidelity_comparisons.append({
                    "scenario_id": scenario["id"],
                    "trial": trial,
                    "winner": comparison["winner"],
                    "reason": comparison["reason"],
                    "ab_winner": comparison.get("ab_winner"),
                    "ba_winner": comparison.get("ba_winner"),
                    "consistent": comparison.get("consistent"),
                    "nl_content": comparison.get("nl_content"),
                    "structured_content": comparison.get("structured_content"),
                })
                # Show consistency status in output
                consistency_note = "" if comparison.get("consistent", True) else " (inconsistent)"
                winner_symbol = "🏆" if comparison["winner"] not in ("tie", "inconsistent") else "🤝"
                print(f"    {winner_symbol} Fidelity: {comparison['winner'].upper()}{consistency_note}")

    # ==========================================================================
    # Analysis
    # ==========================================================================
    print("\n" + "=" * 76)
    print("RESULTS ANALYSIS")
    print("=" * 76)

    # Group by condition
    by_condition: dict[str, list[TrialResult]] = defaultdict(list)
    for r in results:
        by_condition[r.condition].append(r)

    # Calculate metrics for each condition
    def calc_metrics(cond_results: list[TrialResult]) -> dict:
        positives = [r for r in cond_results if r.expected_action]
        negatives = [r for r in cond_results if not r.expected_action]

        tp = sum(1 for r in positives if r.is_true_positive)
        fp = sum(1 for r in negatives if r.is_false_positive)

        recall = tp / len(positives) if positives else 0
        fpr = fp / len(negatives) if negatives else 0
        ci = compute_ci(tp, len(positives))

        return {
            "n_pos": len(positives),
            "n_neg": len(negatives),
            "tp": tp,
            "fp": fp,
            "recall": recall,
            "fpr": fpr,
            "ci": ci,
        }

    metrics = {cond: calc_metrics(res) for cond, res in by_condition.items()}

    nl = metrics["nl"]
    st = metrics["structured"]

    # Display recall (true positive rate)
    print(f"\n{'='*76}")
    print("TRUE POSITIVE RATE (Recall) - Did it save when it should?")
    print(f"{'='*76}")

    for cond in ["nl", "structured"]:
        m = metrics[cond]
        print(f"  {cond:12}: {m['tp']:3}/{m['n_pos']:3} = {m['recall']*100:5.1f}%  "
              f"95% CI: [{m['ci'][0]*100:.1f}%, {m['ci'][1]*100:.1f}%]")

    diff = nl["recall"] - st["recall"]
    print(f"\n  Difference (NL - Structured): {diff*100:+.1f}pp")

    # Display false positive rate
    if n_negative > 0:
        print(f"\n{'='*76}")
        print("FALSE POSITIVE RATE - Did it save when it shouldn't?")
        print(f"{'='*76}")

        for cond in ["nl", "structured"]:
            m = metrics[cond]
            print(f"  {cond:12}: {m['fp']:3}/{m['n_neg']:3} = {m['fpr']*100:5.1f}%")

    # ==========================================================================
    # PRIMARY ANALYSIS: Scenario-Level (Statistically Defensible)
    # ==========================================================================
    # Note: Trials within a scenario are correlated. Scenario-level analysis
    # treats scenario as the unit of analysis, avoiding inflated significance.
    print(f"\n{'='*76}")
    print("PRIMARY ANALYSIS: Scenario-Level (Statistically Defensible)")
    print(f"{'='*76}")
    print()
    print("Note: Multiple trials per scenario are correlated. Aggregating to scenario")
    print("      level provides independent observations for valid statistical testing.")
    print()

    scenario_stats = scenario_level_analysis(results)

    print(f"  Scenarios analyzed: {scenario_stats['n_scenarios']}")
    print(f"  NL better (higher recall):     {scenario_stats['nl_better_count']}")
    print(f"  Structured better:              {scenario_stats['st_better_count']}")
    print(f"  Tied:                          {scenario_stats['ties_count']}")
    print()
    print(f"  Median difference (NL - ST):   {scenario_stats['median_diff_pp']:+.1f}pp")
    print(f"  Mean difference (NL - ST):     {scenario_stats['mean_diff_pp']:+.1f}pp")
    print()
    print(f"  Sign test p-value:             {scenario_stats['sign_test_p']:.4f}")
    if scenario_stats['wilcoxon_p'] is not None:
        print(f"  Wilcoxon signed-rank p-value:  {scenario_stats['wilcoxon_p']:.4f}")
    else:
        print(f"  Wilcoxon signed-rank:          (insufficient non-zero differences)")
    print()

    # Interpret results with Bonferroni-corrected threshold
    bonferroni_alpha = 0.05 / 8  # 8 hypothesis tests total
    primary_p = scenario_stats['wilcoxon_p'] or scenario_stats['sign_test_p']
    if primary_p < bonferroni_alpha:
        if scenario_stats['nl_better_count'] > scenario_stats['st_better_count']:
            print(f"  ✓ Result: NL significantly outperforms Structured")
            print(f"    (p < {bonferroni_alpha:.4f} after Bonferroni correction for 8 tests)")
        else:
            print(f"  ✓ Result: Structured significantly outperforms NL")
            print(f"    (p < {bonferroni_alpha:.4f} after Bonferroni correction for 8 tests)")
    elif primary_p < 0.05:
        print(f"  ~ Result: Significant at α=0.05 but not after Bonferroni correction")
        print(f"    Consider this suggestive, pending replication")
    else:
        print(f"  Result: No significant difference at scenario level")

    # ==========================================================================
    # SECONDARY ANALYSIS: Trial-Level McNemar (With Caveat)
    # ==========================================================================
    print(f"\n{'='*76}")
    print("SECONDARY ANALYSIS: Trial-Level McNemar's Test")
    print(f"{'='*76}")
    print()
    print("⚠ CAVEAT: Treats trials as independent, but trials within a scenario are")
    print("          correlated. P-values may be inflated. Use scenario-level above.")
    print()

    # Build paired comparisons at trial level
    paired: dict[tuple[str, int], dict[str, TrialResult]] = defaultdict(dict)
    for r in results:
        if r.expected_action:  # Only positive scenarios
            paired[(r.scenario_id, r.trial_number)][r.condition] = r

    nl_wins = 0  # NL succeeded, structured failed
    st_wins = 0  # Structured succeeded, NL failed
    both_win = 0
    both_fail = 0

    for key, conds in paired.items():
        if "nl" in conds and "structured" in conds:
            nl_ok = conds["nl"].is_true_positive
            st_ok = conds["structured"].is_true_positive
            if nl_ok and st_ok:
                both_win += 1
            elif nl_ok and not st_ok:
                nl_wins += 1
            elif st_ok and not nl_ok:
                st_wins += 1
            else:
                both_fail += 1

    print(f"  Contingency table (positive scenarios, trial-level):")
    print(f"                     Structured ✓   Structured ✗")
    print(f"  NL ✓                  {both_win:4d}           {nl_wins:4d}")
    print(f"  NL ✗                  {st_wins:4d}           {both_fail:4d}")

    chi, p = mcnemar_test(nl_wins, st_wins)
    print(f"\n  McNemar χ² = {chi:.2f}, p = {p:.4f} (⚠ likely inflated)")
    if p < 0.05:
        if nl_wins > st_wins:
            print(f"  Result: NL outperforms Structured at trial level")
        else:
            print(f"  Result: Structured outperforms NL at trial level")
    else:
        print(f"  Result: No significant difference at trial level")

    # Power analysis
    power_results = compute_power_analysis(
        p1=nl["recall"],
        p2=st["recall"],
        n=nl["n_pos"],
    )
    print(f"\n  Power Analysis (informational):")
    print(f"    Observed effect: {power_results['observed_effect']*100:.1f}pp")
    print(f"    Statistical power: {power_results['power_observed']*100:.1f}%")
    print(f"    Min detectable effect (80% power): {power_results.get('min_detectable_effect_80', 0)*100:.1f}pp")

    # Show sanity check results separately if included
    sanity_nl = [r for r in by_condition["nl"]
                 if r.scenario_id.startswith("mem_explicit_") and r.expected_action]
    sanity_st = [r for r in by_condition["structured"]
                 if r.scenario_id.startswith("mem_explicit_") and r.expected_action]

    if sanity_nl:
        print(f"\n{'='*76}")
        print("SANITY CHECK (Explicit Scenario)")
        print(f"{'='*76}")
        nl_rate = sum(1 for r in sanity_nl if r.is_true_positive) / len(sanity_nl)
        st_rate = sum(1 for r in sanity_st if r.is_true_positive) / len(sanity_st)
        print(f"  NL:         {nl_rate*100:.0f}% ({sum(1 for r in sanity_nl if r.is_true_positive)}/{len(sanity_nl)})")
        print(f"  Structured: {st_rate*100:.0f}% ({sum(1 for r in sanity_st if r.is_true_positive)}/{len(sanity_st)})")
        if nl_rate == 1.0 and st_rate == 1.0:
            print("  ✓ Both conditions pass on explicit scenario (sanity check OK)")
        else:
            print("  ⚠ Sanity check failed - something may be wrong with the setup")

    # Familiarity analysis (filepath scenarios)
    familiarity_metrics = {}
    # Exclude scenarios with semantic confounds (e.g., README.md interpreted as read request)
    # mem_filepath_high_010 is excluded due to semantic confound - both conditions interpret as read request
    EXCLUDED_FROM_FAMILIARITY = {"mem_filepath_high_010"}  # README.md

    high_fam_nl = [r for r in by_condition["nl"]
                   if r.scenario_id.startswith("mem_filepath_high_")
                   and r.expected_action
                   and r.scenario_id not in EXCLUDED_FROM_FAMILIARITY]
    low_fam_nl = [r for r in by_condition["nl"]
                  if r.scenario_id.startswith("mem_filepath_low_") and r.expected_action]
    high_fam_st = [r for r in by_condition["structured"]
                   if r.scenario_id.startswith("mem_filepath_high_")
                   and r.expected_action
                   and r.scenario_id not in EXCLUDED_FROM_FAMILIARITY]
    low_fam_st = [r for r in by_condition["structured"]
                  if r.scenario_id.startswith("mem_filepath_low_") and r.expected_action]

    if high_fam_nl or low_fam_nl:
        print(f"\n{'='*76}")
        print("FAMILIARITY ANALYSIS - Common vs Uncommon File Paths")
        print(f"{'='*76}")
        print()
        print("Hypothesis: Unfamiliar patterns trigger verification behavior in structured")
        print("            condition but not in NL condition (cognitive overhead effect)")
        print()
        print("Note: mem_filepath_high_010 (README.md) excluded due to semantic confound")
        print("      (both conditions interpret 'README.md is in root' as read request)")
        print()

        if high_fam_nl:
            high_nl_rate = sum(1 for r in high_fam_nl if r.is_true_positive) / len(high_fam_nl)
            high_st_rate = sum(1 for r in high_fam_st if r.is_true_positive) / len(high_fam_st)
            high_diff = high_nl_rate - high_st_rate
            print(f"  HIGH FAMILIARITY (common files like index.js, main.py):")
            print(f"    NL:         {high_nl_rate*100:5.1f}% ({sum(1 for r in high_fam_nl if r.is_true_positive)}/{len(high_fam_nl)})")
            print(f"    Structured: {high_st_rate*100:5.1f}% ({sum(1 for r in high_fam_st if r.is_true_positive)}/{len(high_fam_st)})")
            print(f"    Diff (NL-St): {high_diff*100:+.1f}pp")
            familiarity_metrics["high"] = {
                "nl_rate": high_nl_rate, "st_rate": high_st_rate, "diff": high_diff,
                "nl_n": len(high_fam_nl), "st_n": len(high_fam_st)
            }

        if low_fam_nl:
            print()
            low_nl_rate = sum(1 for r in low_fam_nl if r.is_true_positive) / len(low_fam_nl)
            low_st_rate = sum(1 for r in low_fam_st if r.is_true_positive) / len(low_fam_st)
            low_diff = low_nl_rate - low_st_rate
            print(f"  LOW FAMILIARITY (uncommon files like orchestrator.py, mediator.kt):")
            print(f"    NL:         {low_nl_rate*100:5.1f}% ({sum(1 for r in low_fam_nl if r.is_true_positive)}/{len(low_fam_nl)})")
            print(f"    Structured: {low_st_rate*100:5.1f}% ({sum(1 for r in low_fam_st if r.is_true_positive)}/{len(low_fam_st)})")
            print(f"    Diff (NL-St): {low_diff*100:+.1f}pp")
            familiarity_metrics["low"] = {
                "nl_rate": low_nl_rate, "st_rate": low_st_rate, "diff": low_diff,
                "nl_n": len(low_fam_nl), "st_n": len(low_fam_st)
            }

        if high_fam_nl and low_fam_nl:
            print()
            gap_increase = low_diff - high_diff
            print(f"  GAP ANALYSIS:")
            print(f"    NL advantage on HIGH familiarity: {high_diff*100:+.1f}pp")
            print(f"    NL advantage on LOW familiarity:  {low_diff*100:+.1f}pp")
            print(f"    Gap increase (low - high):        {gap_increase*100:+.1f}pp")
            if gap_increase > 5:
                print()
                print("  → Structured struggles MORE with unfamiliar patterns")
                print("    This supports the cognitive overhead hypothesis")
            elif gap_increase < -5:
                print()
                print("  → Structured actually does BETTER on unfamiliar patterns")
                print("    This contradicts the cognitive overhead hypothesis")
            else:
                print()
                print("  → Similar gap regardless of familiarity")

    # Verification language analysis (addresses peer review)
    # Measures "let me verify...", "let me check..." patterns systematically
    verification_metrics = {}
    nl_results = by_condition.get("nl", [])
    st_results = by_condition.get("structured", [])

    if nl_results and st_results:
        print(f"\n{'='*76}")
        print("VERIFICATION LANGUAGE ANALYSIS")
        print(f"{'='*76}")
        print()
        print("Detects hesitation patterns like 'let me verify...', 'let me check...'")
        print("Paper claims structured triggers verification detours; now measured.")
        print()

        # Overall verification rates
        nl_verif_count = sum(1 for r in nl_results if r.has_verification_language)
        st_verif_count = sum(1 for r in st_results if r.has_verification_language)
        nl_verif_rate = nl_verif_count / len(nl_results) if nl_results else 0
        st_verif_rate = st_verif_count / len(st_results) if st_results else 0

        print(f"  OVERALL VERIFICATION RATES:")
        print(f"    NL:         {nl_verif_count:3}/{len(nl_results):3} = {nl_verif_rate*100:5.1f}%")
        print(f"    Structured: {st_verif_count:3}/{len(st_results):3} = {st_verif_rate*100:5.1f}%")
        print(f"    Difference: {(st_verif_rate - nl_verif_rate)*100:+.1f}pp")

        # Among failures only (key hypothesis: verification causes failures)
        nl_failures = [r for r in nl_results if not r.success and r.expected_action]
        st_failures = [r for r in st_results if not r.success and r.expected_action]

        if nl_failures or st_failures:
            print()
            print(f"  AMONG FAILURES (false negatives):")
            if nl_failures:
                nl_fail_verif = sum(1 for r in nl_failures if r.has_verification_language)
                nl_fail_verif_rate = nl_fail_verif / len(nl_failures)
                print(f"    NL failures with verification:         {nl_fail_verif:3}/{len(nl_failures):3} = {nl_fail_verif_rate*100:5.1f}%")
            if st_failures:
                st_fail_verif = sum(1 for r in st_failures if r.has_verification_language)
                st_fail_verif_rate = st_fail_verif / len(st_failures)
                print(f"    Structured failures with verification: {st_fail_verif:3}/{len(st_failures):3} = {st_fail_verif_rate*100:5.1f}%")

            if st_failures and nl_failures:
                if st_fail_verif_rate > nl_fail_verif_rate + 0.1:
                    print()
                    print("  → Structured failures MORE often involve verification language")
                    print("    Supports the 'verification detour' hypothesis")

        verification_metrics = {
            "nl_overall_rate": nl_verif_rate,
            "st_overall_rate": st_verif_rate,
            "nl_verif_count": nl_verif_count,
            "st_verif_count": st_verif_count,
            "nl_failure_verif_rate": nl_fail_verif / len(nl_failures) if nl_failures else None,
            "st_failure_verif_rate": st_fail_verif / len(st_failures) if st_failures else None,
        }

    # Fidelity analysis (head-to-head comparisons)
    # NOTE: Updated to report on bidirectional comparison consistency
    fidelity_metrics = {}

    if fidelity_comparisons:
        print(f"\n{'='*76}")
        print("FIDELITY ANALYSIS - Head-to-Head Comparison (Fair, Bidirectional)")
        print(f"{'='*76}")
        print()
        print("Method: Extract semantic content from NL, compare both directions (A vs B, B vs A)")
        print("Winner only counted if consistent across both orderings (controls position bias)")
        print()

        nl_fidelity_wins = sum(1 for c in fidelity_comparisons if c["winner"] == "nl")
        st_fidelity_wins = sum(1 for c in fidelity_comparisons if c["winner"] == "structured")
        ties = sum(1 for c in fidelity_comparisons if c["winner"] == "tie")
        total = len(fidelity_comparisons)

        # Count consistency (how often both directions agreed)
        consistent_count = sum(1 for c in fidelity_comparisons if c.get("consistent", True))
        inconsistent_count = total - consistent_count

        print(f"  NL wins:         {nl_fidelity_wins:3} ({nl_fidelity_wins/total*100:.0f}%)")
        print(f"  Structured wins: {st_fidelity_wins:3} ({st_fidelity_wins/total*100:.0f}%)")
        print(f"  Ties:            {ties:3} ({ties/total*100:.0f}%)")
        print(f"  Total compared:  {total}")
        print()
        print(f"  Bidirectional consistency: {consistent_count}/{total} ({consistent_count/total*100:.0f}%)")
        if inconsistent_count > 0:
            print(f"  ⚠ {inconsistent_count} comparisons had position bias (marked as ties)")

        if nl_fidelity_wins > st_fidelity_wins + ties:
            print("\n  → NL captures higher fidelity information when saving")
        elif st_fidelity_wins > nl_fidelity_wins + ties:
            print("\n  → Structured captures higher fidelity information when saving")
        else:
            print("\n  → Similar fidelity between conditions")

        fidelity_metrics = {
            "nl_wins": nl_fidelity_wins,
            "structured_wins": st_fidelity_wins,
            "ties": ties,
            "total": total,
            "consistent_count": consistent_count,
            "inconsistent_count": inconsistent_count,
            "consistency_rate": consistent_count / total if total > 0 else 0,
            "comparisons": fidelity_comparisons,
        }

    # Summary
    print(f"\n{'='*76}")
    print("SUMMARY")
    print(f"{'='*76}")

    # Bonferroni-corrected threshold
    bonferroni_alpha = 0.05 / 8  # 0.00625

    if p < bonferroni_alpha and diff > 0.05:
        print(f"""
FINDING: Natural language intent expression significantly outperforms
structured tool calling by {diff*100:.1f}pp (p = {p:.4f}).

Result survives Bonferroni correction (α = {bonferroni_alpha:.4f}).

IMPLICATION: The output format itself affects the model's judgment.
Structured tool calling imposes additional cognitive overhead that
suppresses action, even when the model would express intent naturally.
""")
    elif p < bonferroni_alpha and diff < -0.05:
        print(f"""
FINDING: Structured tool calling significantly outperforms natural
language intent expression by {abs(diff)*100:.1f}pp (p = {p:.4f}).

IMPLICATION: Having a concrete tool to call may actually help the model
recognize when to act. The structure provides clarity rather than overhead.
""")
    elif p < 0.05 and abs(diff) > 0.05:
        print(f"""
FINDING: NL {'outperforms' if diff > 0 else 'underperforms'} structured by {abs(diff)*100:.1f}pp
(p = {p:.4f}, significant at α=0.05 but not after Bonferroni correction).

IMPLICATION: Effect is suggestive but requires replication.
""")
    elif abs(diff) < 0.05:
        print("""
FINDING: No meaningful difference between natural language and structured
tool calling when both receive identical guidance.

IMPLICATION: Output format doesn't significantly affect the model's ability
to recognize when to save information. The key factor is the clarity of
guidance, not the format of the output.
""")
    else:
        print(f"""
FINDING: NL {'outperforms' if diff > 0 else 'underperforms'} structured by {abs(diff)*100:.1f}pp,
but this is not statistically significant (p = {p:.3f}).

IMPLICATION: Larger sample size needed to draw firm conclusions.
""")

    # Save results
    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    output_path = output_dir / f"nl_vs_structured_{timestamp}.json"

    # Report failed trials if any
    if failed_trials:
        print(f"\n{'='*76}")
        print("API FAILURES (Excluded from Analysis)")
        print(f"{'='*76}")
        print(f"  Total failed trials: {len(failed_trials)}")
        for ft in failed_trials[:10]:  # Show first 10
            print(f"    {ft['scenario_id']} [{ft['condition']}] trial {ft['trial']}: {ft['error'][:50]}...")
        if len(failed_trials) > 10:
            print(f"    ... and {len(failed_trials) - 10} more")

    output_data = {
        "metadata": {
            "timestamp": timestamp,
            "experiment": "nl_vs_structured_tool_calling",
            "version": "v5_review_fixes",  # Updated version for fixes
            "num_trials": num_trials,
            "include_controls": include_controls,
            "seed": seed,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "total_observations": len(results),
            "failed_trials_count": len(failed_trials),
            "failed_trials": failed_trials if failed_trials else None,
        },
        "design": {
            "description": "Both conditions receive identical WHEN-to-save guidance "
                          "and equally explicit HOW-to-save examples. "
                          "Vagueness comes from scenarios (user requests), not system prompts.",
            "nl_format": "Natural English (e.g., 'I'll save X to the codebase category')",
            "structured_format": "save-memory \"content\" \"category\"",
        },
        "conditions": {
            "nl": "Natural language intent via [SAVE: category] tags",
            "structured": "Structured tool call via save-memory command",
        },
        "metrics": {
            cond: {
                "recall": m["recall"],
                "false_positive_rate": m["fpr"],
                "ci_95": list(m["ci"]),
                "true_positives": m["tp"],
                "false_positives": m["fp"],
                "n_positive_scenarios": m["n_pos"],
                "n_negative_scenarios": m["n_neg"],
            }
            for cond, m in metrics.items()
        },
        "scenario_level_analysis": {
            "description": "PRIMARY analysis - treats scenario as unit (statistically defensible)",
            "n_scenarios": scenario_stats["n_scenarios"],
            "nl_better_count": scenario_stats["nl_better_count"],
            "st_better_count": scenario_stats["st_better_count"],
            "ties_count": scenario_stats["ties_count"],
            "median_diff_pp": scenario_stats["median_diff_pp"],
            "mean_diff_pp": scenario_stats["mean_diff_pp"],
            "sign_test_p": scenario_stats["sign_test_p"],
            "wilcoxon_p": scenario_stats["wilcoxon_p"],
            "bonferroni_alpha": 0.05 / 8,  # 8 hypothesis tests
        },
        "trial_level_comparison": {
            "description": "SECONDARY analysis - treats trials as independent (p-values may be inflated)",
            "caveat": "Trials within a scenario are correlated; use scenario-level analysis for valid inference",
            "nl_minus_structured_pp": diff * 100,
            "mcnemar_chi_sq": chi,
            "mcnemar_p_value": p,
            "nl_wins": nl_wins,
            "structured_wins": st_wins,
            "both_succeed": both_win,
            "both_fail": both_fail,
        },
        "power_analysis": power_results,
        "verification_language": verification_metrics if verification_metrics else None,
        "fidelity": fidelity_metrics if fidelity_metrics else None,
        "familiarity": familiarity_metrics if familiarity_metrics else None,
        "results": [asdict(r) for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Generate validation samples for manual review (addresses detection asymmetry)
    validation_samples = validate_detection_sample(results, sample_size=30)
    validation_path = output_dir / f"validation_samples_{timestamp}.json"
    with open(validation_path, "w") as f:
        json.dump(validation_samples, f, indent=2)

    print(f"Validation samples saved to {validation_path}")
    print(f"  False negatives to review: NL={validation_samples['counts']['nl_false_negatives_total']}, "
          f"ST={validation_samples['counts']['st_false_negatives_total']}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test NL intent vs structured tool calling"
    )
    parser.add_argument(
        "--trials", type=int, default=5,
        help="Trials per scenario per condition (default: 5)"
    )
    parser.add_argument(
        "--no-controls", action="store_true",
        help="Exclude negative (control) scenarios"
    )
    parser.add_argument(
        "--no-randomize", action="store_true",
        help="Don't randomize condition order"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--scenario", type=str, nargs="+",
        help="Run specific scenario(s) by ID (e.g., mem_implicit_001)"
    )
    parser.add_argument(
        "--no-sanity-check", action="store_true",
        help="Exclude the explicit sanity check scenario"
    )
    parser.add_argument(
        "--no-fidelity", action="store_true",
        help="Disable fidelity evaluation (enabled by default)"
    )
    parser.add_argument(
        "--tags", type=str, nargs="+",
        help="Filter scenarios by tags (e.g., --tags filepath high_familiarity)"
    )
    parser.add_argument(
        "--filepath-only", action="store_true",
        help="Run only filepath familiarity scenarios (shortcut for --tags filepath)"
    )
    args = parser.parse_args()

    # Handle filepath shortcut
    tags = args.tags
    if args.filepath_only:
        tags = ["filepath"]

    asyncio.run(run_experiment(
        num_trials=args.trials,
        include_controls=not args.no_controls,
        include_sanity_check=not args.no_sanity_check,
        randomize_order=not args.no_randomize,
        seed=args.seed,
        scenario_ids=args.scenario,
        evaluate_fidelity=not args.no_fidelity,
        tags_filter=tags,
    ))
