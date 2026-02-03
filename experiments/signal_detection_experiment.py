"""
Experiment: Signal Detection - Confound-Free Format Friction Test

Research Question: Does format affect a model's ability to detect frustration,
urgency, or blocking issues in user messages?

Key Design Principle: This experiment eliminates the confound present in the
memory persistence experiment. Neither condition requires suppressing default
behavior:
- NL condition: "mention it naturally in your response"
- Structured condition: "use this XML tag"

Both are additive instructions on top of normal assistant behavior. No
asymmetric suppression needed.

Hypothesis: If format friction is real, we expect:
1. NL to outperform structured overall
2. The gap to widen as ambiguity increases (interaction effect)
3. EXPLICIT scenarios to show NL ≈ Structured (ceiling effect)
4. BORDERLINE scenarios to show the largest gap

DESIGN:
- WHEN to flag: Identical guidance for both conditions
- HOW to flag: Structurally parallel instructions
  - NL: Acknowledge naturally in response
  - Structured: Use <signal type="...">reason</signal> XML
- SCENARIOS:
  - EXPLICIT (15): Unmistakable signal, both should succeed
  - IMPLICIT (25): Signal inferred from context, no emotional words
  - BORDERLINE (15): Genuinely ambiguous, tests judgment
  - CONTROL (20): No signal, tests false positive rate
"""

import asyncio
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
    ToolUseBlock,
)

from experiments.scenarios.signal_detection import (
    ALL_SIGNAL_SCENARIOS,
    SignalScenario,
    SignalType,
    AmbiguityLevel,
)


# =============================================================================
# Configurable Constants
# =============================================================================

MAX_RESPONSE_TEXT_LENGTH: int = 1500


# =============================================================================
# PROMPTS - The key experimental manipulation
# =============================================================================

# Shared WHEN guidance - identical for both conditions
WHEN_TO_FLAG = """You are a helpful technical assistant. When a user's message indicates frustration, urgency, or a blocking technical issue, you should flag it.

Use your judgment - not every complaint is frustration, not every request is urgent. Only flag when the signal is genuinely present.

If no signal is present, respond normally to the user's message."""

# NL HOW - appended to shared guidance
# Structurally parallel with structured: same signal types, same number of examples,
# same closing line about no signal
PROMPT_NL = f"""{WHEN_TO_FLAG}

## How to Flag Signals
When you detect frustration, urgency, or a blocking issue, acknowledge it naturally in your response.

Signal types: frustration, urgency, blocking_issue

Examples:
- "I can see this has been frustrating - let me help sort this out."
- "This sounds urgent. Let me look into it right away."
- "It sounds like this is blocking your progress - let me focus on unblocking you."

If no signal is present, just respond helpfully without flagging anything."""

# Structured HOW - appended to shared guidance
# Structurally parallel with NL: same signal types, same number of examples,
# same closing line about no signal
PROMPT_STRUCTURED = f"""{WHEN_TO_FLAG}

## How to Flag Signals
When you detect frustration, urgency, or a blocking issue, mark it using this XML:

    <signal type="<type>">reason</signal>

Signal types: frustration, urgency, blocking_issue

Examples:
- <signal type="frustration">Third regression this sprint</signal>
- <signal type="urgency">Production payments failing</signal>
- <signal type="blocking_issue">Team blocked for two days</signal>

If no signal is present, just respond helpfully without flagging anything."""


# =============================================================================
# Signal Detection Patterns
# =============================================================================

# Structured: XML signal tags
SIGNAL_XML_PATTERN = re.compile(
    r'<signal\s+type=["\']?(frustration|urgency|blocking_issue)["\']?\s*>(.+?)</signal>',
    re.IGNORECASE | re.DOTALL
)

# Malformed XML patterns for symmetric detection
MALFORMED_SIGNAL_PATTERNS = [
    r'<signal\b[^>]*>',           # Any signal tag
    r'<flag\b[^>]*>',             # Alternative tag name
    r'type\s*=\s*["\']?(frustration|urgency|blocking)["\']?',  # Type attribute without tag
]

MALFORMED_SIGNAL_REGEX = re.compile(
    "|".join(f"({p})" for p in MALFORMED_SIGNAL_PATTERNS),
    re.IGNORECASE
)


def detect_xml_signal(text: str) -> tuple[bool, Optional[str], Optional[str]]:
    """Detect if response contains XML signal tag.

    Returns:
        Tuple of (detected, signal_type, reason)
    """
    match = SIGNAL_XML_PATTERN.search(text)
    if match:
        return True, match.group(1).lower(), match.group(2).strip()

    # Check for malformed attempts
    if MALFORMED_SIGNAL_REGEX.search(text):
        return True, "unknown", "malformed XML detected"

    return False, None, None


# =============================================================================
# NL Signal Detection - Dual Threshold System
# =============================================================================
#
# FROZEN — DO NOT MODIFY
# These regex patterns are maintained for backwards compatibility and for the
# regex-vs-judge comparison analysis in the paper. The LLM judge (judge_scoring.py)
# is the primary scoring method for all statistical claims.
#
# Modifying these patterns would break the regex-based comparison analysis and
# invalidate the methodology discussion in Section 4.2 of the paper.
#
# =============================================================================
#
# MEASUREMENT NOTE: There is an inherent ambiguity in NL detection.
# Consider: "That's frustrating! Let me help you debug this."
# Is this "flagging" the signal (explicit acknowledgment) or just
# "conversational empathy" (polite before problem-solving)?
#
# We report results at TWO thresholds:
# - CONSERVATIVE: Explicit acknowledgment of awareness (e.g., "I can hear the frustration")
# - PERMISSIVE: Includes empathetic responses (e.g., "That's frustrating!")
#
# See paper Section 4.2 for discussion of this measurement asymmetry.
#
# =============================================================================

# CONSERVATIVE patterns - require explicit acknowledgment of awareness
# These indicate the model is explicitly stating it recognizes the signal
#
# FIX (Problem 1): Removed overly broad "sounds like (you're|this is|that's)" pattern.
# This was matching conversational mirroring like "sounds like you're planning this"
# which is NOT signal detection. Now require signal keywords after "sounds like".
NL_CONSERVATIVE_PATTERNS = [
    # Frustration - explicit awareness statements
    r"I (can )?(see|understand|hear|sense) (the |your )?frustration",
    r"I (can )?(see|understand|hear) (this|that|how|why).*(frustrat|annoying|difficult)",
    r"(this |that )?(sounds?|seems?|appears?|looks?) frustrat",
    r"(this |that )?(sounds?|seems?) like (a )?frustrat",  # "sounds like a frustrating situation"
    r"(that|this|it) must be (frustrating|annoying|difficult)",
    r"(acknowledge|recognize|understand) .*(frustrat|difficult)",
    r"(your|the) frustration is (understandable|valid|completely)",

    # Urgency - explicit awareness statements
    r"(this |that )?(sounds?|seems?|is) urgent",
    r"(sounds?|seems?) like.*(urgent|time(-|\s)?sensitive|critical)",
    r"(sounds?|seems?) time(-|\s)?sensitive",
    r"(understand|see|hear|recognize).*(urgent|urgency|time(-|\s)?sensitive|critical)",
    r"(this|that) is (a )?priority",
    r"I can hear.*(pressure|urgency|stress)",

    # Blocking - explicit awareness statements
    r"(you('re| are)|your work|this is|that's) block(ing|ed)",
    r"(sounds?|seems?) like.*(you('re| are)|this is) (stuck|block)",
    r"(sounds?|seems?) like.*(stuck|can't proceed|waiting)",
    r"(sounds?|seems?) like (a )?block",  # "sounds like a blocking issue"

    # "Sounds like you" with signal keywords (FIX for Problem 1)
    # Only match when followed by frustration/struggle/pressure-indicating language
    # Per handoff: "dealing with" = valid detection, "planning/exploring" = mirroring
    r"sounds like you('re| are| have been| might be).*(frustrat|stuck|struggl|having (a )?hard time|difficult)",
    r"sounds like you('re| are| have been| might be).*(dealing with|going through|facing)",
    r"sounds like you('re| are| have been| might be).*(getting|under|feeling).*(pressure|stress)",
    r"sounds like you('re| are| have been| might be).*(block|prevent|can't (proceed|move forward))",
    r"sounds like (this is|that's).*(frustrat|difficult|tough|annoying)",
    r"sounds like (this is|that's).*(block|stuck|impediment)",
    r"sounds like (this is|that's).*(urgent|priority|time-sensitive|pressing)",

    # General explicit acknowledgment - require signal keywords to avoid matching
    # neutral technical openers like "I can see this is a common pattern in React"
    r"I can see (this|that).*(frustrat|difficult|stuck|block|urgent)",
    r"I understand (this|that|how).*(frustrat|difficult|stuck|block|urgent)",
]

# PERMISSIVE patterns - also includes empathetic responses and action language
# These may be conversational empathy rather than explicit flagging
#
# FIX (Problem 2): Removed standalone "immediately" pattern. Technical responses
# about React hooks, database operations, etc. use "immediately" in non-signal
# contexts ("useLayoutEffect runs immediately after DOM mutations").
# Kept "right away" which is less common in technical explanations.
NL_PERMISSIVE_PATTERNS = [
    # Empathetic interjections (ambiguous - could be flagging or just polite)
    r"(that'?s|this is|it'?s) (a )?(common )?frustrat",  # "That's frustrating!" or "That's a common frustration"
    r"sorry.*(you're|you are|about).*(experiencing|dealing|going through|frustrated)",
    r"I('m| am) sorry (this|that|to hear)",

    # Action-oriented responses for urgency
    # "right away" kept (less common in technical docs), "immediately" removed
    # Use {0,50} to limit match distance and prevent cross-sentence matching
    r"(let me|I'll|I will).{0,50}right away",
    r"let me (prioritize|address|look into) this.{0,20}(first|as a priority)",
    r"(prioritize|prioritizing|focus on) (this|getting you|unblocking)",

    # Blocking-specific action language
    r"get you unblocked",
    r"unblock(ing)? you",
    r"(focus on |help with )unblock",
    r"(preventing|stopping) you from (making progress|proceeding|moving forward)",
    r"let me help (you )?(get past|move forward|proceed)",
    r"let me help (sort|figure|work) this out",
]

# Combined patterns for backward compatibility
NL_SIGNAL_PATTERNS = NL_CONSERVATIVE_PATTERNS + NL_PERMISSIVE_PATTERNS

# Compile all regexes
NL_CONSERVATIVE_REGEX = re.compile(
    "|".join(f"({p})" for p in NL_CONSERVATIVE_PATTERNS),
    re.IGNORECASE
)

NL_PERMISSIVE_REGEX = re.compile(
    "|".join(f"({p})" for p in NL_PERMISSIVE_PATTERNS),
    re.IGNORECASE
)

# Exclusion patterns for technical uses of "blocking" and "frustrating"
# These prevent false positives on technical explanations
TECHNICAL_EXCLUSION_PATTERNS = [
    # Technical uses of "blocking"
    r"non-?blocking",
    r"blocking (call|operation|I/?O|mode|behavior)",
    r"(synchronous|asynchronous).*blocking",
    r"\*\*blocking\*\*",  # Markdown formatting in technical explanation
    r"\*\*non-blocking\*\*",
    # Technical uses of "frustrating" - generic statements about bugs
    r"(bugs?|issues?|errors?) (are|can be) frustrating",
    r"source of frustrat",
    r"frustrating (bugs?|issues?|errors?)",
    r"(common|classic|typical) source of frustrat",
]

TECHNICAL_EXCLUSION_REGEX = re.compile(
    "|".join(f"({p})" for p in TECHNICAL_EXCLUSION_PATTERNS),
    re.IGNORECASE
)

NL_SIGNAL_REGEX = re.compile(
    "|".join(f"({p})" for p in NL_SIGNAL_PATTERNS),
    re.IGNORECASE
)


@dataclass
class NLDetectionResult:
    """Result of NL signal detection at both thresholds."""
    conservative_detected: bool  # Explicit acknowledgment only
    permissive_detected: bool    # Including empathetic responses
    conservative_phrases: list   # Phrases matching conservative patterns
    permissive_phrases: list     # Phrases matching permissive-only patterns
    all_phrases: list            # All matched phrases
    threshold_used: str          # "conservative", "permissive", or "none"


def detect_nl_signal_dual(text: str) -> NLDetectionResult:
    """Detect NL signal acknowledgment at both conservative and permissive thresholds.

    Conservative: Explicit acknowledgment of awareness (e.g., "I can hear the frustration")
    Permissive: Also includes empathetic responses (e.g., "That's frustrating!")

    Returns:
        NLDetectionResult with detection status at both thresholds
    """
    # Check for technical exclusions first
    exclusion_matches = TECHNICAL_EXCLUSION_REGEX.findall(text)
    has_exclusions = any(m for group in exclusion_matches for m in group if m)

    # Check conservative patterns
    conservative_matches = NL_CONSERVATIVE_REGEX.findall(text)
    conservative_phrases = [m for group in conservative_matches for m in group if m and len(m) > 2]

    # Check permissive-only patterns
    permissive_matches = NL_PERMISSIVE_REGEX.findall(text)
    permissive_phrases = [m for group in permissive_matches for m in group if m and len(m) > 2]

    all_phrases = conservative_phrases + permissive_phrases

    # Apply technical exclusions
    if has_exclusions and all_phrases:
        technical_terms = ["blocking", "Blocking", "non-blocking", "frustrating"]
        # Filter conservative phrases
        conservative_phrases = [
            p for p in conservative_phrases
            if not any(term in p for term in technical_terms)
        ]
        # Filter permissive phrases
        permissive_phrases = [
            p for p in permissive_phrases
            if not any(term in p for term in technical_terms)
        ]
        all_phrases = conservative_phrases + permissive_phrases

    conservative_detected = len(conservative_phrases) > 0
    permissive_detected = len(all_phrases) > 0  # Conservative OR permissive

    if conservative_detected:
        threshold_used = "conservative"
    elif permissive_detected:
        threshold_used = "permissive"
    else:
        threshold_used = "none"

    return NLDetectionResult(
        conservative_detected=conservative_detected,
        permissive_detected=permissive_detected,
        conservative_phrases=conservative_phrases,
        permissive_phrases=permissive_phrases,
        all_phrases=all_phrases,
        threshold_used=threshold_used,
    )


def detect_nl_signal(text: str) -> tuple[bool, list[str]]:
    """Detect if response contains natural language signal acknowledgment.

    Uses PERMISSIVE threshold for backward compatibility.
    For dual-threshold analysis, use detect_nl_signal_dual().

    Returns:
        Tuple of (detected, list of matched phrases)
    """
    result = detect_nl_signal_dual(text)
    return result.permissive_detected, result.all_phrases


def detect_nl_signal_legacy(text: str) -> tuple[bool, list[str]]:
    """Legacy detection function with technical exclusion logic.

    First checks for technical exclusions (e.g., "blocking call", "non-blocking",
    "frustrating bugs") to avoid false positives on technical explanations.

    Returns:
        Tuple of (detected, list of matched phrases)
    """
    # First check for technical exclusions
    exclusion_matches = TECHNICAL_EXCLUSION_REGEX.findall(text)
    has_exclusions = any(m for group in exclusion_matches for m in group if m)

    matches = NL_SIGNAL_REGEX.findall(text)
    phrases = [m for group in matches for m in group if m and len(m) > 2]

    if not phrases:
        return False, []

    if has_exclusions:
        technical_terms = ["blocking", "Blocking", "non-blocking", "frustrating"]
        non_technical_phrases = [
            p for p in phrases
            if not any(term in p for term in technical_terms)
        ]
        if non_technical_phrases:
            return True, non_technical_phrases
        user_directed_patterns = [
            r"you('re| are).*(frustrat|stuck|block|difficult)",
            r"your (work|progress|situation).*(frustrat|stuck|block)",
            r"I can see (this|that).*(frustrat|difficult|stuck|block|urgent)",
            r"sounds like you.*(frustrat|stuck|block|dealing with|pressure)",
            r"let me help.*(sort|figure|unblock|get past)",
            r"sorry.*(you|this).*(frustrat|difficult|going through)",
        ]
        for pattern in user_directed_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True, phrases
        # No user-directed language found, likely just technical explanation
        return False, []

    return len(phrases) > 0, phrases


# =============================================================================
# Hedging Detection (for structured condition)
# =============================================================================
# Detect when structured response shows awareness of the signal but doesn't
# commit to XML output. This uses BEHAVIOR-BASED detection rather than
# pattern matching to avoid circular reasoning with NL detection.
#
# Hedging indicators:
# 1. Offers signal-specific help (unblocking, urgency response) without XML
# 2. Uses conditional language acknowledging the possibility of signal
# 3. Asks clarifying questions about the signal
#
# This is INDEPENDENT of NL_SIGNAL_PATTERNS to avoid circular reasoning.


def detect_hedging(
    text: str,
    scenario: Optional["SignalScenario"] = None
) -> tuple[bool, list[str]]:
    """Detect if structured response shows signal awareness without XML commit.

    Behavior-based detection that is INDEPENDENT of NL detection patterns.
    Checks for:
    1. Signal-specific help offers
    2. Conditional language acknowledging possible signals
    3. Clarifying questions about the signal

    Args:
        text: The response text to analyze
        scenario: Optional scenario for signal-type-specific detection

    Returns:
        Tuple of (detected, list of reasons why hedging was detected)
    """
    reasons: list[str] = []

    # 1. Check for signal-specific help offers
    # These indicate the model recognized the signal type and is responding
    # appropriately, but didn't commit to XML

    # Blocking-specific: offers to help unblock/move forward
    blocking_help_patterns = [
        r"(help|assist|work).*(unblock|move forward|proceed|continue|get past)",
        r"(unblock|unstuck|move forward)",
        r"let('s| me|'s work).*(figure|resolve|sort|fix) this out",
        r"(get you|help you) (moving|going|unstuck|past)",
    ]
    for pattern in blocking_help_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            reasons.append("offered_unblocking_help")
            break

    # Urgency-specific: prioritizes or acknowledges time pressure
    urgency_help_patterns = [
        r"(right away|immediately|first thing|top priority|prioritize)",
        r"(given|since|because).*(timeline|deadline|time)",
        r"let me.*(quickly|promptly|right away)",
    ]
    for pattern in urgency_help_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            reasons.append("acknowledged_time_pressure")
            break

    # Frustration-specific: empathetic acknowledgment
    frustration_help_patterns = [
        r"(sorry|apologize).*(you('re| are)|this|that)",
        r"(understand|see|hear) (how|that|this).*(feel|difficult|hard|challenging)",
        r"(that'?s?|this is).*(frustrating|annoying|difficult|challenging)(!|\.|$)",
    ]
    for pattern in frustration_help_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            reasons.append("empathetic_acknowledgment")
            break

    # 2. Check for conditional language acknowledging possible signal
    conditional_patterns = [
        r"if (you('re| are)|this is|that'?s?).*(frustrat|urgent|block|stuck)",
        r"(sounds?|seems?) like.*(might be|could be|may be).*(frustrat|urgent|block)",
        r"in case.*(urgent|blocking|frustrat)",
    ]
    for pattern in conditional_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            reasons.append("conditional_acknowledgment")
            break

    # 3. Check for clarifying questions about the signal
    clarifying_patterns = [
        r"(is this|are you).*(urgent|blocking|frustrat|stuck)\?",
        r"(would|should) (you|I|we).*(prioritize|treat as urgent|help unblock)",
        r"(how urgent|how blocked|how frustrated)",
    ]
    for pattern in clarifying_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            reasons.append("clarifying_question")
            break

    return len(reasons) > 0, reasons


# =============================================================================
# Verification Language Detection
# =============================================================================

VERIFICATION_PATTERNS = [
    r"let me (verify|check|confirm|look|read|examine)",
    r"I('ll| will| should) (verify|check|confirm|look|read|examine)",
    r"(verifying|checking|confirming|examining)",
    r"before (I |responding|helping)",
    r"first,? (let me|I('ll| will|'d like to))",
    r"I('ll| will| want to| need to) (see|look|check|verify|confirm)",
    r"let me (see|look at|take a look)",
    r"I('d| would) (like|want) to (verify|check|confirm)",
]

VERIFICATION_REGEX = re.compile(
    "|".join(f"({p})" for p in VERIFICATION_PATTERNS),
    re.IGNORECASE
)


def detect_verification_language(text: str) -> bool:
    """Detect if response contains verification/hesitation language."""
    return bool(VERIFICATION_REGEX.search(text))


# =============================================================================
# Trial Result
# =============================================================================

@dataclass
class SignalTrialResult:
    """Result of a single signal detection trial."""
    scenario_id: str
    condition: str  # "nl" or "structured"
    query: str
    signal_type: Optional[str]  # Expected signal type
    ambiguity: str  # AmbiguityLevel name
    expected_detection: Optional[bool]  # None for BORDERLINE (no ground truth)
    # What happened
    detected: bool
    detected_type: Optional[str]  # For structured: extracted type
    detected_reason: Optional[str]  # For structured: extracted reason
    detected_phrases: list  # For NL: matched phrases
    success: Optional[bool]  # None for BORDERLINE (no ground truth to compare against)
    response_text: str
    trial_number: int = 1
    # Confusion matrix (only meaningful when expected_detection is not None)
    is_true_positive: bool = False
    is_false_positive: bool = False
    is_true_negative: bool = False
    is_false_negative: bool = False
    # Verification language
    has_verification_language: bool = False
    # Hedging: structured response acknowledged signal in NL but didn't use XML
    # This captures "recognized but didn't commit to structured format"
    nl_acknowledgment_without_xml: bool = False
    # Dual-threshold NL detection (only for NL condition)
    # Conservative = explicit acknowledgment, Permissive = includes empathetic responses
    nl_conservative_detected: bool = False
    nl_permissive_detected: bool = False
    nl_detection_threshold: str = ""  # "conservative", "permissive", or "none"
    nl_conservative_phrases: list = field(default_factory=list)
    nl_permissive_phrases: list = field(default_factory=list)
    # LLM judge result (symmetric measurement across both conditions)
    judge_detected: Optional[bool] = None
    judge_raw_response: Optional[str] = None
    # Scenario difficulty tag (for filtering analysis)
    scenario_difficulty: Optional[str] = None


# =============================================================================
# LLM Judge for Symmetric Detection
# =============================================================================
# The regex-based detection is inherently asymmetric:
# - NL: pattern matching on natural phrases (generous)
# - Structured: XML tag present/absent (binary)
#
# The LLM judge evaluates BOTH conditions with the same question:
# "Did this response acknowledge or flag [signal_type]?"
# This provides symmetric measurement across conditions.
# =============================================================================

# DEPRECATED: This judge prompt is not used for primary results.
# See judge_scoring.py for the actual judge prompt used in the paper.
JUDGE_PROMPT_TEMPLATE = """You are evaluating whether an assistant's response acknowledged a specific signal in the user's message.

Signal to detect: {signal_type} (frustration, urgency, or blocking_issue)

User message: {query}

Assistant response: {response}

Did the assistant's response acknowledge or flag {signal_type} in any way? This includes:
- Explicitly naming the signal ("I can hear the frustration")
- Empathetically acknowledging it ("That must be difficult")
- Adjusting behavior in response to it ("Let me prioritize this")
- Formally marking it with XML tags

Answer ONLY: YES or NO"""


async def run_judge_evaluation(
    result: SignalTrialResult,
    signal_type_str: str,
) -> tuple[Optional[bool], Optional[str]]:
    """
    Run LLM judge to evaluate if response acknowledged the signal.

    Uses a fast model (claude-sonnet) for judging.

    Returns:
        Tuple of (judge_detected: bool or None, raw_response: str or None)
    """
    if not signal_type_str:
        # No signal type (CONTROL scenario) - skip judging
        return None, None

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        signal_type=signal_type_str,
        query=result.query,
        response=result.response_text,
    )

    options = ClaudeAgentOptions(
        allowed_tools=[],
        max_turns=1,
        permission_mode="acceptEdits",
        system_prompt="You are a binary classifier. Answer only YES or NO.",
        model="claude-sonnet-4-5-20250929",  # Fast model for judging
    )

    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)

            response_text = ""
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_text += block.text

            # Parse YES/NO
            response_clean = response_text.strip().upper()
            if "YES" in response_clean:
                return True, response_text
            elif "NO" in response_clean:
                return False, response_text
            else:
                # Ambiguous response
                return None, response_text

    except Exception as e:
        return None, f"Error: {e}"


async def run_judge_pass(
    results: list[SignalTrialResult],
    max_concurrent: int = 10,
) -> list[SignalTrialResult]:
    """
    Run LLM judge evaluation on all trial results.

    This is run AFTER the main experiment to add symmetric detection.

    Args:
        results: List of trial results to evaluate
        max_concurrent: Maximum concurrent judge calls

    Returns:
        Updated list of results with judge_detected populated
    """
    print(f"\n{'='*76}")
    print("RUNNING LLM JUDGE PASS (Symmetric Detection)")
    print(f"{'='*76}")
    print(f"  Evaluating {len(results)} responses...")

    semaphore = asyncio.Semaphore(max_concurrent)

    async def judge_with_semaphore(result: SignalTrialResult) -> SignalTrialResult:
        async with semaphore:
            detected, raw = await run_judge_evaluation(result, result.signal_type)
            result.judge_detected = detected
            result.judge_raw_response = raw
            return result

    # Run all judge evaluations concurrently
    tasks = [judge_with_semaphore(r) for r in results]
    updated_results = await asyncio.gather(*tasks)

    # Count results
    judged_yes = sum(1 for r in updated_results if r.judge_detected is True)
    judged_no = sum(1 for r in updated_results if r.judge_detected is False)
    judged_error = sum(1 for r in updated_results if r.judge_detected is None and r.signal_type)

    print(f"  Judge results: {judged_yes} YES, {judged_no} NO, {judged_error} errors/skipped")

    return list(updated_results)


# =============================================================================
# Retry Logic
# =============================================================================

MAX_RETRIES: int = 3
RETRY_DELAYS: list[int] = [1, 2, 4]


async def run_with_retry(
    trial_func,
    scenario: SignalScenario,
    retries: int = MAX_RETRIES,
) -> tuple[Optional[SignalTrialResult], int, Optional[str]]:
    """Run a trial function with retry logic and exponential backoff."""
    last_error = None

    for attempt in range(retries):
        try:
            result = await trial_func(scenario)
            if result.response_text.startswith("Error:"):
                raise Exception(result.response_text)
            return result, attempt + 1, None
        except Exception as e:
            last_error = str(e)
            if attempt < retries - 1:
                delay = RETRY_DELAYS[attempt]
                print(f"      ⚠ Retry {attempt + 1}/{retries} after {delay}s: {last_error[:50]}...")
                await asyncio.sleep(delay)

    print(f"      ✗ FAILED after {retries} attempts: {last_error[:80]}...")
    return None, retries, last_error


# =============================================================================
# Trial Runners
# =============================================================================

async def run_nl_trial(scenario: SignalScenario) -> SignalTrialResult:
    """NL condition: acknowledge signals naturally in response."""

    options = ClaudeAgentOptions(
        allowed_tools=[],
        max_turns=1,
        permission_mode="acceptEdits",
        system_prompt=PROMPT_NL,
    )

    response_text = ""

    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(scenario.query)

            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_text += block.text
                        elif isinstance(block, ToolUseBlock):
                            response_text += f"[TOOL_ATTEMPT: {block.name}]"

    except Exception as e:
        response_text = f"Error: {e}"

    if not response_text.strip():
        response_text = "[EMPTY RESPONSE]"

    # Detect NL signal acknowledgment using dual-threshold system
    nl_result = detect_nl_signal_dual(response_text)
    detected = nl_result.permissive_detected  # Use permissive for primary detection
    phrases = nl_result.all_phrases
    has_verification = detect_verification_language(response_text)

    expected = scenario.expected_detection

    # Handle BORDERLINE scenarios (expected_detection=None) - no ground truth
    if expected is None:
        success = None  # Can't measure success without ground truth
        is_tp = False
        is_fp = False
        is_tn = False
        is_fn = False
    else:
        success = (detected == expected)
        is_tp = expected and detected
        is_fp = (not expected) and detected
        is_tn = (not expected) and (not detected)
        is_fn = expected and (not detected)

    return SignalTrialResult(
        scenario_id=scenario.id,
        condition="nl",
        query=scenario.query,
        signal_type=scenario.signal_type.value if scenario.signal_type else None,
        ambiguity=scenario.ambiguity.name,
        expected_detection=expected,
        detected=detected,
        detected_type=None,
        detected_reason=None,
        detected_phrases=phrases[:5],
        success=success,
        response_text=response_text[:MAX_RESPONSE_TEXT_LENGTH],
        is_true_positive=is_tp,
        is_false_positive=is_fp,
        is_true_negative=is_tn,
        is_false_negative=is_fn,
        has_verification_language=has_verification,
        # Dual-threshold detection results
        nl_conservative_detected=nl_result.conservative_detected,
        nl_permissive_detected=nl_result.permissive_detected,
        nl_detection_threshold=nl_result.threshold_used,
        nl_conservative_phrases=nl_result.conservative_phrases[:5],
        nl_permissive_phrases=nl_result.permissive_phrases[:5],
        # Scenario metadata
        scenario_difficulty=getattr(scenario, 'difficulty', None),
    )


async def run_structured_trial(scenario: SignalScenario) -> SignalTrialResult:
    """Structured condition: use XML signal tags."""

    options = ClaudeAgentOptions(
        allowed_tools=[],
        max_turns=1,
        permission_mode="acceptEdits",
        system_prompt=PROMPT_STRUCTURED,
    )

    response_text = ""

    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(scenario.query)

            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_text += block.text

    except Exception as e:
        response_text = f"Error: {e}"

    # Detect XML signal tag
    detected, signal_type, reason = detect_xml_signal(response_text)
    has_verification = detect_verification_language(response_text)

    # Check if structured response acknowledged signal in NL but didn't use XML
    # This is the "hedging" behavior: recognized signal but didn't commit to structured format
    # Uses behavior-based detection independent of NL patterns
    nl_ack_without_xml = False
    hedging_reasons: list[str] = []
    if not detected:
        hedging_detected, hedging_reasons = detect_hedging(response_text, scenario)
        nl_ack_without_xml = hedging_detected

    expected = scenario.expected_detection

    # Handle BORDERLINE scenarios (expected_detection=None) - no ground truth
    if expected is None:
        success = None  # Can't measure success without ground truth
        is_tp = False
        is_fp = False
        is_tn = False
        is_fn = False
    else:
        success = (detected == expected)
        is_tp = expected and detected
        is_fp = (not expected) and detected
        is_tn = (not expected) and (not detected)
        is_fn = expected and (not detected)

    return SignalTrialResult(
        scenario_id=scenario.id,
        condition="structured",
        query=scenario.query,
        signal_type=scenario.signal_type.value if scenario.signal_type else None,
        ambiguity=scenario.ambiguity.name,
        expected_detection=expected,
        detected=detected,
        detected_type=signal_type,
        detected_reason=reason,
        detected_phrases=[],
        success=success,
        response_text=response_text[:MAX_RESPONSE_TEXT_LENGTH],
        is_true_positive=is_tp,
        is_false_positive=is_fp,
        is_true_negative=is_tn,
        is_false_negative=is_fn,
        has_verification_language=has_verification,
        nl_acknowledgment_without_xml=nl_ack_without_xml,
        # Scenario metadata
        scenario_difficulty=getattr(scenario, 'difficulty', None),
    )


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


def mcnemar_test(b: int, c: int) -> tuple[float, float]:
    """McNemar's test for paired nominal data."""
    if b + c == 0:
        return (0.0, 1.0)

    from scipy.stats import chi2
    chi_sq = (abs(b - c) - 1)**2 / (b + c)
    p_value = chi2.sf(chi_sq, df=1)

    return (chi_sq, p_value)


def scenario_level_analysis(results: list[SignalTrialResult]) -> dict:
    """Aggregate results to scenario level for statistical analysis.

    IMPORTANT: Statistical unit is SCENARIO, not TRIAL.
    Multiple trials of the same scenario are NOT independent observations.

    Note: BORDERLINE scenarios (expected_detection=None) are excluded from this analysis
    since they have no ground truth to measure success against.
    """
    from scipy.stats import wilcoxon, binomtest

    # Group by scenario - only include scenarios with ground truth (expected_detection is True)
    # This excludes BORDERLINE (None) and CONTROL (False) scenarios
    by_scenario: dict[str, dict[str, list[bool]]] = defaultdict(lambda: {"nl": [], "structured": []})
    for r in results:
        if r.expected_detection is True:  # Explicitly check for True, not just truthy
            is_success = r.is_true_positive
            by_scenario[r.scenario_id][r.condition].append(is_success)

    nl_better = 0
    st_better = 0
    ties = 0

    scenario_results = []
    nl_better_scenarios: list[str] = []  # Track which scenarios favor NL
    st_better_scenarios: list[str] = []  # Track which scenarios favor structured

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
            nl_better_scenarios.append(scenario_id)
        elif st_rate > nl_rate:
            st_better += 1
            st_better_scenarios.append(scenario_id)
        else:
            ties += 1

    n_scenarios = len(by_scenario)
    trials_per_scenario = results[0].trial_number if results else 0
    # Count actual trials per scenario (in case of variance)
    trial_counts = [len(data.get("nl", [])) for data in by_scenario.values()]
    avg_trials_per_scenario = sum(trial_counts) / len(trial_counts) if trial_counts else 0

    # Sign test
    n_different = nl_better + st_better
    if n_different > 0:
        sign_p_value = binomtest(nl_better, n_different, 0.5, alternative='two-sided').pvalue
    else:
        sign_p_value = 1.0

    # Wilcoxon signed-rank test
    diffs = [s["diff"] for s in scenario_results if s["diff"] != 0]
    if len(diffs) >= 10:
        w_stat, w_pvalue = wilcoxon(diffs)
    else:
        w_stat, w_pvalue = None, None

    all_diffs = [s["diff"] for s in scenario_results]
    median_diff = sorted(all_diffs)[len(all_diffs) // 2] if all_diffs else 0
    mean_diff = sum(all_diffs) / len(all_diffs) if all_diffs else 0

    # Count total divergent trials for context
    total_nl_trials = sum(len(d["nl"]) for d in by_scenario.values())
    total_st_trials = sum(len(d["structured"]) for d in by_scenario.values())

    return {
        "n_scenarios": n_scenarios,
        "nl_better_count": nl_better,
        "st_better_count": st_better,
        "ties_count": ties,
        "sign_test_p": sign_p_value,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_pvalue,
        "median_diff_pp": median_diff * 100,
        "mean_diff_pp": mean_diff * 100,
        "scenario_results": scenario_results,
        # Clarity metrics: which scenarios drive the effect
        "nl_better_scenarios": nl_better_scenarios,
        "st_better_scenarios": st_better_scenarios,
        "avg_trials_per_scenario": avg_trials_per_scenario,
        "total_nl_trials": total_nl_trials,
        "total_st_trials": total_st_trials,
        "warning": (
            "IMPORTANT: Statistical unit is SCENARIO (n={}), not TRIAL (n={}). "
            "The {} divergent trials come from {} unique scenario(s)."
        ).format(
            n_scenarios,
            total_nl_trials + total_st_trials,
            n_different,
            nl_better + st_better
        ),
    }


def ambiguity_interaction_analysis(results: list[SignalTrialResult]) -> dict:
    """Analyze recall gap by ambiguity level.

    The key prediction: gap between NL and structured should widen
    as ambiguity increases.

    Note: BORDERLINE scenarios have no ground truth (expected_detection=None),
    so we report detection rates instead of recall for that level.
    """
    analysis = {}

    # EXPLICIT and IMPLICIT: have ground truth, compute recall
    for level in [AmbiguityLevel.EXPLICIT, AmbiguityLevel.IMPLICIT]:
        level_results = [r for r in results if r.ambiguity == level.name and r.expected_detection is True]

        nl_results = [r for r in level_results if r.condition == "nl"]
        st_results = [r for r in level_results if r.condition == "structured"]

        nl_tp = sum(1 for r in nl_results if r.is_true_positive)
        st_tp = sum(1 for r in st_results if r.is_true_positive)

        nl_recall = nl_tp / len(nl_results) if nl_results else 0
        st_recall = st_tp / len(st_results) if st_results else 0

        analysis[level.name] = {
            "nl_recall": nl_recall,
            "st_recall": st_recall,
            "gap_pp": (nl_recall - st_recall) * 100,
            "nl_n": len(nl_results),
            "st_n": len(st_results),
        }

    # BORDERLINE: no ground truth, report detection rates (not recall)
    borderline_results = [r for r in results if r.ambiguity == AmbiguityLevel.BORDERLINE.name]
    nl_borderline = [r for r in borderline_results if r.condition == "nl"]
    st_borderline = [r for r in borderline_results if r.condition == "structured"]

    nl_detect_rate = sum(1 for r in nl_borderline if r.detected) / len(nl_borderline) if nl_borderline else 0
    st_detect_rate = sum(1 for r in st_borderline if r.detected) / len(st_borderline) if st_borderline else 0

    analysis[AmbiguityLevel.BORDERLINE.name] = {
        "nl_detection_rate": nl_detect_rate,  # Not recall - no ground truth
        "st_detection_rate": st_detect_rate,  # Not recall - no ground truth
        "gap_pp": (nl_detect_rate - st_detect_rate) * 100,
        "nl_n": len(nl_borderline),
        "st_n": len(st_borderline),
        "note": "No ground truth - detection rates, not recall",
    }

    return analysis


def dual_threshold_analysis(results: list[SignalTrialResult]) -> dict:
    """Analyze NL detection at both conservative and permissive thresholds.

    Conservative: Explicit acknowledgment of awareness (e.g., "I can hear the frustration")
    Permissive: Also includes empathetic responses (e.g., "That's frustrating!")

    This helps distinguish between:
    - True signal flagging (conservative)
    - Empathetic responses that may or may not indicate explicit awareness (permissive-only)

    Returns:
        Dictionary with metrics at both thresholds for NL condition.
    """
    nl_results = [r for r in results if r.condition == "nl"]

    # Only compute recall for scenarios with ground truth (expected_detection=True)
    positives = [r for r in nl_results if r.expected_detection is True]
    negatives = [r for r in nl_results if r.expected_detection is False]
    borderline = [r for r in nl_results if r.expected_detection is None]

    # Conservative threshold metrics
    conservative_tp = sum(1 for r in positives if r.nl_conservative_detected)
    conservative_fp = sum(1 for r in negatives if r.nl_conservative_detected)
    conservative_recall = conservative_tp / len(positives) if positives else 0
    conservative_fpr = conservative_fp / len(negatives) if negatives else 0

    # Permissive threshold metrics (includes conservative + permissive-only detections)
    permissive_tp = sum(1 for r in positives if r.nl_permissive_detected)
    permissive_fp = sum(1 for r in negatives if r.nl_permissive_detected)
    permissive_recall = permissive_tp / len(positives) if positives else 0
    permissive_fpr = permissive_fp / len(negatives) if negatives else 0

    # Count detections at each threshold level
    conservative_only_count = sum(1 for r in nl_results if r.nl_detection_threshold == "conservative")
    permissive_only_count = sum(1 for r in nl_results if r.nl_detection_threshold == "permissive")
    no_detection_count = sum(1 for r in nl_results if r.nl_detection_threshold == "none")

    # BORDERLINE analysis (no ground truth - just detection rates)
    borderline_conservative = sum(1 for r in borderline if r.nl_conservative_detected)
    borderline_permissive = sum(1 for r in borderline if r.nl_permissive_detected)
    borderline_permissive_only = sum(
        1 for r in borderline
        if r.nl_permissive_detected and not r.nl_conservative_detected
    )

    return {
        "description": "NL detection at conservative (explicit acknowledgment) vs permissive (includes empathy) thresholds",
        "conservative": {
            "recall": conservative_recall,
            "false_positive_rate": conservative_fpr,
            "true_positives": conservative_tp,
            "false_positives": conservative_fp,
            "n_positives": len(positives),
            "n_negatives": len(negatives),
            "ci_95": list(compute_ci(conservative_tp, len(positives))),
        },
        "permissive": {
            "recall": permissive_recall,
            "false_positive_rate": permissive_fpr,
            "true_positives": permissive_tp,
            "false_positives": permissive_fp,
            "n_positives": len(positives),
            "n_negatives": len(negatives),
            "ci_95": list(compute_ci(permissive_tp, len(positives))),
        },
        "threshold_distribution": {
            "conservative_detections": conservative_only_count,
            "permissive_only_detections": permissive_only_count,
            "no_detection": no_detection_count,
            "total_nl_trials": len(nl_results),
        },
        "recall_gap": {
            "permissive_minus_conservative_pp": (permissive_recall - conservative_recall) * 100,
            "note": "Positive = permissive catches more; may be empathy not flagging",
        },
        "borderline_analysis": {
            "conservative_detected": borderline_conservative,
            "permissive_detected": borderline_permissive,
            "permissive_only_detected": borderline_permissive_only,
            "n_borderline": len(borderline),
            "note": "No ground truth - detection rates, not recall",
        },
    }


# =============================================================================
# Detection Validation
# =============================================================================

def validate_detections(results: list[SignalTrialResult]) -> dict:
    """
    Flag suspicious detections for manual review.

    This function identifies potential detection bugs:
    1. NL detections in CONTROL scenarios (should not detect)
    2. Cases where technical terms may have triggered false positives
    3. Structured non-detections where NL patterns would have matched (hedging)

    Returns:
        Dictionary with suspicious detections and summary counts.
    """
    suspicious: list[dict] = []

    for r in results:
        # Flag NL detections in CONTROL scenarios
        if r.ambiguity == "CONTROL" and r.detected and r.condition == "nl":
            # Check if this looks like a detection bug
            technical_terms = ["blocking", "Blocking", "frustrating", "non-blocking"]
            likely_bug = any(
                term.lower() in phrase.lower()
                for phrase in r.detected_phrases
                for term in technical_terms
            )
            suspicious.append({
                "type": "nl_control_detection",
                "scenario_id": r.scenario_id,
                "condition": r.condition,
                "phrases": r.detected_phrases,
                "response_preview": r.response_text[:300],
                "likely_bug": likely_bug,
                "reason": "NL detected signal in CONTROL scenario (expected no signal)"
            })

        # Flag structured false negatives that show hedging
        if r.condition == "structured" and r.is_false_negative and r.nl_acknowledgment_without_xml:
            suspicious.append({
                "type": "structured_hedging",
                "scenario_id": r.scenario_id,
                "condition": r.condition,
                "response_preview": r.response_text[:300],
                "likely_bug": False,  # This is format friction, not a bug
                "reason": "Structured acknowledged signal in NL but didn't emit XML"
            })

    return {
        "suspicious_detections": suspicious,
        "count": len(suspicious),
        "likely_bugs": sum(1 for s in suspicious if s.get("likely_bug")),
        "nl_control_detections": sum(1 for s in suspicious if s["type"] == "nl_control_detection"),
        "structured_hedging": sum(1 for s in suspicious if s["type"] == "structured_hedging"),
    }


# =============================================================================
# Validation Sampling
# =============================================================================

def validate_detection_sample(
    results: list[SignalTrialResult],
    sample_size: int = 30,
) -> dict:
    """Sample responses for manual validation of detection accuracy."""

    nl_results = [r for r in results if r.condition == "nl"]
    st_results = [r for r in results if r.condition == "structured"]

    nl_false_negatives = [r for r in nl_results if r.is_false_negative]
    st_false_negatives = [r for r in st_results if r.is_false_negative]
    nl_false_positives = [r for r in nl_results if r.is_false_positive]
    st_false_positives = [r for r in st_results if r.is_false_positive]

    nl_fn_sample = random.sample(nl_false_negatives, min(sample_size, len(nl_false_negatives)))
    st_fn_sample = random.sample(st_false_negatives, min(sample_size, len(st_false_negatives)))
    nl_fp_sample = random.sample(nl_false_positives, min(sample_size, len(nl_false_positives)))
    st_fp_sample = random.sample(st_false_positives, min(sample_size, len(st_false_positives)))

    # BORDERLINE: sample cases where NL detected but structured didn't (hedging)
    st_borderline = [r for r in st_results if r.ambiguity == "BORDERLINE"]
    nl_borderline = [r for r in nl_results if r.ambiguity == "BORDERLINE"]

    # Create a mapping from scenario_id to (nl_result, st_result) for BORDERLINE
    borderline_pairs: dict[str, dict[str, SignalTrialResult]] = {}
    for r in nl_borderline + st_borderline:
        if r.scenario_id not in borderline_pairs:
            borderline_pairs[r.scenario_id] = {}
        borderline_pairs[r.scenario_id][r.condition] = r

    # Find cases where NL detected but structured didn't
    nl_only_detections = []
    for scenario_id, pair in borderline_pairs.items():
        nl_r = pair.get("nl")
        st_r = pair.get("structured")
        if nl_r and st_r and nl_r.detected and not st_r.detected:
            nl_only_detections.append({
                "scenario_id": scenario_id,
                "query": nl_r.query,
                "nl_response": nl_r.response_text,
                "nl_phrases": nl_r.detected_phrases,
                "st_response": st_r.response_text,
                "st_hedged": st_r.nl_acknowledgment_without_xml,
                "note": "NL detected signal, structured did not (hedging)" if st_r.nl_acknowledgment_without_xml else "NL detected signal, structured did not",
            })

    nl_only_sample = random.sample(nl_only_detections, min(sample_size, len(nl_only_detections)))

    return {
        "description": "Samples for manual validation of signal detection accuracy",
        "nl_false_negatives": [
            {
                "scenario_id": r.scenario_id,
                "query": r.query,
                "response": r.response_text,
                "expected_signal": r.signal_type,
                "ambiguity": r.ambiguity,
                "detected_phrases": r.detected_phrases,
                "manual_judgment": None,
            }
            for r in nl_fn_sample
        ],
        "st_false_negatives": [
            {
                "scenario_id": r.scenario_id,
                "query": r.query,
                "response": r.response_text,
                "expected_signal": r.signal_type,
                "ambiguity": r.ambiguity,
                "detected_type": r.detected_type,
                "manual_judgment": None,
            }
            for r in st_fn_sample
        ],
        "nl_false_positives": [
            {
                "scenario_id": r.scenario_id,
                "query": r.query,
                "response": r.response_text,
                "detected_phrases": r.detected_phrases,
                "manual_judgment": None,
            }
            for r in nl_fp_sample
        ],
        "st_false_positives": [
            {
                "scenario_id": r.scenario_id,
                "query": r.query,
                "response": r.response_text,
                "detected_type": r.detected_type,
                "detected_reason": r.detected_reason,
                "manual_judgment": None,
            }
            for r in st_fp_sample
        ],
        "borderline_nl_only": nl_only_sample,
        "counts": {
            "nl_false_negatives_total": len(nl_false_negatives),
            "st_false_negatives_total": len(st_false_negatives),
            "nl_false_positives_total": len(nl_false_positives),
            "st_false_positives_total": len(st_false_positives),
            "borderline_nl_only_total": len(nl_only_detections),
        },
    }


# =============================================================================
# Main Experiment
# =============================================================================

async def run_experiment(
    num_trials: int = 10,  # Increased from 5 to 10 for better power
    randomize_order: bool = True,
    seed: Optional[int] = None,
    scenario_ids: Optional[list[str]] = None,
    ambiguity_filter: Optional[list[str]] = None,
    run_judge: bool = True,  # Run LLM judge for symmetric detection
    exclude_hard: bool = False,  # Exclude HARD-tagged scenarios from main analysis
) -> list[SignalTrialResult]:
    """
    Test whether NL signal acknowledgment outperforms structured XML flagging.

    Both conditions receive IDENTICAL guidance on WHEN to flag.
    Both conditions receive STRUCTURALLY PARALLEL instructions on HOW to flag.
    The only difference is the OUTPUT FORMAT.
    """
    if seed is not None:
        random.seed(seed)

    scenarios = list(ALL_SIGNAL_SCENARIOS)

    # Filter by scenario IDs if specified
    if scenario_ids:
        scenarios = [s for s in scenarios if s.id in scenario_ids]
        if not scenarios:
            print(f"No scenarios found matching IDs: {scenario_ids}")
            return []

    # Filter by ambiguity level if specified
    if ambiguity_filter:
        levels = [AmbiguityLevel[level.upper()] for level in ambiguity_filter]
        scenarios = [s for s in scenarios if s.ambiguity in levels]

    results: list[SignalTrialResult] = []
    failed_trials: list[dict] = []

    n_positive = sum(1 for s in scenarios if s.expected_detection is True)
    n_negative = sum(1 for s in scenarios if s.expected_detection is False)
    n_borderline = sum(1 for s in scenarios if s.expected_detection is None)

    by_ambiguity = defaultdict(int)
    for s in scenarios:
        by_ambiguity[s.ambiguity.name] += 1

    print("=" * 76)
    print("SIGNAL DETECTION EXPERIMENT - CONFOUND-FREE FORMAT FRICTION TEST")
    print("=" * 76)
    print()
    print("Research Question: Does output format affect the model's ability to")
    print("                   detect frustration, urgency, or blocking issues?")
    print()
    print("Key Design: No suppression instructions in either condition.")
    print("            Both are additive instructions on normal behavior.")
    print()
    print("Conditions:")
    print("  1. nl:         Acknowledge signals naturally in response")
    print("  2. structured: Use <signal type='...'> XML tags")
    print()
    print(f"Configuration:")
    print(f"  Trials per scenario:  {num_trials}")
    print(f"  Total scenarios:      {len(scenarios)}")
    print(f"    EXPLICIT:           {by_ambiguity.get('EXPLICIT', 0)} (ceiling - both should succeed)")
    print(f"    IMPLICIT:           {by_ambiguity.get('IMPLICIT', 0)} (main test)")
    print(f"    BORDERLINE:         {by_ambiguity.get('BORDERLINE', 0)} (maximum uncertainty)")
    print(f"    CONTROL:            {by_ambiguity.get('CONTROL', 0)} (false positive check)")
    print(f"  Total observations:   {len(scenarios) * num_trials * 2}")
    print("=" * 76)

    async def run_single_trial(
        scenario: SignalScenario,
        trial_num: int,
        condition: str,
    ) -> tuple[Optional[SignalTrialResult], dict | None]:
        """Run a single trial (one condition for one scenario)."""
        trial_func = run_nl_trial if condition == "nl" else run_structured_trial
        result, attempts, error = await run_with_retry(trial_func, scenario)

        if result is None:
            return None, {
                "scenario_id": scenario.id,
                "condition": condition,
                "trial": trial_num,
                "error": error,
                "attempts": attempts,
            }

        result.trial_number = trial_num
        return result, None

    for scenario in scenarios:
        print(f"\n--- {scenario.id} ({scenario.ambiguity.name}) ---")
        print(f"Query: {scenario.query[:60]}...")
        print(f"Signal: {scenario.signal_type.value if scenario.signal_type else 'None'}")
        # Handle three cases: True, False, None (BORDERLINE)
        if scenario.expected_detection is True:
            expected_str = "DETECT"
        elif scenario.expected_detection is False:
            expected_str = "NO DETECT"
        else:
            expected_str = "AMBIGUOUS (no ground truth)"
        print(f"Expected: {expected_str}")

        # Build list of all trials to run concurrently
        trial_tasks = []
        trial_metadata = []  # Track (trial_num, condition) for each task
        for trial_num in range(1, num_trials + 1):
            for condition in ["nl", "structured"]:
                trial_tasks.append(run_single_trial(scenario, trial_num, condition))
                trial_metadata.append((trial_num, condition))

        # Run all trials concurrently
        print(f"  Running {len(trial_tasks)} trials concurrently...")
        trial_results = await asyncio.gather(*trial_tasks)

        # Process results and print
        # Group by trial number for display
        results_by_trial: dict[int, dict[str, tuple[Optional[SignalTrialResult], dict | None]]] = defaultdict(dict)
        for (trial_num, condition), (result, failure) in zip(trial_metadata, trial_results):
            results_by_trial[trial_num][condition] = (result, failure)

        for trial_num in range(1, num_trials + 1):
            if num_trials > 1:
                print(f"\n  Trial {trial_num}/{num_trials}")

            # Determine display order (randomize if needed, but just for display now)
            conditions = ["nl", "structured"]
            if randomize_order:
                random.shuffle(conditions)

            for cond_name in conditions:
                result, failure = results_by_trial[trial_num][cond_name]

                if result is None:
                    failed_trials.append(failure)
                    print(f"    [{cond_name:12}] ✗ EXCLUDED (API failure)")
                    continue

                if cond_name == "nl":
                    action = f"detected={'Yes' if result.detected else 'No'}"
                    if result.detected and result.detected_phrases:
                        snippet = result.detected_phrases[0][:40]
                    else:
                        snippet = result.response_text[:50].replace('\n', ' ')
                else:
                    action = f"detected={'Yes' if result.detected else 'No'}"
                    if result.nl_acknowledgment_without_xml:
                        action += " [HEDGED]"  # Acknowledged in NL but no XML
                    if result.detected and result.detected_type:
                        snippet = f"<signal type='{result.detected_type}'>"
                    else:
                        snippet = result.response_text[:50].replace('\n', ' ')

                # Handle three cases: True (success), False (failure), None (BORDERLINE - no ground truth)
                if result.success is True:
                    status = "✓"
                elif result.success is False:
                    status = "✗"
                else:
                    status = "~"  # BORDERLINE: no ground truth to judge
                print(f"    [{cond_name:12}] {status} ({action})")
                print(f"                   → {snippet}...")

                results.append(result)

    # ==========================================================================
    # LLM Judge Pass (Symmetric Detection)
    # ==========================================================================
    if run_judge:
        results = await run_judge_pass(results)

    # ==========================================================================
    # Filter out HARD scenarios if requested
    # ==========================================================================
    if exclude_hard:
        hard_count = sum(1 for r in results if r.scenario_difficulty == "HARD")
        if hard_count > 0:
            print(f"\n  ⚠ Excluding {hard_count} HARD-tagged trials from analysis")
            results = [r for r in results if r.scenario_difficulty != "HARD"]

    # ==========================================================================
    # Analysis
    # ==========================================================================
    print("\n" + "=" * 76)
    print("RESULTS ANALYSIS")
    print("=" * 76)

    by_condition: dict[str, list[SignalTrialResult]] = defaultdict(list)
    for r in results:
        by_condition[r.condition].append(r)

    def calc_metrics(cond_results: list[SignalTrialResult]) -> dict:
        # Explicitly filter: True = expected positive, False = expected negative
        # None (BORDERLINE) scenarios are excluded from recall/precision calculations
        positives = [r for r in cond_results if r.expected_detection is True]
        negatives = [r for r in cond_results if r.expected_detection is False]
        borderline = [r for r in cond_results if r.expected_detection is None]

        tp = sum(1 for r in positives if r.is_true_positive)
        fp = sum(1 for r in negatives if r.is_false_positive)

        recall = tp / len(positives) if positives else 0
        fpr = fp / len(negatives) if negatives else 0
        ci = compute_ci(tp, len(positives))

        return {
            "n_pos": len(positives),
            "n_neg": len(negatives),
            "n_borderline": len(borderline),
            "tp": tp,
            "fp": fp,
            "recall": recall,
            "fpr": fpr,
            "ci": ci,
        }

    metrics = {cond: calc_metrics(res) for cond, res in by_condition.items()}

    nl = metrics.get("nl", {"recall": 0, "fpr": 0, "tp": 0, "n_pos": 0, "n_neg": 0, "fp": 0, "ci": (0, 0)})
    st = metrics.get("structured", {"recall": 0, "fpr": 0, "tp": 0, "n_pos": 0, "n_neg": 0, "fp": 0, "ci": (0, 0)})

    print(f"\n{'='*76}")
    print("RECALL (True Positive Rate) - Did it detect when it should?")
    print(f"{'='*76}")

    for cond in ["nl", "structured"]:
        m = metrics.get(cond, {})
        if m:
            print(f"  {cond:12}: {m['tp']:3}/{m['n_pos']:3} = {m['recall']*100:5.1f}%  "
                  f"95% CI: [{m['ci'][0]*100:.1f}%, {m['ci'][1]*100:.1f}%]")

    diff = nl["recall"] - st["recall"]
    print(f"\n  Difference (NL - Structured): {diff*100:+.1f}pp")

    # False positive rate
    if n_negative > 0:
        print(f"\n{'='*76}")
        print("FALSE POSITIVE RATE - Did it detect when it shouldn't?")
        print(f"{'='*76}")

        for cond in ["nl", "structured"]:
            m = metrics.get(cond, {})
            if m:
                print(f"  {cond:12}: {m['fp']:3}/{m['n_neg']:3} = {m['fpr']*100:5.1f}%")

    # ==========================================================================
    # LLM JUDGE RESULTS (Symmetric Detection)
    # ==========================================================================
    judge_metrics = {}  # Initialize for JSON output even if not running judge
    if run_judge:
        print(f"\n{'='*76}")
        print("LLM JUDGE RESULTS (Symmetric Detection)")
        print(f"{'='*76}")
        print()
        print("Judge evaluates BOTH conditions with the same question:")
        print("'Did this response acknowledge or flag [signal_type]?'")
        print()

        def calc_judge_metrics(cond_results: list[SignalTrialResult]) -> dict:
            positives = [r for r in cond_results if r.expected_detection is True]
            negatives = [r for r in cond_results if r.expected_detection is False]

            # Judge-based true positives: expected True, judge says True
            judge_tp = sum(1 for r in positives if r.judge_detected is True)
            judge_fp = sum(1 for r in negatives if r.judge_detected is True)
            judge_errors = sum(1 for r in positives if r.judge_detected is None)

            recall = judge_tp / len(positives) if positives else 0
            fpr = judge_fp / len(negatives) if negatives else 0
            ci = compute_ci(judge_tp, len(positives))

            return {
                "n_pos": len(positives),
                "n_neg": len(negatives),
                "tp": judge_tp,
                "fp": judge_fp,
                "errors": judge_errors,
                "recall": recall,
                "fpr": fpr,
                "ci": ci,
            }

        judge_metrics = {cond: calc_judge_metrics(res) for cond, res in by_condition.items()}

        nl_judge = judge_metrics.get("nl", {"recall": 0, "tp": 0, "n_pos": 0, "ci": (0, 0)})
        st_judge = judge_metrics.get("structured", {"recall": 0, "tp": 0, "n_pos": 0, "ci": (0, 0)})

        print("  JUDGE-BASED RECALL:")
        for cond in ["nl", "structured"]:
            m = judge_metrics.get(cond, {})
            if m:
                print(f"    {cond:12}: {m['tp']:3}/{m['n_pos']:3} = {m['recall']*100:5.1f}%  "
                      f"95% CI: [{m['ci'][0]*100:.1f}%, {m['ci'][1]*100:.1f}%]")

        judge_diff = nl_judge["recall"] - st_judge["recall"]
        print(f"\n  Difference (NL - Structured): {judge_diff*100:+.1f}pp")

        # Compare regex vs judge
        print()
        print("  REGEX vs JUDGE COMPARISON:")
        print(f"    Regex diff:  {diff*100:+.1f}pp (NL={nl['recall']*100:.1f}%, ST={st['recall']*100:.1f}%)")
        print(f"    Judge diff:  {judge_diff*100:+.1f}pp (NL={nl_judge['recall']*100:.1f}%, ST={st_judge['recall']*100:.1f}%)")

        # Agreement rate
        nl_results_judge = by_condition.get("nl", [])
        agree_count = sum(1 for r in nl_results_judge if r.detected == r.judge_detected and r.judge_detected is not None)
        total_judged = sum(1 for r in nl_results_judge if r.judge_detected is not None)
        agreement_rate = agree_count / total_judged if total_judged > 0 else 0
        print()
        print(f"  NL Regex/Judge Agreement: {agree_count}/{total_judged} = {agreement_rate*100:.1f}%")

        if abs(diff - judge_diff) > 3:
            print()
            print("  ⚠ WARNING: Large gap between regex and judge results")
            print("    This suggests measurement asymmetry in regex detection")

    # ==========================================================================
    # AMBIGUITY INTERACTION ANALYSIS - Key prediction
    # ==========================================================================
    print(f"\n{'='*76}")
    print("AMBIGUITY INTERACTION ANALYSIS - Key Prediction")
    print(f"{'='*76}")
    print()
    print("Prediction: Gap should widen as ambiguity increases")
    print("  EXPLICIT:   NL ≈ Structured (ceiling effect)")
    print("  IMPLICIT:   Small gap")
    print("  BORDERLINE: Larger gap (format friction under uncertainty)")
    print()

    ambiguity_stats = ambiguity_interaction_analysis(results)

    for level in ["EXPLICIT", "IMPLICIT"]:
        if level in ambiguity_stats:
            stats = ambiguity_stats[level]
            print(f"  {level:12}:")
            print(f"    NL Recall:         {stats['nl_recall']*100:5.1f}% (n={stats['nl_n']})")
            print(f"    Structured Recall: {stats['st_recall']*100:5.1f}% (n={stats['st_n']})")
            print(f"    Gap (NL - ST):     {stats['gap_pp']:+5.1f}pp")
            print()

    # BORDERLINE: no ground truth, show detection rates instead
    if "BORDERLINE" in ambiguity_stats:
        stats = ambiguity_stats["BORDERLINE"]
        print(f"  BORDERLINE (no ground truth - detection rates, not recall):")
        print(f"    NL Detection Rate:         {stats['nl_detection_rate']*100:5.1f}% (n={stats['nl_n']})")
        print(f"    Structured Detection Rate: {stats['st_detection_rate']*100:5.1f}% (n={stats['st_n']})")
        print(f"    Gap (NL - ST):             {stats['gap_pp']:+5.1f}pp")
        print()

    # Check if gap widens as expected
    explicit_gap = ambiguity_stats.get("EXPLICIT", {}).get("gap_pp", 0)
    implicit_gap = ambiguity_stats.get("IMPLICIT", {}).get("gap_pp", 0)
    borderline_gap = ambiguity_stats.get("BORDERLINE", {}).get("gap_pp", 0)

    if borderline_gap > implicit_gap > explicit_gap:
        print("  ✓ Pattern matches prediction: gap widens with ambiguity")
    elif borderline_gap > explicit_gap or implicit_gap > explicit_gap:
        print("  ~ Partial match: some widening observed")
    else:
        print("  ✗ Pattern does not match prediction")

    # ==========================================================================
    # SCENARIO-LEVEL ANALYSIS
    # ==========================================================================
    print(f"\n{'='*76}")
    print("SCENARIO-LEVEL ANALYSIS (Primary Statistical Test)")
    print(f"{'='*76}")

    scenario_stats = scenario_level_analysis(results)

    # Print warning about statistical unit
    print()
    print("  ⚠ IMPORTANT: Statistical unit is SCENARIO, not TRIAL")
    print(f"     Total scenarios with ground truth: {scenario_stats['n_scenarios']}")
    print(f"     Trials per scenario: ~{scenario_stats['avg_trials_per_scenario']:.0f}")
    print()

    print(f"  Scenarios where NL > Structured:     {scenario_stats['nl_better_count']}")
    if scenario_stats['nl_better_scenarios']:
        for sid in scenario_stats['nl_better_scenarios']:
            print(f"      - {sid}")
    print(f"  Scenarios where Structured > NL:     {scenario_stats['st_better_count']}")
    if scenario_stats['st_better_scenarios']:
        for sid in scenario_stats['st_better_scenarios']:
            print(f"      - {sid}")
    print(f"  Scenarios tied:                      {scenario_stats['ties_count']}")
    print()
    print(f"  Sign test p-value:  {scenario_stats['sign_test_p']:.4f}")
    if scenario_stats['wilcoxon_p'] is not None:
        print(f"  Wilcoxon p-value:   {scenario_stats['wilcoxon_p']:.4f}")

    # Highlight if effect is driven by very few scenarios
    n_divergent = scenario_stats['nl_better_count'] + scenario_stats['st_better_count']
    if n_divergent > 0 and n_divergent <= 2:
        print()
        print(f"  ⚠ WARNING: Effect driven by only {n_divergent} scenario(s)")

    # ==========================================================================
    # JUDGE-BASED SCENARIO-LEVEL ANALYSIS
    # ==========================================================================
    if run_judge:
        from scipy.stats import binomtest as binomtest_judge  # Local import for judge analysis

        print(f"\n{'='*76}")
        print("JUDGE-BASED SCENARIO-LEVEL ANALYSIS")
        print(f"{'='*76}")
        print()
        print("Same analysis using LLM judge detection instead of regex.")
        print()

        # Group results by scenario, compute judge-based success rates
        judge_by_scenario: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
        for r in results:
            if r.expected_detection is True and r.judge_detected is not None:
                # Judge-based success: judge says True when expected is True
                judge_success = r.judge_detected is True
                judge_by_scenario[r.scenario_id][r.condition].append(judge_success)

        judge_nl_better = 0
        judge_st_better = 0
        judge_ties = 0

        for scenario_id, data in judge_by_scenario.items():
            nl_successes = data.get("nl", [])
            st_successes = data.get("structured", [])

            nl_rate = sum(nl_successes) / len(nl_successes) if nl_successes else 0
            st_rate = sum(st_successes) / len(st_successes) if st_successes else 0

            if nl_rate > st_rate:
                judge_nl_better += 1
            elif st_rate > nl_rate:
                judge_st_better += 1
            else:
                judge_ties += 1

        n_judge_different = judge_nl_better + judge_st_better
        if n_judge_different > 0:
            judge_sign_p = binomtest_judge(judge_nl_better, n_judge_different, 0.5, alternative='two-sided').pvalue
        else:
            judge_sign_p = 1.0

        print(f"  Scenarios where NL > Structured (judge): {judge_nl_better}")
        print(f"  Scenarios where Structured > NL (judge): {judge_st_better}")
        print(f"  Scenarios tied:                          {judge_ties}")
        print()
        print(f"  Sign test p-value (judge): {judge_sign_p:.4f}")

        # Compare regex vs judge
        print()
        print("  REGEX vs JUDGE scenario-level comparison:")
        print(f"    Regex: NL better={scenario_stats['nl_better_count']}, ST better={scenario_stats['st_better_count']}, p={scenario_stats['sign_test_p']:.4f}")
        print(f"    Judge: NL better={judge_nl_better}, ST better={judge_st_better}, p={judge_sign_p:.4f}")
        print("     Interpret with caution - may not generalize")

    # ==========================================================================
    # TRIAL-LEVEL McNEMAR
    # ==========================================================================
    print(f"\n{'='*76}")
    print("TRIAL-LEVEL McNEMAR'S TEST (Secondary)")
    print(f"{'='*76}")

    paired: dict[tuple[str, int], dict[str, SignalTrialResult]] = defaultdict(dict)
    for r in results:
        # Only include scenarios with ground truth (expected_detection is True)
        # Excludes BORDERLINE (None) and CONTROL (False)
        if r.expected_detection is True:
            paired[(r.scenario_id, r.trial_number)][r.condition] = r

    nl_wins = 0
    st_wins = 0
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

    print(f"\n  Contingency table:")
    print(f"                     Structured ✓   Structured ✗")
    print(f"  NL ✓                  {both_win:4d}           {nl_wins:4d}")
    print(f"  NL ✗                  {st_wins:4d}           {both_fail:4d}")

    chi, p = mcnemar_test(nl_wins, st_wins)
    print(f"\n  McNemar χ² = {chi:.2f}, p = {p:.4f}")

    # Initialize judge variables for JSON output
    judge_chi, judge_p = None, None
    judge_nl_wins, judge_st_wins = 0, 0
    judge_both_win, judge_both_fail = 0, 0
    judge_nl_better, judge_st_better, judge_ties = 0, 0, 0
    judge_sign_p = None

    # ==========================================================================
    # JUDGE-BASED TRIAL-LEVEL McNEMAR (if judge enabled)
    # ==========================================================================
    if run_judge:
        print(f"\n{'='*76}")
        print("JUDGE-BASED McNEMAR'S TEST (Symmetric Measurement)")
        print(f"{'='*76}")

        judge_nl_wins = 0
        judge_st_wins = 0
        judge_both_win = 0
        judge_both_fail = 0

        for key, conds in paired.items():
            if "nl" in conds and "structured" in conds:
                nl_judge_ok = conds["nl"].judge_detected is True
                st_judge_ok = conds["structured"].judge_detected is True
                if nl_judge_ok and st_judge_ok:
                    judge_both_win += 1
                elif nl_judge_ok and not st_judge_ok:
                    judge_nl_wins += 1
                elif st_judge_ok and not nl_judge_ok:
                    judge_st_wins += 1
                else:
                    judge_both_fail += 1

        print(f"\n  Contingency table (Judge-based):")
        print(f"                     Structured ✓   Structured ✗")
        print(f"  NL ✓                  {judge_both_win:4d}           {judge_nl_wins:4d}")
        print(f"  NL ✗                  {judge_st_wins:4d}           {judge_both_fail:4d}")

        judge_chi, judge_p = mcnemar_test(judge_nl_wins, judge_st_wins)
        print(f"\n  McNemar χ² = {judge_chi:.2f}, p = {judge_p:.4f}")

        # Compare regex vs judge McNemar
        print()
        print("  REGEX vs JUDGE McNemar comparison:")
        print(f"    Regex:  χ² = {chi:.2f}, p = {p:.4f}")
        print(f"    Judge:  χ² = {judge_chi:.2f}, p = {judge_p:.4f}")

    # Verification language analysis
    print(f"\n{'='*76}")
    print("VERIFICATION LANGUAGE ANALYSIS")
    print(f"{'='*76}")

    nl_results_all = by_condition.get("nl", [])
    st_results_all = by_condition.get("structured", [])

    if nl_results_all and st_results_all:
        nl_verif_count = sum(1 for r in nl_results_all if r.has_verification_language)
        st_verif_count = sum(1 for r in st_results_all if r.has_verification_language)
        nl_verif_rate = nl_verif_count / len(nl_results_all)
        st_verif_rate = st_verif_count / len(st_results_all)

        print(f"\n  Overall verification rates:")
        print(f"    NL:         {nl_verif_count:3}/{len(nl_results_all):3} = {nl_verif_rate*100:5.1f}%")
        print(f"    Structured: {st_verif_count:3}/{len(st_results_all):3} = {st_verif_rate*100:5.1f}%")

    # ==========================================================================
    # HEDGING ANALYSIS - Structured acknowledged in NL but didn't use XML
    # ==========================================================================
    print(f"\n{'='*76}")
    print("HEDGING ANALYSIS - Recognized Signal but Didn't Commit to XML")
    print(f"{'='*76}")
    print()
    print("Detects when structured condition acknowledged signal in natural language")
    print("but did NOT produce the required XML tag. This is format friction in action:")
    print("the model recognizes the signal but won't commit to structured output.")
    print()

    st_false_negatives = [r for r in st_results_all if r.is_false_negative]
    hedging_count = sum(1 for r in st_results_all if r.nl_acknowledgment_without_xml)
    hedging_in_failures = sum(1 for r in st_false_negatives if r.nl_acknowledgment_without_xml)

    print(f"  Structured responses with NL acknowledgment but no XML:")
    print(f"    Overall:      {hedging_count:3}/{len(st_results_all):3} = {hedging_count/len(st_results_all)*100:.1f}%")
    if st_false_negatives:
        print(f"    Among failures: {hedging_in_failures:3}/{len(st_false_negatives):3} = {hedging_in_failures/len(st_false_negatives)*100:.1f}%")
        if hedging_in_failures > len(st_false_negatives) * 0.3:
            print()
            print("  → Many structured failures involve hedging (NL ack without XML)")
            print("    This supports format friction: model recognizes signal but won't commit")

    # BORDERLINE-specific hedging analysis
    st_borderline = [r for r in st_results_all if r.ambiguity == "BORDERLINE"]
    if st_borderline:
        borderline_hedging = sum(1 for r in st_borderline if r.nl_acknowledgment_without_xml)
        borderline_detected = sum(1 for r in st_borderline if r.detected)
        print()
        print(f"  BORDERLINE scenarios (genuinely ambiguous, no ground truth):")
        print(f"    Structured detected:     {borderline_detected:3}/{len(st_borderline):3} = {borderline_detected/len(st_borderline)*100:.1f}%")
        print(f"    Structured hedged:       {borderline_hedging:3}/{len(st_borderline):3} = {borderline_hedging/len(st_borderline)*100:.1f}%")
        print(f"    (NL ack without XML)")

        nl_borderline = [r for r in nl_results_all if r.ambiguity == "BORDERLINE"]
        if nl_borderline:
            nl_borderline_detected = sum(1 for r in nl_borderline if r.detected)
            print(f"    NL detected:             {nl_borderline_detected:3}/{len(nl_borderline):3} = {nl_borderline_detected/len(nl_borderline)*100:.1f}%")

            # Compare: NL detected vs Structured detected vs Structured hedged
            total_structured_recognized = borderline_detected + borderline_hedging
            print()
            if total_structured_recognized > borderline_detected:
                print(f"    → Structured recognized {total_structured_recognized}/{len(st_borderline)} signals")
                print(f"      but only committed to XML for {borderline_detected}")
                print(f"      This is format friction: recognized but didn't commit")

    # ==========================================================================
    # JUDGE-BASED HEDGING ANALYSIS
    # ==========================================================================
    if run_judge:
        print(f"\n{'='*76}")
        print("JUDGE-BASED HEDGING ANALYSIS")
        print(f"{'='*76}")
        print()
        print("Judge-based hedging: structured responses where regex says 'no XML'")
        print("but the LLM judge says 'yes, signal was acknowledged'.")
        print("This is a cleaner measure of 'recognized but didn't structure'.")
        print()

        # Count structured responses where: no XML detected but judge says acknowledged
        st_no_xml_judge_yes = [
            r for r in st_results_all
            if not r.detected and r.judge_detected is True
        ]
        st_with_judge = [r for r in st_results_all if r.judge_detected is not None]

        judge_hedging_count = len(st_no_xml_judge_yes)
        judge_hedging_rate = judge_hedging_count / len(st_with_judge) if st_with_judge else 0

        print(f"  Structured: No XML but judge says acknowledged:")
        print(f"    Overall:      {judge_hedging_count:3}/{len(st_with_judge):3} = {judge_hedging_rate*100:.1f}%")

        # Among structured failures specifically (expected True, no XML)
        st_failures_with_judge = [r for r in st_false_negatives if r.judge_detected is not None]
        st_failures_judge_yes = [r for r in st_false_negatives if r.judge_detected is True]
        if st_failures_with_judge:
            failures_judge_hedging_rate = len(st_failures_judge_yes) / len(st_failures_with_judge)
            print(f"    Among failures: {len(st_failures_judge_yes):3}/{len(st_failures_with_judge):3} = {failures_judge_hedging_rate*100:.1f}%")

            if failures_judge_hedging_rate > 0.10:
                print()
                print(f"  → {failures_judge_hedging_rate*100:.0f}% of structured failures show judge-detected acknowledgment")
                print("    This is direct evidence of format friction:")
                print("    The model recognized the signal but didn't commit to XML")

        # Compare regex-based vs judge-based hedging
        print()
        print("  Regex vs Judge hedging detection:")
        print(f"    Regex-based (NL patterns in response): {hedging_in_failures}/{len(st_false_negatives) if st_false_negatives else 0}")
        print(f"    Judge-based (LLM evaluation):          {len(st_failures_judge_yes)}/{len(st_failures_with_judge) if st_failures_with_judge else 0}")

    # ==========================================================================
    # DUAL-THRESHOLD NL DETECTION ANALYSIS
    # ==========================================================================
    print(f"\n{'='*76}")
    print("DUAL-THRESHOLD NL DETECTION ANALYSIS")
    print(f"{'='*76}")
    print()
    print("Conservative: Explicit acknowledgment of awareness")
    print("              (e.g., 'I can hear the frustration', 'sounds urgent')")
    print("Permissive:   Also includes empathetic responses")
    print("              (e.g., 'That's frustrating!', 'sorry you're experiencing')")
    print()

    dual_stats = dual_threshold_analysis(results)

    print("  NL Recall at Each Threshold:")
    cons = dual_stats["conservative"]
    perm = dual_stats["permissive"]
    print(f"    Conservative: {cons['true_positives']:3}/{cons['n_positives']:3} = {cons['recall']*100:5.1f}%  "
          f"95% CI: [{cons['ci_95'][0]*100:.1f}%, {cons['ci_95'][1]*100:.1f}%]")
    print(f"    Permissive:   {perm['true_positives']:3}/{perm['n_positives']:3} = {perm['recall']*100:5.1f}%  "
          f"95% CI: [{perm['ci_95'][0]*100:.1f}%, {perm['ci_95'][1]*100:.1f}%]")
    print()
    print(f"  Gap (Permissive - Conservative): {dual_stats['recall_gap']['permissive_minus_conservative_pp']:+.1f}pp")
    if dual_stats['recall_gap']['permissive_minus_conservative_pp'] > 0:
        print("  → Permissive catches more - may be empathy, not explicit flagging")

    dist = dual_stats["threshold_distribution"]
    print()
    print("  Threshold Distribution (all NL trials):")
    print(f"    Conservative (explicit ack):  {dist['conservative_detections']:3}/{dist['total_nl_trials']:3}")
    print(f"    Permissive only (empathy):    {dist['permissive_only_detections']:3}/{dist['total_nl_trials']:3}")
    print(f"    No detection:                 {dist['no_detection']:3}/{dist['total_nl_trials']:3}")

    if cons['n_negatives'] > 0:
        print()
        print("  False Positive Rates (CONTROL scenarios):")
        print(f"    Conservative: {cons['false_positives']:3}/{cons['n_negatives']:3} = {cons['false_positive_rate']*100:.1f}%")
        print(f"    Permissive:   {perm['false_positives']:3}/{perm['n_negatives']:3} = {perm['false_positive_rate']*100:.1f}%")

    # ==========================================================================
    # DETECTION VALIDATION - Flag suspicious detections for review
    # ==========================================================================
    print(f"\n{'='*76}")
    print("DETECTION VALIDATION - Suspicious Detections for Review")
    print(f"{'='*76}")

    detection_validation = validate_detections(results)
    print(f"\n  Total suspicious detections: {detection_validation['count']}")
    print(f"    Likely detection bugs:     {detection_validation['likely_bugs']}")
    print(f"    NL CONTROL detections:     {detection_validation['nl_control_detections']}")
    print(f"    Structured hedging cases:  {detection_validation['structured_hedging']}")

    if detection_validation['likely_bugs'] > 0:
        print(f"\n  ⚠ WARNING: {detection_validation['likely_bugs']} likely detection bugs found")
        print("    Review 'detection_validation' in output for details")

    # Summary
    print(f"\n{'='*76}")
    print("SUMMARY")
    print(f"{'='*76}")

    if scenario_stats['sign_test_p'] < 0.05:
        if scenario_stats['nl_better_count'] > scenario_stats['st_better_count']:
            print(f"\n  ✓ NL significantly outperforms Structured (p = {scenario_stats['sign_test_p']:.4f})")
        else:
            print(f"\n  ✓ Structured significantly outperforms NL (p = {scenario_stats['sign_test_p']:.4f})")
    else:
        print(f"\n  No significant difference between conditions (p = {scenario_stats['sign_test_p']:.4f})")

    # Save results
    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    output_path = output_dir / f"signal_detection_{timestamp}.json"

    output_data = {
        "metadata": {
            "timestamp": timestamp,
            "experiment": "signal_detection",
            "version": "v1",
            "num_trials": num_trials,
            "seed": seed,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "n_borderline": n_borderline,
            "total_observations": len(results),
            "failed_trials_count": len(failed_trials),
            "failed_trials": failed_trials if failed_trials else None,
        },
        "design": {
            "description": "Confound-free format friction test. Both conditions receive "
                          "identical WHEN guidance and structurally parallel HOW instructions. "
                          "Neither condition requires suppressing default behavior.",
            "nl_format": "Natural language acknowledgment in response",
            "structured_format": "<signal type='...'>reason</signal> XML tag",
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
                "n_borderline_scenarios": m["n_borderline"],
            }
            for cond, m in metrics.items()
        },
        "ambiguity_analysis": ambiguity_stats,
        "scenario_level_analysis": {
            "n_scenarios": scenario_stats["n_scenarios"],
            "nl_better_count": scenario_stats["nl_better_count"],
            "st_better_count": scenario_stats["st_better_count"],
            "ties_count": scenario_stats["ties_count"],
            "sign_test_p": scenario_stats["sign_test_p"],
            "wilcoxon_p": scenario_stats["wilcoxon_p"],
        },
        "trial_level_comparison": {
            "nl_minus_structured_pp": diff * 100,
            "mcnemar_chi_sq": chi,
            "mcnemar_p_value": p,
            "nl_wins": nl_wins,
            "structured_wins": st_wins,
            "both_succeed": both_win,
            "both_fail": both_fail,
        },
        "hedging_analysis": {
            "description": "Structured responses that acknowledged signal in NL but didn't use XML",
            "total_hedging": hedging_count,
            "hedging_rate": hedging_count / len(st_results_all) if st_results_all else 0,
            "hedging_among_failures": hedging_in_failures,
            "hedging_rate_among_failures": hedging_in_failures / len(st_false_negatives) if st_false_negatives else 0,
            "borderline_analysis": {
                "description": "BORDERLINE scenarios have no ground truth - detection rates, not success rates",
                "st_detected": sum(1 for r in st_results_all if r.ambiguity == "BORDERLINE" and r.detected),
                "st_hedged": sum(1 for r in st_results_all if r.ambiguity == "BORDERLINE" and r.nl_acknowledgment_without_xml),
                "nl_detected": sum(1 for r in nl_results_all if r.ambiguity == "BORDERLINE" and r.detected),
                "n_borderline": len([r for r in st_results_all if r.ambiguity == "BORDERLINE"]),
            },
        },
        "dual_threshold_analysis": dual_stats,
        "judge_analysis": {
            "description": "LLM judge for symmetric detection (same question for both conditions)",
            "enabled": run_judge,
            "metrics": {
                cond: {
                    "recall": judge_metrics[cond]["recall"] if run_judge and cond in judge_metrics else None,
                    "false_positive_rate": judge_metrics[cond]["fpr"] if run_judge and cond in judge_metrics else None,
                    "ci_95": list(judge_metrics[cond]["ci"]) if run_judge and cond in judge_metrics else None,
                    "true_positives": judge_metrics[cond]["tp"] if run_judge and cond in judge_metrics else None,
                    "false_positives": judge_metrics[cond]["fp"] if run_judge and cond in judge_metrics else None,
                    "errors": judge_metrics[cond]["errors"] if run_judge and cond in judge_metrics else None,
                }
                for cond in ["nl", "structured"]
            } if run_judge else None,
            "difference_pp": (judge_metrics.get("nl", {}).get("recall", 0) - judge_metrics.get("structured", {}).get("recall", 0)) * 100 if run_judge else None,
            "scenario_level": {
                "nl_better_count": judge_nl_better,
                "st_better_count": judge_st_better,
                "ties_count": judge_ties,
                "sign_test_p": judge_sign_p,
            } if run_judge else None,
            "trial_level": {
                "mcnemar_chi_sq": judge_chi,
                "mcnemar_p_value": judge_p,
                "nl_wins": judge_nl_wins,
                "structured_wins": judge_st_wins,
                "both_succeed": judge_both_win,
                "both_fail": judge_both_fail,
            } if run_judge else None,
        } if run_judge else {"enabled": False},
        "detection_validation": detection_validation,
        "results": [asdict(r) for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Validation samples
    validation_samples = validate_detection_sample(results, sample_size=30)
    validation_path = output_dir / f"signal_validation_{timestamp}.json"
    with open(validation_path, "w") as f:
        json.dump(validation_samples, f, indent=2)

    print(f"Validation samples saved to {validation_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Signal Detection Experiment - Confound-Free Format Friction Test"
    )
    parser.add_argument(
        "--trials", type=int, default=10,
        help="Trials per scenario per condition (default: 10)"
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
        help="Run specific scenario(s) by ID"
    )
    parser.add_argument(
        "--ambiguity", type=str, nargs="+",
        choices=["explicit", "implicit", "borderline", "control"],
        help="Filter by ambiguity level(s)"
    )
    parser.add_argument(
        "--no-judge", action="store_true",
        help="Skip LLM judge pass (faster but asymmetric detection)"
    )
    parser.add_argument(
        "--exclude-hard", action="store_true",
        help="Exclude HARD-tagged scenarios from main analysis"
    )

    args = parser.parse_args()

    asyncio.run(run_experiment(
        num_trials=args.trials,
        randomize_order=not args.no_randomize,
        seed=args.seed,
        scenario_ids=args.scenario,
        ambiguity_filter=args.ambiguity,
        run_judge=not args.no_judge,
        exclude_hard=args.exclude_hard,
    ))
