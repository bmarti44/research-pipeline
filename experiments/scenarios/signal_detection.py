"""
Scenario definitions for signal detection experiment.

This module defines scenarios for testing whether NL flagging outperforms
structured XML flagging when detecting frustration, urgency, or blocking
technical issues in user messages.

Key design principle: This experiment eliminates the confound present in
the memory persistence experiment. Neither condition requires suppressing
default behavior - NL says "mention it naturally" and structured says
"use this XML tag." Both are additive instructions on top of normal
assistant behavior.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import json


class SignalType(Enum):
    """Type of signal to detect in user messages."""
    FRUSTRATION = "frustration"
    URGENCY = "urgency"
    BLOCKING_ISSUE = "blocking_issue"


class AmbiguityLevel(Enum):
    """How ambiguous the signal is in the user message."""
    EXPLICIT = 4    # Unmistakable signal, direct emotional language
    IMPLICIT = 3    # Signal present but must be inferred, no emotional words
    BORDERLINE = 2  # Genuinely ambiguous, reasonable people disagree
    CONTROL = 1     # No signal present, neutral technical interactions


@dataclass
class SignalScenario:
    """A scenario for signal detection testing."""
    id: str
    query: str
    signal_type: Optional[SignalType]  # None for CONTROL scenarios
    ambiguity: AmbiguityLevel
    expected_detection: Optional[bool]  # Should the signal be flagged? None = no ground truth (BORDERLINE)
    domain: str  # e.g., "frontend", "api", "deployment", "ci", "code_review", "dependency"
    tags: list = field(default_factory=list)
    notes: Optional[str] = None  # For documenting why this is at a particular ambiguity level
    difficulty: Optional[str] = None  # "HARD" for scenarios that may be too subtle; exclude from main analysis

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "query": self.query,
            "signal_type": self.signal_type.value if self.signal_type else None,
            "ambiguity": self.ambiguity.name,
            "expected_detection": self.expected_detection,
            "domain": self.domain,
            "tags": self.tags,
            "notes": self.notes,
            "difficulty": self.difficulty,
        }


# =============================================================================
# EXPLICIT SCENARIOS (15) - Unmistakable signal
# Direct emotional language or explicit urgency markers.
# Both conditions should detect these near 100%.
# =============================================================================

EXPLICIT_SCENARIOS = [
    # Frustration - 5 scenarios
    SignalScenario(
        id="sig_explicit_frust_001",
        query="I'm really frustrated - this is the third regression this sprint and it's breaking the same feature we already fixed twice.",
        signal_type=SignalType.FRUSTRATION,
        ambiguity=AmbiguityLevel.EXPLICIT,
        expected_detection=True,
        domain="frontend",
        notes="Direct use of 'frustrated' plus pattern of repeated issues",
    ),
    SignalScenario(
        id="sig_explicit_frust_002",
        query="This is incredibly frustrating. I've spent 4 hours on what should be a 10 minute fix because the documentation is completely wrong.",
        signal_type=SignalType.FRUSTRATION,
        ambiguity=AmbiguityLevel.EXPLICIT,
        expected_detection=True,
        domain="api",
        notes="Explicit 'frustrating' plus time waste and documentation complaint",
    ),
    SignalScenario(
        id="sig_explicit_frust_003",
        query="I'm so annoyed right now. The CI keeps failing on tests that pass locally and nobody can explain why.",
        signal_type=SignalType.FRUSTRATION,
        ambiguity=AmbiguityLevel.EXPLICIT,
        expected_detection=True,
        domain="ci",
        notes="Explicit 'annoyed' plus mysterious CI failures",
    ),
    SignalScenario(
        id="sig_explicit_frust_004",
        query="This is driving me crazy - the deployment fails silently with no error logs and I have no idea what's wrong.",
        signal_type=SignalType.FRUSTRATION,
        ambiguity=AmbiguityLevel.EXPLICIT,
        expected_detection=True,
        domain="deployment",
        notes="Explicit 'driving me crazy' plus silent failure",
    ),
    SignalScenario(
        id="sig_explicit_frust_005",
        query="I'm at my wit's end with this dependency conflict. Every time I upgrade one package, three others break.",
        signal_type=SignalType.FRUSTRATION,
        ambiguity=AmbiguityLevel.EXPLICIT,
        expected_detection=True,
        domain="dependency",
        notes="Explicit 'at my wit's end' plus cascading dependency issues",
    ),

    # Urgency - 5 scenarios
    SignalScenario(
        id="sig_explicit_urg_001",
        query="URGENT: Production payments are failing and we're losing transactions. Need help immediately.",
        signal_type=SignalType.URGENCY,
        ambiguity=AmbiguityLevel.EXPLICIT,
        expected_detection=True,
        domain="api",
        notes="All caps URGENT, production impact, 'immediately'",
    ),
    SignalScenario(
        id="sig_explicit_urg_002",
        query="This is time-sensitive - the client demo is in 2 hours and the login page is completely broken.",
        signal_type=SignalType.URGENCY,
        ambiguity=AmbiguityLevel.EXPLICIT,
        expected_detection=True,
        domain="frontend",
        notes="Explicit 'time-sensitive' plus concrete deadline",
    ),
    SignalScenario(
        id="sig_explicit_urg_003",
        query="Emergency: the main database is returning connection timeouts and users can't access their data.",
        signal_type=SignalType.URGENCY,
        ambiguity=AmbiguityLevel.EXPLICIT,
        expected_detection=True,
        domain="api",
        notes="Explicit 'Emergency' plus user-facing data access issue",
    ),
    SignalScenario(
        id="sig_explicit_urg_004",
        query="Critical priority - we need to hotfix this security vulnerability before the disclosure deadline tomorrow.",
        signal_type=SignalType.URGENCY,
        ambiguity=AmbiguityLevel.EXPLICIT,
        expected_detection=True,
        domain="deployment",
        notes="Explicit 'Critical priority' plus security deadline",
    ),
    SignalScenario(
        id="sig_explicit_urg_005",
        query="This needs to be fixed ASAP - the CEO is presenting this feature to the board in 30 minutes.",
        signal_type=SignalType.URGENCY,
        ambiguity=AmbiguityLevel.EXPLICIT,
        expected_detection=True,
        domain="frontend",
        notes="Explicit 'ASAP' plus executive visibility and time pressure",
    ),

    # Blocking issue - 5 scenarios
    SignalScenario(
        id="sig_explicit_block_001",
        query="I've been completely blocked on this for two days and the whole team is waiting on me to finish.",
        signal_type=SignalType.BLOCKING_ISSUE,
        ambiguity=AmbiguityLevel.EXPLICIT,
        expected_detection=True,
        domain="code_review",
        notes="Explicit 'completely blocked' plus team dependency",
    ),
    SignalScenario(
        id="sig_explicit_block_002",
        query="This is a blocker - I can't proceed with any of my other tasks until this API issue is resolved.",
        signal_type=SignalType.BLOCKING_ISSUE,
        ambiguity=AmbiguityLevel.EXPLICIT,
        expected_detection=True,
        domain="api",
        notes="Explicit 'blocker' plus inability to proceed",
    ),
    SignalScenario(
        id="sig_explicit_block_003",
        query="My work is completely stuck. The build has been failing for 3 days and I can't test anything.",
        signal_type=SignalType.BLOCKING_ISSUE,
        ambiguity=AmbiguityLevel.EXPLICIT,
        expected_detection=True,
        domain="ci",
        notes="Explicit 'completely stuck' plus multi-day impact",
    ),
    SignalScenario(
        id="sig_explicit_block_004",
        query="This is blocking the entire release. We can't ship until the deployment pipeline is fixed.",
        signal_type=SignalType.BLOCKING_ISSUE,
        ambiguity=AmbiguityLevel.EXPLICIT,
        expected_detection=True,
        domain="deployment",
        notes="Explicit 'blocking the entire release'",
    ),
    SignalScenario(
        id="sig_explicit_block_005",
        query="I'm totally blocked - this circular dependency means I can't even run the app locally.",
        signal_type=SignalType.BLOCKING_ISSUE,
        ambiguity=AmbiguityLevel.EXPLICIT,
        expected_detection=True,
        domain="dependency",
        notes="Explicit 'totally blocked' plus inability to run locally",
    ),
]


# =============================================================================
# IMPLICIT SCENARIOS (25) - Signal present but must be inferred
# No emotional words. The signal is clear from context but never stated.
# This is where we expect format friction to have the most effect.
# =============================================================================

IMPLICIT_SCENARIOS = [
    # Frustration - 8 scenarios
    #
    # FIX (Problem 4): sig_implicit_frust_001 had <30% detection for both conditions.
    # "Another ticket" was too ambiguous - could be factual or frustration.
    # Added concrete context ("third one this month") to make repetition clearer
    # while still avoiding explicit frustration language.
    SignalScenario(
        id="sig_implicit_frust_001",
        query="I opened another ticket about the 500 errors on that endpoint. This is the third one this month.",
        signal_type=SignalType.FRUSTRATION,
        ambiguity=AmbiguityLevel.IMPLICIT,
        expected_detection=True,
        domain="api",
        notes="'third one this month' makes repetition concrete - clear implicit frustration signal",
    ),
    SignalScenario(
        id="sig_implicit_frust_002",
        query="I went through the troubleshooting guide but I'm still seeing the same behavior.",
        signal_type=SignalType.FRUSTRATION,
        ambiguity=AmbiguityLevel.IMPLICIT,
        expected_detection=True,
        domain="deployment",
        notes="Followed instructions, didn't work. Neutral report or frustration?",
    ),
    # NOTE: This scenario shows anomalous behavior discovered during REVIEW.md remediation.
    # The model consistently tags this as "blocking_issue" not "frustration" because
    # the query reads as an environment discrepancy that blocks deployment.
    # When "Again." was stripped, the frustration signal may have been lost.
    # This scenario shows -50pp friction due to signal type mismatch between
    # scenario definition (frustration) and model's interpretation (blocking_issue).
    # Keeping original label for data consistency; noted in paper §3.6.1.
    SignalScenario(
        id="sig_implicit_frust_003",
        query="The tests pass locally and in staging but not in production.",
        signal_type=SignalType.FRUSTRATION,  # Model interprets as blocking_issue
        ambiguity=AmbiguityLevel.IMPLICIT,
        expected_detection=True,
        domain="ci",
        notes="Stripped 'Again.' — now it's just an environment discrepancy report. Model tags as blocking_issue.",
    ),
    SignalScenario(
        id="sig_implicit_frust_004",
        query="I updated the component based on the review feedback and it's still not passing review.",
        signal_type=SignalType.FRUSTRATION,
        ambiguity=AmbiguityLevel.IMPLICIT,
        expected_detection=True,
        domain="code_review",
        notes="Single iteration mentioned, not three. Could be normal back-and-forth.",
    ),
    # RELABELED: Was IMPLICIT frustration, now CONTROL - just a factual bug report
    SignalScenario(
        id="sig_implicit_frust_005",
        query="Something in the build broke after the React update.",
        signal_type=None,  # Changed: no signal present
        ambiguity=AmbiguityLevel.CONTROL,  # Changed: relabeled
        expected_detection=False,  # Changed: should NOT detect
        domain="dependency",
        notes="RELABELED from IMPLICIT. 0/0 in pilot - this is just a factual bug report, no frustration signal.",
    ),
    SignalScenario(
        id="sig_implicit_frust_006",
        query="The documentation doesn't quite match what the API actually returns.",
        signal_type=SignalType.FRUSTRATION,
        ambiguity=AmbiguityLevel.IMPLICIT,
        expected_detection=True,
        domain="api",
        notes="Softened with 'quite'. Observation or complaint?",
    ),
    SignalScenario(
        id="sig_implicit_frust_007",
        query="I've been looking into this error but can't find anything about it online.",
        signal_type=SignalType.FRUSTRATION,
        ambiguity=AmbiguityLevel.IMPLICIT,
        expected_detection=True,
        domain="frontend",
        notes="Stripped '6 hours'. Vague duration, no dramatic framing.",
        difficulty="HARD",  # 0/0 in pilot - signal may be too subtle
    ),
    SignalScenario(
        id="sig_implicit_frust_008",
        query="The PR feedback seems different from what was approved on a similar change before.",
        signal_type=SignalType.FRUSTRATION,
        ambiguity=AmbiguityLevel.IMPLICIT,
        expected_detection=True,
        domain="code_review",
        notes="'seems' and 'similar' add uncertainty. Observation or grievance?",
    ),

    # Urgency - 8 scenarios
    SignalScenario(
        id="sig_implicit_urg_001",
        query="The feature I was going to demo to leadership isn't quite working yet.",
        signal_type=SignalType.URGENCY,
        ambiguity=AmbiguityLevel.IMPLICIT,
        expected_detection=True,
        domain="frontend",
        notes="No 'yesterday', no 'VP'. Vague timeline, vague audience.",
    ),
    # NOTE: This scenario had a labeling bug discovered during REVIEW.md remediation.
    # Original: signal_type=SignalType.FRUSTRATION (ID mismatch - "urg" in ID)
    # This caused -100pp "negative friction" because the model correctly tagged as urgency
    # but evaluation expected frustration. Fixed 2026-02-03.
    SignalScenario(
        id="sig_implicit_urg_002",
        query="The deployment has been queued for a bit and there's a customer go-live coming up.",
        signal_type=SignalType.URGENCY,  # FIXED: was FRUSTRATION, should be URGENCY per ID
        ambiguity=AmbiguityLevel.IMPLICIT,
        expected_detection=True,
        domain="deployment",
        notes="'a bit' and 'coming up' are deliberately vague. When? How soon?",
    ),
    # RELABELED: Was IMPLICIT urgency, now CONTROL - just informational context
    SignalScenario(
        id="sig_implicit_urg_003",
        query="We're running a little below our uptime target for the month.",
        signal_type=None,  # Changed: no signal present
        ambiguity=AmbiguityLevel.CONTROL,  # Changed: relabeled
        expected_detection=False,  # Changed: should NOT detect
        domain="api",
        notes="RELABELED from IMPLICIT. 0/0 in pilot - this is informational context, not urgency.",
    ),
    SignalScenario(
        id="sig_implicit_urg_004",
        query="The security audit is coming up and there are still some critical findings open.",
        signal_type=SignalType.URGENCY,
        ambiguity=AmbiguityLevel.IMPLICIT,
        expected_detection=True,
        domain="dependency",
        notes="No '3 days', no '47 vulnerabilities'. Vague timeline, vague count.",
    ),
    SignalScenario(
        id="sig_implicit_urg_005",
        query="It's a busy sales period and the cart page is having some issues.",
        signal_type=SignalType.URGENCY,
        ambiguity=AmbiguityLevel.IMPLICIT,
        expected_detection=True,
        domain="frontend",
        notes="No 'Black Friday', no '80%'. Generic business pressure.",
    ),
    SignalScenario(
        id="sig_implicit_urg_006",
        query="The migration window is tonight and we're still working through it.",
        signal_type=SignalType.URGENCY,
        ambiguity=AmbiguityLevel.IMPLICIT,
        expected_detection=True,
        domain="deployment",
        notes="Deadline present but 'still working through it' could mean on track.",
    ),
    SignalScenario(
        id="sig_implicit_urg_007",
        query="The other office is going to need this fix when they come online in the morning.",
        signal_type=SignalType.URGENCY,
        ambiguity=AmbiguityLevel.IMPLICIT,
        expected_detection=True,
        domain="deployment",
        notes="No '4 hours'. Vague timeline. Need or preference?",
    ),
    SignalScenario(
        id="sig_implicit_urg_008",
        query="There are a few other PRs waiting on this review and the sprint is wrapping up.",
        signal_type=SignalType.URGENCY,
        ambiguity=AmbiguityLevel.IMPLICIT,
        expected_detection=True,
        domain="code_review",
        notes="'a few', 'wrapping up' — deliberately imprecise.",
    ),

    # Blocking issue - 9 scenarios
    SignalScenario(
        id="sig_implicit_block_001",
        query="I'm writing the tests but the test database isn't available yet.",
        signal_type=SignalType.BLOCKING_ISSUE,
        ambiguity=AmbiguityLevel.IMPLICIT,
        expected_detection=True,
        domain="ci",
        notes="Separated cause and effect. 'Writing tests' implies work continues around it.",
        difficulty="HARD",  # 0/0 in pilot - 'yet' softens the blocking implication
    ),
    SignalScenario(
        id="sig_implicit_block_002",
        query="The API endpoint I need to integrate with returns 'coming soon' for every request.",
        signal_type=SignalType.BLOCKING_ISSUE,
        ambiguity=AmbiguityLevel.IMPLICIT,
        expected_detection=True,
        domain="api",
        notes="KEEP — this is the one that showed friction in original results.",
    ),
    SignalScenario(
        id="sig_implicit_block_003",
        query="My feature branch has drifted pretty far from main and the merges are getting messy.",
        signal_type=SignalType.BLOCKING_ISSUE,
        ambiguity=AmbiguityLevel.IMPLICIT,
        expected_detection=True,
        domain="code_review",
        notes="No '200 commits'. 'Pretty far' and 'getting messy' — how bad?",
    ),
    SignalScenario(
        id="sig_implicit_block_004",
        query="The npm registry is returning 403 for some of our packages.",
        signal_type=SignalType.BLOCKING_ISSUE,
        ambiguity=AmbiguityLevel.IMPLICIT,
        expected_detection=True,
        domain="dependency",
        notes="'some' not 'all'. Partial failure — blocking or workaround-able?",
    ),
    # RELABELED: Was IMPLICIT blocking, now CONTROL - flaky ≠ blocked
    SignalScenario(
        id="sig_implicit_block_005",
        query="The staging environment has been a bit flaky since the network change.",
        signal_type=None,  # Changed: no signal present
        ambiguity=AmbiguityLevel.CONTROL,  # Changed: relabeled
        expected_detection=False,  # Changed: should NOT detect
        domain="deployment",
        notes="RELABELED from IMPLICIT. 0/0 in pilot - 'a bit flaky' is intermittent issues, not blocking.",
    ),
    SignalScenario(
        id="sig_implicit_block_006",
        query="I'm waiting on the new design tokens from the design team to finish this up.",
        signal_type=SignalType.BLOCKING_ISSUE,
        ambiguity=AmbiguityLevel.IMPLICIT,
        expected_detection=True,
        domain="frontend",
        notes="'waiting on' could be normal workflow, not blocked.",
    ),
    SignalScenario(
        id="sig_implicit_block_007",
        query="The CI queue is pretty backed up today and my builds are slow.",
        signal_type=SignalType.BLOCKING_ISSUE,
        ambiguity=AmbiguityLevel.IMPLICIT,
        expected_detection=True,
        domain="ci",
        notes="'pretty backed up' and 'slow' — not 'timing out'. Inconvenience or blocked?",
    ),
    SignalScenario(
        id="sig_implicit_block_008",
        query="I put in a request for production log access but it's still pending.",
        signal_type=SignalType.BLOCKING_ISSUE,
        ambiguity=AmbiguityLevel.IMPLICIT,
        expected_detection=True,
        domain="deployment",
        notes="No 'a week'. Just 'still pending'. Could be yesterday.",
        difficulty="HARD",  # 0/0 in pilot - blocking may not be clear from context
    ),
    SignalScenario(
        id="sig_implicit_block_009",
        query="The package we use dropped support for our Node version and the upgrade hasn't been approved yet.",
        signal_type=SignalType.BLOCKING_ISSUE,
        ambiguity=AmbiguityLevel.IMPLICIT,
        expected_detection=True,
        domain="dependency",
        notes="Removed specific versions. Factual statement of dependency situation.",
    ),
]


# =============================================================================
# BORDERLINE SCENARIOS (15) - Genuinely ambiguous
# Reasonable people would disagree on whether these contain a signal.
# Tests the model's judgment under maximum uncertainty.
#
# CALIBRATION: Detection rate should be ~40-60% for both conditions.
# If both detect at 80%+, the scenario is too obvious.
# If both detect at <20%, it's effectively a control.
#
# PREVIOUS VERSION had 78.7% detection rate - too easy. Replaced with
# genuinely ambiguous scenarios per review feedback.
# =============================================================================

BORDERLINE_SCENARIOS = [
    # =============================================================================
    # CALIBRATION FIX (Problem 3): Previous borderline scenarios had 0% detection
    # from both conditions because they lacked technical context. The model would
    # respond "I don't have enough context" and move on.
    #
    # Fix: Add technical detail so the model can form an opinion. A borderline
    # scenario should describe a real situation where frustration/urgency/blocking
    # MIGHT be present, but hasn't been stated. Target detection rate: 30-70%.
    # =============================================================================

    # Frustration-borderline: Technical context, no emotional language
    # These describe situations that MIGHT cause frustration but don't express it
    SignalScenario(
        id="sig_borderline_001",
        query="I've gone through the migration guide, updated the config, and verified the dependencies, but the module still isn't resolving correctly.",
        signal_type=SignalType.FRUSTRATION,
        ambiguity=AmbiguityLevel.BORDERLINE,
        expected_detection=None,
        domain="dependency",
        notes="Multiple steps taken without success - could be frustrating or just factual report",
    ),
    SignalScenario(
        id="sig_borderline_002",
        query="I ran the linter, fixed the warnings it flagged, and the same three test failures are still showing up in CI.",
        signal_type=SignalType.FRUSTRATION,
        ambiguity=AmbiguityLevel.BORDERLINE,
        expected_detection=None,
        domain="ci",
        notes="Effort invested without resolution - pattern of repeated issues",
    ),
    SignalScenario(
        id="sig_borderline_003",
        query="I've checked the docs for this API and the response format doesn't match what's documented for the v3 endpoint.",
        signal_type=SignalType.FRUSTRATION,
        ambiguity=AmbiguityLevel.BORDERLINE,
        expected_detection=None,
        domain="api",
        notes="Documentation mismatch found - could imply frustration or just observation",
    ),
    SignalScenario(
        id="sig_borderline_004",
        query="Hmm, that didn't work either.",
        signal_type=SignalType.FRUSTRATION,
        ambiguity=AmbiguityLevel.BORDERLINE,
        expected_detection=None,
        domain="deployment",
        notes="Brief update - 'hmm' is mild, 'either' implies prior attempts",
    ),
    SignalScenario(
        id="sig_borderline_005",
        query="I've gone through the standard debugging steps - cleared caches, restarted services, checked the logs - and I'm still seeing intermittent 504s.",
        signal_type=SignalType.FRUSTRATION,
        ambiguity=AmbiguityLevel.BORDERLINE,
        expected_detection=None,
        domain="api",
        notes="Systematic debugging without success - could be frustrating or methodical",
    ),

    # Urgency-borderline: Timeline pressure without explicit urgency markers
    SignalScenario(
        id="sig_borderline_006",
        query="The feature freeze is Thursday and I still need to get the auth changes through code review.",
        signal_type=SignalType.URGENCY,
        ambiguity=AmbiguityLevel.BORDERLINE,
        expected_detection=None,
        domain="code_review",
        notes="Concrete deadline mentioned with pending work - implies time pressure",
    ),
    SignalScenario(
        id="sig_borderline_007",
        query="The team was asking about the timeline.",
        signal_type=SignalType.URGENCY,
        ambiguity=AmbiguityLevel.BORDERLINE,
        expected_detection=None,
        domain="code_review",
        notes="Third-party pressure mentioned casually",
    ),
    SignalScenario(
        id="sig_borderline_008",
        query="We told the client we'd have the fix deployed by end of week. Still working through the edge cases.",
        signal_type=SignalType.URGENCY,
        ambiguity=AmbiguityLevel.BORDERLINE,
        expected_detection=None,
        domain="deployment",
        notes="Client commitment with incomplete work - implies but doesn't state urgency",
    ),
    SignalScenario(
        id="sig_borderline_009",
        query="This is on the roadmap for next sprint.",
        signal_type=SignalType.URGENCY,
        ambiguity=AmbiguityLevel.BORDERLINE,
        expected_detection=None,
        domain="frontend",
        notes="Timeline context - informational or hinting at priority?",
    ),
    SignalScenario(
        id="sig_borderline_010",
        query="The release train leaves Monday and this PR has two outstanding review comments.",
        signal_type=SignalType.URGENCY,
        ambiguity=AmbiguityLevel.BORDERLINE,
        expected_detection=None,
        domain="code_review",
        notes="Deadline + pending blockers - urgency implied but not stated",
    ),

    # Blocking-borderline: Describes impediments without using "blocked"
    SignalScenario(
        id="sig_borderline_011",
        query="I can write the integration tests but the sandbox API credentials haven't been provisioned yet.",
        signal_type=SignalType.BLOCKING_ISSUE,
        ambiguity=AmbiguityLevel.BORDERLINE,
        expected_detection=None,
        domain="ci",
        notes="Work possible in parallel vs true blocker - ambiguous dependency",
    ),
    SignalScenario(
        id="sig_borderline_012",
        query="The refactor is ready to go but it depends on the schema migration that's still in review.",
        signal_type=SignalType.BLOCKING_ISSUE,
        ambiguity=AmbiguityLevel.BORDERLINE,
        expected_detection=None,
        domain="api",
        notes="Dependency on pending work - blocking or just sequencing?",
    ),
    SignalScenario(
        id="sig_borderline_013",
        query="I've got the component working locally but the shared design tokens package hasn't been published to the registry.",
        signal_type=SignalType.BLOCKING_ISSUE,
        ambiguity=AmbiguityLevel.BORDERLINE,
        expected_detection=None,
        domain="frontend",
        notes="Local success + missing dependency - blocked or workaround possible?",
    ),
    SignalScenario(
        id="sig_borderline_014",
        query="I'll need to figure this out before moving on.",
        signal_type=SignalType.BLOCKING_ISSUE,
        ambiguity=AmbiguityLevel.BORDERLINE,
        expected_detection=None,
        domain="deployment",
        notes="Dependency stated - blocking or just next step?",
    ),
    SignalScenario(
        id="sig_borderline_015",
        query="I'm looking into alternatives in the meantime.",
        signal_type=SignalType.BLOCKING_ISSUE,
        ambiguity=AmbiguityLevel.BORDERLINE,
        expected_detection=None,
        domain="api",
        notes="Workaround mentioned - blocked or just being proactive?",
    ),
]


# =============================================================================
# CONTROL SCENARIOS (20) - No signal present
# Neutral technical interactions. Should NOT trigger detection.
# =============================================================================

CONTROL_SCENARIOS = [
    # Neutral technical questions - 5 scenarios
    SignalScenario(
        id="sig_control_001",
        query="Can you explain how connection pooling works in this context?",
        signal_type=None,
        ambiguity=AmbiguityLevel.CONTROL,
        expected_detection=False,
        domain="api",
        notes="Pure technical question, no signal",
    ),
    SignalScenario(
        id="sig_control_002",
        query="What's the difference between useEffect and useLayoutEffect?",
        signal_type=None,
        ambiguity=AmbiguityLevel.CONTROL,
        expected_detection=False,
        domain="frontend",
        notes="Conceptual question, no signal",
    ),
    SignalScenario(
        id="sig_control_003",
        query="How should I structure the test fixtures for this module?",
        signal_type=None,
        ambiguity=AmbiguityLevel.CONTROL,
        expected_detection=False,
        domain="ci",
        notes="Asking for guidance, no signal",
    ),
    SignalScenario(
        id="sig_control_004",
        query="What are the best practices for blue-green deployments?",
        signal_type=None,
        ambiguity=AmbiguityLevel.CONTROL,
        expected_detection=False,
        domain="deployment",
        notes="General knowledge question, no signal",
    ),
    SignalScenario(
        id="sig_control_005",
        query="Should I use yarn or npm for this project?",
        signal_type=None,
        ambiguity=AmbiguityLevel.CONTROL,
        expected_detection=False,
        domain="dependency",
        notes="Seeking recommendation, no signal",
    ),

    # Neutral status updates - 5 scenarios
    SignalScenario(
        id="sig_control_006",
        query="Here's the stack trace from the latest error.",
        signal_type=None,
        ambiguity=AmbiguityLevel.CONTROL,
        expected_detection=False,
        domain="api",
        notes="Sharing information, no emotional content",
    ),
    SignalScenario(
        id="sig_control_007",
        query="I pushed the changes to the feature branch.",
        signal_type=None,
        ambiguity=AmbiguityLevel.CONTROL,
        expected_detection=False,
        domain="code_review",
        notes="Status update, neutral",
    ),
    SignalScenario(
        id="sig_control_008",
        query="The build is currently running. Should be done in about 10 minutes.",
        signal_type=None,
        ambiguity=AmbiguityLevel.CONTROL,
        expected_detection=False,
        domain="ci",
        notes="Normal workflow update, no urgency",
    ),
    SignalScenario(
        id="sig_control_009",
        query="I'm setting up the local environment now.",
        signal_type=None,
        ambiguity=AmbiguityLevel.CONTROL,
        expected_detection=False,
        domain="deployment",
        notes="Progress update, neutral",
    ),
    SignalScenario(
        id="sig_control_010",
        query="Let me try updating to the latest version and see if that helps.",
        signal_type=None,
        ambiguity=AmbiguityLevel.CONTROL,
        expected_detection=False,
        domain="dependency",
        notes="Active troubleshooting, neutral tone",
    ),

    # Positive/resolution messages - 5 scenarios
    SignalScenario(
        id="sig_control_011",
        query="Thanks, that fixed it.",
        signal_type=None,
        ambiguity=AmbiguityLevel.CONTROL,
        expected_detection=False,
        domain="frontend",
        notes="Resolution confirmation, positive",
    ),
    SignalScenario(
        id="sig_control_012",
        query="The tests are passing now. I'll open the PR.",
        signal_type=None,
        ambiguity=AmbiguityLevel.CONTROL,
        expected_detection=False,
        domain="ci",
        notes="Positive progress, no signal",
    ),
    SignalScenario(
        id="sig_control_013",
        query="Good catch! I'll add that to the implementation.",
        signal_type=None,
        ambiguity=AmbiguityLevel.CONTROL,
        expected_detection=False,
        domain="code_review",
        notes="Positive feedback, collaborative",
    ),
    SignalScenario(
        id="sig_control_014",
        query="The deployment went smoothly. All services are healthy.",
        signal_type=None,
        ambiguity=AmbiguityLevel.CONTROL,
        expected_detection=False,
        domain="deployment",
        notes="Success report, positive",
    ),
    SignalScenario(
        id="sig_control_015",
        query="I found the issue - it was a typo in the config file.",
        signal_type=None,
        ambiguity=AmbiguityLevel.CONTROL,
        expected_detection=False,
        domain="api",
        notes="Resolution with explanation, neutral/positive",
    ),

    # Neutral planning/coordination - 5 scenarios
    SignalScenario(
        id="sig_control_016",
        query="I'll work on this tomorrow morning.",
        signal_type=None,
        ambiguity=AmbiguityLevel.CONTROL,
        expected_detection=False,
        domain="frontend",
        notes="Schedule statement, no urgency implied",
    ),
    SignalScenario(
        id="sig_control_017",
        query="Let's sync up after the standup to discuss the approach.",
        signal_type=None,
        ambiguity=AmbiguityLevel.CONTROL,
        expected_detection=False,
        domain="code_review",
        notes="Meeting coordination, neutral",
    ),
    SignalScenario(
        id="sig_control_018",
        query="I've added some comments to the PR for context.",
        signal_type=None,
        ambiguity=AmbiguityLevel.CONTROL,
        expected_detection=False,
        domain="code_review",
        notes="Information sharing, neutral",
    ),
    SignalScenario(
        id="sig_control_019",
        query="I'm going to grab lunch and then continue debugging.",
        signal_type=None,
        ambiguity=AmbiguityLevel.CONTROL,
        expected_detection=False,
        domain="api",
        notes="Casual update, no signal",
    ),
    SignalScenario(
        id="sig_control_020",
        query="This looks good to me. Approving the PR.",
        signal_type=None,
        ambiguity=AmbiguityLevel.CONTROL,
        expected_detection=False,
        domain="code_review",
        notes="Approval, positive/neutral",
    ),
]


# =============================================================================
# ALL SCENARIOS
# =============================================================================

ALL_SIGNAL_SCENARIOS = (
    BORDERLINE_SCENARIOS +
	IMPLICIT_SCENARIOS +
    EXPLICIT_SCENARIOS +
    CONTROL_SCENARIOS
)


def get_scenarios_by_ambiguity(level: AmbiguityLevel) -> list[SignalScenario]:
    """Get all scenarios at a specific ambiguity level."""
    return [s for s in ALL_SIGNAL_SCENARIOS if s.ambiguity == level]


def get_scenarios_by_signal_type(signal_type: SignalType) -> list[SignalScenario]:
    """Get all scenarios for a specific signal type."""
    return [s for s in ALL_SIGNAL_SCENARIOS if s.signal_type == signal_type]


def get_scenarios_by_domain(domain: str) -> list[SignalScenario]:
    """Get all scenarios for a specific domain."""
    return [s for s in ALL_SIGNAL_SCENARIOS if s.domain == domain]


def export_scenarios(filepath: str) -> None:
    """Export all scenarios to JSON."""
    data = {
        "total": len(ALL_SIGNAL_SCENARIOS),
        "by_ambiguity": {
            level.name: len(get_scenarios_by_ambiguity(level))
            for level in AmbiguityLevel
        },
        "by_signal_type": {
            "frustration": len(get_scenarios_by_signal_type(SignalType.FRUSTRATION)),
            "urgency": len(get_scenarios_by_signal_type(SignalType.URGENCY)),
            "blocking_issue": len(get_scenarios_by_signal_type(SignalType.BLOCKING_ISSUE)),
        },
        "scenarios": [s.to_dict() for s in ALL_SIGNAL_SCENARIOS],
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    print(f"Total scenarios: {len(ALL_SIGNAL_SCENARIOS)}")
    print(f"\nBy ambiguity level:")
    for level in AmbiguityLevel:
        count = len(get_scenarios_by_ambiguity(level))
        print(f"  {level.name}: {count}")
    print(f"\nBy signal type:")
    for signal_type in SignalType:
        count = len(get_scenarios_by_signal_type(signal_type))
        print(f"  {signal_type.name}: {count}")
