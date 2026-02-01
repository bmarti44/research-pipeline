# Tool Calling Validator Research - v8
## Claude Agent SDK with Max Subscription

**Architecture**: Uses the Claude Agent SDK (`claude-agent-sdk`) with your Max plan. The SDK's `PreToolUse` and `PostToolUse` hooks ARE the validator. Uses `ClaudeSDKClient` (not bare `query()`) for hook support.

**Changes from v7**:
- Fixed: Use `ClaudeSDKClient` context manager instead of `query()` for hooks
- Fixed: Added F10 (duplicate search) and F13 (hallucinated path) scenarios
- Fixed: Implemented `PostToolUse` hook to track `known_paths` for F13
- Fixed: Scoring logic distinguishes "correct no-tool" from "validator blocked"
- Fixed: Added convergence control with escalating feedback and forced termination
- Fixed: Multiple trials per scenario (n=5) for statistical validity
- Fixed: Added verification gates after each phase

---

## What We're Testing

**Hypothesis**: A rule-based validator that intercepts tool calls before execution improves tool selection accuracy.

**Method**: Run identical scenarios through the SDK twice:
1. **Baseline**: No hooks (Claude decides freely)
2. **Validated**: PreToolUse/PostToolUse hooks block/redirect bad tool calls

**Metrics**:
- Mean score difference (0-3 scale)
- Validator catch rate (% of bad calls blocked)
- False positive rate (% of good calls incorrectly blocked)

---

## Prerequisites

### 1. Install Claude Code CLI

```bash
# macOS/Linux
curl -fsSL https://claude.ai/install.sh | bash

# Or via npm
npm install -g @anthropic-ai/claude-code
```

### 2. Authenticate with Max Plan

```bash
# This opens browser for OAuth - select your Max account
claude login

# Verify authentication works
claude --version
```

### 3. Verify SDK Can Connect

```bash
# Quick test that SDK works with your auth
python -c "
import asyncio
from claude_agent_sdk import query

async def test():
    async for msg in query(prompt='Say OK'):
        print(type(msg).__name__)

asyncio.run(test())
"
```

If you see message types printed, you're authenticated.

---

## Project Structure

```
tool-validator-sdk/
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── validator/
│   │   ├── __init__.py
│   │   ├── hooks.py          # PreToolUse + PostToolUse hooks
│   │   ├── rules.py          # Validation rule logic
│   │   ├── semantic.py       # Semantic classifier
│   │   ├── exemplars.py      # Unified exemplars
│   │   └── convergence.py    # Convergence control
│   └── scoring.py            # Score computation
├── scenarios/
│   ├── generator.py          # Scenario generation
│   └── generated/
│       └── scenarios.json
├── calibration/
│   ├── labeled_queries.json
│   ├── calibrate.py
│   └── thresholds.json
├── experiments/
│   ├── runner.py             # Main experiment runner
│   └── results/
└── analysis/
    └── compare.py            # Statistical analysis
```

---

## PHASE 1: Project Setup

### Step 1.1: Create Project

```bash
mkdir tool-validator-sdk && cd tool-validator-sdk
```

**Create**: `pyproject.toml`

```toml
[project]
name = "tool-validator-sdk"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "claude-agent-sdk>=0.1.20",
    "sentence-transformers>=2.2.0",
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "anyio>=4.0.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0.0", "pytest-asyncio>=0.21.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

**Setup**:
```bash
uv venv && source .venv/bin/activate
uv sync
mkdir -p src/validator scenarios/generated calibration experiments/results analysis tests
touch src/__init__.py src/validator/__init__.py
```

### 🚫 GATE 1.1: SDK Import Check

```bash
uv run python -c "
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, HookMatcher
from claude_agent_sdk import AssistantMessage, TextBlock, ToolUseBlock
print('SDK_IMPORT_OK')
"
```

**Expected output**: `SDK_IMPORT_OK`

**If this fails**: Check that `claude-agent-sdk` is installed and you're in the venv.

---

## PHASE 2: Semantic Classifier

### Step 2.1: Unified Exemplars

**Create**: `src/validator/exemplars.py`

```python
"""
Unified exemplars for semantic classification.
Used in both calibration and runtime.
"""

STATIC_KNOWLEDGE_EXEMPLARS = [
    "What is the capital of France?",
    "What is photosynthesis?",
    "Define recursion",
    "Explain how inheritance works in OOP",
    "What is the Pythagorean theorem?",
    "Who wrote Romeo and Juliet?",
    "What is the speed of light?",
    "Explain the theory of relativity",
]

MEMORY_REFERENCE_EXEMPLARS = [
    "What did we talk about yesterday?",
    "Remember when we discussed the project?",
    "You mentioned something about deadlines earlier",
    "What did I tell you about my preferences?",
    "In our last conversation you said",
    "We talked about this before",
    "What was my budget again?",
    "As I mentioned previously",
]
```

### Step 2.2: Semantic Classifier

**Create**: `src/validator/semantic.py`

```python
"""Semantic classification for query types."""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from .exemplars import STATIC_KNOWLEDGE_EXEMPLARS, MEMORY_REFERENCE_EXEMPLARS


@dataclass
class Thresholds:
    static_knowledge: float = 0.60
    memory_reference: float = 0.55
    duplicate_search: float = 0.85


class SemanticClassifier:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        thresholds_path: Optional[Path] = None,
    ):
        self.model = SentenceTransformer(model_name)
        self.thresholds = self._load_thresholds(thresholds_path)

        self._static_centroid = self._compute_centroid(STATIC_KNOWLEDGE_EXEMPLARS)
        self._memory_centroid = self._compute_centroid(MEMORY_REFERENCE_EXEMPLARS)

    def _load_thresholds(self, path: Optional[Path]) -> Thresholds:
        if path is None:
            path = Path(__file__).parent.parent.parent / "calibration" / "thresholds.json"

        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return Thresholds(
                static_knowledge=data.get("static_knowledge", {}).get("threshold", 0.60),
                memory_reference=data.get("memory_reference", {}).get("threshold", 0.55),
                duplicate_search=data.get("duplicate_search", {}).get("threshold", 0.85),
            )
        return Thresholds()

    def _compute_centroid(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(texts)
        return np.mean(embeddings, axis=0)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def is_static_knowledge_query(self, query: str) -> tuple[bool, float]:
        query_embedding = self.model.encode(query)
        similarity = self._cosine_similarity(query_embedding, self._static_centroid)
        return similarity >= self.thresholds.static_knowledge, similarity

    def is_memory_reference_query(self, query: str) -> tuple[bool, float]:
        query_embedding = self.model.encode(query)
        similarity = self._cosine_similarity(query_embedding, self._memory_centroid)
        return similarity >= self.thresholds.memory_reference, similarity

    def is_duplicate_search(self, query: str, prior_queries: list[str]) -> tuple[bool, float]:
        if not prior_queries:
            return False, 0.0

        query_embedding = self.model.encode(query)
        prior_embeddings = self.model.encode(prior_queries)

        similarities = [self._cosine_similarity(query_embedding, pe) for pe in prior_embeddings]
        best_score = max(similarities)

        return best_score >= self.thresholds.duplicate_search, best_score
```

### 🚫 GATE 2.2: Semantic Classifier Check

```bash
uv run python -c "
from src.validator.semantic import SemanticClassifier

c = SemanticClassifier()

# Static knowledge should classify as static
is_static, score = c.is_static_knowledge_query('What is recursion?')
assert is_static, f'Expected static=True, got {is_static} (score={score:.3f})'
assert score > 0.5, f'Expected score > 0.5, got {score:.3f}'

# Current events should NOT classify as static
is_static2, score2 = c.is_static_knowledge_query('What is the stock price of Apple today?')
assert not is_static2, f'Expected static=False, got {is_static2} (score={score2:.3f})'

print('SEMANTIC_OK')
"
```

**Expected output**: `SEMANTIC_OK`

---

## PHASE 3: Convergence Control

### Step 3.1: Convergence Controller

**Create**: `src/validator/convergence.py`

```python
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
```

### 🚫 GATE 3.1: Convergence Control Check

```bash
uv run python -c "
from src.validator.convergence import ConvergenceState, FeedbackLevel

state = ConvergenceState(max_rejections_per_rule=2, max_total_rejections=5)

# First rejection - gentle
level1 = state.record_rejection('F1')
assert level1 == FeedbackLevel.GENTLE, f'Expected GENTLE, got {level1}'
assert not state.should_force_direct_answer()

# Second rejection - direct
level2 = state.record_rejection('F1')
assert level2 == FeedbackLevel.DIRECT, f'Expected DIRECT, got {level2}'
assert not state.should_force_direct_answer()

# Third rejection - forceful AND triggers termination
level3 = state.record_rejection('F1')
assert level3 == FeedbackLevel.FORCEFUL, f'Expected FORCEFUL, got {level3}'
assert state.should_force_direct_answer(), 'Should force direct answer after 3 violations'

print('CONVERGENCE_OK')
"
```

**Expected output**: `CONVERGENCE_OK`

---

## PHASE 4: Validation Rules

### Step 4.1: Rule Logic

**Create**: `src/validator/rules.py`

```python
"""Validation rules for tool calls."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import re

from .semantic import SemanticClassifier


class Decision(Enum):
    ALLOW = "allow"
    DENY = "deny"


@dataclass
class ValidationResult:
    decision: Decision
    rule_id: Optional[str]
    reason: str


@dataclass
class ValidationContext:
    """Mutable context tracking state across tool calls in a session."""
    user_query: str
    tool_history: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    known_paths: set[str] = field(default_factory=set)

    def add_known_paths(self, paths: list[str]) -> None:
        """Add paths discovered via Glob/LS."""
        self.known_paths.update(paths)


class RuleValidator:
    """Validates tool calls against rules."""

    def __init__(self, semantic: Optional[SemanticClassifier] = None):
        self.semantic = semantic or SemanticClassifier()

    def validate(
        self,
        tool_name: str,
        tool_input: dict,
        ctx: ValidationContext,
    ) -> ValidationResult:
        """Run all rules. First denial wins."""

        rules = [
            self._rule_f15_binary_file,
            self._rule_f1_static_knowledge,
            self._rule_f4_memory_vs_web,
            self._rule_f8_missing_location,
            self._rule_f10_duplicate_search,
            self._rule_f13_hallucinated_path,
        ]

        for rule in rules:
            result = rule(tool_name, tool_input, ctx)
            if result is not None:
                return result

        return ValidationResult(
            decision=Decision.ALLOW,
            rule_id=None,
            reason="All rules passed"
        )

    def _rule_f1_static_knowledge(
        self, tool_name: str, tool_input: dict, ctx: ValidationContext
    ) -> Optional[ValidationResult]:
        """Block web search for static knowledge queries."""
        if tool_name not in ("WebSearch", "web_search"):
            return None

        is_static, score = self.semantic.is_static_knowledge_query(ctx.user_query)

        if is_static:
            return ValidationResult(
                decision=Decision.DENY,
                rule_id="F1",
                reason=f"Static knowledge query (score: {score:.2f}). Answer directly."
            )
        return None

    def _rule_f4_memory_vs_web(
        self, tool_name: str, tool_input: dict, ctx: ValidationContext
    ) -> Optional[ValidationResult]:
        """Block web search when memory search is appropriate."""
        if tool_name not in ("WebSearch", "web_search"):
            return None

        is_memory, score = self.semantic.is_memory_reference_query(ctx.user_query)

        if is_memory:
            return ValidationResult(
                decision=Decision.DENY,
                rule_id="F4",
                reason=f"Memory reference (score: {score:.2f}). Use memory search instead."
            )
        return None

    def _rule_f8_missing_location(
        self, tool_name: str, tool_input: dict, ctx: ValidationContext
    ) -> Optional[ValidationResult]:
        """Block location-dependent search without getting location first."""
        if tool_name not in ("WebSearch", "web_search"):
            return None

        query = tool_input.get("query", "").lower()
        user_query = ctx.user_query.lower()

        location_keywords = ["near me", "nearby", "closest", "local", "in my area"]

        if any(kw in user_query or kw in query for kw in location_keywords):
            # Check if we've already gotten location
            if not any("location" in h.lower() for h in ctx.tool_history):
                return ValidationResult(
                    decision=Decision.DENY,
                    rule_id="F8",
                    reason="Location-dependent query. Get user location first."
                )
        return None

    def _rule_f10_duplicate_search(
        self, tool_name: str, tool_input: dict, ctx: ValidationContext
    ) -> Optional[ValidationResult]:
        """Block duplicate searches."""
        if tool_name not in ("WebSearch", "web_search"):
            return None

        query = tool_input.get("query", "")

        is_dup, score = self.semantic.is_duplicate_search(query, ctx.search_queries)
        if is_dup:
            return ValidationResult(
                decision=Decision.DENY,
                rule_id="F10",
                reason=f"Duplicate search (similarity: {score:.2f}). Use previous results."
            )
        return None

    def _rule_f13_hallucinated_path(
        self, tool_name: str, tool_input: dict, ctx: ValidationContext
    ) -> Optional[ValidationResult]:
        """Block access to paths that look invented."""
        if tool_name not in ("Read", "View", "view_file", "read_file"):
            return None

        path = tool_input.get("file_path", "") or tool_input.get("path", "")

        # Allow if path was mentioned by user or discovered
        if path in ctx.known_paths:
            return None
        if path in ctx.user_query:
            return None

        # Allow common/reasonable paths
        common_patterns = [
            r"^\.?/?README",
            r"^\.?/?package\.json$",
            r"^\.?/?pyproject\.toml$",
            r"^\.?/?Cargo\.toml$",
            r"^\.?/?\.env",
        ]
        for pattern in common_patterns:
            if re.search(pattern, path, re.IGNORECASE):
                return None

        # Flag suspicious generic paths
        suspicious = [
            (r"^/project/", "Generic /project/ path"),
            (r"^/app/", "Generic /app/ path"),
            (r"/src/main\.(py|js|ts)$", "Generic main file"),
            (r"^/home/user/", "Generic /home/user/ path"),
            (r"^/var/www/", "Generic /var/www/ path"),
        ]

        for pattern, desc in suspicious:
            if re.search(pattern, path):
                return ValidationResult(
                    decision=Decision.DENY,
                    rule_id="F13",
                    reason=f"Hallucinated path ({desc}). List directory first."
                )

        # If path is absolute and not in known_paths, block it
        if path.startswith("/") and len(ctx.known_paths) > 0:
            return ValidationResult(
                decision=Decision.DENY,
                rule_id="F13",
                reason="Unknown absolute path. List directory first to discover valid paths."
            )

        return None

    def _rule_f15_binary_file(
        self, tool_name: str, tool_input: dict, ctx: ValidationContext
    ) -> Optional[ValidationResult]:
        """Block reading binary files."""
        if tool_name not in ("Read", "View", "view_file", "read_file"):
            return None

        path = (tool_input.get("file_path", "") or tool_input.get("path", "")).lower()

        binary_patterns = [
            r"\.(exe|dll|so|dylib|bin|o|a)$",
            r"\.(png|jpg|jpeg|gif|bmp|ico|webp|svg)$",
            r"\.(pdf|doc|docx|xls|xlsx|ppt|pptx)$",
            r"\.(zip|tar|gz|7z|rar|bz2)$",
            r"\.(mp3|mp4|wav|avi|mov|mkv|flac)$",
            r"\.(pyc|pyo|class|wasm)$",
            r"^/usr/bin/",
            r"^/bin/",
            r"^/sbin/",
        ]

        for pattern in binary_patterns:
            if re.search(pattern, path, re.IGNORECASE):
                return ValidationResult(
                    decision=Decision.DENY,
                    rule_id="F15",
                    reason=f"Binary file cannot be read as text."
                )
        return None
```

### 🚫 GATE 4.1: Rules Check

```bash
uv run python -c "
from src.validator.rules import RuleValidator, ValidationContext, Decision

v = RuleValidator()

# F1: Static knowledge
ctx = ValidationContext(user_query='What is the capital of France?')
r = v.validate('WebSearch', {'query': 'capital of France'}, ctx)
assert r.rule_id == 'F1' and r.decision == Decision.DENY, f'F1 failed: {r}'

# F4: Memory reference
ctx = ValidationContext(user_query='What did we discuss yesterday?')
r = v.validate('WebSearch', {'query': 'yesterday discussion'}, ctx)
assert r.rule_id == 'F4' and r.decision == Decision.DENY, f'F4 failed: {r}'

# F8: Location without getting location first
ctx = ValidationContext(user_query='Find restaurants near me')
r = v.validate('WebSearch', {'query': 'restaurants near me'}, ctx)
assert r.rule_id == 'F8' and r.decision == Decision.DENY, f'F8 failed: {r}'

# F10: Duplicate search
ctx = ValidationContext(user_query='test', search_queries=['Apple stock price today'])
r = v.validate('WebSearch', {'query': 'Apple stock price current'}, ctx)
assert r.rule_id == 'F10' and r.decision == Decision.DENY, f'F10 failed: {r}'

# F13: Hallucinated path (generic suspicious path)
ctx = ValidationContext(user_query='Read the config', known_paths={'/home/claude/readme.txt'})
r = v.validate('Read', {'path': '/project/src/main.py'}, ctx)
assert r.rule_id == 'F13' and r.decision == Decision.DENY, f'F13 failed: {r}'

# F15: Binary file
ctx = ValidationContext(user_query='Show me /usr/bin/python')
r = v.validate('Read', {'file_path': '/usr/bin/python'}, ctx)
assert r.rule_id == 'F15' and r.decision == Decision.DENY, f'F15 failed: {r}'

# Valid search (current events)
ctx = ValidationContext(user_query='What is the stock price of Apple today?')
r = v.validate('WebSearch', {'query': 'Apple stock price'}, ctx)
assert r.decision == Decision.ALLOW, f'Valid search blocked: {r}'

print('RULES_OK')
"
```

**Expected output**: `RULES_OK`

---

## PHASE 5: SDK Hooks

### Step 5.1: Hook Implementation

**Create**: `src/validator/hooks.py`

```python
"""
PreToolUse and PostToolUse hooks for the Claude Agent SDK.

Uses ClaudeSDKClient (not bare query()) for hook support.
"""

from typing import Any
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
    def create(cls, user_query: str) -> "HookState":
        return cls(
            validator=RuleValidator(),
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
```

### 🚫 GATE 5.1: Hooks Check

```bash
uv run python -c "
import asyncio
from src.validator.hooks import HookState, create_pre_tool_use_hook

state = HookState.create('What is the capital of France?')
hook = create_pre_tool_use_hook(state)

# Should block WebSearch for static knowledge
result = asyncio.run(hook(
    {'tool_name': 'WebSearch', 'tool_input': {'query': 'capital France'}},
    'test-id',
    None
))

assert 'hookSpecificOutput' in result, f'Expected denial, got {result}'
assert result['hookSpecificOutput']['permissionDecision'] == 'deny'
assert len(state.rejections) == 1
assert state.rejections[0]['rule_id'] == 'F1'
assert state.rejections[0]['feedback_level'] == 'GENTLE'

# Second rejection should escalate
result2 = asyncio.run(hook(
    {'tool_name': 'WebSearch', 'tool_input': {'query': 'France capital city'}},
    'test-id-2',
    None
))
assert state.rejections[1]['feedback_level'] == 'DIRECT'

print('HOOKS_OK')
"
```

**Expected output**: `HOOKS_OK`

---

## PHASE 6: Scenarios

### Step 6.1: Scenario Generator

**Create**: `scenarios/generator.py`

```python
"""Generate test scenarios for the validator experiment."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class Scenario:
    id: str
    query: str
    category: str
    expected_behavior: str  # "no_tools", "web_search", "file_read", etc.
    failure_mode: Optional[str]  # F1, F4, F8, F10, F13, F15 or None
    validator_should_block: bool
    notes: str = ""


# Scenarios where the validator SHOULD block
SHOULD_BLOCK_SCENARIOS = [
    # F1: Static knowledge - should not search
    Scenario("f1_001", "What is the capital of France?", "static", "no_tools", "F1", True),
    Scenario("f1_002", "Explain recursion in programming", "static", "no_tools", "F1", True),
    Scenario("f1_003", "What is photosynthesis?", "static", "no_tools", "F1", True),
    Scenario("f1_004", "Who wrote Romeo and Juliet?", "static", "no_tools", "F1", True),
    Scenario("f1_005", "What is the Pythagorean theorem?", "static", "no_tools", "F1", True),
    Scenario("f1_006", "Define machine learning", "static", "no_tools", "F1", True),
    Scenario("f1_007", "How does gravity work?", "static", "no_tools", "F1", True),
    Scenario("f1_008", "What is a linked list?", "static", "no_tools", "F1", True),

    # F4: Memory reference - should use memory, not web
    Scenario("f4_001", "What did we discuss yesterday?", "memory", "memory_search", "F4", True),
    Scenario("f4_002", "Remember when we talked about the project?", "memory", "memory_search", "F4", True),
    Scenario("f4_003", "What was my budget again?", "memory", "memory_search", "F4", True),
    Scenario("f4_004", "You mentioned something about deadlines earlier", "memory", "memory_search", "F4", True),

    # F8: Location-dependent without location
    Scenario("f8_001", "Find restaurants near me", "location", "get_location_then_search", "F8", True),
    Scenario("f8_002", "What coffee shops are nearby?", "location", "get_location_then_search", "F8", True),
    Scenario("f8_003", "Show me local events", "location", "get_location_then_search", "F8", True),

    # F10: Duplicate search - model should use previous results
    Scenario("f10_001", "Search for Python tutorials again", "duplicate", "use_previous", "F10", True,
             notes="Requires prior search for 'Python tutorials' in context"),
    Scenario("f10_002", "Look up that same stock price", "duplicate", "use_previous", "F10", True,
             notes="Requires prior search for stock price in context"),
    Scenario("f10_003", "Search for React documentation one more time", "duplicate", "use_previous", "F10", True,
             notes="Requires prior search for 'React documentation' in context"),

    # F13: Hallucinated paths - model invents file paths
    Scenario("f13_001", "Read /project/src/main.py", "hallucinated_path", "list_first", "F13", True,
             notes="Generic path not mentioned by user or discovered"),
    Scenario("f13_002", "Show me /app/config/settings.json", "hallucinated_path", "list_first", "F13", True,
             notes="Generic path not mentioned by user or discovered"),
    Scenario("f13_003", "Open /home/user/documents/report.txt", "hallucinated_path", "list_first", "F13", True,
             notes="Generic path not mentioned by user or discovered"),

    # F15: Binary files
    Scenario("f15_001", "Read /usr/bin/python", "binary", "no_tools", "F15", True),
    Scenario("f15_002", "Show me the contents of /bin/bash", "binary", "no_tools", "F15", True),
    Scenario("f15_003", "What's in image.png?", "binary", "no_tools", "F15", True),
    Scenario("f15_004", "Read the compiled output.pyc file", "binary", "no_tools", "F15", True),
]

# Scenarios where the validator should NOT block (valid tool use)
SHOULD_ALLOW_SCENARIOS = [
    # Current events - should search
    Scenario("valid_001", "What is the current stock price of Apple?", "current", "web_search", None, False),
    Scenario("valid_002", "What are today's top news headlines?", "current", "web_search", None, False),
    Scenario("valid_003", "What is the weather in Tokyo right now?", "current", "web_search", None, False),
    Scenario("valid_004", "Who won the game last night?", "current", "web_search", None, False),
    Scenario("valid_005", "What is Bitcoin trading at?", "current", "web_search", None, False),
    Scenario("valid_006", "Latest developments in AI regulation", "current", "web_search", None, False),

    # File operations with user-provided paths
    Scenario("valid_010", "Read the file at /home/user/myconfig.json", "file", "file_read", None, False,
             notes="Path provided by user"),
    Scenario("valid_011", "What's in README.md?", "file", "file_read", None, False,
             notes="Common file, reasonable to try"),
    Scenario("valid_012", "Show me package.json", "file", "file_read", None, False,
             notes="Common file, reasonable to try"),

    # Code generation - no tools needed
    Scenario("valid_020", "Write a Python function to sort a list", "code", "no_tools", None, False),
    Scenario("valid_021", "Create a bash script to backup files", "code", "no_tools", None, False),
    Scenario("valid_022", "Generate a SQL query to find duplicate rows", "code", "no_tools", None, False),
]

ALL_SCENARIOS = SHOULD_BLOCK_SCENARIOS + SHOULD_ALLOW_SCENARIOS


def generate_scenarios() -> list[Scenario]:
    return ALL_SCENARIOS


def save_scenarios(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    scenarios = generate_scenarios()
    with open(path, "w") as f:
        json.dump([asdict(s) for s in scenarios], f, indent=2)
    return len(scenarios)


def load_scenarios(path: Path) -> list[Scenario]:
    with open(path) as f:
        data = json.load(f)
    return [Scenario(**d) for d in data]


if __name__ == "__main__":
    path = Path("scenarios/generated/scenarios.json")
    n = save_scenarios(path)
    print(f"Generated {n} scenarios to {path}")
```

### 🚫 GATE 6.1: Scenarios Check

```bash
uv run python scenarios/generator.py
uv run python -c "
from scenarios.generator import load_scenarios
from pathlib import Path

scenarios = load_scenarios(Path('scenarios/generated/scenarios.json'))
should_block = [s for s in scenarios if s.validator_should_block]
should_allow = [s for s in scenarios if not s.validator_should_block]

# Check counts
assert len(scenarios) >= 30, f'Expected >= 30 scenarios, got {len(scenarios)}'
assert len(should_block) >= 20, f'Expected >= 20 block scenarios, got {len(should_block)}'
assert len(should_allow) >= 10, f'Expected >= 10 allow scenarios, got {len(should_allow)}'

# Check all failure modes covered
failure_modes = set(s.failure_mode for s in should_block if s.failure_mode)
required_modes = {'F1', 'F4', 'F8', 'F10', 'F13', 'F15'}
missing = required_modes - failure_modes
assert not missing, f'Missing failure modes: {missing}'

print(f'Total: {len(scenarios)}')
print(f'Should block: {len(should_block)}')
print(f'Should allow: {len(should_allow)}')
print(f'Failure modes: {sorted(failure_modes)}')
print('SCENARIOS_OK')
"
```

**Expected output**:
```
Total: 34
Should block: 22
Should allow: 12
Failure modes: ['F1', 'F10', 'F13', 'F15', 'F4', 'F8']
SCENARIOS_OK
```

---

## PHASE 7: Scoring

### Step 7.1: Score Computation

**Create**: `src/scoring.py`

```python
"""Score tool calling outcomes."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ScoreCategory(Enum):
    PERFECT = 3      # Correct tool selection and appropriate use
    GOOD = 2         # Mostly correct, minor issues
    PARTIAL = 1      # Some correct elements
    WRONG = 0        # Incorrect tool selection


@dataclass
class TrialScore:
    scenario_id: str
    approach: str  # "baseline" or "validated"
    score: float
    category: ScoreCategory
    tools_used: list[str]
    tools_attempted: list[str]  # Including blocked ones
    validator_blocked: bool
    validator_should_block: bool
    correct_without_validator: bool  # Did Claude make the right choice independently?
    notes: list[str]


def score_trial(
    scenario_id: str,
    approach: str,
    expected_behavior: str,
    tools_used: list[str],
    tools_attempted: list[str],
    validator_blocked: bool,
    validator_should_block: bool,
) -> TrialScore:
    """
    Score a trial based on expected vs actual behavior.

    Key distinction:
    - tools_used: Tools that actually executed
    - tools_attempted: Tools Claude tried to use (including blocked ones)

    expected_behavior: "no_tools", "web_search", "memory_search", "file_read", etc.
    """
    notes = []

    # Determine what Claude tried to do (before validation)
    attempted_web_search = any("WebSearch" in t or "Search" in t or "search" in t.lower()
                               for t in tools_attempted)
    attempted_file_read = any("Read" in t or "View" in t for t in tools_attempted)
    attempted_any_tool = len(tools_attempted) > 0

    # Determine what actually executed
    used_web_search = any("WebSearch" in t or "Search" in t or "search" in t.lower()
                          for t in tools_used)
    used_file_read = any("Read" in t or "View" in t for t in tools_used)
    used_any_tool = len(tools_used) > 0

    # Key insight: Did Claude make the right choice INDEPENDENTLY (without validator)?
    # This is determined by what Claude attempted, not what executed
    if expected_behavior == "no_tools":
        correct_without_validator = not attempted_any_tool
    elif expected_behavior == "web_search":
        correct_without_validator = attempted_web_search
    elif expected_behavior == "memory_search":
        correct_without_validator = any("memory" in t.lower() for t in tools_attempted)
    elif expected_behavior == "file_read":
        correct_without_validator = attempted_file_read
    else:
        correct_without_validator = False  # Unknown behavior

    # Score based on expected behavior and outcome
    if expected_behavior == "no_tools":
        if not used_any_tool:
            if correct_without_validator:
                # Claude correctly chose not to use tools
                score = 3.0
                category = ScoreCategory.PERFECT
                notes.append("Correctly answered without tools")
            elif validator_blocked:
                # Validator prevented bad tool use
                score = 2.5
                category = ScoreCategory.GOOD
                notes.append("Validator blocked inappropriate tool use")
            else:
                # No tools used but unclear why
                score = 2.0
                category = ScoreCategory.GOOD
        else:
            # Used tools when shouldn't have
            score = 0.0
            category = ScoreCategory.WRONG
            notes.append(f"Used tools when shouldn't have: {tools_used}")

    elif expected_behavior == "web_search":
        if used_web_search:
            score = 3.0
            category = ScoreCategory.PERFECT
        elif validator_blocked and not validator_should_block:
            # FALSE POSITIVE: Validator incorrectly blocked valid search
            score = 0.0
            category = ScoreCategory.WRONG
            notes.append("Validator incorrectly blocked valid search (false positive)")
        elif not used_any_tool:
            score = 1.0
            category = ScoreCategory.PARTIAL
            notes.append("Should have searched but didn't")
        else:
            score = 1.0
            category = ScoreCategory.PARTIAL
            notes.append(f"Used wrong tool: {tools_used}")

    elif expected_behavior == "memory_search":
        used_memory = any("Memory" in t or "memory" in t.lower() or "conversation" in t.lower()
                          for t in tools_used)
        if used_memory:
            score = 3.0
            category = ScoreCategory.PERFECT
        elif not used_any_tool:
            # Acceptable - might answer from context
            score = 2.0
            category = ScoreCategory.GOOD
            notes.append("No memory search but may have answered from context")
        elif used_web_search and validator_blocked:
            # Validator caught web search for memory query
            score = 2.0
            category = ScoreCategory.GOOD
            notes.append("Validator blocked inappropriate web search")
        else:
            score = 1.0
            category = ScoreCategory.PARTIAL

    elif expected_behavior == "file_read":
        if used_file_read:
            score = 3.0
            category = ScoreCategory.PERFECT
        elif not used_any_tool:
            score = 1.0
            category = ScoreCategory.PARTIAL
            notes.append("Should have read file")
        else:
            score = 1.0
            category = ScoreCategory.PARTIAL

    elif expected_behavior == "get_location_then_search":
        used_location = any("Location" in t or "location" in t.lower() for t in tools_used)
        if used_location and used_web_search:
            score = 3.0
            category = ScoreCategory.PERFECT
        elif validator_blocked and validator_should_block:
            # Blocked search without location - correct
            score = 2.0
            category = ScoreCategory.GOOD
            notes.append("Validator blocked search without location")
        elif used_web_search and not used_location:
            score = 0.5
            category = ScoreCategory.WRONG
            notes.append("Searched without getting location first")
        else:
            score = 1.0
            category = ScoreCategory.PARTIAL

    elif expected_behavior == "use_previous":
        # For duplicate search scenarios
        if not used_web_search:
            score = 3.0
            category = ScoreCategory.PERFECT
            notes.append("Correctly used previous results")
        elif validator_blocked:
            score = 2.5
            category = ScoreCategory.GOOD
            notes.append("Validator blocked duplicate search")
        else:
            score = 0.0
            category = ScoreCategory.WRONG
            notes.append("Made duplicate search")

    elif expected_behavior == "list_first":
        # For hallucinated path scenarios
        used_list = any("Glob" in t or "LS" in t or "list" in t.lower() for t in tools_used)
        if used_list:
            score = 3.0
            category = ScoreCategory.PERFECT
        elif validator_blocked:
            score = 2.5
            category = ScoreCategory.GOOD
            notes.append("Validator blocked hallucinated path")
        elif not used_any_tool:
            score = 1.5
            category = ScoreCategory.PARTIAL
            notes.append("Should have listed directory first")
        else:
            score = 0.0
            category = ScoreCategory.WRONG
            notes.append("Attempted to read unknown path")

    else:
        # Unknown expected behavior
        score = 1.5
        category = ScoreCategory.PARTIAL
        notes.append(f"Unknown expected behavior: {expected_behavior}")

    return TrialScore(
        scenario_id=scenario_id,
        approach=approach,
        score=score,
        category=category,
        tools_used=tools_used,
        tools_attempted=tools_attempted,
        validator_blocked=validator_blocked,
        validator_should_block=validator_should_block,
        correct_without_validator=correct_without_validator,
        notes=notes,
    )
```

### 🚫 GATE 7.1: Scoring Check

```bash
uv run python -c "
from src.scoring import score_trial, ScoreCategory

# Test 1: Correct no-tool answer
score = score_trial(
    scenario_id='f1_001',
    approach='validated',
    expected_behavior='no_tools',
    tools_used=[],
    tools_attempted=[],
    validator_blocked=False,
    validator_should_block=True,
)
assert score.score == 3.0, f'Expected 3.0, got {score.score}'
assert score.correct_without_validator == True

# Test 2: Validator blocked bad tool use
score = score_trial(
    scenario_id='f1_001',
    approach='validated',
    expected_behavior='no_tools',
    tools_used=[],
    tools_attempted=['WebSearch'],
    validator_blocked=True,
    validator_should_block=True,
)
assert score.score == 2.5, f'Expected 2.5, got {score.score}'
assert score.correct_without_validator == False

# Test 3: False positive (validator blocked valid search)
score = score_trial(
    scenario_id='valid_001',
    approach='validated',
    expected_behavior='web_search',
    tools_used=[],
    tools_attempted=['WebSearch'],
    validator_blocked=True,
    validator_should_block=False,
)
assert score.score == 0.0, f'Expected 0.0 for false positive, got {score.score}'
assert 'false positive' in score.notes[0].lower()

print('SCORING_OK')
"
```

**Expected output**: `SCORING_OK`

---

## PHASE 8: Experiment Runner

### Step 8.1: Main Runner

**Create**: `experiments/runner.py`

```python
"""
Run the validator experiment using Claude Agent SDK.

CRITICAL: Uses ClaudeSDKClient (not bare query()) for hook support.

Compares:
- Baseline: No validation hooks (Claude decides freely)
- Validated: PreToolUse/PostToolUse hooks block bad tool calls
"""

import asyncio
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    HookMatcher,
    AssistantMessage,
    TextBlock,
    ToolUseBlock,
    ResultMessage,
)

from src.validator.hooks import (
    HookState,
    create_pre_tool_use_hook,
    create_post_tool_use_hook,
    create_logging_hooks,
)
from src.scoring import score_trial, TrialScore
from scenarios.generator import load_scenarios, Scenario


# Number of trials per scenario for statistical validity
N_TRIALS_PER_SCENARIO = 5


@dataclass
class TrialResult:
    scenario_id: str
    trial_number: int
    approach: str
    query: str
    tools_used: list[str]
    tools_attempted: list[str]
    response_summary: str
    validator_rejections: list[dict]
    score: TrialScore
    error: Optional[str] = None


async def run_baseline_trial(scenario: Scenario, trial_num: int) -> TrialResult:
    """Run a trial without validation hooks (logging only)."""
    log_pre, log_post, calls = create_logging_hooks()

    # Set up prior context for F10 scenarios
    prior_searches = []
    if scenario.failure_mode == "F10":
        prior_searches = _get_prior_searches_for_scenario(scenario)

    options = ClaudeAgentOptions(
        allowed_tools=["WebSearch", "Read", "Glob", "Bash", "LS"],
        max_turns=3,
        permission_mode="acceptEdits",
        hooks={
            "PreToolUse": [
                HookMatcher(matcher="*", hooks=[log_pre])
            ],
            "PostToolUse": [
                HookMatcher(matcher="*", hooks=[log_post])
            ],
        }
    )

    response_text = ""
    error = None

    try:
        async with ClaudeSDKClient(options=options) as client:
            # Add context for duplicate search scenarios
            if prior_searches:
                context_prompt = f"Previous search results for '{prior_searches[0]}': [some results]. Now: {scenario.query}"
            else:
                context_prompt = scenario.query

            await client.query(context_prompt)

            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_text += block.text[:200]
                elif isinstance(message, ResultMessage):
                    response_text = str(message)[:500]

    except Exception as e:
        error = str(e)

    # All tools attempted = all tools used (no validator)
    tools_used = [c["tool_name"] for c in calls if c["event"] == "pre"]

    score = score_trial(
        scenario_id=scenario.id,
        approach="baseline",
        expected_behavior=scenario.expected_behavior,
        tools_used=tools_used,
        tools_attempted=tools_used,
        validator_blocked=False,
        validator_should_block=scenario.validator_should_block,
    )

    return TrialResult(
        scenario_id=scenario.id,
        trial_number=trial_num,
        approach="baseline",
        query=scenario.query,
        tools_used=tools_used,
        tools_attempted=tools_used,
        response_summary=response_text[:200],
        validator_rejections=[],
        score=score,
        error=error,
    )


async def run_validated_trial(scenario: Scenario, trial_num: int) -> TrialResult:
    """Run a trial with validation hooks."""
    state = HookState.create(scenario.query)
    pre_hook = create_pre_tool_use_hook(state)
    post_hook = create_post_tool_use_hook(state)

    # Set up prior context for F10 scenarios
    if scenario.failure_mode == "F10":
        prior_searches = _get_prior_searches_for_scenario(scenario)
        state.context.search_queries.extend(prior_searches)

    # Set up known paths for F13 scenarios (simulate having listed a directory)
    if scenario.failure_mode == "F13":
        state.context.add_known_paths(["/home/claude/actual_file.txt"])

    options = ClaudeAgentOptions(
        allowed_tools=["WebSearch", "Read", "Glob", "Bash", "LS"],
        max_turns=5,  # More turns to allow for retry after rejection
        permission_mode="acceptEdits",
        hooks={
            "PreToolUse": [
                HookMatcher(matcher="*", hooks=[pre_hook])
            ],
            "PostToolUse": [
                HookMatcher(matcher="*", hooks=[post_hook])
            ],
        }
    )

    response_text = ""
    error = None

    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(scenario.query)

            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_text += block.text[:200]
                elif isinstance(message, ResultMessage):
                    response_text = str(message)[:500]

    except Exception as e:
        error = str(e)

    # Tools that actually executed
    tools_used = [h.split(":")[0] for h in state.context.tool_history]

    # Tools that were attempted (including blocked)
    tools_attempted = tools_used + [r["tool_name"] for r in state.rejections]

    validator_blocked = len(state.rejections) > 0

    score = score_trial(
        scenario_id=scenario.id,
        approach="validated",
        expected_behavior=scenario.expected_behavior,
        tools_used=tools_used,
        tools_attempted=tools_attempted,
        validator_blocked=validator_blocked,
        validator_should_block=scenario.validator_should_block,
    )

    return TrialResult(
        scenario_id=scenario.id,
        trial_number=trial_num,
        approach="validated",
        query=scenario.query,
        tools_used=tools_used,
        tools_attempted=tools_attempted,
        response_summary=response_text[:200],
        validator_rejections=state.rejections,
        score=score,
        error=error,
    )


def _get_prior_searches_for_scenario(scenario: Scenario) -> list[str]:
    """Get prior search queries for duplicate search scenarios."""
    if "Python tutorials" in scenario.query:
        return ["Python tutorials"]
    if "stock price" in scenario.query:
        return ["AAPL stock price"]
    if "React documentation" in scenario.query:
        return ["React documentation"]
    return []


async def run_experiment(
    scenarios: list[Scenario],
    output_dir: Path,
    n_trials: int = N_TRIALS_PER_SCENARIO,
    delay_between_trials: float = 1.0,
) -> list[TrialResult]:
    """Run all scenarios in both baseline and validated modes."""

    results = []
    total_trials = len(scenarios) * n_trials * 2
    current = 0

    for scenario in scenarios:
        for trial_num in range(n_trials):
            current += 1
            print(f"\n[{current}/{total_trials}] Baseline: {scenario.id} (trial {trial_num + 1})")
            baseline_result = await run_baseline_trial(scenario, trial_num)
            results.append(baseline_result)
            print(f"  Tools: {baseline_result.tools_used}, Score: {baseline_result.score.score}")

            await asyncio.sleep(delay_between_trials)

            current += 1
            print(f"[{current}/{total_trials}] Validated: {scenario.id} (trial {trial_num + 1})")
            validated_result = await run_validated_trial(scenario, trial_num)
            results.append(validated_result)
            print(f"  Tools: {validated_result.tools_used}, Score: {validated_result.score.score}")
            print(f"  Rejections: {len(validated_result.validator_rejections)}")

            await asyncio.sleep(delay_between_trials)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    results_data = []
    for r in results:
        d = asdict(r)
        d["score"] = asdict(r.score)
        d["score"]["category"] = r.score.category.name
        results_data.append(d)

    output_path = output_dir / f"results_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return results


def summarize_results(results: list[TrialResult]):
    """Print summary statistics."""
    baseline = [r for r in results if r.approach == "baseline"]
    validated = [r for r in results if r.approach == "validated"]

    baseline_scores = [r.score.score for r in baseline]
    validated_scores = [r.score.score for r in validated]

    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)

    print(f"\nTrials per approach: {len(baseline)}")

    print(f"\nBaseline:")
    print(f"  Mean score: {sum(baseline_scores)/len(baseline_scores):.3f}")

    print(f"\nValidated:")
    print(f"  Mean score: {sum(validated_scores)/len(validated_scores):.3f}")

    # Validator metrics
    should_block = [r for r in validated if r.score.validator_should_block]
    actually_blocked = [r for r in should_block if r.validator_rejections]

    if should_block:
        catch_rate = len(actually_blocked) / len(should_block)
        print(f"  Catch rate: {catch_rate:.1%} ({len(actually_blocked)}/{len(should_block)})")

    should_allow = [r for r in validated if not r.score.validator_should_block]
    incorrectly_blocked = [r for r in should_allow if r.validator_rejections]

    if should_allow:
        false_positive_rate = len(incorrectly_blocked) / len(should_allow)
        print(f"  False positive rate: {false_positive_rate:.1%} ({len(incorrectly_blocked)}/{len(should_allow)})")

    # Independent correct rate (did Claude get it right without validator help?)
    correct_without = [r for r in validated if r.score.correct_without_validator]
    independent_rate = len(correct_without) / len(validated) if validated else 0
    print(f"  Independent correct rate: {independent_rate:.1%}")

    # Score improvement
    improvement = sum(validated_scores)/len(validated_scores) - sum(baseline_scores)/len(baseline_scores)
    print(f"\nScore improvement: {improvement:+.3f}")


async def main():
    scenarios_path = Path("scenarios/generated/scenarios.json")

    if not scenarios_path.exists():
        print("Generating scenarios...")
        from scenarios.generator import save_scenarios
        save_scenarios(scenarios_path)

    scenarios = load_scenarios(scenarios_path)
    print(f"Loaded {len(scenarios)} scenarios")
    print(f"Running {N_TRIALS_PER_SCENARIO} trials per scenario")

    output_dir = Path("experiments/results")

    results = await run_experiment(scenarios, output_dir)
    summarize_results(results)


if __name__ == "__main__":
    asyncio.run(main())
```

### 🚫 GATE 8.1: Runner Structure Check

```bash
uv run python -c "
from experiments.runner import (
    run_baseline_trial,
    run_validated_trial,
    run_experiment,
    summarize_results,
    N_TRIALS_PER_SCENARIO,
)
from scenarios.generator import load_scenarios
from pathlib import Path

# Verify imports work
assert N_TRIALS_PER_SCENARIO >= 3, f'Need at least 3 trials, got {N_TRIALS_PER_SCENARIO}'

scenarios = load_scenarios(Path('scenarios/generated/scenarios.json'))
print(f'Ready to run {len(scenarios)} scenarios x {N_TRIALS_PER_SCENARIO} trials = {len(scenarios) * N_TRIALS_PER_SCENARIO * 2} total trials')
print('RUNNER_STRUCTURE_OK')
"
```

**Expected output**: `RUNNER_STRUCTURE_OK`

---

## PHASE 9: Statistical Analysis

### Step 9.1: Comparison Analysis

**Create**: `analysis/compare.py`

```python
"""Statistical comparison of baseline vs validated results."""

import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from scipy import stats


@dataclass
class ComparisonResult:
    n_pairs: int
    n_trials_per_scenario: int
    baseline_mean: float
    validated_mean: float
    improvement: float
    t_statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    significant: bool
    catch_rate: float
    false_positive_rate: float
    independent_correct_rate: float


def load_results(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def analyze_results(results: list[dict], alpha: float = 0.05) -> ComparisonResult:
    """Perform paired comparison of baseline vs validated."""

    # Group by scenario and trial
    by_scenario_trial = {}
    for r in results:
        key = (r["scenario_id"], r["trial_number"])
        if key not in by_scenario_trial:
            by_scenario_trial[key] = {}
        by_scenario_trial[key][r["approach"]] = r

    # Extract paired scores
    baseline_scores = []
    validated_scores = []

    for key, approaches in by_scenario_trial.items():
        if "baseline" in approaches and "validated" in approaches:
            baseline_scores.append(approaches["baseline"]["score"]["score"])
            validated_scores.append(approaches["validated"]["score"]["score"])

    baseline_arr = np.array(baseline_scores)
    validated_arr = np.array(validated_scores)
    diff = validated_arr - baseline_arr

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(baseline_arr, validated_arr)

    # Confidence interval for difference
    se = stats.sem(diff)
    if se > 0:
        ci = stats.t.interval(1 - alpha, len(diff) - 1, loc=np.mean(diff), scale=se)
    else:
        ci = (0, 0)

    # Validator metrics
    validated_only = [r for r in results if r["approach"] == "validated"]

    should_block = [r for r in validated_only if r["score"]["validator_should_block"]]
    actually_blocked = [r for r in should_block if r["validator_rejections"]]
    catch_rate = len(actually_blocked) / len(should_block) if should_block else 0

    should_allow = [r for r in validated_only if not r["score"]["validator_should_block"]]
    incorrectly_blocked = [r for r in should_allow if r["validator_rejections"]]
    false_positive_rate = len(incorrectly_blocked) / len(should_allow) if should_allow else 0

    # Independent correct rate
    correct_without = [r for r in validated_only if r["score"]["correct_without_validator"]]
    independent_correct_rate = len(correct_without) / len(validated_only) if validated_only else 0

    # Count unique scenarios and trials
    unique_scenarios = len(set(r["scenario_id"] for r in results))
    trials_per_scenario = len(results) // (unique_scenarios * 2) if unique_scenarios > 0 else 0

    return ComparisonResult(
        n_pairs=len(baseline_scores),
        n_trials_per_scenario=trials_per_scenario,
        baseline_mean=float(np.mean(baseline_arr)),
        validated_mean=float(np.mean(validated_arr)),
        improvement=float(np.mean(diff)),
        t_statistic=float(t_stat),
        p_value=float(p_value),
        ci_lower=float(ci[0]),
        ci_upper=float(ci[1]),
        significant=p_value < alpha,
        catch_rate=catch_rate,
        false_positive_rate=false_positive_rate,
        independent_correct_rate=independent_correct_rate,
    )


def print_analysis(result: ComparisonResult):
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS: Validator Hypothesis")
    print("="*60)

    print(f"\nSample size:")
    print(f"  Paired comparisons: {result.n_pairs}")
    print(f"  Trials per scenario: {result.n_trials_per_scenario}")

    print(f"\nScores:")
    print(f"  Baseline mean:  {result.baseline_mean:.3f}")
    print(f"  Validated mean: {result.validated_mean:.3f}")
    print(f"  Improvement:    {result.improvement:+.3f}")

    print(f"\nStatistical test (paired t-test):")
    print(f"  t-statistic: {result.t_statistic:.3f}")
    print(f"  p-value:     {result.p_value:.4f}")
    print(f"  95% CI:      [{result.ci_lower:+.3f}, {result.ci_upper:+.3f}]")
    print(f"  Significant: {'Yes' if result.significant else 'No'} (α=0.05)")

    print(f"\nValidator metrics:")
    print(f"  Catch rate:            {result.catch_rate:.1%}")
    print(f"  False positive rate:   {result.false_positive_rate:.1%}")
    print(f"  Independent correct:   {result.independent_correct_rate:.1%}")

    print("\n" + "-"*60)
    print("HYPOTHESIS EVALUATION")
    print("-"*60)

    criteria = [
        ("Improvement ≥ 0.15", result.improvement >= 0.15),
        ("p-value < 0.05", result.p_value < 0.05),
        ("Catch rate ≥ 60%", result.catch_rate >= 0.60),
        ("False positive < 10%", result.false_positive_rate < 0.10),
    ]

    for criterion, passed in criteria:
        status = "✓" if passed else "✗"
        print(f"  {status} {criterion}")

    all_passed = all(passed for _, passed in criteria)
    print(f"\nConclusion: {'VALIDATED' if all_passed else 'NOT VALIDATED'}")


def main():
    results_dir = Path("experiments/results")
    result_files = sorted(results_dir.glob("results_*.json"))

    if not result_files:
        print("No results found. Run experiments first:")
        print("  uv run python -m experiments.runner")
        return

    latest = result_files[-1]
    print(f"Analyzing: {latest.name}")

    results = load_results(latest)
    analysis = analyze_results(results)
    print_analysis(analysis)


if __name__ == "__main__":
    main()
```

### 🚫 GATE 9.1: Analysis Check

```bash
uv run python -c "
from analysis.compare import analyze_results, print_analysis

# Mock data to verify analysis works
mock_results = []
for i in range(10):
    for trial in range(3):
        # Baseline: sometimes makes mistakes
        mock_results.append({
            'scenario_id': f's{i}',
            'trial_number': trial,
            'approach': 'baseline',
            'score': {'score': 2.0 if i % 3 == 0 else 3.0, 'validator_should_block': i < 5, 'correct_without_validator': i % 2 == 0},
            'validator_rejections': [],
        })
        # Validated: usually better
        mock_results.append({
            'scenario_id': f's{i}',
            'trial_number': trial,
            'approach': 'validated',
            'score': {'score': 2.5 if i % 3 == 0 else 3.0, 'validator_should_block': i < 5, 'correct_without_validator': i % 2 == 0},
            'validator_rejections': [{'rule': 'F1'}] if i < 3 else [],
        })

result = analyze_results(mock_results)
assert result.n_pairs == 30, f'Expected 30 pairs, got {result.n_pairs}'
assert result.n_trials_per_scenario == 3, f'Expected 3 trials/scenario, got {result.n_trials_per_scenario}'
assert 0 <= result.catch_rate <= 1
assert 0 <= result.false_positive_rate <= 1

print('ANALYSIS_OK')
"
```

**Expected output**: `ANALYSIS_OK`

---

## Running the Experiment

### Quick Start

```bash
# 1. Ensure all gates pass
# (Run each gate check from above)

# 2. Generate scenarios
uv run python scenarios/generator.py

# 3. Run experiment (uses your Max plan via SDK)
uv run python -m experiments.runner

# 4. Analyze results
uv run python -m analysis.compare
```

### Expected Output

```
Loaded 34 scenarios
Running 5 trials per scenario

[1/340] Baseline: f1_001 (trial 1)
  Tools: ['WebSearch'], Score: 0.0
[2/340] Validated: f1_001 (trial 1)
  Tools: [], Score: 2.5
  Rejections: 1

...

==================================================
EXPERIMENT SUMMARY
==================================================

Trials per approach: 170

Baseline:
  Mean score: 1.850

Validated:
  Mean score: 2.420
  Catch rate: 73.3% (88/120)
  False positive rate: 4.0% (2/50)
  Independent correct rate: 45.3%

Score improvement: +0.570

============================================================
STATISTICAL ANALYSIS: Validator Hypothesis
============================================================

Sample size:
  Paired comparisons: 170
  Trials per scenario: 5

Scores:
  Baseline mean:  1.850
  Validated mean: 2.420
  Improvement:    +0.570

Statistical test (paired t-test):
  t-statistic: 8.234
  p-value:     0.0001
  95% CI:      [+0.432, +0.708]
  Significant: Yes (α=0.05)

Validator metrics:
  Catch rate:            73.3%
  False positive rate:   4.0%
  Independent correct:   45.3%

------------------------------------------------------------
HYPOTHESIS EVALUATION
------------------------------------------------------------
  ✓ Improvement ≥ 0.15
  ✓ p-value < 0.05
  ✓ Catch rate ≥ 60%
  ✓ False positive < 10%

Conclusion: VALIDATED
```

### Success Criteria

| Metric | Threshold |
|--------|-----------|
| Mean improvement | ≥ 0.15 |
| p-value | < 0.05 |
| Catch rate | ≥ 60% |
| False positive rate | < 10% |

---

## Troubleshooting

### "CLINotFoundError"

Claude Code CLI not installed or not in PATH.

```bash
# Install CLI
curl -fsSL https://claude.ai/install.sh | bash

# Or specify path in code
options = ClaudeAgentOptions(cli_path="/path/to/claude")
```

### "Authentication failed"

Run `claude login` and authenticate with your Max plan.

### Rate limiting

Add longer delays between trials:

```python
results = await run_experiment(scenarios, output_dir, delay_between_trials=2.0)
```

### Hook not firing

Verify you're using `ClaudeSDKClient`, not bare `query()`. Hooks only work with the client.

---

## Files Summary

```
tool-validator-sdk/
├── pyproject.toml              # Dependencies including claude-agent-sdk
├── src/
│   ├── validator/
│   │   ├── exemplars.py        # Static knowledge / memory exemplars
│   │   ├── semantic.py         # Sentence transformer classifier
│   │   ├── rules.py            # F1, F4, F8, F10, F13, F15 rules
│   │   ├── convergence.py      # Escalating feedback, termination control
│   │   └── hooks.py            # PreToolUse + PostToolUse hooks
│   └── scoring.py              # Trial scoring with correct_without_validator
├── scenarios/
│   ├── generator.py            # 34 scenarios covering all failure modes
│   └── generated/
│       └── scenarios.json
├── experiments/
│   ├── runner.py               # ClaudeSDKClient-based runner, 5 trials/scenario
│   └── results/
│       └── results_*.json
└── analysis/
    └── compare.py              # Paired t-test with proper sample size
```

---

## Changes from v7

| Issue | v7 | v8 |
|-------|----|----|
| SDK API | Bare `query()` | `ClaudeSDKClient` context manager |
| F10/F13 scenarios | Missing | Added 6 scenarios |
| PostToolUse hook | Not implemented | Tracks `known_paths` |
| Scoring logic | Conflated validator/correct | Separate `correct_without_validator` |
| Convergence | None | Escalating feedback, forced termination |
| Statistical validity | n=1 per scenario | n=5 per scenario |
| Verification gates | None | 9 gates with deterministic checks |

---

**This is v8. Uses ClaudeSDKClient for hooks. All failure modes covered. Multiple trials. Verification gates.**
