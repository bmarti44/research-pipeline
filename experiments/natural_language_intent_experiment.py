"""
Experiment: Does Claude express tool intent more often in natural language
than it actually makes tool calls?

Hypothesis: Claude will express intent to save/remember information more frequently
when it can use natural language vs when it must make structured tool calls.

This tests the "decoupled tool calling" approach suggested by NLT and SLOT papers.
"""

import asyncio
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
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


# Intent detection patterns - what Claude might say naturally
MEMORY_INTENT_PATTERNS = [
    # Direct statements
    r"I should (save|remember|store|note|record)",
    r"I('ll| will) (save|remember|store|note|record)",
    r"(save|remember|store|note|record) this",
    r"worth (remembering|saving|noting|recording)",
    r"important to (remember|save|note|record)",
    r"keep (this|that) in mind",
    r"I'll keep .* in mind",
    r"note (this|that) (down|for later)",
    r"for future reference",
    r"(adding|add) (this|that) to memory",
    r"(storing|store) (this|that)",
    # Passive acknowledgment patterns (Claude without tools)
    r"I'm noting",
    r"noting (this|that)",
    r"useful (context|information) for .* future",
    r"good to know",
    r"helpful (context|to know)",
    r"I understand",
    # Conditional/hypothetical
    r"would (save|remember|store|note)",
    r"should be (saved|remembered|stored|noted)",
    r"(this|that) is (important|notable|significant)",
    r"key (information|detail|point)",
    # Task completion
    r"task (completed|done|finished)",
    r"(completed|finished|done) (the|this) task",
    r"successfully (completed|finished)",
]

# Stricter patterns - actual persistence intent
STRONG_INTENT_PATTERNS = [
    r"I('ll| will) (save|remember|store)",
    r"(save|saving) (this|that|it) to memory",
    r"let me (save|store|record)",
    r"I should (save|remember|store)",
]

STRONG_INTENT_REGEX = re.compile(
    "|".join(f"({p})" for p in STRONG_INTENT_PATTERNS),
    re.IGNORECASE
)

# Compile patterns for efficiency
INTENT_REGEX = re.compile(
    "|".join(f"({p})" for p in MEMORY_INTENT_PATTERNS),
    re.IGNORECASE
)


# Same scenarios as the memory tool experiment
SCENARIOS = [
    {
        "id": "implicit_001",
        "query": "The main entry point is src/app.py and it uses FastAPI. The database models are in src/models/.",
        "expected_intent": True,
        "prompt_level": "implicit",
    },
    {
        "id": "implicit_002",
        "query": "This is important: the authentication system uses bcrypt for password hashing and JWT for session tokens.",
        "expected_intent": True,
        "prompt_level": "implicit",
    },
    {
        "id": "implicit_003",
        "query": "I just finished refactoring the user service. All tests pass now.",
        "expected_intent": True,
        "prompt_level": "implicit",
    },
    {
        "id": "hint_001",
        "query": "For your reference, the API endpoints are defined in src/routes/.",
        "expected_intent": True,
        "prompt_level": "hint",
    },
    {
        "id": "hint_002",
        "query": "I prefer using type hints everywhere and descriptive variable names. Keep that in mind for future work.",
        "expected_intent": True,
        "prompt_level": "hint",
    },
    {
        "id": "hint_003",
        "query": "Going forward, we'll use PostgreSQL for all database operations.",
        "expected_intent": True,
        "prompt_level": "hint",
    },
    {
        "id": "request_001",
        "query": "We decided to use PostgreSQL instead of SQLite. Please remember this decision for later.",
        "expected_intent": True,
        "prompt_level": "request",
    },
    {
        "id": "control_001",
        "query": "What is 2 + 2?",
        "expected_intent": False,
        "prompt_level": "control",
    },
]


@dataclass
class IntentTrialResult:
    scenario_id: str
    condition: str  # "natural_language" or "with_tool"
    query: str
    expected_intent: bool
    # Natural language condition
    expressed_intent: bool
    intent_phrases: list
    # Tool condition (for comparison)
    actual_tool_call: bool
    # Full response
    response_text: str


async def run_natural_language_trial(scenario: dict) -> IntentTrialResult:
    """Run trial where Claude has NO tools - just responds naturally."""

    system_prompt = """You are a helpful assistant. When the user shares information,
respond naturally and think aloud about what you're learning and what might be
useful to remember for future conversations.

You do NOT have any tools available - just respond in natural language."""

    options = ClaudeAgentOptions(
        allowed_tools=[],  # NO TOOLS
        max_turns=1,
        permission_mode="acceptEdits",
        system_prompt=system_prompt,
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

    # Detect intent expressions
    matches = INTENT_REGEX.findall(response_text)
    intent_phrases = [m for group in matches for m in group if m]
    expressed_intent = len(intent_phrases) > 0

    # Also check strong intent
    strong_matches = STRONG_INTENT_REGEX.findall(response_text)
    strong_phrases = [m for group in strong_matches for m in group if m]

    return IntentTrialResult(
        scenario_id=scenario["id"],
        condition="natural_language",
        query=scenario["query"],
        expected_intent=scenario["expected_intent"],
        expressed_intent=expressed_intent,
        intent_phrases=intent_phrases,
        actual_tool_call=False,
        response_text=response_text[:500],
    )


async def run_explicit_persistence_trial(scenario: dict) -> IntentTrialResult:
    """Run trial asking Claude to IDENTIFY what should be persisted (reasoning task)."""

    system_prompt = """You are a helpful assistant. When the user shares information,
respond naturally but ALSO explicitly state what information (if any) should be
saved to a persistent memory system for future conversations.

Format your response as:
1. Your natural response to the user
2. [PERSIST]: List any information that should be saved, or "None" if nothing needs saving

You do NOT have any tools - just identify what SHOULD be saved."""

    options = ClaudeAgentOptions(
        allowed_tools=[],
        max_turns=1,
        permission_mode="acceptEdits",
        system_prompt=system_prompt,
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

    # Check for [PERSIST] section with actual content
    persist_match = re.search(r'\[PERSIST\]:?\s*(.+?)(?:\n\n|$)', response_text, re.IGNORECASE | re.DOTALL)
    has_persist_content = False
    persist_content = ""
    if persist_match:
        persist_content = persist_match.group(1).strip()
        has_persist_content = persist_content.lower() not in ["none", "nothing", "n/a", ""] and "nothing" not in persist_content.lower()

    return IntentTrialResult(
        scenario_id=scenario["id"],
        condition="identify_persist",
        query=scenario["query"],
        expected_intent=scenario["expected_intent"],
        expressed_intent=has_persist_content,
        intent_phrases=[persist_content[:100]] if persist_content else [],
        actual_tool_call=False,
        response_text=response_text[:500],
    )


async def run_tool_trial(scenario: dict) -> IntentTrialResult:
    """Run trial where Claude HAS the memory tool available."""

    tool_calls = []

    async def track_tool_calls(input_data, tool_use_id, context):
        tool_name = input_data.get("tool_name", "")
        if tool_name == "Bash":
            command = input_data.get("tool_input", {}).get("command", "")
            if "save-memory" in command:
                tool_calls.append(command)
                return {"result": "Memory saved successfully."}
        return {}

    system_prompt = """You are a helpful assistant with access to a memory tool.

## Memory Tool
To save important information for future conversations, run:
    save-memory "content" "category"

Where category is one of: 'codebase', 'task_completed', 'user_preference', 'decision', 'other'

IMPORTANT: Use this tool whenever you learn something important that should be remembered."""

    options = ClaudeAgentOptions(
        allowed_tools=["Bash"],
        max_turns=2,
        permission_mode="acceptEdits",
        system_prompt=system_prompt,
        hooks={
            "PreToolUse": [
                HookMatcher(matcher="*", hooks=[track_tool_calls])
            ],
        }
    )

    response_text = ""

    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(scenario["query"])

            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_text += block.text[:500]

    except Exception as e:
        response_text = f"Error: {e}"

    # Also check for intent in the response (even if tool was called)
    matches = INTENT_REGEX.findall(response_text)
    intent_phrases = [m for group in matches for m in group if m]

    return IntentTrialResult(
        scenario_id=scenario["id"],
        condition="with_tool",
        query=scenario["query"],
        expected_intent=scenario["expected_intent"],
        expressed_intent=len(intent_phrases) > 0,
        intent_phrases=intent_phrases,
        actual_tool_call=len(tool_calls) > 0,
        response_text=response_text[:500],
    )


async def run_experiment():
    """Compare natural language intent expression vs actual tool calls."""

    results = []

    print("=" * 70)
    print("NATURAL LANGUAGE INTENT vs TOOL CALL EXPERIMENT")
    print("Testing: Does Claude express intent more often than it calls tools?")
    print("=" * 70)

    for scenario in SCENARIOS:
        print(f"\n--- Scenario: {scenario['id']} ({scenario['prompt_level']}) ---")
        print(f"Query: {scenario['query'][:60]}...")

        # Condition 1: Natural language only (no tools)
        print("\n  [1. Natural Language - No Tools]")
        result_nl = await run_natural_language_trial(scenario)
        print(f"    Expressed intent: {result_nl.expressed_intent}")
        if result_nl.intent_phrases:
            print(f"    Phrases: {result_nl.intent_phrases[:3]}")
        results.append(result_nl)

        await asyncio.sleep(1)

        # Condition 2: Identify what should be persisted (reasoning task)
        print("\n  [2. Identify Persist - Reasoning Task]")
        result_explicit = await run_explicit_persistence_trial(scenario)
        print(f"    Identified persist: {result_explicit.expressed_intent}")
        if result_explicit.intent_phrases and result_explicit.intent_phrases[0]:
            print(f"    Content: {result_explicit.intent_phrases[0][:60]}...")
        results.append(result_explicit)

        await asyncio.sleep(1)

        # Condition 3: With tool available
        print("\n  [3. With Memory Tool]")
        result_tool = await run_tool_trial(scenario)
        print(f"    Tool called: {result_tool.actual_tool_call}")
        print(f"    Also expressed intent: {result_tool.expressed_intent}")
        results.append(result_tool)

        await asyncio.sleep(1)

    # Analysis
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    nl_results = [r for r in results if r.condition == "natural_language"]
    explicit_results = [r for r in results if r.condition == "identify_persist"]
    tool_results = [r for r in results if r.condition == "with_tool"]

    # Filter to scenarios where we expect intent
    nl_should_intent = [r for r in nl_results if r.expected_intent]
    explicit_should = [r for r in explicit_results if r.expected_intent]
    tool_should_call = [r for r in tool_results if r.expected_intent]

    nl_expressed = sum(1 for r in nl_should_intent if r.expressed_intent)
    explicit_identified = sum(1 for r in explicit_should if r.expressed_intent)
    tool_called = sum(1 for r in tool_should_call if r.actual_tool_call)
    tool_expressed = sum(1 for r in tool_should_call if r.expressed_intent)

    print(f"\n--- Three Conditions Comparison ---")
    print(f"Scenarios where memory should be triggered: {len(nl_should_intent)}")
    print(f"")
    print(f"1. Natural Language (baseline, no instruction):")
    print(f"   Expressed any intent: {nl_expressed}/{len(nl_should_intent)} ({100*nl_expressed/len(nl_should_intent):.1f}%)")
    print(f"")
    print(f"2. Identify Persist (reasoning task - 'what SHOULD be saved?'):")
    print(f"   Identified persist content: {explicit_identified}/{len(explicit_should)} ({100*explicit_identified/len(explicit_should):.1f}%)")
    print(f"")
    print(f"3. With Structured Tool Available:")
    print(f"   Actually called tool: {tool_called}/{len(tool_should_call)} ({100*tool_called/len(tool_should_call):.1f}%)")

    # The key comparison
    print(f"\n--- KEY FINDINGS ---")
    print(f"")
    print(f"Comparing 'identify what to save' (reasoning) vs 'call tool' (action):")
    print(f"")
    explicit_vs_tool = explicit_identified - tool_called
    if explicit_vs_tool > 0:
        print(f"  REASONING > ACTION by: {explicit_vs_tool} scenarios ({100*explicit_vs_tool/len(explicit_should):.1f}pp)")
        print(f"")
        print(f"  KEY INSIGHT: Claude KNOWS what should be saved more often than it ACTS.")
        print(f"  The gap represents tool-calling overhead that a two-stage system could recover.")
        print(f"")
        print(f"  Two-stage architecture potential:")
        print(f"  - Stage 1: Claude identifies what to persist (reasoning) -> {100*explicit_identified/len(explicit_should):.1f}%")
        print(f"  - Stage 2: Lightweight model converts to tool calls")
        print(f"  - vs Single-stage tool calling -> {100*tool_called/len(tool_should_call):.1f}%")
    elif explicit_vs_tool < 0:
        print(f"  ACTION > REASONING by: {-explicit_vs_tool}")
        print(f"  -> Tools help Claude act (hypothesis not supported)")
    else:
        print(f"  Perfect match between reasoning and action")

    # By prompt level
    print(f"\n--- By Prompt Level ---")
    levels = ["implicit", "hint", "request"]
    for level in levels:
        nl_level = [r for r in nl_should_intent if r.scenario_id.startswith(level[:3])]
        explicit_level = [r for r in explicit_should if r.scenario_id.startswith(level[:3])]
        tool_level = [r for r in tool_should_call if r.scenario_id.startswith(level[:3])]

        if nl_level:
            nl_rate = sum(1 for r in nl_level if r.expressed_intent) / len(nl_level)
            explicit_rate = sum(1 for r in explicit_level if r.expressed_intent) / len(explicit_level)
            tool_rate = sum(1 for r in tool_level if r.actual_tool_call) / len(tool_level)
            print(f"  {level:12}: NL {nl_rate*100:5.1f}% | Explicit {explicit_rate*100:5.1f}% | Tool {tool_rate*100:5.1f}%")

    # Save results
    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    output_path = output_dir / f"intent_vs_tool_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    asyncio.run(run_experiment())
