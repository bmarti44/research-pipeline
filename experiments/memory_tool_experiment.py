"""
Experiment: Does prompt explicitness affect proactive tool usage?

Hypothesis: Claude may be reluctant to use tools proactively, even when explicitly
instructed. This tests different levels of prompt explicitness.

The "memory tool" is simulated via Bash - Claude should call:
    save-memory "content" "category"

This tests the "missing tool call" problem - when Claude SHOULD call a tool
but doesn't.
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


# Two different instruction styles for the same functionality
INSTRUCTION_NATURAL_LANGUAGE = """
## Memory Tool

You have access to a memory tool via bash. To save important information, run:

    save-memory "content" "category"

Where:
- content: The information to remember
- category: One of 'codebase', 'task_completed', 'user_preference', 'decision', 'other'

IMPORTANT: You should use this tool whenever:
- You learn something important about the codebase
- A task is completed
- The user shares preferences
- Important decisions are made

Be proactive about saving important information.
"""

INSTRUCTION_PYTHON_SYNTAX = """
## Memory Tool

You have access to a memory function via bash:

```python
def save_memory(content: str, category: str) -> None:
    '''
    Save information to persistent memory for future conversations.

    Args:
        content: The information to remember (e.g., "User prefers tabs over spaces")
        category: One of 'codebase', 'task_completed', 'user_preference', 'decision', 'other'

    Returns:
        None

    Example:
        # Via bash:
        save-memory "The main entry point is src/main.py" "codebase"
        save-memory "Fixed the authentication bug" "task_completed"
    '''
```

To call it, run via bash: `save-memory "content" "category"`

IMPORTANT: Call this function whenever you encounter important information.
"""

# Scenarios where the memory tool SHOULD be called
# Different levels of explicitness in prompting memory tool use
MEMORY_SCENARIOS = [
    # Level 1a: Pure implicit - just information, no forward-looking language
    {
        "id": "implicit_001",
        "query": "The main entry point is src/app.py and it uses FastAPI. The database models are in src/models/.",
        "expected_memory_call": True,
        "prompt_level": "implicit",
    },
    # Level 1b: Important information but still purely declarative
    {
        "id": "implicit_002",
        "query": "This is important: the authentication system uses bcrypt for password hashing and JWT for session tokens.",
        "expected_memory_call": True,
        "prompt_level": "implicit",
    },
    # Level 1c: Information about completed work (task completed trigger)
    {
        "id": "implicit_003",
        "query": "I just finished refactoring the user service. All tests pass now.",
        "expected_memory_call": True,
        "prompt_level": "implicit",
    },
    # Level 2a: Very weak hint - "for your reference"
    {
        "id": "hint_001",
        "query": "For your reference, the API endpoints are defined in src/routes/.",
        "expected_memory_call": True,
        "prompt_level": "hint",
    },
    # Level 2b: Forward-looking hint - "keep in mind"
    {
        "id": "hint_002",
        "query": "I prefer using type hints everywhere and descriptive variable names. Keep that in mind for future work.",
        "expected_memory_call": True,
        "prompt_level": "hint",
    },
    # Level 2c: Continuity hint - "going forward"
    {
        "id": "hint_003",
        "query": "Going forward, we'll use PostgreSQL for all database operations.",
        "expected_memory_call": True,
        "prompt_level": "hint",
    },
    # Level 3: Request - explicitly asks to remember
    {
        "id": "request_001",
        "query": "We decided to use PostgreSQL instead of SQLite. Please remember this decision for later.",
        "expected_memory_call": True,
        "prompt_level": "request",
    },
    # Level 4: Command - direct instruction to use the tool
    {
        "id": "command_001",
        "query": "The API uses JWT tokens for auth. Use the save-memory tool to save this information now.",
        "expected_memory_call": True,
        "prompt_level": "command",
    },
    # Level 5: Explicit tool call - names the tool and parameters
    {
        "id": "explicit_001",
        "query": "Run this bash command now: save-memory \"Config is in /etc/app/config.yaml\" \"codebase\"",
        "expected_memory_call": True,
        "prompt_level": "explicit",
    },
    # Control - should NOT call
    {
        "id": "control_001",
        "query": "What is 2 + 2?",
        "expected_memory_call": False,
        "prompt_level": "control",
    },
]


@dataclass
class MemoryTrialResult:
    scenario_id: str
    instruction_type: str  # "natural_language" or "python_syntax"
    query: str
    expected_memory_call: bool
    actual_memory_call: bool
    memory_call_content: Optional[str]
    memory_call_category: Optional[str]
    response_text: str
    correct: bool
    bash_commands: list  # All bash commands attempted


async def run_memory_trial(
    scenario: dict,
    system_instructions: str,
    instruction_type: str,
) -> MemoryTrialResult:
    """Run a single trial testing if Claude calls the memory tool via Bash."""

    memory_calls = []
    all_bash_commands = []

    async def track_bash_calls(input_data, tool_use_id, context):
        tool_name = input_data.get("tool_name", "")
        if tool_name == "Bash":
            tool_input = input_data.get("tool_input", {})
            command = tool_input.get("command", "")
            all_bash_commands.append(command)

            # Check if this is a save-memory call
            if "save-memory" in command or "save_memory" in command:
                # Parse the command to extract content and category
                # Expected: save-memory "content" "category"
                match = re.search(r'save[-_]memory\s+"([^"]+)"\s+"([^"]+)"', command)
                if match:
                    memory_calls.append({
                        "content": match.group(1),
                        "category": match.group(2),
                    })
                else:
                    # Partial match - still counts as an attempt
                    memory_calls.append({
                        "content": command,
                        "category": "unparsed",
                    })
                # Return simulated success
                return {"result": "Memory saved successfully."}
        return {}  # Allow all calls

    options = ClaudeAgentOptions(
        allowed_tools=["Bash"],
        max_turns=2,
        permission_mode="acceptEdits",
        system_prompt=system_instructions,
        hooks={
            "PreToolUse": [
                HookMatcher(matcher="*", hooks=[track_bash_calls])
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
                elif isinstance(message, ResultMessage):
                    pass

    except Exception as e:
        response_text = f"Error: {e}"

    actual_memory_call = len(memory_calls) > 0

    return MemoryTrialResult(
        scenario_id=scenario["id"],
        instruction_type=instruction_type,
        query=scenario["query"],
        expected_memory_call=scenario["expected_memory_call"],
        actual_memory_call=actual_memory_call,
        memory_call_content=memory_calls[0]["content"] if memory_calls else None,
        memory_call_category=memory_calls[0]["category"] if memory_calls else None,
        response_text=response_text[:200],
        correct=(actual_memory_call == scenario["expected_memory_call"]),
        bash_commands=all_bash_commands,
    )


async def run_experiment():
    """Compare memory tool usage between different instruction styles and prompt levels."""

    results = []

    print("=" * 60)
    print("MEMORY TOOL EXPERIMENT")
    print("Testing: Does prompt explicitness affect proactive tool usage?")
    print("=" * 60)

    for scenario in MEMORY_SCENARIOS:
        print(f"\n--- Scenario: {scenario['id']} (level: {scenario['prompt_level']}) ---")
        print(f"Query: {scenario['query'][:60]}...")
        print(f"Expected memory call: {scenario['expected_memory_call']}")

        # Test with natural language instructions
        print("\n  [Natural Language Instructions]")
        result_nl = await run_memory_trial(
            scenario,
            INSTRUCTION_NATURAL_LANGUAGE,
            "natural_language"
        )
        print(f"    Memory called: {result_nl.actual_memory_call}")
        if result_nl.bash_commands:
            print(f"    Bash commands: {result_nl.bash_commands}")
        print(f"    Correct: {result_nl.correct}")
        results.append(result_nl)

        await asyncio.sleep(1)

        # Test with Python syntax instructions
        print("\n  [Python Syntax Instructions]")
        result_py = await run_memory_trial(
            scenario,
            INSTRUCTION_PYTHON_SYNTAX,
            "python_syntax"
        )
        print(f"    Memory called: {result_py.actual_memory_call}")
        if result_py.bash_commands:
            print(f"    Bash commands: {result_py.bash_commands}")
        print(f"    Correct: {result_py.correct}")
        results.append(result_py)

        await asyncio.sleep(1)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    nl_results = [r for r in results if r.instruction_type == "natural_language"]
    py_results = [r for r in results if r.instruction_type == "python_syntax"]

    # Overall by instruction type
    nl_should_call = [r for r in nl_results if r.expected_memory_call]
    py_should_call = [r for r in py_results if r.expected_memory_call]

    nl_actual_calls = sum(1 for r in nl_should_call if r.actual_memory_call)
    py_actual_calls = sum(1 for r in py_should_call if r.actual_memory_call)

    print(f"\n--- By Instruction Type ---")
    print(f"Natural Language: {nl_actual_calls}/{len(nl_should_call)} called when should")
    print(f"Python Syntax:    {py_actual_calls}/{len(py_should_call)} called when should")

    # By prompt level (the key insight)
    print(f"\n--- By Prompt Explicitness ---")
    prompt_levels = ["implicit", "hint", "request", "command", "explicit"]
    for level in prompt_levels:
        level_results = [r for r in results if r.scenario_id.startswith(level[:3]) and r.expected_memory_call]
        if level_results:
            called = sum(1 for r in level_results if r.actual_memory_call)
            print(f"  {level:12}: {called}/{len(level_results)} called ({100*called/len(level_results):5.1f}%)")

    # Show all bash commands for debugging
    print(f"\n--- All Bash Commands Observed ---")
    for r in results:
        if r.bash_commands:
            print(f"  {r.scenario_id} ({r.instruction_type}): {r.bash_commands}")

    # Save results
    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    output_path = output_dir / f"memory_experiment_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    asyncio.run(run_experiment())
