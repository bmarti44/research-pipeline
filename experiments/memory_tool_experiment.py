"""
Experiment: Does prompt explicitness affect proactive tool usage?

Hypothesis: Claude may be reluctant to use tools proactively, even when explicitly
instructed. This tests different levels of prompt explicitness.

METHODOLOGY FIXES (v2):
1. Multiple trials per scenario (default: 5)
2. Control scenarios for false positive testing
3. Proper statistical analysis with confidence intervals
4. Randomized condition order

The "memory tool" is simulated via Bash - Claude should call:
    save-memory "content" "category"

This tests the "missing tool call" problem - when Claude SHOULD call a tool
but doesn't.
"""

import asyncio
import json
import random
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

# Import canonical scenarios from proactive_tools - including controls
from experiments.scenarios.proactive_tools import (
    MEMORY_SCENARIOS as PROACTIVE_MEMORY_SCENARIOS,
    CONTROL_SCENARIOS as PROACTIVE_CONTROL_SCENARIOS,
    ExplicitnessLevel,
)


# EXPLICIT instruction style - matches the reasoning condition
INSTRUCTION_EXPLICIT = """
## Memory Tool

You have access to a memory tool via bash. To save important information, run:

    save-memory "content" "category"

Where:
- content: The information to remember
- category: One of 'codebase', 'task_completed', 'user_preference', 'decision', 'other'

## When to Use This Tool
You MUST use this tool whenever the user shares:
- Codebase structure or conventions (category: codebase)
- User preferences like coding style, tools, etc. (category: user_preference)
- Important decisions that were made (category: decision)
- Technical constraints or requirements (category: codebase)

If the user shares any of the above information, use the save-memory tool immediately.
"""

# VAGUE instruction style - original problematic version for comparison
INSTRUCTION_VAGUE = """
## Memory Tool

You have access to a memory tool via bash. To save important information, run:

    save-memory "content" "category"

Where:
- content: The information to remember
- category: One of 'codebase', 'task_completed', 'user_preference', 'decision', 'other'

IMPORTANT: Use this tool whenever you learn something important that should be remembered.
Be proactive about saving important information.
"""

# Map ExplicitnessLevel to prompt_level strings
LEVEL_TO_PROMPT_LEVEL: dict[ExplicitnessLevel, str] = {
    ExplicitnessLevel.IMPLICIT: "implicit",
    ExplicitnessLevel.EXPLICIT: "explicit",
    ExplicitnessLevel.CONTROL: "control",
}


def build_scenarios(include_controls: bool = True) -> list[dict]:
    """Build scenarios from proactive_tools.py.

    Args:
        include_controls: If True, include negative examples for false positive testing.
    """
    scenarios = []

    # Positive examples
    for scenario in PROACTIVE_MEMORY_SCENARIOS:
        scenarios.append({
            "id": scenario.id,
            "query": scenario.query,
            "expected_memory_call": scenario.expected_action,
            "prompt_level": LEVEL_TO_PROMPT_LEVEL.get(scenario.level, "unknown"),
            "trigger_pattern": scenario.trigger_pattern,
            "category": scenario.category,
            "is_control": False,
        })

    # Negative examples (should NOT trigger tool)
    if include_controls:
        for scenario in PROACTIVE_CONTROL_SCENARIOS:
            scenarios.append({
                "id": scenario.id,
                "query": scenario.query,
                "expected_memory_call": scenario.expected_action,  # Should be False
                "prompt_level": LEVEL_TO_PROMPT_LEVEL.get(scenario.level, "control"),
                "trigger_pattern": scenario.trigger_pattern,
                "category": scenario.category,
                "is_control": True,
            })

    return scenarios


@dataclass
class MemoryTrialResult:
    scenario_id: str
    instruction_type: str  # "explicit" or "vague"
    query: str
    expected_memory_call: bool
    actual_memory_call: bool
    memory_call_content: Optional[str]
    memory_call_category: Optional[str]
    response_text: str
    correct: bool
    bash_commands: list
    is_control: bool
    trial_number: int = 1
    # For analysis
    is_true_positive: bool = False
    is_false_positive: bool = False
    is_true_negative: bool = False
    is_false_negative: bool = False


async def run_memory_trial(
    scenario: dict,
    system_instructions: str,
    instruction_type: str,
) -> MemoryTrialResult:
    """Run a single trial testing if Claude calls the memory tool via Bash."""

    memory_calls: list[dict] = []
    all_bash_commands: list[str] = []

    async def track_bash_calls(
        input_data: dict, tool_use_id: str, context: dict
    ) -> dict:
        tool_name = input_data.get("tool_name", "")
        if tool_name == "Bash":
            tool_input = input_data.get("tool_input", {})
            command = tool_input.get("command", "")
            all_bash_commands.append(command)

            # Check if this is a save-memory call
            if "save-memory" in command or "save_memory" in command:
                # Parse the command to extract content and category
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
    expected = scenario["expected_memory_call"]

    return MemoryTrialResult(
        scenario_id=scenario["id"],
        instruction_type=instruction_type,
        query=scenario["query"],
        expected_memory_call=expected,
        actual_memory_call=actual_memory_call,
        memory_call_content=memory_calls[0]["content"] if memory_calls else None,
        memory_call_category=memory_calls[0]["category"] if memory_calls else None,
        response_text=response_text[:200],
        correct=(actual_memory_call == expected),
        bash_commands=all_bash_commands,
        is_control=scenario.get("is_control", False),
        is_true_positive=(expected and actual_memory_call),
        is_false_positive=(not expected and actual_memory_call),
        is_true_negative=(not expected and not actual_memory_call),
        is_false_negative=(expected and not actual_memory_call),
    )


def compute_confidence_interval(
    successes: int,
    total: int,
    confidence: float = 0.95
) -> tuple[float, float]:
    """Compute Wilson score confidence interval for a proportion."""
    if total == 0:
        return (0.0, 0.0)

    from math import sqrt

    z = 1.96 if confidence == 0.95 else 2.576
    p_hat = successes / total

    denominator = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denominator
    margin = z * sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total) / denominator

    return (max(0, center - margin), min(1, center + margin))


async def run_experiment(
    num_trials: int = 5,
    include_controls: bool = True,
    randomize_order: bool = True,
    seed: Optional[int] = None,
) -> list[MemoryTrialResult]:
    """Compare memory tool usage between explicit and vague instruction styles.

    Args:
        num_trials: Number of trials per scenario per instruction type (default: 5)
        include_controls: Include negative examples for false positive testing
        randomize_order: Randomize instruction type order
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)

    scenarios = build_scenarios(include_controls=include_controls)
    results: list[MemoryTrialResult] = []

    n_positive = sum(1 for s in scenarios if s["expected_memory_call"])
    n_negative = sum(1 for s in scenarios if not s["expected_memory_call"])

    print("=" * 70)
    print("MEMORY TOOL EXPERIMENT (v2 - Fixed Methodology)")
    print("Testing: Does instruction explicitness affect proactive tool usage?")
    print("=" * 70)
    print(f"Trials per scenario per condition: {num_trials}")
    print(f"Positive scenarios (should save): {n_positive}")
    print(f"Negative scenarios (should NOT save): {n_negative}")
    print(f"Total scenarios: {len(scenarios)}")
    print(f"Total observations: {len(scenarios) * num_trials * 2}")
    print("=" * 70)

    for scenario in scenarios:
        print(f"\n--- {scenario['id']} ({scenario['prompt_level']}) ---")
        print(f"Query: {scenario['query'][:50]}...")
        print(f"Expected: {scenario['expected_memory_call']}")

        for trial in range(1, num_trials + 1):
            if num_trials > 1:
                print(f"\n  Trial {trial}/{num_trials}")

            # Randomize instruction order
            instruction_types = [
                ("explicit", INSTRUCTION_EXPLICIT),
                ("vague", INSTRUCTION_VAGUE),
            ]
            if randomize_order:
                random.shuffle(instruction_types)

            for inst_type, inst_prompt in instruction_types:
                result = await run_memory_trial(scenario, inst_prompt, inst_type)
                result.trial_number = trial

                status = "CORRECT" if result.correct else "WRONG"
                print(f"    [{inst_type}] {status} (called={result.actual_memory_call})")

                results.append(result)
                await asyncio.sleep(0.5)

    # Analysis
    print("\n" + "=" * 70)
    print("RESULTS ANALYSIS")
    print("=" * 70)

    explicit_results = [r for r in results if r.instruction_type == "explicit"]
    vague_results = [r for r in results if r.instruction_type == "vague"]

    # Positive scenarios (recall)
    explicit_positives = [r for r in explicit_results if r.expected_memory_call]
    vague_positives = [r for r in vague_results if r.expected_memory_call]

    explicit_tp = sum(1 for r in explicit_positives if r.is_true_positive)
    vague_tp = sum(1 for r in vague_positives if r.is_true_positive)

    # Negative scenarios (false positive rate)
    explicit_negatives = [r for r in explicit_results if not r.expected_memory_call]
    vague_negatives = [r for r in vague_results if not r.expected_memory_call]

    explicit_fp = sum(1 for r in explicit_negatives if r.is_false_positive)
    vague_fp = sum(1 for r in vague_negatives if r.is_false_positive)

    n_pos = len(explicit_positives)
    n_neg = len(explicit_negatives)

    print(f"\n--- Recall (True Positive Rate) ---")
    print(f"On {n_pos} positive scenarios:")

    explicit_recall = explicit_tp / n_pos if n_pos > 0 else 0
    vague_recall = vague_tp / n_pos if n_pos > 0 else 0

    e_ci = compute_confidence_interval(explicit_tp, n_pos)
    v_ci = compute_confidence_interval(vague_tp, n_pos)

    print(f"  Explicit: {explicit_tp}/{n_pos} ({explicit_recall*100:.1f}%) "
          f"95% CI: [{e_ci[0]*100:.1f}%, {e_ci[1]*100:.1f}%]")
    print(f"  Vague:    {vague_tp}/{n_pos} ({vague_recall*100:.1f}%) "
          f"95% CI: [{v_ci[0]*100:.1f}%, {v_ci[1]*100:.1f}%]")

    improvement = explicit_recall - vague_recall
    print(f"  Improvement: {improvement*100:+.1f}pp")

    if n_neg > 0:
        print(f"\n--- False Positive Rate ---")
        print(f"On {n_neg} negative (control) scenarios:")

        explicit_fpr = explicit_fp / n_neg
        vague_fpr = vague_fp / n_neg

        print(f"  Explicit: {explicit_fp}/{n_neg} ({explicit_fpr*100:.1f}%)")
        print(f"  Vague:    {vague_fp}/{n_neg} ({vague_fpr*100:.1f}%)")

    # By prompt level
    print(f"\n--- By Prompt Level (Positive Scenarios) ---")
    level_prefixes = [
        ("mem_implicit_", "implicit"),
        ("mem_explicit_", "explicit"),
    ]

    for prefix, level_name in level_prefixes:
        e_level = [r for r in explicit_positives if r.scenario_id.startswith(prefix)]
        v_level = [r for r in vague_positives if r.scenario_id.startswith(prefix)]

        if e_level:
            e_rate = sum(1 for r in e_level if r.is_true_positive) / len(e_level)
            v_rate = sum(1 for r in v_level if r.is_true_positive) / len(v_level)
            level_diff = e_rate - v_rate
            print(f"  {level_name:10}: Explicit {e_rate*100:5.1f}% | "
                  f"Vague {v_rate*100:5.1f}% | Diff {level_diff*100:+5.1f}pp")

    # Save results
    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    output_path = output_dir / f"memory_experiment_{timestamp}.json"

    output_data = {
        "metadata": {
            "timestamp": timestamp,
            "num_trials": num_trials,
            "include_controls": include_controls,
            "randomize_order": randomize_order,
            "seed": seed,
            "n_positive_scenarios": n_positive,
            "n_negative_scenarios": n_negative,
            "total_observations": len(results),
        },
        "summary": {
            "explicit_recall": explicit_recall,
            "vague_recall": vague_recall,
            "improvement_pp": improvement * 100,
            "explicit_ci_95": [e_ci[0], e_ci[1]],
            "vague_ci_95": [v_ci[0], v_ci[1]],
            "explicit_false_positive_rate": explicit_fpr if n_neg > 0 else None,
            "vague_false_positive_rate": vague_fpr if n_neg > 0 else None,
        },
        "results": [asdict(r) for r in results],
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test proactive memory tool usage with explicit vs vague instructions"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of trials per scenario per condition (default: 5)",
    )
    parser.add_argument(
        "--no-controls",
        action="store_true",
        help="Exclude negative (control) scenarios",
    )
    parser.add_argument(
        "--no-randomize",
        action="store_true",
        help="Don't randomize condition order",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    asyncio.run(run_experiment(
        num_trials=args.trials,
        include_controls=not args.no_controls,
        randomize_order=not args.no_randomize,
        seed=args.seed,
    ))
