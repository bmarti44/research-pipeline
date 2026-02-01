"""
Run the validator experiment using Claude Agent SDK.

CRITICAL: Uses ClaudeSDKClient (not bare query()) for hook support.

Compares:
- Baseline: No validation hooks (Claude decides freely)
- Validated: PreToolUse/PostToolUse hooks block bad tool calls
"""

import asyncio
import json
import logging
import os
import warnings
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Suppress noisy model loading output
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

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
from src.validator.semantic import SemanticClassifier
from src.validator.rules import RuleValidator
from src.scoring import score_trial, TrialScore
from scenarios.generator import load_scenarios, Scenario

# Global cached classifier (loaded once)
_cached_classifier: SemanticClassifier | None = None


def get_cached_classifier() -> SemanticClassifier:
    global _cached_classifier
    if _cached_classifier is None:
        print("Loading semantic classifier (one-time)...")
        _cached_classifier = SemanticClassifier()
        print("Classifier loaded.")
    return _cached_classifier


# Number of trials per scenario for statistical validity
# Set to 1 for quick testing, 5 for full experiment
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
    # Use cached classifier to avoid reloading model each trial
    classifier = get_cached_classifier()
    validator = RuleValidator(semantic=classifier)

    state = HookState.create(scenario.query, validator=validator)
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
    delay_between_trials: float = 0.5,
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
    import sys

    # Preload the classifier once at startup
    get_cached_classifier()

    scenarios_path = Path("scenarios/generated/scenarios.json")

    if not scenarios_path.exists():
        print("Generating scenarios...")
        from scenarios.generator import save_scenarios
        save_scenarios(scenarios_path)

    scenarios = load_scenarios(scenarios_path)

    # Support --quick flag for fast testing (just 6 scenarios, one per failure mode)
    if "--quick" in sys.argv:
        # Pick one scenario per failure mode + 2 valid scenarios
        quick_ids = ["f1_001", "f4_001", "f8_001", "f10_001", "f13_001", "f15_001", "valid_001", "valid_020"]
        scenarios = [s for s in scenarios if s.id in quick_ids]
        print(f"Quick mode: {len(scenarios)} scenarios")

    print(f"Loaded {len(scenarios)} scenarios")
    print(f"Running {N_TRIALS_PER_SCENARIO} trials per scenario")
    print(f"Total trials: {len(scenarios) * N_TRIALS_PER_SCENARIO * 2}")

    output_dir = Path("experiments/results")

    results = await run_experiment(scenarios, output_dir)
    summarize_results(results)


if __name__ == "__main__":
    asyncio.run(main())
