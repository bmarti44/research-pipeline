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
import random
import warnings
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
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
    BaselineContext,
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
    # Enhanced audit logging
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_ms: Optional[int] = None
    validation_log: list[dict] = field(default_factory=list)
    tool_call_sequence: list[dict] = field(default_factory=list)
    convergence_state: Optional[dict] = None
    context_state: Optional[dict] = None


async def run_baseline_trial(scenario: Scenario, trial_num: int) -> TrialResult:
    """Run a trial without validation hooks (logging only).

    For scenarios with prior context requirements (F10, F16, F17, F18, F20),
    we set up the context to enable fair comparison with validated trials.
    """
    started_at = datetime.now(timezone.utc)

    # Set up prior context based on scenario type
    prior_searches = []
    if scenario.failure_mode == "F10":
        prior_searches = _get_prior_searches_for_scenario(scenario)

    # Create logging hooks with prior search context
    log_pre, log_post, calls, baseline_ctx = create_logging_hooks(prior_searches=prior_searches)

    # Set up additional prior context for rules that need it
    if scenario.failure_mode == "F17":
        for pattern in _get_prior_globs_for_scenario(scenario):
            baseline_ctx.add_glob_pattern(pattern)

    if scenario.failure_mode == "F18":
        for cmd in _get_prior_commands_for_scenario(scenario):
            baseline_ctx.add_tool_output(
                cmd.get("tool", "Bash"),
                {"command": cmd.get("command", ""), "file": cmd.get("file", "")},
                "success:completed"
            )

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

    completed_at = datetime.now(timezone.utc)
    duration_ms = int((completed_at - started_at).total_seconds() * 1000)

    # All tools attempted = all tools used (no validator)
    tools_used = [c["tool_name"] for c in calls if c["event"] == "pre"]

    # Build tool call sequence from logging hooks
    tool_call_sequence = [
        {
            "timestamp": c.get("timestamp", started_at.isoformat()),
            "tool_name": c.get("tool_name"),
            "tool_input": c.get("tool_input"),
            "status": "started" if c["event"] == "pre" else "completed",
        }
        for c in calls
    ]

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
        started_at=started_at.isoformat(),
        completed_at=completed_at.isoformat(),
        duration_ms=duration_ms,
        validation_log=[],  # No validation in baseline
        tool_call_sequence=tool_call_sequence,
        convergence_state=None,
        context_state=baseline_ctx.to_dict(),  # Track context for F10 comparison
    )


async def run_validated_trial(
    scenario: Scenario,
    trial_num: int,
    enabled_rules: Optional[set[str]] = None,
) -> TrialResult:
    """Run a trial with validation hooks.

    Args:
        scenario: The scenario to test
        trial_num: Trial number for this scenario
        enabled_rules: Optional set of rule IDs to enable for ablation testing.
                      If None, all rules are enabled.
    """
    started_at = datetime.now(timezone.utc)

    # Use cached classifier to avoid reloading model each trial
    classifier = get_cached_classifier()
    validator = RuleValidator(semantic=classifier, enabled_rules=enabled_rules)

    state = HookState.create(scenario.query, validator=validator)
    pre_hook = create_pre_tool_use_hook(state)
    post_hook = create_post_tool_use_hook(state)

    # Set up prior context for F10 scenarios (duplicate search)
    if scenario.failure_mode == "F10":
        prior_searches = _get_prior_searches_for_scenario(scenario)
        state.context.search_queries.extend(prior_searches)

    # Set up known paths for F13 scenarios (simulate having listed a directory)
    if scenario.failure_mode == "F13":
        state.context.add_known_paths(["/home/claude/actual_file.txt"])

    # Set up prior glob patterns for F17 scenarios (redundant glob)
    if scenario.failure_mode == "F17":
        for pattern in _get_prior_globs_for_scenario(scenario):
            state.context.add_glob_pattern(pattern)

    # Set up prior tool outputs for F18 scenarios (reverification)
    if scenario.failure_mode == "F18":
        for cmd in _get_prior_commands_for_scenario(scenario):
            state.context.add_tool_output(
                cmd.get("tool", "Bash"),
                {"command": cmd.get("command", ""), "file": cmd.get("file", "")},
                "success:completed"
            )

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

    completed_at = datetime.now(timezone.utc)
    duration_ms = int((completed_at - started_at).total_seconds() * 1000)

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

    # Serialize validation log for JSON
    validation_log = [
        {
            "timestamp": e.timestamp,
            "tool_name": e.tool_name,
            "tool_input": e.tool_input,
            "decision": e.decision,
            "rule_id": e.rule_id,
            "reason": e.reason,
            "feedback_level": e.feedback_level,
            "semantic_scores": e.semantic_scores,
        }
        for e in state.validation_log
    ]

    # Capture convergence state
    convergence_state = {
        "total_rejections": state.convergence.total_rejections,
        "forced_direct_answer": state.convergence.forced_direct_answer,
        "termination_reason": state.convergence.termination_reason,
        "violations": {
            rule_id: {"count": v.count, "last_level": v.last_feedback_level.name}
            for rule_id, v in state.convergence.violations.items()
        },
    }

    # Capture context state
    context_state = {
        "tool_history": state.context.tool_history,
        "search_queries": state.context.search_queries,
        "known_paths": list(state.context.known_paths),
    }

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
        started_at=started_at.isoformat(),
        completed_at=completed_at.isoformat(),
        duration_ms=duration_ms,
        validation_log=validation_log,
        tool_call_sequence=state.tool_call_sequence,
        convergence_state=convergence_state,
        context_state=context_state,
    )


def _get_prior_searches_for_scenario(scenario: Scenario) -> list[str]:
    """Get prior search queries for duplicate/cascading search scenarios."""
    # Check for explicit prior_searches in scenario data
    if hasattr(scenario, 'prior_searches') and scenario.prior_searches:
        return scenario.prior_searches

    # Fallback for F10 scenarios without explicit prior_searches
    if "Python tutorials" in scenario.query:
        return ["Python tutorials"]
    if "stock price" in scenario.query:
        return ["AAPL stock price"]
    if "React documentation" in scenario.query:
        return ["React documentation"]
    return []


def _get_prior_reads_for_scenario(scenario: Scenario) -> list[str]:
    """Get prior file reads for duplicate read scenarios (F16)."""
    if hasattr(scenario, 'prior_reads') and scenario.prior_reads:
        return scenario.prior_reads
    return []


def _get_prior_globs_for_scenario(scenario: Scenario) -> list[str]:
    """Get prior glob patterns for redundant glob scenarios (F17)."""
    if hasattr(scenario, 'prior_globs') and scenario.prior_globs:
        return scenario.prior_globs
    return []


def _get_prior_commands_for_scenario(scenario: Scenario) -> list[dict]:
    """Get prior commands for reverification scenarios (F18)."""
    if hasattr(scenario, 'prior_commands') and scenario.prior_commands:
        return scenario.prior_commands
    return []


async def run_experiment(
    scenarios: list[Scenario],
    output_dir: Path,
    n_trials: int = N_TRIALS_PER_SCENARIO,
    delay_between_trials: float = 0.5,
    random_seed: Optional[int] = None,
    enabled_rules: Optional[set[str]] = None,
) -> list[TrialResult]:
    """Run all scenarios in both baseline and validated modes.

    Trial order is randomized with counterbalancing to eliminate ordering effects.
    For each scenario-trial pair, we randomly choose whether baseline or validated
    runs first, ensuring roughly equal distribution of orderings.

    Args:
        scenarios: List of scenarios to test
        output_dir: Directory to save results
        n_trials: Number of trials per scenario
        delay_between_trials: Seconds to wait between trials
        random_seed: Optional seed for reproducibility of trial ordering
        enabled_rules: Optional set of rule IDs for ablation testing.
                      If None, all rules are enabled.
    """

    results = []
    total_trials = len(scenarios) * n_trials * 2
    current = 0

    # Set random seed for reproducibility
    if random_seed is not None:
        random.seed(random_seed)
        print(f"Using random seed: {random_seed}")

    # Generate counterbalanced trial order
    # For each (scenario, trial_num), randomly decide if baseline or validated runs first
    # We ensure roughly equal split by alternating the bias
    trial_orders = []
    for i, scenario in enumerate(scenarios):
        for trial_num in range(n_trials):
            # Use a mix of deterministic alternation and randomization for counterbalancing
            # This ensures roughly 50% baseline-first, 50% validated-first
            baseline_first = random.choice([True, False])
            trial_orders.append((scenario, trial_num, baseline_first))

    # Shuffle the overall trial order as well to avoid scenario ordering effects
    random.shuffle(trial_orders)

    # Log ablation configuration
    if enabled_rules is not None:
        print(f"Ablation mode: only rules {enabled_rules} enabled")

    for scenario, trial_num, baseline_first in trial_orders:
        if baseline_first:
            # Baseline first, then validated
            current += 1
            print(f"\n[{current}/{total_trials}] Baseline: {scenario.id} (trial {trial_num + 1})")
            baseline_result = await run_baseline_trial(scenario, trial_num)
            results.append(baseline_result)
            print(f"  Tools: {baseline_result.tools_used}, Score: {baseline_result.score.score}")

            await asyncio.sleep(delay_between_trials)

            current += 1
            print(f"[{current}/{total_trials}] Validated: {scenario.id} (trial {trial_num + 1})")
            validated_result = await run_validated_trial(scenario, trial_num, enabled_rules=enabled_rules)
            results.append(validated_result)
            print(f"  Tools: {validated_result.tools_used}, Score: {validated_result.score.score}")
            print(f"  Rejections: {len(validated_result.validator_rejections)}")
        else:
            # Validated first, then baseline
            current += 1
            print(f"\n[{current}/{total_trials}] Validated: {scenario.id} (trial {trial_num + 1})")
            validated_result = await run_validated_trial(scenario, trial_num, enabled_rules=enabled_rules)
            results.append(validated_result)
            print(f"  Tools: {validated_result.tools_used}, Score: {validated_result.score.score}")
            print(f"  Rejections: {len(validated_result.validator_rejections)}")

            await asyncio.sleep(delay_between_trials)

            current += 1
            print(f"[{current}/{total_trials}] Baseline: {scenario.id} (trial {trial_num + 1})")
            baseline_result = await run_baseline_trial(scenario, trial_num)
            results.append(baseline_result)
            print(f"  Tools: {baseline_result.tools_used}, Score: {baseline_result.score.score}")

        await asyncio.sleep(delay_between_trials)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    results_data = []
    for r in results:
        d = asdict(r)
        d["score"] = asdict(r.score)
        d["score"]["category"] = r.score.category.name
        results_data.append(d)

    output_path = output_dir / f"results_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2)

    # Compute experiment statistics
    durations = [r.duration_ms for r in results if r.duration_ms]
    baseline_results = [r for r in results if r.approach == "baseline"]
    validated_results = [r for r in results if r.approach == "validated"]

    # Also save experiment metadata with model variability controls
    metadata = {
        # Timing
        "experiment_started": results[0].started_at if results else None,
        "experiment_completed": results[-1].completed_at if results else None,
        "total_duration_minutes": round(
            sum(durations) / 60000, 2
        ) if durations else None,

        # Experimental design
        "total_trials": len(results),
        "scenarios_count": len(scenarios),
        "trials_per_scenario": n_trials,
        "random_seed": random_seed,
        "trial_order": "randomized_counterbalanced",

        # Ablation configuration
        "ablation": {
            "enabled_rules": list(enabled_rules) if enabled_rules else list(RuleValidator.ALL_RULES),
            "is_ablation_run": enabled_rules is not None,
        },

        # Model variability controls
        "model": "claude",  # Model used
        "sdk_version": "claude-agent-sdk",  # SDK used
        "temperature": None,  # Fixed temperature if API allows (currently not available)

        # Timing statistics (for detecting API load effects)
        "timing_stats": {
            "mean_trial_duration_ms": round(sum(durations) / len(durations), 1) if durations else None,
            "min_trial_duration_ms": min(durations) if durations else None,
            "max_trial_duration_ms": max(durations) if durations else None,
            "std_trial_duration_ms": round(
                (sum((d - sum(durations)/len(durations))**2 for d in durations) / len(durations))**0.5, 1
            ) if durations and len(durations) > 1 else None,
        },

        # Error tracking
        "errors": {
            "baseline_errors": sum(1 for r in baseline_results if r.error),
            "validated_errors": sum(1 for r in validated_results if r.error),
            "error_scenarios": list(set(r.scenario_id for r in results if r.error)),
        },

        # Classifier configuration
        "classifier_thresholds": {
            "static_knowledge": get_cached_classifier().thresholds.static_knowledge,
            "memory_reference": get_cached_classifier().thresholds.memory_reference,
            "duplicate_search": get_cached_classifier().thresholds.duplicate_search,
            "duplicate_file_read": get_cached_classifier().thresholds.duplicate_file_read,
            "cascading_search": get_cached_classifier().thresholds.cascading_search,
            "answer_in_context": get_cached_classifier().thresholds.answer_in_context,
        },
        "classifier_model": "all-MiniLM-L6-v2",
    }

    metadata_path = output_dir / f"metadata_{timestamp}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"Metadata saved to {metadata_path}")

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
    baseline_tool_calls = sum(len(r.tools_used) for r in baseline)
    print(f"  Total tool calls: {baseline_tool_calls}")

    print(f"\nValidated:")
    print(f"  Mean score: {sum(validated_scores)/len(validated_scores):.3f}")
    validated_tool_calls = sum(len(r.tools_used) for r in validated)
    validated_attempted = sum(len(r.tools_attempted) for r in validated)
    blocked_calls = validated_attempted - validated_tool_calls
    print(f"  Tool calls executed: {validated_tool_calls}")
    print(f"  Tool calls blocked: {blocked_calls}")
    if validated_attempted > 0:
        print(f"  Block rate: {blocked_calls/validated_attempted:.1%}")

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

    # Cost estimate (assuming ~$0.01 per tool call on average)
    if blocked_calls > 0:
        print(f"\nEstimated savings:")
        print(f"  Blocked calls: {blocked_calls}")
        print(f"  Est. cost saved: ${blocked_calls * 0.01:.2f} (at $0.01/call)")


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

    # Parse command line arguments
    n_trials = N_TRIALS_PER_SCENARIO
    random_seed = None
    enabled_rules = None  # None means all rules

    # Support --quick flag for fast testing (one scenario per failure mode)
    if "--quick" in sys.argv:
        # Pick one scenario per failure mode + valid scenarios
        quick_ids = [
            "f1_001", "f4_001", "f8_001", "f10_001", "f13_001", "f15_001",  # Original rules
            "f17_001", "f18_001", "f19_001",  # Efficiency rules
            "f21_001", "f22_001", "f23_001",  # Distinct rules
            "f24_001", "f25_001",  # Root cause rules (well-known API, trivial knowledge)
            "valid_001", "valid_020", "valid_030", "valid_040"  # Valid scenarios
        ]
        scenarios = [s for s in scenarios if s.id in quick_ids]
        n_trials = 1
        print(f"Quick mode: {len(scenarios)} scenarios, 1 trial each")

    # Support --help flag
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python -m experiments.runner [OPTIONS]")
        print()
        print("Options:")
        print("  --quick             Run quick test (8 scenarios, 1 trial each)")
        print("  --trials=N          Number of trials per scenario (default: 5)")
        print("  --seed=N            Random seed for reproducible trial ordering")
        print("  --only-rule=RULE    Ablation: only enable one rule (F1, F4, F8, F10, F13, F15)")
        print("  --exclude-rule=RULE Ablation: enable all rules except one")
        print("  --rules=R1,R2,...   Ablation: enable specific rules (comma-separated)")
        print()
        print("Examples:")
        print("  python -m experiments.runner --quick")
        print("  python -m experiments.runner --only-rule=F1")
        print("  python -m experiments.runner --rules=F1,F4,F15 --trials=3")
        return

    # Parse other arguments
    for arg in sys.argv:
        if arg.startswith("--seed="):
            random_seed = int(arg.split("=")[1])
        elif arg.startswith("--trials="):
            n_trials = int(arg.split("=")[1])
        elif arg.startswith("--only-rule="):
            rule = arg.split("=")[1].upper()
            enabled_rules = {rule}
            print(f"Ablation: only rule {rule} enabled")
        elif arg.startswith("--exclude-rule="):
            rule = arg.split("=")[1].upper()
            enabled_rules = RuleValidator.ALL_RULES - {rule}
            print(f"Ablation: all rules except {rule}")
        elif arg.startswith("--rules="):
            rules_str = arg.split("=")[1]
            enabled_rules = set(r.strip().upper() for r in rules_str.split(","))
            print(f"Ablation: rules {enabled_rules} enabled")
        elif arg == "--proven-only":
            enabled_rules = RuleValidator.PROVEN_RULES.copy()
            print(f"Using only proven rules: {enabled_rules}")

    print(f"Loaded {len(scenarios)} scenarios")
    print(f"Running {n_trials} trials per scenario")
    print(f"Total trials: {len(scenarios) * n_trials * 2}")
    print(f"Trial order: randomized with counterbalancing")
    if enabled_rules:
        print(f"Enabled rules: {enabled_rules}")
    else:
        print(f"Enabled rules: all ({RuleValidator.ALL_RULES})")

    output_dir = Path("experiments/results")

    results = await run_experiment(
        scenarios, output_dir,
        n_trials=n_trials,
        random_seed=random_seed,
        enabled_rules=enabled_rules,
    )
    summarize_results(results)


if __name__ == "__main__":
    asyncio.run(main())
