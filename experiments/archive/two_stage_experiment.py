"""
Two-Stage Tool Calling Experiment

Compares single-model tool calling vs two-stage (reasoner + extractor) approach.

Hypothesis: Separating reasoning from formatting improves tool-calling accuracy
by eliminating format friction during the reasoning phase.

Usage:
    # Run pilot (5 scenarios, 2 trials)
    python experiments/two_stage_experiment.py --pilot

    # Run full experiment
    python experiments/two_stage_experiment.py

    # Run with specific number of trials
    python experiments/two_stage_experiment.py --trials 5
"""

import asyncio
import json
import argparse
import logging
import random
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
)

from tool_definitions import (
    TOOLS,
    get_all_tools_schema,
    parse_tool_xml,
    check_no_tool,
)
from scenarios.tool_calling import (
    ALL_SCENARIOS,
    ALL_TOOL_SCENARIOS,
    ALL_CONTROL_SCENARIOS,
    ToolScenario,
    get_scenario_stats,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Prompts
# =============================================================================

SINGLE_MODEL_SYSTEM = """You are a helpful assistant with access to these tools:

{tool_schema}

When appropriate, call tools using this format:
<tool name="tool_name">
  <param name="param1">value1</param>
  <param name="param2">value2</param>
</tool>

If no tool is needed, just respond naturally without any XML tags."""


TWO_STAGE_REASONER_SYSTEM = """You are a helpful assistant. Respond naturally to the user.
If you would save information, search for something, run code, read/write files,
or make API calls, describe what you would do in plain English.
Do not use any special formatting or XML tags."""


TWO_STAGE_EXTRACTOR_SYSTEM = """You are a tool extraction system. Given a natural language response,
extract any tool calls that should be made.

Available tools:
{tool_schema}

Original user query:
"{user_query}"

Assistant's natural language response:
"{reasoner_response}"

If tools should be called, output them in this format:
<tool name="tool_name">
  <param name="param1">value1</param>
</tool>

If no tools should be called, output: <no_tool/>

Output ONLY the tool XML or <no_tool/>. No other text."""


JUDGE_SYSTEM = """You are evaluating whether a tool call matches the expected call.

Expected tool: {expected_tool}
Expected parameters: {expected_params}

Actual tool: {actual_tool}
Actual parameters: {actual_params}

Evaluate:
1. Is the correct tool selected? (YES/NO)
2. Do the parameters capture equivalent information? (YES/PARTIAL/NO)
3. Are there hallucinated parameters not in the original? (YES/NO)

Respond with a JSON object:
{{"tool_correct": true/false, "params_match": "YES"/"PARTIAL"/"NO", "hallucination": true/false, "verdict": "MATCH"/"PARTIAL_MATCH"/"MISMATCH"}}

Output only the JSON, nothing else."""


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TrialResult:
    """Result of a single trial."""
    scenario_id: str
    trial_number: int
    condition: str  # "single" or "two_stage"
    query: str
    expected_tool: Optional[str]
    expected_params: dict
    ambiguity: str
    domain: str

    # Response data
    response_text: str
    reasoner_response: Optional[str] = None  # Only for two_stage
    extractor_response: Optional[str] = None  # Only for two_stage

    # Parsed results
    actual_tool: Optional[str] = None
    actual_params: dict = None

    # Scoring
    tool_detected: bool = False  # Any tool was called
    tool_correct: bool = False   # Correct tool was called
    params_match: str = "NO"     # YES/PARTIAL/NO
    no_tool_correct: bool = False  # Correctly did NOT call a tool (controls)
    verdict: str = "MISMATCH"    # Final verdict

    # Metadata
    error: Optional[str] = None
    duration_ms: int = 0

    def __post_init__(self):
        if self.actual_params is None:
            self.actual_params = {}


# =============================================================================
# API Calls
# =============================================================================

async def call_model(
    prompt: str,
    system_prompt: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, Optional[str]]:
    """Call the model with given prompt and system.

    Returns:
        Tuple of (response_text, error_message)
    """
    async with semaphore:
        options = ClaudeAgentOptions(
            allowed_tools=[],
            max_turns=1,
            permission_mode="acceptEdits",
            system_prompt=system_prompt,
            model="claude-sonnet-4-5-20250929",
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

                return response_text.strip(), None

        except Exception as e:
            logger.error(f"API call failed: {e}")
            return "", str(e)


async def run_single_model_trial(
    scenario: ToolScenario,
    trial_number: int,
    semaphore: asyncio.Semaphore,
) -> TrialResult:
    """Run a single trial with the single-model condition."""
    import time
    start_time = time.time()

    system = SINGLE_MODEL_SYSTEM.format(tool_schema=get_all_tools_schema())
    response_text, error = await call_model(scenario.query, system, semaphore)

    duration_ms = int((time.time() - start_time) * 1000)

    result = TrialResult(
        scenario_id=scenario.id,
        trial_number=trial_number,
        condition="single",
        query=scenario.query,
        expected_tool=scenario.expected_tool,
        expected_params=scenario.expected_params,
        ambiguity=scenario.ambiguity,
        domain=scenario.domain,
        response_text=response_text,
        error=error,
        duration_ms=duration_ms,
    )

    # Parse tool call
    if error:
        return result

    parsed = parse_tool_xml(response_text)
    if parsed:
        result.actual_tool, result.actual_params = parsed
        result.tool_detected = True
        result.tool_correct = (result.actual_tool == scenario.expected_tool)
    else:
        result.tool_detected = False
        # For controls, not calling a tool is correct
        if scenario.expected_tool is None:
            result.no_tool_correct = not result.tool_detected

    return result


async def run_two_stage_trial(
    scenario: ToolScenario,
    trial_number: int,
    semaphore: asyncio.Semaphore,
) -> TrialResult:
    """Run a single trial with the two-stage condition."""
    import time
    start_time = time.time()

    # Stage 1: Reasoner
    reasoner_response, error = await call_model(
        scenario.query,
        TWO_STAGE_REASONER_SYSTEM,
        semaphore,
    )

    if error:
        return TrialResult(
            scenario_id=scenario.id,
            trial_number=trial_number,
            condition="two_stage",
            query=scenario.query,
            expected_tool=scenario.expected_tool,
            expected_params=scenario.expected_params,
            ambiguity=scenario.ambiguity,
            domain=scenario.domain,
            response_text="",
            reasoner_response=reasoner_response,
            error=f"Reasoner error: {error}",
            duration_ms=int((time.time() - start_time) * 1000),
        )

    # Stage 2: Extractor
    extractor_system = TWO_STAGE_EXTRACTOR_SYSTEM.format(
        tool_schema=get_all_tools_schema(),
        user_query=scenario.query,
        reasoner_response=reasoner_response,
    )

    extractor_response, error = await call_model(
        "Extract any tool calls from the response above.",
        extractor_system,
        semaphore,
    )

    duration_ms = int((time.time() - start_time) * 1000)

    result = TrialResult(
        scenario_id=scenario.id,
        trial_number=trial_number,
        condition="two_stage",
        query=scenario.query,
        expected_tool=scenario.expected_tool,
        expected_params=scenario.expected_params,
        ambiguity=scenario.ambiguity,
        domain=scenario.domain,
        response_text=extractor_response,
        reasoner_response=reasoner_response,
        extractor_response=extractor_response,
        error=error,
        duration_ms=duration_ms,
    )

    if error:
        result.error = f"Extractor error: {error}"
        return result

    # Parse tool call from extractor output
    parsed = parse_tool_xml(extractor_response)
    if parsed:
        result.actual_tool, result.actual_params = parsed
        result.tool_detected = True
        result.tool_correct = (result.actual_tool == scenario.expected_tool)
    else:
        result.tool_detected = False
        # Check for explicit no-tool marker
        if check_no_tool(extractor_response):
            result.no_tool_correct = (scenario.expected_tool is None)
        elif scenario.expected_tool is None:
            # No tool called and none expected
            result.no_tool_correct = True

    return result


async def judge_result(
    result: TrialResult,
    semaphore: asyncio.Semaphore,
) -> TrialResult:
    """Use LLM judge to evaluate parameter matching."""
    if result.error or not result.tool_correct:
        # Can't judge params if wrong tool or error
        if result.expected_tool is None and not result.tool_detected:
            result.verdict = "MATCH"  # Correctly didn't call a tool
        return result

    # Use judge for parameter comparison
    judge_prompt = JUDGE_SYSTEM.format(
        expected_tool=result.expected_tool,
        expected_params=json.dumps(result.expected_params),
        actual_tool=result.actual_tool,
        actual_params=json.dumps(result.actual_params),
    )

    response, error = await call_model(
        "Evaluate the tool call.",
        judge_prompt,
        semaphore,
    )

    if error:
        logger.warning(f"Judge error for {result.scenario_id}: {error}")
        return result

    # Parse judge response
    try:
        # Extract JSON from response
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        judge_result = json.loads(response)

        result.params_match = judge_result.get("params_match", "NO")
        result.verdict = judge_result.get("verdict", "MISMATCH")

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to parse judge response: {e}")

    return result


# =============================================================================
# Experiment Runner
# =============================================================================

async def run_experiment(
    scenarios: list[ToolScenario],
    trials_per_scenario: int,
    max_concurrent: int = 20,
) -> list[TrialResult]:
    """Run the full experiment.

    Args:
        scenarios: List of scenarios to test
        trials_per_scenario: Number of trials per scenario per condition
        max_concurrent: Maximum concurrent API calls

    Returns:
        List of all trial results
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    all_results = []

    total_trials = len(scenarios) * trials_per_scenario * 2  # 2 conditions
    logger.info(f"Running {total_trials} trials ({len(scenarios)} scenarios × {trials_per_scenario} trials × 2 conditions)")

    # Build all trial tasks
    tasks = []
    for scenario in scenarios:
        for trial_num in range(1, trials_per_scenario + 1):
            # Single-model condition
            tasks.append(run_single_model_trial(scenario, trial_num, semaphore))
            # Two-stage condition
            tasks.append(run_two_stage_trial(scenario, trial_num, semaphore))

    # Run all trials
    logger.info("Running trials...")
    results = await asyncio.gather(*tasks)
    all_results.extend(results)

    # Judge all results
    logger.info("Judging results...")
    judge_tasks = [judge_result(r, semaphore) for r in all_results]
    all_results = await asyncio.gather(*judge_tasks)

    return list(all_results)


def compute_metrics(results: list[TrialResult]) -> dict:
    """Compute metrics from trial results."""
    metrics = {
        "single": {"total": 0, "tool_correct": 0, "full_match": 0, "false_positive": 0},
        "two_stage": {"total": 0, "tool_correct": 0, "full_match": 0, "false_positive": 0},
        "by_ambiguity": {},
        "by_domain": {},
    }

    for result in results:
        cond = result.condition
        metrics[cond]["total"] += 1

        if result.tool_correct:
            metrics[cond]["tool_correct"] += 1

        if result.verdict == "MATCH":
            metrics[cond]["full_match"] += 1

        # False positive: called a tool when none expected
        if result.expected_tool is None and result.tool_detected:
            metrics[cond]["false_positive"] += 1

        # By ambiguity
        amb = result.ambiguity
        if amb not in metrics["by_ambiguity"]:
            metrics["by_ambiguity"][amb] = {
                "single": {"total": 0, "full_match": 0},
                "two_stage": {"total": 0, "full_match": 0},
            }
        metrics["by_ambiguity"][amb][cond]["total"] += 1
        if result.verdict == "MATCH":
            metrics["by_ambiguity"][amb][cond]["full_match"] += 1

        # By domain
        dom = result.domain
        if dom not in metrics["by_domain"]:
            metrics["by_domain"][dom] = {
                "single": {"total": 0, "full_match": 0},
                "two_stage": {"total": 0, "full_match": 0},
            }
        metrics["by_domain"][dom][cond]["total"] += 1
        if result.verdict == "MATCH":
            metrics["by_domain"][dom][cond]["full_match"] += 1

    # Compute rates
    for cond in ["single", "two_stage"]:
        total = metrics[cond]["total"]
        if total > 0:
            metrics[cond]["tool_detection_rate"] = metrics[cond]["tool_correct"] / total
            metrics[cond]["full_accuracy"] = metrics[cond]["full_match"] / total
            metrics[cond]["false_positive_rate"] = metrics[cond]["false_positive"] / total

    return metrics


def run_statistical_tests(results: list[TrialResult]) -> dict:
    """Run statistical tests comparing conditions."""
    from collections import defaultdict

    # Pair results by (scenario_id, trial_number)
    paired = defaultdict(dict)
    for r in results:
        key = (r.scenario_id, r.trial_number)
        paired[key][r.condition] = r

    # McNemar's test: count discordant pairs
    # a = single correct, two_stage wrong
    # b = single wrong, two_stage correct
    a, b = 0, 0
    for key, pair in paired.items():
        if "single" not in pair or "two_stage" not in pair:
            continue
        single_correct = pair["single"].verdict == "MATCH"
        two_correct = pair["two_stage"].verdict == "MATCH"

        if single_correct and not two_correct:
            a += 1
        elif not single_correct and two_correct:
            b += 1

    # McNemar's test statistic
    if a + b > 0:
        chi2 = (abs(a - b) - 1) ** 2 / (a + b)
        # p-value from chi-squared distribution with 1 df
        # Using approximation: p ≈ exp(-chi2/2) for chi2 > 0
        import math
        p_value = math.exp(-chi2 / 2) if chi2 > 0 else 1.0
    else:
        chi2 = 0
        p_value = 1.0

    # Sign test at scenario level
    scenario_wins = {"single": 0, "two_stage": 0, "tie": 0}
    scenario_results = defaultdict(lambda: {"single": 0, "two_stage": 0})

    for r in results:
        if r.verdict == "MATCH":
            scenario_results[r.scenario_id][r.condition] += 1

    for scenario_id, scores in scenario_results.items():
        if scores["single"] > scores["two_stage"]:
            scenario_wins["single"] += 1
        elif scores["two_stage"] > scores["single"]:
            scenario_wins["two_stage"] += 1
        else:
            scenario_wins["tie"] += 1

    return {
        "mcnemar": {
            "single_better": a,
            "two_stage_better": b,
            "chi2": chi2,
            "p_value": p_value,
        },
        "sign_test": scenario_wins,
    }


def print_results(metrics: dict, stats: dict) -> None:
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("TWO-STAGE TOOL CALLING EXPERIMENT RESULTS")
    print("=" * 70)

    print("\nOVERALL ACCURACY")
    print("-" * 40)
    for cond in ["single", "two_stage"]:
        m = metrics[cond]
        print(f"{cond:12} | "
              f"Full accuracy: {m.get('full_accuracy', 0):.1%} | "
              f"Tool detection: {m.get('tool_detection_rate', 0):.1%} | "
              f"FP rate: {m.get('false_positive_rate', 0):.1%}")

    gap = (metrics["two_stage"].get("full_accuracy", 0) -
           metrics["single"].get("full_accuracy", 0))
    print(f"\nGap (two_stage - single): {gap:+.1%}")

    print("\nBY AMBIGUITY LEVEL")
    print("-" * 40)
    for amb in ["EXPLICIT", "IMPLICIT", "CONTROL"]:
        if amb not in metrics["by_ambiguity"]:
            continue
        amb_data = metrics["by_ambiguity"][amb]
        single_acc = (amb_data["single"]["full_match"] / amb_data["single"]["total"]
                      if amb_data["single"]["total"] > 0 else 0)
        two_acc = (amb_data["two_stage"]["full_match"] / amb_data["two_stage"]["total"]
                   if amb_data["two_stage"]["total"] > 0 else 0)
        print(f"{amb:12} | single: {single_acc:.1%} | two_stage: {two_acc:.1%} | gap: {two_acc - single_acc:+.1%}")

    print("\nBY TOOL DOMAIN")
    print("-" * 40)
    for dom in metrics["by_domain"]:
        dom_data = metrics["by_domain"][dom]
        single_acc = (dom_data["single"]["full_match"] / dom_data["single"]["total"]
                      if dom_data["single"]["total"] > 0 else 0)
        two_acc = (dom_data["two_stage"]["full_match"] / dom_data["two_stage"]["total"]
                   if dom_data["two_stage"]["total"] > 0 else 0)
        print(f"{dom:15} | single: {single_acc:.1%} | two_stage: {two_acc:.1%}")

    print("\nSTATISTICAL TESTS")
    print("-" * 40)
    mcnemar = stats["mcnemar"]
    print(f"McNemar's test:")
    print(f"  Single better: {mcnemar['single_better']}")
    print(f"  Two-stage better: {mcnemar['two_stage_better']}")
    print(f"  χ² = {mcnemar['chi2']:.2f}, p = {mcnemar['p_value']:.4f}")

    sign = stats["sign_test"]
    print(f"\nSign test (scenario level):")
    print(f"  Single wins: {sign['single']}")
    print(f"  Two-stage wins: {sign['two_stage']}")
    print(f"  Ties: {sign['tie']}")

    print("\n" + "=" * 70)


async def main():
    parser = argparse.ArgumentParser(
        description="Run two-stage tool calling experiment"
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run pilot with 5 scenarios, 2 trials"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Trials per scenario (default: 10)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=20,
        help="Maximum concurrent API calls (default: 20)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)

    # Select scenarios
    if args.pilot:
        # Pilot: 5 random scenarios (mix of types)
        scenarios = random.sample(ALL_TOOL_SCENARIOS, 4) + random.sample(ALL_CONTROL_SCENARIOS, 1)
        trials = 2
        logger.info("Running PILOT experiment")
    else:
        scenarios = ALL_SCENARIOS
        trials = args.trials
        logger.info("Running FULL experiment")

    # Print scenario stats
    stats = get_scenario_stats()
    logger.info(f"Scenarios: {len(scenarios)} total")
    logger.info(f"Trials per scenario: {trials}")
    logger.info(f"Total API calls: {len(scenarios) * trials * 3}")  # 3 = single + reasoner + extractor

    # Run experiment
    results = await run_experiment(
        scenarios=scenarios,
        trials_per_scenario=trials,
        max_concurrent=args.max_concurrent,
    )

    # Compute metrics
    metrics = compute_metrics(results)
    stat_tests = run_statistical_tests(results)

    # Print results
    print_results(metrics, stat_tests)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path("experiments/results") / f"two_stage_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "metadata": {
            "timestamp": timestamp,
            "model": "claude-sonnet-4-5-20250929",
            "pilot": args.pilot,
            "trials_per_scenario": trials,
            "num_scenarios": len(scenarios),
            "seed": args.seed,
        },
        "metrics": metrics,
        "statistical_tests": stat_tests,
        "results": [asdict(r) for r in results],
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
