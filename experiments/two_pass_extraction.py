"""
Two-Pass Extraction Experiment

Evaluates whether a second model can extract structured signal information
from natural language responses, recovering "silent failures" where the
original model detected but didn't comply with structured output requirements.

This extends the signal detection experiment by showing that NL responses
can be post-processed to recover tool calls that would otherwise be lost.

Multi-Model Comparison:
- Sonnet: Frontier model extraction (ceiling)
- Qwen 7B: Off-the-shelf small model (cost argument)

Usage:
    # Run Sonnet extraction on judged results
    python experiments/two_pass_extraction.py <results_json> --model sonnet

    # Run Qwen 7B extraction (requires Ollama)
    python experiments/two_pass_extraction.py <results_json> --model qwen-7b

    # Run both models
    python experiments/two_pass_extraction.py <results_json> --model all

    # Pilot run (10 samples)
    python experiments/two_pass_extraction.py <results_json> --model sonnet --pilot
"""

import asyncio
import json
import argparse
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

import requests

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Model Configuration
# =============================================================================

MODELS = {
    "sonnet": {
        "type": "anthropic",
        "model": "claude-sonnet-4-5-20250929",
        "prompt_version": "v1",
    },
    "qwen-7b": {
        "type": "ollama",
        "model": "qwen2.5:7b-instruct",
        "prompt_version": "v1-7b",
    }
}

OLLAMA_BASE_URL = "http://localhost:11434"


# =============================================================================
# Extraction Prompts
# =============================================================================

# Standard extraction prompt for Sonnet
EXTRACTION_PROMPT_V1 = """Extract any signal acknowledgment from this assistant response.

Signal types (if present):
- frustration: User is frustrated, annoyed, or exasperated
- urgency: User is under time pressure
- blocking_issue: User is stuck or blocked on their work

If the assistant acknowledged frustration, urgency, or a blocking issue, output:
<signal type="TYPE">brief reason</signal>

If the assistant did NOT acknowledge any such signal, output:
NO_SIGNAL

Only extract signals the assistant actually acknowledged. Do not infer signals the assistant ignored.

Assistant response:
{response_text}"""

# Stricter extraction prompt for 7B models
EXTRACTION_PROMPT_V1_7B = """Extract any signal acknowledgment from this assistant response.

Rules:
1. If the assistant acknowledged frustration, urgency, or a blocking issue, output EXACTLY this format:
<signal type="TYPE">brief reason</signal>
Where TYPE is one of: frustration, urgency, blocking_issue

2. If the assistant did NOT acknowledge any such signal, output EXACTLY:
NO_SIGNAL

3. Output NOTHING else. No explanation. No preamble. Just the XML tag or NO_SIGNAL.

Only extract signals the assistant actually acknowledged. Do not infer signals the assistant ignored.

Assistant response:
\"\"\"
{response_text}
\"\"\""""


def get_extraction_prompt(response_text: str, prompt_version: str) -> str:
    """Get the extraction prompt for a given version."""
    if prompt_version == "v1":
        return EXTRACTION_PROMPT_V1.format(response_text=response_text)
    elif prompt_version == "v1-7b":
        return EXTRACTION_PROMPT_V1_7B.format(response_text=response_text)
    else:
        raise ValueError(f"Unknown prompt version: {prompt_version}")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExtractionResult:
    """Result of extracting a signal from an NL response."""
    # Input data
    scenario_id: str
    trial_number: int
    condition: str  # "nl" or "st" (which response we extracted from)
    response_text: str

    # Ground truth
    expected_signal_type: Optional[str]  # From scenario
    judge_detected: Optional[bool]  # Did the judge say the response acknowledged the signal?
    ambiguity: str

    # Extraction results
    extraction_model: str
    prompt_version: str
    raw_extraction: str
    extracted: bool  # Did we extract a signal?
    extracted_type: Optional[str]  # Signal type extracted
    extracted_reason: Optional[str]  # Reason text
    parse_error: bool = False

    # Timing
    latency_ms: int = 0
    error: Optional[str] = None


@dataclass
class ExtractionStats:
    """Statistics for extraction results."""
    model: str
    total: int = 0
    extracted: int = 0
    not_extracted: int = 0
    parse_errors: int = 0

    # Ground truth comparison
    true_positives: int = 0  # Extracted when judge said yes
    true_negatives: int = 0  # Not extracted when judge said no
    false_positives: int = 0  # Extracted when judge said no
    false_negatives: int = 0  # Not extracted when judge said yes

    # Signal type accuracy (among true positives)
    type_correct: int = 0
    type_incorrect: int = 0

    # Timing
    total_latency_ms: int = 0

    @property
    def extraction_rate(self) -> float:
        return self.extracted / self.total if self.total > 0 else 0

    @property
    def precision(self) -> float:
        tp_fp = self.true_positives + self.false_positives
        return self.true_positives / tp_fp if tp_fp > 0 else 0

    @property
    def recall(self) -> float:
        tp_fn = self.true_positives + self.false_negatives
        return self.true_positives / tp_fn if tp_fn > 0 else 0

    @property
    def type_accuracy(self) -> float:
        total = self.type_correct + self.type_incorrect
        return self.type_correct / total if total > 0 else 0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total if self.total > 0 else 0


# =============================================================================
# Parsing
# =============================================================================

def parse_extraction(raw: str) -> dict:
    """Parse extraction output from model.

    Returns dict with:
        extracted: bool
        signal_type: Optional[str]
        reason: Optional[str]
        raw: str
        parse_error: bool
    """
    raw = raw.strip()

    # Strip markdown code blocks if present
    raw = re.sub(r'^```\w*\n?', '', raw)
    raw = re.sub(r'\n?```$', '', raw)
    raw = raw.strip()

    # Check for signal XML
    match = re.search(
        r'<signal\s+type=["\']?(\w+)["\']?\s*>(.+?)</signal>',
        raw, re.IGNORECASE | re.DOTALL
    )
    if match:
        return {
            "extracted": True,
            "signal_type": match.group(1).lower().strip(),
            "reason": match.group(2).strip(),
            "raw": raw,
            "parse_error": False,
        }

    # Check for NO_SIGNAL (flexible matching)
    if re.search(r'no.?signal', raw, re.IGNORECASE):
        return {
            "extracted": False,
            "signal_type": None,
            "reason": None,
            "raw": raw,
            "parse_error": False,
        }

    # Neither — parse error (treat as no extraction, conservative)
    return {
        "extracted": False,
        "signal_type": None,
        "reason": None,
        "raw": raw,
        "parse_error": True,
    }


# =============================================================================
# Model Calls
# =============================================================================

async def call_sonnet(
    prompt: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, Optional[str]]:
    """Call Sonnet for extraction.

    Returns:
        Tuple of (response_text, error_message)
    """
    async with semaphore:
        options = ClaudeAgentOptions(
            allowed_tools=[],
            max_turns=1,
            permission_mode="acceptEdits",
            system_prompt="You extract structured signals from text. Output only the extraction result.",
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
            logger.error(f"Sonnet call failed: {e}")
            return "", str(e)


def call_ollama_sync(prompt: str, model: str) -> tuple[str, Optional[str]]:
    """Call Ollama model synchronously.

    Returns:
        Tuple of (response_text, error_message)
    """
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 150,
                }
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"].strip(), None

    except requests.exceptions.ConnectionError:
        return "", "Ollama not running. Start with: ollama serve"
    except requests.exceptions.Timeout:
        return "", "Ollama request timed out"
    except Exception as e:
        return "", str(e)


async def call_ollama(
    prompt: str,
    model: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str, Optional[str]]:
    """Call Ollama model asynchronously (wraps sync call)."""
    async with semaphore:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, call_ollama_sync, prompt, model)


# =============================================================================
# Extraction Runner
# =============================================================================

async def extract_single(
    trial: dict,
    model_config: dict,
    model_name: str,
    semaphore: asyncio.Semaphore,
    condition: str = "nl",
) -> ExtractionResult:
    """Run extraction on a single trial.

    Args:
        trial: Trial data from judged results
        model_config: Model configuration dict
        model_name: Name of extraction model
        semaphore: Concurrency limiter
        condition: Which response to extract from ("nl" or "st")
    """
    start_time = time.time()

    # Get response text based on condition
    if condition == "nl":
        response_text = trial.get("nl_response_text", trial.get("response_text", ""))
        judge_detected = trial.get("nl_judge_detected")
    else:
        response_text = trial.get("st_response_text", "")
        judge_detected = trial.get("st_judge_detected")

    # Build extraction prompt
    prompt = get_extraction_prompt(response_text, model_config["prompt_version"])

    # Call model
    if model_config["type"] == "anthropic":
        raw_extraction, error = await call_sonnet(prompt, semaphore)
    else:  # ollama
        raw_extraction, error = await call_ollama(prompt, model_config["model"], semaphore)

    latency_ms = int((time.time() - start_time) * 1000)

    # Parse extraction
    if error:
        parsed = {"extracted": False, "signal_type": None, "reason": None, "raw": "", "parse_error": True}
    else:
        parsed = parse_extraction(raw_extraction)

    return ExtractionResult(
        scenario_id=trial.get("scenario_id", ""),
        trial_number=trial.get("trial_number", 0),
        condition=condition,
        response_text=response_text,
        expected_signal_type=trial.get("signal_type"),
        judge_detected=judge_detected,
        ambiguity=trial.get("ambiguity", ""),
        extraction_model=model_name,
        prompt_version=model_config["prompt_version"],
        raw_extraction=raw_extraction if not error else "",
        extracted=parsed["extracted"],
        extracted_type=parsed["signal_type"],
        extracted_reason=parsed["reason"],
        parse_error=parsed["parse_error"],
        latency_ms=latency_ms,
        error=error,
    )


async def run_extraction(
    trials: list[dict],
    model_name: str,
    max_concurrent: int = 20,
    condition: str = "nl",
) -> list[ExtractionResult]:
    """Run extraction on all trials.

    Args:
        trials: List of trial dicts from judged results
        model_name: Name of extraction model to use
        max_concurrent: Max concurrent calls (1 for Ollama)
        condition: Which response to extract from

    Returns:
        List of ExtractionResult objects
    """
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

    model_config = MODELS[model_name]

    # Ollama processes one request at a time
    if model_config["type"] == "ollama":
        max_concurrent = 1

    semaphore = asyncio.Semaphore(max_concurrent)

    logger.info(f"Running extraction with {model_name} on {len(trials)} trials (max concurrent: {max_concurrent})")

    tasks = [
        extract_single(trial, model_config, model_name, semaphore, condition)
        for trial in trials
    ]

    results = await asyncio.gather(*tasks)
    return list(results)


# =============================================================================
# Analysis
# =============================================================================

def compute_stats(
    results: list[ExtractionResult],
    model_name: str,
) -> ExtractionStats:
    """Compute statistics from extraction results."""
    stats = ExtractionStats(model=model_name)

    for r in results:
        stats.total += 1
        stats.total_latency_ms += r.latency_ms

        if r.parse_error:
            stats.parse_errors += 1

        if r.extracted:
            stats.extracted += 1
        else:
            stats.not_extracted += 1

        # Ground truth comparison
        if r.judge_detected is True:
            if r.extracted:
                stats.true_positives += 1
                # Check signal type
                if r.extracted_type == r.expected_signal_type:
                    stats.type_correct += 1
                else:
                    stats.type_incorrect += 1
            else:
                stats.false_negatives += 1
        elif r.judge_detected is False:
            if r.extracted:
                stats.false_positives += 1
            else:
                stats.true_negatives += 1

    return stats


def compute_recovery_stats(
    nl_results: list[ExtractionResult],
    judged_trials: list[dict],
) -> dict:
    """Compute recovery statistics for silent failures.

    Silent failures = ST judge detected, ST regex did not detect
    Recovery = extraction from NL response recovers the signal
    """
    # Find silent failure trials
    silent_failure_ids = set()
    for trial in judged_trials:
        st_judge = trial.get("st_judge_detected")
        st_regex = trial.get("st_regex_detected")
        if st_judge is True and st_regex is not True:
            key = (trial.get("scenario_id"), trial.get("trial_number"))
            silent_failure_ids.add(key)

    # Count recoveries
    recovered = 0
    total_silent = len(silent_failure_ids)

    for r in nl_results:
        key = (r.scenario_id, r.trial_number)
        if key in silent_failure_ids and r.extracted:
            recovered += 1

    return {
        "total_silent_failures": total_silent,
        "recovered": recovered,
        "recovery_rate": recovered / total_silent if total_silent > 0 else 0,
    }


def print_report(
    stats: ExtractionStats,
    recovery: dict,
    condition: str,
) -> None:
    """Print formatted extraction report."""
    print("\n" + "=" * 70)
    print(f"TWO-PASS EXTRACTION: {stats.model.upper()}")
    print(f"Condition: {condition.upper()} responses")
    print("=" * 70)

    print(f"\nEXTRACTION RATES")
    print("-" * 40)
    print(f"Total trials:        {stats.total}")
    print(f"Extracted:           {stats.extracted} ({stats.extraction_rate:.1%})")
    print(f"Not extracted:       {stats.not_extracted}")
    print(f"Parse errors:        {stats.parse_errors} ({stats.parse_errors/stats.total:.1%})")

    print(f"\nGROUND TRUTH COMPARISON (vs judge)")
    print("-" * 40)
    print(f"True positives:      {stats.true_positives}")
    print(f"True negatives:      {stats.true_negatives}")
    print(f"False positives:     {stats.false_positives}")
    print(f"False negatives:     {stats.false_negatives}")
    print(f"Precision:           {stats.precision:.1%}")
    print(f"Recall:              {stats.recall:.1%}")

    print(f"\nSIGNAL TYPE ACCURACY")
    print("-" * 40)
    print(f"Correct type:        {stats.type_correct}")
    print(f"Incorrect type:      {stats.type_incorrect}")
    print(f"Type accuracy:       {stats.type_accuracy:.1%}")

    print(f"\nSILENT FAILURE RECOVERY")
    print("-" * 40)
    print(f"Total silent failures:  {recovery['total_silent_failures']}")
    print(f"Recovered:              {recovery['recovered']}")
    print(f"Recovery rate:          {recovery['recovery_rate']:.1%}")

    print(f"\nLATENCY")
    print("-" * 40)
    print(f"Total time:          {stats.total_latency_ms/1000:.1f}s")
    print(f"Avg per extraction:  {stats.avg_latency_ms:.0f}ms")

    print("\n" + "=" * 70)


def print_comparison_report(
    all_stats: dict[str, ExtractionStats],
    all_recovery: dict[str, dict],
    direct_st_compliance: float,
) -> None:
    """Print comparison report across models."""
    print("\n" + "=" * 70)
    print("TWO-PASS EXTRACTION: MODEL COMPARISON")
    print("=" * 70)

    print(f"\nEXTRACTION RATES (NL responses)")
    print("-" * 60)
    print(f"{'Model':<20} {'Extraction':<15} {'Precision':<15} {'Recall':<15}")
    print("-" * 60)
    for model_name, stats in all_stats.items():
        print(f"{model_name:<20} {stats.extraction_rate:.1%}{'':<10} "
              f"{stats.precision:.1%}{'':<10} {stats.recall:.1%}")
    print(f"{'Direct ST (XML)':<20} {direct_st_compliance:.1%}{'':<10} {'—':<15} {'—':<15}")

    print(f"\nSILENT FAILURE RECOVERY")
    print("-" * 60)
    print(f"{'Model':<20} {'Recovered':<20} {'Recovery Rate':<20}")
    print("-" * 60)
    for model_name, recovery in all_recovery.items():
        print(f"{model_name:<20} "
              f"{recovery['recovered']}/{recovery['total_silent_failures']:<12} "
              f"{recovery['recovery_rate']:.1%}")

    print(f"\nFALSE POSITIVE RATES (Control scenarios)")
    print("-" * 60)
    for model_name, stats in all_stats.items():
        fp_rate = stats.false_positives / (stats.true_negatives + stats.false_positives) \
            if (stats.true_negatives + stats.false_positives) > 0 else 0
        print(f"{model_name:<20} {fp_rate:.1%}")

    print(f"\nSIGNAL TYPE ACCURACY")
    print("-" * 60)
    for model_name, stats in all_stats.items():
        print(f"{model_name:<20} {stats.type_accuracy:.1%}")

    print(f"\nPARSE ERROR RATES")
    print("-" * 60)
    for model_name, stats in all_stats.items():
        print(f"{model_name:<20} {stats.parse_errors}/{stats.total} = {stats.parse_errors/stats.total:.1%}")

    print(f"\nLATENCY (avg per extraction)")
    print("-" * 60)
    for model_name, stats in all_stats.items():
        print(f"{model_name:<20} {stats.avg_latency_ms:.0f}ms")

    print("\n" + "=" * 70)


# =============================================================================
# Main
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Run two-pass extraction on judged signal detection results"
    )
    parser.add_argument(
        "results_file",
        type=Path,
        help="Path to judged results JSON file"
    )
    parser.add_argument(
        "--model",
        choices=["sonnet", "qwen-7b", "all"],
        default="sonnet",
        help="Extraction model to use (default: sonnet)"
    )
    parser.add_argument(
        "--condition",
        choices=["nl", "st", "both"],
        default="nl",
        help="Which responses to extract from (default: nl)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=20,
        help="Maximum concurrent API calls for Sonnet (default: 20)"
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run pilot with 10 samples only"
    )

    args = parser.parse_args()

    # Load judged results
    logger.info(f"Loading results from {args.results_file}")
    with open(args.results_file) as f:
        data = json.load(f)

    trials = data.get("results", [])
    logger.info(f"Found {len(trials)} trial records")

    # Pilot mode
    if args.pilot:
        trials = trials[:10]
        logger.info(f"Pilot mode: using {len(trials)} trials")

    # Determine models to run
    models_to_run = ["sonnet", "qwen-7b"] if args.model == "all" else [args.model]
    conditions_to_run = ["nl", "st"] if args.condition == "both" else [args.condition]

    # Store all results
    all_results = {}
    all_stats = {}
    all_recovery = {}

    # Run extraction for each model/condition combination
    for model_name in models_to_run:
        for condition in conditions_to_run:
            logger.info(f"\nRunning {model_name} extraction on {condition} responses...")

            results = await run_extraction(
                trials,
                model_name,
                max_concurrent=args.max_concurrent,
                condition=condition,
            )

            key = f"{model_name}_{condition}"
            all_results[key] = results

            # Compute stats
            stats = compute_stats(results, model_name)
            all_stats[key] = stats

            # Compute recovery (only for NL condition)
            if condition == "nl":
                recovery = compute_recovery_stats(results, trials)
                all_recovery[model_name] = recovery

            # Print individual report
            print_report(stats, recovery if condition == "nl" else {"total_silent_failures": 0, "recovered": 0, "recovery_rate": 0}, condition)

    # Print comparison report if multiple models
    if len(models_to_run) > 1 and "nl" in conditions_to_run:
        # Compute direct ST compliance rate
        st_regex_true = sum(1 for t in trials if t.get("st_regex_detected") is True)
        ground_truth_trials = [t for t in trials if t.get("ambiguity") in ("EXPLICIT", "IMPLICIT")]
        direct_st_compliance = st_regex_true / len(ground_truth_trials) if ground_truth_trials else 0

        print_comparison_report(
            {k: v for k, v in all_stats.items() if k.endswith("_nl")},
            all_recovery,
            direct_st_compliance,
        )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for key, results in all_results.items():
        model_name, condition = key.rsplit("_", 1)
        output_path = Path("experiments/results") / f"two_pass_{model_name.replace('-', '')}_{condition}_{timestamp}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "metadata": {
                "timestamp": timestamp,
                "extraction_model": model_name,
                "condition": condition,
                "prompt_version": MODELS[model_name]["prompt_version"],
                "source_file": str(args.results_file),
                "pilot": args.pilot,
                "num_trials": len(trials),
            },
            "stats": asdict(all_stats[key]),
            "recovery": all_recovery.get(model_name, {}),
            "results": [asdict(r) for r in results],
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Saved {key} results to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
