"""
LLM Judge Scoring for Signal Detection Experiment

This script takes an existing results JSON file and adds judge scores to every trial.
The judge replaces regex as the primary scoring method for statistical claims.

Usage:
    python experiments/judge_scoring.py experiments/results/signal_detection_{timestamp}.json
"""

import asyncio
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

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
# Judge Prompt - FROZEN, do not modify without re-validation
# =============================================================================

JUDGE_PROMPT_TEMPLATE = """You are scoring whether an AI assistant's response acknowledged a specific signal in the user's message.

Signal type: {signal_type}
Signal definitions:
- frustration: The user is frustrated, annoyed, or exasperated with a technical situation
- urgency: The user is under time pressure or needs something handled quickly
- blocking_issue: The user is blocked, stuck, or unable to proceed on their work

User message:
"{query}"

Assistant response:
"{response}"

Did the assistant's response acknowledge or flag {signal_type} in the user's message?

Acknowledgment includes ANY of:
- Naming the signal ("I can see this is frustrating")
- Empathizing with it ("That must be difficult")
- Adjusting behavior because of it ("Let me prioritize this")
- Formally tagging it with XML

Acknowledgment does NOT include:
- Simply answering the technical question without addressing the signal
- Paraphrasing what the user said without recognizing the emotional/situational subtext
- Generic helpfulness ("I'd be happy to help") with no signal-specific language

Answer ONLY with YES or NO. Nothing else."""


@dataclass
class JudgeResult:
    """Result from a single judge call."""
    detected: Optional[bool]  # True=YES, False=NO, None=indeterminate
    raw_response: str
    error: Optional[str] = None


async def call_judge(
    query: str,
    response: str,
    signal_type: str,
    semaphore: asyncio.Semaphore,
) -> JudgeResult:
    """Call the LLM judge for a single response.

    Args:
        query: The user's original message
        response: The assistant's response to judge
        signal_type: The signal type being tested (frustration, urgency, blocking_issue)
        semaphore: Concurrency limiter

    Returns:
        JudgeResult with detected status and raw response
    """
    async with semaphore:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            signal_type=signal_type,
            query=query,
            response=response,
        )

        options = ClaudeAgentOptions(
            allowed_tools=[],
            max_turns=1,
            permission_mode="acceptEdits",
            system_prompt="You are a binary classifier. Answer only YES or NO.",
            model="claude-sonnet-4-5-20250929",
        )

        try:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(prompt)

                # Receive response
                raw_text = ""
                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                raw_text += block.text

            raw_text = raw_text.strip().upper()

            # Parse YES/NO
            if raw_text == "YES":
                return JudgeResult(detected=True, raw_response=raw_text)
            elif raw_text == "NO":
                return JudgeResult(detected=False, raw_response=raw_text)
            else:
                logger.warning(f"Indeterminate judge response: '{raw_text}'")
                return JudgeResult(detected=None, raw_response=raw_text)

        except Exception as e:
            logger.error(f"Judge call failed: {e}")
            return JudgeResult(detected=None, raw_response="", error=str(e))


async def score_trial(
    trial: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Score a single trial with the judge.

    Args:
        trial: Trial data dict containing query, response_text, signal_type, condition
        semaphore: Concurrency limiter

    Returns:
        Updated trial dict with judge scores added
    """
    query = trial.get("query", "")
    response = trial.get("response_text", "")
    signal_type = trial.get("signal_type", "")
    condition = trial.get("condition", "")

    # Handle None signal_type for CONTROL scenarios
    if signal_type is None:
        signal_type = "frustration"  # Default for controls - they should still say NO

    result = await call_judge(query, response, signal_type, semaphore)

    # Add judge result to trial
    trial_copy = trial.copy()

    if condition == "nl":
        trial_copy["nl_judge_detected"] = result.detected
        trial_copy["nl_judge_raw"] = result.raw_response
        # Preserve existing regex result
        trial_copy["nl_regex_detected"] = trial.get("detected", trial.get("nl_detected", None))
    else:  # structured
        trial_copy["st_judge_detected"] = result.detected
        trial_copy["st_judge_raw"] = result.raw_response
        # Preserve existing regex result
        trial_copy["st_regex_detected"] = trial.get("detected", trial.get("tool_called", None))

    return trial_copy


async def score_all_trials(
    results: list[dict],
    max_concurrent: int = 30,
) -> list[dict]:
    """Score all trials with the judge.

    Args:
        results: List of trial dicts
        max_concurrent: Maximum concurrent judge calls

    Returns:
        List of scored trial dicts
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    logger.info(f"Scoring {len(results)} trials with LLM judge (max {max_concurrent} concurrent)...")

    tasks = [score_trial(trial, semaphore) for trial in results]
    scored = await asyncio.gather(*tasks)

    return list(scored)


def merge_paired_trials(scored_trials: list[dict]) -> list[dict]:
    """Merge NL and structured trials for the same scenario/trial into unified records.

    The input has separate records for NL and structured conditions.
    The output has one record per trial with both conditions' judge results.
    """
    # Group by (scenario_id, trial_number)
    from collections import defaultdict
    grouped = defaultdict(dict)

    for trial in scored_trials:
        key = (trial.get("scenario_id"), trial.get("trial_number"))
        condition = trial.get("condition")

        if condition == "nl":
            grouped[key]["nl"] = trial
        else:
            grouped[key]["st"] = trial

    # Merge into unified records
    merged = []
    for key, pair in grouped.items():
        nl_trial = pair.get("nl", {})
        st_trial = pair.get("st", {})

        # Use NL trial as base, add structured judge results
        merged_record = nl_trial.copy()

        # Add structured condition results
        merged_record["st_judge_detected"] = st_trial.get("st_judge_detected")
        merged_record["st_judge_raw"] = st_trial.get("st_judge_raw")
        merged_record["st_regex_detected"] = st_trial.get("st_regex_detected")
        merged_record["st_response_text"] = st_trial.get("response_text")

        # Rename NL fields for clarity
        merged_record["nl_judge_detected"] = nl_trial.get("nl_judge_detected")
        merged_record["nl_judge_raw"] = nl_trial.get("nl_judge_raw")
        merged_record["nl_regex_detected"] = nl_trial.get("nl_regex_detected")
        merged_record["nl_response_text"] = nl_trial.get("response_text")

        merged.append(merged_record)

    return merged


def compute_summary_stats(merged_trials: list[dict]) -> dict:
    """Compute summary statistics from judge scores."""

    stats = {
        "total_trials": len(merged_trials),
        "judge_calls": len(merged_trials) * 2,
    }

    # Count by condition
    nl_yes = sum(1 for t in merged_trials if t.get("nl_judge_detected") is True)
    nl_no = sum(1 for t in merged_trials if t.get("nl_judge_detected") is False)
    nl_none = sum(1 for t in merged_trials if t.get("nl_judge_detected") is None)

    st_yes = sum(1 for t in merged_trials if t.get("st_judge_detected") is True)
    st_no = sum(1 for t in merged_trials if t.get("st_judge_detected") is False)
    st_none = sum(1 for t in merged_trials if t.get("st_judge_detected") is None)

    stats["nl"] = {"yes": nl_yes, "no": nl_no, "indeterminate": nl_none}
    stats["st"] = {"yes": st_yes, "no": st_no, "indeterminate": st_none}

    # Detection rates
    nl_total = nl_yes + nl_no
    st_total = st_yes + st_no

    if nl_total > 0:
        stats["nl"]["detection_rate"] = nl_yes / nl_total
    if st_total > 0:
        stats["st"]["detection_rate"] = st_yes / st_total

    # Agreement with regex
    nl_agree = sum(1 for t in merged_trials
                   if t.get("nl_judge_detected") is not None
                   and t.get("nl_regex_detected") is not None
                   and t.get("nl_judge_detected") == t.get("nl_regex_detected"))
    st_agree = sum(1 for t in merged_trials
                   if t.get("st_judge_detected") is not None
                   and t.get("st_regex_detected") is not None
                   and t.get("st_judge_detected") == t.get("st_regex_detected"))

    nl_comparable = sum(1 for t in merged_trials
                        if t.get("nl_judge_detected") is not None
                        and t.get("nl_regex_detected") is not None)
    st_comparable = sum(1 for t in merged_trials
                        if t.get("st_judge_detected") is not None
                        and t.get("st_regex_detected") is not None)

    if nl_comparable > 0:
        stats["nl"]["regex_agreement"] = nl_agree / nl_comparable
    if st_comparable > 0:
        stats["st"]["regex_agreement"] = st_agree / st_comparable

    return stats


async def main():
    parser = argparse.ArgumentParser(
        description="Add LLM judge scores to signal detection results"
    )
    parser.add_argument(
        "results_file",
        type=Path,
        help="Path to results JSON file"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=30,
        help="Maximum concurrent judge calls (default: 30)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate without making judge calls"
    )

    args = parser.parse_args()

    # Load results
    logger.info(f"Loading results from {args.results_file}")
    with open(args.results_file) as f:
        data = json.load(f)

    results = data.get("results", [])
    logger.info(f"Found {len(results)} trial records")

    if args.dry_run:
        logger.info("Dry run - not making judge calls")
        return

    # Score all trials
    scored = await score_all_trials(results, args.max_concurrent)

    # Merge paired trials
    merged = merge_paired_trials(scored)
    logger.info(f"Merged into {len(merged)} paired trial records")

    # Compute summary stats
    stats = compute_summary_stats(merged)
    logger.info(f"Judge summary: NL detection rate={stats['nl'].get('detection_rate', 'N/A'):.1%}, "
                f"ST detection rate={stats['st'].get('detection_rate', 'N/A'):.1%}")

    # Build output data
    output = {
        "metadata": data.get("metadata", {}),
        "judge_metadata": {
            "judge_model": "claude-sonnet-4-5-20250929",
            "judge_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "source_file": str(args.results_file),
        },
        "judge_summary": stats,
        "design": data.get("design", {}),
        "results": merged,
    }

    # Copy other top-level fields
    for key in data:
        if key not in output and key != "results":
            output[key] = data[key]

    # Write output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.results_file.parent / f"signal_detection_{timestamp}_judged.json"

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Wrote judged results to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("JUDGE SCORING COMPLETE")
    print("=" * 60)
    print(f"Total trials: {stats['total_trials']}")
    print(f"Judge calls: {stats['judge_calls']}")
    print(f"\nNL condition:")
    print(f"  Detection rate: {stats['nl'].get('detection_rate', 0):.1%}")
    print(f"  YES: {stats['nl']['yes']}, NO: {stats['nl']['no']}, Indeterminate: {stats['nl']['indeterminate']}")
    print(f"  Agreement with regex: {stats['nl'].get('regex_agreement', 0):.1%}")
    print(f"\nStructured condition:")
    print(f"  Detection rate: {stats['st'].get('detection_rate', 0):.1%}")
    print(f"  YES: {stats['st']['yes']}, NO: {stats['st']['no']}, Indeterminate: {stats['st']['indeterminate']}")
    print(f"  Agreement with regex: {stats['st'].get('regex_agreement', 0):.1%}")
    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
