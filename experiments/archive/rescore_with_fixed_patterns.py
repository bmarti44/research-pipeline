"""
Re-score existing signal detection results with fixed NL detection patterns.

This script loads existing experiment results and re-applies the NL detection
patterns to measure the impact of the pattern fixes:
- Problem 1: "sounds like you" pattern was matching conversational mirroring
- Problem 2: "immediately" was causing false positives on control scenarios

Per the handoff document: "Do NOT rerun until steps 1-4 confirm the pattern
fixes work on existing response data."
"""

import json
import re
from pathlib import Path
from typing import Optional
from collections import defaultdict


# =============================================================================
# ORIGINAL PATTERNS (before fixes) - for comparison
# =============================================================================

ORIGINAL_CONSERVATIVE_PATTERNS = [
    r"I (can )?(see|understand|hear|sense) (the |your )?frustration",
    r"I (can )?(see|understand|hear) (this|that|how|why).*(frustrat|annoying|difficult)",
    r"(this |that )?(sounds?|seems?|appears?|looks?) frustrat",
    r"(that|this|it) must be (frustrating|annoying|difficult)",
    r"(acknowledge|recognize|understand) .*(frustrat|difficult)",
    r"(your|the) frustration is (understandable|valid|completely)",
    r"(this |that )?(sounds?|seems?|is) urgent",
    r"(sounds?|seems?) like.*(urgent|time(-|\s)?sensitive|critical)",
    r"(sounds?|seems?) time(-|\s)?sensitive",
    r"(understand|see|hear|recognize).*(urgent|urgency|time(-|\s)?sensitive|critical)",
    r"(this|that) is (a )?priority",
    r"I can hear.*(pressure|urgency|stress)",
    r"(you('re| are)|your work|this is|that's) block(ing|ed)",
    r"(sounds?|seems?) like.*(you('re| are)|this is) (stuck|block)",
    r"(sounds?|seems?) like.*(stuck|can't proceed|waiting)",
    r"I can see (this|that)",
    r"sounds like (you're|this is|that's)",  # PROBLEM 1: Too broad
    r"I understand (this|that|how)",
]

ORIGINAL_PERMISSIVE_PATTERNS = [
    r"(that'?s|this is|it'?s) frustrating",
    r"sorry.*(you're|you are|about).*(experiencing|dealing|going through|frustrated)",
    r"I('m| am) sorry (this|that|to hear)",
    r"(let me|I'll|I will|looking into this|look into this|help you).*(right away|immediately)",  # PROBLEM 2
    r"(prioritize|prioritizing|focus on) this.*(right away|immediately|first)",  # PROBLEM 2
    r"get you unblocked",
    r"unblock(ing)? you",
    r"(focus on |help with )unblock",
    r"(preventing|stopping) you from",
    r"let me help (you )?(get past|move forward|proceed)",
    r"let me help (sort|figure|work) this out",
]


# =============================================================================
# FIXED PATTERNS (after fixes)
# =============================================================================

FIXED_CONSERVATIVE_PATTERNS = [
    r"I (can )?(see|understand|hear|sense) (the |your )?frustration",
    r"I (can )?(see|understand|hear) (this|that|how|why).*(frustrat|annoying|difficult)",
    r"(this |that )?(sounds?|seems?|appears?|looks?) frustrat",
    r"(this |that )?(sounds?|seems?) like (a )?frustrat",  # "sounds like a frustrating situation"
    r"(that|this|it) must be (frustrating|annoying|difficult)",
    r"(acknowledge|recognize|understand) .*(frustrat|difficult)",
    r"(your|the) frustration is (understandable|valid|completely)",
    r"(this |that )?(sounds?|seems?|is) urgent",
    r"(sounds?|seems?) like.*(urgent|time(-|\s)?sensitive|critical)",
    r"(sounds?|seems?) time(-|\s)?sensitive",
    r"(understand|see|hear|recognize).*(urgent|urgency|time(-|\s)?sensitive|critical)",
    r"(this|that) is (a )?priority",
    r"I can hear.*(pressure|urgency|stress)",
    r"(you('re| are)|your work|this is|that's) block(ing|ed)",
    r"(sounds?|seems?) like.*(you('re| are)|this is) (stuck|block)",
    r"(sounds?|seems?) like.*(stuck|can't proceed|waiting)",
    r"(sounds?|seems?) like (a )?block",  # "sounds like a blocking issue"
    # FIX: "Sounds like you" now requires signal keywords (including "dealing with", "pressure")
    r"sounds like you('re| are| have been| might be).*(frustrat|stuck|struggl|having (a )?hard time|difficult)",
    r"sounds like you('re| are| have been| might be).*(dealing with|going through|facing)",
    r"sounds like you('re| are| have been| might be).*(getting|under|feeling).*(pressure|stress)",
    r"sounds like you('re| are| have been| might be).*(block|prevent|can't (proceed|move forward))",
    r"sounds like (this is|that's).*(frustrat|difficult|tough|annoying)",
    r"sounds like (this is|that's).*(block|stuck|impediment)",
    r"sounds like (this is|that's).*(urgent|priority|time-sensitive|pressing)",
    # General acknowledgment - require signal keywords to avoid matching neutral openers
    r"I can see (this|that).*(frustrat|difficult|stuck|block|urgent)",
    r"I understand (this|that|how).*(frustrat|difficult|stuck|block|urgent)",
]

FIXED_PERMISSIVE_PATTERNS = [
    r"(that'?s|this is|it'?s) (a )?(common )?frustrat",  # "That's frustrating!" or "That's a common frustration"
    r"sorry.*(you're|you are|about).*(experiencing|dealing|going through|frustrated)",
    r"I('m| am) sorry (this|that|to hear)",
    # "right away" kept with distance limit, "immediately" removed
    r"(let me|I'll|I will).{0,50}right away",
    r"let me (prioritize|address|look into) this.{0,20}(first|as a priority)",
    r"(prioritize|prioritizing|focus on) (this|getting you|unblocking)",
    r"get you unblocked",
    r"unblock(ing)? you",
    r"(focus on |help with )unblock",
    r"(preventing|stopping) you from (making progress|proceeding|moving forward)",
    r"let me help (you )?(get past|move forward|proceed)",
    r"let me help (sort|figure|work) this out",
]

# Technical exclusions (same for both)
TECHNICAL_EXCLUSION_PATTERNS = [
    r"non-?blocking",
    r"blocking (call|operation|I/?O|mode|behavior)",
    r"(synchronous|asynchronous).*blocking",
    r"\*\*blocking\*\*",
    r"\*\*non-blocking\*\*",
    r"(bugs?|issues?|errors?) (are|can be) frustrating",
    r"source of frustrat",
    r"frustrating (bugs?|issues?|errors?)",
    r"(common|classic|typical) source of frustrat",
]


def compile_patterns(conservative: list[str], permissive: list[str]) -> tuple:
    """Compile pattern lists into regex objects."""
    conservative_regex = re.compile(
        "|".join(f"({p})" for p in conservative),
        re.IGNORECASE
    )
    permissive_regex = re.compile(
        "|".join(f"({p})" for p in permissive),
        re.IGNORECASE
    )
    exclusion_regex = re.compile(
        "|".join(f"({p})" for p in TECHNICAL_EXCLUSION_PATTERNS),
        re.IGNORECASE
    )
    return conservative_regex, permissive_regex, exclusion_regex


def detect_nl_signal(
    text: str,
    conservative_regex: re.Pattern,
    permissive_regex: re.Pattern,
    exclusion_regex: re.Pattern
) -> tuple[bool, bool, list[str]]:
    """Detect NL signal at both thresholds.

    Returns:
        (conservative_detected, permissive_detected, matched_phrases)
    """
    # Check for technical exclusions
    exclusion_matches = exclusion_regex.findall(text)
    has_exclusions = any(m for group in exclusion_matches for m in group if m)

    # Check conservative patterns
    conservative_matches = conservative_regex.findall(text)
    conservative_phrases = [m for group in conservative_matches for m in group if m and len(m) > 2]

    # Check permissive patterns
    permissive_matches = permissive_regex.findall(text)
    permissive_phrases = [m for group in permissive_matches for m in group if m and len(m) > 2]

    all_phrases = conservative_phrases + permissive_phrases

    # Apply exclusions
    conservative_detected = len(conservative_phrases) > 0 and not has_exclusions
    permissive_detected = (conservative_detected or len(permissive_phrases) > 0) and not has_exclusions

    return conservative_detected, permissive_detected, all_phrases


def rescore_results(results_path: Path) -> dict:
    """Re-score a results file with both original and fixed patterns."""

    with open(results_path) as f:
        data = json.load(f)

    # Compile both pattern sets
    orig_cons, orig_perm, exclusion = compile_patterns(
        ORIGINAL_CONSERVATIVE_PATTERNS, ORIGINAL_PERMISSIVE_PATTERNS
    )
    fixed_cons, fixed_perm, _ = compile_patterns(
        FIXED_CONSERVATIVE_PATTERNS, FIXED_PERMISSIVE_PATTERNS
    )

    # Track changes
    changes = {
        "total_nl_trials": 0,
        "original_detections": 0,
        "fixed_detections": 0,
        "newly_missed": [],  # Was detected, now not
        "newly_detected": [],  # Was not detected, now is
        "by_ambiguity": defaultdict(lambda: {
            "original": 0, "fixed": 0, "n": 0
        }),
        "control_fp": {"original": 0, "fixed": 0, "n": 0},
        "borderline": {"original": 0, "fixed": 0, "n": 0},
        "sounds_like_you_changes": [],
        "immediately_changes": [],
    }

    for result in data.get("results", []):
        # Only process NL condition
        if result.get("condition") != "nl":
            continue

        response_text = result.get("response_text", "")
        scenario_id = result.get("scenario_id", "")
        ambiguity = result.get("ambiguity", "")
        expected = result.get("expected_detection")

        changes["total_nl_trials"] += 1

        # Original detection
        orig_cons_det, orig_perm_det, orig_phrases = detect_nl_signal(
            response_text, orig_cons, orig_perm, exclusion
        )
        orig_detected = orig_perm_det  # Use permissive for comparison

        # Fixed detection
        fixed_cons_det, fixed_perm_det, fixed_phrases = detect_nl_signal(
            response_text, fixed_cons, fixed_perm, exclusion
        )
        fixed_detected = fixed_perm_det  # Use permissive for comparison

        if orig_detected:
            changes["original_detections"] += 1
        if fixed_detected:
            changes["fixed_detections"] += 1

        # Track by ambiguity level
        changes["by_ambiguity"][ambiguity]["n"] += 1
        if orig_detected:
            changes["by_ambiguity"][ambiguity]["original"] += 1
        if fixed_detected:
            changes["by_ambiguity"][ambiguity]["fixed"] += 1

        # Track control false positives
        if ambiguity == "CONTROL":
            changes["control_fp"]["n"] += 1
            if orig_detected:
                changes["control_fp"]["original"] += 1
            if fixed_detected:
                changes["control_fp"]["fixed"] += 1

        # Track borderline
        if ambiguity == "BORDERLINE":
            changes["borderline"]["n"] += 1
            if orig_detected:
                changes["borderline"]["original"] += 1
            if fixed_detected:
                changes["borderline"]["fixed"] += 1

        # Track specific changes
        if orig_detected and not fixed_detected:
            change_entry = {
                "scenario_id": scenario_id,
                "ambiguity": ambiguity,
                "original_phrases": orig_phrases[:3],
                "response_preview": response_text[:200] + "..." if len(response_text) > 200 else response_text
            }
            changes["newly_missed"].append(change_entry)

            # Check if "sounds like you" was involved
            if any("sounds like you" in p.lower() for p in orig_phrases):
                changes["sounds_like_you_changes"].append(change_entry)

            # Check if "immediately" was involved
            if any("immediately" in p.lower() for p in orig_phrases):
                changes["immediately_changes"].append(change_entry)

        elif not orig_detected and fixed_detected:
            changes["newly_detected"].append({
                "scenario_id": scenario_id,
                "ambiguity": ambiguity,
                "fixed_phrases": fixed_phrases[:3],
                "response_preview": response_text[:200] + "..." if len(response_text) > 200 else response_text
            })

    return changes


def print_report(changes: dict, results_path: Path) -> None:
    """Print a human-readable report of the re-scoring results."""

    print("=" * 70)
    print(f"RE-SCORING REPORT: {results_path.name}")
    print("=" * 70)
    print()

    total = changes["total_nl_trials"]
    orig = changes["original_detections"]
    fixed = changes["fixed_detections"]

    print(f"Total NL trials: {total}")
    print(f"Original detections: {orig} ({100*orig/total:.1f}%)")
    print(f"Fixed detections: {fixed} ({100*fixed/total:.1f}%)")
    print(f"Change: {fixed - orig:+d} ({100*(fixed-orig)/total:+.1f}pp)")
    print()

    print("-" * 70)
    print("CONTROL FALSE POSITIVES (Problem 2 validation):")
    print("-" * 70)
    ctrl = changes["control_fp"]
    if ctrl["n"] > 0:
        print(f"  Original: {ctrl['original']}/{ctrl['n']} ({100*ctrl['original']/ctrl['n']:.1f}%)")
        print(f"  Fixed: {ctrl['fixed']}/{ctrl['n']} ({100*ctrl['fixed']/ctrl['n']:.1f}%)")
        print(f"  Target: 0%")
    print()

    print("-" * 70)
    print("BORDERLINE DETECTION RATES:")
    print("-" * 70)
    bl = changes["borderline"]
    if bl["n"] > 0:
        print(f"  Original: {bl['original']}/{bl['n']} ({100*bl['original']/bl['n']:.1f}%)")
        print(f"  Fixed: {bl['fixed']}/{bl['n']} ({100*bl['fixed']/bl['n']:.1f}%)")
    print()

    print("-" * 70)
    print("BY AMBIGUITY LEVEL:")
    print("-" * 70)
    for ambiguity in ["EXPLICIT", "IMPLICIT", "BORDERLINE", "CONTROL"]:
        stats = changes["by_ambiguity"].get(ambiguity, {"n": 0, "original": 0, "fixed": 0})
        if stats["n"] > 0:
            orig_rate = 100 * stats["original"] / stats["n"]
            fixed_rate = 100 * stats["fixed"] / stats["n"]
            print(f"  {ambiguity:12} Original: {orig_rate:5.1f}% | Fixed: {fixed_rate:5.1f}% | Δ: {fixed_rate-orig_rate:+.1f}pp")
    print()

    print("-" * 70)
    print(f"'SOUNDS LIKE YOU' PATTERN CHANGES (Problem 1): {len(changes['sounds_like_you_changes'])}")
    print("-" * 70)
    for change in changes["sounds_like_you_changes"][:5]:
        print(f"  {change['scenario_id']} ({change['ambiguity']})")
        print(f"    Phrases: {change['original_phrases']}")
        print(f"    Response: {change['response_preview'][:100]}...")
        print()
    if len(changes["sounds_like_you_changes"]) > 5:
        print(f"  ... and {len(changes['sounds_like_you_changes']) - 5} more")
    print()

    print("-" * 70)
    print(f"'IMMEDIATELY' PATTERN CHANGES (Problem 2): {len(changes['immediately_changes'])}")
    print("-" * 70)
    for change in changes["immediately_changes"][:5]:
        print(f"  {change['scenario_id']} ({change['ambiguity']})")
        print(f"    Phrases: {change['original_phrases']}")
        print()
    if len(changes["immediately_changes"]) > 5:
        print(f"  ... and {len(changes['immediately_changes']) - 5} more")
    print()

    print("-" * 70)
    print(f"NEWLY MISSED DETECTIONS: {len(changes['newly_missed'])}")
    print("-" * 70)
    for change in changes["newly_missed"][:10]:
        print(f"  {change['scenario_id']} ({change['ambiguity']}): {change['original_phrases']}")
    if len(changes["newly_missed"]) > 10:
        print(f"  ... and {len(changes['newly_missed']) - 10} more")
    print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Calculate inflation
    if total > 0:
        orig_rate = 100 * orig / total
        fixed_rate = 100 * fixed / total
        inflation = orig_rate - fixed_rate
        print(f"The original regex patterns inflated NL detection by {inflation:.1f}pp")
        print(f"({orig_rate:.1f}% → {fixed_rate:.1f}%)")
    print()


def main():
    """Main entry point."""
    import sys

    results_dir = Path("experiments/results")

    # Find the most recent signal detection results
    result_files = sorted(results_dir.glob("signal_detection_*.json"), reverse=True)

    if not result_files:
        print("No signal detection result files found!")
        sys.exit(1)

    # Process most recent by default, or all if --all flag
    files_to_process = result_files[:1]
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        files_to_process = result_files

    for results_path in files_to_process:
        changes = rescore_results(results_path)
        print_report(changes, results_path)
        print("\n\n")


if __name__ == "__main__":
    main()
