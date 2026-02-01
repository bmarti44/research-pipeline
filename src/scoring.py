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
