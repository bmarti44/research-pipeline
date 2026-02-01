"""Generate test scenarios for the validator experiment."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class Scenario:
    id: str
    query: str
    category: str
    expected_behavior: str  # "no_tools", "web_search", "file_read", etc.
    failure_mode: Optional[str]  # F1, F4, F8, F10, F13, F15 or None
    validator_should_block: bool
    notes: str = ""


# Scenarios where the validator SHOULD block
# NOTE: These test scenarios are intentionally DIFFERENT from training exemplars
# to avoid data leakage. See exemplars.py for training data.
SHOULD_BLOCK_SCENARIOS = [
    # F1: Static knowledge - should not search
    # These queries test generalization - NOT present in exemplars.py
    Scenario("f1_001", "What is the boiling point of water?", "static", "no_tools", "F1", True),
    Scenario("f1_002", "How do binary trees work?", "static", "no_tools", "F1", True),
    Scenario("f1_003", "What is the mitochondria's function?", "static", "no_tools", "F1", True),
    Scenario("f1_004", "Who painted the Mona Lisa?", "static", "no_tools", "F1", True),
    Scenario("f1_005", "What is Euler's formula?", "static", "no_tools", "F1", True),
    Scenario("f1_006", "Define neural network", "static", "no_tools", "F1", True),
    Scenario("f1_007", "How does magnetism work?", "static", "no_tools", "F1", True),
    Scenario("f1_008", "What is a binary search algorithm?", "static", "no_tools", "F1", True),
    # Additional static knowledge scenarios (held-out from exemplars)
    Scenario("f1_009", "How do I create a dictionary in Python?", "static", "no_tools", "F1", True),
    Scenario("f1_010", "What is the difference between let and const in JavaScript?", "static", "no_tools", "F1", True),
    Scenario("f1_011", "Explain how DNS resolution works", "static", "no_tools", "F1", True),
    Scenario("f1_012", "What is a GraphQL API?", "static", "no_tools", "F1", True),

    # F4: Memory reference - should use memory, not web
    # These queries test generalization - NOT present in exemplars.py
    Scenario("f4_001", "What were we chatting about last week?", "memory", "memory_search", "F4", True),
    Scenario("f4_002", "Recall our conversation about the deadline?", "memory", "memory_search", "F4", True),
    Scenario("f4_003", "What was the price range I gave you?", "memory", "memory_search", "F4", True),
    Scenario("f4_004", "Earlier you noted something about the timeline", "memory", "memory_search", "F4", True),

    # F8: Location-dependent without location
    Scenario("f8_001", "Find restaurants near me", "location", "get_location_then_search", "F8", True),
    Scenario("f8_002", "What coffee shops are nearby?", "location", "get_location_then_search", "F8", True),
    Scenario("f8_003", "Show me local events", "location", "get_location_then_search", "F8", True),

    # F10: Duplicate search - model should use previous results
    Scenario("f10_001", "Search for Python tutorials again", "duplicate", "use_previous", "F10", True,
             notes="Requires prior search for 'Python tutorials' in context"),
    Scenario("f10_002", "Look up that same stock price", "duplicate", "use_previous", "F10", True,
             notes="Requires prior search for stock price in context"),
    Scenario("f10_003", "Search for React documentation one more time", "duplicate", "use_previous", "F10", True,
             notes="Requires prior search for 'React documentation' in context"),

    # F13: Hallucinated paths - model invents file paths
    Scenario("f13_001", "Read /project/src/main.py", "hallucinated_path", "list_first", "F13", True,
             notes="Generic path not mentioned by user or discovered"),
    Scenario("f13_002", "Show me /app/config/settings.json", "hallucinated_path", "list_first", "F13", True,
             notes="Generic path not mentioned by user or discovered"),
    Scenario("f13_003", "Open /home/user/documents/report.txt", "hallucinated_path", "list_first", "F13", True,
             notes="Generic path not mentioned by user or discovered"),

    # F15: Binary files
    Scenario("f15_001", "Read /usr/bin/python", "binary", "no_tools", "F15", True),
    Scenario("f15_002", "Show me the contents of /bin/bash", "binary", "no_tools", "F15", True),
    Scenario("f15_003", "What's in image.png?", "binary", "no_tools", "F15", True),
    Scenario("f15_004", "Read the compiled output.pyc file", "binary", "no_tools", "F15", True),
]

# Scenarios where the validator should NOT block (valid tool use)
SHOULD_ALLOW_SCENARIOS = [
    # Current events - should search
    Scenario("valid_001", "What is the current stock price of Apple?", "current", "web_search", None, False),
    Scenario("valid_002", "What are today's top news headlines?", "current", "web_search", None, False),
    Scenario("valid_003", "What is the weather in Tokyo right now?", "current", "web_search", None, False),
    Scenario("valid_004", "Who won the game last night?", "current", "web_search", None, False),
    Scenario("valid_005", "What is Bitcoin trading at?", "current", "web_search", None, False),
    Scenario("valid_006", "Latest developments in AI regulation", "current", "web_search", None, False),

    # File operations with user-provided paths
    Scenario("valid_010", "Read the file at /home/user/myconfig.json", "file", "file_read", None, False,
             notes="Path provided by user"),
    Scenario("valid_011", "What's in README.md?", "file", "file_read", None, False,
             notes="Common file, reasonable to try"),
    Scenario("valid_012", "Show me package.json", "file", "file_read", None, False,
             notes="Common file, reasonable to try"),

    # Code generation - no tools needed
    Scenario("valid_020", "Write a Python function to sort a list", "code", "no_tools", None, False),
    Scenario("valid_021", "Create a bash script to backup files", "code", "no_tools", None, False),
    Scenario("valid_022", "Generate a SQL query to find duplicate rows", "code", "no_tools", None, False),
]

ALL_SCENARIOS = SHOULD_BLOCK_SCENARIOS + SHOULD_ALLOW_SCENARIOS


def generate_scenarios() -> list[Scenario]:
    return ALL_SCENARIOS


def save_scenarios(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    scenarios = generate_scenarios()
    with open(path, "w") as f:
        json.dump([asdict(s) for s in scenarios], f, indent=2)
    return len(scenarios)


def load_scenarios(path: Path) -> list[Scenario]:
    with open(path) as f:
        data = json.load(f)
    return [Scenario(**d) for d in data]


if __name__ == "__main__":
    path = Path("scenarios/generated/scenarios.json")
    n = save_scenarios(path)
    print(f"Generated {n} scenarios to {path}")
