"""
Scenario definitions for proactive tool evaluation.

This module defines scenarios for testing whether NL intent expression
outperforms structured tool calling on vague user messages.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import json


class ToolType(Enum):
    MEMORY = "memory"           # Save information for future


class ExplicitnessLevel(Enum):
    IMPLICIT = 1        # Vague, no hint to save
    EXPLICIT = 5        # Direct command to save
    CONTROL = 0         # Should NOT trigger tool


@dataclass
class Scenario:
    id: str
    tool_type: ToolType
    level: ExplicitnessLevel
    query: str
    expected_action: bool
    expected_content: Optional[str] = None
    trigger_pattern: Optional[str] = None
    category: Optional[str] = None
    tags: list = field(default_factory=list)

    def to_dict(self):
        return {
            "id": self.id,
            "tool_type": self.tool_type.value,
            "level": self.level.value,
            "level_name": self.level.name,
            "query": self.query,
            "expected_action": self.expected_action,
            "expected_content": self.expected_content,
            "trigger_pattern": self.trigger_pattern,
            "category": self.category,
            "tags": self.tags,
        }


# =============================================================================
# MEMORY TOOL SCENARIOS - IMPLICIT (20 vague scenarios)
# =============================================================================

MEMORY_SCENARIOS = [
    # Original implicit (5) - very terse
    Scenario(
            id="mem_implicit_000",
            tool_type=ToolType.MEMORY,
            level=ExplicitnessLevel.IMPLICIT,
            query="app.py is in ./src",
            expected_action=True,
            expected_content="app.py in src folder",
            category="codebase",
        ),
    Scenario(
        id="mem_implicit_001",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="app.ts is in ./src",
        expected_action=True,
        expected_content="app.ts in src folder",
        category="codebase",
    ),
    Scenario(
        id="mem_implicit_002",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="pg is db",
        expected_action=True,
        expected_content="using Postgres",
        category="codebase",
    ),
    Scenario(
        id="mem_implicit_003",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="clerk is used for auth",
        expected_action=True,
        expected_content="Clerk for auth",
        category="codebase",
    ),
    Scenario(
        id="mem_implicit_004",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="rate limit is about 100 per minute",
        expected_action=True,
        expected_content="rate limit ~100/min",
        category="constraint",
    ),
    Scenario(
        id="mem_implicit_005",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="the config is in .env.local",
        expected_action=True,
        expected_content="config in .env.local",
        category="codebase",
    ),

    # Converted from weak (5) - made vague
    Scenario(
        id="mem_implicit_006",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="deploy uses CDK",
        expected_action=True,
        expected_content="deploy with CDK",
        category="codebase",
    ),
    Scenario(
        id="mem_implicit_007",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="prefer functional components",
        expected_action=True,
        expected_content="prefers functional React components",
        category="user_preference",
    ),
    Scenario(
        id="mem_implicit_008",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="tests in __tests__ folders",
        expected_action=True,
        expected_content="tests in __tests__ folders",
        category="codebase",
    ),
    Scenario(
        id="mem_implicit_009",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="two space indent",
        expected_action=True,
        expected_content="2-space indentation",
        category="codebase",
    ),
    Scenario(
        id="mem_implicit_010",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="dates are UTC in db",
        expected_action=True,
        expected_content="dates in UTC",
        category="codebase",
    ),

    # Converted from moderate (5) - made vague
    Scenario(
        id="mem_implicit_011",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="bcrypt for passwords, jwt for sessions",
        expected_action=True,
        expected_content="bcrypt for passwords, JWT for sessions",
        category="codebase",
    ),
    Scenario(
        id="mem_implicit_012",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="type hints everywhere, descriptive names",
        expected_action=True,
        expected_content="prefers type hints, descriptive names",
        category="user_preference",
    ),
    Scenario(
        id="mem_implicit_013",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="postgres for new db stuff",
        expected_action=True,
        expected_content="PostgreSQL for new DB ops",
        category="decision",
    ),
    Scenario(
        id="mem_implicit_014",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="staging is at staging.example.com",
        expected_action=True,
        expected_content="staging at staging.example.com",
        category="codebase",
    ),
    Scenario(
        id="mem_implicit_015",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="no console.log in prod",
        expected_action=True,
        expected_content="no console.log in production",
        category="codebase",
    ),

    # Converted from strong (5) - made vague
    Scenario(
        id="mem_implicit_016",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="ci/cd is in .github/workflows",
        expected_action=True,
        expected_content="CI/CD in .github/workflows",
        category="codebase",
    ),
    Scenario(
        id="mem_implicit_017",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="tabs not spaces",
        expected_action=True,
        expected_content="prefers tabs over spaces",
        category="user_preference",
    ),
    Scenario(
        id="mem_implicit_018",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="graphql for the new api",
        expected_action=True,
        expected_content="GraphQL for new API",
        category="decision",
    ),
    Scenario(
        id="mem_implicit_019",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="admin panel needs 2fa",
        expected_action=True,
        expected_content="admin panel requires 2FA",
        category="codebase",
    ),
    Scenario(
        id="mem_implicit_020",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="db connection is postgres://user:pass@host:5432/db",
        expected_action=True,
        expected_content="DB connection format",
        category="codebase",
    ),

    # =========================================================================
    # FILE PATH FAMILIARITY TEST - High vs Low familiarity
    # Testing hypothesis: familiar conventions trigger less verification behavior
    # =========================================================================

    # HIGH FAMILIARITY - Very common file names/paths across ecosystems
    Scenario(
        id="mem_filepath_high_001",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="index.js is in ./src",
        expected_action=True,
        expected_content="index.js in src",
        category="codebase",
        tags=["filepath", "high_familiarity"],
    ),
    Scenario(
        id="mem_filepath_high_002",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="main.py is in the root",
        expected_action=True,
        expected_content="main.py in root",
        category="codebase",
        tags=["filepath", "high_familiarity"],
    ),
    Scenario(
        id="mem_filepath_high_003",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="package.json is in root",
        expected_action=True,
        expected_content="package.json in root",
        category="codebase",
        tags=["filepath", "high_familiarity"],
    ),
    Scenario(
        id="mem_filepath_high_004",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="__init__.py is in ./src",
        expected_action=True,
        expected_content="__init__.py in src",
        category="codebase",
        tags=["filepath", "high_familiarity"],
    ),
    Scenario(
        id="mem_filepath_high_005",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="settings.py is in ./config",
        expected_action=True,
        expected_content="settings.py in config",
        category="codebase",
        tags=["filepath", "high_familiarity"],
    ),
    Scenario(
        id="mem_filepath_high_006",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="routes.py is in ./api",
        expected_action=True,
        expected_content="routes.py in api",
        category="codebase",
        tags=["filepath", "high_familiarity"],
    ),
    Scenario(
        id="mem_filepath_high_007",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="models.py is in ./app",
        expected_action=True,
        expected_content="models.py in app",
        category="codebase",
        tags=["filepath", "high_familiarity"],
    ),
    Scenario(
        id="mem_filepath_high_008",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="index.html is in ./public",
        expected_action=True,
        expected_content="index.html in public",
        category="codebase",
        tags=["filepath", "high_familiarity"],
    ),
    Scenario(
        id="mem_filepath_high_009",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="Dockerfile is in root",
        expected_action=True,
        expected_content="Dockerfile in root",
        category="codebase",
        tags=["filepath", "high_familiarity"],
    ),
    Scenario(
        id="mem_filepath_high_010",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="README.md is in root",
        expected_action=True,
        expected_content="README.md in root",
        category="codebase",
        tags=["filepath", "high_familiarity"],
    ),

    # LOW FAMILIARITY - Uncommon/custom file names, unusual paths
    Scenario(
        id="mem_filepath_low_001",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="orchestrator.py is in ./core",
        expected_action=True,
        expected_content="orchestrator.py in core",
        category="codebase",
        tags=["filepath", "low_familiarity"],
    ),
    Scenario(
        id="mem_filepath_low_002",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="dispatcher.ts is in ./lib",
        expected_action=True,
        expected_content="dispatcher.ts in lib",
        category="codebase",
        tags=["filepath", "low_familiarity"],
    ),
    Scenario(
        id="mem_filepath_low_003",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="coordinator.go is in ./internal",
        expected_action=True,
        expected_content="coordinator.go in internal",
        category="codebase",
        tags=["filepath", "low_familiarity"],
    ),
    Scenario(
        id="mem_filepath_low_004",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="bootstrap.php is in ./app",
        expected_action=True,
        expected_content="bootstrap.php in app",
        category="codebase",
        tags=["filepath", "low_familiarity"],
    ),
    Scenario(
        id="mem_filepath_low_005",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="entrypoint.rs is in ./src",
        expected_action=True,
        expected_content="entrypoint.rs in src",
        category="codebase",
        tags=["filepath", "low_familiarity"],
    ),
    Scenario(
        id="mem_filepath_low_006",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="processor.scala is in ./app",
        expected_action=True,
        expected_content="processor.scala in app",
        category="codebase",
        tags=["filepath", "low_familiarity"],
    ),
    Scenario(
        id="mem_filepath_low_007",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="handler.ex is in ./lib",
        expected_action=True,
        expected_content="handler.ex in lib",
        category="codebase",
        tags=["filepath", "low_familiarity"],
    ),
    Scenario(
        id="mem_filepath_low_008",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="mediator.kt is in ./src",
        expected_action=True,
        expected_content="mediator.kt in src",
        category="codebase",
        tags=["filepath", "low_familiarity"],
    ),
    Scenario(
        id="mem_filepath_low_009",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="aggregator.rb is in ./services",
        expected_action=True,
        expected_content="aggregator.rb in services",
        category="codebase",
        tags=["filepath", "low_familiarity"],
    ),
    Scenario(
        id="mem_filepath_low_010",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="reconciler.zig is in ./core",
        expected_action=True,
        expected_content="reconciler.zig in core",
        category="codebase",
        tags=["filepath", "low_familiarity"],
    ),

    # EXPLICIT (5) - Direct commands (for sanity check)
    Scenario(
        id="mem_explicit_001",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.EXPLICIT,
        query="Save this to memory: we use Stripe for payments with webhook endpoint at /api/webhooks/stripe.",
        expected_action=True,
        expected_content="Stripe payments, webhook at /api/webhooks/stripe",
        trigger_pattern="save this to memory",
        category="codebase",
    ),
    Scenario(
        id="mem_explicit_002",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.EXPLICIT,
        query="Please remember this decision: we're using Tailwind CSS instead of styled-components.",
        expected_action=True,
        expected_content="Tailwind CSS (not styled-components)",
        trigger_pattern="please remember this",
        category="decision",
    ),
    Scenario(
        id="mem_explicit_003",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.EXPLICIT,
        query="Add to your notes: my timezone is America/Los_Angeles.",
        expected_action=True,
        expected_content="timezone America/Los_Angeles",
        trigger_pattern="add to your notes",
        category="user_preference",
    ),
    Scenario(
        id="mem_explicit_004",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.EXPLICIT,
        query="Record this: the production database is read-only replicated to analytics-db.",
        expected_action=True,
        expected_content="prod DB replicated to analytics-db (read-only)",
        trigger_pattern="record this",
        category="codebase",
    ),
    Scenario(
        id="mem_explicit_005",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.EXPLICIT,
        query="Use your memory tool to save: error logs go to Sentry, project ID is abc123.",
        expected_action=True,
        expected_content="Sentry for errors, project abc123",
        trigger_pattern="use your memory tool",
        category="codebase",
    ),
]


# =============================================================================
# CONTROL SCENARIOS (5) - Should NOT trigger any tool
# =============================================================================

CONTROL_SCENARIOS = [
    # One from each category for diversity
    Scenario(
        id="ctrl_question_001",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.CONTROL,
        query="What is 2 + 2?",
        expected_action=False,
        tags=["simple_question"],
    ),
    Scenario(
        id="ctrl_action_001",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.CONTROL,
        query="Please write a function that sorts an array.",
        expected_action=False,
        tags=["action_request"],
    ),
    Scenario(
        id="ctrl_temp_001",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.CONTROL,
        query="I'm just testing something real quick.",
        expected_action=False,
        tags=["temporary"],
    ),
    Scenario(
        id="ctrl_chat_001",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.CONTROL,
        query="Thanks for your help!",
        expected_action=False,
        tags=["chitchat"],
    ),
    Scenario(
        id="ctrl_known_001",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.CONTROL,
        query="As I mentioned before, we use React.",
        expected_action=False,
        tags=["already_known"],
    ),
]


# =============================================================================
# ALL SCENARIOS
# =============================================================================

ALL_SCENARIOS = MEMORY_SCENARIOS + CONTROL_SCENARIOS


def get_scenarios_by_level(level: ExplicitnessLevel) -> list[Scenario]:
    """Get all scenarios at a specific explicitness level."""
    return [s for s in ALL_SCENARIOS if s.level == level]


def export_scenarios(filepath: str):
    """Export all scenarios to JSON."""
    data = {
        "total": len(ALL_SCENARIOS),
        "by_level": {
            level.name: len(get_scenarios_by_level(level))
            for level in ExplicitnessLevel
        },
        "scenarios": [s.to_dict() for s in ALL_SCENARIOS],
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    print(f"Total scenarios: {len(ALL_SCENARIOS)}")
    print(f"\nBy explicitness level:")
    for level in ExplicitnessLevel:
        count = len(get_scenarios_by_level(level))
        if count > 0:
            print(f"  {level.name}: {count}")
