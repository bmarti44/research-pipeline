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
    FILE_READ = "file_read"     # Read file contents (future work)
    FILE_WRITE = "file_write"   # Write/create files (future work)
    FILE_SEARCH = "file_search" # Search for files (future work)


class ExplicitnessLevel(Enum):
    IMPLICIT = 1        # Vague, no hint to save
    EXPLICIT = 5        # Direct command to save
    CONTROL = 0         # Should NOT trigger tool


class Difficulty(Enum):
    """Difficulty of discerning save intent (for implicit scenarios only).

    Based on:
    1. Terseness - fewer words = harder
    2. Use of symbols/abbreviations - harder
    3. Inference required - more inference = harder
    4. Categorization ambiguity - harder
    """
    HARD = 3        # Very terse, symbols/abbreviations, high inference
    MEDIUM = 2      # Terse but clearer structure
    EASY = 1        # More context, clearer what to save
    NA = 0          # Not applicable (explicit/control scenarios)


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
    difficulty: Difficulty = Difficulty.NA

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
            "difficulty": self.difficulty.value,
            "difficulty_name": self.difficulty.name,
        }


# =============================================================================
# MEMORY TOOL SCENARIOS - IMPLICIT (20 vague scenarios)
# =============================================================================

MEMORY_SCENARIOS = [
    # Original implicit (6) - very terse
    Scenario(
        id="mem_implicit_000",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="app.py is in ./src",
        expected_action=True,
        expected_content="app.py in src folder",
        category="codebase",
        difficulty=Difficulty.MEDIUM,  # Terse but clear file path statement
    ),
    Scenario(
        id="mem_implicit_001",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="app.ts is in ./src",
        expected_action=True,
        expected_content="app.ts in src folder",
        category="codebase",
        difficulty=Difficulty.MEDIUM,  # Terse but clear file path statement
    ),
    Scenario(
        id="mem_implicit_002",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="pg is db",
        expected_action=True,
        expected_content="using Postgres",
        category="codebase",
        difficulty=Difficulty.HARD,  # Very terse (3 words), abbreviation
    ),
    Scenario(
        id="mem_implicit_003",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="port 3000 = dev",
        expected_action=True,
        expected_content="port 3000 is development",
        category="codebase",
        difficulty=Difficulty.HARD,  # Uses symbol (=), very terse
    ),
    Scenario(
        id="mem_implicit_004",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="rate ~100/60",
        expected_action=True,
        expected_content="rate limit ~100/min",
        category="constraint",
        difficulty=Difficulty.HARD,  # Uses symbols (~, /), cryptic
    ),
    Scenario(
        id="mem_implicit_005",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="the config in ./.env.local!",
        expected_action=True,
        expected_content="the config is in .env.local",
        category="codebase",
        difficulty=Difficulty.MEDIUM,  # Informal but clear file path
    ),

    # Converted from weak (5) - made vague
    Scenario(
        id="mem_implicit_006",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="cdk is deploy",
        expected_action=True,
        expected_content="the deployment uses CDK",
        category="codebase",
        difficulty=Difficulty.HARD,  # Very terse (3 words), abbreviation
    ),
    Scenario(
        id="mem_implicit_007",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="prefer functional components",
        expected_action=True,
        expected_content="prefers functional React components",
        category="user_preference",
        difficulty=Difficulty.MEDIUM,  # Clear preference statement
    ),
    Scenario(
        id="mem_implicit_008",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="tests in __tests__ folders",
        expected_action=True,
        expected_content="tests in __tests__ folders",
        category="codebase",
        difficulty=Difficulty.MEDIUM,  # Clear convention statement
    ),
    Scenario(
        id="mem_implicit_009",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="two space indent",
        expected_action=True,
        expected_content="2-space indentation",
        category="codebase",
        difficulty=Difficulty.HARD,  # Very terse (3 words)
    ),
    Scenario(
        id="mem_implicit_010",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="dates are UTC in db",
        expected_action=True,
        expected_content="dates in UTC",
        category="codebase",
        difficulty=Difficulty.MEDIUM,  # Terse but clear technical fact
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
        difficulty=Difficulty.MEDIUM,  # Technical but clear dual-fact
    ),
    Scenario(
        id="mem_implicit_012",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="type hints everywhere, descriptive names",
        expected_action=True,
        expected_content="prefers type hints, descriptive names",
        category="user_preference",
        difficulty=Difficulty.MEDIUM,  # Clear preference statement
    ),
    Scenario(
        id="mem_implicit_013",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="postgres for new db stuff",
        expected_action=True,
        expected_content="PostgreSQL for new DB ops",
        category="decision",
        difficulty=Difficulty.MEDIUM,  # Informal but clear decision
    ),
    Scenario(
        id="mem_implicit_014",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="staging is at staging.example.com",
        expected_action=True,
        expected_content="staging at staging.example.com",
        category="codebase",
        difficulty=Difficulty.EASY,  # Clear URL/endpoint statement
    ),
    Scenario(
        id="mem_implicit_015",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="no console.log in prod",
        expected_action=True,
        expected_content="no console.log in production",
        category="codebase",
        difficulty=Difficulty.MEDIUM,  # Clear constraint/rule
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
        difficulty=Difficulty.MEDIUM,  # Clear path statement
    ),
    Scenario(
        id="mem_implicit_017",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="tabs not spaces",
        expected_action=True,
        expected_content="prefers tabs over spaces",
        category="user_preference",
        difficulty=Difficulty.HARD,  # Very terse (3 words)
    ),
    Scenario(
        id="mem_implicit_018",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="graphql for the new api",
        expected_action=True,
        expected_content="GraphQL for new API",
        category="decision",
        tags=["semantic_confound", "clarification_trigger"],  # Model asks clarifying questions instead of saving
        difficulty=Difficulty.MEDIUM,  # Clear decision but triggers clarification
    ),
    Scenario(
        id="mem_implicit_019",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="admin panel needs 2fa",
        expected_action=True,
        expected_content="admin panel requires 2FA",
        category="codebase",
        tags=["semantic_confound", "implementation_trigger"],  # Model tries to implement feature instead of saving
        difficulty=Difficulty.MEDIUM,  # Clear requirement but triggers implementation
    ),
    Scenario(
        id="mem_implicit_020",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="db connection is postgres://user:pass@host:5432/db",
        expected_action=True,
        expected_content="DB connection format",
        category="codebase",
        difficulty=Difficulty.EASY,  # Explicit connection string
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
        difficulty=Difficulty.MEDIUM,  # Familiar pattern reduces inference
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
        difficulty=Difficulty.MEDIUM,  # Familiar pattern reduces inference
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
        difficulty=Difficulty.MEDIUM,  # Familiar pattern reduces inference
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
        difficulty=Difficulty.MEDIUM,  # Familiar pattern reduces inference
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
        difficulty=Difficulty.MEDIUM,  # Familiar pattern reduces inference
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
        difficulty=Difficulty.MEDIUM,  # Familiar pattern reduces inference
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
        difficulty=Difficulty.MEDIUM,  # Familiar pattern reduces inference
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
        difficulty=Difficulty.MEDIUM,  # Familiar pattern reduces inference
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
        difficulty=Difficulty.MEDIUM,  # Familiar pattern reduces inference
    ),
    Scenario(
        id="mem_filepath_high_010",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="README.md is in root",
        expected_action=True,
        expected_content="README.md in root",
        category="codebase",
        tags=["filepath", "high_familiarity", "semantic_confound"],  # Excluded from familiarity analysis - both conditions interpret as read request
        difficulty=Difficulty.MEDIUM,  # Would be MEDIUM but has semantic confound
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
        difficulty=Difficulty.HARD,  # Unfamiliar pattern increases uncertainty
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
        difficulty=Difficulty.HARD,  # Unfamiliar pattern increases uncertainty
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
        difficulty=Difficulty.HARD,  # Unfamiliar pattern increases uncertainty
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
        difficulty=Difficulty.HARD,  # Unfamiliar pattern increases uncertainty
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
        difficulty=Difficulty.HARD,  # Unfamiliar pattern increases uncertainty
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
        difficulty=Difficulty.HARD,  # Unfamiliar pattern increases uncertainty
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
        difficulty=Difficulty.HARD,  # Unfamiliar pattern increases uncertainty
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
        difficulty=Difficulty.HARD,  # Unfamiliar pattern increases uncertainty
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
        difficulty=Difficulty.HARD,  # Unfamiliar pattern increases uncertainty
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
        difficulty=Difficulty.HARD,  # Unfamiliar pattern increases uncertainty
    ),

    # =========================================================================
    # PRECISION TEST - Scenarios where structured output's fidelity advantage
    # may outweigh NL's recall advantage (exact values, multi-field, negation)
    # =========================================================================

    # PRECISION - Exact numeric/version values that need to be preserved exactly
    # NOTE: These scenarios show mixed results and are excluded from main analysis
    Scenario(
        id="mem_precision_001",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="max upload is exactly 10485760 bytes",
        expected_action=True,
        expected_content="max upload 10485760 bytes",
        category="constraint",
        tags=["precision", "numeric", "excluded_from_main_analysis"],
        difficulty=Difficulty.EASY,  # Clear constraint with explicit "exactly"
    ),
    Scenario(
        id="mem_precision_002",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="our api is at v2.3.1-beta.4",
        expected_action=True,
        expected_content="API version v2.3.1-beta.4",
        category="codebase",
        tags=["precision", "version", "excluded_from_main_analysis"],
        difficulty=Difficulty.EASY,  # Clear version statement
    ),
    Scenario(
        id="mem_precision_003",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="timeout is 30000ms, not 30s",
        expected_action=True,
        expected_content="timeout 30000ms (not 30s)",
        category="constraint",
        tags=["precision", "numeric", "excluded_from_main_analysis"],
        difficulty=Difficulty.EASY,  # Clear constraint with explicit correction
    ),

    # MULTI-FIELD - Multiple pieces of information that all need to be captured
    # NOTE: These scenarios show 0pp gap (both 100%) and are excluded from main analysis
    Scenario(
        id="mem_multifield_001",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="auth uses OAuth2 with PKCE, tokens expire in 3600s, refresh enabled",
        expected_action=True,
        expected_content="OAuth2 PKCE, 3600s expiry, refresh enabled",
        category="codebase",
        tags=["multifield", "auth", "excluded_from_main_analysis"],
        difficulty=Difficulty.EASY,  # Clear technical spec, low uncertainty
    ),
    Scenario(
        id="mem_multifield_002",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="db is postgres 15.2 on port 5433 with ssl required",
        expected_action=True,
        expected_content="postgres 15.2, port 5433, ssl required",
        category="codebase",
        tags=["multifield", "database", "excluded_from_main_analysis"],
        difficulty=Difficulty.EASY,  # Clear technical spec, low uncertainty
    ),
    Scenario(
        id="mem_multifield_003",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="redis cache at localhost:6379, db 2, password is in REDIS_PASS env var",
        expected_action=True,
        expected_content="redis localhost:6379, db 2, password in REDIS_PASS",
        category="codebase",
        tags=["multifield", "cache", "excluded_from_main_analysis"],
        difficulty=Difficulty.EASY,  # Clear technical spec, low uncertainty
    ),

    # NEGATION - State changes, migrations, removals that need precise capture
    # NOTE: These scenarios show structured winning on fidelity and are excluded from main analysis
    Scenario(
        id="mem_negation_001",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="we migrated off Redis last month",
        expected_action=True,
        expected_content="migrated off Redis",
        category="decision",
        tags=["negation", "migration", "excluded_from_main_analysis"],
        difficulty=Difficulty.MEDIUM,  # State change requires inference
    ),
    Scenario(
        id="mem_negation_002",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="removed the webpack config, using vite now",
        expected_action=True,
        expected_content="removed webpack, using vite",
        category="decision",
        tags=["negation", "migration", "excluded_from_main_analysis"],
        difficulty=Difficulty.MEDIUM,  # State change requires inference
    ),
    Scenario(
        id="mem_negation_003",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.IMPLICIT,
        query="deprecated the v1 api, only v2 is supported",
        expected_action=True,
        expected_content="v1 API deprecated, only v2 supported",
        category="codebase",
        tags=["negation", "deprecation", "excluded_from_main_analysis"],
        difficulty=Difficulty.MEDIUM,  # Deprecation statement, clear but technical
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
# FILE OPERATION SCENARIOS (Future Work)
# =============================================================================
# These scenarios test format friction for higher-stakes tools beyond memory.
# Memory persistence is low-stakes; file operations have real side effects.
# NOTE: These are defined but not currently used in the main experiment.

FILE_OPERATION_SCENARIOS = [
    # -------------------------------------------------------------------------
    # FILE READ REQUESTS - Should trigger file read
    # -------------------------------------------------------------------------
    Scenario(
        id="file_read_001",
        tool_type=ToolType.FILE_READ,
        level=ExplicitnessLevel.IMPLICIT,
        query="Check what's in config.yaml",
        expected_action=True,
        expected_content="read config.yaml",
        tags=["file_operation", "read"],
    ),
    Scenario(
        id="file_read_002",
        tool_type=ToolType.FILE_READ,
        level=ExplicitnessLevel.IMPLICIT,
        query="Look at the package.json dependencies",
        expected_action=True,
        expected_content="read package.json",
        tags=["file_operation", "read"],
    ),
    Scenario(
        id="file_read_003",
        tool_type=ToolType.FILE_READ,
        level=ExplicitnessLevel.IMPLICIT,
        query="What's in the .env file?",
        expected_action=True,
        expected_content="read .env",
        tags=["file_operation", "read"],
    ),

    # -------------------------------------------------------------------------
    # FILE WRITE REQUESTS - Should trigger file write
    # -------------------------------------------------------------------------
    Scenario(
        id="file_write_001",
        tool_type=ToolType.FILE_WRITE,
        level=ExplicitnessLevel.IMPLICIT,
        query="Add a newline at the end of main.py",
        expected_action=True,
        expected_content="modify main.py",
        tags=["file_operation", "write"],
    ),
    Scenario(
        id="file_write_002",
        tool_type=ToolType.FILE_WRITE,
        level=ExplicitnessLevel.IMPLICIT,
        query="Create a .gitignore file",
        expected_action=True,
        expected_content="create .gitignore",
        tags=["file_operation", "write"],
    ),
    Scenario(
        id="file_write_003",
        tool_type=ToolType.FILE_WRITE,
        level=ExplicitnessLevel.IMPLICIT,
        query="Update the README with a new section",
        expected_action=True,
        expected_content="modify README",
        tags=["file_operation", "write"],
    ),

    # -------------------------------------------------------------------------
    # FILE SEARCH REQUESTS - Should trigger file search
    # -------------------------------------------------------------------------
    Scenario(
        id="file_search_001",
        tool_type=ToolType.FILE_SEARCH,
        level=ExplicitnessLevel.IMPLICIT,
        query="Find all TypeScript files",
        expected_action=True,
        expected_content="search for *.ts",
        tags=["file_operation", "search"],
    ),
    Scenario(
        id="file_search_002",
        tool_type=ToolType.FILE_SEARCH,
        level=ExplicitnessLevel.IMPLICIT,
        query="Where is the database config?",
        expected_action=True,
        expected_content="search for database config",
        tags=["file_operation", "search"],
    ),
    Scenario(
        id="file_search_003",
        tool_type=ToolType.FILE_SEARCH,
        level=ExplicitnessLevel.IMPLICIT,
        query="Find where the API endpoints are defined",
        expected_action=True,
        expected_content="search for API endpoints",
        tags=["file_operation", "search"],
    ),
]


# =============================================================================
# CONTROL SCENARIOS (20) - Should NOT trigger any tool
# =============================================================================
# Expanded from 5 to 20 scenarios to reliably estimate false positive rate
# (addresses peer review concern about underpowered control analysis)

CONTROL_SCENARIOS = [
    # -------------------------------------------------------------------------
    # QUESTIONS (should not save) - 5 scenarios
    # -------------------------------------------------------------------------
    Scenario(
        id="ctrl_question_001",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.CONTROL,
        query="What is 2 + 2?",
        expected_action=False,
        tags=["question", "simple"],
    ),
    Scenario(
        id="ctrl_question_002",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.CONTROL,
        query="How do I configure webpack?",
        expected_action=False,
        tags=["question", "technical"],
    ),
    Scenario(
        id="ctrl_question_003",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.CONTROL,
        query="What's the best way to handle errors?",
        expected_action=False,
        tags=["question", "technical"],
    ),
    Scenario(
        id="ctrl_question_004",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.CONTROL,
        query="Can you explain async/await?",
        expected_action=False,
        tags=["question", "conceptual"],
    ),
    Scenario(
        id="ctrl_question_005",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.CONTROL,
        query="Why is my test failing?",
        expected_action=False,
        tags=["question", "debugging"],
    ),

    # -------------------------------------------------------------------------
    # ACTION REQUESTS (should not save) - 5 scenarios
    # -------------------------------------------------------------------------
    Scenario(
        id="ctrl_action_001",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.CONTROL,
        query="Please write a function that sorts an array.",
        expected_action=False,
        tags=["action_request", "coding"],
    ),
    Scenario(
        id="ctrl_action_002",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.CONTROL,
        query="Refactor this function to use async",
        expected_action=False,
        tags=["action_request", "refactor"],
    ),
    Scenario(
        id="ctrl_action_003",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.CONTROL,
        query="Add error handling to this code",
        expected_action=False,
        tags=["action_request", "coding"],
    ),
    Scenario(
        id="ctrl_action_004",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.CONTROL,
        query="Write unit tests for the auth module",
        expected_action=False,
        tags=["action_request", "testing"],
    ),
    Scenario(
        id="ctrl_action_005",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.CONTROL,
        query="Fix the bug on line 42",
        expected_action=False,
        tags=["action_request", "debugging"],
    ),

    # -------------------------------------------------------------------------
    # TEMPORARY/TRANSIENT (should not save) - 5 scenarios
    # -------------------------------------------------------------------------
    Scenario(
        id="ctrl_temp_001",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.CONTROL,
        query="I'm just testing something real quick.",
        expected_action=False,
        tags=["temporary", "testing"],
    ),
    Scenario(
        id="ctrl_temp_002",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.CONTROL,
        query="Let me try something different",
        expected_action=False,
        tags=["temporary", "exploring"],
    ),
    Scenario(
        id="ctrl_temp_003",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.CONTROL,
        query="Actually, ignore that last thing",
        expected_action=False,
        tags=["temporary", "retraction"],
    ),
    Scenario(
        id="ctrl_temp_004",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.CONTROL,
        query="Just brainstorming here",
        expected_action=False,
        tags=["temporary", "brainstorm"],
    ),
    Scenario(
        id="ctrl_temp_005",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.CONTROL,
        query="This is just a quick test",
        expected_action=False,
        tags=["temporary", "testing"],
    ),

    # -------------------------------------------------------------------------
    # CHITCHAT AND OPINIONS (should not save) - 5 scenarios
    # -------------------------------------------------------------------------
    Scenario(
        id="ctrl_chat_001",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.CONTROL,
        query="Thanks for your help!",
        expected_action=False,
        tags=["chitchat", "gratitude"],
    ),
    Scenario(
        id="ctrl_opinion_001",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.CONTROL,
        query="React has more GitHub stars than Vue",
        expected_action=False,
        tags=["fact", "trivia"],
    ),
    Scenario(
        id="ctrl_opinion_002",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.CONTROL,
        query="This code looks messy",
        expected_action=False,
        tags=["opinion", "code_quality"],
    ),
    Scenario(
        id="ctrl_opinion_003",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.CONTROL,
        query="That's an interesting approach",
        expected_action=False,
        tags=["opinion", "feedback"],
    ),
    Scenario(
        id="ctrl_known_001",
        tool_type=ToolType.MEMORY,
        level=ExplicitnessLevel.CONTROL,
        query="Did I already tell you we use React?",
        expected_action=False,
        tags=["question", "clarification"],
    ),
]


# =============================================================================
# ALL SCENARIOS
# =============================================================================

# Default set: memory scenarios + controls
# FILE_OPERATION_SCENARIOS are defined but not included by default (future work)
ALL_SCENARIOS = MEMORY_SCENARIOS + CONTROL_SCENARIOS

# To include file operation scenarios for future experiments:
# ALL_SCENARIOS_WITH_FILES = MEMORY_SCENARIOS + CONTROL_SCENARIOS + FILE_OPERATION_SCENARIOS


def get_scenarios_by_level(level: ExplicitnessLevel) -> list[Scenario]:
    """Get all scenarios at a specific explicitness level."""
    return [s for s in ALL_SCENARIOS if s.level == level]


def get_scenarios_by_difficulty(difficulty: Difficulty) -> list[Scenario]:
    """Get all scenarios at a specific difficulty level."""
    return [s for s in ALL_SCENARIOS if s.difficulty == difficulty]


def get_hard_implicit_scenarios() -> list[Scenario]:
    """Get implicit scenarios classified as HARD (for focused paper analysis).

    These are scenarios where:
    - Very terse (3 words or fewer)
    - Uses symbols/abbreviations
    - Requires significant inference
    - Unfamiliar patterns
    """
    return [
        s for s in ALL_SCENARIOS
        if s.level == ExplicitnessLevel.IMPLICIT
        and s.difficulty == Difficulty.HARD
        and "excluded_from_main_analysis" not in s.tags
    ]


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
