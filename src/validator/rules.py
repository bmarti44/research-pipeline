"""Validation rules for tool calls."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import re

from .semantic import SemanticClassifier


class Decision(Enum):
    ALLOW = "allow"
    DENY = "deny"


@dataclass
class ValidationResult:
    decision: Decision
    rule_id: Optional[str]
    reason: str


@dataclass
class ValidationContext:
    """Mutable context tracking state across tool calls in a session."""
    user_query: str
    tool_history: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    known_paths: set[str] = field(default_factory=set)

    def add_known_paths(self, paths: list[str]) -> None:
        """Add paths discovered via Glob/LS."""
        self.known_paths.update(paths)


class RuleValidator:
    """Validates tool calls against rules.

    Supports ablation testing by enabling/disabling individual rules.
    """

    # All available rules with their IDs
    ALL_RULES = {"F1", "F4", "F8", "F10", "F13", "F15"}

    def __init__(
        self,
        semantic: Optional[SemanticClassifier] = None,
        enabled_rules: Optional[set[str]] = None,
    ):
        """
        Initialize the rule validator.

        Args:
            semantic: Semantic classifier for F1, F4, F10 rules
            enabled_rules: Set of rule IDs to enable. If None, all rules are enabled.
                          Use for ablation studies, e.g., {"F1"} to test only F1.
        """
        self.semantic = semantic or SemanticClassifier()
        self.enabled_rules = enabled_rules if enabled_rules is not None else self.ALL_RULES.copy()

        # Validate rule IDs
        invalid_rules = self.enabled_rules - self.ALL_RULES
        if invalid_rules:
            raise ValueError(f"Unknown rule IDs: {invalid_rules}. Valid rules: {self.ALL_RULES}")

    def validate(
        self,
        tool_name: str,
        tool_input: dict,
        ctx: ValidationContext,
    ) -> ValidationResult:
        """Run enabled rules. First denial wins."""

        # Map rule IDs to rule methods
        rule_map = {
            "F15": self._rule_f15_binary_file,
            "F1": self._rule_f1_static_knowledge,
            "F4": self._rule_f4_memory_vs_web,
            "F8": self._rule_f8_missing_location,
            "F10": self._rule_f10_duplicate_search,
            "F13": self._rule_f13_hallucinated_path,
        }

        # Run rules in order (F15 first for efficiency)
        rule_order = ["F15", "F1", "F4", "F8", "F10", "F13"]

        for rule_id in rule_order:
            if rule_id not in self.enabled_rules:
                continue

            rule_fn = rule_map[rule_id]
            result = rule_fn(tool_name, tool_input, ctx)
            if result is not None:
                return result

        return ValidationResult(
            decision=Decision.ALLOW,
            rule_id=None,
            reason="All rules passed"
        )

    @classmethod
    def for_ablation(cls, only_rule: str, semantic: Optional[SemanticClassifier] = None) -> "RuleValidator":
        """Create a validator with only one rule enabled for ablation testing.

        Args:
            only_rule: The single rule ID to enable (e.g., "F1")
            semantic: Optional semantic classifier

        Returns:
            RuleValidator with only the specified rule enabled
        """
        return cls(semantic=semantic, enabled_rules={only_rule})

    @classmethod
    def without_rule(cls, exclude_rule: str, semantic: Optional[SemanticClassifier] = None) -> "RuleValidator":
        """Create a validator with all rules except one for ablation testing.

        Args:
            exclude_rule: The rule ID to exclude (e.g., "F1")
            semantic: Optional semantic classifier

        Returns:
            RuleValidator with all rules except the specified one
        """
        enabled = cls.ALL_RULES - {exclude_rule}
        return cls(semantic=semantic, enabled_rules=enabled)

    def _rule_f1_static_knowledge(
        self, tool_name: str, tool_input: dict, ctx: ValidationContext
    ) -> Optional[ValidationResult]:
        """Block web search for static knowledge queries."""
        if tool_name not in ("WebSearch", "web_search"):
            return None

        is_static, score = self.semantic.is_static_knowledge_query(ctx.user_query)

        if is_static:
            return ValidationResult(
                decision=Decision.DENY,
                rule_id="F1",
                reason=f"Static knowledge query (score: {score:.2f}). Answer directly."
            )
        return None

    def _rule_f4_memory_vs_web(
        self, tool_name: str, tool_input: dict, ctx: ValidationContext
    ) -> Optional[ValidationResult]:
        """Block web search when memory search is appropriate."""
        if tool_name not in ("WebSearch", "web_search"):
            return None

        is_memory, score = self.semantic.is_memory_reference_query(ctx.user_query)

        if is_memory:
            return ValidationResult(
                decision=Decision.DENY,
                rule_id="F4",
                reason=f"Memory reference (score: {score:.2f}). Use memory search instead."
            )
        return None

    def _rule_f8_missing_location(
        self, tool_name: str, tool_input: dict, ctx: ValidationContext
    ) -> Optional[ValidationResult]:
        """Block location-dependent search without getting location first."""
        if tool_name not in ("WebSearch", "web_search"):
            return None

        query = tool_input.get("query", "").lower()
        user_query = ctx.user_query.lower()

        location_keywords = ["near me", "nearby", "closest", "local", "in my area"]

        if any(kw in user_query or kw in query for kw in location_keywords):
            # Check if we've already gotten location
            if not any("location" in h.lower() for h in ctx.tool_history):
                return ValidationResult(
                    decision=Decision.DENY,
                    rule_id="F8",
                    reason="Location-dependent query. Get user location first."
                )
        return None

    def _rule_f10_duplicate_search(
        self, tool_name: str, tool_input: dict, ctx: ValidationContext
    ) -> Optional[ValidationResult]:
        """Block duplicate searches."""
        if tool_name not in ("WebSearch", "web_search"):
            return None

        query = tool_input.get("query", "")

        is_dup, score = self.semantic.is_duplicate_search(query, ctx.search_queries)
        if is_dup:
            return ValidationResult(
                decision=Decision.DENY,
                rule_id="F10",
                reason=f"Duplicate search (similarity: {score:.2f}). Use previous results."
            )
        return None

    def _rule_f13_hallucinated_path(
        self, tool_name: str, tool_input: dict, ctx: ValidationContext
    ) -> Optional[ValidationResult]:
        """Block access to paths that look invented."""
        if tool_name not in ("Read", "View", "view_file", "read_file"):
            return None

        path = tool_input.get("file_path", "") or tool_input.get("path", "")

        # Allow if path was mentioned by user or discovered
        if path in ctx.known_paths:
            return None
        if path in ctx.user_query:
            return None

        # Allow common/reasonable paths
        common_patterns = [
            r"^\.?/?README",
            r"^\.?/?package\.json$",
            r"^\.?/?pyproject\.toml$",
            r"^\.?/?Cargo\.toml$",
            r"^\.?/?\.env",
        ]
        for pattern in common_patterns:
            if re.search(pattern, path, re.IGNORECASE):
                return None

        # Flag suspicious generic paths
        suspicious = [
            (r"^/project/", "Generic /project/ path"),
            (r"^/app/", "Generic /app/ path"),
            (r"/src/main\.(py|js|ts)$", "Generic main file"),
            (r"^/home/user/", "Generic /home/user/ path"),
            (r"^/var/www/", "Generic /var/www/ path"),
        ]

        for pattern, desc in suspicious:
            if re.search(pattern, path):
                return ValidationResult(
                    decision=Decision.DENY,
                    rule_id="F13",
                    reason=f"Hallucinated path ({desc}). List directory first."
                )

        # If path is absolute and not in known_paths, block it
        if path.startswith("/") and len(ctx.known_paths) > 0:
            return ValidationResult(
                decision=Decision.DENY,
                rule_id="F13",
                reason="Unknown absolute path. List directory first to discover valid paths."
            )

        return None

    def _rule_f15_binary_file(
        self, tool_name: str, tool_input: dict, ctx: ValidationContext
    ) -> Optional[ValidationResult]:
        """Block reading binary files via Read or Bash."""

        # Check Read tool
        if tool_name in ("Read", "View", "view_file", "read_file"):
            path = (tool_input.get("file_path", "") or tool_input.get("path", "")).lower()
            if self._is_binary_path(path):
                return ValidationResult(
                    decision=Decision.DENY,
                    rule_id="F15",
                    reason=f"Binary file cannot be read as text."
                )

        # Check Bash commands that try to read binary files
        if tool_name in ("Bash", "bash", "shell", "execute"):
            command = tool_input.get("command", "")

            # Patterns that directly read binary files (command followed by binary path)
            # More specific patterns to avoid false positives from heredocs with shebangs
            direct_binary_reads = [
                # cat/head/tail directly reading from binary locations
                r"\b(cat|head|tail|less|more)\s+(/usr/bin/|/bin/|/sbin/|/usr/lib/)\S+",
                r"\b(cat|head|tail|less|more)\s+\S+\.(pyc|so|dylib|exe|dll|bin)\b",
                # xxd/hexdump/strings on binary files
                r"\b(xxd|hexdump|strings)\s+\S+",
            ]

            for pattern in direct_binary_reads:
                if re.search(pattern, command, re.IGNORECASE):
                    return ValidationResult(
                        decision=Decision.DENY,
                        rule_id="F15",
                        reason=f"Cannot read binary file via shell command."
                    )

        return None

    def _is_binary_path(self, path: str) -> bool:
        """Check if a path looks like a binary file."""
        binary_patterns = [
            r"\.(exe|dll|so|dylib|bin|o|a)$",
            r"\.(png|jpg|jpeg|gif|bmp|ico|webp|svg)$",
            r"\.(pdf|doc|docx|xls|xlsx|ppt|pptx)$",
            r"\.(zip|tar|gz|7z|rar|bz2)$",
            r"\.(mp3|mp4|wav|avi|mov|mkv|flac)$",
            r"\.(pyc|pyo|class|wasm)$",
            r"^/usr/bin/",
            r"^/bin/",
            r"^/sbin/",
        ]
        for pattern in binary_patterns:
            if re.search(pattern, path, re.IGNORECASE):
                return True
        return False
