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
    """Validates tool calls against rules."""

    def __init__(self, semantic: Optional[SemanticClassifier] = None):
        self.semantic = semantic or SemanticClassifier()

    def validate(
        self,
        tool_name: str,
        tool_input: dict,
        ctx: ValidationContext,
    ) -> ValidationResult:
        """Run all rules. First denial wins."""

        rules = [
            self._rule_f15_binary_file,
            self._rule_f1_static_knowledge,
            self._rule_f4_memory_vs_web,
            self._rule_f8_missing_location,
            self._rule_f10_duplicate_search,
            self._rule_f13_hallucinated_path,
        ]

        for rule in rules:
            result = rule(tool_name, tool_input, ctx)
            if result is not None:
                return result

        return ValidationResult(
            decision=Decision.ALLOW,
            rule_id=None,
            reason="All rules passed"
        )

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
        """Block reading binary files."""
        if tool_name not in ("Read", "View", "view_file", "read_file"):
            return None

        path = (tool_input.get("file_path", "") or tool_input.get("path", "")).lower()

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
                return ValidationResult(
                    decision=Decision.DENY,
                    rule_id="F15",
                    reason=f"Binary file cannot be read as text."
                )
        return None
