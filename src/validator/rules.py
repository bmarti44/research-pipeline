"""Validation rules for tool calls."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import re
import fnmatch
from pathlib import Path

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
class ToolOutput:
    """Record of a tool call and its output summary."""
    tool_name: str
    tool_input: dict
    output_summary: str  # Brief summary of what the output showed


@dataclass
class ValidationContext:
    """Mutable context tracking state across tool calls in a session."""
    user_query: str
    tool_history: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    known_paths: set[str] = field(default_factory=set)

    # F16: Track file reads
    read_files: list[str] = field(default_factory=list)

    # F17: Track glob patterns
    glob_patterns: list[str] = field(default_factory=list)

    # F18: Track recent tool outputs for verification detection
    recent_outputs: list[ToolOutput] = field(default_factory=list)

    def add_known_paths(self, paths: list[str]) -> None:
        """Add paths discovered via Glob/LS."""
        self.known_paths.update(paths)

    def add_file_read(self, file_path: str) -> None:
        """Track a file that was read."""
        self.read_files.append(file_path)

    def add_glob_pattern(self, pattern: str) -> None:
        """Track a glob pattern that was used."""
        self.glob_patterns.append(pattern)

    def add_tool_output(self, tool_name: str, tool_input: dict, output_summary: str) -> None:
        """Track a tool's output for verification detection."""
        self.recent_outputs.append(ToolOutput(tool_name, tool_input, output_summary))
        # Keep only last 5 outputs
        if len(self.recent_outputs) > 5:
            self.recent_outputs = self.recent_outputs[-5:]


class RuleValidator:
    """Validates tool calls against rules.

    Supports ablation testing by enabling/disabling individual rules.
    """

    # All available rules with their IDs
    # PROVEN: Rules with demonstrated catch rates in experiments
    # THEORETICAL: Rules that guard against behaviors Claude already avoids
    PROVEN_RULES = {"F10", "F17", "F18", "F21"}  # Actually catch bad behavior
    THEORETICAL_RULES = {"F1", "F4", "F8", "F13", "F15", "F19", "F22", "F23", "F24", "F25"}  # 0% catch rate
    ALL_RULES = PROVEN_RULES | THEORETICAL_RULES

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
            "F17": self._rule_f17_redundant_glob,
            "F18": self._rule_f18_unnecessary_reverification,
            "F19": self._rule_f19_search_after_answer,
            "F21": self._rule_f21_overly_broad_search,
            "F22": self._rule_f22_bash_tool_misuse,
            "F23": self._rule_f23_destructive_command,
            "F24": self._rule_f24_well_known_api,
            "F25": self._rule_f25_answer_in_training,
        }

        # Run rules in order (F15/F23 first for safety, then efficiency rules)
        rule_order = ["F15", "F23", "F1", "F4", "F8", "F10", "F13", "F17", "F18", "F19", "F21", "F22", "F24", "F25"]

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

    # ===== EFFICIENCY RULES =====

    def _rule_f17_redundant_glob(
        self, tool_name: str, tool_input: dict, ctx: ValidationContext
    ) -> Optional[ValidationResult]:
        """F17: Block glob patterns that overlap with prior searches."""
        if tool_name not in ("Glob", "glob", "find_files", "list_files"):
            return None

        pattern = tool_input.get("pattern", "") or tool_input.get("glob", "")
        if not pattern:
            return None

        for prior_pattern in ctx.glob_patterns:
            # Check if patterns are equivalent or one subsumes the other
            relationship = self._glob_relationship(pattern, prior_pattern)

            if relationship == "equivalent":
                return ValidationResult(
                    decision=Decision.DENY,
                    rule_id="F17",
                    reason=f"Equivalent glob pattern already searched: '{prior_pattern}'"
                )
            elif relationship == "subset":
                # New pattern is more specific, prior already covered it
                return ValidationResult(
                    decision=Decision.DENY,
                    rule_id="F17",
                    reason=f"Pattern '{pattern}' already covered by broader search '{prior_pattern}'"
                )
            elif relationship == "superset":
                # New pattern is broader - could be wasteful if prior was sufficient
                # Allow but could warn in future
                pass

        return None

    def _glob_relationship(self, pattern1: str, pattern2: str) -> str:
        """Determine relationship between two glob patterns.

        Returns:
            'equivalent': patterns match same files
            'subset': pattern1 matches subset of pattern2
            'superset': pattern1 matches superset of pattern2
            'disjoint': patterns are unrelated
        """
        # Normalize patterns
        p1, p2 = pattern1.strip(), pattern2.strip()

        # Exact match
        if p1 == p2:
            return "equivalent"

        # Common subsumption patterns
        # **/*.py is superset of *.py, src/*.py, etc.
        if "**" in p1 and "**" not in p2:
            # p1 is recursive, p2 is not
            p1_base = p1.replace("**/", "").replace("**", "*")
            if fnmatch.fnmatch(p2, p1_base) or p1_base in p2:
                return "superset"

        if "**" in p2 and "**" not in p1:
            # p2 is recursive, p1 is not
            p2_base = p2.replace("**/", "").replace("**", "*")
            if fnmatch.fnmatch(p1, p2_base) or p2_base in p1:
                return "subset"

        # Both recursive
        if "**" in p1 and "**" in p2:
            # Compare extensions
            ext1 = p1.split(".")[-1] if "." in p1 else ""
            ext2 = p2.split(".")[-1] if "." in p2 else ""
            if ext1 == ext2:
                # Same extension, compare directory specificity
                dir1 = p1.split("**")[0]
                dir2 = p2.split("**")[0]
                if dir1 == dir2:
                    return "equivalent"
                elif dir1.startswith(dir2):
                    return "subset"  # p1 is in subdirectory of p2
                elif dir2.startswith(dir1):
                    return "superset"

        return "disjoint"

    def _rule_f18_unnecessary_reverification(
        self, tool_name: str, tool_input: dict, ctx: ValidationContext
    ) -> Optional[ValidationResult]:
        """F18: Block redundant verification commands."""

        # Common reverification patterns
        reverification_patterns = [
            # git status after git add/commit/rm
            {
                "check_tool": ("Bash", "bash", "shell"),
                "check_pattern": r"\bgit\s+status\b",
                "after_tools": [("Bash", r"\bgit\s+(add|commit|rm|reset|checkout)\b")],
                "reason": "git status unnecessary - prior git command output shows state",
            },
            # ls after write/mkdir
            {
                "check_tool": ("Bash", "bash", "shell"),
                "check_pattern": r"\bls\b",
                "after_tools": [
                    ("Write", None),
                    ("Bash", r"\bmkdir\b"),
                ],
                "reason": "ls unnecessary - file operation confirmed success",
            },
            # cat after write (to verify)
            {
                "check_tool": ("Read", "read_file", "View", "cat"),
                "check_pattern": None,
                "after_tools": [("Write", None), ("Edit", None)],
                "reason": "File read unnecessary - write/edit operation shows result",
            },
            # npm list after npm install
            {
                "check_tool": ("Bash", "bash", "shell"),
                "check_pattern": r"\bnpm\s+(list|ls)\b",
                "after_tools": [("Bash", r"\bnpm\s+install\b")],
                "reason": "npm list unnecessary - install output shows what was installed",
            },
        ]

        for pattern_config in reverification_patterns:
            # Check if current tool matches the verification pattern
            if tool_name not in pattern_config["check_tool"]:
                continue

            # For bash commands, check the command pattern
            if pattern_config["check_pattern"]:
                command = tool_input.get("command", "")
                if not re.search(pattern_config["check_pattern"], command, re.IGNORECASE):
                    continue

            # Check if a triggering tool was recently used
            for recent in ctx.recent_outputs[-3:]:  # Check last 3 outputs
                for after_tool, after_pattern in pattern_config["after_tools"]:
                    if recent.tool_name != after_tool:
                        continue

                    # If pattern specified, check it
                    if after_pattern:
                        cmd = recent.tool_input.get("command", "")
                        if not re.search(after_pattern, cmd, re.IGNORECASE):
                            continue

                    # Match found - this is unnecessary reverification
                    return ValidationResult(
                        decision=Decision.DENY,
                        rule_id="F18",
                        reason=pattern_config["reason"],
                    )

        return None

    def _rule_f19_search_after_answer(
        self, tool_name: str, tool_input: dict, ctx: ValidationContext
    ) -> Optional[ValidationResult]:
        """F19: Block web search when answer is already in user's message.

        Only triggers when user PROVIDES information (statement), not when
        user ASKS for information (question).
        """
        if tool_name not in ("WebSearch", "web_search"):
            return None

        query = tool_input.get("query", "")
        if not query:
            return None

        user_query = ctx.user_query.strip()

        # Don't trigger if user query is a question (asking for info, not providing it)
        question_indicators = [
            user_query.endswith("?"),
            user_query.lower().startswith(("what ", "who ", "where ", "when ", "why ", "how ", "which ", "is ", "are ", "can ", "could ", "do ", "does ", "will ", "would ")),
        ]
        if any(question_indicators):
            return None

        # Don't trigger if user query is too short (likely just a question/request)
        if len(user_query.split()) < 8:
            return None

        # Check if the search query is semantically similar to user's context
        is_in_context, score = self.semantic.is_answer_in_context(query, ctx.user_query)

        if is_in_context:
            return ValidationResult(
                decision=Decision.DENY,
                rule_id="F19",
                reason=f"Search query matches info in context (similarity: {score:.2f}). Use provided info."
            )

        return None

    # ===== NEW DISTINCT RULES (F21-F23) =====

    def _rule_f21_overly_broad_search(
        self, tool_name: str, tool_input: dict, ctx: ValidationContext
    ) -> Optional[ValidationResult]:
        """F21: Block overly vague/broad search queries unlikely to return useful results.

        Different from F10: Targets query QUALITY, not redundancy.
        """
        if tool_name not in ("WebSearch", "web_search"):
            return None

        query = tool_input.get("query", "").strip()
        if not query:
            return None

        # Check for overly short queries (likely too vague)
        words = query.split()
        if len(words) <= 2:
            # Allow short queries if they contain specific terms
            specific_indicators = [
                r"\d{4}",  # Year
                r"v\d+",  # Version
                r"[A-Z]{2,}",  # Acronyms
                r"@\w+",  # Handles
                r"#\w+",  # Hashtags
                r"\.\w{2,4}$",  # File extensions
            ]
            is_specific = any(re.search(p, query) for p in specific_indicators)

            if not is_specific:
                return ValidationResult(
                    decision=Decision.DENY,
                    rule_id="F21",
                    reason=f"Search query too vague ({len(words)} words). Add specific terms for useful results."
                )

        # Check for generic single-word queries
        generic_terms = {
            "python", "javascript", "java", "code", "programming", "tutorial",
            "help", "error", "fix", "issue", "problem", "how", "what", "best",
            "example", "documentation", "guide", "learn", "install", "setup"
        }
        if len(words) == 1 and words[0].lower() in generic_terms:
            return ValidationResult(
                decision=Decision.DENY,
                rule_id="F21",
                reason=f"Search query '{query}' is too generic. Add context for useful results."
            )

        return None

    def _rule_f22_bash_tool_misuse(
        self, tool_name: str, tool_input: dict, ctx: ValidationContext
    ) -> Optional[ValidationResult]:
        """F22: Block using Bash for operations that have dedicated tools.

        Different from other rules: Targets TOOL SELECTION efficiency.
        Encourages use of specialized tools over shell commands.
        """
        if tool_name not in ("Bash", "bash", "shell", "execute"):
            return None

        command = tool_input.get("command", "")
        if not command:
            return None

        # Patterns where a dedicated tool should be used instead
        misuse_patterns = [
            # cat/head/tail for reading -> Use Read tool
            {
                "pattern": r"^\s*(cat|head|tail)\s+[\"']?[\w\.\/\-]+[\"']?\s*$",
                "reason": "Use the Read tool instead of cat/head/tail for reading files",
                "exception": r"\|",  # Allow if piped
            },
            # find for file search -> Use Glob tool
            {
                "pattern": r"^\s*find\s+.*-name\s+",
                "reason": "Use the Glob tool instead of find for file searches",
                "exception": r"-exec|-delete",  # Allow if doing operations
            },
            # grep for content search -> Use Grep tool
            {
                "pattern": r"^\s*(grep|rg)\s+.*[\"'].*[\"']",
                "reason": "Use the Grep tool instead of grep/rg for content searches",
                "exception": r"\|",  # Allow if piped
            },
            # echo > for writing -> Use Write tool
            {
                "pattern": r"echo\s+.*>\s*[\"']?[\w\.\/\-]+[\"']?\s*$",
                "reason": "Use the Write tool instead of echo redirection for creating files",
                "exception": r">>",  # Allow append
            },
        ]

        for mp in misuse_patterns:
            if re.search(mp["pattern"], command, re.IGNORECASE):
                # Check for exception pattern
                if mp.get("exception") and re.search(mp["exception"], command):
                    continue
                return ValidationResult(
                    decision=Decision.DENY,
                    rule_id="F22",
                    reason=mp["reason"],
                )

        return None

    def _rule_f23_destructive_command(
        self, tool_name: str, tool_input: dict, ctx: ValidationContext
    ) -> Optional[ValidationResult]:
        """F23: Block potentially destructive commands without safeguards.

        Different from other rules: SAFETY-focused, not efficiency.
        Prevents accidental data loss from dangerous commands.
        """
        if tool_name not in ("Bash", "bash", "shell", "execute"):
            return None

        command = tool_input.get("command", "")
        if not command:
            return None

        # Dangerous patterns that should be blocked or warned
        destructive_patterns = [
            # rm with force/recursive on broad paths
            {
                "pattern": r"\brm\s+(-[rf]+\s+)*(/|~|\*|\.\.|\.)\s*$",
                "reason": "Destructive rm command targets root, home, or broad paths",
            },
            {
                "pattern": r"\brm\s+-[rf]*\s+-[rf]*\s+",
                "reason": "rm with -rf flags is destructive - verify path is correct",
            },
            # git reset --hard / git clean -f
            {
                "pattern": r"\bgit\s+(reset\s+--hard|clean\s+-[fd])",
                "reason": "Destructive git command will lose uncommitted changes",
            },
            # DROP/DELETE without WHERE
            {
                "pattern": r"\b(DROP\s+(TABLE|DATABASE)|DELETE\s+FROM\s+\w+\s*;)",
                "reason": "Destructive SQL command - verify this is intentional",
            },
            # chmod 777 / chmod -R
            {
                "pattern": r"\bchmod\s+(-R\s+)?777\s+",
                "reason": "chmod 777 is insecure - use more restrictive permissions",
            },
            # dd command (disk destroyer)
            {
                "pattern": r"\bdd\s+.*of=/dev/",
                "reason": "dd to device is destructive - verify target is correct",
            },
            # Format/mkfs commands
            {
                "pattern": r"\b(mkfs|format)\s+",
                "reason": "Filesystem format command is destructive",
            },
            # kill -9 / killall
            {
                "pattern": r"\b(kill\s+-9|killall)\s+",
                "reason": "Forceful process termination - consider graceful shutdown first",
            },
        ]

        for dp in destructive_patterns:
            if re.search(dp["pattern"], command, re.IGNORECASE):
                return ValidationResult(
                    decision=Decision.DENY,
                    rule_id="F23",
                    reason=dp["reason"],
                )

        return None

    def _rule_f24_well_known_api(
        self, tool_name: str, tool_input: dict, ctx: ValidationContext
    ) -> Optional[ValidationResult]:
        """F24: Block searching for extremely well-known APIs/libraries.

        ROOT CAUSE: LLMs often search for documentation they definitely have
        in training data. This feels "unnatural" - like looking up how to
        spell your own name.

        Targets the "I know this but let me check anyway" pattern.
        """
        if tool_name not in ("WebSearch", "web_search"):
            return None

        query = tool_input.get("query", "").lower()
        if not query:
            return None

        # Extremely common APIs/libraries that any LLM definitely knows
        # These are so fundamental they appear in millions of training examples
        well_known_patterns = [
            # Python stdlib - appears in virtually every Python tutorial
            (r"\b(python|py)\b.*(open|read|write)\s*(file)?", "Python file I/O"),
            (r"\b(python|py)\b.*\b(list|dict|set|tuple)\b", "Python built-in types"),
            (r"\b(python|py)\b.*\b(for|while)\s*loop", "Python loops"),
            (r"\b(python|py)\b.*\b(import|from)\b", "Python imports"),
            (r"\b(python|py)\b.*\b(print|input)\b", "Python print/input"),
            (r"\b(python|py)\b.*\b(if|else|elif)\b", "Python conditionals"),
            (r"\b(python|py)\b.*\b(def|function|lambda)\b", "Python functions"),
            (r"\b(python|py)\b.*\b(class|__init__|self)\b", "Python classes"),
            (r"\b(python|py)\b.*\b(try|except|exception)\b", "Python exceptions"),
            (r"\b(python|py)\b.*\b(str|int|float|bool)\b", "Python type conversion"),

            # JavaScript fundamentals
            (r"\b(javascript|js)\b.*\b(var|let|const)\b", "JavaScript variables"),
            (r"\b(javascript|js)\b.*\b(function|arrow|=>)\b", "JavaScript functions"),
            (r"\b(javascript|js)\b.*\b(array|map|filter|reduce)\b", "JavaScript arrays"),
            (r"\b(javascript|js)\b.*\b(object|json)\b", "JavaScript objects"),
            (r"\b(javascript|js)\b.*\b(if|else|switch)\b", "JavaScript conditionals"),
            (r"\b(javascript|js)\b.*\b(for|while|foreach)\b", "JavaScript loops"),
            (r"\b(javascript|js)\b.*\b(async|await|promise)\b", "JavaScript async"),

            # Git basics
            (r"\bgit\b.*(add|commit|push|pull|clone)", "Git basics"),
            (r"\bgit\b.*(branch|checkout|merge)", "Git branching"),
            (r"\bgit\b.*(status|log|diff)", "Git inspection"),

            # Shell basics
            (r"\b(bash|shell)\b.*(cd|ls|mkdir|rm|cp|mv)", "Shell basics"),
            (r"\b(bash|shell)\b.*(echo|cat|grep|find)", "Shell commands"),

            # HTML/CSS basics
            (r"\b(html)\b.*(div|span|p|a|img|table)", "HTML elements"),
            (r"\b(css)\b.*(color|font|margin|padding|display)", "CSS properties"),

            # SQL basics
            (r"\b(sql)\b.*(select|insert|update|delete|from|where)", "SQL basics"),
        ]

        for pattern, topic in well_known_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return ValidationResult(
                    decision=Decision.DENY,
                    rule_id="F24",
                    reason=f"Searching for {topic} - this is fundamental knowledge. Answer directly from training."
                )

        return None

    def _rule_f25_answer_in_training(
        self, tool_name: str, tool_input: dict, ctx: ValidationContext
    ) -> Optional[ValidationResult]:
        """F25: Block searching for common programming questions with obvious answers.

        ROOT CAUSE: LLMs search for things they definitely know because of
        uncertainty/hedging behavior. But some questions have such universal
        answers that searching is clearly unnecessary.

        Targets: "how to X in Y" where X is trivial and Y is common.
        """
        if tool_name not in ("WebSearch", "web_search"):
            return None

        query = tool_input.get("query", "").lower()
        if not query:
            return None

        # Pattern: "how to [trivial operation] in [common language]"
        trivial_operations = [
            "declare variable", "create variable", "define variable",
            "create function", "define function", "write function",
            "create class", "define class",
            "create array", "create list", "make list", "make array",
            "create dictionary", "create dict", "create object", "create map",
            "create loop", "write loop", "make loop",
            "print", "console log", "output",
            "read file", "write file", "open file",
            "concatenate string", "join string", "split string",
            "convert to string", "convert to int", "convert to number",
            "check if", "test if", "verify if",
            "import", "include", "require",
            "return value", "return from function",
            "handle error", "catch exception", "try catch",
            "comment", "add comment",
        ]

        common_languages = [
            "python", "javascript", "js", "java", "c++", "cpp", "c#", "csharp",
            "ruby", "go", "golang", "rust", "typescript", "ts", "php", "swift",
            "kotlin", "scala", "perl", "r", "matlab", "bash", "shell", "sql"
        ]

        # Check for "how to [trivial] in [language]" pattern
        for op in trivial_operations:
            if op in query:
                for lang in common_languages:
                    if lang in query:
                        return ValidationResult(
                            decision=Decision.DENY,
                            rule_id="F25",
                            reason=f"'{op}' in {lang} is fundamental knowledge. Answer directly."
                        )

        return None
