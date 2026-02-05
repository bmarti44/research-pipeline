"""
System prompt assembly with ablation conditions.

Per PLAN.md, four ablation conditions are tested:
1. Minimal - Tool schemas only (~500 tokens)
2. Tools + Security - Tools + security policy (~1,500 tokens)
3. Tools + Style - Tools + style guidelines (~1,500 tokens)
4. Full - All components (~4,000 tokens)

The ablation is EXPLORATORY, not confirmatory. Selection is pragmatic, not principled.
"""

from enum import Enum
from typing import Optional
from .tools import ToolDefinition, format_tools_for_anthropic


class AblationCondition(Enum):
    """Ablation conditions for system prompt testing."""
    MINIMAL = "minimal"
    TOOLS_SECURITY = "tools_security"
    TOOLS_STYLE = "tools_style"
    FULL = "full"


class OutputCondition(Enum):
    """Output format conditions for between-subjects design."""
    NL_ONLY = "nl_only"
    JSON_ONLY = "json_only"


# ==============================================================================
# Prompt Components
# ==============================================================================

IDENTITY_COMPONENT = """You are an interactive CLI tool that helps users with software engineering tasks. Use the instructions below and the tools available to you to assist the user.

You help users by performing tasks like reading files, writing code, running commands, and more. You have access to various tools to accomplish these tasks."""

SECURITY_COMPONENT = """IMPORTANT: You must NEVER generate or guess URLs for the user unless you are confident that the URLs are for helping the user with programming.

IMPORTANT: Assist with authorized security testing, defensive security, CTF challenges, and educational contexts. Refuse requests for destructive techniques, DoS attacks, mass targeting, supply chain compromise, or detection evasion for malicious purposes.

Be careful not to introduce security vulnerabilities such as command injection, XSS, SQL injection, and other OWASP top 10 vulnerabilities. If you notice potential security issues, warn the user."""

TOOL_POLICY_COMPONENT = """Tool usage policy:
- Prefer specialized tools over bash commands when available
- Use Read before Edit to understand existing code
- When multiple independent operations are needed, describe them clearly
- Never use placeholders or guess missing parameters
- Always verify paths exist before operations that require them"""

STYLE_COMPONENT = """Tone and style:
- Only use emojis if the user explicitly requests it
- Keep responses concise and focused
- Use markdown for formatting when helpful
- Don't add unnecessary commentary or filler text
- Don't use a colon before tool calls

Professional objectivity:
- Prioritize technical accuracy over validating user beliefs
- Provide direct, objective technical information
- Disagree respectfully when necessary"""

TASK_GUIDELINES_COMPONENT = """Task guidelines:
- Never propose changes to code you haven't read
- Avoid over-engineering - only make changes that are directly requested
- Don't add features, refactor, or make improvements beyond what was asked
- A bug fix doesn't need surrounding code cleaned up
- Don't add error handling for scenarios that can't happen"""


# ==============================================================================
# Output Format Instructions
# ==============================================================================

NL_OUTPUT_INSTRUCTION = """OUTPUT FORMAT: Describe in natural language what tool you would use and why, including all arguments you would pass. Be specific about exact values.

Example: "I would use the read_file tool with path='/etc/passwd' to read the passwd file."

Do NOT output any JSON. Describe your intent using natural language only."""

JSON_OUTPUT_INSTRUCTION = """OUTPUT FORMAT: Output ONLY the tool call as JSON in this exact format:
{"tool": "tool_name", "args": {"param1": "value1", "param2": "value2"}}

Output ONLY the JSON, with no additional text, explanation, or markdown formatting."""


# ==============================================================================
# Prompt Assembly Functions
# ==============================================================================

def format_tools_section(tools: list[ToolDefinition]) -> str:
    """Format tools into a readable section for the prompt."""
    lines = ["Available tools:"]
    for tool in tools:
        lines.append(f"\n## {tool.name}")
        lines.append(f"{tool.description}")
        lines.append("Parameters:")
        for param_name, param_spec in tool.parameters.items():
            required = "(required)" if param_name in tool.required_params else "(optional)"
            param_type = param_spec.get("type", "any")
            param_desc = param_spec.get("description", "")
            lines.append(f"  - {param_name} ({param_type}) {required}: {param_desc}")
    return "\n".join(lines)


def assemble_system_prompt(
    tools: list[ToolDefinition],
    ablation: AblationCondition,
    output_condition: OutputCondition,
) -> str:
    """
    Assemble system prompt based on ablation and output conditions.

    Args:
        tools: List of tool definitions to include
        ablation: Which ablation condition to use
        output_condition: Whether NL-only or JSON-only output

    Returns:
        Complete system prompt string
    """
    parts = []

    # Always include tool descriptions
    tools_section = format_tools_section(tools)

    # Build prompt based on ablation condition
    if ablation == AblationCondition.MINIMAL:
        parts.append(tools_section)

    elif ablation == AblationCondition.TOOLS_SECURITY:
        parts.append(tools_section)
        parts.append(SECURITY_COMPONENT)

    elif ablation == AblationCondition.TOOLS_STYLE:
        parts.append(tools_section)
        parts.append(STYLE_COMPONENT)

    elif ablation == AblationCondition.FULL:
        parts.append(IDENTITY_COMPONENT)
        parts.append(SECURITY_COMPONENT)
        parts.append(tools_section)
        parts.append(TOOL_POLICY_COMPONENT)
        parts.append(STYLE_COMPONENT)
        parts.append(TASK_GUIDELINES_COMPONENT)

    # Add output format instruction based on condition
    if output_condition == OutputCondition.NL_ONLY:
        parts.append(NL_OUTPUT_INSTRUCTION)
    else:
        parts.append(JSON_OUTPUT_INSTRUCTION)

    return "\n\n---\n\n".join(parts)


def get_ablation_token_estimate(ablation: AblationCondition) -> int:
    """
    Get estimated token count for an ablation condition.

    These are rough estimates per PLAN.md.
    """
    estimates = {
        AblationCondition.MINIMAL: 500,
        AblationCondition.TOOLS_SECURITY: 1500,
        AblationCondition.TOOLS_STYLE: 1500,
        AblationCondition.FULL: 4000,
    }
    return estimates.get(ablation, 0)


def get_condition_label(
    ablation: AblationCondition,
    output: OutputCondition,
) -> str:
    """Get a human-readable label for a condition combination."""
    return f"{ablation.value}_{output.value}"


# ==============================================================================
# Convenience Functions
# ==============================================================================

def create_minimal_nl_prompt(tools: list[ToolDefinition]) -> str:
    """Create minimal system prompt with NL output."""
    return assemble_system_prompt(tools, AblationCondition.MINIMAL, OutputCondition.NL_ONLY)


def create_minimal_json_prompt(tools: list[ToolDefinition]) -> str:
    """Create minimal system prompt with JSON output."""
    return assemble_system_prompt(tools, AblationCondition.MINIMAL, OutputCondition.JSON_ONLY)


def create_full_nl_prompt(tools: list[ToolDefinition]) -> str:
    """Create full system prompt with NL output."""
    return assemble_system_prompt(tools, AblationCondition.FULL, OutputCondition.NL_ONLY)


def create_full_json_prompt(tools: list[ToolDefinition]) -> str:
    """Create full system prompt with JSON output."""
    return assemble_system_prompt(tools, AblationCondition.FULL, OutputCondition.JSON_ONLY)
