"""
Tool Definitions for Two-Stage Experiment

Defines the tools available to the model, their parameters, and XML format.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    description: str
    required: bool = True
    param_type: str = "string"


@dataclass
class ToolDefinition:
    """Definition of a tool."""
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)

    def to_schema_text(self) -> str:
        """Generate human-readable schema for prompts."""
        params_text = ""
        for param in self.parameters:
            req = "(required)" if param.required else "(optional)"
            params_text += f"    - {param.name}: {param.description} {req}\n"

        return f"""Tool: {self.name}
  Description: {self.description}
  Parameters:
{params_text}"""


# =============================================================================
# Tool Definitions
# =============================================================================

MEMORY_SAVE = ToolDefinition(
    name="memory_save",
    description="Save information for later retrieval. Use when the user shares important information they want remembered.",
    parameters=[
        ToolParameter(
            name="key",
            description="A short identifier for what is being saved (e.g., 'api_key', 'project_name')",
        ),
        ToolParameter(
            name="value",
            description="The actual information to save",
        ),
        ToolParameter(
            name="category",
            description="Category for organization (e.g., 'credentials', 'preferences', 'notes')",
            required=False,
        ),
    ],
)

WEB_SEARCH = ToolDefinition(
    name="web_search",
    description="Search the web for information. Use when the user needs current information or facts you don't know.",
    parameters=[
        ToolParameter(
            name="query",
            description="The search query",
        ),
        ToolParameter(
            name="num_results",
            description="Number of results to return (default: 5)",
            required=False,
        ),
    ],
)

CODE_EXECUTE = ToolDefinition(
    name="code_execute",
    description="Execute code in a sandboxed environment. Use when the user wants to run code or see output.",
    parameters=[
        ToolParameter(
            name="language",
            description="Programming language (python, javascript, bash)",
        ),
        ToolParameter(
            name="code",
            description="The code to execute",
        ),
    ],
)

FILE_OPERATION = ToolDefinition(
    name="file_operation",
    description="Read or write files. Use when the user wants to access or modify file contents.",
    parameters=[
        ToolParameter(
            name="operation",
            description="The operation to perform: 'read' or 'write'",
        ),
        ToolParameter(
            name="path",
            description="File path",
        ),
        ToolParameter(
            name="content",
            description="Content to write (only for write operation)",
            required=False,
        ),
    ],
)

API_CALL = ToolDefinition(
    name="api_call",
    description="Make an HTTP API request. Use when the user needs to interact with external services.",
    parameters=[
        ToolParameter(
            name="method",
            description="HTTP method (GET, POST, PUT, DELETE)",
        ),
        ToolParameter(
            name="url",
            description="The API endpoint URL",
        ),
        ToolParameter(
            name="body",
            description="Request body (for POST/PUT)",
            required=False,
        ),
        ToolParameter(
            name="headers",
            description="Request headers as JSON",
            required=False,
        ),
    ],
)


# All available tools
TOOLS: dict[str, ToolDefinition] = {
    "memory_save": MEMORY_SAVE,
    "web_search": WEB_SEARCH,
    "code_execute": CODE_EXECUTE,
    "file_operation": FILE_OPERATION,
    "api_call": API_CALL,
}


def get_all_tools_schema() -> str:
    """Generate schema text for all tools."""
    schemas = [tool.to_schema_text() for tool in TOOLS.values()]
    return "\n".join(schemas)


def generate_tool_xml(tool_name: str, params: dict[str, str]) -> str:
    """Generate XML for a tool call.

    Args:
        tool_name: Name of the tool
        params: Dictionary of parameter name -> value

    Returns:
        XML string for the tool call
    """
    param_xml = "\n".join(
        f'  <param name="{name}">{value}</param>'
        for name, value in params.items()
    )
    return f'<tool name="{tool_name}">\n{param_xml}\n</tool>'


def parse_tool_xml(xml_text: str) -> Optional[tuple[str, dict[str, str]]]:
    """Parse tool XML to extract tool name and parameters.

    Args:
        xml_text: Raw text potentially containing tool XML

    Returns:
        Tuple of (tool_name, params_dict) or None if no valid tool found
    """
    import re

    # Match tool tag
    tool_match = re.search(r'<tool\s+name="([^"]+)">(.*?)</tool>', xml_text, re.DOTALL)
    if not tool_match:
        return None

    tool_name = tool_match.group(1)
    inner_content = tool_match.group(2)

    # Extract parameters
    params = {}
    param_pattern = r'<param\s+name="([^"]+)">(.*?)</param>'
    for param_match in re.finditer(param_pattern, inner_content, re.DOTALL):
        param_name = param_match.group(1)
        param_value = param_match.group(2).strip()
        params[param_name] = param_value

    return (tool_name, params)


def check_no_tool(xml_text: str) -> bool:
    """Check if response contains explicit no-tool marker."""
    return "<no_tool/>" in xml_text or "<no_tool />" in xml_text
