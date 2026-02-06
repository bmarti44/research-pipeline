"""
Prompt building for tool-calling studies.

Builds system and user prompts for NL and JSON conditions.
"""

import json
import sys
from pathlib import Path

# Import tasks module from same directory (works with dynamic loading)
_this_dir = Path(__file__).parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))
import tasks as tasks_module
get_tools = tasks_module.get_tools


def build_system_prompt(task: dict, condition: str) -> str:
    """
    Build system prompt based on condition.

    Args:
        task: Task dictionary
        condition: 'nl_only' or 'json_only'

    Returns:
        System prompt string
    """
    tools = get_tools()
    tools_desc = "\n".join([
        f"- {t['name']}: {t['description']}"
        for t in tools
    ])

    tools_schema = json.dumps(tools, indent=2)

    base_prompt = f"""You are an assistant with access to the following tools:

{tools_desc}

Tool schemas:
{tools_schema}
"""

    if condition == "nl_only":
        return base_prompt + """
When the user requests an action, describe in natural language:
1. Which tool you would use
2. What arguments you would pass
3. Why this is the appropriate tool

Do NOT output JSON. Describe your intended action in plain English.
"""
    elif condition == "json_only":
        return base_prompt + """
When the user requests an action, output ONLY a JSON tool call in this exact format:
{"tool": "tool_name", "args": {"arg1": "value1", "arg2": "value2"}}

Output ONLY the JSON. No explanation, no markdown, no other text.
"""
    else:
        return base_prompt


def build_prompt(task: dict, condition: str) -> str:
    """
    Build user prompt.

    Args:
        task: Task dictionary
        condition: 'nl_only' or 'json_only'

    Returns:
        User prompt string
    """
    return task.get("user_prompt", "")
