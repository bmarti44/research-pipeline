"""
Task definitions for tool-calling studies.

Each task defines a tool call scenario with:
- task_id: Unique identifier
- category: Operation type (control, read, create, edit)
- user_prompt: The scenario description
- expected_tool: The tool that should be called
- expected_args: Arguments that should be passed
"""

# Tool definitions
TOOLS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file",
        "parameters": {
            "path": {"type": "string", "description": "Path to the file"},
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file (creates or overwrites)",
        "parameters": {
            "path": {"type": "string", "description": "Path to the file"},
            "content": {"type": "string", "description": "Content to write"},
        },
    },
    {
        "name": "edit_file",
        "description": "Edit an existing file by replacing text",
        "parameters": {
            "path": {"type": "string", "description": "Path to the file"},
            "old_text": {"type": "string", "description": "Text to find"},
            "new_text": {"type": "string", "description": "Text to replace with"},
        },
    },
]


def get_tasks() -> list[dict]:
    """Return list of tasks for the study."""
    return [
        # Control tasks
        {
            "task_id": "control_001",
            "category": "control",
            "user_prompt": "What tools are available to you?",
            "expected_tool": None,
            "expected_args": {},
        },

        # Read operations
        {
            "task_id": "read_001",
            "category": "read_operations",
            "user_prompt": "Read the configuration file at /etc/config.yaml",
            "expected_tool": "read_file",
            "expected_args": {"path": "/etc/config.yaml"},
        },
        {
            "task_id": "read_002",
            "category": "read_operations",
            "user_prompt": "Show me what's in the README.md file",
            "expected_tool": "read_file",
            "expected_args": {"path": "README.md"},
        },

        # Create operations
        {
            "task_id": "create_001",
            "category": "create_operations",
            "user_prompt": "Create a new file called hello.txt with the content 'Hello, World!'",
            "expected_tool": "write_file",
            "expected_args": {"path": "hello.txt", "content": "Hello, World!"},
        },
        {
            "task_id": "create_002",
            "category": "create_operations",
            "user_prompt": "Write a Python script to /tmp/test.py that prints 'test'",
            "expected_tool": "write_file",
            "expected_args": {"path": "/tmp/test.py", "content": "print('test')"},
        },

        # Edit operations (key test for execution context effects)
        {
            "task_id": "edit_001",
            "category": "edit_operations",
            "user_prompt": "In the file config.yaml, change 'debug: false' to 'debug: true'",
            "expected_tool": "edit_file",
            "expected_args": {
                "path": "config.yaml",
                "old_text": "debug: false",
                "new_text": "debug: true",
            },
        },
        {
            "task_id": "edit_002",
            "category": "edit_operations",
            "user_prompt": "Update the version in package.json from '1.0.0' to '1.1.0'",
            "expected_tool": "edit_file",
            "expected_args": {
                "path": "package.json",
                "old_text": "'1.0.0'",
                "new_text": "'1.1.0'",
            },
        },
        {
            "task_id": "edit_003",
            "category": "edit_operations",
            "user_prompt": "Fix the typo in main.py: change 'pritn' to 'print'",
            "expected_tool": "edit_file",
            "expected_args": {
                "path": "main.py",
                "old_text": "pritn",
                "new_text": "print",
            },
        },
    ]


def get_tools() -> list[dict]:
    """Return tool definitions for prompts."""
    return TOOLS
