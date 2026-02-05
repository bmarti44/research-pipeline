"""
Task definitions for the format friction experiment.

Per PLAN.md:
- Tasks with blind categorization metadata
- Sample tasks across complexity tiers
- Factorial adversarial categories (JSON/Escape/Unicode isolated)

Blind categorization protocol:
1. Two independent raters categorize each task using only the definitions
2. Raters are blind to expected friction and hypothesis direction
3. Disagreements resolved by third rater
4. Report inter-rater κ for categorization
5. Categories locked before data collection
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tools import (
    ToolDefinition,
    ToolComplexity,
    READ_FILE_TOOL,
    WRITE_FILE_TOOL,
    DELETE_FILE_TOOL,
    LIST_DIRECTORY_TOOL,
    SEARCH_FILES_TOOL,
    EDIT_FILE_TOOL,
    RUN_COMMAND_TOOL,
    HTTP_REQUEST_TOOL,
    CREATE_FILES_TOOL,
    NOOP_TOOL,
    WRITE_JSON_FILE_TOOL,
    WRITE_ESCAPED_TOOL,
    WRITE_UNICODE_TOOL,
    WRITE_COMPLEX_CONFIG_TOOL,
)
from core.harness import Task


# ==============================================================================
# Control Tasks (Baseline)
# ==============================================================================

CONTROL_TASKS = [
    Task(
        task_id="ctrl_001",
        tool=NOOP_TOOL,
        user_prompt="Acknowledge this message without taking any action.",
        expected_args={},
        category="control",
    ),
    Task(
        task_id="ctrl_002",
        tool=NOOP_TOOL,
        user_prompt="Just confirm you received this, no action needed.",
        expected_args={},
        category="control",
    ),
    Task(
        task_id="ctrl_003",
        tool=NOOP_TOOL,
        user_prompt="This is a test message. Please acknowledge without performing any operation.",
        expected_args={},
        category="control",
    ),
]

# ==============================================================================
# Simple Tasks (Single required parameter)
# ==============================================================================

SIMPLE_TASKS = [
    Task(
        task_id="simple_001",
        tool=READ_FILE_TOOL,
        user_prompt="Read the contents of /etc/passwd",
        expected_args={"path": "/etc/passwd"},
        category="simple",
    ),
    Task(
        task_id="simple_002",
        tool=READ_FILE_TOOL,
        user_prompt="I need to see what's in the config file at /home/user/.bashrc",
        expected_args={"path": "/home/user/.bashrc"},
        category="simple",
    ),
    Task(
        task_id="simple_003",
        tool=READ_FILE_TOOL,
        user_prompt="Show me the contents of /var/log/syslog",
        expected_args={"path": "/var/log/syslog"},
        category="simple",
    ),
    Task(
        task_id="simple_004",
        tool=DELETE_FILE_TOOL,
        user_prompt="Delete the file at /tmp/old_backup.tar.gz",
        expected_args={"path": "/tmp/old_backup.tar.gz"},
        category="simple",
    ),
    Task(
        task_id="simple_005",
        tool=DELETE_FILE_TOOL,
        user_prompt="Remove /home/user/Downloads/temp_file.txt",
        expected_args={"path": "/home/user/Downloads/temp_file.txt"},
        category="simple",
    ),
    Task(
        task_id="simple_006",
        tool=LIST_DIRECTORY_TOOL,
        user_prompt="List all files in /var/log",
        expected_args={"path": "/var/log"},
        category="simple",
    ),
    Task(
        task_id="simple_007",
        tool=LIST_DIRECTORY_TOOL,
        user_prompt="Show me what's in the /home/user/Documents directory",
        expected_args={"path": "/home/user/Documents"},
        category="simple",
    ),
    Task(
        task_id="simple_008",
        tool=READ_FILE_TOOL,
        user_prompt="Can you read /etc/hosts for me?",
        expected_args={"path": "/etc/hosts"},
        category="simple",
    ),
    Task(
        task_id="simple_009",
        tool=LIST_DIRECTORY_TOOL,
        user_prompt="What files are in /tmp?",
        expected_args={"path": "/tmp"},
        category="simple",
    ),
    Task(
        task_id="simple_010",
        tool=READ_FILE_TOOL,
        user_prompt="Display the contents of /etc/resolv.conf",
        expected_args={"path": "/etc/resolv.conf"},
        category="simple",
    ),
    Task(
        task_id="simple_011",
        tool=DELETE_FILE_TOOL,
        user_prompt="Delete /var/tmp/cache.db",
        expected_args={"path": "/var/tmp/cache.db"},
        category="simple",
    ),
    Task(
        task_id="simple_012",
        tool=LIST_DIRECTORY_TOOL,
        user_prompt="List the contents of /usr/local/bin",
        expected_args={"path": "/usr/local/bin"},
        category="simple",
    ),
]

# ==============================================================================
# Medium Tasks (Multiple parameters)
# ==============================================================================

MEDIUM_TASKS = [
    Task(
        task_id="medium_001",
        tool=WRITE_FILE_TOOL,
        user_prompt="Write 'Hello, World!' to /tmp/greeting.txt",
        expected_args={"path": "/tmp/greeting.txt", "content": "Hello, World!"},
        category="medium",
    ),
    Task(
        task_id="medium_002",
        tool=WRITE_FILE_TOOL,
        user_prompt="Create a file at /home/user/notes.txt with the content 'Remember to buy milk'",
        expected_args={"path": "/home/user/notes.txt", "content": "Remember to buy milk"},
        category="medium",
    ),
    Task(
        task_id="medium_003",
        tool=SEARCH_FILES_TOOL,
        user_prompt="Find all Python files in /home/user/projects",
        expected_args={"pattern": "*.py", "path": "/home/user/projects"},
        category="medium",
    ),
    Task(
        task_id="medium_004",
        tool=SEARCH_FILES_TOOL,
        user_prompt="Search for files matching *.log in /var/log",
        expected_args={"pattern": "*.log", "path": "/var/log"},
        category="medium",
    ),
    Task(
        task_id="medium_005",
        tool=RUN_COMMAND_TOOL,
        user_prompt="Run the command 'ls -la' in /home/user",
        expected_args={"command": "ls -la", "working_dir": "/home/user"},
        category="medium",
    ),
    Task(
        task_id="medium_006",
        tool=RUN_COMMAND_TOOL,
        user_prompt="Execute 'git status' in /home/user/myproject",
        expected_args={"command": "git status", "working_dir": "/home/user/myproject"},
        category="medium",
    ),
    Task(
        task_id="medium_007",
        tool=WRITE_FILE_TOOL,
        user_prompt="Save the text 'Configuration complete' to /etc/app/status.txt",
        expected_args={"path": "/etc/app/status.txt", "content": "Configuration complete"},
        category="medium",
    ),
    Task(
        task_id="medium_008",
        tool=SEARCH_FILES_TOOL,
        user_prompt="Find all markdown files in /home/user/docs",
        expected_args={"pattern": "*.md", "path": "/home/user/docs"},
        category="medium",
    ),
    Task(
        task_id="medium_009",
        tool=RUN_COMMAND_TOOL,
        user_prompt="Run 'npm install' in /home/user/webapp",
        expected_args={"command": "npm install", "working_dir": "/home/user/webapp"},
        category="medium",
    ),
    Task(
        task_id="medium_010",
        tool=WRITE_FILE_TOOL,
        user_prompt="Create /tmp/log.txt with 'Started at 10:00'",
        expected_args={"path": "/tmp/log.txt", "content": "Started at 10:00"},
        category="medium",
    ),
    Task(
        task_id="medium_011",
        tool=SEARCH_FILES_TOOL,
        user_prompt="Find *.json files in /etc/config",
        expected_args={"pattern": "*.json", "path": "/etc/config"},
        category="medium",
    ),
    Task(
        task_id="medium_012",
        tool=RUN_COMMAND_TOOL,
        user_prompt="Execute 'make build' in /home/user/project",
        expected_args={"command": "make build", "working_dir": "/home/user/project"},
        category="medium",
    ),
]

# ==============================================================================
# Complex Tasks (Nested objects, arrays)
# ==============================================================================

COMPLEX_TASKS = [
    Task(
        task_id="complex_001",
        tool=EDIT_FILE_TOOL,
        user_prompt="In /home/user/app.py, replace 'DEBUG = True' with 'DEBUG = False'",
        expected_args={
            "path": "/home/user/app.py",
            "edits": [{"old_string": "DEBUG = True", "new_string": "DEBUG = False"}],
        },
        category="complex",
    ),
    Task(
        task_id="complex_002",
        tool=EDIT_FILE_TOOL,
        user_prompt="Edit /etc/config.ini to change 'timeout=30' to 'timeout=60'",
        expected_args={
            "path": "/etc/config.ini",
            "edits": [{"old_string": "timeout=30", "new_string": "timeout=60"}],
        },
        category="complex",
    ),
    Task(
        task_id="complex_003",
        tool=CREATE_FILES_TOOL,
        user_prompt="Create two files: /tmp/file1.txt with 'content1' and /tmp/file2.txt with 'content2'",
        expected_args={
            "files": [
                {"path": "/tmp/file1.txt", "content": "content1"},
                {"path": "/tmp/file2.txt", "content": "content2"},
            ]
        },
        category="complex",
    ),
    Task(
        task_id="complex_004",
        tool=HTTP_REQUEST_TOOL,
        user_prompt="Make a GET request to https://api.example.com/users",
        expected_args={"method": "GET", "url": "https://api.example.com/users"},
        category="complex",
    ),
    Task(
        task_id="complex_005",
        tool=HTTP_REQUEST_TOOL,
        user_prompt="Send a POST request to https://api.example.com/data with body '{\"key\": \"value\"}'",
        expected_args={
            "method": "POST",
            "url": "https://api.example.com/data",
            "body": '{"key": "value"}',
        },
        category="complex",
    ),
    Task(
        task_id="complex_006",
        tool=EDIT_FILE_TOOL,
        user_prompt="In /home/user/main.js, replace 'const PORT = 3000' with 'const PORT = 8080'",
        expected_args={
            "path": "/home/user/main.js",
            "edits": [{"old_string": "const PORT = 3000", "new_string": "const PORT = 8080"}],
        },
        category="complex",
    ),
    Task(
        task_id="complex_007",
        tool=CREATE_FILES_TOOL,
        user_prompt="Create /tmp/a.txt with 'alpha' and /tmp/b.txt with 'beta' and /tmp/c.txt with 'gamma'",
        expected_args={
            "files": [
                {"path": "/tmp/a.txt", "content": "alpha"},
                {"path": "/tmp/b.txt", "content": "beta"},
                {"path": "/tmp/c.txt", "content": "gamma"},
            ]
        },
        category="complex",
    ),
    Task(
        task_id="complex_008",
        tool=HTTP_REQUEST_TOOL,
        user_prompt="Make a DELETE request to https://api.example.com/item/42",
        expected_args={"method": "DELETE", "url": "https://api.example.com/item/42"},
        category="complex",
    ),
    Task(
        task_id="complex_009",
        tool=EDIT_FILE_TOOL,
        user_prompt="In /etc/nginx/nginx.conf, change 'worker_processes 1' to 'worker_processes 4'",
        expected_args={
            "path": "/etc/nginx/nginx.conf",
            "edits": [{"old_string": "worker_processes 1", "new_string": "worker_processes 4"}],
        },
        category="complex",
    ),
    Task(
        task_id="complex_010",
        tool=HTTP_REQUEST_TOOL,
        user_prompt="Send a PUT request to https://api.example.com/user/1 with body '{\"name\": \"updated\"}'",
        expected_args={
            "method": "PUT",
            "url": "https://api.example.com/user/1",
            "body": '{"name": "updated"}',
        },
        category="complex",
    ),
    Task(
        task_id="complex_011",
        tool=CREATE_FILES_TOOL,
        user_prompt="Create /home/user/test/index.html with '<html></html>' and /home/user/test/style.css with 'body {}'",
        expected_args={
            "files": [
                {"path": "/home/user/test/index.html", "content": "<html></html>"},
                {"path": "/home/user/test/style.css", "content": "body {}"},
            ]
        },
        category="complex",
    ),
    Task(
        task_id="complex_012",
        tool=EDIT_FILE_TOOL,
        user_prompt="In /home/user/settings.py, replace 'ALLOWED_HOSTS = []' with 'ALLOWED_HOSTS = [\"*\"]'",
        expected_args={
            "path": "/home/user/settings.py",
            "edits": [{"old_string": "ALLOWED_HOSTS = []", "new_string": 'ALLOWED_HOSTS = ["*"]'}],
        },
        category="complex",
    ),
]

# ==============================================================================
# Adversarial Tasks - JSON Content (Adv-JSON)
# Tests: JSON content that must be nested in args
# ==============================================================================

ADV_JSON_TASKS = [
    Task(
        task_id="adv_json_001",
        tool=WRITE_JSON_FILE_TOOL,
        user_prompt='Write {"name": "test", "value": 123} to /tmp/config.json',
        expected_args={
            "path": "/tmp/config.json",
            "json_content": {"name": "test", "value": 123},
        },
        category="adv_json",
    ),
    Task(
        task_id="adv_json_002",
        tool=WRITE_JSON_FILE_TOOL,
        user_prompt='Save the JSON object {"users": ["alice", "bob"], "active": true} to /data/users.json',
        expected_args={
            "path": "/data/users.json",
            "json_content": {"users": ["alice", "bob"], "active": True},
        },
        category="adv_json",
    ),
    Task(
        task_id="adv_json_003",
        tool=WRITE_JSON_FILE_TOOL,
        user_prompt='Write {"nested": {"deep": {"value": "found"}}} to /tmp/deep.json',
        expected_args={
            "path": "/tmp/deep.json",
            "json_content": {"nested": {"deep": {"value": "found"}}},
        },
        category="adv_json",
    ),
    Task(
        task_id="adv_json_004",
        tool=WRITE_JSON_FILE_TOOL,
        user_prompt='Write {"items": [1, 2, 3], "count": 3} to /data/items.json',
        expected_args={
            "path": "/data/items.json",
            "json_content": {"items": [1, 2, 3], "count": 3},
        },
        category="adv_json",
    ),
    Task(
        task_id="adv_json_005",
        tool=WRITE_JSON_FILE_TOOL,
        user_prompt='Save {"enabled": false, "timeout": null} to /tmp/settings.json',
        expected_args={
            "path": "/tmp/settings.json",
            "json_content": {"enabled": False, "timeout": None},
        },
        category="adv_json",
    ),
    Task(
        task_id="adv_json_006",
        tool=WRITE_JSON_FILE_TOOL,
        user_prompt='Write {"tags": ["a", "b", "c"], "meta": {"version": 2}} to /config/tags.json',
        expected_args={
            "path": "/config/tags.json",
            "json_content": {"tags": ["a", "b", "c"], "meta": {"version": 2}},
        },
        category="adv_json",
    ),
    Task(
        task_id="adv_json_007",
        tool=WRITE_JSON_FILE_TOOL,
        user_prompt='Write {"empty_array": [], "empty_object": {}} to /tmp/empty.json',
        expected_args={
            "path": "/tmp/empty.json",
            "json_content": {"empty_array": [], "empty_object": {}},
        },
        category="adv_json",
    ),
]

# ==============================================================================
# Adversarial Tasks - Escaping (Adv-Escape)
# Tests: Quotes, backslashes requiring escaping
# ==============================================================================

ADV_ESCAPE_TASKS = [
    Task(
        task_id="adv_escape_001",
        tool=WRITE_ESCAPED_TOOL,
        user_prompt='Write the text "Hello \\"World\\"" to /tmp/quoted.txt',
        expected_args={
            "path": "/tmp/quoted.txt",
            "content": 'Hello "World"',
        },
        category="adv_escape",
    ),
    Task(
        task_id="adv_escape_002",
        tool=WRITE_ESCAPED_TOOL,
        user_prompt="Write 'It\\'s a test' to /tmp/apostrophe.txt",
        expected_args={
            "path": "/tmp/apostrophe.txt",
            "content": "It's a test",
        },
        category="adv_escape",
    ),
    Task(
        task_id="adv_escape_003",
        tool=WRITE_ESCAPED_TOOL,
        user_prompt="Write 'C:\\\\Users\\\\name' to /tmp/path.txt",
        expected_args={
            "path": "/tmp/path.txt",
            "content": "C:\\Users\\name",
        },
        category="adv_escape",
    ),
    Task(
        task_id="adv_escape_004",
        tool=WRITE_ESCAPED_TOOL,
        user_prompt='Write the text with newline "line1\\nline2" to /tmp/newline.txt',
        expected_args={
            "path": "/tmp/newline.txt",
            "content": "line1\nline2",
        },
        category="adv_escape",
    ),
    Task(
        task_id="adv_escape_005",
        tool=WRITE_ESCAPED_TOOL,
        user_prompt='Write "Tab\\there" to /tmp/tab.txt',
        expected_args={
            "path": "/tmp/tab.txt",
            "content": "Tab\there",
        },
        category="adv_escape",
    ),
    Task(
        task_id="adv_escape_006",
        tool=WRITE_ESCAPED_TOOL,
        user_prompt="Write 'She said \"yes\" and \"no\"' to /tmp/multiquote.txt",
        expected_args={
            "path": "/tmp/multiquote.txt",
            "content": 'She said "yes" and "no"',
        },
        category="adv_escape",
    ),
    Task(
        task_id="adv_escape_007",
        tool=WRITE_ESCAPED_TOOL,
        user_prompt="Write 'D:\\\\Documents\\\\file.txt' to /tmp/winpath.txt",
        expected_args={
            "path": "/tmp/winpath.txt",
            "content": "D:\\Documents\\file.txt",
        },
        category="adv_escape",
    ),
]

# ==============================================================================
# Adversarial Tasks - Unicode (Adv-Unicode)
# Tests: Emoji, non-ASCII characters
# ==============================================================================

ADV_UNICODE_TASKS = [
    Task(
        task_id="adv_unicode_001",
        tool=WRITE_UNICODE_TOOL,
        user_prompt="Write 'Hello 👋 World 🌍' to /tmp/emoji.txt",
        expected_args={
            "path": "/tmp/emoji.txt",
            "content": "Hello 👋 World 🌍",
        },
        category="adv_unicode",
    ),
    Task(
        task_id="adv_unicode_002",
        tool=WRITE_UNICODE_TOOL,
        user_prompt="Write '日本語テスト' to /tmp/japanese.txt",
        expected_args={
            "path": "/tmp/japanese.txt",
            "content": "日本語テスト",
        },
        category="adv_unicode",
    ),
    Task(
        task_id="adv_unicode_003",
        tool=WRITE_UNICODE_TOOL,
        user_prompt="Write 'Ελληνικά' to /tmp/greek.txt",
        expected_args={
            "path": "/tmp/greek.txt",
            "content": "Ελληνικά",
        },
        category="adv_unicode",
    ),
    Task(
        task_id="adv_unicode_004",
        tool=WRITE_UNICODE_TOOL,
        user_prompt="Write '✓ Check ✗ Cross' to /tmp/symbols.txt",
        expected_args={
            "path": "/tmp/symbols.txt",
            "content": "✓ Check ✗ Cross",
        },
        category="adv_unicode",
    ),
    Task(
        task_id="adv_unicode_005",
        tool=WRITE_UNICODE_TOOL,
        user_prompt="Write '🚀 Launch 🎯 Target 💡 Idea' to /tmp/emojis.txt",
        expected_args={
            "path": "/tmp/emojis.txt",
            "content": "🚀 Launch 🎯 Target 💡 Idea",
        },
        category="adv_unicode",
    ),
    Task(
        task_id="adv_unicode_006",
        tool=WRITE_UNICODE_TOOL,
        user_prompt="Write '中文测试' to /tmp/chinese.txt",
        expected_args={
            "path": "/tmp/chinese.txt",
            "content": "中文测试",
        },
        category="adv_unicode",
    ),
    Task(
        task_id="adv_unicode_007",
        tool=WRITE_UNICODE_TOOL,
        user_prompt="Write '→ arrow ← back ↑ up ↓ down' to /tmp/arrows.txt",
        expected_args={
            "path": "/tmp/arrows.txt",
            "content": "→ arrow ← back ↑ up ↓ down",
        },
        category="adv_unicode",
    ),
]

# ==============================================================================
# Adversarial Tasks - Combined (Adv-Combined)
# Tests: JSON + Escaping + Unicode combined
# ==============================================================================

ADV_COMBINED_TASKS = [
    Task(
        task_id="adv_combined_001",
        tool=WRITE_COMPLEX_CONFIG_TOOL,
        user_prompt='Write a JSON config with {"emoji": "🎉", "path": "C:\\\\test"} to /tmp/complex.json',
        expected_args={
            "path": "/tmp/complex.json",
            "config": {"emoji": "🎉", "path": "C:\\test"},
            "format": "json",
        },
        category="adv_combined",
    ),
    Task(
        task_id="adv_combined_002",
        tool=WRITE_COMPLEX_CONFIG_TOOL,
        user_prompt='Create YAML config with {"message": "Say \\"Hello\\" 👋"} at /tmp/config.yaml',
        expected_args={
            "path": "/tmp/config.yaml",
            "config": {"message": 'Say "Hello" 👋'},
            "format": "yaml",
        },
        category="adv_combined",
    ),
    Task(
        task_id="adv_combined_003",
        tool=WRITE_COMPLEX_CONFIG_TOOL,
        user_prompt='Write JSON config {"path": "C:\\\\Users\\\\test", "emoji": "🎉", "quote": "He said \\"hi\\""} to /tmp/full.json',
        expected_args={
            "path": "/tmp/full.json",
            "config": {"path": "C:\\Users\\test", "emoji": "🎉", "quote": 'He said "hi"'},
            "format": "json",
        },
        category="adv_combined",
    ),
    Task(
        task_id="adv_combined_004",
        tool=WRITE_COMPLEX_CONFIG_TOOL,
        user_prompt='Create YAML with {"title": "日本語 🇯🇵", "path": "D:\\\\Data"} at /tmp/intl.yaml',
        expected_args={
            "path": "/tmp/intl.yaml",
            "config": {"title": "日本語 🇯🇵", "path": "D:\\Data"},
            "format": "yaml",
        },
        category="adv_combined",
    ),
    Task(
        task_id="adv_combined_005",
        tool=WRITE_COMPLEX_CONFIG_TOOL,
        user_prompt='Write JSON config {"newline": "line1\\nline2", "tab": "col1\\tcol2", "emoji": "✅"} to /tmp/mixed.json',
        expected_args={
            "path": "/tmp/mixed.json",
            "config": {"newline": "line1\nline2", "tab": "col1\tcol2", "emoji": "✅"},
            "format": "json",
        },
        category="adv_combined",
    ),
]

# ==============================================================================
# All Tasks
# ==============================================================================

ALL_TASKS = (
    CONTROL_TASKS
    + SIMPLE_TASKS
    + MEDIUM_TASKS
    + COMPLEX_TASKS
    + ADV_JSON_TASKS
    + ADV_ESCAPE_TASKS
    + ADV_UNICODE_TASKS
    + ADV_COMBINED_TASKS
)

TASKS_BY_CATEGORY = {
    "control": CONTROL_TASKS,
    "simple": SIMPLE_TASKS,
    "medium": MEDIUM_TASKS,
    "complex": COMPLEX_TASKS,
    "adv_json": ADV_JSON_TASKS,
    "adv_escape": ADV_ESCAPE_TASKS,
    "adv_unicode": ADV_UNICODE_TASKS,
    "adv_combined": ADV_COMBINED_TASKS,
}


def get_tasks_by_category(category: str) -> list[Task]:
    """Get all tasks for a category."""
    return TASKS_BY_CATEGORY.get(category, [])


def get_all_tasks() -> list[Task]:
    """Get all tasks."""
    return ALL_TASKS


def get_task_by_id(task_id: str) -> Optional[Task]:
    """Get a task by its ID."""
    for task in ALL_TASKS:
        if task.task_id == task_id:
            return task
    return None


def get_category_counts() -> dict[str, int]:
    """Get count of tasks per category."""
    return {cat: len(tasks) for cat, tasks in TASKS_BY_CATEGORY.items()}
