"""
Tasks for Tool Phase Transition Study.

Defines 10 core tasks with correct tools, plus distractor tool pools.
"""

from typing import Any
import random

random.seed(42)


# =============================================================================
# Core Tasks (10 tasks with known correct tools)
# =============================================================================

CORE_TASKS = [
    {
        "task_id": "core_001",
        "description": "Get the current weather in New York",
        "correct_tool": "get_weather",
        "correct_args": {"location": "New York"},
        "category": "data_retrieval"
    },
    {
        "task_id": "core_002",
        "description": "Send an email to john@example.com with subject 'Meeting Tomorrow'",
        "correct_tool": "send_email",
        "correct_args": {"to": "john@example.com", "subject": "Meeting Tomorrow"},
        "category": "communication"
    },
    {
        "task_id": "core_003",
        "description": "Create a new file called 'report.txt' with content 'Q4 Summary'",
        "correct_tool": "create_file",
        "correct_args": {"filename": "report.txt", "content": "Q4 Summary"},
        "category": "file_operations"
    },
    {
        "task_id": "core_004",
        "description": "Search the database for users with status 'active'",
        "correct_tool": "query_database",
        "correct_args": {"table": "users", "filter": {"status": "active"}},
        "category": "database"
    },
    {
        "task_id": "core_005",
        "description": "Schedule a meeting for tomorrow at 2pm",
        "correct_tool": "schedule_meeting",
        "correct_args": {"time": "2pm", "date": "tomorrow"},
        "category": "calendar"
    },
    {
        "task_id": "core_006",
        "description": "Calculate 15% tip on a $47.50 bill",
        "correct_tool": "calculate",
        "correct_args": {"expression": "47.50 * 0.15"},
        "category": "math"
    },
    {
        "task_id": "core_007",
        "description": "Translate 'Hello world' to Spanish",
        "correct_tool": "translate_text",
        "correct_args": {"text": "Hello world", "target_language": "Spanish"},
        "category": "language"
    },
    {
        "task_id": "core_008",
        "description": "Get the stock price for AAPL",
        "correct_tool": "get_stock_price",
        "correct_args": {"symbol": "AAPL"},
        "category": "finance"
    },
    {
        "task_id": "core_009",
        "description": "Resize image.png to 800x600 pixels",
        "correct_tool": "resize_image",
        "correct_args": {"filename": "image.png", "width": 800, "height": 600},
        "category": "media"
    },
    {
        "task_id": "core_010",
        "description": "Set a reminder for 5pm to 'Call mom'",
        "correct_tool": "set_reminder",
        "correct_args": {"time": "5pm", "message": "Call mom"},
        "category": "productivity"
    },
]


# =============================================================================
# Core Tool Definitions
# =============================================================================

CORE_TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {"location": "string"}
    },
    {
        "name": "send_email",
        "description": "Send an email message",
        "parameters": {"to": "string", "subject": "string", "body": "string"}
    },
    {
        "name": "create_file",
        "description": "Create a new file with content",
        "parameters": {"filename": "string", "content": "string"}
    },
    {
        "name": "query_database",
        "description": "Query a database table",
        "parameters": {"table": "string", "filter": "object"}
    },
    {
        "name": "schedule_meeting",
        "description": "Schedule a calendar meeting",
        "parameters": {"time": "string", "date": "string", "attendees": "array"}
    },
    {
        "name": "calculate",
        "description": "Perform a mathematical calculation",
        "parameters": {"expression": "string"}
    },
    {
        "name": "translate_text",
        "description": "Translate text to another language",
        "parameters": {"text": "string", "target_language": "string"}
    },
    {
        "name": "get_stock_price",
        "description": "Get current stock price",
        "parameters": {"symbol": "string"}
    },
    {
        "name": "resize_image",
        "description": "Resize an image file",
        "parameters": {"filename": "string", "width": "integer", "height": "integer"}
    },
    {
        "name": "set_reminder",
        "description": "Set a reminder notification",
        "parameters": {"time": "string", "message": "string"}
    },
]


# =============================================================================
# Distractor Tools (Similar and Unrelated)
# =============================================================================

# Semantically similar tools (confusable with core tools)
SIMILAR_DISTRACTORS = [
    {"name": "check_weather", "description": "Check weather forecast", "similar_to": "get_weather"},
    {"name": "fetch_weather_data", "description": "Fetch weather information", "similar_to": "get_weather"},
    {"name": "weather_lookup", "description": "Look up weather conditions", "similar_to": "get_weather"},

    {"name": "compose_email", "description": "Compose an email draft", "similar_to": "send_email"},
    {"name": "mail_message", "description": "Mail a message to recipient", "similar_to": "send_email"},
    {"name": "email_user", "description": "Email a user", "similar_to": "send_email"},

    {"name": "write_file", "description": "Write content to file", "similar_to": "create_file"},
    {"name": "new_file", "description": "Create new file", "similar_to": "create_file"},
    {"name": "save_file", "description": "Save content as file", "similar_to": "create_file"},

    {"name": "search_database", "description": "Search database records", "similar_to": "query_database"},
    {"name": "db_query", "description": "Execute database query", "similar_to": "query_database"},
    {"name": "find_records", "description": "Find database records", "similar_to": "query_database"},

    {"name": "book_meeting", "description": "Book a meeting slot", "similar_to": "schedule_meeting"},
    {"name": "create_event", "description": "Create calendar event", "similar_to": "schedule_meeting"},
    {"name": "add_appointment", "description": "Add appointment to calendar", "similar_to": "schedule_meeting"},

    {"name": "compute", "description": "Compute mathematical expression", "similar_to": "calculate"},
    {"name": "math_eval", "description": "Evaluate math expression", "similar_to": "calculate"},
    {"name": "solve", "description": "Solve mathematical problem", "similar_to": "calculate"},

    {"name": "convert_language", "description": "Convert text language", "similar_to": "translate_text"},
    {"name": "language_translate", "description": "Translate language", "similar_to": "translate_text"},
    {"name": "text_translation", "description": "Perform text translation", "similar_to": "translate_text"},

    {"name": "stock_quote", "description": "Get stock quote", "similar_to": "get_stock_price"},
    {"name": "fetch_stock", "description": "Fetch stock data", "similar_to": "get_stock_price"},
    {"name": "ticker_price", "description": "Get ticker price", "similar_to": "get_stock_price"},

    {"name": "scale_image", "description": "Scale image dimensions", "similar_to": "resize_image"},
    {"name": "image_resize", "description": "Resize image file", "similar_to": "resize_image"},
    {"name": "transform_image", "description": "Transform image size", "similar_to": "resize_image"},

    {"name": "create_reminder", "description": "Create a reminder", "similar_to": "set_reminder"},
    {"name": "add_reminder", "description": "Add reminder to list", "similar_to": "set_reminder"},
    {"name": "remind_me", "description": "Set up reminder notification", "similar_to": "set_reminder"},
]

# Unrelated tools (different domains entirely)
UNRELATED_DISTRACTORS = [
    {"name": "compress_video", "description": "Compress video file size"},
    {"name": "extract_audio", "description": "Extract audio from video"},
    {"name": "merge_pdfs", "description": "Merge multiple PDF files"},
    {"name": "ocr_image", "description": "Extract text from image via OCR"},
    {"name": "generate_qr", "description": "Generate QR code"},
    {"name": "validate_json", "description": "Validate JSON structure"},
    {"name": "minify_css", "description": "Minify CSS code"},
    {"name": "format_code", "description": "Format source code"},
    {"name": "lint_python", "description": "Lint Python code"},
    {"name": "run_tests", "description": "Run test suite"},
    {"name": "deploy_app", "description": "Deploy application"},
    {"name": "backup_data", "description": "Create data backup"},
    {"name": "encrypt_file", "description": "Encrypt a file"},
    {"name": "hash_password", "description": "Hash a password"},
    {"name": "verify_signature", "description": "Verify digital signature"},
    {"name": "ping_server", "description": "Ping a server"},
    {"name": "trace_route", "description": "Trace network route"},
    {"name": "dns_lookup", "description": "Perform DNS lookup"},
    {"name": "whois_query", "description": "Query WHOIS database"},
    {"name": "port_scan", "description": "Scan open ports"},
    {"name": "generate_uuid", "description": "Generate UUID"},
    {"name": "base64_encode", "description": "Encode to base64"},
    {"name": "url_shorten", "description": "Shorten a URL"},
    {"name": "parse_html", "description": "Parse HTML content"},
    {"name": "scrape_page", "description": "Scrape web page"},
    {"name": "cache_clear", "description": "Clear cache"},
    {"name": "log_event", "description": "Log an event"},
    {"name": "track_metric", "description": "Track a metric"},
    {"name": "alert_team", "description": "Send alert to team"},
    {"name": "rotate_logs", "description": "Rotate log files"},
]


def build_tool_set(
    task: dict[str, Any],
    tool_count: int,
    distractor_type: str
) -> list[dict[str, Any]]:
    """
    Build a tool set for a given task and conditions.

    Args:
        task: The core task
        tool_count: Total number of tools to include
        distractor_type: 'similar_tools', 'unrelated_tools', or 'no_distractors'

    Returns:
        List of tool definitions
    """
    tools = list(CORE_TOOLS)  # Always include all core tools

    if tool_count <= 10:
        # Just use core tools
        return tools[:tool_count]

    # Need to add distractors
    distractors_needed = tool_count - 10

    if distractor_type == "no_distractors":
        # Add random mix
        all_distractors = SIMILAR_DISTRACTORS + UNRELATED_DISTRACTORS
        random.shuffle(all_distractors)
        tools.extend(all_distractors[:distractors_needed])

    elif distractor_type == "similar_tools":
        # Prioritize similar distractors
        random.shuffle(SIMILAR_DISTRACTORS)
        tools.extend(SIMILAR_DISTRACTORS[:distractors_needed])
        if distractors_needed > len(SIMILAR_DISTRACTORS):
            extra = distractors_needed - len(SIMILAR_DISTRACTORS)
            tools.extend(UNRELATED_DISTRACTORS[:extra])

    elif distractor_type == "unrelated_tools":
        # Only unrelated distractors
        random.shuffle(UNRELATED_DISTRACTORS)
        tools.extend(UNRELATED_DISTRACTORS[:distractors_needed])

    return tools[:tool_count]


def get_tasks() -> list[dict[str, Any]]:
    """Generate all task configurations."""
    tasks = []
    task_id = 0

    tool_counts = [5, 10, 15, 20, 25, 30, 40, 50]
    distractor_types = ["no_distractors", "similar_tools", "unrelated_tools"]

    for core_task in CORE_TASKS:
        for tool_count in tool_counts:
            for distractor_type in distractor_types:
                tool_set = build_tool_set(core_task, tool_count, distractor_type)

                tasks.append({
                    "task_id": f"task_{task_id:04d}",
                    "core_task_id": core_task["task_id"],
                    "description": core_task["description"],
                    "correct_tool": core_task["correct_tool"],
                    "correct_args": core_task["correct_args"],
                    "category": core_task["category"],
                    "tool_count": tool_count,
                    "distractor_type": distractor_type,
                    "available_tools": tool_set,
                })
                task_id += 1

    return tasks


def format_tools_for_prompt(tools: list[dict[str, Any]]) -> str:
    """Format tool definitions for inclusion in prompt."""
    lines = ["Available tools:"]
    for tool in tools:
        params = ", ".join(f"{k}: {v}" for k, v in tool.get("parameters", {}).items())
        lines.append(f"- {tool['name']}: {tool['description']} ({params})")
    return "\n".join(lines)


if __name__ == "__main__":
    tasks = get_tasks()
    print(f"Total tasks: {len(tasks)}")

    # Count by tool count
    for count in [5, 10, 15, 20, 25, 30, 40, 50]:
        n = len([t for t in tasks if t["tool_count"] == count])
        print(f"  {count} tools: {n}")

    # Count by distractor type
    for dtype in ["no_distractors", "similar_tools", "unrelated_tools"]:
        n = len([t for t in tasks if t["distractor_type"] == dtype])
        print(f"  {dtype}: {n}")
