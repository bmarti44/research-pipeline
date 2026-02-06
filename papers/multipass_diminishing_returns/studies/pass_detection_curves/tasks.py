"""
Tasks for Multi-Pass Diminishing Returns Study.

Generates code samples with seeded errors of known types for detection tracking.
"""

from typing import Any
import random

random.seed(42)


# =============================================================================
# Error Type Definitions
# =============================================================================

ERROR_TYPES = {
    "syntax_error": {
        "description": "Missing brackets, typos, invalid syntax",
        "expected_detectability": "high",
        "examples": ["missing_bracket", "typo_in_keyword", "invalid_operator"]
    },
    "type_error": {
        "description": "Type mismatches, wrong argument types",
        "expected_detectability": "high",
        "examples": ["string_int_concat", "wrong_arg_type", "null_dereference"]
    },
    "logic_error": {
        "description": "Wrong operators, off-by-one, wrong conditions",
        "expected_detectability": "moderate",
        "examples": ["off_by_one", "wrong_comparison", "inverted_condition"]
    },
    "edge_case_error": {
        "description": "Unhandled null, empty input, boundary conditions",
        "expected_detectability": "moderate",
        "examples": ["empty_array", "null_input", "boundary_overflow"]
    },
    "security_error": {
        "description": "Injection, XSS, auth bypass",
        "expected_detectability": "low",
        "examples": ["sql_injection", "xss_vulnerability", "path_traversal"]
    },
    "performance_error": {
        "description": "N+1 queries, unnecessary loops, memory leaks",
        "expected_detectability": "low",
        "examples": ["n_plus_one", "quadratic_loop", "unbounded_cache"]
    },
    "concurrency_error": {
        "description": "Race conditions, deadlocks, thread safety",
        "expected_detectability": "low",
        "examples": ["race_condition", "missing_lock", "deadlock_potential"]
    },
}


# =============================================================================
# Code Templates with Seeded Errors
# =============================================================================

CODE_SAMPLES = {
    # =========== SYNTAX ERRORS (High detectability) ===========
    "syntax_error": [
        {
            "id": "syn_001",
            "complexity": "simple",
            "language": "python",
            "correct_code": '''
def add_numbers(a, b):
    return a + b
''',
            "buggy_code": '''
def add_numbers(a, b):
    return a + b
)
''',
            "error_location": "line 3",
            "error_description": "Extra closing parenthesis",
        },
        {
            "id": "syn_002",
            "complexity": "simple",
            "language": "python",
            "correct_code": '''
def greet(name):
    print(f"Hello, {name}!")
''',
            "buggy_code": '''
def greet(name):
    prnt(f"Hello, {name}!")
''',
            "error_location": "line 2",
            "error_description": "Typo: 'prnt' instead of 'print'",
        },
        {
            "id": "syn_003",
            "complexity": "moderate",
            "language": "python",
            "correct_code": '''
def process_items(items):
    results = []
    for item in items:
        if item > 0:
            results.append(item * 2)
    return results
''',
            "buggy_code": '''
def process_items(items):
    results = []
    for item in items:
        if item > 0:
            results.append(item * 2)
    return results
]
''',
            "error_location": "line 7",
            "error_description": "Extra closing bracket",
        },
    ],

    # =========== TYPE ERRORS (High detectability) ===========
    "type_error": [
        {
            "id": "type_001",
            "complexity": "simple",
            "language": "python",
            "correct_code": '''
def format_message(count):
    return f"Found {count} items"
''',
            "buggy_code": '''
def format_message(count):
    return "Found " + count + " items"
''',
            "error_location": "line 2",
            "error_description": "String concatenation with integer (should use str() or f-string)",
        },
        {
            "id": "type_002",
            "complexity": "moderate",
            "language": "python",
            "correct_code": '''
def get_user_age(user_data):
    if user_data and "age" in user_data:
        return user_data["age"]
    return 0
''',
            "buggy_code": '''
def get_user_age(user_data):
    return user_data["age"]
''',
            "error_location": "line 2",
            "error_description": "No null check - will raise KeyError/TypeError if user_data is None or missing 'age'",
        },
        {
            "id": "type_003",
            "complexity": "moderate",
            "language": "javascript",
            "correct_code": '''
function calculateTotal(prices) {
    return prices.reduce((sum, p) => sum + p, 0);
}
''',
            "buggy_code": '''
function calculateTotal(prices) {
    return prices.reduce((sum, p) => sum + p);
}
''',
            "error_location": "line 2",
            "error_description": "Missing initial value for reduce - fails on empty array",
        },
    ],

    # =========== LOGIC ERRORS (Moderate detectability) ===========
    "logic_error": [
        {
            "id": "logic_001",
            "complexity": "simple",
            "language": "python",
            "correct_code": '''
def is_adult(age):
    return age >= 18
''',
            "buggy_code": '''
def is_adult(age):
    return age > 18
''',
            "error_location": "line 2",
            "error_description": "Off-by-one: should be >= 18, not > 18 (18-year-olds are adults)",
        },
        {
            "id": "logic_002",
            "complexity": "moderate",
            "language": "python",
            "correct_code": '''
def find_max(numbers):
    if not numbers:
        return None
    max_val = numbers[0]
    for n in numbers[1:]:
        if n > max_val:
            max_val = n
    return max_val
''',
            "buggy_code": '''
def find_max(numbers):
    if not numbers:
        return None
    max_val = numbers[0]
    for n in numbers[1:]:
        if n < max_val:
            max_val = n
    return max_val
''',
            "error_location": "line 6",
            "error_description": "Wrong comparison operator: < instead of > (finds min instead of max)",
        },
        {
            "id": "logic_003",
            "complexity": "complex",
            "language": "python",
            "correct_code": '''
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
''',
            "buggy_code": '''
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid
        else:
            right = mid
    return -1
''',
            "error_location": "lines 8, 10",
            "error_description": "Missing +1/-1 in boundary updates causes infinite loop",
        },
    ],

    # =========== EDGE CASE ERRORS (Moderate detectability) ===========
    "edge_case_error": [
        {
            "id": "edge_001",
            "complexity": "simple",
            "language": "python",
            "correct_code": '''
def get_first(items):
    if items:
        return items[0]
    return None
''',
            "buggy_code": '''
def get_first(items):
    return items[0]
''',
            "error_location": "line 2",
            "error_description": "No check for empty list - raises IndexError",
        },
        {
            "id": "edge_002",
            "complexity": "moderate",
            "language": "python",
            "correct_code": '''
def calculate_average(numbers):
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)
''',
            "buggy_code": '''
def calculate_average(numbers):
    return sum(numbers) / len(numbers)
''',
            "error_location": "line 2",
            "error_description": "Division by zero when list is empty",
        },
        {
            "id": "edge_003",
            "complexity": "moderate",
            "language": "javascript",
            "correct_code": '''
function safeParse(jsonString) {
    try {
        return JSON.parse(jsonString);
    } catch (e) {
        return null;
    }
}
''',
            "buggy_code": '''
function safeParse(jsonString) {
    return JSON.parse(jsonString);
}
''',
            "error_location": "line 2",
            "error_description": "No try-catch for invalid JSON input",
        },
    ],

    # =========== SECURITY ERRORS (Low detectability) ===========
    "security_error": [
        {
            "id": "sec_001",
            "complexity": "moderate",
            "language": "python",
            "correct_code": '''
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = ?"
    return db.execute(query, (user_id,))
''',
            "buggy_code": '''
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)
''',
            "error_location": "line 2",
            "error_description": "SQL injection vulnerability - user_id not parameterized",
        },
        {
            "id": "sec_002",
            "complexity": "moderate",
            "language": "python",
            "correct_code": '''
import os
def read_file(filename):
    safe_path = os.path.basename(filename)
    with open(os.path.join('/data', safe_path)) as f:
        return f.read()
''',
            "buggy_code": '''
def read_file(filename):
    with open(f'/data/{filename}') as f:
        return f.read()
''',
            "error_location": "line 2",
            "error_description": "Path traversal vulnerability - filename not sanitized",
        },
        {
            "id": "sec_003",
            "complexity": "complex",
            "language": "javascript",
            "correct_code": '''
function displayMessage(message) {
    const sanitized = message.replace(/</g, '&lt;').replace(/>/g, '&gt;');
    document.getElementById('output').textContent = sanitized;
}
''',
            "buggy_code": '''
function displayMessage(message) {
    document.getElementById('output').innerHTML = message;
}
''',
            "error_location": "line 2",
            "error_description": "XSS vulnerability - unsanitized HTML insertion",
        },
    ],

    # =========== PERFORMANCE ERRORS (Low detectability) ===========
    "performance_error": [
        {
            "id": "perf_001",
            "complexity": "moderate",
            "language": "python",
            "correct_code": '''
def find_duplicates(items):
    seen = set()
    duplicates = []
    for item in items:
        if item in seen:
            duplicates.append(item)
        seen.add(item)
    return duplicates
''',
            "buggy_code": '''
def find_duplicates(items):
    duplicates = []
    for i, item in enumerate(items):
        if item in items[:i]:
            duplicates.append(item)
    return duplicates
''',
            "error_location": "line 4",
            "error_description": "O(n²) algorithm instead of O(n) - checking list slice instead of set",
        },
        {
            "id": "perf_002",
            "complexity": "complex",
            "language": "python",
            "correct_code": '''
def get_users_with_orders(user_ids):
    users = User.query.filter(User.id.in_(user_ids)).all()
    orders = Order.query.filter(Order.user_id.in_(user_ids)).all()
    orders_by_user = {}
    for order in orders:
        orders_by_user.setdefault(order.user_id, []).append(order)
    return [(u, orders_by_user.get(u.id, [])) for u in users]
''',
            "buggy_code": '''
def get_users_with_orders(user_ids):
    users = User.query.filter(User.id.in_(user_ids)).all()
    result = []
    for user in users:
        orders = Order.query.filter(Order.user_id == user.id).all()
        result.append((user, orders))
    return result
''',
            "error_location": "line 5",
            "error_description": "N+1 query problem - querying orders inside loop",
        },
    ],

    # =========== CONCURRENCY ERRORS (Low detectability) ===========
    "concurrency_error": [
        {
            "id": "conc_001",
            "complexity": "complex",
            "language": "python",
            "correct_code": '''
import threading

counter = 0
lock = threading.Lock()

def increment():
    global counter
    with lock:
        counter += 1
''',
            "buggy_code": '''
import threading

counter = 0

def increment():
    global counter
    counter += 1
''',
            "error_location": "line 7",
            "error_description": "Race condition - counter increment not atomic/locked",
        },
        {
            "id": "conc_002",
            "complexity": "complex",
            "language": "python",
            "correct_code": '''
import threading

class BankAccount:
    def __init__(self):
        self.balance = 0
        self.lock = threading.Lock()

    def transfer(self, other, amount):
        # Always lock in consistent order to prevent deadlock
        first, second = sorted([self, other], key=id)
        with first.lock:
            with second.lock:
                if self.balance >= amount:
                    self.balance -= amount
                    other.balance += amount
''',
            "buggy_code": '''
import threading

class BankAccount:
    def __init__(self):
        self.balance = 0
        self.lock = threading.Lock()

    def transfer(self, other, amount):
        with self.lock:
            with other.lock:
                if self.balance >= amount:
                    self.balance -= amount
                    other.balance += amount
''',
            "error_location": "lines 9-10",
            "error_description": "Potential deadlock - locks acquired in inconsistent order",
        },
    ],
}


def get_tasks() -> list[dict[str, Any]]:
    """Generate all task configurations."""
    tasks = []
    task_id = 0

    for error_type, samples in CODE_SAMPLES.items():
        for sample in samples:
            tasks.append({
                "task_id": f"task_{task_id:04d}",
                "sample_id": sample["id"],
                "error_type": error_type,
                "complexity": sample["complexity"],
                "language": sample["language"],
                "buggy_code": sample["buggy_code"].strip(),
                "correct_code": sample["correct_code"].strip(),
                "error_location": sample["error_location"],
                "error_description": sample["error_description"],
                "expected_detectability": ERROR_TYPES[error_type]["expected_detectability"],
            })
            task_id += 1

    return tasks


def get_tasks_by_error_type(error_type: str) -> list[dict[str, Any]]:
    """Get tasks for a specific error type."""
    return [t for t in get_tasks() if t["error_type"] == error_type]


def get_tasks_by_detectability(detectability: str) -> list[dict[str, Any]]:
    """Get tasks by expected detectability level."""
    return [t for t in get_tasks() if t["expected_detectability"] == detectability]


def build_review_prompt(task: dict[str, Any]) -> str:
    """Build the code review prompt."""
    return f'''Review the following {task["language"]} code for bugs, errors, and issues.
Be thorough and identify any problems you find.

```{task["language"]}
{task["buggy_code"]}
```

List any bugs or errors you find, with:
1. Location (line number or function name)
2. Description of the issue
3. Suggested fix

If you find no issues, say "No issues found."'''


if __name__ == "__main__":
    tasks = get_tasks()
    print(f"Total tasks: {len(tasks)}")

    # Count by error type
    for error_type in ERROR_TYPES.keys():
        count = len([t for t in tasks if t["error_type"] == error_type])
        detectability = ERROR_TYPES[error_type]["expected_detectability"]
        print(f"  {error_type} ({detectability}): {count}")

    # Count by complexity
    for complexity in ["simple", "moderate", "complex"]:
        count = len([t for t in tasks if t["complexity"] == complexity])
        print(f"  {complexity}: {count}")
