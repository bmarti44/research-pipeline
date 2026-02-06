"""
Tasks for Phase Separation Workflow Study.

Defines coding tasks across complexity levels and types for workflow comparison.
"""

from typing import Any
from dataclasses import dataclass


# =============================================================================
# Workflow Definitions
# =============================================================================

@dataclass
class WorkflowConfig:
    """Configuration for a workflow variant."""
    name: str
    phases: list[str]
    context_clearing: bool
    external_memory: bool
    description: str


WORKFLOWS = {
    "continuous_context": WorkflowConfig(
        name="continuous_context",
        phases=["plan_and_execute"],
        context_clearing=False,
        external_memory=False,
        description="All phases in one continuous context"
    ),
    "phases_no_clearing": WorkflowConfig(
        name="phases_no_clearing",
        phases=["plan", "review", "resolve", "execute"],
        context_clearing=False,
        external_memory=False,
        description="Distinct phases but no context clearing"
    ),
    "phases_with_clearing": WorkflowConfig(
        name="phases_with_clearing",
        phases=["plan", "review", "resolve", "execute"],
        context_clearing=True,
        external_memory=False,
        description="Distinct phases with context clearing"
    ),
    "external_memory_no_clearing": WorkflowConfig(
        name="external_memory_no_clearing",
        phases=["plan", "review", "resolve", "execute"],
        context_clearing=False,
        external_memory=True,
        description="External memory but no context clearing"
    ),
    "full_workflow": WorkflowConfig(
        name="full_workflow",
        phases=["plan", "review", "resolve", "execute"],
        context_clearing=True,
        external_memory=True,
        description="Full workflow: phases + clearing + external memory"
    ),
    "no_review": WorkflowConfig(
        name="no_review",
        phases=["plan", "execute"],
        context_clearing=True,
        external_memory=True,
        description="Plan → execute with clearing, no review"
    ),
    # Context collapse test conditions (same mode, tests memory effect)
    "iterative_refine_in_context": WorkflowConfig(
        name="iterative_refine_in_context",
        phases=["plan", "refine", "refine", "refine", "execute"],
        context_clearing=False,
        external_memory=False,
        description="Context collapse test: multiple refinements in continuous context"
    ),
    "iterative_refine_external": WorkflowConfig(
        name="iterative_refine_external",
        phases=["plan", "refine", "refine", "refine", "execute"],
        context_clearing=True,
        external_memory=True,
        description="Context collapse control: same refinements with external memory"
    ),
}


# =============================================================================
# Coding Tasks
# =============================================================================

CODING_TASKS = [
    # =========== SIMPLE TASKS ===========
    {
        "task_id": "simple_001",
        "complexity": "simple",
        "task_type": "new_feature",
        "title": "Add a function to calculate factorial",
        "description": """
Create a Python function `factorial(n)` that:
- Takes a non-negative integer n
- Returns n! (n factorial)
- Handles edge cases (0! = 1, negative numbers raise ValueError)
- Include docstring and type hints
""",
        "test_cases": [
            {"input": 0, "expected": 1},
            {"input": 1, "expected": 1},
            {"input": 5, "expected": 120},
            {"input": 10, "expected": 3628800},
            {"input": -1, "expected": "ValueError"},
        ],
        "language": "python",
    },
    {
        "task_id": "simple_002",
        "complexity": "simple",
        "task_type": "bug_fix",
        "title": "Fix the palindrome checker",
        "description": """
The following palindrome checker has bugs. Fix them:

```python
def is_palindrome(s):
    return s == s.reverse()
```

It should:
- Handle case-insensitively
- Ignore non-alphanumeric characters
- Return True for palindromes, False otherwise
""",
        "test_cases": [
            {"input": "racecar", "expected": True},
            {"input": "A man, a plan, a canal: Panama", "expected": True},
            {"input": "hello", "expected": False},
            {"input": "", "expected": True},
        ],
        "language": "python",
    },
    {
        "task_id": "simple_003",
        "complexity": "simple",
        "task_type": "refactoring",
        "title": "Refactor nested conditionals",
        "description": """
Refactor this deeply nested code to be more readable:

```python
def process_user(user):
    if user is not None:
        if user.is_active:
            if user.email is not None:
                if '@' in user.email:
                    return send_welcome_email(user)
                else:
                    return "Invalid email"
            else:
                return "No email"
        else:
            return "User inactive"
    else:
        return "No user"
```

Use early returns or guard clauses.
""",
        "test_cases": [],  # Manual review for refactoring
        "language": "python",
    },

    # =========== MODERATE TASKS ===========
    {
        "task_id": "moderate_001",
        "complexity": "moderate",
        "task_type": "new_feature",
        "title": "Implement a LRU Cache",
        "description": """
Implement an LRU (Least Recently Used) cache class with:
- `__init__(self, capacity: int)` - Initialize with given capacity
- `get(self, key: int) -> int` - Return value or -1 if not found
- `put(self, key: int, value: int) -> None` - Insert/update key-value
- O(1) time complexity for both operations

When cache is full, evict the least recently used item.
""",
        "test_cases": [
            {"operations": ["put(1,1)", "put(2,2)", "get(1)", "put(3,3)", "get(2)"],
             "expected": [None, None, 1, None, -1]},  # 2 was evicted
        ],
        "language": "python",
    },
    {
        "task_id": "moderate_002",
        "complexity": "moderate",
        "task_type": "bug_fix",
        "title": "Fix the async rate limiter",
        "description": """
This rate limiter has race conditions. Fix them:

```python
import asyncio
import time

class RateLimiter:
    def __init__(self, rate: int, per: float):
        self.rate = rate
        self.per = per
        self.tokens = rate
        self.last_update = time.time()

    async def acquire(self):
        now = time.time()
        elapsed = now - self.last_update
        self.tokens += elapsed * (self.rate / self.per)
        self.tokens = min(self.tokens, self.rate)
        self.last_update = now

        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False
```

Add proper locking for thread/coroutine safety.
""",
        "test_cases": [],
        "language": "python",
    },
    {
        "task_id": "moderate_003",
        "complexity": "moderate",
        "task_type": "integration",
        "title": "Add pagination to API endpoint",
        "description": """
Add pagination support to this FastAPI endpoint:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
items_db = [{"id": i, "name": f"Item {i}"} for i in range(100)]

@app.get("/items")
async def get_items():
    return items_db
```

Requirements:
- Accept `page` (default 1) and `page_size` (default 10, max 50) query params
- Return paginated response with `items`, `total`, `page`, `page_size`, `total_pages`
- Handle edge cases (page out of range, invalid params)
""",
        "test_cases": [
            {"params": {}, "expected_total": 100, "expected_count": 10},
            {"params": {"page": 2}, "expected_count": 10},
            {"params": {"page_size": 20}, "expected_count": 20},
            {"params": {"page": 100}, "expected_count": 0},
        ],
        "language": "python",
    },

    # =========== COMPLEX TASKS ===========
    {
        "task_id": "complex_001",
        "complexity": "complex",
        "task_type": "new_feature",
        "title": "Implement a task scheduler with dependencies",
        "description": """
Implement a task scheduler that handles dependencies:

```python
class TaskScheduler:
    def add_task(self, task_id: str, duration: int, dependencies: list[str] = None):
        '''Add a task with optional dependencies'''
        pass

    def get_execution_order(self) -> list[str]:
        '''Return valid execution order (topological sort)'''
        pass

    def get_minimum_completion_time(self) -> int:
        '''Return minimum time to complete all tasks (parallel execution)'''
        pass

    def detect_cycles(self) -> bool:
        '''Return True if there are circular dependencies'''
        pass
```

Requirements:
- Handle parallel execution where possible
- Detect circular dependencies
- Calculate critical path for minimum completion time
""",
        "test_cases": [
            {
                "tasks": [("A", 3, []), ("B", 2, ["A"]), ("C", 4, ["A"]), ("D", 1, ["B", "C"])],
                "expected_order_valid": True,  # A must come before B, C; B, C before D
                "expected_min_time": 8,  # A(3) + C(4) + D(1) = critical path
            },
        ],
        "language": "python",
    },
    {
        "task_id": "complex_002",
        "complexity": "complex",
        "task_type": "bug_fix",
        "title": "Fix the distributed lock implementation",
        "description": """
This Redis-based distributed lock has several bugs. Fix them:

```python
import redis
import time
import uuid

class DistributedLock:
    def __init__(self, redis_client, lock_name, expire_time=10):
        self.redis = redis_client
        self.lock_name = lock_name
        self.expire_time = expire_time
        self.lock_id = str(uuid.uuid4())

    def acquire(self):
        return self.redis.set(self.lock_name, self.lock_id)

    def release(self):
        self.redis.delete(self.lock_name)

    def __enter__(self):
        while not self.acquire():
            time.sleep(0.1)
        return self

    def __exit__(self, *args):
        self.release()
```

Bugs to fix:
- acquire() doesn't use NX flag (allows overwrite)
- No expiration set (lock never expires if holder crashes)
- release() doesn't check ownership (can release others' locks)
- No retry limit in __enter__ (infinite loop possible)
- Race condition in release (check-then-delete)
""",
        "test_cases": [],
        "language": "python",
    },
    {
        "task_id": "complex_003",
        "complexity": "complex",
        "task_type": "refactoring",
        "title": "Extract microservice from monolith",
        "description": """
Refactor this monolithic order processing code into a clean service interface
that could be deployed as a microservice:

```python
# Current monolithic code
class OrderProcessor:
    def __init__(self, db):
        self.db = db

    def process_order(self, order_data):
        # Validate
        if not order_data.get('items'):
            raise ValueError("No items")

        # Calculate total
        total = 0
        for item in order_data['items']:
            product = self.db.query(f"SELECT price FROM products WHERE id={item['product_id']}")
            total += product['price'] * item['quantity']

        # Apply discount
        user = self.db.query(f"SELECT * FROM users WHERE id={order_data['user_id']}")
        if user['is_premium']:
            total *= 0.9

        # Create order
        order_id = self.db.execute(f"INSERT INTO orders ...")

        # Update inventory
        for item in order_data['items']:
            self.db.execute(f"UPDATE inventory SET quantity = quantity - {item['quantity']} ...")

        # Send notification
        self.send_email(user['email'], f"Order {order_id} confirmed")

        return order_id
```

Create:
- Clean service interface with dependency injection
- Separate concerns (validation, pricing, inventory, notifications)
- Error handling and transaction management
- Type hints and documentation
""",
        "test_cases": [],
        "language": "python",
    },
    {
        "task_id": "complex_004",
        "complexity": "complex",
        "task_type": "integration",
        "title": "Add WebSocket support for real-time updates",
        "description": """
Add WebSocket support to this REST API for real-time order status updates:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
orders = {}  # order_id -> order_data

class Order(BaseModel):
    id: str
    status: str
    items: list

@app.post("/orders")
async def create_order(order: Order):
    orders[order.id] = order.dict()
    return {"id": order.id}

@app.patch("/orders/{order_id}/status")
async def update_status(order_id: str, status: str):
    orders[order_id]["status"] = status
    return orders[order_id]
```

Add:
- WebSocket endpoint at `/ws/orders/{order_id}` for subscribing to order updates
- Broadcast status changes to all connected clients for that order
- Handle connection/disconnection gracefully
- Add heartbeat/ping mechanism
- Support multiple orders per connection
""",
        "test_cases": [],
        "language": "python",
    },
]


# =============================================================================
# Workflow Phase Prompts
# =============================================================================

PHASE_PROMPTS = {
    "plan": """You are a senior software engineer creating a detailed implementation plan.

TASK:
{task_description}

Create a comprehensive plan that includes:
1. Understanding of requirements and edge cases
2. High-level approach
3. Detailed step-by-step implementation plan
4. Potential pitfalls and how to avoid them
5. Testing strategy

Output your plan in markdown format.
""",

    "review": """You are a critical code reviewer examining an implementation plan.

PLAN TO REVIEW:
{plan_content}

ORIGINAL TASK:
{task_description}

Critically review this plan. Look for:
1. Missing edge cases or requirements
2. Potential bugs or logic errors
3. Performance issues
4. Security concerns
5. Better approaches that were missed

Be thorough and critical. Don't assume the plan is correct.
Output specific issues with severity (critical/major/minor) and suggestions.
""",

    "resolve": """You are resolving issues identified in a code review.

ORIGINAL PLAN:
{plan_content}

REVIEW FINDINGS:
{review_content}

ORIGINAL TASK:
{task_description}

Address each issue from the review:
1. For each finding, explain how you'll address it
2. Update the plan to fix critical and major issues
3. Note any findings you disagree with and why

Output the revised plan in markdown format.
""",

    "execute": """You are implementing code based on a plan.

PLAN:
{plan_content}

ORIGINAL TASK:
{task_description}

Implement the code following the plan exactly.
Include all necessary imports, error handling, and documentation.
Output only the final code.
""",

    "plan_and_execute": """You are a senior software engineer implementing a feature.

TASK:
{task_description}

First, think through your approach, then implement the solution.
Include all necessary imports, error handling, and documentation.
Output your complete implementation.
""",

    "refine": """You are refining an implementation plan to make it more robust.

CURRENT PLAN:
{plan_content}

ORIGINAL TASK:
{task_description}

Improve this plan by:
1. Adding more specific implementation details
2. Considering additional edge cases
3. Strengthening error handling approach
4. Improving the testing strategy

Output the refined plan in markdown format. Be thorough and detailed.
""",
}


def get_tasks() -> list[dict[str, Any]]:
    """Generate all task configurations."""
    tasks = []
    task_id = 0

    for coding_task in CODING_TASKS:
        for workflow_name, workflow in WORKFLOWS.items():
            tasks.append({
                "task_id": f"task_{task_id:04d}",
                "coding_task_id": coding_task["task_id"],
                "workflow": workflow_name,
                "workflow_config": {
                    "phases": workflow.phases,
                    "context_clearing": workflow.context_clearing,
                    "external_memory": workflow.external_memory,
                },
                "title": coding_task["title"],
                "description": coding_task["description"],
                "complexity": coding_task["complexity"],
                "task_type": coding_task["task_type"],
                "test_cases": coding_task["test_cases"],
                "language": coding_task["language"],
            })
            task_id += 1

    return tasks


def get_phase_prompt(phase: str, **kwargs) -> str:
    """Get the prompt for a specific phase."""
    template = PHASE_PROMPTS.get(phase, "")
    return template.format(**kwargs)


def get_tasks_by_workflow(workflow: str) -> list[dict[str, Any]]:
    """Get all tasks for a specific workflow."""
    return [t for t in get_tasks() if t["workflow"] == workflow]


def get_tasks_by_complexity(complexity: str) -> list[dict[str, Any]]:
    """Get all tasks for a specific complexity level."""
    return [t for t in get_tasks() if t["complexity"] == complexity]


if __name__ == "__main__":
    tasks = get_tasks()
    print(f"Total tasks: {len(tasks)}")

    # Count by workflow
    for workflow in WORKFLOWS.keys():
        count = len([t for t in tasks if t["workflow"] == workflow])
        print(f"  {workflow}: {count}")

    # Count by complexity
    for complexity in ["simple", "moderate", "complex"]:
        count = len([t for t in tasks if t["complexity"] == complexity])
        print(f"  {complexity}: {count}")

    # Count by task type
    for task_type in ["new_feature", "bug_fix", "refactoring", "integration"]:
        count = len([t for t in tasks if t["task_type"] == task_type])
        print(f"  {task_type}: {count}")
