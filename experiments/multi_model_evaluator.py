"""
Multi-model evaluation harness for reasoning-action gap experiments.

Supports: Claude, GPT-4, Llama, Gemini
"""

import asyncio
import json
import re
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any
from enum import Enum

# Import scenarios
from experiments.scenarios.proactive_tools import (
    ALL_SCENARIOS, Scenario, ToolType, ExplicitnessLevel,
    get_scenarios_by_tool, get_scenarios_by_level
)


class ModelProvider(Enum):
    CLAUDE = "claude"
    OPENAI = "openai"
    # GOOGLE = "google"
    # TOGETHER = "together"


@dataclass
class TrialResult:
    scenario_id: str
    model: str
    condition: str  # "reasoning" or "action"
    query: str
    tool_type: str
    level: str
    expected_action: bool
    # Results
    success: bool
    identified_content: Optional[str] = None
    tool_called: bool = False
    tool_call_content: Optional[str] = None
    response_text: str = ""
    error: Optional[str] = None
    # Metadata
    latency_ms: Optional[float] = None
    tokens_used: Optional[int] = None


@dataclass
class ExperimentResults:
    experiment_id: str
    model: str
    timestamp: str
    total_scenarios: int
    # Aggregate metrics
    reasoning_accuracy: float
    action_accuracy: float
    gap: float
    # By level
    by_level: dict = field(default_factory=dict)
    # By tool type
    by_tool: dict = field(default_factory=dict)
    # Raw results
    trials: list = field(default_factory=list)


class ModelEvaluator(ABC):
    """Abstract base class for model evaluators."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    async def evaluate_reasoning(self, scenario: Scenario) -> TrialResult:
        """Evaluate reasoning condition: identify what should be done."""
        pass

    @abstractmethod
    async def evaluate_action(self, scenario: Scenario) -> TrialResult:
        """Evaluate action condition: actually call the tool."""
        pass


class ClaudeEvaluator(ModelEvaluator):
    """Evaluator for Claude models using the Agent SDK."""

    def __init__(self, model_name: str = "claude-sonnet-4-5-20250929"):
        super().__init__(model_name)
        # Import here to avoid dependency issues
        from claude_agent_sdk import (
            ClaudeSDKClient, ClaudeAgentOptions, HookMatcher,
            AssistantMessage, TextBlock
        )
        self.ClaudeSDKClient = ClaudeSDKClient
        self.ClaudeAgentOptions = ClaudeAgentOptions
        self.HookMatcher = HookMatcher
        self.AssistantMessage = AssistantMessage
        self.TextBlock = TextBlock

    async def evaluate_reasoning(self, scenario: Scenario) -> TrialResult:
        """Reasoning condition: identify what should be saved."""

        system_prompt = self._get_reasoning_prompt(scenario.tool_type)

        options = self.ClaudeAgentOptions(
            allowed_tools=[],
            max_turns=1,
            permission_mode="acceptEdits",
            system_prompt=system_prompt,
        )

        response_text = ""
        start_time = datetime.now()

        try:
            async with self.ClaudeSDKClient(options=options) as client:
                await client.query(scenario.query)
                async for message in client.receive_response():
                    if isinstance(message, self.AssistantMessage):
                        for block in message.content:
                            if isinstance(block, self.TextBlock):
                                response_text += block.text
        except Exception as e:
            return TrialResult(
                scenario_id=scenario.id,
                model=self.model_name,
                condition="reasoning",
                query=scenario.query,
                tool_type=scenario.tool_type.value,
                level=scenario.level.name,
                expected_action=scenario.expected_action,
                success=False,
                error=str(e),
            )

        latency = (datetime.now() - start_time).total_seconds() * 1000

        # Parse response for [PERSIST], [LOG], [ACTION] tags
        identified, content = self._parse_reasoning_response(response_text, scenario.tool_type)

        # Determine success
        if scenario.expected_action:
            success = identified
        else:
            success = not identified

        return TrialResult(
            scenario_id=scenario.id,
            model=self.model_name,
            condition="reasoning",
            query=scenario.query,
            tool_type=scenario.tool_type.value,
            level=scenario.level.name,
            expected_action=scenario.expected_action,
            success=success,
            identified_content=content,
            response_text=response_text[:500],
            latency_ms=latency,
        )

    async def evaluate_action(self, scenario: Scenario) -> TrialResult:
        """Action condition: actually call the tool."""

        tool_calls = []

        async def track_tool_calls(input_data, tool_use_id, context):
            tool_name = input_data.get("tool_name", "")
            if tool_name == "Bash":
                command = input_data.get("tool_input", {}).get("command", "")
                # Check for our tool commands
                if any(cmd in command for cmd in ["save-memory", "log-task", "track-analytics", "bookmark-file"]):
                    tool_calls.append(command)
                    return {"result": "Success: Action recorded."}
            return {}

        system_prompt = self._get_action_prompt(scenario.tool_type)

        options = self.ClaudeAgentOptions(
            allowed_tools=["Bash"],
            max_turns=2,
            permission_mode="acceptEdits",
            system_prompt=system_prompt,
            hooks={
                "PreToolUse": [
                    self.HookMatcher(matcher="*", hooks=[track_tool_calls])
                ],
            }
        )

        response_text = ""
        start_time = datetime.now()

        try:
            async with self.ClaudeSDKClient(options=options) as client:
                await client.query(scenario.query)
                async for message in client.receive_response():
                    if isinstance(message, self.AssistantMessage):
                        for block in message.content:
                            if isinstance(block, self.TextBlock):
                                response_text += block.text[:500]
        except Exception as e:
            return TrialResult(
                scenario_id=scenario.id,
                model=self.model_name,
                condition="action",
                query=scenario.query,
                tool_type=scenario.tool_type.value,
                level=scenario.level.name,
                expected_action=scenario.expected_action,
                success=False,
                error=str(e),
            )

        latency = (datetime.now() - start_time).total_seconds() * 1000

        tool_called = len(tool_calls) > 0

        # Determine success
        if scenario.expected_action:
            success = tool_called
        else:
            success = not tool_called

        return TrialResult(
            scenario_id=scenario.id,
            model=self.model_name,
            condition="action",
            query=scenario.query,
            tool_type=scenario.tool_type.value,
            level=scenario.level.name,
            expected_action=scenario.expected_action,
            success=success,
            tool_called=tool_called,
            tool_call_content=tool_calls[0] if tool_calls else None,
            response_text=response_text[:500],
            latency_ms=latency,
        )

    def _get_reasoning_prompt(self, tool_type: ToolType) -> str:
        """Get system prompt for reasoning condition based on tool type."""

        if tool_type == ToolType.MEMORY:
            return """You are a helpful assistant. When the user shares information,
respond naturally but ALSO explicitly state what information (if any) should be
saved to a persistent memory system for future conversations.

Format your response as:
1. Your natural response to the user
2. [PERSIST]: List any information that should be saved, or "None" if nothing needs saving

Categories for memory: codebase, user_preference, decision, task_completed, other

You do NOT have any tools - just identify what SHOULD be saved."""

        elif tool_type == ToolType.TASK_LOG:
            return """You are a helpful assistant. When the user mentions completed work,
respond naturally but ALSO explicitly state what tasks (if any) should be
logged for record-keeping.

Format your response as:
1. Your natural response to the user
2. [LOG]: Describe the completed task to log, or "None" if nothing to log

You do NOT have any tools - just identify what SHOULD be logged."""

        else:
            return """You are a helpful assistant. Respond naturally to the user."""

    def _get_action_prompt(self, tool_type: ToolType) -> str:
        """Get system prompt for action condition based on tool type."""

        if tool_type == ToolType.MEMORY:
            return """You are a helpful assistant with access to a memory tool.

To save important information for future conversations, run:
    save-memory "content" "category"

Where category is one of: codebase, user_preference, decision, task_completed, other

IMPORTANT: Use this tool whenever you learn something important about:
- The codebase structure or conventions
- User preferences
- Decisions that were made
- Completed tasks"""

        elif tool_type == ToolType.TASK_LOG:
            return """You are a helpful assistant with access to a task logging tool.

To log completed tasks, run:
    log-task "description" "category"

Where category is one of: bugfix, feature, refactor, docs, testing, deployment, other

IMPORTANT: Use this tool whenever the user mentions:
- Completing a task
- Fixing a bug
- Finishing implementation
- Successful deployments"""

        else:
            return """You are a helpful assistant."""

    def _parse_reasoning_response(self, response: str, tool_type: ToolType) -> tuple[bool, Optional[str]]:
        """Parse reasoning response for identification tags."""

        tag = "[PERSIST]" if tool_type == ToolType.MEMORY else "[LOG]"

        # Try to find the tag
        pattern = rf'\[(?:PERSIST|LOG)\]:?\s*(.+?)(?:\n\n|$)'
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)

        if match:
            content = match.group(1).strip()
            # Check if it's "None" or similar
            if content.lower() in ["none", "nothing", "n/a", "none."]:
                return False, None
            if "nothing" in content.lower() and len(content) < 50:
                return False, None
            return True, content

        return False, None


async def run_experiment(
    evaluator: ModelEvaluator,
    scenarios: list[Scenario],
    conditions: list[str] = ["reasoning", "action"],
    delay_between_calls: float = 1.0,
) -> ExperimentResults:
    """Run full experiment with given evaluator and scenarios."""

    experiment_id = f"{evaluator.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    trials = []

    print(f"\n{'='*60}")
    print(f"Running experiment: {experiment_id}")
    print(f"Model: {evaluator.model_name}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Conditions: {conditions}")
    print(f"{'='*60}\n")

    for i, scenario in enumerate(scenarios):
        print(f"[{i+1}/{len(scenarios)}] {scenario.id} ({scenario.level.name})")

        for condition in conditions:
            if condition == "reasoning":
                result = await evaluator.evaluate_reasoning(scenario)
            else:
                result = await evaluator.evaluate_action(scenario)

            trials.append(result)
            status = "✓" if result.success else "✗"
            print(f"  {condition}: {status}")

            await asyncio.sleep(delay_between_calls)

    # Compute aggregate metrics
    reasoning_trials = [t for t in trials if t.condition == "reasoning" and t.expected_action]
    action_trials = [t for t in trials if t.condition == "action" and t.expected_action]

    reasoning_acc = sum(1 for t in reasoning_trials if t.success) / len(reasoning_trials) if reasoning_trials else 0
    action_acc = sum(1 for t in action_trials if t.success) / len(action_trials) if action_trials else 0

    # By level
    by_level = {}
    for level in ExplicitnessLevel:
        if level == ExplicitnessLevel.CONTROL:
            continue
        level_reasoning = [t for t in reasoning_trials if t.level == level.name]
        level_action = [t for t in action_trials if t.level == level.name]
        if level_reasoning:
            by_level[level.name] = {
                "reasoning": sum(1 for t in level_reasoning if t.success) / len(level_reasoning),
                "action": sum(1 for t in level_action if t.success) / len(level_action) if level_action else 0,
                "n": len(level_reasoning),
            }

    # By tool type
    by_tool = {}
    for tool in ToolType:
        tool_reasoning = [t for t in reasoning_trials if t.tool_type == tool.value]
        tool_action = [t for t in action_trials if t.tool_type == tool.value]
        if tool_reasoning:
            by_tool[tool.value] = {
                "reasoning": sum(1 for t in tool_reasoning if t.success) / len(tool_reasoning),
                "action": sum(1 for t in tool_action if t.success) / len(tool_action) if tool_action else 0,
                "n": len(tool_reasoning),
            }

    results = ExperimentResults(
        experiment_id=experiment_id,
        model=evaluator.model_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_scenarios=len(scenarios),
        reasoning_accuracy=reasoning_acc,
        action_accuracy=action_acc,
        gap=reasoning_acc - action_acc,
        by_level=by_level,
        by_tool=by_tool,
        trials=[asdict(t) for t in trials],
    )

    return results


def print_results(results: ExperimentResults):
    """Print formatted experiment results."""

    print(f"\n{'='*60}")
    print(f"RESULTS: {results.model}")
    print(f"{'='*60}")

    print(f"\n--- Overall ---")
    print(f"Reasoning Accuracy: {results.reasoning_accuracy:.1%}")
    print(f"Action Accuracy:    {results.action_accuracy:.1%}")
    print(f"Gap:                {results.gap:+.1%}")

    print(f"\n--- By Explicitness Level ---")
    for level, data in results.by_level.items():
        gap = data["reasoning"] - data["action"]
        print(f"  {level:15}: R={data['reasoning']:.1%} A={data['action']:.1%} Gap={gap:+.1%} (n={data['n']})")

    print(f"\n--- By Tool Type ---")
    for tool, data in results.by_tool.items():
        gap = data["reasoning"] - data["action"]
        print(f"  {tool:15}: R={data['reasoning']:.1%} A={data['action']:.1%} Gap={gap:+.1%} (n={data['n']})")


def save_results(results: ExperimentResults, output_dir: str = "experiments/results"):
    """Save experiment results to JSON."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(output_dir) / f"{results.experiment_id}.json"

    with open(filepath, "w") as f:
        json.dump(asdict(results), f, indent=2)

    print(f"\nResults saved to {filepath}")
    return filepath


async def main():
    """Run the main experiment."""

    # Use Claude for now
    evaluator = ClaudeEvaluator()

    # Filter to scenarios that should trigger action (exclude controls for main analysis)
    test_scenarios = [s for s in ALL_SCENARIOS if s.expected_action]

    # Add some control scenarios
    control_scenarios = [s for s in ALL_SCENARIOS if not s.expected_action][:10]
    all_test = test_scenarios + control_scenarios

    print(f"Testing {len(test_scenarios)} positive scenarios + {len(control_scenarios)} controls")

    results = await run_experiment(
        evaluator=evaluator,
        scenarios=all_test,
        conditions=["reasoning", "action"],
        delay_between_calls=1.0,
    )

    print_results(results)
    save_results(results)

    return results


if __name__ == "__main__":
    asyncio.run(main())
