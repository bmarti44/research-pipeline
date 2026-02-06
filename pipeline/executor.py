"""
Trial execution with batching and concurrency.

Handles efficient execution of trials with:
- Configurable batch sizes
- Concurrent API calls with rate limiting
- Automatic retry with exponential backoff
- Progress tracking and checkpointing
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from .models import Trial, Response, Task, Condition, StudyDesign


logger = logging.getLogger(__name__)


# =============================================================================
# Execution Configuration
# =============================================================================

@dataclass
class ExecutionConfig:
    """Configuration for trial execution."""

    # Batching
    batch_size: int = 10  # Trials per batch
    max_concurrent: int = 5  # Concurrent API calls within batch

    # Retry
    max_retries: int = 3
    base_delay: float = 1.0  # Base delay for exponential backoff
    max_delay: float = 60.0  # Maximum delay between retries

    # Rate limiting
    requests_per_minute: int = 60
    tokens_per_minute: int = 100000

    # Timeouts
    request_timeout: float = 120.0  # Per-request timeout
    batch_timeout: float = 600.0  # Per-batch timeout

    # Checkpointing
    checkpoint_frequency: int = 10  # Checkpoint every N trials
    checkpoint_path: Optional[Path] = None

    # Progress
    show_progress: bool = True
    progress_callback: Optional[Callable[[int, int, str], None]] = None

    def to_dict(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "max_concurrent": self.max_concurrent,
            "max_retries": self.max_retries,
            "base_delay": self.base_delay,
            "max_delay": self.max_delay,
            "requests_per_minute": self.requests_per_minute,
            "tokens_per_minute": self.tokens_per_minute,
            "request_timeout": self.request_timeout,
            "batch_timeout": self.batch_timeout,
            "checkpoint_frequency": self.checkpoint_frequency,
        }


# =============================================================================
# Rate Limiter
# =============================================================================

class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        tokens_per_minute: int = 100000,
    ):
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute

        # Token buckets
        self.request_tokens = requests_per_minute
        self.token_tokens = tokens_per_minute

        # Refill rate (per second)
        self.request_refill = requests_per_minute / 60.0
        self.token_refill = tokens_per_minute / 60.0

        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int = 1000) -> float:
        """
        Acquire permission to make a request.

        Returns the delay (if any) before the request should be made.
        """
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_refill

            # Refill buckets
            self.request_tokens = min(
                self.rpm,
                self.request_tokens + elapsed * self.request_refill
            )
            self.token_tokens = min(
                self.tpm,
                self.token_tokens + elapsed * self.token_refill
            )
            self.last_refill = now

            # Calculate delay needed
            delay = 0.0

            if self.request_tokens < 1:
                delay = max(delay, (1 - self.request_tokens) / self.request_refill)

            if self.token_tokens < estimated_tokens:
                delay = max(delay, (estimated_tokens - self.token_tokens) / self.token_refill)

            if delay > 0:
                return delay

            # Consume tokens
            self.request_tokens -= 1
            self.token_tokens -= estimated_tokens

            return 0.0

    def sync_acquire(self, estimated_tokens: int = 1000) -> float:
        """Synchronous version of acquire."""
        now = time.time()
        elapsed = now - self.last_refill

        # Refill buckets
        self.request_tokens = min(
            self.rpm,
            self.request_tokens + elapsed * self.request_refill
        )
        self.token_tokens = min(
            self.tpm,
            self.token_tokens + elapsed * self.token_refill
        )
        self.last_refill = now

        # Calculate delay needed
        delay = 0.0

        if self.request_tokens < 1:
            delay = max(delay, (1 - self.request_tokens) / self.request_refill)

        if self.token_tokens < estimated_tokens:
            delay = max(delay, (estimated_tokens - self.token_tokens) / self.token_refill)

        if delay > 0:
            return delay

        # Consume tokens
        self.request_tokens -= 1
        self.token_tokens -= estimated_tokens

        return 0.0


# =============================================================================
# Checkpoint Manager
# =============================================================================

@dataclass
class ExecutionCheckpoint:
    """Checkpoint state for resumable execution."""
    study_name: str
    total_trials: int
    completed_trial_ids: list[str]
    failed_trial_ids: list[str]
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "study_name": self.study_name,
            "total_trials": self.total_trials,
            "completed_trial_ids": self.completed_trial_ids,
            "failed_trial_ids": self.failed_trial_ids,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExecutionCheckpoint":
        return cls(**data)

    def save(self, path: Path) -> None:
        """Save checkpoint to file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ExecutionCheckpoint":
        """Load checkpoint from file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# Execution Result
# =============================================================================

@dataclass
class ExecutionResult:
    """Result of executing all trials."""
    total: int
    completed: int
    failed: int
    skipped: int
    responses: list[Response]
    errors: list[dict]
    duration_seconds: float
    checkpoint: Optional[ExecutionCheckpoint] = None

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "skipped": self.skipped,
            "duration_seconds": self.duration_seconds,
            "completion_rate": self.completed / self.total if self.total > 0 else 0,
            "error_rate": self.failed / self.total if self.total > 0 else 0,
        }


# =============================================================================
# Trial Executor
# =============================================================================

class TrialExecutor:
    """
    Executes trials with batching and concurrency.

    Usage:
        executor = TrialExecutor(api_client, config)
        result = executor.execute(trials, study_design)
    """

    def __init__(
        self,
        api_client: Any,  # LLM API client
        config: Optional[ExecutionConfig] = None,
    ):
        self.api = api_client
        self.config = config or ExecutionConfig()
        self.rate_limiter = RateLimiter(
            requests_per_minute=self.config.requests_per_minute,
            tokens_per_minute=self.config.tokens_per_minute,
        )

    def execute(
        self,
        trials: list[Trial],
        study_design: StudyDesign,
        resume_from: Optional[ExecutionCheckpoint] = None,
    ) -> ExecutionResult:
        """
        Execute all trials with batching and concurrency.

        Args:
            trials: List of trials to execute
            study_design: Study design (for task/condition lookup)
            resume_from: Optional checkpoint to resume from
        """
        start_time = time.time()

        # Filter out already completed trials if resuming
        if resume_from:
            completed_ids = set(resume_from.completed_trial_ids)
            trials = [t for t in trials if t.trial_id not in completed_ids]
            logger.info(f"Resuming: {len(trials)} trials remaining")

        responses = []
        errors = []
        completed_ids = list(resume_from.completed_trial_ids) if resume_from else []
        failed_ids = list(resume_from.failed_trial_ids) if resume_from else []

        # Split into batches
        batches = self._create_batches(trials)
        total_batches = len(batches)

        for batch_idx, batch in enumerate(batches):
            batch_num = batch_idx + 1
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} trials)")

            # Execute batch with concurrency
            batch_responses, batch_errors = self._execute_batch(batch, study_design)

            responses.extend(batch_responses)
            errors.extend(batch_errors)

            # Update tracking
            for resp in batch_responses:
                if resp.error:
                    failed_ids.append(resp.trial_id)
                else:
                    completed_ids.append(resp.trial_id)

            # Checkpoint
            if self.config.checkpoint_path and batch_num % self.config.checkpoint_frequency == 0:
                checkpoint = ExecutionCheckpoint(
                    study_name=study_design.name,
                    total_trials=len(trials),
                    completed_trial_ids=completed_ids,
                    failed_trial_ids=failed_ids,
                )
                checkpoint.save(self.config.checkpoint_path)
                logger.info(f"Checkpoint saved: {len(completed_ids)} completed")

            # Progress callback
            if self.config.progress_callback:
                self.config.progress_callback(
                    len(completed_ids),
                    len(trials),
                    f"Batch {batch_num}/{total_batches}",
                )

        duration = time.time() - start_time

        return ExecutionResult(
            total=len(trials) + len(completed_ids) if resume_from else len(trials),
            completed=len([r for r in responses if not r.error]),
            failed=len([r for r in responses if r.error]),
            skipped=len(completed_ids) if resume_from else 0,
            responses=responses,
            errors=errors,
            duration_seconds=duration,
        )

    def _create_batches(self, trials: list[Trial]) -> list[list[Trial]]:
        """Split trials into batches."""
        batches = []
        for i in range(0, len(trials), self.config.batch_size):
            batches.append(trials[i:i + self.config.batch_size])
        return batches

    def _execute_batch(
        self,
        batch: list[Trial],
        study_design: StudyDesign,
    ) -> tuple[list[Response], list[dict]]:
        """Execute a single batch with concurrency."""
        responses = []
        errors = []

        with ThreadPoolExecutor(max_workers=self.config.max_concurrent) as executor:
            futures = {
                executor.submit(self._execute_trial, trial, study_design): trial
                for trial in batch
            }

            for future in as_completed(futures, timeout=self.config.batch_timeout):
                trial = futures[future]
                try:
                    response = future.result()
                    responses.append(response)
                except Exception as e:
                    logger.error(f"Trial {trial.trial_id} failed: {e}")
                    errors.append({
                        "trial_id": trial.trial_id,
                        "error": str(e),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                    # Create error response
                    responses.append(Response(
                        trial_id=trial.trial_id,
                        model="unknown",
                        raw_output="",
                        error=str(e),
                    ))

        return responses, errors

    def _execute_trial(
        self,
        trial: Trial,
        study_design: StudyDesign,
    ) -> Response:
        """Execute a single trial with retry logic."""
        task = trial.task or study_design.get_task(trial.task_id)
        condition = trial.condition or study_design.get_condition(trial.condition_name)

        if not task:
            raise ValueError(f"Task not found: {trial.task_id}")
        if not condition:
            raise ValueError(f"Condition not found: {trial.condition_name}")

        # Build prompt
        prompt = self._build_prompt(task, condition)

        # Execute with retry
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                # Rate limiting
                delay = self.rate_limiter.sync_acquire(estimated_tokens=1000)
                if delay > 0:
                    time.sleep(delay)

                # Make API call
                start_time = time.time()
                result = self._call_api(prompt, task, condition)
                latency = (time.time() - start_time) * 1000

                return Response(
                    trial_id=trial.trial_id,
                    model=result.get("model", "unknown"),
                    raw_output=result.get("content", ""),
                    latency_ms=latency,
                    input_tokens=result.get("input_tokens"),
                    output_tokens=result.get("output_tokens"),
                    request_id=result.get("request_id"),
                    retry_count=attempt,
                )

            except Exception as e:
                last_error = e
                logger.warning(f"Trial {trial.trial_id} attempt {attempt + 1} failed: {e}")

                # Exponential backoff
                delay = min(
                    self.config.base_delay * (2 ** attempt),
                    self.config.max_delay,
                )
                time.sleep(delay)

        # All retries failed
        raise last_error or Exception("Unknown error")

    def _build_prompt(self, task: Task, condition: Condition) -> dict:
        """Build the prompt for a trial."""
        system = condition.system_instructions or ""
        user = task.prompt

        if task.context:
            user = f"{task.context}\n\n{user}"

        return {
            "system": system,
            "user": user,
            "tools": task.tools if task.tools else None,
        }

    def _call_api(
        self,
        prompt: dict,
        task: Task,
        condition: Condition,
    ) -> dict:
        """Make the actual API call. Override for different providers."""
        # This is a placeholder - actual implementation depends on API client
        # In practice, this would call self.api.messages.create() or similar
        raise NotImplementedError("Subclass must implement _call_api")


# =============================================================================
# Async Executor (for higher throughput)
# =============================================================================

class AsyncTrialExecutor(TrialExecutor):
    """
    Async version of TrialExecutor for maximum throughput.

    Use when you need to execute many trials quickly and your API
    client supports async operations.
    """

    async def execute_async(
        self,
        trials: list[Trial],
        study_design: StudyDesign,
        resume_from: Optional[ExecutionCheckpoint] = None,
    ) -> ExecutionResult:
        """Execute all trials asynchronously."""
        start_time = time.time()

        # Filter out already completed trials if resuming
        if resume_from:
            completed_ids = set(resume_from.completed_trial_ids)
            trials = [t for t in trials if t.trial_id not in completed_ids]

        responses = []
        errors = []
        completed_ids = list(resume_from.completed_trial_ids) if resume_from else []
        failed_ids = list(resume_from.failed_trial_ids) if resume_from else []

        # Process in batches
        batches = self._create_batches(trials)

        for batch_idx, batch in enumerate(batches):
            # Execute batch concurrently
            tasks = [
                self._execute_trial_async(trial, study_design)
                for trial in batch
            ]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for trial, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    errors.append({
                        "trial_id": trial.trial_id,
                        "error": str(result),
                    })
                    failed_ids.append(trial.trial_id)
                    responses.append(Response(
                        trial_id=trial.trial_id,
                        model="unknown",
                        raw_output="",
                        error=str(result),
                    ))
                else:
                    responses.append(result)
                    if result.error:
                        failed_ids.append(result.trial_id)
                    else:
                        completed_ids.append(result.trial_id)

        duration = time.time() - start_time

        return ExecutionResult(
            total=len(trials),
            completed=len([r for r in responses if not r.error]),
            failed=len([r for r in responses if r.error]),
            skipped=0,
            responses=responses,
            errors=errors,
            duration_seconds=duration,
        )

    async def _execute_trial_async(
        self,
        trial: Trial,
        study_design: StudyDesign,
    ) -> Response:
        """Execute a single trial asynchronously."""
        task = trial.task or study_design.get_task(trial.task_id)
        condition = trial.condition or study_design.get_condition(trial.condition_name)

        if not task or not condition:
            raise ValueError(f"Task or condition not found for trial {trial.trial_id}")

        prompt = self._build_prompt(task, condition)

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                delay = await self.rate_limiter.acquire(estimated_tokens=1000)
                if delay > 0:
                    await asyncio.sleep(delay)

                start_time = time.time()
                result = await self._call_api_async(prompt, task, condition)
                latency = (time.time() - start_time) * 1000

                return Response(
                    trial_id=trial.trial_id,
                    model=result.get("model", "unknown"),
                    raw_output=result.get("content", ""),
                    latency_ms=latency,
                    input_tokens=result.get("input_tokens"),
                    output_tokens=result.get("output_tokens"),
                    request_id=result.get("request_id"),
                    retry_count=attempt,
                )

            except Exception as e:
                last_error = e
                delay = min(
                    self.config.base_delay * (2 ** attempt),
                    self.config.max_delay,
                )
                await asyncio.sleep(delay)

        raise last_error or Exception("Unknown error")

    async def _call_api_async(
        self,
        prompt: dict,
        task: Task,
        condition: Condition,
    ) -> dict:
        """Async API call. Override for different providers."""
        raise NotImplementedError("Subclass must implement _call_api_async")


# =============================================================================
# Convenience Functions
# =============================================================================

def execute_trials(
    trials: list[Trial],
    study_design: StudyDesign,
    api_client: Any,
    config: Optional[ExecutionConfig] = None,
    checkpoint_path: Optional[Path] = None,
) -> ExecutionResult:
    """
    Convenience function to execute trials.

    Args:
        trials: List of trials to execute
        study_design: Study design
        api_client: LLM API client
        config: Optional execution config
        checkpoint_path: Optional path for checkpointing
    """
    if config is None:
        config = ExecutionConfig()

    if checkpoint_path:
        config.checkpoint_path = checkpoint_path

    executor = TrialExecutor(api_client, config)

    # Check for existing checkpoint
    resume_from = None
    if checkpoint_path and checkpoint_path.exists():
        resume_from = ExecutionCheckpoint.load(checkpoint_path)
        logger.info(f"Resuming from checkpoint: {len(resume_from.completed_trial_ids)} already completed")

    return executor.execute(trials, study_design, resume_from)


def estimate_execution_time(
    n_trials: int,
    config: Optional[ExecutionConfig] = None,
    avg_latency_ms: float = 2000,
) -> dict:
    """
    Estimate execution time for a study.

    Returns dict with estimates for different scenarios.
    """
    config = config or ExecutionConfig()

    # Calculate based on rate limits
    requests_per_second = config.requests_per_minute / 60

    # Serial execution time
    serial_time = n_trials * (avg_latency_ms / 1000)

    # Rate-limited time
    rate_limited_time = n_trials / requests_per_second

    # Concurrent time (with batching)
    batch_time = (n_trials / config.batch_size) * (avg_latency_ms / 1000) / min(
        config.max_concurrent,
        config.requests_per_minute // 10,  # Conservative concurrent requests
    )

    # Actual time is max of rate limit and concurrent execution
    estimated_time = max(rate_limited_time, batch_time)

    return {
        "n_trials": n_trials,
        "serial_seconds": serial_time,
        "rate_limited_seconds": rate_limited_time,
        "estimated_seconds": estimated_time,
        "estimated_minutes": estimated_time / 60,
        "config": config.to_dict(),
    }
