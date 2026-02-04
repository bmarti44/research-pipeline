"""
Core modules for format friction analysis.

This package provides shared utilities for statistical analysis,
LLM judging, CLI wrappers for subscription-based model access,
and checkpoint management.
"""

from .statistics import (
    wilson_ci,
    sign_test,
    cohens_kappa,
    bootstrap_ci,
    dip_test,
    compute_distribution_metrics,
)
from .bootstrap import bootstrap_mean_ci, bootstrap_difference_ci
from .judge import claude_judge, gpt4_judge, agnostic_judge
from .cli_wrappers import call_claude, call_codex, call_gemini, CLIResponse
from .checkpoint import (
    write_checkpoint,
    read_checkpoint,
    compute_file_sha256,
    verify_checksum,
)

__all__ = [
    # Statistics
    "wilson_ci",
    "sign_test",
    "cohens_kappa",
    "bootstrap_ci",
    "dip_test",
    "compute_distribution_metrics",
    # Bootstrap
    "bootstrap_mean_ci",
    "bootstrap_difference_ci",
    # Judge
    "claude_judge",
    "gpt4_judge",
    "agnostic_judge",
    # CLI Wrappers
    "call_claude",
    "call_codex",
    "call_gemini",
    "CLIResponse",
    # Checkpoint
    "write_checkpoint",
    "read_checkpoint",
    "compute_file_sha256",
    "verify_checksum",
]
