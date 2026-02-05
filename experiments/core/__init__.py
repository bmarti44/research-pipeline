"""
Core modules for format friction analysis.

This package provides:
- Statistical utilities (bootstrap, Wilson CI, Cohen's kappa)
- LLM API providers with retry logic
- Tool-call correctness judging
- NL intent extraction
- System prompt assembly with ablation
- Experiment harness for between-subjects design
- Configuration management with model locking
"""

from .statistics import (
    wilson_ci,
    sign_test,
    cohens_kappa,
    bootstrap_ci,
    dip_test,
    compute_distribution_metrics,
    benjamini_hochberg,
    mcnemar_test,
)
from .bootstrap import (
    bootstrap_mean_ci,
    bootstrap_difference_ci,
    bootstrap_proportion_ci,
)
from .api_providers import (
    call_model,
    call_model_with_retry,
    check_api_keys,
    list_available_models,
    MODELS,
    APIResponse,
    RetryConfig,
    Provider,
    ModelConfig,
)
from .checkpoint import (
    write_checkpoint,
    read_checkpoint,
    compute_file_sha256,
    verify_checksum,
)
from .config import (
    get_model_config,
    get_config_manager,
    ConfigManager,
    ExperimentConfig,
    ModelVersions,
    EnvironmentInfo,
)
from .tools import (
    ToolDefinition,
    ToolComplexity,
    ALL_TOOLS,
    TOOLS_BY_COMPLEXITY,
    get_tool_by_name,
    get_tools_for_experiment,
    validate_tool_call,
    format_tools_for_anthropic,
    format_tools_for_openai,
)
from .prompts import (
    AblationCondition,
    OutputCondition,
    assemble_system_prompt,
    create_minimal_nl_prompt,
    create_minimal_json_prompt,
    create_full_nl_prompt,
    create_full_json_prompt,
)
from .extractor import (
    ExtractedIntent,
    ExtractionRubric,
    extract_intent,
    is_correct_intent,
    validate_extractor,
)
from .judge import (
    JudgmentResult,
    judge_response,
    judge_json_response,
    judge_nl_response,
    judge_with_llm,
    judge_with_all_families,
    validate_judge_human_agreement,
    CrossFamilyAgreement,
    compute_cross_family_agreement,
)
from .harness import (
    ExperimentHarness,
    Task,
    TrialResult,
    ManipulationCheck,
    compute_cluster_stats,
)
from .evaluation import (
    EvaluationMode,
    EvaluationResult,
    evaluate_response,
    evaluate_trial_all_modes,
    TOOL_EQUIVALENCE,
)

__all__ = [
    # Statistics
    "wilson_ci",
    "sign_test",
    "cohens_kappa",
    "bootstrap_ci",
    "dip_test",
    "compute_distribution_metrics",
    "benjamini_hochberg",
    "mcnemar_test",
    # Bootstrap
    "bootstrap_mean_ci",
    "bootstrap_difference_ci",
    "bootstrap_proportion_ci",
    # API Providers
    "call_model",
    "call_model_with_retry",
    "check_api_keys",
    "list_available_models",
    "MODELS",
    "APIResponse",
    "RetryConfig",
    "Provider",
    "ModelConfig",
    # Checkpoint
    "write_checkpoint",
    "read_checkpoint",
    "compute_file_sha256",
    "verify_checksum",
    # Config
    "get_model_config",
    "get_config_manager",
    "ConfigManager",
    "ExperimentConfig",
    "ModelVersions",
    "EnvironmentInfo",
    # Tools
    "ToolDefinition",
    "ToolComplexity",
    "ALL_TOOLS",
    "TOOLS_BY_COMPLEXITY",
    "get_tool_by_name",
    "get_tools_for_experiment",
    "validate_tool_call",
    "format_tools_for_anthropic",
    "format_tools_for_openai",
    # Prompts
    "AblationCondition",
    "OutputCondition",
    "assemble_system_prompt",
    "create_minimal_nl_prompt",
    "create_minimal_json_prompt",
    "create_full_nl_prompt",
    "create_full_json_prompt",
    # Extractor
    "ExtractedIntent",
    "ExtractionRubric",
    "extract_intent",
    "is_correct_intent",
    "validate_extractor",
    # Judge
    "JudgmentResult",
    "judge_response",
    "judge_json_response",
    "judge_nl_response",
    "judge_with_llm",
    "judge_with_all_families",
    "validate_judge_human_agreement",
    "CrossFamilyAgreement",
    "compute_cross_family_agreement",
    # Harness
    "ExperimentHarness",
    "Task",
    "TrialResult",
    "ManipulationCheck",
    "compute_cluster_stats",
    # Evaluation
    "EvaluationMode",
    "EvaluationResult",
    "evaluate_response",
    "evaluate_trial_all_modes",
    "TOOL_EQUIVALENCE",
]
