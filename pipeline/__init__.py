"""
Research Pipeline for LLM Behavioral Studies.

A turn-key pipeline for running reproducible LLM experiments with:
- Generic, reusable components (Conditions, Tasks, Trials, Scenarios)
- Study isolation (each hypothesis in its own folder)
- Stage-gated execution with automated review gates
- Deterministic reproducibility (locked environments, seeded RNG)
- Preregistration with tamper-evident hashing
- Pilot studies with automatic progression criteria
- Adaptive stopping rules (O'Brien-Fleming, Pocock)
- Multi-model support (model as IV)
- Replication framework (direct, conceptual, extension)
- Efficient execution with batching and concurrency

Usage:
    python -m pipeline new my_study --template basic
    python -m pipeline run my_study --all
    python -m pipeline run my_study --full  # Full pipeline with gates
    python -m pipeline verify my_study --all
    python -m pipeline prereg my_study  # Create preregistration
    python -m pipeline pilot my_study   # Run pilot study
    python -m pipeline replicate my_study --type direct
    python -m pipeline status
"""

__version__ = "0.4.0"

# Utilities
from .utils import (
    setup_logging,
    get_logger,
    hash_string,
    hash_file as util_hash_file,
    hash_directory as util_hash_directory,
    hash_dict,
    verify_hash,
    now_utc,
    now_iso,
    parse_iso,
    format_duration,
    PipelineJSONEncoder,
    json_dumps,
    json_dump,
    json_load,
    ensure_dir,
    safe_write,
    read_yaml,
    write_yaml,
    get_env,
    get_project_root,
    get_papers_dir,
    get_templates_dir,
    is_valid_study_name,
    is_valid_condition_name,
    validate_probability,
    validate_positive_int,
)

# Schema validation
from .schemas import (
    ValidationError,
    ValidationResult,
    validate_identifier,
    validate_study_config,
    validate_condition,
    validate_model_config,
    validate_task,
    validate_tasks_file,
    validate_tool,
    validate_trial,
    validate_response,
    validate_study_directory,
    validate_config_file,
    validate_all,
)

# Analysis module
from .analysis import (
    ConditionAggregate,
    AnalysisResult,
    aggregate_scores,
    wilson_ci,
    run_statistical_tests,
    check_assumptions,
    run_analysis as analysis_run_analysis,
    analyze_by_category,
    generate_analysis_summary,
)

# Manuscript generation
from .manuscript import (
    ManuscriptSection,
    Table,
    Figure,
    Manuscript,
    generate_introduction,
    generate_methods,
    generate_results,
    generate_discussion,
    generate_results_table,
    generate_test_table,
    generate_manuscript,
    save_manuscript,
)

# PDF export
from .pdf_export import (
    PDFConfig,
    check_pandoc_available,
    check_latex_available,
    check_wkhtmltopdf_available,
    markdown_to_pdf,
    markdown_string_to_pdf,
    html_to_pdf,
    markdown_to_html,
    export_results_pdf,
    export_manuscript_pdf,
    export_all_studies_pdf,
    get_available_pdf_engines,
    print_pdf_setup_instructions,
)

# Core verification
from .verification import (
    VerificationResult,
    CheckResult,
    verify_stage,
    verify_all_stages,
    hash_file,
    hash_directory,
)

# Stage execution
from .runner import (
    run_stage,
    run_all_stages,
    run_full_pipeline,
    run_stage_with_gates,
    run_preregistration,
    run_pilot,
    run_with_adaptive_stopping,
    StageStatus,
    StageResult,
    get_study_status,
    load_study_config,
)

# Preregistration
from .preregistration import (
    PreregistrationRecord,
    PreregistrationVerification,
    create_preregistration,
    verify_preregistration,
    check_preregistration_exists,
    get_preregistration_hash,
)

# Pilot studies
from .pilot import (
    PilotCriteria,
    PilotResult,
    evaluate_pilot,
    create_pilot_config,
    check_pilot_gate,
    save_pilot_result,
)

# Adaptive stopping
from .adaptive import (
    AdaptiveConfig,
    InterimAnalysis,
    StoppingRule,
    should_stop,
    perform_interim_analysis,
    save_interim_analysis,
    get_adjusted_alpha,
    compute_alpha_spending,
)

# Multi-model support
from .multimodel import (
    ModelConfig,
    ModelAsIV,
    load_models_config,
    create_model_iv,
    generate_model_trials,
    analyze_model_effects,
    verify_model_versions,
)

# Replication
from .replication import (
    ReplicationRecord,
    ReplicationResult,
    create_direct_replication,
    create_conceptual_replication,
    compare_replication_results,
    verify_replication_integrity,
    generate_replication_report,
)

# Review gates
from .review_gates import (
    GateResult,
    run_gates,
    save_gate_results,
    print_gate_results,
    check_interview_gate,
    check_preregistration_gate,
    check_pilot_gate,
    check_execution_integrity_gate,
    check_evaluation_determinism_gate,
    check_preregistration_compliance_gate,
    check_adaptive_stopping_gate,
)

# Interview system
from .interview import (
    InterviewQuestion,
    HypothesisSpec,
    get_interview_questions,
    check_completeness,
    build_hypothesis_spec,
    save_interview,
    load_interview,
    format_interview_for_display,
    verify_interview_complete,
)

# Review system
from .review import (
    Review,
    ReviewFinding,
    ReviewSeverity,
    ReviewCategory,
    REVIEWER_PERSPECTIVES,
    get_review_prompt,
    aggregate_reviews,
    save_reviews,
    load_reviews,
    get_unresolved_findings,
    generate_review_summary,
)

# Research orchestrator
from .orchestrator import (
    ResearchState,
    generate_research_plan,
    save_research_state,
    load_research_state,
    get_next_action,
    format_review_prompt_for_agent,
    check_all_reviews_complete,
    check_all_critical_resolved,
)

# Generic models (core building blocks)
from .models import (
    Condition,
    Task,
    Trial,
    Response,
    Evaluation,
    EvaluationMode,
    Scenario,
    StudyDesign,
    Tool,
    ToolParameter,
    load_study_design,
    load_tasks_from_file,
    load_tools_from_file,
)

# Statistical tests registry
from .stats import (
    TestResult,
    register_test,
    get_test,
    list_tests,
    run_test,
    run_analysis,
    compute_power,
)

# Evaluator registry
from .evaluators import (
    EvalResult,
    register_evaluator,
    get_evaluator,
    list_evaluators,
    evaluate_response,
    evaluate_batch,
    run_all_evaluators,
)

# Execution with batching/concurrency
from .executor import (
    ExecutionConfig,
    ExecutionCheckpoint,
    ExecutionResult,
    RateLimiter,
    TrialExecutor,
    AsyncTrialExecutor,
    execute_trials,
    estimate_execution_time,
)

__all__ = [
    # Version
    "__version__",
    # Utilities
    "setup_logging",
    "get_logger",
    "hash_string",
    "hash_dict",
    "verify_hash",
    "now_utc",
    "now_iso",
    "parse_iso",
    "format_duration",
    "PipelineJSONEncoder",
    "json_dumps",
    "json_dump",
    "json_load",
    "ensure_dir",
    "safe_write",
    "read_yaml",
    "write_yaml",
    "get_env",
    "get_project_root",
    "get_papers_dir",
    "get_templates_dir",
    "is_valid_study_name",
    "is_valid_condition_name",
    "validate_probability",
    "validate_positive_int",
    # Schema validation
    "ValidationError",
    "ValidationResult",
    "validate_identifier",
    "validate_study_config",
    "validate_condition",
    "validate_model_config",
    "validate_task",
    "validate_tasks_file",
    "validate_tool",
    "validate_trial",
    "validate_response",
    "validate_study_directory",
    "validate_config_file",
    "validate_all",
    # Analysis
    "ConditionAggregate",
    "AnalysisResult",
    "aggregate_scores",
    "wilson_ci",
    "run_statistical_tests",
    "check_assumptions",
    "analyze_by_category",
    "generate_analysis_summary",
    # Manuscript
    "ManuscriptSection",
    "Table",
    "Figure",
    "Manuscript",
    "generate_introduction",
    "generate_methods",
    "generate_results",
    "generate_discussion",
    "generate_results_table",
    "generate_test_table",
    "generate_manuscript",
    "save_manuscript",
    # PDF export
    "PDFConfig",
    "check_pandoc_available",
    "check_latex_available",
    "check_wkhtmltopdf_available",
    "markdown_to_pdf",
    "markdown_string_to_pdf",
    "html_to_pdf",
    "markdown_to_html",
    "export_results_pdf",
    "export_manuscript_pdf",
    "export_all_studies_pdf",
    "get_available_pdf_engines",
    "print_pdf_setup_instructions",
    # Verification
    "VerificationResult",
    "CheckResult",
    "verify_stage",
    "verify_all_stages",
    "hash_file",
    "hash_directory",
    # Runner
    "run_stage",
    "run_all_stages",
    "run_full_pipeline",
    "run_stage_with_gates",
    "run_preregistration",
    "run_pilot",
    "run_with_adaptive_stopping",
    "StageStatus",
    "StageResult",
    "get_study_status",
    "load_study_config",
    # Preregistration
    "PreregistrationRecord",
    "PreregistrationVerification",
    "create_preregistration",
    "verify_preregistration",
    "check_preregistration_exists",
    "get_preregistration_hash",
    # Pilot
    "PilotCriteria",
    "PilotResult",
    "evaluate_pilot",
    "create_pilot_config",
    "check_pilot_gate",
    "save_pilot_result",
    # Adaptive
    "AdaptiveConfig",
    "InterimAnalysis",
    "StoppingRule",
    "should_stop",
    "perform_interim_analysis",
    "save_interim_analysis",
    "get_adjusted_alpha",
    "compute_alpha_spending",
    # Multi-model
    "ModelConfig",
    "ModelAsIV",
    "load_models_config",
    "create_model_iv",
    "generate_model_trials",
    "analyze_model_effects",
    "verify_model_versions",
    # Replication
    "ReplicationRecord",
    "ReplicationResult",
    "create_direct_replication",
    "create_conceptual_replication",
    "compare_replication_results",
    "verify_replication_integrity",
    "generate_replication_report",
    # Review gates
    "GateResult",
    "run_gates",
    "save_gate_results",
    "print_gate_results",
    "check_preregistration_gate",
    "check_pilot_gate",
    "check_execution_integrity_gate",
    "check_evaluation_determinism_gate",
    "check_preregistration_compliance_gate",
    "check_adaptive_stopping_gate",
    # Interview
    "InterviewQuestion",
    "HypothesisSpec",
    "get_interview_questions",
    "check_completeness",
    "build_hypothesis_spec",
    "save_interview",
    "load_interview",
    "format_interview_for_display",
    "verify_interview_complete",
    # Review
    "Review",
    "ReviewFinding",
    "ReviewSeverity",
    "ReviewCategory",
    "REVIEWER_PERSPECTIVES",
    "get_review_prompt",
    "aggregate_reviews",
    "save_reviews",
    "load_reviews",
    "get_unresolved_findings",
    "generate_review_summary",
    # Orchestrator
    "ResearchState",
    "generate_research_plan",
    "save_research_state",
    "load_research_state",
    "get_next_action",
    "format_review_prompt_for_agent",
    "check_all_reviews_complete",
    "check_all_critical_resolved",
    # Generic models
    "Condition",
    "Task",
    "Trial",
    "Response",
    "Evaluation",
    "EvaluationMode",
    "Scenario",
    "StudyDesign",
    "Tool",
    "ToolParameter",
    "load_study_design",
    "load_tasks_from_file",
    "load_tools_from_file",
    # Statistical tests
    "TestResult",
    "register_test",
    "get_test",
    "list_tests",
    "run_test",
    "run_analysis",
    # Evaluators
    "EvalResult",
    "register_evaluator",
    "get_evaluator",
    "list_evaluators",
    "evaluate_response",
    "evaluate_batch",
    "run_all_evaluators",
    # Execution
    "ExecutionConfig",
    "ExecutionCheckpoint",
    "ExecutionResult",
    "RateLimiter",
    "TrialExecutor",
    "AsyncTrialExecutor",
    "execute_trials",
    "estimate_execution_time",
]
