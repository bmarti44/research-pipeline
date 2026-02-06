# Research Pipeline: Architecture & Design

## Vision

A **fully autonomous, academically rigorous research pipeline** that takes a natural language hypothesis about LLM/AI behavior and produces a complete, PDF-ready academic paper. Orchestrated by a Claude Code agent operating as a hierarchical supervisor with 9 specialized expert sub-agents.

**Design Principles:**
1. **Paper-first organization** — Each paper is a self-contained project; only its data lives in its directory
2. **Study-scoped isolation** — Only data/config relevant to a study lives in that study's directory
3. **Generic, reusable components** — Core building blocks (Conditions, Tasks, Trials, Scenarios) defined once in `pipeline/`, instantiated via YAML
4. **Stage-gated execution** — No stage proceeds without programmatic proof the previous stage passed
5. **5-round expert review** — Every major milestone reviewed by 5 specialized sub-agents with structured, verifiable output
6. **Deterministic verification** — Every gate is a boolean check against data on disk; nothing proceeds on trust
7. **Checkpoint-driven resumability** — State written to disk after every milestone; pipeline can resume from any checkpoint
8. **Reproducibility by construction** — Seeded randomization, environment locks, hashed preregistration, archived data

---

## Pipeline Overview

```
HYPOTHESIS → INTERVIEW → DESIGN → [5-REVIEW] → PREREGISTRATION → PILOT → [5-REVIEW]
    → CHECKPOINT → MAIN STUDY → [5-REVIEW] → CHECKPOINT → (ADD STUDIES?) → CROSS-STUDY
    → MANUSCRIPT → [5-REVIEW] → CHECKPOINT → PUBLISHABILITY → PDF EXPORT → CHECKPOINT
```

| Stage | Purpose | Expert Agents Involved | Produces |
|-------|---------|----------------------|----------|
| Interview | Clarify hypothesis with user | Orchestrator | `interview.yaml` |
| Design | Create experimental specification | Designer, Methodologist, Statistician, Domain Expert | `config.yaml`, `tasks.yaml`, `analysis_plan.yaml` |
| Preregistration | Lock design before data collection | Orchestrator | `preregistration/` (hashed, timestamped) |
| Pilot | Validate apparatus works | Domain Expert, Technical Reviewer | `pilot/` (data + pass/fail) |
| Main Study | Collect full dataset | Domain Expert, Technical Reviewer | `stages/` (responses, scores, analysis) |
| Cross-Study | Synthesize across studies | Statistician | `combined_analysis/` |
| Manuscript | Write the paper | Writing Specialist | `manuscript/` |
| PDF Export | Final output | Orchestrator | `paper_final.pdf` |

---

## Agent Architecture

### Supervisor Pattern

The Orchestrator spawns sub-agents via Claude Code's Task tool. Sub-agents are stateless — they receive full context in their task description and produce structured file output.

```
                    ┌──────────────────────┐
                    │    ORCHESTRATOR       │
                    │  (this agent)         │
                    │                       │
                    │  • Plan stages        │
                    │  • Spawn sub-agents   │
                    │  • Verify gates       │
                    │  • Write checkpoints  │
                    │  • Make decisions      │
                    └──────────┬───────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                     │
    ┌─────▼──────┐     ┌──────▼───────┐     ┌──────▼───────┐
    │ DESIGN TEAM│     │EXECUTION TEAM│     │ REVIEW TEAM  │
    │            │     │              │     │              │
    │ Designer   │     │ Domain Expert│     │ Methodologist│
    │ Method.    │     │ Statistician │     │ Statistician │
    │ Statist.   │     │ Technical    │     │ Skeptic      │
    │ Domain Exp.│     │              │     │ Technical    │
    │            │     │              │     │ Ethics/Domain│
    └────────────┘     └──────────────┘     └──────────────┘
```

### Sub-Agent Personas (9 total)

| Persona | Primary Responsibility | Key Questions They Ask |
|---------|----------------------|----------------------|
| **Research Designer** | Hypothesis → research plan | "Is this falsifiable? What's the comparison?" |
| **Methodologist** | Design rigor, confounds | "What else could explain this? Is the design balanced?" |
| **Statistician** | Analysis planning, power | "Is N sufficient? Are assumptions met? What corrections?" |
| **Domain Expert** | LLM-specific knowledge | "Does this prompt actually test what we think? API gotchas?" |
| **Skeptic** | Adversarial critique | "What's the weakest link? What would a reviewer attack?" |
| **Technical Reviewer** | Code, reproducibility | "Does this actually run? Can someone else reproduce it?" |
| **Writing Specialist** | Scientific prose, APA | "Is this clear? Is the logic sound? Are claims supported?" |
| **Ethics Reviewer** | Bias, responsible AI | "Are we reporting honestly? Any societal concerns?" |
| **Replication Specialist** | Independent verification | "Can I reproduce these results from the archived data?" |

---

## Core Data Models

All models are defined in `pipeline/models.py` and instantiated via YAML configuration.

### Condition
```python
@dataclass
class Condition:
    name: str                           # e.g., "json_mode", "nl_mode"
    description: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    prompt_template: Optional[str] = None
    system_instructions: Optional[str] = None
```

### Task
```python
@dataclass
class Task:
    id: str                             # e.g., "task_001"
    prompt: str                         # The actual prompt text
    category: str = "default"           # Grouping (e.g., "file_ops", "arithmetic")
    expected: dict[str, Any] = field(default_factory=dict)  # Ground truth
    metadata: dict[str, Any] = field(default_factory=dict)
    context: Optional[str] = None       # Additional context
    tools: list[dict] = field(default_factory=list)          # Tool definitions
```

### Trial
```python
@dataclass
class Trial:
    trial_id: str                       # Unique ID
    task_id: str                        # References Task.id
    condition_name: str                 # References Condition.name
    repetition: int                     # Which repetition (1-indexed)
    seed: Optional[int] = None          # Per-trial seed
    order: Optional[int] = None         # Randomized presentation order
```

### Scenario
```python
@dataclass
class Scenario:
    id: str                             # e.g., "file_operations"
    description: str = ""
    shared_context: Optional[str] = None
    shared_tools: list[dict] = field(default_factory=list)
    task_ids: list[str] = field(default_factory=list)
```

### StudyDesign
```python
@dataclass
class StudyDesign:
    name: str
    conditions: list[Condition]
    tasks: list[Task]
    scenarios: list[Scenario]
    design_type: str                    # "between", "within", "mixed"
    repetitions: int
    seed: int
    evaluation_modes: list[str]
    statistical_tests: list[str]        # Names from stats registry
    alpha: float
```

---

## Registries

### Statistical Tests (`pipeline/stats.py`)

Tests are selected by name in `config.yaml`. The registry provides:

| Test Name | When to Use | Data Type |
|-----------|-------------|-----------|
| `two_proportion_z` | Compare two success rates | Binary |
| `chi_square` | Independence in contingency tables | Categorical |
| `fisher_exact` | Small sample 2×2 tables | Binary |
| `mcnemar` | Paired nominal data (within-subjects) | Binary |
| `independent_t` | Compare two group means | Continuous |
| `paired_t` | Before/after comparisons | Continuous |
| `mann_whitney` | Non-parametric two-group comparison | Ordinal/continuous |
| `wilcoxon` | Non-parametric paired comparison | Ordinal/continuous |
| `one_way_anova` | Compare 3+ group means | Continuous |
| `kruskal_wallis` | Non-parametric 3+ group comparison | Ordinal/continuous |
| `pearson_correlation` | Linear relationship | Continuous |
| `spearman_correlation` | Monotonic relationship | Ordinal |
| `bootstrap_proportion_diff` | CI for proportion differences | Binary |
| `bootstrap_mean_diff` | CI for mean differences | Continuous |
| `cohens_kappa` | Inter-rater agreement | Categorical |
| `icc` | Intraclass correlation | Continuous |
| `logistic_regression` | Binary outcome with predictors | Binary |
| `mixed_effects_logistic` | Nested binary outcome data | Binary |

**Effect sizes** are computed automatically with the appropriate test:
- Binary/categorical: Cramér's V, φ, odds ratio
- Continuous: Cohen's d, η², r
- All with 95% confidence intervals

### Evaluator Registry (`pipeline/evaluators.py`)

| Evaluator | What It Checks | Modes |
|-----------|---------------|-------|
| `exact_match` | Exact string equality | strict, case_insensitive |
| `contains` | Substring presence | strict, any, all |
| `regex` | Pattern match | strict |
| `tool_call` | Correct tool + parameters | strict, intent, functional |
| `tool_call_nl` | NL description of tool use | semantic |
| `json_valid` | Valid JSON syntax | strict |
| `json_schema` | JSON matches schema | strict, intent |
| `numeric` | Numeric answer correctness | strict, tolerance |
| `classification` | Category label match | strict, intent |
| `llm_judge` | LLM evaluates quality | rubric-based, pairwise |

---

## Execution Engine (`pipeline/executor.py`)

### Configuration
```python
@dataclass
class ExecutionConfig:
    batch_size: int = 10            # Trials per batch
    max_concurrent: int = 5         # Parallel API calls
    max_retries: int = 3            # Per-trial retry limit
    base_delay: float = 1.0         # Backoff base (seconds)
    max_delay: float = 60.0         # Max backoff (seconds)
    requests_per_minute: int = 60   # Rate limit
    tokens_per_minute: int = 100000
    checkpoint_frequency: int = 10  # Save every N trials
```

### Execution Flow
```
Trials (N=500)
    ├── Batch 1 (10 trials) ──┬── Concurrent (max 5)
    │                         └── [Checkpoint at trial 10]
    ├── Batch 2 (10 trials)
    └── ... (50 batches)
```

### Features
- **Rate limiting**: Token bucket algorithm (requests/min + tokens/min)
- **Checkpointing**: Resumable execution; auto-save every N trials
- **Retry logic**: Exponential backoff with jitter
- **Async support**: `AsyncTrialExecutor` for maximum throughput
- **Response validation**: Each response validated for structure before saving

---

## Verification Gates

Every stage transition has a programmatic gate. Gates return boolean pass/fail with structured reasons.

### Gate Registry

| Gate ID | Before Stage | Checks | Blocking |
|---------|-------------|--------|----------|
| `G01_interview` | Design | Interview complete, all categories answered | Yes |
| `G02_design_schema` | Preregistration | Config validates against Pydantic schema | Yes |
| `G03_design_review` | Preregistration | 5-round review complete, no unresolved critical | Yes |
| `G04_prereg_lock` | Pilot | Preregistration hashed and locked | Yes |
| `G05_pilot_pass` | Main Study | Pilot metrics meet success criteria | Yes |
| `G06_pilot_review` | Main Study | 5-round review complete | Yes |
| `G07_prereg_intact` | Execute | Preregistration hash unchanged | Yes |
| `G08_config_locked` | Generate | Environment lock exists | Yes |
| `G09_trials_complete` | Execute | All trial combinations generated | Yes |
| `G10_execution_integrity` | Evaluate | ≥95% completion, ≤5% errors | Yes |
| `G11_data_integrity` | Evaluate | SHA-256 checksums valid | Yes |
| `G12_eval_determinism` | Analyze | Re-run 10% evaluations = identical | Yes |
| `G13_assumptions_checked` | Report | Statistical assumptions documented | Yes |
| `G14_study_review` | Manuscript | 5-round review complete | Yes |
| `G15_manuscript_complete` | PDF Export | All sections present, no placeholders | Yes |
| `G16_manuscript_review` | Publishability | 5-round review complete | Yes |
| `G17_prereg_compliance` | PDF Export | No undocumented deviations | Warning |
| `G18_cross_references` | PDF Export | All stats trace to data files | Yes |

### Gate Implementation Pattern

```python
def verify_gate(gate_id: str, study_path: Path) -> GateResult:
    """Returns GateResult with passed: bool, reasons: list[str], blocking: bool"""
    checks = GATE_REGISTRY[gate_id]
    results = [check(study_path) for check in checks]
    passed = all(r.passed for r in results)
    return GateResult(
        gate_id=gate_id,
        passed=passed,
        blocking=checks.blocking,
        reasons=[r.reason for r in results if not r.passed],
        timestamp=now_iso()
    )
```

---

## 5-Round Review Protocol

Triggered after: Design, Pilot, Main Study, Manuscript.

| Round | Reviewer | Focus Area |
|-------|----------|-----------|
| 1 | Methodologist | Design validity, confounds, threats to internal validity |
| 2 | Statistician | Analysis correctness, assumptions, power, corrections |
| 3 | Skeptic | Alternative explanations, logical gaps, fatal flaws |
| 4 | Technical Reviewer | Code correctness, data integrity, reproducibility |
| 5 | Ethics / Domain Expert | Bias, responsible reporting, LLM-specific concerns |

### Output Structure
```
reviews/<checkpoint>/
├── round_1_methodologist.json    # Structured findings
├── round_2_statistician.json
├── round_3_skeptic.json
├── round_4_technical.json
├── round_5_ethics_domain.json
└── review_summary.json           # Orchestrator's synthesis
```

### Finding Schema
```json
{
  "id": "F001",
  "severity": "critical|major|minor|suggestion",
  "category": "design|analysis|code|writing|ethics",
  "description": "...",
  "location": "file:line",
  "recommendation": "...",
  "resolution_required": true
}
```

### Resolution Rules
- **Critical**: Must resolve before proceeding. No exceptions.
- **Major**: Must resolve OR explicitly acknowledge as limitation in paper.
- **Minor/Suggestion**: Discretionary; document decision either way.
- **Re-review**: If a round returns "fail", revise and re-run that round only (max 3 retries).

---

## State Checkpoints

Written to disk after every major milestone for resumability and audit trail.

### Checkpoint Locations
```
papers/<paper>/checkpoints/
├── checkpoint_001_design_complete.json
├── checkpoint_002_pilot_complete.json
├── checkpoint_003_study1_complete.json
├── checkpoint_004_manuscript_complete.json
├── checkpoint_005_published.json
└── PROGRESS.md
```

### Checkpoint Schema
```json
{
  "checkpoint_id": "NNN",
  "milestone": "<name>",
  "timestamp": "<ISO 8601>",
  "paper": "<paper_name>",
  "status": {
    "current_stage": "<stage>",
    "completed_stages": [],
    "next_stage": "<stage>",
    "blocking_issues": []
  },
  "studies": {
    "<name>": {
      "status": "<designed|preregistered|piloted|executed|analyzed|written>",
      "hypothesis_supported": null,
      "key_findings": [],
      "effect_sizes": {}
    }
  },
  "reviews": {
    "total_rounds_completed": 5,
    "unresolved_critical": 0,
    "unresolved_major": 0
  },
  "data_integrity": {
    "config_hash": "<sha256>",
    "preregistration_hash": "<sha256>",
    "data_hash": "<sha256>"
  },
  "next_actions": []
}
```

---

## Directory Structure

```
<repository_root>/
├── AGENTS.md                              # Orchestrator master prompt
├── PLAN.md                                # This file
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── pipeline/                              # Generic reusable code (v0.4.0)
│   ├── __init__.py                        # Package exports, version
│   ├── __main__.py                        # CLI entry point
│   ├── models.py                          # Condition, Task, Trial, Scenario, StudyDesign
│   ├── schemas.py                         # Validation for configs and data
│   ├── stats.py                           # Statistical tests registry (16 tests)
│   ├── evaluators.py                      # Evaluator registry (10 evaluators)
│   ├── executor.py                        # Batching, concurrency, checkpointing
│   ├── runner.py                          # Stage execution (6 stages)
│   ├── verification.py                    # Gate logic (6 stage gates)
│   ├── preregistration.py                 # Hash-based preregistration
│   ├── pilot.py                           # Pilot management
│   ├── analysis.py                        # Statistical analysis module
│   ├── manuscript.py                      # Manuscript generation
│   ├── pdf_export.py                      # Markdown/HTML → PDF
│   ├── api.py                             # LLM API wrappers (Anthropic, OpenAI, Google)
│   ├── utils.py                           # Hashing, timestamps, JSON, I/O
│   ├── cli.py                             # CLI commands (8 commands)
│   ├── adaptive.py                        # Adaptive stopping rules
│   ├── multimodel.py                      # Multi-model support
│   ├── replication.py                     # Replication framework
│   ├── interview.py                       # Interview system
│   ├── review.py                          # 5-round review system
│   ├── review_gates.py                    # Review gate logic
│   └── orchestrator.py                    # Research orchestration
│
├── templates/
│   └── study/
│       ├── basic/
│       └── llm_behavioral/
│
└── papers/
    └── <paper_name>/
        ├── paper.yaml                     # Paper metadata
        ├── interview.yaml                 # User interview
        ├── checkpoints/                   # State + PROGRESS.md
        ├── studies/
        │   └── <study_name>/
        │       ├── config.yaml            # Study configuration
        │       ├── tasks.yaml             # Task definitions
        │       ├── analysis_plan.yaml     # Pre-specified analysis
        │       ├── research_plan.md       # Narrative rationale
        │       ├── preregistration/       # Hashed, locked design
        │       ├── pilot/                 # Pilot data + results
        │       ├── stages/                # Main study data (1-5)
        │       ├── reviews/               # 5-round reviews per checkpoint
        │       ├── future_research.yaml   # Serendipitous findings
        │       └── deviations.yaml        # Post-prereg changes (if any)
        ├── combined_analysis/             # Cross-study synthesis
        ├── manuscript/
        │   ├── sections/                  # Individual .md files
        │   ├── figures/
        │   ├── tables/
        │   ├── paper.md                   # Combined
        │   ├── references.bib
        │   └── output/
        │       └── paper_final.pdf
        └── supplementary/
```

---

## Configuration Reference

### paper.yaml
```yaml
paper:
  title: "<paper title>"
  short_title: "<running head>"
  authors:
    - name: "<author>"
      affiliation: "<institution>"
      email: "<email>"
  keywords: ["<keyword1>", "<keyword2>"]

studies:
  - name: "<study_name>"
    role: "Study 1"
    title: "<study title>"
    status: "designed|preregistered|piloted|executed|analyzed|written"
```

### config.yaml (per study)
```yaml
study:
  name: "<name>"
  version: "1.0.0"
  paper: "<paper_name>"
  hypothesis: "<testable hypothesis>"
  design_type: "within|between|mixed"

conditions:
  - name: "<condition>"
    description: "<what this condition does>"
    params: {}
    prompt_template: |
      <exact prompt text>
    system_instructions: |
      <exact system prompt>

scenarios:
  - id: "<scenario_id>"
    description: "<description>"
    shared_context: "<context>"
    shared_tools: []
    task_ids: []

trials:
  repetitions: <N>
  seed: <int>
  randomize_order: true

models:
  - provider: "anthropic|openai|google"
    model: "<exact model string>"
    parameters:
      temperature: 0.0
      max_tokens: <N>

evaluation:
  primary_evaluator: "<from registry>"
  modes: []
  llm_judge:
    enabled: false
    model: "<model>"
    rubric: "<rubric>"
    position_debiasing: true

analysis:
  tests:
    - name: "<test from registry>"
      params: {}
  alpha: 0.05
  correction: "holm-bonferroni|bonferroni|benjamini-hochberg|none"
  effect_sizes: []

execution:
  batch_size: 10
  max_concurrent: 5
  max_retries: 3
  requests_per_minute: 60
  checkpoint_frequency: 10

pilot:
  required: true
  fraction: 0.2
  seed: <different from main>
  success_criteria:
    min_trials: 20
    response_rate: 0.95
    error_rate_max: 0.05
    evaluation_determinism: 1.0
```

### tasks.yaml (per study)
```yaml
tasks:
  - id: "<task_id>"
    prompt: "<exact prompt text>"
    category: "<grouping>"
    expected:
      <evaluator-specific expected output>
    metadata:
      difficulty: "easy|medium|hard"
      domain: "<domain>"
    context: "<optional additional context>"
    tools:
      - name: "<tool_name>"
        description: "<tool description>"
        parameters: {}
```

### analysis_plan.yaml (per study)
```yaml
primary_analysis:
  test: "<test name>"
  hypothesis: "<directional prediction>"
  dv: "<dependent variable name>"
  iv: "<independent variable name>"
  alpha: 0.05
  effect_size_metric: "<metric>"
  minimum_effect_size: <value>

secondary_analyses:
  - test: "<test name>"
    purpose: "<why this test>"

assumption_checks:
  - test: "shapiro_wilk"
    variable: "<which variable>"
    action_if_violated: "use mann_whitney instead"

corrections:
  method: "holm-bonferroni"
  family: ["<which tests are in the same family>"]

power_analysis:
  method: "simulation|closed_form"
  target_power: 0.80
  effect_size: <value>
  alpha: 0.05
  computed_n: <result>

exclusion_criteria:
  - rule: "<description>"
    implementation: "<how to detect>"

exploratory:
  - description: "<what might be explored post-hoc>"
    tests: ["<potential tests>"]
```

---

## CLI Reference

```bash
# Study lifecycle
python -m pipeline new <study> --paper <paper> --template <template>
python -m pipeline validate <paper>/<study>/config.yaml
python -m pipeline prereg <study>
python -m pipeline prereg <study> --verify
python -m pipeline pilot <study>
python -m pipeline run <study> --full
python -m pipeline run <study> --stage <stage> [--resume]
python -m pipeline verify <study> --all

# Status and inspection
python -m pipeline status
python -m pipeline status <study>
python -m pipeline checkpoint <paper>

# Multi-study
python -m pipeline cross-analyze <paper>
python -m pipeline replicate <original> <new> --type direct|conceptual

# Manuscript
python -m pipeline manuscript <paper> --generate
python -m pipeline manuscript <paper> --pdf

# Autonomous mode
python -m pipeline research "<hypothesis>" --paper <paper>
```

---

## Technology Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| Configuration | Pydantic + YAML | Schema validation, type safety |
| Data versioning | Git + SHA-256 | Immutable data trail |
| Experiment tracking | Built-in JSON logs | Lightweight, self-contained |
| Statistical analysis | scipy, statsmodels, pingouin | Comprehensive test coverage |
| Effect sizes | pingouin, custom | With confidence intervals |
| Visualization | matplotlib, seaborn | Publication-quality figures |
| Manuscript | Markdown + Jinja2 | Templated sections |
| PDF generation | weasyprint or pandoc | Markdown → PDF |
| API clients | httpx (async) | Anthropic, OpenAI, Google |
| CLI | click or typer | Command-line interface |

---

## Key Constraints

1. **No data collection before preregistration is locked.** Period.
2. **No proceeding past a failed blocking gate.** Fix it or halt.
3. **No modifying preregistered files after lock.** Document deviations separately.
4. **No statistical results without assumption checks.** Every test must verify its assumptions.
5. **No claims in the paper without a traceable data file.** Every number must have a source.
6. **No skipping review rounds.** Every checkpoint gets exactly 5 rounds.
7. **No review files are ever deleted or modified.** Append-only.
8. **Negative results are publishable.** Report them honestly; do not p-hack.

---

*Last Updated: 2026-02-04*
*Architecture Version: 1.0.0*
