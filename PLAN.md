# Format Friction Research: Autonomous Execution Plan

**Plan Version**: 7.0
**Date**: 2026-02-03
**Status**: Ready for Execution
**Cost**: $0 (subscription-based CLIs only)

---

## Executive Summary

This plan pre-registers and executes a fully reproducible replication of the "Format Friction" effect: LLMs detect implicit user signals but fail to report them in structured XML. The study uses subscription-only CLIs (no API keys), locks hypotheses before data collection on OSF, and produces publication-ready analysis with deterministic seeds, full distributions, and cross-family judging. Manual gates are limited to OSF pre-registration and human annotation; everything else is automated through a single pipeline script with checkpointed resume.

**Key improvements over v6.0**: Removed time estimates, added code implementation phase, added self-correction protocols, added signal-type-agnostic metric, specified exact labeling fixes, defined checkpoint schemas per phase.

---

## How to Use This Plan

This plan is designed for **fully autonomous execution** by an AI agent. It is organized as follows:

1. **Part 1: Context** - What problem we're solving and why
2. **Part 2: Strategy** - High-level approach, authentication, parallelization
3. **Part 3: Pre-Registration** - Locked hypotheses and analysis plan (must complete before data)
4. **Part 4: Execution Phases** - Step-by-step implementation (10 phases: -1 through 8)
5. **Part 5: Code Modules** - What code to write
6. **Part 6: Verification** - How to validate each step
7. **Part 7: Deliverables** - What files are produced
8. **Part 8: Reference** - Technical specs, review mapping, appendices

**To execute**:
1. Complete **Phase -1** (Code Implementation) - creates all scripts and modules
2. Run `bash scripts/setup_env.sh` once to validate environment
3. Run `./scripts/run_pipeline.sh` after completing pre-registration (Phase 0 manual gate)

---

## Part 1: Context

### 1.1 The Problem

The original Format Friction paper found that LLMs detect user signals but fail to report them in structured XML format. The finding (29.4pp friction, p=0.004) is statistically significant but methodologically contested:

| Issue | Impact | This Plan's Fix |
|-------|--------|-----------------|
| No pre-registration | Can't distinguish data cleaning from p-hacking | Pre-register on OSF before data collection |
| Low κ on IMPLICIT (0.41) | 24% measurement error where effect concentrates | Increase N to 300, target κ≥0.50 |
| Same-family judge | Claude judging Claude may share biases | Add GPT-4 cross-family judge |
| Small N (10/scenario) | Wide confidence intervals | Increase to N=30/scenario |
| Bimodal distribution hidden | Mean misrepresents 41% zero / 41% severe split | Report full distribution |
| 2 labeling bugs | -150pp artificial negative friction | Fix before data collection |

### 1.2 The Goal

Produce **publication-quality evidence** that either:
- **Confirms** format friction (p<0.05, CI excludes zero)
- **Refutes** the original finding with properly-powered null result

Either outcome advances scientific understanding. Pre-registration ensures credibility regardless of direction.

### 1.3 Success Criteria

| Criterion | Threshold | Measurement |
|-----------|-----------|-------------|
| Pre-registration | Timestamped before data | OSF URL recorded |
| Sample size | N=30 per scenario | 4,500 total observations |
| Cross-family agreement | κ≥0.75 Claude-GPT4 | Agreement metrics |
| Human validation | κ≥0.70 overall, κ≥0.50 IMPLICIT | Annotation agreement |
| Primary analysis | Pre-registered exactly | Script checksum matches |
| Reproducibility | Re-run = identical output | Deterministic verification |

### 1.4 Experimental Conditions

Each scenario is run under **two conditions**:
1. **Freeform** response (no enforced XML)
2. **XML-constrained** response (explicit XML compliance instructions)

Total observations = 75 scenarios × 30 trials × 2 conditions = 4,500.

---

## Part 2: Strategy

### 2.0 Dependencies and Setup (TURNKEY)

**System tools (required):**
- `python` 3.11+
- `bash`
- `pandoc`
- `git`
- `sha256sum`

**Python dependencies (pinned in `requirements.txt`):**
- `numpy`
- `scipy`
- `statsmodels`
- `scikit-learn`
- `diptest`
- `pandas`
- `matplotlib`
- `jinja2`
- `sentence-transformers`

**Explicitly excluded**: `openai`, `anthropic` SDKs (API keys not allowed)

**Model CLIs (subscription-based, required):**
- `claude`
- `codex`
- `gemini`

**Setup command (must run once before Phase 0):**
```bash
bash scripts/setup_env.sh
```

**`scripts/setup_env.sh` must:**
- Install Python deps from `requirements.txt`
- Validate system tools (`pandoc`, `sha256sum`, `git`)
- Validate CLIs (`claude`, `codex`, `gemini`)

**Environment variables for automation:**
- `OSF_PREREGISTRATION_URL` (required after OSF submission)
- `HUMAN_ANNOTATIONS_FILE` (required after annotation)
- `SKIP_MULTIMODEL=true` (optional)

### 2.1 Authentication (CRITICAL)

**This project uses SUBSCRIPTION-BASED CLIs only. NO API KEYS.**

```bash
# MUST be unset to prevent SDK from attempting API calls
unset ANTHROPIC_API_KEY
unset OPENAI_API_KEY
unset GOOGLE_API_KEY
```

| Model | CLI Tool | Auth Method |
|-------|----------|-------------|
| Claude Sonnet | `claude` | Claude Code subscription |
| GPT-4 | `codex exec` | ChatGPT subscription |
| Gemini | `gemini` | Google subscription |

**Why**: User has subscription accounts, not API credits. SDKs would fail without API keys; CLIs authenticate via browser/OAuth.

### 2.2 Manual Gates (Explicit and Minimal)

Two phases require human action:
1. **OSF pre-registration** (Phase 0): submit `paper/PREREGISTRATION.md` and supply `--osf-url` or `OSF_PREREGISTRATION_URL`.
2. **Human annotation** (Phase 4): complete `human_annotations.csv` and supply `--annotations` or `HUMAN_ANNOTATIONS_FILE`.

All other steps are automated and non-interactive.

**No interactive prompts** are permitted in scripts; manual gates must use `--osf-url` and `--annotations` flags or environment variables.

### 2.3 Subagent Strategy

For parallel execution, spawn multiple subagents:

| Phase | Subagent Pattern | Parallelism |
|-------|------------------|-------------|
| Code writing | 5 modules simultaneously | 5 subagents |
| Judge scoring | Claude + GPT-4 in parallel | 2 subagents |
| Analysis | All stats tests concurrent | 4 subagents |
| Multi-model | Codex + Gemini + Claude | 3 subagents |

**When to parallelize**: Independent tasks with no data dependencies.
**When NOT to parallelize**: Sequential dependencies (e.g., data → judge → analysis).

### 2.4 Batching for Performance

```python
BATCH_CONFIG = {
    "experiment_batch_size": 10,      # 10 scenarios per batch
    "experiment_parallelism": 5,       # 5 concurrent batches
    "judge_batch_size": 20,            # 20 responses per judge call
    "judge_parallelism": 10,           # 10 concurrent judge calls
    "parallel_judges": True,           # Claude & GPT-4 simultaneously
}
```

**Benefit**: Reduces total API calls and improves throughput via concurrent execution.

### 2.5 Directory Structure

```
experiments/
├── scenarios/
│   ├── signal_detection.py       # 75 scenario definitions (fixed)
│   └── exemplars.json            # Prior exemplar texts for leakage check
├── core/
│   ├── __init__.py
│   ├── statistics.py             # Wilson, bootstrap, sign test
│   ├── bootstrap.py              # Bootstrap CI (seed=42)
│   ├── judge.py                  # Claude + GPT-4 judges
│   ├── cli_wrappers.py           # Codex/Gemini CLI wrappers
│   └── checkpoint.py             # Checkpoint write/read utilities
├── cli.py                        # Single CLI entrypoint for all phases
├── signal_detection_experiment.py
├── run_analysis.py
├── compute_agreement.py
├── generate_validation_sample.py
└── results/
    ├── primary/                  # Raw + judged data
    ├── validation/               # Human annotations
    ├── multi_model/              # Cross-model results
    ├── analysis/                 # Pre-registered outputs
    └── archive/                  # Legacy data (read-only)

paper/
├── PREREGISTRATION.md            # Locked before data
├── FORMAT_FRICTION.md            # Generated from data
├── templates/                    # Jinja2 templates
└── figures/                      # Generated figures

verification/
├── preregistration_lock.json     # SHA256 checksums
├── checkpoint_*.json             # Per-phase state
└── FINAL_REPORT.md               # Verification summary

scripts/
├── run_pipeline.sh               # Master turnkey script
├── setup_env.sh                  # Environment setup
├── generate_pdf.sh               # Pandoc build
├── prepare_zenodo.sh             # Zenodo packaging and upload
└── verify_checkpoint.py          # Checkpoint verification
```

### 2.6 Deterministic Run IDs

All outputs use a deterministic `run_id` instead of timestamps.

**Run ID definition**:
```
run_id = first 8 chars of sha256(
  signal_detection.py contents +
  seed +
  n_trials_per_scenario +
  conditions +
  model_name
)
```

**Important**: Each model (Claude, Codex, Gemini) gets its own `run_id` because model_name differs. The primary study uses Claude's run_id; multi-model validation (Phase 7) uses separate run_ids.

This ensures stable filenames across re-runs with identical inputs.

### 2.7 Determinism Clarification for Model Calls

LLM outputs are **inherently non-deterministic**. "Deterministic verification" in this plan means:

1. **Same inputs + cached outputs = identical results**: Once model responses are cached, re-running analysis produces byte-for-byte identical outputs.
2. **No overwrite policy**: If output files exist, phases validate checksums and skip re-calling models unless `--force` is passed.
3. **Fresh runs may differ**: Running with `--force` or on new scenarios may produce different model outputs. This is expected and acceptable.
4. **Verification targets cached data**: All statistical verification operates on cached responses, not live model behavior.

**What IS deterministic**:
- Analysis code (fixed seeds, reproducible bootstrap)
- File checksums (SHA256)
- Checkpoint state

**What is NOT deterministic**:
- Raw model responses (even with temperature=0, minor variations possible)
- Judge classifications on identical inputs across runs

---

## Part 3: Pre-Registration Document

**This section becomes `paper/PREREGISTRATION.md` and is registered on OSF BEFORE data collection.**

### 3.1 Study Information

**Title**: Format Friction in LLM Tool Calling: A Pre-Registered Replication

**Research Questions**:
1. Do LLMs detect implicit user signals but fail to report them in structured format?
2. Is this "format friction" phenomenon robust to cross-family judge validation?
3. Does format friction generalize across model families?

### 3.2 Hypotheses

**Primary Hypothesis (H1)**:
> On IMPLICIT scenarios, the proportion of trials where the signal is detected (by judge) but NOT reported in compliant XML format is greater than zero.

**Formal specification**:
- Unit of analysis: Scenario (not trial)
- Test: One-sided sign test on scenario-level friction
- α = 0.05 (two-sided for secondary)
- Minimum detectable effect: 15pp (based on original 29pp finding)

**Secondary Hypotheses**:
- H2: EXPLICIT scenarios show zero friction (detection = compliance)
- H3: Cross-family judge (GPT-4) agrees with Claude judge (κ≥0.75)
- H4: Friction replicates across model families (Codex, Gemini)

**Exploratory Hypothesis** (robustness check per REVIEW.md §3.3):
- H5: Signal-type-agnostic friction (using agnostic judge) shows same pattern as signal-type-specific

### 3.3 Exclusion Criteria (Pre-Specified)

| Criterion | Scope | Justification |
|-----------|-------|---------------|
| API errors | Exclude trial | Technical failure, not model behavior |
| Empty responses | Exclude trial | API failure |
| HARD scenarios | Exclude 3 scenarios | Cannot be solved by XML compliance |

**NO other exclusions permitted**. All scenarios with valid responses are included.

### 3.4 Analysis Plan

**Primary Analysis**:
```python
def primary_analysis(scenario_results: list[ScenarioStats]) -> HypothesisTest:
    """Pre-registered primary analysis for H1."""
    implicit_scenarios = [s for s in scenario_results if s.ambiguity == "IMPLICIT"]

    # Compute friction per scenario
    frictions = [s.detection_rate - s.compliance_rate for s in implicit_scenarios]

    # Count positive friction scenarios
    n_positive = sum(1 for f in frictions if f > 0)
    n_total = len(frictions)

    # One-sided sign test (H1: friction > 0)
    from scipy.stats import binomtest
    result = binomtest(n_positive, n_total, 0.5, alternative='greater')

    return HypothesisTest(
        hypothesis="H1",
        test="one-sided sign test",
        n_positive=n_positive,
        n_total=n_total,
        p_value=result.pvalue,
        significant=result.pvalue < 0.05
    )
```

**Bootstrap CIs**:
```python
def bootstrap_ci(values: list[float], n_bootstrap: int = 10000,
                 seed: int = 42, ci: float = 0.95) -> tuple[float, float]:
    """Bootstrap CI with fixed seed for reproducibility."""
    np.random.seed(seed)
    bootstrap_means = [
        np.mean(np.random.choice(values, size=len(values), replace=True))
        for _ in range(n_bootstrap)
    ]
    lower = np.percentile(bootstrap_means, (1 - ci) / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 + ci) / 2 * 100)
    return (lower, upper)
```

**Statistical Methods**:
| Method | Application | Specification |
|--------|-------------|---------------|
| Sign test | Primary H1 | Scenario-level, one-sided |
| Bootstrap CI | Effect size | 10,000 replicates, seed=42, percentile |
| Wilson interval | Proportions | For detection/compliance rates |
| Cohen's κ | Agreement | By stratum (EXPLICIT, IMPLICIT, etc.) |
| Benjamini-Hochberg | Multiple tests | For secondary/exploratory |
| Dip test | Bimodality | Hartigan's dip statistic |

### 3.5 Locked Files (Checksums)

Before data collection, compute SHA256 checksums:
```bash
sha256sum paper/PREREGISTRATION.md > verification/preregistration_lock.json
sha256sum experiments/run_analysis.py >> verification/preregistration_lock.json
sha256sum experiments/scenarios/signal_detection.py >> verification/preregistration_lock.json
```

These files **MUST NOT CHANGE** after pre-registration.

---

## Part 4: Execution Phases

### Deterministic Verification (Applies to All Phases)

Every phase must produce a **deterministic verification artifact** that allows an agent to self-correct without human judgment:
1. **Checkpoint JSON** (`verification/checkpoint_<phase>.json`) containing:
   - `phase`, `status`, `inputs_sha256`, `outputs_sha256`, `seed`, `metrics`
2. **Input/Output checksums**: SHA256 for all files touched in the phase.
3. **Determinism rule**: Re-running the phase with the same inputs/seed must produce identical outputs (byte-for-byte) unless the phase is a manual gate (Phase 0 OSF, Phase 4 human annotations). Manual gates still require deterministic validation of the resulting files and metadata.

For manual gates, determinism is enforced by:
- Locked inputs (pre-registration content or annotation template)
- Checksums on returned artifacts
- Explicit validation criteria (counts, κ thresholds, and schema checks)

For phases involving external model calls (Phases 2, 3, 7), determinism is enforced by:
- **Caching** all raw model responses to disk
- **No overwrite policy**: if outputs exist, the phase must validate checksums and exit without re-calling models unless `--force`
- **Self-correction**: if validation fails, re-run only the missing/invalid batches, not the entire phase

### Self-Correction Protocols

Each verification failure has an explicit recovery action:

| Condition | Action | Max Retries |
|-----------|--------|-------------|
| Checksum mismatch | Abort and report; do NOT proceed | 0 |
| CLI error rate > 1% | Retry failed batches only | 3 |
| κ < 0.75 Claude-GPT4 | Document disagreement cases; proceed with warning | 0 |
| IMPLICIT κ < 0.50 (human) | Flag for review; do NOT use judge on affected scenarios | 0 |
| Missing output file | Re-run phase for missing outputs only | 2 |
| Schema validation fail | Abort; report malformed file | 0 |
| OSF URL invalid format | Reject and prompt for valid URL | ∞ (manual gate) |

**Critical**: If a verification fails and retry limit is reached, the pipeline **MUST abort** with a clear error message. Do NOT continue to dependent phases.

---

### Phase -1: Code Implementation (BEFORE EXECUTION)

**Objective**: Create all required scripts and modules before any data collection.

**Rationale**: The pipeline script, CLI, and core modules must exist before Phase 0 can run. This phase creates the execution infrastructure.

**Tasks**:
1. Create directory structure:
   ```
   mkdir -p experiments/core
   mkdir -p verification
   mkdir -p paper/templates
   mkdir -p paper/generated/tables
   mkdir -p paper/output
   mkdir -p scripts
   ```

2. Archive existing results (from current implementation):
   ```bash
   # Move current results to archive to avoid collision
   mv experiments/results/primary/*.json experiments/results/archive/ 2>/dev/null || true
   ```

3. Create `experiments/scenarios/exemplars.json`:
   - Extract unique response patterns from archived results
   - Used in Phase 1 for semantic leakage detection
   - Schema:
   ```json
   {
     "exemplars": [
       {
         "scenario_id": "sig_implicit_frust_001",
         "response_snippet": "I understand this is frustrating...",
         "source": "archive/signal_detection_20260203_judged.json"
       }
     ]
   }
   ```

4. Create core modules (see Part 5 for specifications):
   - `experiments/core/__init__.py`
   - `experiments/core/statistics.py`
   - `experiments/core/bootstrap.py`
   - `experiments/core/judge.py`
   - `experiments/core/cli_wrappers.py`
   - `experiments/core/checkpoint.py`

5. Create CLI entrypoint:
   - `experiments/cli.py`

6. Create pipeline scripts:
   - `scripts/setup_env.sh`
   - `scripts/run_pipeline.sh`
   - `scripts/generate_pdf.sh`
   - `scripts/prepare_zenodo.sh`
   - `scripts/verify_checkpoint.py`

7. Create paper templates:
   - `paper/templates/preregistration.jinja2`
   - `paper/templates/paper.jinja2`

**Verification**:
- [ ] All directories exist
- [ ] All Python modules import without error: `python -c "from experiments.core import statistics, judge, cli_wrappers, checkpoint"`
- [ ] CLI responds to `--help`: `python -m experiments.cli --help`
- [ ] Pipeline script is executable: `bash -n scripts/run_pipeline.sh` (syntax check)
- [ ] Existing results archived (if any)
- [ ] `exemplars.json` exists and has valid schema

**Checkpoint Schema** (`verification/checkpoint_-1.json`):
```json
{
  "phase": -1,
  "phase_name": "code_implementation",
  "status": "passed|failed",
  "outputs_sha256": {
    "experiments/core/statistics.py": "...",
    "experiments/core/judge.py": "...",
    "experiments/cli.py": "...",
    "scripts/run_pipeline.sh": "..."
  },
  "metrics": {
    "modules_created": 6,
    "scripts_created": 5,
    "archived_files": 0
  }
}
```

**On Failure**: Fix the failing module/script and re-run. This phase has no external dependencies.

**Inputs**: This plan (PLAN.md)
**Outputs**: All code modules and scripts listed above

---

### Phase 0: Pre-Registration

**Objective**: Lock hypotheses and analysis plan before data collection.

**Tasks**:
1. Generate `paper/PREREGISTRATION.md` from Part 3 template
2. Compute SHA256 checksums of locked files
3. **MANUAL GATE**: Submit to OSF (https://osf.io/prereg/) and obtain URL
4. Validate OSF URL format and record in `verification/osf_registration.json`
5. Create git tag: `git tag -a pre-registration -m "Locked pre-registration"`
6. Verify all required CLIs are available (claude, codex, gemini)

**Verification**:
- [ ] PREREGISTRATION.md exists and contains all required sections (hypotheses, analysis plan, exclusion criteria)
- [ ] Checksums recorded in `verification/preregistration_lock.json`
- [ ] OSF URL matches regex pattern `^https://osf\.io/[a-z0-9]{5,}/?$`
- [ ] Git tag `pre-registration` exists: `git tag -l pre-registration`
- [ ] CLI availability checked and recorded

**Checkpoint Schema** (`verification/checkpoint_0.json`):
```json
{
  "phase": 0,
  "phase_name": "pre_registration",
  "status": "passed|failed",
  "inputs_sha256": {
    "PLAN.md": "..."
  },
  "outputs_sha256": {
    "paper/PREREGISTRATION.md": "...",
    "verification/preregistration_lock.json": "..."
  },
  "metrics": {
    "osf_url": "https://osf.io/xxxxx/",
    "osf_url_valid": true,
    "git_tag_created": true,
    "cli_claude_available": true,
    "cli_codex_available": true,
    "cli_gemini_available": true
  }
}
```

**On Failure**:
- OSF URL invalid format → Reject and prompt user to provide valid URL via `--osf-url`
- CLI unavailable → Record as `false` in metrics; Phase 7 will skip unavailable models
- Git tag creation fails → Abort; investigate git state

**Inputs**: Part 3 content, PLAN.md
**Outputs**: `paper/PREREGISTRATION.md`, `verification/preregistration_lock.json`, `verification/osf_registration.json`

---

### Phase 1: Scenario Validation

**Objective**: Ensure all 75 scenarios are correctly defined with no metadata bugs. Fix known labeling issues from prior implementation.

**Known Labeling Bugs to Fix** (from REVIEW.md analysis):
| Scenario ID | Bug | Fix |
|-------------|-----|-----|
| `sig_implicit_urg_002` | ID says "urg" but `signal_type="frustration"` | Change `signal_type` to `SignalType.URGENCY` |
| `sig_implicit_frust_003` | Model tags as `blocking_issue`, not `frustration` | Review query; if genuinely ambiguous, change `ambiguity` to `BORDERLINE` |

**Tasks**:
1. **Fix known labeling bugs** in `experiments/scenarios/signal_detection.py`:
   - Apply fixes from table above
   - Document changes in commit message
2. Validate each scenario:
   - ID matches signal_type (no "urg" ID with "frustration" metadata)
   - Ambiguity level correctly assigned
   - Expected detection and ground truth present
3. Check for semantic leakage against `experiments/scenarios/exemplars.json` (cosine similarity < 0.85 using `sentence-transformers/all-MiniLM-L6-v2`)
4. Generate validation report

**Validation Script** (must pass with 0 errors):
```python
def validate_scenarios(scenarios: list[Scenario]) -> list[str]:
    """Validate all scenarios before data collection."""
    errors = []
    for s in scenarios:
        # Check ID matches signal_type
        if s.signal_type and s.signal_type.value.lower() not in s.id.lower():
            errors.append(f"{s.id}: ID doesn't match signal_type {s.signal_type.value}")

        # Check required fields for non-CONTROL scenarios
        if s.ambiguity != AmbiguityLevel.CONTROL and s.expected_detection is None:
            errors.append(f"{s.id}: Missing expected_detection")

        # Check ambiguity is valid enum
        if not isinstance(s.ambiguity, AmbiguityLevel):
            errors.append(f"{s.id}: Invalid ambiguity {s.ambiguity}")

    return errors
```

**Verification** (agent self-check):
- [ ] `python -m experiments.cli validate` exits with code 0
- [ ] Validation report shows `"errors": []` (empty list)
- [ ] All 75 scenarios present: `jq '.scenarios | length' experiments/results/scenario_validation_report.json` returns 75
- [ ] No ID/signal_type mismatches in report
- [ ] Leakage check: all cosine similarities < 0.85
- [ ] Scenario file checksum matches pre-registration (if already locked)
- [ ] Checkpoint written successfully

**Checkpoint Schema** (`verification/checkpoint_1.json`):
```json
{
  "phase": 1,
  "phase_name": "scenario_validation",
  "status": "passed|failed",
  "inputs_sha256": {
    "experiments/scenarios/signal_detection.py": "...",
    "experiments/scenarios/exemplars.json": "..."
  },
  "outputs_sha256": {
    "experiments/results/scenario_validation_report.json": "..."
  },
  "metrics": {
    "total_scenarios": 75,
    "validation_errors": 0,
    "labeling_bugs_fixed": 2,
    "max_leakage_similarity": 0.72,
    "scenarios_by_ambiguity": {
      "EXPLICIT": 15,
      "IMPLICIT": 19,
      "BORDERLINE": 15,
      "CONTROL": 23,
      "HARD": 3
    }
  }
}
```

**On Failure**:
- Validation errors found → Fix each error in `signal_detection.py`, re-run validation
- Leakage detected (similarity ≥ 0.85) → Rewrite the leaking scenario query, re-run
- Scenario count ≠ 75 → Investigate missing/duplicate scenarios

**Inputs**: `experiments/scenarios/signal_detection.py`, `experiments/scenarios/exemplars.json`
**Outputs**: `experiments/results/scenario_validation_report.json`

---

### Phase 2: Data Collection

**Objective**: Collect N=30 trials per scenario (4,500 total observations).

**Tasks**:
1. Compute deterministic `run_id` from scenario file + config
2. Run experiment with batching (10 scenarios/batch, 5 concurrent)
3. Cache all raw model responses to disk
4. Verify data integrity (CLI errors < 1%)
5. Compute output checksums

**Experiment Configuration**:
```python
EXPERIMENT_CONFIG = {
    "model": "claude-sonnet-4-5-20250929",
    "n_trials_per_scenario": 30,
    "n_scenarios": 75,
    "total_observations": 4500,  # 75 × 30 × 2 conditions
    "random_seed": 42,
    "batch_size": 10,
    "parallelism": 5,
}
```

**Verification** (agent self-check):
```bash
# Check total observations
jq '.trials | length' experiments/results/primary/signal_detection_{run_id}.json
# Expected: 4500

# Check per-scenario count
jq '[.trials[].scenario_id] | group_by(.) | map({scenario: .[0], count: length}) | map(select(.count != 30))' experiments/results/primary/signal_detection_{run_id}.json
# Expected: [] (empty - all scenarios have exactly 30)

# Check error rate
jq '[.trials[] | select(.error != null)] | length' experiments/results/primary/signal_detection_{run_id}.json
# Expected: < 45 (1% of 4500)

# Verify both conditions present per scenario
jq '[.trials[] | {scenario: .scenario_id, condition: .condition}] | unique | group_by(.scenario) | map(select(length != 2))' experiments/results/primary/signal_detection_{run_id}.json
# Expected: [] (empty - all scenarios have both conditions)
```

- [ ] Total observations = 4500
- [ ] Each of 75 scenarios has exactly 30 trials (15 per condition)
- [ ] Error rate < 1% (< 45 errors)
- [ ] Both `freeform` and `xml_constrained` conditions present for each scenario
- [ ] Output file exists and is valid JSON
- [ ] Checkpoint written

**Checkpoint Schema** (`verification/checkpoint_2.json`):
```json
{
  "phase": 2,
  "phase_name": "data_collection",
  "status": "passed|failed",
  "inputs_sha256": {
    "experiments/scenarios/signal_detection.py": "..."
  },
  "outputs_sha256": {
    "experiments/results/primary/signal_detection_{run_id}.json": "..."
  },
  "run_id": "a1b2c3d4",
  "seed": 42,
  "metrics": {
    "total_observations": 4500,
    "cli_errors": 12,
    "error_rate": 0.0027,
    "scenarios_complete": 75,
    "trials_per_scenario": 30
  }
}
```

**On Failure**:
- Error rate > 1% → Retry failed batches (max 3 retries per batch)
- Missing scenarios → Re-run only missing scenarios with `--scenarios` flag
- JSON parse error → Investigate corrupted output; re-run affected batch

**Inputs**: Validated scenarios from Phase 1
**Outputs**: `experiments/results/primary/signal_detection_{run_id}.json`

---

### Phase 3: Judge Scoring

**Objective**: Score all responses with Claude AND GPT-4 judges (in parallel). Include both signal-type-specific AND signal-type-agnostic metrics.

**Tasks**:
1. Apply Claude judge to all trials (subscription via `claude` CLI)
2. Apply GPT-4 judge to all trials (subscription via `codex exec`)
3. **NEW**: Apply signal-type-agnostic judge (detects ANY signal, not specific type)
4. Compute cross-judge agreement (κ)
5. Document disagreement cases with full context

**Signal-Type-Agnostic Judge** (addresses REVIEW.md §3.3):
```python
AGNOSTIC_JUDGE_PROMPT = """You are evaluating whether an AI assistant acknowledged ANY emotional signal or user state in their response.

User message: "{query}"
Assistant response: "{response}"

Did the assistant acknowledge ANY of these signals: frustration, urgency, blocking issue, or other emotional state?
Acknowledgment includes: naming it, empathizing, adjusting tone/priority, or formally tagging.
Acknowledgment does NOT include: just answering the question with generic helpfulness.

Answer ONLY: YES or NO"""

def agnostic_judge(query: str, response: str) -> bool:
    """Signal-type-agnostic judge - detects ANY signal acknowledgment."""
    prompt = AGNOSTIC_JUDGE_PROMPT.format(query=query, response=response)
    result = call_claude(prompt)
    return result.success and "YES" in result.response.upper()
```

**Verification** (agent self-check):
```bash
# Check all trials have Claude judge score
jq '[.trials[] | select(.judge_claude == null)] | length' experiments/results/primary/signal_detection_{run_id}_judged_claude.json
# Expected: 0

# Check all trials have GPT-4 judge score
jq '[.trials[] | select(.judge_gpt4 == null)] | length' experiments/results/primary/signal_detection_{run_id}_judged_gpt4.json
# Expected: 0

# Check all trials have agnostic judge score
jq '[.trials[] | select(.judge_agnostic == null)] | length' experiments/results/primary/signal_detection_{run_id}_judged_claude.json
# Expected: 0

# Check cross-judge agreement
jq '.kappa_claude_gpt4' experiments/results/validation/cross_judge_agreement.json
# Expected: >= 0.75
```

- [ ] All 4500 trials scored by Claude judge
- [ ] All 4500 trials scored by GPT-4 judge
- [ ] All 4500 trials scored by agnostic judge
- [ ] Cross-judge κ (Claude-GPT4) ≥ 0.75
- [ ] Disagreement cases exported to `disagreements.json`
- [ ] Checkpoint written

**Checkpoint Schema** (`verification/checkpoint_3.json`):
```json
{
  "phase": 3,
  "phase_name": "judge_scoring",
  "status": "passed|failed",
  "inputs_sha256": {
    "experiments/results/primary/signal_detection_{run_id}.json": "..."
  },
  "outputs_sha256": {
    "experiments/results/primary/signal_detection_{run_id}_judged_claude.json": "...",
    "experiments/results/primary/signal_detection_{run_id}_judged_gpt4.json": "...",
    "experiments/results/validation/cross_judge_agreement.json": "..."
  },
  "metrics": {
    "trials_scored_claude": 4500,
    "trials_scored_gpt4": 4500,
    "trials_scored_agnostic": 4500,
    "kappa_claude_gpt4": 0.82,
    "kappa_claude_gpt4_implicit": 0.78,
    "disagreement_count": 234,
    "cross_judge_agreement_met": true
  }
}
```

**On Failure**:
- κ < 0.75 → Document all disagreement cases; proceed with WARNING (do not abort)
- Missing judge scores → Re-run judge on missing trials only
- CLI timeout → Retry with increased timeout (max 3 retries)

**Inputs**: Raw experiment data from Phase 2
**Outputs**:
- `experiments/results/primary/signal_detection_{run_id}_judged_claude.json`
- `experiments/results/primary/signal_detection_{run_id}_judged_gpt4.json`
- `experiments/results/validation/cross_judge_agreement.json`
- `experiments/results/validation/judge_disagreements.json`

---

### Phase 4: Human Validation

**Objective**: Validate judge reliability with human annotations (N=300).

**Tasks**:
1. Generate stratified sample (seed=42)
2. Export sample CSV for annotation (blind to judge labels)
3. **MANUAL GATE**: Collect annotations from 2 human annotators
4. Validate annotation file schema and completeness
5. Compute inter-annotator agreement
6. Adjudicate disagreements (majority vote or third annotator)
7. Compute judge-human agreement by stratum (especially IMPLICIT)

**Sampling Strategy**:
```python
def generate_validation_sample(data: list[Trial], n: int = 300, seed: int = 42) -> list[Trial]:
    """Stratified sample for human validation."""
    np.random.seed(seed)

    # Stratify by ambiguity and condition
    strata = {}
    for trial in data:
        key = (trial.ambiguity, trial.condition)
        strata.setdefault(key, []).append(trial)

    # Sample proportionally from each stratum
    samples = []
    for key, trials in strata.items():
        n_stratum = max(1, int(n * len(trials) / len(data)))
        samples.extend(np.random.choice(trials, size=n_stratum, replace=False))

    return samples[:n]
```

**Annotation CSV Schema** (required columns):
```csv
trial_id,query,response,annotator_1,annotator_2,adjudicated
abc123,"User query here","Model response here",YES,YES,YES
```

- `trial_id`: Unique identifier matching judged data
- `query`: The user query (for annotator reference)
- `response`: The model response (for annotator reference)
- `annotator_1`: YES or NO
- `annotator_2`: YES or NO
- `adjudicated`: Final label after disagreement resolution

**Verification** (agent self-check):
```bash
# Check annotation count
wc -l < experiments/results/validation/human_annotations.csv
# Expected: 301 (300 + header)

# Check required columns present
head -1 experiments/results/validation/human_annotations.csv | tr ',' '\n' | grep -c -E '^(trial_id|query|response|annotator_1|annotator_2|adjudicated)$'
# Expected: 6

# Check no empty values in required columns
awk -F',' 'NR>1 && ($4=="" || $5=="" || $6=="")' experiments/results/validation/human_annotations.csv | wc -l
# Expected: 0

# Check valid values (YES/NO only)
awk -F',' 'NR>1 && ($4!="YES" && $4!="NO")' experiments/results/validation/human_annotations.csv | wc -l
# Expected: 0
```

- [ ] Annotation file has exactly 300 data rows
- [ ] All required columns present
- [ ] No empty values in annotation columns
- [ ] All values are YES or NO
- [ ] Inter-annotator κ ≥ 0.70 overall
- [ ] IMPLICIT stratum κ ≥ 0.50 (critical threshold from REVIEW.md)
- [ ] Judge-human agreement computed for all strata
- [ ] Checkpoint written

**Checkpoint Schema** (`verification/checkpoint_4.json`):
```json
{
  "phase": 4,
  "phase_name": "human_validation",
  "status": "passed|failed",
  "inputs_sha256": {
    "experiments/results/validation/validation_sample.csv": "...",
    "experiments/results/validation/human_annotations.csv": "..."
  },
  "outputs_sha256": {
    "experiments/results/validation/judge_human_agreement.json": "..."
  },
  "metrics": {
    "samples_annotated": 300,
    "inter_annotator_kappa_overall": 0.78,
    "inter_annotator_kappa_implicit": 0.55,
    "judge_human_kappa_overall": 0.81,
    "judge_human_kappa_explicit": 1.00,
    "judge_human_kappa_implicit": 0.52,
    "judge_human_kappa_control": 1.00,
    "implicit_threshold_met": true
  }
}
```

**On Failure**:
- Schema validation fails → Reject file; provide specific error; prompt for corrected file
- Inter-annotator κ < 0.70 → Proceed with WARNING; document in paper
- IMPLICIT κ < 0.50 → **CRITICAL**: Flag affected scenarios; bounded estimates required in analysis

**Inputs**: Judged data from Phase 3
**Outputs**:
- `experiments/results/validation/validation_sample.csv` (for annotation)
- `experiments/results/validation/human_annotations.csv` (completed by humans)
- `experiments/results/validation/judge_human_agreement.json`

**NOTE**: This phase requires human annotators. Pipeline pauses until annotations provided via `--annotations FILE` flag or `HUMAN_ANNOTATIONS_FILE` environment variable.

---

### Phase 5: Pre-Registered Analysis

**Objective**: Execute exactly the pre-registered analysis plan. Report full distribution (not just mean).

**Tasks**:
1. Verify analysis script checksum matches pre-registration
2. Run primary analysis (sign test on IMPLICIT scenarios)
3. Run secondary analyses (H2, H3, H4)
4. **NEW**: Run signal-type-agnostic analysis (robustness check per REVIEW.md §3.3)
5. Compute bootstrap CIs (seed=42)
6. **NEW**: Compute full distribution metrics (addresses REVIEW.md §2.3 bimodality)
7. Generate bounded estimates accounting for judge uncertainty
8. Save analysis artifacts as separate JSONs
9. Verify reproducibility (re-run produces identical output)

**Distribution Metrics** (required per REVIEW.md):
```python
def compute_distribution_metrics(friction_values: list[float]) -> dict:
    """Compute full distribution metrics to expose bimodality."""
    return {
        "mean": np.mean(friction_values),
        "median": np.median(friction_values),
        "std": np.std(friction_values),
        "iqr": np.percentile(friction_values, 75) - np.percentile(friction_values, 25),
        "min": np.min(friction_values),
        "max": np.max(friction_values),
        "percentiles": {
            "p10": np.percentile(friction_values, 10),
            "p25": np.percentile(friction_values, 25),
            "p50": np.percentile(friction_values, 50),
            "p75": np.percentile(friction_values, 75),
            "p90": np.percentile(friction_values, 90),
        },
        "n_zero": sum(1 for f in friction_values if f == 0),
        "n_severe": sum(1 for f in friction_values if f > 30),  # >30pp threshold
        "pct_zero": sum(1 for f in friction_values if f == 0) / len(friction_values) * 100,
        "pct_severe": sum(1 for f in friction_values if f > 30) / len(friction_values) * 100,
        "dip_statistic": diptest(np.array(friction_values))[0],
        "dip_pvalue": diptest(np.array(friction_values))[1],
    }
```

**Analysis Script** (must match pre-registration):
```python
def run_preregistered_analysis(data_path: str, output_path: str, seed: int = 42):
    """Execute pre-registered analysis exactly as specified."""

    # Verify script hasn't changed
    verify_checksum("experiments/run_analysis.py", "verification/preregistration_lock.json")

    # Load and process data
    data = load_judged_data(data_path)
    scenario_stats = compute_scenario_statistics(data)

    # Primary analysis (H1) - signal-type-specific
    h1_result = primary_analysis(scenario_stats)

    # Secondary analyses
    h2_result = explicit_zero_friction_test(scenario_stats)
    h3_result = cross_family_agreement_test(data)

    # Signal-type-agnostic analysis (robustness check)
    h1_agnostic = primary_analysis_agnostic(scenario_stats)

    # Bootstrap CIs
    np.random.seed(seed)
    friction_values = [s.friction for s in scenario_stats if s.ambiguity == "IMPLICIT"]
    friction_ci = bootstrap_ci(friction_values)

    # Full distribution analysis (exposes bimodality)
    distribution = compute_distribution_metrics(friction_values)

    # Bounded estimates (accounting for judge uncertainty)
    bounded = compute_bounded_estimates(scenario_stats, judge_kappa=checkpoint_4_metrics["judge_human_kappa_implicit"])

    results = {
        "primary": h1_result,
        "primary_agnostic": h1_agnostic,
        "secondary": {"h2": h2_result, "h3": h3_result},
        "bootstrap_ci": friction_ci,
        "distribution": distribution,
        "bounded_estimates": bounded,
        "seed": seed
    }

    save_json(results, output_path)
    return results
```

**Verification** (agent self-check):
```bash
# Verify analysis script checksum
sha256sum experiments/run_analysis.py | awk '{print $1}'
# Compare to value in verification/preregistration_lock.json

# Verify all required keys present in output
jq 'keys' experiments/results/analysis/preregistered_analysis.json
# Expected: ["bounded_estimates", "bootstrap_ci", "distribution", "primary", "primary_agnostic", "secondary", "seed"]

# Verify seed is correct
jq '.seed' experiments/results/analysis/preregistered_analysis.json
# Expected: 42

# Verify distribution metrics present
jq '.distribution | keys' experiments/results/analysis/preregistered_analysis.json
# Expected includes: dip_pvalue, dip_statistic, iqr, max, mean, median, min, n_severe, n_zero, pct_severe, pct_zero, percentiles, std

# Verify reproducibility: re-run and compare checksum
python -m experiments.cli analyze --preregistered --seed 42
sha256sum experiments/results/analysis/preregistered_analysis.json
# Must match previous run
```

- [ ] Analysis script checksum matches pre-registration lock
- [ ] Primary hypothesis (H1) tested with sign test
- [ ] Signal-type-agnostic analysis included
- [ ] All distribution metrics computed (mean, median, IQR, dip test, n_zero, n_severe)
- [ ] Bootstrap CI uses seed=42
- [ ] Bounded estimates computed
- [ ] Re-run produces byte-identical output (checksum match)
- [ ] Checkpoint written

**Checkpoint Schema** (`verification/checkpoint_5.json`):
```json
{
  "phase": 5,
  "phase_name": "preregistered_analysis",
  "status": "passed|failed",
  "inputs_sha256": {
    "experiments/run_analysis.py": "...",
    "experiments/results/primary/signal_detection_{run_id}_judged_claude.json": "..."
  },
  "outputs_sha256": {
    "experiments/results/analysis/preregistered_analysis.json": "..."
  },
  "seed": 42,
  "metrics": {
    "h1_p_value": 0.004,
    "h1_significant": true,
    "h1_agnostic_p_value": 0.006,
    "friction_mean": 29.4,
    "friction_median": 20.0,
    "friction_ci_lower": 13.5,
    "friction_ci_upper": 46.5,
    "pct_zero_friction": 41.2,
    "pct_severe_friction": 41.2,
    "dip_pvalue": 0.02,
    "bounded_lower": 1.0,
    "bounded_upper": 29.4,
    "reproducibility_verified": true
  }
}
```

**On Failure**:
- Checksum mismatch → **ABORT**: Analysis script modified after pre-registration; investigate
- Reproducibility fails → Check for non-determinism in code; fix and re-run
- Missing distribution metrics → Add missing computation; re-run

**Inputs**: Judged data from Phase 3, human validation from Phase 4
**Outputs**:
- `experiments/results/analysis/preregistered_analysis.json`
- `experiments/results/analysis/hypothesis_tests.json`
- `experiments/results/analysis/bootstrap_cis.json`
- `experiments/results/analysis/distribution_metrics.json`

---

### Phase 6: Paper Generation

**Objective**: Generate paper and figures from data (no manual table creation).

**Tasks**:
1. Generate all tables from analysis JSON
2. Generate all figures (matplotlib, with distribution histogram per REVIEW.md)
3. Fill Jinja2 paper template
4. Convert to PDF with pandoc using deterministic build env (`SOURCE_DATE_EPOCH=0`, `TZ=UTC`)

**Required Figures**:
- Figure 1: Detection vs compliance rates by ambiguity level
- Figure 2: Friction distribution histogram (shows bimodality)
- Figure 3: Cross-judge agreement matrix
- Figure 4: Bounded estimates visualization

**Table Generation**:
```python
def generate_tables(analysis_path: str, output_dir: str):
    """Generate all paper tables from analysis results."""
    analysis = load_json(analysis_path)

    # Table 1: Detection rates
    table1 = generate_detection_table(analysis)
    save_markdown(table1, f"{output_dir}/table1_detection.md")

    # Table 2: Format friction with distribution summary
    table2 = generate_friction_table(analysis)
    save_markdown(table2, f"{output_dir}/table2_friction.md")

    # Table 3: Distribution characteristics (bimodality metrics)
    table3 = generate_distribution_table(analysis)
    save_markdown(table3, f"{output_dir}/table3_distribution.md")

    # Table 4: Judge agreement by stratum
    table4 = generate_agreement_table(analysis)
    save_markdown(table4, f"{output_dir}/table4_agreement.md")
```

**Verification** (agent self-check):
```bash
# Check all required tables exist
ls paper/generated/tables/*.md | wc -l
# Expected: >= 4

# Check all required figures exist
ls paper/figures/*.png | wc -l
# Expected: >= 4

# Check paper markdown is not empty
wc -l < paper/generated/FORMAT_FRICTION.md
# Expected: > 100

# Check PDF exists and is valid
file paper/output/FORMAT_FRICTION.pdf
# Expected: "PDF document"

# Check PDF is non-trivial size
ls -la paper/output/FORMAT_FRICTION.pdf | awk '{print $5}'
# Expected: > 50000 (bytes)
```

- [ ] All 4 required tables generated
- [ ] All 4 required figures generated
- [ ] Paper markdown > 100 lines
- [ ] PDF generated and valid (not corrupted)
- [ ] PDF file size > 50KB
- [ ] Checkpoint written

**Checkpoint Schema** (`verification/checkpoint_6.json`):
```json
{
  "phase": 6,
  "phase_name": "paper_generation",
  "status": "passed|failed",
  "inputs_sha256": {
    "experiments/results/analysis/preregistered_analysis.json": "...",
    "paper/templates/paper.jinja2": "..."
  },
  "outputs_sha256": {
    "paper/generated/FORMAT_FRICTION.md": "...",
    "paper/output/FORMAT_FRICTION.pdf": "..."
  },
  "metrics": {
    "tables_generated": 4,
    "figures_generated": 4,
    "paper_lines": 450,
    "pdf_size_bytes": 125000,
    "pdf_valid": true
  }
}
```

**On Failure**:
- Missing table → Check analysis JSON has required fields; re-run table generation
- Figure generation error → Check matplotlib dependencies; inspect error message
- PDF generation fails → Check pandoc installed; check LaTeX dependencies
- PDF corrupted → Re-run pandoc with verbose output

**Inputs**: Analysis results from Phase 5
**Outputs**:
- `paper/generated/FORMAT_FRICTION.md`
- `paper/generated/tables/*.md`
- `paper/figures/*.png`
- `paper/output/FORMAT_FRICTION.pdf`

---

### Phase 7: Multi-Model Validation [CONDITIONAL]

**Objective**: Test friction generalization across model families.

**Condition**: This phase runs only if `cli_codex_available=true` OR `cli_gemini_available=true` in Phase 0 checkpoint. If neither CLI is available, skip this phase entirely (set status="skipped").

**Tasks**:
1. Check CLI availability from Phase 0 checkpoint
2. For each available model:
   - Run experiment with same scenarios and config as Phase 2
   - Apply Claude judge (cross-family judging)
   - Compute friction metrics
3. Compute cross-model comparison
4. Generate comparison report

**CLI Wrappers**:
```python
def call_codex(prompt: str, timeout: int = 120) -> str:
    """Call Codex CLI (ChatGPT subscription)."""
    result = subprocess.run(
        ["codex", "exec"],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout
    )
    return result.stdout.strip()

def call_gemini(prompt: str, timeout: int = 120) -> str:
    """Call Gemini CLI (Google subscription)."""
    result = subprocess.run(
        ["gemini", "--prompt", prompt],
        capture_output=True,
        text=True,
        timeout=timeout
    )
    return result.stdout.strip()
```

**Verification** (agent self-check):
```bash
# Check if phase was skipped
jq '.status' verification/checkpoint_7.json
# "passed" or "skipped" (not "failed")

# If not skipped, check model results exist
ls experiments/results/multi_model/*.json | wc -l
# Expected: >= 1 if any model available

# Check comparison report exists (if models tested)
test -f experiments/results/multi_model/cross_model_comparison.json && echo "exists"
```

- [ ] CLI availability checked from Phase 0 metrics
- [ ] Each available model: experiment run + judged
- [ ] Cross-model comparison generated (if any models tested)
- [ ] Status is "passed" (models tested) or "skipped" (no CLIs available)
- [ ] Checkpoint written

**Checkpoint Schema** (`verification/checkpoint_7.json`):
```json
{
  "phase": 7,
  "phase_name": "multi_model_validation",
  "status": "passed|skipped|failed",
  "inputs_sha256": {
    "experiments/scenarios/signal_detection.py": "..."
  },
  "outputs_sha256": {
    "experiments/results/multi_model/codex_results.json": "...",
    "experiments/results/multi_model/cross_model_comparison.json": "..."
  },
  "metrics": {
    "models_tested": ["codex"],
    "models_skipped": ["gemini"],
    "codex_friction_mean": 25.3,
    "codex_friction_ci": [10.2, 40.1],
    "cross_model_agreement": 0.89
  }
}
```

**On Failure**:
- CLI unavailable at runtime (but was available in Phase 0) → Re-check authentication; retry
- Timeout errors → Increase timeout; retry batch
- All models fail → Set status="failed" with error details

**Inputs**: Validated scenarios from Phase 1
**Outputs**: `experiments/results/multi_model/*.json`

---

### Phase 8: Publication

**Objective**: Prepare and publish to Zenodo.

**Tasks**:
1. Verify all required files exist from prior phases
2. Generate `verification/FINAL_REPORT.md` summarizing all checkpoints
3. Prepare release package with checksums
4. Upload to Zenodo (or create draft)
5. Record DOI in checkpoint

**Release Package Contents**:
```
zenodo_release/
├── README.md                    # Overview and reproduction instructions
├── CHECKSUMS.sha256             # SHA256 of all files
├── data/
│   ├── scenarios.json           # Scenario definitions
│   ├── raw_responses.json       # Raw model outputs
│   ├── judged_responses.json    # Judge scores
│   └── analysis_results.json    # Statistical analysis
├── code/
│   ├── experiments/             # Analysis code
│   └── requirements.txt         # Dependencies
├── paper/
│   ├── FORMAT_FRICTION.pdf      # Final paper
│   └── PREREGISTRATION.md       # Pre-registration document
└── verification/
    ├── checkpoint_*.json        # All checkpoints
    ├── preregistration_lock.json
    └── FINAL_REPORT.md
```

**Verification** (agent self-check):
```bash
# Check all required files present
for f in paper/output/FORMAT_FRICTION.pdf paper/PREREGISTRATION.md experiments/results/analysis/preregistered_analysis.json; do
  test -f "$f" && echo "$f: OK" || echo "$f: MISSING"
done

# Check all phase checkpoints exist and passed
for i in -1 0 1 2 3 4 5 6 7 8; do
  status=$(jq -r '.status' "verification/checkpoint_${i}.json" 2>/dev/null || echo "missing")
  echo "Phase $i: $status"
done
# All should be "passed" or "skipped" (Phase 7)

# Verify FINAL_REPORT.md generated
test -f verification/FINAL_REPORT.md && wc -l < verification/FINAL_REPORT.md
# Expected: > 50 lines

# Verify release archive created
test -f zenodo_release.zip && ls -la zenodo_release.zip
```

- [ ] All prior phase checkpoints have status "passed" or "skipped"
- [ ] FINAL_REPORT.md generated with summary of all phases
- [ ] Release package created with all required files
- [ ] CHECKSUMS.sha256 file included and valid
- [ ] DOI obtained (or draft created on Zenodo)
- [ ] Checkpoint written

**Checkpoint Schema** (`verification/checkpoint_8.json`):
```json
{
  "phase": 8,
  "phase_name": "publication",
  "status": "passed|failed",
  "inputs_sha256": {
    "paper/output/FORMAT_FRICTION.pdf": "...",
    "experiments/results/analysis/preregistered_analysis.json": "..."
  },
  "outputs_sha256": {
    "zenodo_release.zip": "...",
    "verification/FINAL_REPORT.md": "..."
  },
  "metrics": {
    "phases_passed": 9,
    "phases_skipped": 1,
    "phases_failed": 0,
    "release_file_count": 15,
    "release_size_bytes": 5000000,
    "doi": "10.5281/zenodo.xxxxxxx",
    "doi_status": "draft|published"
  }
}
```

**On Failure**:
- Missing checkpoint → Prior phase incomplete; go back and complete it
- Zenodo upload fails → Check credentials; retry
- Checksum mismatch in release → Regenerate release package

**Inputs**: All analysis outputs, paper PDF, all checkpoints
**Outputs**: `zenodo_release.zip`, `verification/FINAL_REPORT.md`, DOI

---

## Part 5: Code Modules

### 5.0 CLI Entrypoint (`experiments/cli.py`)

Single entrypoint for all phases. Each subcommand must:
1. Write a `verification/checkpoint_<phase>.json` file
2. Emit a non-zero exit code on failure

**Subcommands**:
- `generate-preregistration` - Generate PREREGISTRATION.md from template
- `validate` - Validate scenarios, fix labeling bugs, check for leakage
- `validate-annotations --file` - Validate human annotation file schema
- `experiment --n-trials --seed` - Run data collection
- `judge --parallel [--include-agnostic]` - Apply judges to all trials
- `sample --n` - Generate stratified validation sample
- `agreement --annotations` - Compute judge-human agreement
- `analyze --preregistered --seed` - Run pre-registered analysis
- `generate-paper` - Generate tables, figures, markdown
- `multimodel --models` - Run multi-model validation
- `generate-final-report` - Generate FINAL_REPORT.md summarizing all checkpoints
- `checkpoint --phase --status [--metrics-file]` - Write/read checkpoint files
- `--help` - Show all subcommands and options

### 5.1 Core Statistics (`experiments/core/statistics.py`)

```python
"""Statistical functions for format friction analysis."""

from scipy.stats import binomtest
import numpy as np

def wilson_ci(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for proportions."""
    from statsmodels.stats.proportion import proportion_confint
    return proportion_confint(successes, total, alpha=1-confidence, method='wilson')

def sign_test(n_positive: int, n_total: int, alternative: str = 'greater') -> float:
    """Sign test for matched pairs."""
    result = binomtest(n_positive, n_total, 0.5, alternative=alternative)
    return result.pvalue

def cohens_kappa(labels1: list, labels2: list) -> float:
    """Cohen's kappa for inter-rater agreement."""
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(labels1, labels2)

def bootstrap_ci(values: list[float], n_bootstrap: int = 10000,
                 seed: int = 42, ci: float = 0.95) -> tuple[float, float]:
    """Bootstrap confidence interval with fixed seed."""
    np.random.seed(seed)
    means = [np.mean(np.random.choice(values, len(values), replace=True))
             for _ in range(n_bootstrap)]
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return lower, upper

def dip_test(values: list[float]) -> tuple[float, float]:
    """Hartigan's dip test for bimodality."""
    from diptest import diptest
    return diptest(np.array(values))
```

### 5.2 CLI Wrappers (`experiments/core/cli_wrappers.py`)

```python
"""CLI wrappers for subscription-based model access (NO API KEYS)."""

import subprocess
from dataclasses import dataclass
from typing import Optional

@dataclass
class CLIResponse:
    model: str
    response: str
    success: bool
    error: Optional[str] = None

def call_claude(prompt: str, timeout: int = 120) -> CLIResponse:
    """Call Claude via Claude Code CLI (subscription)."""
    try:
        result = subprocess.run(
            ["claude", "--print", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return CLIResponse("claude", result.stdout.strip(), result.returncode == 0)
    except Exception as e:
        return CLIResponse("claude", "", False, str(e))

def call_codex(prompt: str, timeout: int = 120) -> CLIResponse:
    """Call Codex via Codex CLI (ChatGPT subscription)."""
    try:
        result = subprocess.run(
            ["codex", "exec"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return CLIResponse("codex", result.stdout.strip(), result.returncode == 0)
    except Exception as e:
        return CLIResponse("codex", "", False, str(e))

def call_gemini(prompt: str, timeout: int = 120) -> CLIResponse:
    """Call Gemini via Gemini CLI (Google subscription)."""
    try:
        result = subprocess.run(
            ["gemini", "--prompt", prompt],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return CLIResponse("gemini", result.stdout.strip(), result.returncode == 0)
    except Exception as e:
        return CLIResponse("gemini", "", False, str(e))
```

### 5.3 Judge Implementation (`experiments/core/judge.py`)

```python
"""LLM judge implementations (subscription-based CLIs)."""

from .cli_wrappers import call_claude, call_codex

JUDGE_PROMPT = """You are scoring whether an AI assistant's response acknowledged a specific user signal.

Signal type: {signal_type}
User query: {query}
Assistant response: {response}

Did the assistant acknowledge or address the {signal_type} signal in their response?
Answer only YES or NO."""

def claude_judge(query: str, response: str, signal_type: str) -> bool:
    """Claude judge via subscription CLI."""
    prompt = JUDGE_PROMPT.format(signal_type=signal_type, query=query, response=response)
    result = call_claude(prompt)
    return result.success and "YES" in result.response.upper()

def gpt4_judge(query: str, response: str, signal_type: str) -> bool:
    """GPT-4 judge via Codex CLI (ChatGPT subscription)."""
    prompt = JUDGE_PROMPT.format(signal_type=signal_type, query=query, response=response)
    result = call_codex(prompt)
    return result.success and "YES" in result.response.upper()
```

---

## Part 6: Verification

### 6.1 Checkpoint System

Each phase creates a checkpoint file via `python -m experiments.cli checkpoint`:

```json
{
  "checkpoint": 2,
  "phase": "data_collection",
  "status": "passed",
  "inputs_sha256": {
    "experiments/scenarios/signal_detection.py": "abc123..."
  },
  "outputs_sha256": {
    "experiments/results/primary/signal_detection_{run_id}.json": "def456..."
  },
  "seed": 42,
  "metrics": {
    "observations": 4500,
    "error_rate": 0.002
  }
}
```

### 6.2 Verification Matrix

| Phase | What to Verify | Method | Deterministic? |
|-------|----------------|--------|----------------|
| -1 | Code modules created | Import test, syntax check | ✓ |
| 0 | Pre-registration locked | SHA256 checksum, OSF URL regex | ✓ |
| 1 | Scenarios valid | Validation script exit code 0 | ✓ |
| 2 | Data complete | Count observations = 4500 | ✓ |
| 3 | Judges applied | All trials scored, κ computed | ✓ |
| 4 | Annotations complete | Schema validation, κ threshold | Partial (human input) |
| 5 | Analysis reproducible | Re-run checksum match | ✓ |
| 6 | Paper generated | PDF exists, size > 50KB | ✓ |
| 7 | Multi-model complete | Results files exist (or skipped) | ✓ |
| 8 | Publication ready | All checkpoints passed, DOI | ✓ |

### 6.3 Resume Capability

Pipeline can resume from any checkpoint:
```bash
./scripts/run_pipeline.sh --resume-from 3
```

---

## Part 7: Deliverables

### 7.1 File Checklist

**Pre-Registration**:
- [ ] `paper/PREREGISTRATION.md`
- [ ] `verification/preregistration_lock.json`
- [ ] `verification/osf_registration.json`

**Data**:
- [ ] `experiments/results/primary/signal_detection_{run_id}.json`
- [ ] `experiments/results/primary/signal_detection_{run_id}_judged_claude.json`
- [ ] `experiments/results/primary/signal_detection_{run_id}_judged_gpt4.json`

**Validation**:
- [ ] `experiments/results/validation/validation_sample.csv`
- [ ] `experiments/results/validation/human_annotations.csv`
- [ ] `experiments/results/validation/cross_judge_agreement.json`
- [ ] `experiments/results/validation/judge_human_agreement.json`

**Analysis**:
- [ ] `experiments/results/analysis/preregistered_analysis.json`
- [ ] `experiments/results/analysis/hypothesis_tests.json`
- [ ] `experiments/results/analysis/bootstrap_cis.json`
- [ ] `experiments/results/analysis/distribution_metrics.json`

**Paper**:
- [ ] `paper/generated/FORMAT_FRICTION.md`
- [ ] `paper/generated/tables/*.md`
- [ ] `paper/figures/*.png`
- [ ] `paper/output/FORMAT_FRICTION.pdf`

**Verification**:
- [ ] `verification/checkpoint_*.json` (10 files: Phase -1 through 8)
- [ ] `verification/FINAL_REPORT.md`
- [ ] `verification/preregistration_lock.json`
- [ ] `verification/osf_registration.json`

---

## Part 8: Reference

### 8.0 Glossary

- **OSF**: Open Science Framework (pre-registration platform)
- **κ (kappa)**: Cohen’s kappa, inter-rater agreement metric
- **BH**: Benjamini-Hochberg correction for multiple comparisons
- **Dip test**: Hartigan’s dip test for bimodality

### 8.1 Review Issues Addressed

| Issue | Severity | How Addressed |
|-------|----------|---------------|
| R1: Exclusion dependency | Critical | Pre-register exclusion criteria |
| R2: No pre-registration | Critical | OSF registration before data |
| R3: Low κ on IMPLICIT | Critical | Larger N=300, target κ≥0.50 |
| R4: Bimodality hidden | Critical | Report full distribution |
| R5: Generalizability | Critical | Multi-model validation |
| R6: Signal-type specificity | Major | Add type-agnostic metric |
| R7: Same-family judge | Major | GPT-4 cross-family judge |
| R8: Stratum κ hidden | Major | Lead with IMPLICIT κ |
| R9: Heterogeneity | Major | Model friction predictors |
| R10-R18: Minor issues | Minor | Various fixes |

### 8.2 Academic Rigor Standards

| Standard | Implementation | Verification |
|----------|---------------|--------------|
| Pre-registration | OSF before data | Timestamp check |
| Scenario-level primary | Sign test on scenarios | Script checksum |
| Bootstrap CIs | 10k replicates, seed=42 | Re-run identical |
| Wilson intervals | For all proportions | Unit tests |
| Cohen's κ by stratum | Especially IMPLICIT | Per-stratum output |
| BH correction | For multiple tests | Applied to secondaries |
| Distribution reporting | Mean, median, IQR, dip | In analysis output |
| Blinding | Annotators blind to judge | Separate CSV columns |

### 8.3 Master Pipeline Script

```bash
#!/bin/bash
# scripts/run_pipeline.sh

set -e

# CRITICAL: Unset API keys to prevent SDK usage
unset ANTHROPIC_API_KEY OPENAI_API_KEY GOOGLE_API_KEY

# Parse arguments
RESUME_FROM=-1
OSF_URL=${OSF_PREREGISTRATION_URL:-""}
ANNOTATIONS=${HUMAN_ANNOTATIONS_FILE:-""}
SKIP_MULTIMODEL=${SKIP_MULTIMODEL:-false}

while [[ $# -gt 0 ]]; do
    case $1 in
        --resume-from) RESUME_FROM="$2"; shift 2 ;;
        --osf-url) OSF_URL="$2"; shift 2 ;;
        --annotations) ANNOTATIONS="$2"; shift 2 ;;
        --skip-multimodel) SKIP_MULTIMODEL=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# Phase -1: Code Implementation (creates all required modules)
if [ $RESUME_FROM -le -1 ]; then
    echo "=== Phase -1: Code Implementation ==="
    # This phase is manual - agent creates all modules before running pipeline
    # Verify modules exist
    python -c "from experiments.core import statistics, judge, cli_wrappers, checkpoint" || {
        echo "ERROR: Core modules not found. Complete Phase -1 first."
        exit 1
    }
    python -m experiments.cli --help > /dev/null || {
        echo "ERROR: CLI not functional. Complete Phase -1 first."
        exit 1
    }
    python -m experiments.cli checkpoint --phase -1 --status passed
fi

# Setup environment (validates dependencies)
bash scripts/setup_env.sh

# Phase 0: Pre-registration
if [ $RESUME_FROM -le 0 ]; then
    echo "=== Phase 0: Pre-Registration ==="
    python -m experiments.cli generate-preregistration

    if [ -z "$OSF_URL" ]; then
        echo "MANUAL GATE: Submit paper/PREREGISTRATION.md to https://osf.io/prereg/"
        echo "Then re-run with: --osf-url <URL>"
        exit 0
    fi

    # Validate OSF URL format
    if ! echo "$OSF_URL" | grep -qE '^https://osf\.io/[a-z0-9]{5,}/?$'; then
        echo "ERROR: Invalid OSF URL format. Expected: https://osf.io/xxxxx/"
        exit 1
    fi

    echo "{\"osf_url\": \"$OSF_URL\", \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" > verification/osf_registration.json
    git tag -a pre-registration -m "Locked pre-registration" 2>/dev/null || echo "Tag already exists"
    python -m experiments.cli checkpoint --phase 0 --status passed
fi

# Phase 1: Scenario validation
if [ $RESUME_FROM -le 1 ]; then
    echo "=== Phase 1: Scenario Validation ==="
    python -m experiments.cli validate
    # Verify validation passed
    if [ "$(jq -r '.metrics.validation_errors' verification/checkpoint_1.json)" != "0" ]; then
        echo "ERROR: Scenario validation found errors. Fix and re-run."
        exit 1
    fi
fi

# Phase 2: Data collection
if [ $RESUME_FROM -le 2 ]; then
    echo "=== Phase 2: Data Collection ==="
    python -m experiments.cli experiment --n-trials 30 --seed 42
    # Verify data complete
    if [ "$(jq -r '.metrics.total_observations' verification/checkpoint_2.json)" != "4500" ]; then
        echo "ERROR: Data collection incomplete. Check errors and re-run."
        exit 1
    fi
fi

# Phase 3: Judge scoring
if [ $RESUME_FROM -le 3 ]; then
    echo "=== Phase 3: Judge Scoring ==="
    python -m experiments.cli judge --parallel --include-agnostic
    # Check cross-judge agreement (warning only, not fatal)
    KAPPA=$(jq -r '.metrics.kappa_claude_gpt4' verification/checkpoint_3.json)
    if (( $(echo "$KAPPA < 0.75" | bc -l) )); then
        echo "WARNING: Cross-judge κ=$KAPPA < 0.75. Proceeding with caution."
    fi
fi

# Phase 4: Human validation
if [ $RESUME_FROM -le 4 ]; then
    echo "=== Phase 4: Human Validation ==="
    python -m experiments.cli sample --n 300

    if [ -z "$ANNOTATIONS" ]; then
        echo "MANUAL GATE: Complete human annotations for experiments/results/validation/validation_sample.csv"
        echo "Then re-run with: --annotations <path_to_annotations.csv>"
        exit 0
    fi

    # Validate annotation file schema
    python -m experiments.cli validate-annotations --file "$ANNOTATIONS" || {
        echo "ERROR: Annotation file failed schema validation."
        exit 1
    }

    python -m experiments.cli agreement --annotations "$ANNOTATIONS"

    # Check IMPLICIT κ threshold (critical per REVIEW.md)
    IMPLICIT_KAPPA=$(jq -r '.metrics.judge_human_kappa_implicit' verification/checkpoint_4.json)
    if (( $(echo "$IMPLICIT_KAPPA < 0.50" | bc -l) )); then
        echo "WARNING: IMPLICIT κ=$IMPLICIT_KAPPA < 0.50. Bounded estimates required in analysis."
    fi
fi

# Phase 5: Analysis
if [ $RESUME_FROM -le 5 ]; then
    echo "=== Phase 5: Pre-Registered Analysis ==="

    # Verify analysis script checksum matches pre-registration
    EXPECTED_CHECKSUM=$(jq -r '.["experiments/run_analysis.py"]' verification/preregistration_lock.json)
    ACTUAL_CHECKSUM=$(sha256sum experiments/run_analysis.py | awk '{print $1}')
    if [ "$EXPECTED_CHECKSUM" != "$ACTUAL_CHECKSUM" ]; then
        echo "ERROR: Analysis script modified after pre-registration!"
        echo "Expected: $EXPECTED_CHECKSUM"
        echo "Actual:   $ACTUAL_CHECKSUM"
        exit 1
    fi

    python -m experiments.cli analyze --preregistered --seed 42

    # Verify reproducibility by re-running
    FIRST_CHECKSUM=$(sha256sum experiments/results/analysis/preregistered_analysis.json | awk '{print $1}')
    python -m experiments.cli analyze --preregistered --seed 42
    SECOND_CHECKSUM=$(sha256sum experiments/results/analysis/preregistered_analysis.json | awk '{print $1}')
    if [ "$FIRST_CHECKSUM" != "$SECOND_CHECKSUM" ]; then
        echo "ERROR: Analysis not reproducible! Non-determinism detected."
        exit 1
    fi
    echo "Reproducibility verified: $FIRST_CHECKSUM"
fi

# Phase 6: Paper generation
if [ $RESUME_FROM -le 6 ]; then
    echo "=== Phase 6: Paper Generation ==="
    python -m experiments.cli generate-paper

    # Generate PDF with deterministic timestamp
    SOURCE_DATE_EPOCH=0 TZ=UTC bash scripts/generate_pdf.sh

    # Verify outputs exist
    for f in paper/output/FORMAT_FRICTION.pdf paper/generated/FORMAT_FRICTION.md; do
        if [ ! -f "$f" ]; then
            echo "ERROR: Missing output: $f"
            exit 1
        fi
    done

    # Verify PDF is valid and non-trivial
    PDF_SIZE=$(stat -f%z paper/output/FORMAT_FRICTION.pdf 2>/dev/null || stat -c%s paper/output/FORMAT_FRICTION.pdf)
    if [ "$PDF_SIZE" -lt 50000 ]; then
        echo "ERROR: PDF too small ($PDF_SIZE bytes). Generation may have failed."
        exit 1
    fi
fi

# Phase 7: Multi-model (conditional on CLI availability)
if [ $RESUME_FROM -le 7 ]; then
    echo "=== Phase 7: Multi-Model Validation ==="

    # Check CLI availability from Phase 0
    CODEX_AVAILABLE=$(jq -r '.metrics.cli_codex_available' verification/checkpoint_0.json)
    GEMINI_AVAILABLE=$(jq -r '.metrics.cli_gemini_available' verification/checkpoint_0.json)

    if [ "$CODEX_AVAILABLE" = "false" ] && [ "$GEMINI_AVAILABLE" = "false" ]; then
        echo "No additional CLIs available. Skipping Phase 7."
        python -m experiments.cli checkpoint --phase 7 --status skipped
    else
        MODELS=""
        [ "$CODEX_AVAILABLE" = "true" ] && MODELS="codex"
        [ "$GEMINI_AVAILABLE" = "true" ] && MODELS="$MODELS gemini"
        python -m experiments.cli multimodel --models $MODELS
    fi
fi

# Phase 8: Publication
if [ $RESUME_FROM -le 8 ]; then
    echo "=== Phase 8: Publication ==="

    # Verify all prior phases passed
    for phase in -1 0 1 2 3 4 5 6; do
        STATUS=$(jq -r '.status' "verification/checkpoint_${phase}.json" 2>/dev/null || echo "missing")
        if [ "$STATUS" != "passed" ]; then
            echo "ERROR: Phase $phase not completed (status: $STATUS). Cannot publish."
            exit 1
        fi
    done

    # Phase 7 can be passed or skipped
    STATUS_7=$(jq -r '.status' verification/checkpoint_7.json 2>/dev/null || echo "missing")
    if [ "$STATUS_7" != "passed" ] && [ "$STATUS_7" != "skipped" ]; then
        echo "ERROR: Phase 7 not completed (status: $STATUS_7). Cannot publish."
        exit 1
    fi

    python -m experiments.cli generate-final-report
    bash scripts/prepare_zenodo.sh
fi

echo "=== Pipeline Complete ==="
echo "All checkpoints:"
for i in -1 0 1 2 3 4 5 6 7 8; do
    STATUS=$(jq -r '.status' "verification/checkpoint_${i}.json" 2>/dev/null || echo "N/A")
    echo "  Phase $i: $STATUS"
done
```

---

## Phase Summary

| Phase | Name | Description | Dependencies | Parallelism |
|-------|------|-------------|--------------|-------------|
| -1 | Code Implementation | Create all scripts and modules | None | 5 modules parallel |
| 0 | Pre-Registration | Lock hypotheses on OSF | Phase -1 | - |
| 1 | Scenario Validation | Fix bugs, validate scenarios | Phase 0 | - |
| 2 | Data Collection | 4500 observations | Phase 1 | 5 concurrent batches |
| 3 | Judge Scoring | Claude + GPT-4 + agnostic | Phase 2 | 3 judges parallel |
| 4 | Human Validation | N=300 stratified sample | Phase 3 | Manual gate |
| 5 | Analysis | Pre-registered stats | Phase 4 | 4 concurrent tests |
| 6 | Paper Generation | Tables, figures, PDF | Phase 5 | Parallel generation |
| 7 | Multi-Model | Cross-family validation | Phase 1 | Conditional; 3 models |
| 8 | Publication | Zenodo archive | Phase 6, 7 | - |

**Phase Execution Order**: -1 → 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8

**Manual Gates**: Phase 0 (OSF submission), Phase 4 (human annotations)

---

*Plan Version: 7.0*
*Authentication: Subscription-based CLIs only (NO API KEYS)*
*Parallelization: Enabled for data collection, judging, analysis, multi-model*
*Resume: Supported via --resume-from N*
*Deterministic: All computational steps use fixed seeds; agent work verified via checksums and schema validation*
*Self-Correction: Each phase has explicit on_failure actions*
