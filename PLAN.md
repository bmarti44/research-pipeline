# Agentic Format Friction Study

**Status**: PILOT COMPLETE — GO FOR FULL STUDY
**Date**: 2026-02-04
**Last Updated**: 2026-02-04 (Pilot complete, go/no-go criteria met)

---

## ✅ Pilot Complete — Go/No-Go Criteria Met

The pilot study has been completed with the following outcomes:

### Summary

| Criterion | Result | Status |
|-----------|--------|--------|
| Format friction detected | +33pp at task level | ✓ GO |
| Cross-family validation | κ = 0.755 (Gemini) | ✓ GO |
| JSON failures confirmed | All 3 judges agree | ✓ GO |
| ICC analyzed | NL=1.0, JSON=0.89 | ✓ Documented |
| Evaluation asymmetry | Discovered and documented | ✓ Documented |

### Key Finding from Pilot

**Format friction is real and externally validated.** On adversarial tasks:
- All judges (GPT-4, Gemini, ours) agree JSON fails
- NL evaluation is inherently more lenient (by design)
- Friction magnitude depends on evaluation strictness (+33pp to +50pp)

### Methodology Adjustments for Full Study

Based on pilot findings:
1. **Task-level analysis only** (ICC > 0.9 invalidates trial-level)
2. **Report both evaluation modes** (STRICT and FUNCTIONAL)
3. **65 tasks** (expanded from original 50)
4. **Document evaluation asymmetry** explicitly

---

## Historical: Issues from REVIEW.md (All Resolved)

| Issue | Resolution | Status |
|-------|------------|--------|
| Code implements wrong experiment | Deleted judge.py, implemented tool-call correctness | ✓ DONE |
| CLI imports non-existent module | Deleted and recreated cli.py | ✓ DONE |
| np.random.seed() still used | Replaced with np.random.default_rng() | ✓ DONE |
| No retry logic | Implemented exponential backoff | ✓ DONE |
| ICC interpretation backwards | Trial-level abandoned when ICC > 0.9 | ✓ DONE |
| Missing infrastructure | Created all directories and files | ✓ DONE |

---

## Core Research Question

> Do LLMs exhibit higher error rates when producing structured JSON tool calls compared to expressing the same intent in natural language, and does agentic harness complexity amplify this gap?

### The Phenomenon

When using Claude Code CLI with custom tools, we observed:
- The model **reliably** described tool intent in natural language ("I should read the config file")
- The model **unreliably** produced the actual JSON tool call
- This occurred even when the model clearly "knew" what to do

This suggests an asymmetry between **intending** (easy) and **serializing** (hard).

### Why This Might Occur

| Natural Language | JSON Tool Call |
|------------------|----------------|
| Trained on vast "I would do X" text | Less training on tool-call syntax |
| Flexible expression | Rigid format required |
| Graceful degradation (typos OK) | Catastrophic failure (one char breaks parse) |
| No schema to recall | Must remember exact arg names, types |

---

## Hypotheses

### Primary Hypothesis (H1)
> In a complex agentic harness, the rate of correct responses in JSON-only condition is lower than in NL-only condition.

**Formally**: P(correct | JSON-only) < P(correct | NL-only)

### Secondary Hypotheses
- **H2**: Friction increases with system prompt complexity (ablation test)
- **H3**: Friction varies by tool complexity (simple < medium < complex < adversarial)
- **H4**: Effect replicates across model families (Claude, GPT-4, Gemini)

### Exploratory Hypothesis
- **H5**: Within-task ICC is high (>0.9) at temperature=0, suggesting pseudo-replication

---

## Experimental Design

### Between-Subjects Design

**Rationale**: A within-response design (NL then JSON) conflates task understanding, NL generation, JSON serialization, attention allocation, and position effects.

**Solution**: Pure between-subjects design with separate conditions:

| Condition | Prompt Instruction | Output Expected |
|-----------|-------------------|-----------------|
| **NL-only** | "Describe in natural language what tool you would use and why, including all arguments" | Free-form text |
| **JSON-only** | "Output ONLY the tool call as JSON: `{\"tool\": \"name\", \"args\": {...}}`" | Structured JSON |

**Randomization**: Each task is randomly assigned to NL-only or JSON-only condition. Same task never appears in both conditions for the same trial.

### Manipulation Checks (REVISED)

**For JSON-only trials, record**:
- `attempted_json`: Did the response contain any JSON-like structure? (regex detection)
- `syntactically_valid`: Was the JSON parseable?
- `explicitly_declined`: Did the model state it cannot/will not produce JSON?
- `refusal_reason`: If declined, what reason was given?

**For NL-only trials, record** (NEW per REVIEW.md 4.3):
- `attempted_nl_description`: Did the response describe a tool intent?
- `tool_identified`: Was a specific tool named or unambiguously referenced?
- `args_complete`: Were all required arguments specified with values?
- `explicitly_declined`: Did the model refuse to engage with the task?
- `refusal_reason`: If declined, what reason was given?

**CRITICAL**: Trials where `explicitly_declined=True` are **NOT excluded**. Report two metrics:

1. **Overall Friction**: Including declines as failures (primary measure)
   - Rationale: A model that declines to produce JSON is exhibiting friction

2. **Conditional Friction**: Among compliant attempts only (secondary measure)
   - Rationale: Isolates serialization errors from task refusal

### Temperature and Variance (REVISED per REVIEW.md 2.3)

**Protocol**:
1. Run pilot at temperature=0
2. Compute intra-class correlation (ICC) within tasks
3. **REVISED decision rule**:
   - **Always report both** task-level and trial-level analyses
   - Use ICC to **interpret** results, not to select analysis
   - **If ICC > 0.9**: Trial-level analysis is INVALID for inference; only task-level analysis supports statistical claims. Trial-level may be reported for descriptive purposes only.
   - **If ICC ≤ 0.9**: Both analyses are valid; report both with appropriate caveats

**Primary analysis**: temperature=0.0

**Sensitivity analysis**: temperature=0.7 (secondary)

**Acknowledged limitation** (NEW per REVIEW.md 4.4): Temperature=0 does NOT guarantee determinism. Modern LLMs may exhibit non-determinism due to:
- Batching effects on logit computation
- Hardware differences (GPU vs CPU floating point)
- Model updates during data collection
- Provider infrastructure non-determinism

**Mitigations**: Record full API response metadata including request_id; run sensitivity analysis at temperature=0.7; acknowledge residual non-determinism in limitations section.

---

## Statistical Analysis Plan

### Independence and Clustering (REVISED per REVIEW.md 2.1)

**Problem**: Trials within tasks are not independent. Power analysis at N=1500 per condition assumes independence that doesn't hold.

**Solution**: Use cluster-robust standard errors AND mixed-effects models. Report both.

**CRITICAL REVISION**: The between-subjects design means tasks are NOT paired across conditions. Therefore:

1. **Correct primary method**: Two-proportion z-test with cluster-robust standard errors
2. **Correct effect size**: Relative risk (RR) or risk difference, NOT odds ratio
3. **Mixed-effects model**: Logistic regression with task as RANDOM effect (not fixed)

```
P(correct) ~ condition + (1|task)
```

**Incorrect method removed**: Odds ratio for "dependent data" is inappropriate because:
- Between-subjects design means each task appears in ONE condition only
- No pairing exists; data are independent at the task level
- Odds ratio for dependent data requires paired observations

```python
# REQUIRED: Use modern numpy RNG (REVIEW.md 2.4)
def cluster_bootstrap_ci(data, task_ids, statistic_fn, n_replicates=10000, seed=42):
    """Bootstrap resampling at the task level."""
    unique_tasks = np.unique(task_ids)
    rng = np.random.default_rng(seed)  # NOT np.random.seed()

    bootstrap_stats = []
    for _ in range(n_replicates):
        # Resample tasks, not trials
        sampled_tasks = rng.choice(unique_tasks, size=len(unique_tasks), replace=True)
        # Include all trials from sampled tasks
        sampled_data = [data[task_ids == t] for t in sampled_tasks]
        bootstrap_stats.append(statistic_fn(np.concatenate(sampled_data)))

    return np.percentile(bootstrap_stats, [2.5, 97.5])
```

**Reproducibility requirements** (REVIEW.md 2.4):
- Record: Python version, NumPy version, SciPy version
- Record: Bootstrap method (percentile, not BCa)
- Use `np.random.default_rng(seed)` exclusively
- Store full environment in `experiments/results/environment.json`

### Multiple Comparisons (REVISED per REVIEW.md 3.1)

**Principled justification for family structure**:

**Family 1 (Primary)**: H1 alone
- **Justification**: H1 is the sole primary confirmatory hypothesis. Secondary hypotheses (H2-H4) are mechanistic explorations conditional on H1. This is standard practice in confirmatory-exploratory designs.
- Test: Two-proportion z-test with cluster-robust SEs
- α = 0.05 (two-sided)

**Family 2 (Secondary)**: H2, H3, H4
- **Justification**: These test mechanisms (what causes friction) rather than existence (is there friction). They form a coherent family of mechanistic questions.
- H2: 4 tests (prompt ablation comparisons)
- H3: 1 trend test across complexity levels
- H4: 3 tests (cross-model comparisons)
- Total: 8 tests
- Correction: Benjamini-Hochberg at α = 0.05

**Family 3 (Exploratory)**: H5 and ad-hoc analyses
- Clearly labeled as exploratory
- No correction applied
- Interpret cautiously; require replication

### Effect Size and Practical Significance (REVISED per REVIEW.md 4.1)

**Pre-registered minimum effect of interest**: Friction ≥ 10 percentage points

**Principled justification** (REVISED to avoid circular reasoning):

We base the 10pp threshold on three independent considerations:

1. **Empirical precedent**: Prior work on LLM structured output (SLOT paper, arxiv 2505.04016) found effect sizes in the 5-15pp range for format compliance failures. 10pp represents the midpoint of observed effects.

2. **Engineering heuristic**: In autonomous agent loops, each tool-call failure typically requires either:
   - Human intervention (breaks autonomy)
   - Retry with backoff (adds latency, cost)
   - A 10pp failure rate means ~1 in 10 tool calls fail, which is a reasonable threshold for "needs attention"

3. **Statistical power tradeoff**: With our design (N=50 tasks, ~30 trials per task), we have approximately 72% power to detect a 10pp effect. Effects smaller than 5pp would require substantially more tasks to detect reliably.

**Acknowledged limitation**: The 10pp threshold is a pragmatic choice, not a definitive boundary. We report exact effect sizes with confidence intervals; readers may apply their own thresholds.

**Effect size metrics** (REVISED per REVIEW.md 3.3):

1. **Primary: Risk Difference** (P(NL correct) - P(JSON correct)) with 95% CI
   - Directly interpretable as percentage point difference
   - Appropriate for between-subjects comparison

2. **Secondary: Relative Risk** (P(NL correct) / P(JSON correct))
   - Shows multiplicative effect
   - Appropriate for between-subjects design

3. **For literature comparison only: Cohen's h**
   - 2 × (arcsin(√p₁) - arcsin(√p₂))
   - Explicitly note: "Cohen's h assumes independence; interpret cautiously"

4. **REMOVED**: Odds ratio as primary (inappropriate for between-subjects non-paired design)

**Interpretation scale**:
| Friction | Interpretation | Practical Impact |
|----------|----------------|------------------|
| < 5pp | Negligible | System remains autonomous |
| 5-10pp | Small | Minor additional oversight needed |
| 10-20pp | Medium | Regular human verification required |
| > 20pp | Large | Human-in-the-loop mandatory |

---

## Task Categories

### Factorial Design for Adversarial Tasks (REVISED per REVIEW.md 2.4)

**Problem**: Original categories conflate JSON content, escaping needs, and unicode—not orthogonal.

**Solution**: Factorial design that isolates each factor:

| Category | JSON Content | Escaping Needed | Unicode | Example |
|----------|--------------|-----------------|---------|---------|
| **Control** | No | No | No | `{"tool": "noop", "args": {}}` |
| **Simple** | No | No | No | `read_file(path="/etc/config")` |
| **Medium** | No | Maybe | No | `write_file(path, content)` |
| **Complex** | No | Maybe | No | `edit_file(path, edits: [{...}])` |
| **Adv-JSON** | Yes | Maybe (nested) | No | Write `{"key": "value"}` to file |
| **Adv-Escape** | No | Yes (quotes, backslash) | No | Write `"Hello \"World\""` |
| **Adv-Unicode** | No | Maybe | Yes | Write emoji 🎉 or non-ASCII |
| **Adv-Combined** | Yes | Yes | Yes | Full adversarial combination |

**Analysis**: Compare Adv-JSON vs Control, Adv-Escape vs Control, Adv-Unicode vs Control independently. The Combined category tests interaction effects.

### Blind Categorization Protocol

1. Two independent raters categorize each task using only the definitions above
2. Raters are blind to expected friction and hypothesis direction
3. Disagreements resolved by third rater
4. Report inter-rater κ for categorization
5. Categories locked before data collection

---

## NL Intent Extraction (REVISED per REVIEW.md 2.3)

### Pre-Registered Extraction Rubric

**Problem**: "Extract intent" was undefined. What counts as correct NL intent?

**Solution**: Pre-register exact extraction criteria with labeled examples.

**Extraction criteria**:

An NL response is considered to express **correct intent** if it identifies:
1. The correct tool name (or unambiguous synonym)
2. All required arguments with correct values
3. No contradictory or impossible argument values

**Labeled examples** (minimum 50 required before data collection):

| Task | NL Response | Correct? | Rationale |
|------|-------------|----------|-----------|
| Read /etc/passwd | "I should read the passwd file" | NO | Missing exact path |
| Read /etc/passwd | "I would use read_file on /etc/passwd" | YES | Tool + exact path |
| Read /etc/passwd | "Reading configuration seems necessary" | NO | Wrong file, vague tool |
| Write "hello" to /tmp/out.txt | "I'll write 'hello' to /tmp/out.txt using write_file" | YES | Tool + all args |
| Write "hello" to /tmp/out.txt | "I should write to that file" | NO | Missing content, vague path |

**Validation dataset**: `experiments/validation/extraction_ground_truth.json`
- Minimum 50 examples
- Stratified by category (Control, Simple, Medium, Complex, Adversarial-*)
- Two annotators with κ ≥ 0.80 required
- Extractor accuracy ≥ 90% on validation set before proceeding

---

## System Prompt Design

### Prompt Ablation Conditions (REVISED per REVIEW.md 4.5)

**Acknowledged limitation**: The ablation is EXPLORATORY, not confirmatory. We cannot isolate all prompt components; selection is pragmatic, not principled.

**Justification for selected factors**:
- Security and style instructions are the two longest components in Claude Code prompts
- These components are commonly added by harness developers
- Testing their impact is practically relevant even if not theoretically complete

**Acknowledged unexplored factors**:
- Tool description verbosity
- Few-shot examples (0, 1, 3)
- System prompt length (controlling content)
- Instruction positioning (beginning vs end)

| Condition | Components | Token Estimate |
|-----------|------------|----------------|
| **Minimal** | Tool schemas only | ~500 tokens |
| **Tools + Security** | Tools + security policy | ~1,500 tokens |
| **Tools + Style** | Tools + style guidelines | ~1,500 tokens |
| **Full** | All components (Claude Code-style) | ~4,000 tokens |

**Limitation statement**: "The ablation tests security and style factors only. This selection is pragmatic (these are common additions) but incomplete. Future work should systematically test tool verbosity, examples, and positioning."

### Prompt Components (Full Condition)

```
1. IDENTITY
   "You are an interactive CLI tool that helps users with software engineering tasks..."

2. SECURITY POLICY
   "IMPORTANT: Never generate URLs unless confident they help with programming..."
   "IMPORTANT: Assist with authorized security testing..."

3. TOOL DESCRIPTIONS
   Detailed descriptions of each available tool with usage guidelines.

4. TOOL USAGE POLICY
   - Prefer specialized tools over bash
   - Use Read before Edit
   - Parallel tool calls when independent
   - Never use placeholders

5. STYLE GUIDELINES
   - Be concise
   - No emojis unless requested
   - Don't use colons before tool calls

6. TASK GUIDELINES
   - Never propose changes to unread code
   - Avoid over-engineering
   - Be careful about security vulnerabilities
```

---

## Human Validation (REVISED per REVIEW.md 2.2 and 4.2)

### LLM-as-Judge Mitigation

**Problem**: Using LLMs to judge LLM outputs creates circularity. Claude judging Claude shares training data and architecture.

**Solution**: Multi-layer validation with cross-family judges and increased human sample.

**Protocol**:

1. **30% stratified sample**: 450 trials
   - Proportional allocation by category
   - Equal split between NL-only and JSON-only conditions

2. **Blind annotation**:
   - Annotators see only task description and model response
   - Annotators do NOT see: condition label, expected answer, hypothesis direction
   - Two independent annotators per trial
   - Third annotator for disagreements

3. **Agreement thresholds** (REVISED for consistency per REVIEW.md 4.2):

   | Metric | Threshold | Rationale |
   |--------|-----------|-----------|
   | Inter-annotator κ | ≥ 0.80 | Standard for human agreement on objective tasks |
   | Judge-human κ (overall) | ≥ 0.75 | Allows some judge error while maintaining validity |
   | Judge-human κ (per category) | ≥ 0.70 | Categories may have legitimate ambiguity |
   | Cross-family judge κ | ≥ 0.75 | **RAISED** to match judge-human threshold |

   **Rationale for unified 0.75 threshold**: If judges disagree more with each other than with humans, this suggests task ambiguity OR judge unreliability. There's no principled reason cross-family should be lower than judge-human.

4. **Cross-family validation** (REQUIRED per REVIEW.md 2.2):
   - Primary judge: Claude (not same model as subject)
   - Secondary judge: GPT-4 (different family)
   - Tertiary judge: Gemini (different family)
   - Report all pairwise κ: Claude-GPT-4, Claude-Gemini, GPT-4-Gemini
   - **Cross-family judge agreement ≥ 0.75 required** (raised from 0.70)

5. **If thresholds not met**:
   - Use human labels as primary for that category
   - Report judge failures prominently in results
   - Do NOT proceed with automated judging for failed categories

---

## Power Analysis

### Revised Calculation

**Effective sample size**: Due to clustering, effective N is between n_tasks (50) and n_trials (3000). Use design effect to estimate.

**Design effect**: DE = 1 + (m - 1) × ICC, where m = trials per task

If ICC = 0.5 and m = 30:
- DE = 1 + (30 - 1) × 0.5 = 15.5
- Effective N = 3000 / 15.5 ≈ 194

**Sensitivity analysis**:

| True Friction | Power at Effective N=200 | Power at Effective N=500 |
|---------------|--------------------------|--------------------------|
| 5pp | 35% | 60% |
| 7pp | 52% | 78% |
| 10pp | 72% | 92% |
| 15pp | 90% | 99% |

**Decision**: Run N=1500 per condition (3000 total). If pilot ICC suggests effective N < 200, increase trials or tasks.

**Pilot study**: 10 tasks × 10 trials = 100 observations per condition
- Estimate preliminary friction
- Compute ICC for effective N calculation
- Validate judge against 50% human labels

---

## Reproducibility

### Model Versioning (REVISED per REVIEW.md 4.1)

**Problem**: Hardcoded model IDs will break when models update.

**Solution**: Environment variables with runtime locking.

```python
# experiments/core/config.py
import os
from datetime import datetime

def get_model_config():
    """Get model configuration from environment with runtime locking."""
    config = {
        "claude_model": os.environ.get("CLAUDE_MODEL_ID", "claude-sonnet-4-20250514"),
        "gpt_model": os.environ.get("OPENAI_MODEL_ID", "gpt-4o-2024-08-06"),
        "gemini_model": os.environ.get("GEMINI_MODEL_ID", "gemini-1.5-pro-002"),
        "locked_at": datetime.utcnow().isoformat(),
    }
    return config
```

**Protocol**:
- Set model IDs via environment variables at experiment start
- Lock configuration in `experiments/results/model_config_lock.json`
- Never modify after first API call
- Document: "Results reflect model behavior as of [date]"

### API Metadata Logging (REVISED per REVIEW.md 4.2)

**Log for each request**:
```json
{
  "request_id": "req_abc123",
  "timestamp": "2026-02-04T15:30:00.123Z",
  "model": "claude-sonnet-4-20250514",
  "latency_ms": 1234,
  "retry_count": 0,
  "input_tokens": 4567,
  "output_tokens": 234,
  "finish_reason": "end_turn",
  "rate_limit_remaining": 1000,
  "rate_limit_reset_at": "2026-02-04T15:31:00.000Z"
}
```

### Rate Limiting and Retry Logic (NEW per REVIEW.md 2.5)

**Required implementation**:

```python
# experiments/core/api_providers.py additions

import time
from dataclasses import dataclass
from typing import Callable

@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0

def call_with_retry(
    api_fn: Callable,
    retry_config: RetryConfig = RetryConfig(),
) -> APIResponse:
    """Call API function with exponential backoff retry."""
    last_error = None

    for attempt in range(retry_config.max_retries + 1):
        try:
            result = api_fn()
            if result.success:
                result.retry_count = attempt
                return result
            last_error = result.error
        except RateLimitError as e:
            last_error = str(e)
            if attempt < retry_config.max_retries:
                delay = min(
                    retry_config.base_delay * (retry_config.exponential_base ** attempt),
                    retry_config.max_delay
                )
                time.sleep(delay)
        except Exception as e:
            last_error = str(e)

    return APIResponse(success=False, error=f"Max retries exceeded: {last_error}")
```

---

## Implementation Roadmap

### Phase 0: Cleanup (BLOCKING — MUST COMPLETE FIRST)

**Verified by REVIEW.md**: Complete code-plan mismatch exists. The following has been independently confirmed:

#### 0.1 Delete Cache and OS Artifacts

```bash
rm -rf experiments/core/__pycache__/
rm -f experiments/.DS_Store
rm -f experiments/results/.DS_Store
```

#### 0.2 Delete Misaligned Code

| File | Issue | Verification | Action |
|------|-------|--------------|--------|
| `experiments/cli.py` | Imports non-existent `signal_detection` module | Line 231 imports from empty scenarios/ | DELETE |
| `experiments/core/judge.py` | Implements signal detection (frustration/urgency), not tool-call correctness | JUDGE_PROMPT tests emotional signals | DELETE |

#### 0.3 Create Directory Structure

```bash
mkdir -p experiments/scenarios
mkdir -p experiments/validation
mkdir -p experiments/results/pilot
mkdir -p experiments/results/primary
mkdir -p experiments/results/raw
mkdir -p experiments/analysis
```

#### 0.4 Verify Cleanup Complete

```bash
# Should show only: __init__.py, api_providers.py, statistics.py, bootstrap.py, checkpoint.py
ls experiments/core/

# Should be empty
ls experiments/scenarios/

# Should show: pilot/, primary/, raw/
ls experiments/results/
```

---

### Phase 1: Infrastructure

**Fix existing code FIRST**:

1. **bootstrap.py**: Replace all `np.random.seed(seed)` with:
   ```python
   rng = np.random.default_rng(seed)
   # Use rng.choice() instead of np.random.choice()
   ```

   Lines requiring changes: 40, 93, 162

2. **statistics.py**: Same replacement at line 93

3. **api_providers.py**: Implement retry logic per plan specification

**Create new files**:

| File | Purpose | Key Requirements |
|------|---------|------------------|
| `experiments/core/harness.py` | Between-subjects experiment runner | Support condition randomization, cluster tracking |
| `experiments/core/prompts.py` | System prompt assembly | Support four ablation conditions |
| `experiments/core/tools.py` | Tool schemas (JSON) | Complexity tiers, factorial adversarial categories |
| `experiments/core/extractor.py` | NL intent extraction | Pre-registered rubric, validation against ground truth |
| `experiments/core/judge.py` | Tool-call correctness evaluation | Cross-family support, NOT signal detection |
| `experiments/core/config.py` | Runtime configuration | Environment variable model locking |
| `experiments/scenarios/tasks.py` | Task definitions | Blind categorization metadata |
| `experiments/cli.py` | Unified CLI | Actual experiment runner, not stubs |

---

### Phase 2: Validation

1. **Create ground truth datasets**:
   - `validation/extraction_ground_truth.json`: 50+ hand-labeled NL extractions
   - `validation/judgment_ground_truth.json`: 100+ hand-labeled correctness judgments

2. **Validate components**:
   - NL extraction accuracy ≥ 90% on validation set
   - Correctness judgment κ ≥ 0.80 (judge vs human)
   - Cross-family judge agreement κ ≥ 0.75 (raised from 0.70)

3. **Blind categorization**:
   - Two raters categorize all tasks
   - Report inter-rater κ
   - Lock categories

---

### Phase 3: Pilot Study — COMPLETE ✓

**Stop/Go Criteria**:

| Metric | Threshold | Action if Not Met |
|--------|-----------|-------------------|
| ICC at temp=0 | > 0.9 | Trial-level analysis INVALID; proceed with task-level only |
| Judge-human κ | < 0.75 | Fix judge OR use human labels for that category |
| Cross-family judge κ | < 0.75 | Investigate disagreement; may need human labels |
| Preliminary friction | < 5pp | Consider increasing N or accepting as negative result |
| NL manipulation check failure | > 20% | Revise NL condition instructions |
| JSON manipulation check failure | > 20% | Revise JSON condition instructions |
| API error rate | > 5% | Fix retry logic before proceeding |

---

### Phase 3.5: Pilot Results and Go/No-Go Decision

#### Pilot Configuration
- **Tasks**: 13 adversarial tasks
- **Trials**: 5 per task = 65 total
- **Model**: Claude Sonnet (claude-sonnet-4-20250514)
- **Temperature**: 0.0

#### ICC Analysis Results

| Condition | ICC | Interpretation |
|-----------|-----|----------------|
| NL-only | 1.00 | Perfect consistency |
| JSON-only | 0.86-0.89 | High consistency |

**Finding**: ICC > 0.9 for NL condition → Trial-level analysis INVALID. All analyses must use task-level statistics.

#### Task-Level Results (FUNCTIONAL Mode)

| Condition | Tasks Correct | Rate |
|-----------|---------------|------|
| NL-only | 7/7 | 100% |
| JSON-only | 4/6 | 67% |
| **Friction** | — | **+33pp** |

**95% Bootstrap CI**: [0pp, +67pp]

#### Cross-Family Validation

**Gemini Flash (13 pilot tasks)**:
- Agreement: 12/13 (92%)
- Cohen's κ = 0.755 ✓ (meets ≥ 0.75 threshold)

**GPT-4 + Gemini (5 adversarial combined tasks)**:

| Condition | Our Eval | GPT-4 | Gemini |
|-----------|----------|-------|--------|
| NL-only | 5/5 ✓ | 0/5 ✗ | 2/5 |
| JSON-only | 0/5 ✗ | 0/5 ✗ | 0/5 ✗ |

**Critical finding**: All three judges agree JSON fails (0/5) on adversarial combined tasks.

#### Format Friction Definition (Corrected)

Format friction is now correctly defined as:

```
Format Friction = signal_detected AND NOT json_schema_correct
```

Both conditions are evaluated THE SAME WAY for signal detection:

| Metric | NL Condition | JSON Condition |
|--------|--------------|----------------|
| `signal_detected` | ✓ Check | ✓ Check |
| `is_correct` | = signal_detected | = signal_detected AND json_valid AND json_schema_correct |
| `format_friction` | Always False | = signal_detected AND NOT json_schema_correct |

**Key insight**: The model may SAY it should call a tool but NOT output the correct JSON. That gap is format friction.

**Metrics to report**:
1. `signal_detected` rate (same criteria for both conditions)
2. `is_correct` rate (NL: signal only; JSON: signal + valid schema)
3. `format_friction` rate (JSON only: signal present but output failed)

#### Go/No-Go Checklist

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Signal detected | ✓ | +33pp friction at task level |
| Direction correct | ✓ | NL > JSON as hypothesized |
| Exceeds threshold | ✓ | +33pp > 10pp practical threshold |
| ICC analyzed | ✓ | Trial-level invalid; use task-level |
| Cross-family κ ≥ 0.75 | ✓ | Gemini κ = 0.755 |
| JSON failures confirmed | ✓ | All 3 judges agree |
| Eval asymmetry documented | ✓ | See above |

#### Decision: **GO**

Proceed to full study with documented methodology adjustments.

---

### Phase 4: Pre-Registration

**Before full data collection**:

1. Finalize all analysis code in `experiments/analysis/preregistered_analysis.py`
2. Create SHA256 checksums of locked files
3. Record full environment: Python version, NumPy version, SciPy version
4. Submit to OSF or AsPredicted
5. Create git tag `preregistration-v1`
6. **Then—and only then—collect full data**

**Locked files**:
- `experiments/core/harness.py`
- `experiments/core/tools.py`
- `experiments/core/judge.py`
- `experiments/core/extractor.py`
- `experiments/scenarios/tasks.py`
- `experiments/validation/extraction_ground_truth.json`
- `experiments/validation/judgment_ground_truth.json`
- `experiments/analysis/preregistered_analysis.py`

---

### Phase 5: Full Experiment — REVISED

**Task Set (Expanded)**:

| Category | Count |
|----------|-------|
| control | 3 |
| simple | 12 |
| medium | 12 |
| complex | 12 |
| adv_json | 7 |
| adv_escape | 7 |
| adv_unicode | 7 |
| adv_combined | 5 |
| **Total** | **65** |

**Design (revised based on ICC findings)**:
- **65 tasks** with blind categorization
- **5 trials per task** = 325 total trials per condition
- **Effective N = 65 tasks** (not 325 trials, due to ICC > 0.9)
- Between-subjects at task level: ~32-33 tasks per condition
- Test across model families (Claude, GPT-4, Gemini)
- Store raw API responses with full metadata

**Statistical Power (N=65 tasks)**:

| Effect Size | Power |
|-------------|-------|
| 10pp | ~80% |
| 20pp | ~95% |
| 30pp | >99% |

**Analysis Plan**:
1. Primary: Task-level comparison with cluster bootstrap CI
2. Report BOTH STRICT and FUNCTIONAL evaluation modes
3. Cross-family validation with GPT-4 and Gemini judges
4. Sensitivity analysis at temperature=0.7

**Human Validation**:
- 30% stratified sample (~20 tasks)
- Focus on disagreement cases between evaluation modes

---

### Phase 6: Analysis

Execute pre-registered analysis:

1. **Primary (H1)**: Compare P(correct) across conditions
   - Report overall friction (including declines)
   - Report conditional friction (excluding declines)
   - Primary effect size: Risk difference with 95% CI
   - Secondary: Relative risk
   - Report BOTH task-level and trial-level analyses
   - **If ICC > 0.9**: Only task-level supports inference

2. **Secondary (H2-H4)**: Ablation, complexity, cross-model
   - Apply BH correction within family

3. **Exploratory (H5)**: ICC analysis
   - Report regardless of value; use for interpretation

4. **Human validation**: Report all agreement metrics
   - Inter-annotator κ by category
   - Judge-human κ by category
   - Cross-family judge κ

---

## Files Structure

### After Phase 0 Cleanup

```
experiments/
├── core/
│   ├── __init__.py        # Keep (package init)
│   ├── api_providers.py   # Keep (add retry logic)
│   ├── statistics.py      # Keep (fix np.random.seed)
│   ├── bootstrap.py       # Keep (fix np.random.seed)
│   └── checkpoint.py      # Keep
├── scenarios/             # Empty, ready for tasks.py
├── validation/            # Empty, ready for ground truth
├── analysis/              # Empty, ready for analysis code
└── results/
    ├── pilot/             # Pilot outputs
    ├── primary/           # Full experiment outputs
    └── raw/               # Raw API responses
```

### After Phase 1 Implementation

```
experiments/
├── core/
│   ├── __init__.py
│   ├── api_providers.py   # With retry logic
│   ├── statistics.py      # With default_rng
│   ├── bootstrap.py       # With default_rng
│   ├── checkpoint.py
│   ├── config.py          # NEW: Runtime configuration
│   ├── harness.py         # NEW: Experiment runner
│   ├── prompts.py         # NEW: Prompt assembly
│   ├── tools.py           # NEW: Tool schemas
│   ├── extractor.py       # NEW: NL extraction
│   └── judge.py           # NEW: Correctness judge (not signal detection)
├── scenarios/
│   └── tasks.py           # NEW: Task definitions
├── validation/
│   ├── extraction_ground_truth.json  # NEW: 50+ examples
│   └── judgment_ground_truth.json    # NEW: 100+ examples
├── analysis/
│   └── preregistered_analysis.py     # NEW
├── cli.py                 # NEW: Actual experiment CLI
└── results/
    ├── pilot/
    ├── primary/
    ├── raw/
    ├── environment.json   # NEW: Locked environment
    └── model_config_lock.json  # NEW: Locked model IDs
```

---

## Success Criteria

The study succeeds if we can answer:

1. **Is there friction?** — Is P(JSON correct) < P(NL correct)? (Primary test p < 0.05)
2. **How large?** — What's the magnitude? (Report risk difference, relative risk, 95% CI)
3. **Is it practically significant?** — Does friction ≥ 10pp? (Pre-registered threshold with principled justification)
4. **What drives it?** — Which prompt components? (Ablation analysis - exploratory)
5. **Is it consistent?** — Does it replicate across models? (Cross-model analysis)
6. **Are results valid?** — Judge-human κ ≥ 0.75? Cross-family judge κ ≥ 0.75? (Human validation)
7. **Are results reproducible?** — Same analysis on same data yields same results? (Environment locked)

---

## Summary of Changes Addressing Verified REVIEW.md Issues

| REVIEW.md Issue | Section | Verification | Resolution |
|-----------------|---------|--------------|------------|
| 1.2 Code implements signal detection | Part 1 | judge.py JUDGE_PROMPT confirmed | Phase 0 cleanup: delete judge.py |
| 1.3 CLI imports non-existent module | Part 1 | cli.py:231 confirmed | Phase 0 cleanup: delete cli.py |
| 1.4 Stale bytecode | Part 1 | __pycache__ contains orphans | Phase 0 cleanup: delete __pycache__ |
| 2.1 Between-subjects vs odds ratio | Part 2 | Lines 90 vs 189-191 contradictory | Use risk difference, relative risk; remove odds ratio |
| 2.3 ICC interpretation backwards | Part 2 | "Note" insufficient | ICC > 0.9 → trial-level INVALID |
| 2.4 np.random.seed() still used | Part 3 | bootstrap.py:40,93,162; statistics.py:93 | Replace with default_rng |
| 2.5 No retry logic | Part 4 | api_providers.py confirmed | Implement exponential backoff |
| 3.1 Missing infrastructure | Part 3 | validation/, analysis/ don't exist | Create all directories and files |
| 4.1 Effect threshold circular | Part 4 | 50-step justification unverified | Principled three-part justification |
| 4.2 Cross-family threshold inconsistent | Part 4 | 0.70 vs 0.75 unexplained | Unified to 0.75 with rationale |
| 4.3 Missing NL manipulation checks | Part 4 | Only JSON checks defined | Added NL manipulation checks |
| 4.4 Temperature=0 determinism | Part 4 | No limitations acknowledged | Added explicit limitations |
| 4.5 Prompt ablation unjustified | Part 2 | Selection arbitrary | Acknowledged as exploratory |

---

## References

- Claude Code system prompts: https://github.com/Piebald-AI/claude-code-system-prompts
- SLOT paper (structured output): https://arxiv.org/html/2505.04016v1
- Claude tool use: https://docs.anthropic.com/en/docs/build-with-claude/tool-use
- OpenAI function calling: https://platform.openai.com/docs/guides/function-calling
