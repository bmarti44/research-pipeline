# Agent Instructions

Instructions for AI agents working with this repository.

## Overview

This repository contains the **Agentic Format Friction** research project — investigating whether requiring structured JSON tool calls causes failures that wouldn't occur with natural language output.

**Status**: Prototype / Exploratory Phase
**Last Updated**: 2026-02-04

## Critical: Read Before Working

**The codebase currently requires Phase 0 cleanup.** Before implementing ANY new code:

1. Read `PLAN.md` — Contains the complete experimental design and implementation roadmap
2. Read `REVIEW.md` — Contains the methodological critique that informed the plan revision
3. Execute Phase 0 cleanup (see below)

## Environment Setup

### API Keys

```bash
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

### Virtual Environment

```bash
source .venv/bin/activate

# If venv doesn't exist:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Verify Setup

```bash
python -c "from experiments.core import api_providers; print(api_providers.check_api_keys())"
```

## Project Structure

### Current State (REQUIRES CLEANUP)

```
experiments/
├── cli.py                 # DELETE - imports non-existent module
├── core/
│   ├── __init__.py        # Keep
│   ├── api_providers.py   # Keep - LLM API wrapper
│   ├── statistics.py      # Keep - Statistical utilities
│   ├── bootstrap.py       # Keep - Bootstrap CI utilities
│   ├── checkpoint.py      # Keep - Checkpointing
│   ├── judge.py           # DELETE - signal detection (wrong experiment)
│   └── __pycache__/       # DELETE - entire directory
├── scenarios/             # Empty (correct)
├── results/
│   └── .DS_Store          # DELETE
└── .DS_Store              # DELETE
```

### Target State (After Phase 1)

```
experiments/
├── cli.py                 # NEW: Unified CLI for tool-call experiment
├── core/
│   ├── __init__.py
│   ├── api_providers.py   # Updated: Add retry logic with exponential backoff
│   ├── statistics.py
│   ├── bootstrap.py       # Updated: Use np.random.default_rng() not np.random.seed()
│   ├── checkpoint.py
│   ├── config.py          # NEW: Runtime configuration with model locking
│   ├── harness.py         # NEW: Between-subjects experiment runner
│   ├── prompts.py         # NEW: System prompt assembly with ablation
│   ├── tools.py           # NEW: Tool schemas (JSON)
│   ├── extractor.py       # NEW: NL intent extraction
│   └── judge.py           # NEW: Tool-call correctness evaluation (NOT signal detection)
├── scenarios/
│   └── tasks.py           # NEW: Task definitions with categories
├── validation/
│   ├── extraction_ground_truth.json  # 50+ hand-labeled examples
│   └── judgment_ground_truth.json    # 100+ hand-labeled examples
├── analysis/
│   └── preregistered_analysis.py
└── results/
    ├── pilot/
    ├── primary/
    ├── raw/
    ├── environment.json         # NEW: Locked Python/NumPy/SciPy versions
    └── model_config_lock.json   # NEW: Locked model IDs
```

## Phase 0: Cleanup (BLOCKING)

Execute these commands BEFORE any other work:

```bash
# 1. Delete pycache
rm -rf experiments/core/__pycache__/

# 2. Delete macOS artifacts
rm -f experiments/.DS_Store
rm -f experiments/results/.DS_Store

# 3. Delete misaligned code
rm experiments/cli.py
rm experiments/core/judge.py

# 4. Create directory structure
mkdir -p experiments/scenarios
mkdir -p experiments/validation
mkdir -p experiments/results/pilot
mkdir -p experiments/results/primary
mkdir -p experiments/results/raw
mkdir -p experiments/analysis

# 5. Verify cleanup
ls experiments/core/  # Should show: __init__.py, api_providers.py, statistics.py, bootstrap.py, checkpoint.py
ls experiments/scenarios/  # Should be empty
```

## Available Models

| Provider | SDK | Models |
|----------|-----|--------|
| Anthropic | `anthropic` | claude-sonnet, claude-haiku |
| OpenAI | `openai` | gpt-4o, gpt-4o-mini |
| Google | `google-generativeai` | gemini-flash, gemini-pro |

## Key Design Decisions (from REVIEW.md)

1. **Between-subjects design**: NL-only vs JSON-only conditions (not within-response)
2. **Manipulation checks**: Record whether model attempted/declined JSON; report BOTH overall and conditional friction (don't exclude declines)
3. **Cluster-robust analysis**: Bootstrap at task level, not trial level; report both task-level and trial-level analyses
4. **Human validation**: 30% stratified sample (450 trials) with blinded annotators; cross-family judges required (κ ≥ 0.70)
5. **Prompt ablation**: Four conditions (minimal, +security, +style, full); acknowledged as incomplete
6. **ICC check**: Always report both analyses; use ICC to interpret, NOT to select analysis
7. **Effect size**: Primary = odds ratio (not Cohen's h); practical threshold = 10pp with justification
8. **Reproducibility**: Use `np.random.default_rng()`, lock model versions via env vars, record full environment

## Reference Documents

- **PLAN.md** — Complete research plan, experimental design, and roadmap
- **REVIEW.md** — Methodological critique with all issues addressed in plan
- **CLAUDE.md** — Points to this file

## Implementation Notes

When implementing Phase 1 infrastructure:

1. **harness.py**: Must support between-subjects design with condition randomization and cluster tracking
2. **prompts.py**: Must support four ablation conditions
3. **tools.py**: Tool schemas with factorial adversarial categories (JSON/Escape/Unicode isolated)
4. **extractor.py**: NL intent extraction with pre-registered rubric; validate against 50+ ground truth examples
5. **judge.py**: Correctness evaluation (NOT signal detection); cross-family support required
6. **tasks.py**: Tasks with blind categorization metadata
7. **config.py**: Runtime model locking via environment variables
8. **api_providers.py**: Add exponential backoff retry logic
9. **bootstrap.py**: Replace `np.random.seed()` with `np.random.default_rng()`

### Critical Requirements

- **NL extraction must have ≥90% accuracy** on validation set before proceeding
- **Judge-human κ ≥ 0.75** required; cross-family judge κ ≥ 0.70 required
- **Always log**: request_id, timestamp, model, latency, retry_count, tokens, rate_limit_info
- **Lock before first API call**: model versions, environment versions

See PLAN.md for detailed specifications.
