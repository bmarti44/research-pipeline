# Contributing Guide

Thank you for your interest in contributing to the Autonomous Research Pipeline.

## Project Overview

This repository is an autonomous research pipeline that produces academically rigorous papers investigating LLM/AI behavior. The core infrastructure lives in `pipeline/`, study templates in `templates/`, and individual research papers in `papers/`.

## What You Can Contribute

| Area | Welcome? | Notes |
|------|----------|-------|
| Bug fixes | Yes | Pipeline code, documentation |
| Documentation | Yes | Improvements, clarifications, typos |
| New pipeline modules | Yes | Statistical tests, evaluators, exporters |
| Replication studies | Yes | Using different models or domains |
| New analyses | Yes | Must be clearly marked as exploratory |
| Visualization | Yes | New figures, improved plots |

## What You Cannot Modify

Each paper may have **preregistration locks**. After a study's preregistration is created (via `python -m pipeline prereg <study>`), these files are hash-locked and must not be changed:

- `papers/<paper>/studies/<study>/config.yaml`
- `papers/<paper>/studies/<study>/tasks.yaml`
- `papers/<paper>/studies/<study>/analysis_plan.yaml`

Locks are stored in `papers/<paper>/studies/<study>/preregistration/preregistration_lock.json`. Any post-preregistration change must be documented as a deviation in `deviations.yaml`.

---

## Getting Started

### 1. Environment Setup

```bash
# Clone the repository
git clone <repo-url>
cd tool-calling

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set API keys for the LLM providers you plan to use
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"

# Verify setup
python -c "import pipeline; print('OK')"
```

### 2. Understand the Codebase

- **`README.md`** — Project overview, CLI reference, directory layout
- **`AGENTS.md`** — Orchestrator protocol (how the agent pipeline works end-to-end)
- **`PLAN.md`** — Architecture reference and design rationale

### 3. Run the Pipeline

```bash
# Check status of existing studies
python -m pipeline status

# Create a new study from a template
python -m pipeline new my_study --template basic

# Verify a study's stage gates
python -m pipeline verify my_study --all
```

---

## Directory Layout

```
pipeline/          # Reusable pipeline infrastructure (stats, evaluators, runner, etc.)
templates/study/   # Study templates (basic, tool_calling)
papers/            # Individual research papers, each self-contained
  <paper_name>/
    paper.yaml
    studies/<study_name>/
      config.yaml, tasks.yaml, analysis_plan.yaml
      preregistration/    # Hash-locked design
      pilot/              # Pilot study data
      stages/             # Main study data (configure, generate, execute, evaluate, analyze)
      reviews/            # 5-round review files
```

---

## Adding a New Study

1. **Create from template:**
   ```bash
   python -m pipeline new my_study --paper my_paper --template basic
   ```

2. **Edit the generated files:**
   - `config.yaml` — Conditions, design, models, trial parameters
   - `tasks.yaml` — Task definitions with expected outputs
   - `analysis_plan.yaml` — Statistical tests, alpha, corrections

3. **Preregister (locks the design):**
   ```bash
   python -m pipeline prereg my_study
   ```

4. **Run pilot, then full pipeline:**
   ```bash
   python -m pipeline pilot my_study
   python -m pipeline run my_study --full
   ```

---

## Coding Standards

### Style

- **Type hints**: Always include type annotations
- **Descriptive names**: Prefer clarity over brevity
- **Docstrings**: Required for public functions

---

## Submitting Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Keep commits focused and atomic
- Write clear commit messages
- Don't modify preregistration-locked files

### 3. Verify Integrity

```bash
# Check that preregistration locks are intact
python -m pipeline prereg <study_name> --verify

# Run stage verification gates
python -m pipeline verify <study_name> --all
```

### 4. Submit Pull Request

- Describe what changed and why
- Reference any related issues
- Mark exploratory analyses clearly

---

## Exploratory Analyses

New analyses beyond a preregistered plan are welcome but must be:

1. **Clearly labeled** as exploratory (not confirmatory)
2. **Placed in separate files** (not modifying locked configs)
3. **Documented** with rationale and limitations

---

## Questions?

- Open an issue for bugs or feature requests
- See `AGENTS.md` for the full orchestrator protocol
- See `PLAN.md` for architecture and design rationale
