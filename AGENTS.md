# Agent Instructions

This document provides instructions for AI agents working with this repository.

## Overview

This repository contains the **Format Friction** research project - a pre-registered study investigating whether LLMs detect user signals but fail to report them in structured XML format.

## Critical: Environment Setup

### Unset API Keys

**IMPORTANT**: This project uses **subscription-based CLIs only** (not API SDKs). You MUST unset API keys to prevent accidental SDK usage:

```bash
unset ANTHROPIC_API_KEY
unset OPENAI_API_KEY
unset GOOGLE_API_KEY
```

If these are set, the CLI wrappers may behave unexpectedly or fail.

### Virtual Environment

Always activate the virtual environment before running Python:

```bash
source .venv/bin/activate
```

If the venv doesn't exist or is broken:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Verify Setup

```bash
source .venv/bin/activate
python -c "from experiments.core import statistics, judge, cli_wrappers, checkpoint; print('OK')"
python -m experiments.cli --help
```

## Available CLIs

This project uses subscription-based model CLIs:

| Model | CLI Command | Auth Method |
|-------|-------------|-------------|
| Claude | `claude` | Claude Code subscription |
| GPT-4 | `codex exec` | ChatGPT subscription |
| Gemini | `gemini` | Google subscription |

Check availability:
```bash
which claude codex gemini
```

## Project Structure

```
experiments/
├── cli.py                    # Main CLI entrypoint
├── core/                     # Pipeline modules (statistics, judge, checkpoint)
├── run_analysis.py           # Analysis script (locked in pre-registration)
├── scenarios/
│   └── signal_detection.py   # 75 scenario definitions
└── results/
    ├── primary/              # Main experiment data
    ├── validation/           # Human annotations
    └── analysis/             # Analysis outputs

scripts/
├── run_pipeline.sh           # Master pipeline script
├── setup_env.sh              # Environment validation
└── verify_checkpoint.py      # Checkpoint verification

verification/
├── checkpoint_*.json         # Phase checkpoints
├── preregistration_lock.json # SHA256 locks for pre-registered files
└── FINAL_REPORT.md           # Summary report
```

## Running Commands

### CLI Usage

The main entrypoint is `experiments/cli.py`:

```bash
# Show all commands
python -m experiments.cli --help

# Validate scenarios
python -m experiments.cli validate

# Run analysis
python -m experiments.cli analyze --preregistered --seed 42

# Generate paper artifacts
python -m experiments.cli generate-paper
```

### Pipeline Execution

For full pipeline execution:

```bash
# First, unset API keys
unset ANTHROPIC_API_KEY OPENAI_API_KEY GOOGLE_API_KEY

# Run setup
bash scripts/setup_env.sh

# Run pipeline (will pause at manual gates)
bash scripts/run_pipeline.sh

# Resume from specific phase
bash scripts/run_pipeline.sh --resume-from 3
```

### Manual Gates

The pipeline has two manual gates:

1. **Phase 0 (OSF Pre-Registration)**: Submit `paper/PREREGISTRATION.md` to OSF, then:
   ```bash
   bash scripts/run_pipeline.sh --osf-url https://osf.io/xxxxx/
   ```

2. **Phase 4 (Human Annotations)**: Complete annotations, then:
   ```bash
   bash scripts/run_pipeline.sh --annotations path/to/annotations.csv
   ```

## Key Files (Do Not Modify)

These files are **locked** after pre-registration:

- `experiments/run_analysis.py` - Analysis code
- `experiments/scenarios/signal_detection.py` - Scenario definitions
- `paper/PREREGISTRATION.md` - Pre-registration document

Checksums are stored in `verification/preregistration_lock.json`.

## Data Files

| File | Description |
|------|-------------|
| `results/primary/signal_detection_*_judged.json` | Main experiment data with judge scores |
| `results/validation/validation_annotation_*.csv` | Human annotations |
| `results/analysis/preregistered_analysis.json` | Statistical analysis output |

## Common Tasks

### Run Analysis on Existing Data

```bash
source .venv/bin/activate
python experiments/run_analysis.py experiments/results/primary/signal_detection_20260203_074411_judged.json
```

### Compute Judge-Human Agreement

```bash
python experiments/compute_agreement.py \
  experiments/results/validation/validation_annotation_20260203_074834.csv \
  experiments/results/validation/validation_key_20260203_074834.csv
```

### Verify Checkpoints

```bash
python scripts/verify_checkpoint.py
python scripts/verify_checkpoint.py --phase 5
python scripts/verify_checkpoint.py --lock-only
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### CLI Not Found

If `claude`, `codex`, or `gemini` commands not found:
- These require subscription-based accounts
- Check they're installed and in PATH
- The pipeline will skip unavailable models

### Checksum Mismatch

If pre-registration checksums don't match:
- **DO NOT modify locked files** after pre-registration
- Check `verification/preregistration_lock.json` for expected values
- This is a critical error - investigate before proceeding

## Human Documentation

For human reviewers and annotators, see:

- **`verification/MANUAL_ACTIONS.md`**: Step-by-step instructions for the two manual gates (OSF submission, human annotations)
- **`experiments/results/validation/ANNOTATION_INSTRUCTIONS.md`**: Detailed guide for human annotators on labeling criteria

## Reference

- **PLAN.md**: Full research plan with all phases
- **CLAUDE.md**: User preferences for code style
- **verification/FINAL_REPORT.md**: Pipeline execution summary
- **verification/MANUAL_ACTIONS.md**: Human manual action checklist
- **experiments/results/validation/ANNOTATION_INSTRUCTIONS.md**: Annotation guide
