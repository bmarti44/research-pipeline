# Agent Instructions

Instructions for AI agents working with this repository.

## Overview

This repository contains the **Agentic Format Friction** research project — investigating whether requiring structured JSON tool calls causes failures in agentic task loops that wouldn't occur with natural language output.

**Status**: Prototype / Exploratory Phase

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

```
experiments/
├── cli.py                 # CLI entrypoint
├── core/
│   ├── api_providers.py   # API wrappers for Claude, GPT-4, Gemini
│   ├── judge.py           # LLM judge functions
│   ├── statistics.py      # Statistical utilities
│   └── checkpoint.py      # Checkpointing
├── scenarios/             # Task definitions (to be created)
└── results/               # Experiment outputs
```

## Available Models

| Provider | SDK | Models |
|----------|-----|--------|
| Anthropic | `anthropic` | claude-sonnet, claude-haiku |
| OpenAI | `openai` | gpt-4o, gpt-4o-mini |
| Google | `google-generativeai` | gemini-flash, gemini-pro |

## Current Tasks

See `PLAN.md` for the research plan and implementation roadmap.

Phase 1 (Infrastructure):
- Create `SimulatedEnvironment` class
- Create `AgenticHarness` class
- Implement tool definitions
- Create `ActionExtractor`

## Reference

- **PLAN.md** — Research plan and hypotheses
- **CLAUDE.md** — Points to this file
