# Agentic Format Friction

Investigating whether requiring structured JSON tool calls causes failures in agentic task loops that wouldn't occur with natural language output.

**Status**: Prototype / Exploratory Phase

## Research Question

> When LLMs operate in an agentic tool-use loop, does requiring structured output format (JSON tool calls) cause failures that would not occur if the model could express its intent in natural language?

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set API keys
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

## Project Structure

```
experiments/
├── core/
│   ├── api_providers.py   # API wrappers
│   ├── judge.py           # LLM judge
│   └── statistics.py      # Stats utilities
├── scenarios/             # Task definitions
└── results/               # Outputs
```

## Documentation

- **PLAN.md** — Research plan, hypotheses, and roadmap
- **AGENTS.md** — Instructions for AI agents
