# Agentic Format Friction Study

**Status**: Prototype / Exploratory Phase
**Date**: 2026-02-04

---

## Research Question

> When LLMs operate in an agentic tool-use loop, does requiring structured output format (JSON tool calls) cause failures that would not occur if the model could express its intent in natural language?

This tests whether the constraint of producing valid, parseable tool calls causes models to fail even when they understand what needs to be done.

---

## Current Phase: Prototyping

We are in early exploration. The goal is to:

1. Build a working agentic harness with simulated tools
2. Test the basic experimental setup on a small number of tasks
3. Iterate on task design and measurement approach
4. Identify issues before committing to a full study design

**Pre-registration will happen after prototyping is complete** and we have confidence in the methodology.

---

## Hypotheses (Draft)

### Primary (H1)

> In an agentic task loop, the rate of correct action selection is higher when models describe intended actions in natural language than when required to produce structured tool calls.

### Secondary

| ID | Hypothesis |
|----|------------|
| H2 | Structured condition causes more omissions (no action when action needed) |
| H3 | Structured condition causes more argument errors (right tool, wrong args) |
| H4 | Effect replicates across model families (Claude, GPT-4, Gemini) |

### Exploratory

| ID | Hypothesis |
|----|------------|
| H5 | Friction is larger for ambiguous tasks than clear tasks |
| H6 | Friction increases with task complexity |

---

## Experimental Design

### Conditions

| Condition | Format | Implementation |
|-----------|--------|----------------|
| **Structured (ST)** | Native tool calling | Provider's `tools` API parameter |
| **Free-form (NL)** | Natural language | Prose description of intended action |

### Simulated Tools

| Tool | Description | Arguments |
|------|-------------|-----------|
| `read_file` | Read file contents | `path: str` |
| `write_file` | Write to file | `path: str, content: str` |
| `search_codebase` | Search for pattern | `query: str, path?: str` |
| `run_command` | Execute shell command | `command: str` |
| `ask_user` | Request clarification | `question: str` |
| `task_complete` | Signal completion | `summary: str` |

### Task Categories (Draft)

| Category | Description |
|----------|-------------|
| Clear single-step | Unambiguous, one tool needed |
| Clear multi-step | Unambiguous, 2-4 tools needed |
| Ambiguous | Multiple valid approaches |
| Error recovery | Tool returns failure, model must adapt |
| Information gathering | Requires search before action |

---

## Implementation Roadmap

### Phase 1: Infrastructure (Current)

- [ ] Create `SimulatedEnvironment` class (stateful tool simulation)
- [ ] Create `AgenticHarness` class (task loop with conditions)
- [ ] Implement tool definitions for each provider
- [ ] Create `ActionExtractor` (NL → structured action)
- [ ] Write basic tests

### Phase 2: Task Design

- [ ] Write 10-15 pilot tasks
- [ ] Define simulation states and success criteria
- [ ] Run pilot experiments
- [ ] Iterate on task difficulty and measurement

### Phase 3: Pre-registration

- [ ] Finalize hypotheses and analysis plan
- [ ] Lock analysis code
- [ ] Submit to OSF
- [ ] Expand to full 50 tasks

### Phase 4: Data Collection

- [ ] Run full experiment across 3 model families
- [ ] 5 trials per task-condition
- [ ] Apply action extraction and correctness judgment

### Phase 5: Analysis & Write-up

- [ ] Run pre-registered analysis
- [ ] Write paper

---

## Infrastructure to Build

### New Files

```
experiments/
├── core/
│   ├── simulation.py      # SimulatedEnvironment class
│   └── harness.py         # AgenticHarness class
├── scenarios/
│   └── agentic_tasks.py   # Task definitions
├── run_agentic.py         # Experiment runner
└── results/
    └── agentic/           # Results directory
```

### Reusable Code

Keep and adapt:
- `experiments/core/api_providers.py` — API wrappers
- `experiments/core/judge.py` — LLM judge (adapt for action evaluation)
- `experiments/core/statistics.py` — Statistical functions
- `experiments/core/checkpoint.py` — Checkpointing

---

## Key Design Decisions

### Prompt Symmetry

Both conditions receive identical information:
- Same tool descriptions
- Same task description
- Same simulated tool results

The only difference is output format.

### Symmetric Evaluation

Both conditions go through identical evaluation:
1. Extract intended action (trivial for ST, regex+LLM for NL)
2. Apply same correctness criteria
3. Use same judge model

### Simulation Fidelity

The simulation should:
- Maintain state (files created are readable)
- Introduce realistic errors (20% failure rate)
- Provide partial information (require follow-up)
- Support multi-step tasks

---

## Open Questions

1. **How to handle NL extraction failures?** Should they count as model failures or be excluded?

2. **What's the right number of tasks per category?** Need to balance power with API costs.

3. **Should we test temperature variations?** Or fix temperature and treat as within-subject.

4. **How to define "correct" for ambiguous tasks?** Need a rubric or multiple valid answers.

---

## References

- Claude tool use: https://docs.anthropic.com/en/docs/build-with-claude/tool-use
- OpenAI function calling: https://platform.openai.com/docs/guides/function-calling
- Gemini function calling: https://ai.google.dev/gemini-api/docs/function-calling
