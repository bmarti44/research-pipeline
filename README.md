# Autonomous Research Pipeline

A fully autonomous, academically rigorous research pipeline that takes a natural language hypothesis about LLM/AI behavior and produces a complete, PDF-ready academic paper. Orchestrated by a Claude Code agent operating as a hierarchical supervisor with 9 specialized expert sub-agents.

## How It Works

You provide a hypothesis in natural language. The orchestrator agent interviews you to clarify the research design, then autonomously executes the entire pipeline — study design, preregistration, pilot study, main data collection, statistical analysis, manuscript writing, and PDF export — with deterministic verification gates at every stage.

```
HYPOTHESIS → INTERVIEW → STUDY DESIGN → [5-ROUND REVIEW] → PREREGISTRATION (hash-locked)
    → PILOT STUDY → [5-ROUND REVIEW] → CHECKPOINT
    → MAIN STUDY (configure → generate trials → execute → evaluate → analyze)
    → [5-ROUND REVIEW] → CHECKPOINT
    → (additional studies?) → CROSS-STUDY ANALYSIS
    → MANUSCRIPT → [5-ROUND REVIEW] → PUBLISHABILITY DECISION → PDF EXPORT
```

No stage proceeds without programmatic proof that the previous stage passed. Every gate is a boolean check against data on disk.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set API keys for the models you want to study
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

Requires Python 3.10+.

### Optional: PDF Export

The pipeline can export manuscripts to PDF. This requires one of the following system tools (not Python packages):

- [pandoc](https://pandoc.org/installing.html) (recommended)
- [wkhtmltopdf](https://wkhtmltopdf.org/downloads.html)

If neither is installed, the pipeline will still generate Markdown and HTML output but will skip PDF generation.

## Usage

### Starting a New Paper

Open Claude Code in this repository and describe your hypothesis. The orchestrator (defined in `AGENTS.md`) will:

1. **Interview you** to clarify IVs, DVs, effect sizes, confounds, and scope
2. **Design the study** by spawning Research Designer, Methodologist, Statistician, and Domain Expert sub-agents
3. **Run 5 rounds of review** (Methodologist → Statistician → Skeptic → Technical Reviewer → Ethics/Domain Expert)
4. **Preregister** the design with SHA-256 hash locking
5. **Run a pilot study** (20% of trials, different seed) to validate the apparatus
6. **Execute the main study** with batching, concurrency, rate limiting, and checkpointing
7. **Analyze results** using preregistered statistical tests
8. **Write the manuscript** with proper APA formatting and traceable statistical claims
9. **Export to PDF**

### Pipeline CLI

The `pipeline/` module also provides a CLI for direct study management:

```bash
# Create a new study from a template
python -m pipeline new <study_name> [--template basic|tool_calling] [--paper <paper_name>]

# Run a specific stage
python -m pipeline run <study_name> --stage <stage_name>

# Run the full pipeline
python -m pipeline run <study_name> --full

# Resume from a checkpoint
python -m pipeline run <study_name> --resume

# Run and evaluate a pilot study
python -m pipeline pilot <study_name>

# Lock a preregistration
python -m pipeline prereg <study_name>

# Verify preregistration integrity
python -m pipeline prereg <study_name> --verify

# Verify stage gates
python -m pipeline verify <study_name> [--stage <stage>] [--all]

# Check study status
python -m pipeline status [<study_name>]

# Create a replication study
python -m pipeline replicate <original_study> <new_study> [--type direct|conceptual]
```

## Project Structure

```
├── AGENTS.md                  # Orchestrator instructions (agent protocol)
├── PLAN.md                    # Architecture and design reference
├── pipeline/                  # Reusable pipeline infrastructure
│   ├── __main__.py            #   CLI entry point
│   ├── cli.py                 #   Command-line interface
│   ├── models.py              #   Data classes: Condition, Task, Trial, Scenario, StudyDesign
│   ├── schemas.py             #   Pydantic validation for all YAML configs
│   ├── runner.py              #   Stage execution engine
│   ├── executor.py            #   Batching, concurrency, rate limiting, checkpointing
│   ├── evaluators.py          #   Evaluator registry (strict, intent, functional modes)
│   ├── stats.py               #   Statistical tests, power analysis, effect sizes
│   ├── analysis.py            #   Statistical analysis runner
│   ├── verification.py        #   Verification gate logic
│   ├── preregistration.py     #   Hash-based preregistration with tamper detection
│   ├── pilot.py               #   Pilot study management
│   ├── review.py              #   5-round expert review orchestration
│   ├── review_gates.py        #   Review gate enforcement
│   ├── interview.py           #   User interview for hypothesis clarification
│   ├── orchestrator.py        #   Supervisor agent logic
│   ├── manuscript.py          #   Manuscript generation
│   ├── pdf_export.py          #   PDF generation from markdown
│   ├── api.py                 #   LLM API wrappers (Anthropic, OpenAI, Google)
│   ├── adaptive.py            #   Adaptive stopping rules (O'Brien-Fleming, Pocock)
│   ├── replication.py         #   Replication framework (direct, conceptual, extension)
│   ├── multimodel.py          #   Multi-model comparison support
│   └── utils.py               #   Hashing, timestamps, file I/O
├── templates/                 # Study templates
│   └── study/
│       ├── basic/             #   Minimal study config
│       └── tool_calling/      #   Tool-use specific config
└── papers/                    # Research papers (each self-contained)
    └── <paper_name>/
        ├── paper.yaml         #   Paper metadata, hypotheses, results
        ├── interview.yaml     #   User interview data
        ├── checkpoints/       #   State checkpoints + PROGRESS.md
        ├── studies/
        │   └── <study_name>/
        │       ├── config.yaml          # Study configuration
        │       ├── tasks.yaml           # Task definitions
        │       ├── analysis_plan.yaml   # Preregistered analysis plan
        │       ├── research_plan.md     # Narrative rationale
        │       ├── preregistration/     # Hash-locked design
        │       ├── pilot/               # Pilot study data
        │       ├── stages/              # Main study execution data
        │       │   ├── 1_configure/
        │       │   ├── 2_generate/
        │       │   ├── 3_execute/
        │       │   ├── 4_evaluate/
        │       │   └── 5_analyze/
        │       ├── reviews/             # 5 review files per checkpoint
        │       └── deviations.yaml      # Post-prereg changes (if any)
        ├── combined_analysis/   # Cross-study synthesis
        └── manuscript/          # Paper output
            ├── sections/
            ├── figures/
            ├── tables/
            └── output/
                └── paper_final.pdf
```

## Sub-Agent Architecture

The orchestrator spawns stateless sub-agents for domain-specific tasks. Each receives full context in its task description and produces structured, verifiable file output.

| Agent | Role |
|-------|------|
| Research Designer | Hypothesis refinement, IV/DV operationalization |
| Methodologist | Experimental design, confound control, counterbalancing |
| Statistician | Power analysis, test selection, assumption checking |
| Domain Expert | Model-specific knowledge, prompt engineering |
| Skeptic | Adversarial critique, alternative explanations, fatal flaws |
| Technical Reviewer | Code correctness, reproducibility, data integrity |
| Writing Specialist | Scientific prose, APA formatting, logical flow |
| Ethics Reviewer | Bias detection, responsible disclosure, limitations |
| Replication Specialist | Independent verification, reproduction from archived data |

## Key Design Principles

- **Paper-first organization** — each paper is a self-contained project
- **Stage-gated execution** — no stage proceeds without passing its verification gate
- **5-round expert review** — every milestone reviewed by 5 specialized agents
- **Preregistration integrity** — SHA-256 hash locking before data collection; any post-hoc deviation is documented
- **Checkpoint-driven resumability** — state written to disk after every milestone
- **Reproducibility by construction** — seeded randomization, environment locks, archived data

## Documentation

- **`AGENTS.md`** — Complete orchestrator protocol, stage definitions, review rules, and file schemas
- **`PLAN.md`** — Architecture reference and design rationale
- **`CONTRIBUTING.md`** — How to contribute (respecting preregistration locks)

## License

- **Papers and figures**: CC-BY-4.0
- **Code**: MIT
- **Data**: CC0-1.0

See [LICENSE](LICENSE) for details.
