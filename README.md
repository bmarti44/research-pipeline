# Format Friction: Do LLMs Detect Signals They Fail to Report?

A pre-registered study investigating whether Large Language Models detect user signals (frustration, urgency, blocking issues) but fail to report them when constrained to structured XML output.

## For Academic Reviewers

This repository contains a **pre-registered replication study** with full transparency measures. Below is everything you need to evaluate the scientific rigor of this work.

### Study Overview

| Aspect | Details |
|--------|---------|
| **Research Question** | Do LLMs detect implicit user signals but fail to report them in structured format? |
| **Design** | Within-subject comparison: same scenarios, two output formats (natural language vs. XML) |
| **Sample Size** | 75 scenarios × 10 trials × 2 conditions = 1,500 observations |
| **Primary Outcome** | "Format friction" = detection rate − compliance rate on IMPLICIT scenarios |
| **Pre-registration** | Hypotheses, analysis code, and scenarios locked before data collection |

### Key Findings

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Primary test (H1) | p = 0.011 | Significant format friction exists |
| Mean friction | 20.5 pp | [CI: 0.9, 39.1] |
| Effect direction | 11 positive, 2 negative | 85% of scenarios show friction |
| Human validation | κ = 0.81 overall | Substantial judge-human agreement |

---

## Verifying Scientific Rigor

### 1. Pre-Registration Integrity

All hypotheses and analysis code were locked before data collection:

```bash
# View locked file checksums
cat verification/preregistration_lock.json

# Verify current files match pre-registered versions
python scripts/verify_checkpoint.py --lock-only
```

**Locked files** (must not be modified post-registration):
- `experiments/run_analysis.py` - Analysis code
- `experiments/scenarios/signal_detection.py` - 75 scenario definitions
- `paper/PREREGISTRATION.md` - Hypotheses and statistical plan

### 2. Reproducibility

All results are deterministically reproducible:

```bash
# Activate environment
source .venv/bin/activate

# Re-run pre-registered analysis (must produce identical output)
python experiments/run_analysis.py \
  experiments/results/primary/signal_detection_20260203_074411_judged.json

# Compare with stored results
cat experiments/results/analysis/preregistered_analysis.json
```

**Determinism guarantees**:
- Fixed random seed (42) for all stochastic operations
- 10,000 bootstrap replicates for confidence intervals
- Cached model responses (no re-querying required)

### 3. Data Availability

| File | Description | Size |
|------|-------------|------|
| `experiments/results/primary/signal_detection_*_judged.json` | Full experiment data with judge scores | 2.5 MB |
| `experiments/results/validation/validation_annotation_*.csv` | Human annotations (N=150) | ~500 KB |
| `experiments/results/analysis/preregistered_analysis.json` | Statistical analysis output | 1 KB |

### 4. Human Validation

Judge reliability was validated against human annotations:

```bash
# View human annotation data
head experiments/results/validation/validation_annotation_*.csv

# View agreement metrics
cat experiments/results/analysis/preregistered_analysis.json
```

| Stratum | κ | Agreement | N |
|---------|---|-----------|---|
| Overall | 0.81 | 90.7% | 150 |
| EXPLICIT | 1.00 | 100% | 30 |
| IMPLICIT | 0.41 | 76% | 66 |
| CONTROL | 1.00 | 100% | 54 |

**Note**: Lower IMPLICIT agreement reflects genuine ambiguity in what constitutes "signal detection" vs. "signal handling."

### 5. Pipeline Verification

Every phase of the experiment has a checkpoint with SHA256 verification:

```bash
# View all phase statuses
cat verification/FINAL_REPORT.md

# Inspect individual checkpoints
ls verification/checkpoint_*.json
cat verification/checkpoint_5.json  # Analysis phase
```

---

## Methodology

### Experimental Design

1. **Scenarios**: 75 software engineering help requests with varying signal ambiguity
   - 15 EXPLICIT (clear emotional language)
   - 22 IMPLICIT (contextual signals only)
   - 23 CONTROL (neutral, no signal)
   - 15 BORDERLINE (ambiguous)

2. **Conditions**: Each scenario tested in two formats
   - **Natural Language (NL)**: "If you notice frustration, acknowledge it naturally"
   - **Structured (ST)**: "If you notice frustration, output `<signal type='frustration'>reason</signal>`"

3. **Measurement**:
   - **Detection**: Did the model recognize the signal? (judged by LLM + human validation)
   - **Compliance**: Did the model output the required XML tag?
   - **Friction**: Detection rate − Compliance rate

### Statistical Analysis

| Test | Application | Specification |
|------|-------------|---------------|
| Sign test | Primary H1 | One-sided, scenario-level |
| Bootstrap CI | Effect size | 10,000 replicates, percentile method |
| Cohen's κ | Agreement | Stratified by ambiguity level |
| Wilson interval | Proportions | For detection/compliance rates |

### Potential Limitations

1. **Same-family judge**: Claude judges Claude (mitigated by human validation)
2. **Single model tested**: Results specific to Claude Sonnet 4
3. **Software domain only**: May not generalize to other domains
4. **IMPLICIT ambiguity**: Human-judge disagreement on implicit signals

See `backup/paper/LIMITATIONS.md` for full discussion.

---

## Repository Structure

```
├── README.md                    # This file (reviewer guide)
├── AGENTS.md                    # Instructions for AI agents
├── PLAN.md                      # Full research plan (71 KB)
├── experiments/
│   ├── cli.py                   # Main CLI entrypoint
│   ├── run_analysis.py          # Pre-registered analysis (LOCKED)
│   ├── core/                    # Statistics, judge, checkpoint modules
│   ├── scenarios/
│   │   └── signal_detection.py  # 75 scenarios (LOCKED)
│   └── results/
│       ├── primary/             # Experiment data
│       ├── validation/          # Human annotations + instructions
│       └── analysis/            # Statistical outputs
├── scripts/
│   ├── run_pipeline.sh          # Master pipeline script
│   └── verify_checkpoint.py     # Verification tools
├── verification/
│   ├── FINAL_REPORT.md          # Pipeline completion status
│   ├── MANUAL_ACTIONS.md        # Human action checklist
│   ├── checkpoint_*.json        # Phase verification
│   └── preregistration_lock.json # SHA256 of locked files
└── backup/
    └── paper/                   # Paper drafts and figures
```

---

## Running the Analysis

### Quick Start

```bash
# 1. Setup environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Verify setup
python -c "from experiments.core import statistics; print('OK')"

# 3. Run pre-registered analysis
python experiments/run_analysis.py \
  experiments/results/primary/signal_detection_20260203_074411_judged.json
```

### Full Pipeline (for replication)

```bash
# Unset API keys (uses subscription CLIs only)
unset ANTHROPIC_API_KEY OPENAI_API_KEY GOOGLE_API_KEY

# Run pipeline
./scripts/run_pipeline.sh
```

See `AGENTS.md` for detailed instructions.

---

## References

Johnson, A., Pain, E., & West, M. (2025). Natural Language Tools: Decoupling Tool Selection from Response Generation. *arXiv preprint arXiv:2510.14453*.

Sclar, M., Choi, Y., Tsvetkov, Y., & Suhr, A. (2024). Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design. *NAACL 2024 Findings*.

Tam, Z. R., et al. (2024). Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models. *arXiv preprint arXiv:2408.02442*.

Wang, D. Y., et al. (2025). SLOT: Structuring the Output of Large Language Models. *arXiv preprint arXiv:2505.04016*.

---

## Citation

```bibtex
@misc{formatfriction2026,
  title={Format Friction: Do LLMs Detect Signals They Fail to Report?},
  author={[Authors]},
  year={2026},
  howpublished={Pre-registered study},
  note={OSF registration: [URL]}
}
```

---

## Contact

For questions about methodology or data access, please open an issue in this repository.
