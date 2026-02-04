# Contributing Guide

Thank you for your interest in contributing to the Format Friction research project.

## Project Status

This is a **pre-registered study** with locked analysis code. Contributions are welcome but must respect the pre-registration integrity.

## What You Can Contribute

| Area | Welcome? | Notes |
|------|----------|-------|
| Bug fixes | ✅ Yes | For non-locked files only |
| Documentation | ✅ Yes | Improvements, clarifications, typos |
| Replication studies | ✅ Yes | Using different models or domains |
| New analyses | ✅ Yes | Must be clearly marked as exploratory |
| Visualization | ✅ Yes | New figures, improved plots |

## What You Cannot Modify

These files are **locked** after pre-registration and must not be changed:

- `experiments/run_analysis.py` - Pre-registered analysis code
- `experiments/scenarios/signal_detection.py` - Scenario definitions
- `paper/PREREGISTRATION.md` - Hypotheses and statistical plan

Checksums are verified in `verification/preregistration_lock.json`.

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

# CRITICAL: Unset API keys (uses subscription CLIs only)
unset ANTHROPIC_API_KEY OPENAI_API_KEY GOOGLE_API_KEY

# Verify setup
python -c "from experiments.core import statistics, judge, cli_wrappers, checkpoint; print('OK')"
```

### 2. Understand the Codebase

- **`README.md`** - Project overview and verification guide
- **`AGENTS.md`** - Detailed instructions for working with the code
- **`PLAN.md`** - Full research plan (71 KB)

### 3. Run Tests

```bash
# Validate scenarios
python -m experiments.cli validate

# Run analysis on existing data
python experiments/run_analysis.py \
  experiments/results/primary/signal_detection_20260203_074411_judged.json
```

---

## Coding Standards

### Style

- **Type hints**: Always include type annotations
- **Descriptive names**: Prefer clarity over brevity
- **Docstrings**: Required for public functions

### Example

```python
def compute_friction(
    detection_rate: float,
    compliance_rate: float
) -> float:
    """
    Compute format friction as the gap between detection and compliance.

    Args:
        detection_rate: Proportion of trials where signal was detected
        compliance_rate: Proportion of trials with compliant XML output

    Returns:
        Friction value (detection_rate - compliance_rate)
    """
    return detection_rate - compliance_rate
```

---

## Submitting Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Keep commits focused and atomic
- Write clear commit messages
- Don't modify locked files

### 3. Verify Integrity

```bash
# Ensure pre-registration locks are intact
python scripts/verify_checkpoint.py --lock-only
```

### 4. Submit Pull Request

- Describe what changed and why
- Reference any related issues
- Mark exploratory analyses clearly

---

## Exploratory Analyses

New analyses beyond the pre-registered plan are welcome but must be:

1. **Clearly labeled** as exploratory (not confirmatory)
2. **Placed in separate files** (not modifying locked code)
3. **Documented** with rationale and limitations

Example directory structure:
```
experiments/
├── run_analysis.py           # LOCKED - pre-registered
├── exploratory/              # New exploratory analyses go here
│   └── your_analysis.py
```

---

## Questions?

- Open an issue for bugs or feature requests
- See `AGENTS.md` for detailed technical documentation
- Check `verification/MANUAL_ACTIONS.md` for pipeline operations
