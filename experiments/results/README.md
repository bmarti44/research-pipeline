# Experiment Results

## Canonical Results

- **nl_vs_structured_20260201_224648.json**: Full experiment results for the Format Friction paper
  - 51 scenarios × 5 trials × 2 conditions = 510 observations
  - Includes scenario-level analysis, verification language metrics, fidelity comparisons
  - Version: v5_review_fixes (includes all peer review fixes)

## Archive

Old development files are stored in `archive/` for reference:
- `results_*.json`, `metadata_*.json`: Early experiment iterations
- `memory_experiment_*.json`: Initial memory tool experiments
- `intent_vs_tool_*.json`: Intermediate experiments
- Earlier `nl_vs_structured_*.json` files: Development runs
- `analysis_*.md`: Old analysis summaries

## Running New Experiments

```bash
python -m experiments.natural_language_intent_experiment --trials 5
```

Results are automatically timestamped and saved to this directory.
