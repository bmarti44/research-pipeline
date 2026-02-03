# Experiment Results

This directory contains experimental data for the Format Friction research paper.

**For detailed documentation of all data files, see [DATA_MANIFEST.md](DATA_MANIFEST.md).**

## Directory Structure

```
experiments/results/
├── primary/          # Main evidence for paper claims (N=750 judged, N=1500 raw)
├── replication/      # Supporting signal detection runs
├── exploratory/      # Small-scale exploration runs
├── failed/           # API errors, incomplete runs (UNUSABLE)
├── pilot/            # Pilot runs (NOT for claims without caveats)
├── validation/       # Human annotation data
├── legacy/           # Old nl_vs_structured Study 1 data
└── archive/          # Pre-existing early development files
```

## Primary Evidence

The main experimental evidence is in `primary/`:

- **signal_detection_20260203_074411_judged.json**: N=750 trials with LLM judge annotations
  - Supports Tables 1-5, Sections 4.4-4.8
  - Key metrics: NL 87.0%, ST 81.4%, Format Friction 12.2pp

- **signal_detection_20260203_121413.json**: N=1500 observations (raw, regex-based)
  - Supports Section 4.7 measurement comparison

## Important Notes

1. **Failed runs**: Files in `failed/` have API errors and are unusable
2. **Pilot data**: Files in `pilot/` are preliminary (n=10) and cannot support quantitative claims
3. **Paper alignment**: Section 4.9 (Two-Pass Recovery) was revised to reflect pilot status

## Running New Experiments

```bash
# Signal detection experiment
python -m experiments.signal_detection_experiment --trials 10

# Original nl_vs_structured experiment
python -m experiments.natural_language_intent_experiment --trials 5
```

Results are automatically timestamped and saved to this directory.

## Documentation

- [DATA_MANIFEST.md](DATA_MANIFEST.md) - Complete file documentation and claim mapping
- [CHANGELOG.md](CHANGELOG.md) - Data reorganization history
