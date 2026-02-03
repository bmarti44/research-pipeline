# Data Changelog

This document tracks changes to experimental data organization for academic transparency.

---

## 2026-02-03: Judge Validation Completed

### Purpose

Complete human annotation of validation samples to establish inter-rater reliability for the LLM judge.

### Data Collected

**In `validation/`**:
- `validation_annotation_20260203_074834.csv` - 150 samples with human_label column completed

### Key Results

| Metric | Value |
|--------|-------|
| Overall κ | 0.812 |
| Overall Agreement | 90.7% |
| EXPLICIT κ | 1.000 (perfect) |
| CONTROL κ | 1.000 (perfect) |
| BORDERLINE κ | 0.867 |
| IMPLICIT κ | 0.406 (flagged) |

**Confusion Matrix (Judge vs Human)**:
- True Positives: 74
- True Negatives: 62
- False Positives: 6 (judge over-detected)
- False Negatives: 8 (judge missed)

### Paper Updates

- Section 4.3 updated: κ = 0.81 with breakdown by ambiguity level
- Limitation #3 revised: Notes IMPLICIT stratum disagreement reflects genuine ambiguity
- Removed "[κ PENDING]" placeholder

### Finding: Implicit Signal Ambiguity

Human annotators tended to mark "NO" when the model addressed an implicit issue helpfully without explicitly acknowledging it as a signal. This reflects a distinction between *detecting* a signal and *handling* it—the model may resolve an issue without categorizing it, which our binary judge counts as detection but humans may not.

---

## 2026-02-03: Two-Pass Extraction Data Collection

### Purpose

Run full two-pass extraction experiment to replace pilot data with valid evidence.

### Data Collected

**To `primary/`** (New valid two-pass results):
- `two_pass_sonnet_nl_20260203_125603.json` - N=750 trials, Sonnet extraction
- `two_pass_qwen7b_nl_20260203_131141.json` - N=750 trials, Qwen 7B extraction

### Key Results

| Model | Recovery Rate | Precision | Type Accuracy |
|-------|---------------|-----------|---------------|
| Sonnet | 64.9% (48/74) | 97.3% | 88.5% |
| Qwen 7B | 39.2% (29/74) | 96.7% | 82.6% |

### Paper Updates

Section 4.9 updated from "Preliminary" to validated results:
- Table 6 now contains real data
- Abstract, contributions, conclusion, and discussion updated with actual recovery rates
- Limitations section revised to note single-task scope rather than pilot status

### Technical Note

Ran with `ANTHROPIC_API_KEY=` (empty) to use Claude Code subscription instead of API credits.

---

## 2026-02-03: Academic Cleanup and Reorganization

**Prior Commit**: `f9cc502dcc9f8b18a1f1600a9731059176326007`

### Purpose

Reorganize experimental data to:
1. Clearly distinguish valid primary evidence from pilot/failed runs
2. Align paper claims with available evidence
3. Improve academic rigor and transparency

### Changes Made

#### Directory Structure Created

```
experiments/results/
├── primary/       # Main evidence for paper claims
├── replication/   # Supporting runs
├── exploratory/   # Small-scale exploration
├── failed/        # API errors, incomplete
├── pilot/         # Pilot runs (not for claims)
├── validation/    # Human annotation data
└── legacy/        # Old Study 1 data
```

#### Files Moved

**To `primary/`** (Main evidence):
- `signal_detection_20260203_074411_judged.json` - N=750 judged trials
- `signal_detection_20260203_121413.json` - N=1500 raw observations

**To `replication/`** (Supporting signal detection runs):
- `signal_detection_20260203_010117.json`
- `signal_detection_20260203_024909.json`
- `signal_detection_20260203_012453.json`
- `signal_detection_20260203_012650.json`
- `signal_detection_20260203_013054.json`
- Corresponding `signal_validation_*.json` files

**To `exploratory/`** (Small runs, analysis outputs):
- `signal_detection_20260202_221456.json`
- `signal_detection_20260202_221655.json`
- `signal_validation_20260202_*.json`
- `signal_validation_20260203_121413.json`
- `quick_recovery_test.json`
- `analysis_report_*.txt`

**To `failed/`** (API errors):
- `two_pass_sonnet_nl_20260203_113046.json` - All responses: "Credit balance is too low"
- `two_stage_20260203_094518.json` - All responses: "Credit balance is too low"

**To `pilot/`** (Pilot runs):
- `two_pass_qwen7b_nl_20260203_113119.json` - n=10, pilot
- `two_pass_qwen7b_nl_20260203_114549.json` - n=10, pilot

**To `validation/`** (Human annotation):
- `validation_annotation_20260203_074834.csv`
- `validation_key_20260203_074834.csv`

**To `legacy/`** (Study 1 data):
- All `nl_vs_structured_*.json` files (~30 files)
- All `validation_samples_*.json` files (~30 files)

### Documentation Created

- `DATA_MANIFEST.md` - Complete documentation of all data files
- `CHANGELOG.md` - This file

### Paper Revisions

**Section 4.9 (Two-Pass Recovery)** was revised because:
1. Only pilot runs exist (n=10 each)
2. Pilot samples had `total_silent_failures: 0` - no actual silent failures to recover
3. Specific statistics (55% recovery, 49 failures) were not supported by data

The section was reframed as "preliminary exploration" demonstrating extraction viability rather than validated recovery rates.

### Data Integrity

- **No files deleted** - All data preserved
- **Git history preserved** - Full provenance available
- **Failed experiments documented** - Marked as unusable with explanation
- **Pilot data distinguished** - Cannot support quantitative claims

### Verification

After reorganization:
- All files accounted for in DATA_MANIFEST.md
- Paper claims verified against evidence files
- Experiment scripts verified to still function

### Dead Code Archived

The following experiment code was archived to `experiments/archive/` because it only produced failed/unusable data:

| File | Reason |
|------|--------|
| `two_stage_experiment.py` | Produced only API error data (failed/) |
| `tool_definitions.py` | Only used by two_stage_experiment.py |
| `scenarios/tool_calling.py` | Only used by two_stage_experiment.py |

These files are preserved for reference but are not part of the active codebase.

---

## Pre-2026-02-03: Original State

Prior to this cleanup:
- All result files were in flat `experiments/results/` directory
- Some files in pre-existing `archive/` subdirectory
- No systematic documentation of file validity
- Paper Section 4.9 contained statistics not supported by available data
