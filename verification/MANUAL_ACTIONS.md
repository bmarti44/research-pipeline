# Manual Actions Required

This document describes the manual human actions required to complete the Format Friction experiment pipeline.

## Overview

The pipeline has **two manual gates** that require human intervention:

| Phase | Gate | Action Required |
|-------|------|-----------------|
| 0 | OSF Pre-Registration | Submit pre-registration to OSF and provide URL |
| 4 | Human Validation | Complete annotation of validation samples |

All other phases are fully automated.

---

## Manual Gate 1: OSF Pre-Registration (Phase 0)

### When This Happens
After Phase -1 (code implementation) completes, the pipeline generates `paper/PREREGISTRATION.md` and pauses.

### What You Need To Do

1. **Review the pre-registration document**:
   ```bash
   cat paper/PREREGISTRATION.md
   ```

2. **Submit to OSF**:
   - Go to https://osf.io/prereg/
   - Create a new pre-registration
   - Copy the content from `paper/PREREGISTRATION.md`
   - Submit and obtain the registration URL

3. **Resume the pipeline** with the OSF URL:
   ```bash
   ./scripts/run_pipeline.sh --osf-url https://osf.io/xxxxx/
   ```

   Or set the environment variable:
   ```bash
   export OSF_PREREGISTRATION_URL="https://osf.io/xxxxx/"
   ./scripts/run_pipeline.sh
   ```

### Verification
The pipeline will validate that the URL matches the expected format (`https://osf.io/...`).

---

## Manual Gate 2: Human Validation (Phase 4)

### When This Happens
After Phase 3 (judge scoring) completes, the pipeline generates a validation sample and pauses.

### What You Need To Do

1. **Locate the validation sample**:
   ```bash
   ls experiments/results/validation/validation_sample*.csv
   ```

2. **Read the annotation instructions**:
   ```bash
   cat experiments/results/validation/ANNOTATION_INSTRUCTIONS.md
   ```

3. **Complete annotations**:
   - Open the CSV file
   - For each sample, add your label in the `human_label` column
   - Labels must be exactly `YES` or `NO`
   - See `ANNOTATION_INSTRUCTIONS.md` for detailed criteria

4. **Save the completed annotations** (e.g., `human_annotations.csv`)

5. **Resume the pipeline** with the annotations file:
   ```bash
   ./scripts/run_pipeline.sh --annotations experiments/results/validation/human_annotations.csv
   ```

   Or set the environment variable:
   ```bash
   export HUMAN_ANNOTATIONS_FILE="experiments/results/validation/human_annotations.csv"
   ./scripts/run_pipeline.sh
   ```

### Verification
The pipeline will validate:
- All samples have labels
- Labels are exactly `YES` or `NO`
- No missing or empty values

---

## Current Status

Check `verification/FINAL_REPORT.md` for the current pipeline status:

```bash
cat verification/FINAL_REPORT.md
```

Check individual phase checkpoints:

```bash
ls verification/checkpoint_*.json
```

---

## Troubleshooting

### Pipeline stopped unexpectedly
Check which phase failed:
```bash
cat verification/FINAL_REPORT.md
```

Resume from that phase:
```bash
./scripts/run_pipeline.sh --resume-from <phase_number>
```

### OSF URL rejected
Ensure the URL matches the format `https://osf.io/xxxxx/` (with trailing slash).

### Annotations rejected
Check for:
- Empty cells in `human_label` column
- Values other than `YES` or `NO` (case-sensitive)
- Missing rows

Validate your file:
```bash
python -m experiments.cli validate-annotations --file your_annotations.csv
```

---

## File Locations

| File | Purpose |
|------|---------|
| `paper/PREREGISTRATION.md` | Pre-registration document to submit to OSF |
| `experiments/results/validation/validation_sample*.csv` | Samples to annotate |
| `experiments/results/validation/ANNOTATION_INSTRUCTIONS.md` | How to annotate |
| `verification/FINAL_REPORT.md` | Pipeline status summary |
| `verification/checkpoint_*.json` | Individual phase checkpoints |
