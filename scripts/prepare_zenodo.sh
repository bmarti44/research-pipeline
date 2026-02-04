#!/bin/bash
# Prepare release package for Zenodo upload
#
# Creates a zip archive containing:
# - README.md with reproduction instructions
# - CHECKSUMS.sha256 with SHA256 of all files
# - data/ directory with scenarios and results
# - code/ directory with analysis code
# - paper/ directory with PDF and pre-registration
# - verification/ directory with checkpoints
#
# Usage: bash scripts/prepare_zenodo.sh

set -e

RELEASE_DIR="zenodo_release"
ARCHIVE_NAME="zenodo_release.zip"

echo "=== Preparing Zenodo Release Package ==="
echo ""

# Clean up previous release
rm -rf "$RELEASE_DIR" "$ARCHIVE_NAME"

# Create directory structure
mkdir -p "$RELEASE_DIR"/{data,code/experiments,paper,verification}

# =============================================================================
# README
# =============================================================================
cat > "$RELEASE_DIR/README.md" << 'EOF'
# Format Friction in LLM Tool Calling: Replication Package

This archive contains all data, code, and documentation for the pre-registered
replication study of format friction in LLM tool calling.

## Contents

- `data/` - Raw and processed experiment data
  - `scenarios.json` - Scenario definitions
  - `raw_responses.json` - Raw model outputs
  - `judged_responses.json` - Judge scores
  - `analysis_results.json` - Statistical analysis results

- `code/` - Analysis code
  - `experiments/` - Python experiment and analysis code
  - `requirements.txt` - Python dependencies

- `paper/` - Publication materials
  - `FORMAT_FRICTION.pdf` - Final paper
  - `PREREGISTRATION.md` - Pre-registration document

- `verification/` - Reproducibility verification
  - `checkpoint_*.json` - Per-phase verification files
  - `preregistration_lock.json` - Checksums of locked files
  - `FINAL_REPORT.md` - Summary of all verifications

## Reproduction Instructions

1. Install Python 3.11+
2. Install dependencies: `pip install -r code/requirements.txt`
3. Run analysis: `python code/experiments/run_analysis.py data/judged_responses.json`

## Citation

[Citation information]

## License

[License information]

## Contact

[Contact information]
EOF

# =============================================================================
# Data files
# =============================================================================
echo "Copying data files..."

# Scenarios
if [ -f experiments/scenarios/signal_detection.py ]; then
    python -c "
import json
import sys
sys.path.insert(0, '.')
from experiments.scenarios.signal_detection import (
    EXPLICIT_SCENARIOS, IMPLICIT_SCENARIOS, BORDERLINE_SCENARIOS, CONTROL_SCENARIOS
)
scenarios = [s.to_dict() for s in EXPLICIT_SCENARIOS + IMPLICIT_SCENARIOS + BORDERLINE_SCENARIOS + CONTROL_SCENARIOS]
print(json.dumps({'scenarios': scenarios}, indent=2))
" > "$RELEASE_DIR/data/scenarios.json" 2>/dev/null || echo "Could not export scenarios"
fi

# Raw and judged results (copy most recent)
for pattern in "signal_detection_*_judged.json" "signal_detection_*.json"; do
    LATEST=$(ls -t experiments/results/primary/$pattern 2>/dev/null | head -1)
    if [ -n "$LATEST" ] && [ -f "$LATEST" ]; then
        case "$pattern" in
            *judged*)
                cp "$LATEST" "$RELEASE_DIR/data/judged_responses.json"
                ;;
            *)
                cp "$LATEST" "$RELEASE_DIR/data/raw_responses.json"
                ;;
        esac
    fi
done

# Analysis results
if [ -f experiments/results/analysis/preregistered_analysis.json ]; then
    cp experiments/results/analysis/preregistered_analysis.json "$RELEASE_DIR/data/analysis_results.json"
fi

# =============================================================================
# Code
# =============================================================================
echo "Copying code files..."

# Copy Python modules
cp -r experiments/core "$RELEASE_DIR/code/experiments/" 2>/dev/null || true
cp experiments/run_analysis.py "$RELEASE_DIR/code/experiments/" 2>/dev/null || true
cp experiments/cli.py "$RELEASE_DIR/code/experiments/" 2>/dev/null || true

# Copy or create requirements.txt
if [ -f requirements.txt ]; then
    cp requirements.txt "$RELEASE_DIR/code/"
else
    cat > "$RELEASE_DIR/code/requirements.txt" << 'EOF'
numpy>=1.24.0
scipy>=1.10.0
statsmodels>=0.14.0
scikit-learn>=1.2.0
pandas>=2.0.0
matplotlib>=3.7.0
jinja2>=3.1.0
diptest>=0.7.0
sentence-transformers>=2.2.0
EOF
fi

# =============================================================================
# Paper
# =============================================================================
echo "Copying paper files..."

if [ -f paper/output/FORMAT_FRICTION.pdf ]; then
    cp paper/output/FORMAT_FRICTION.pdf "$RELEASE_DIR/paper/"
elif [ -f paper/generated/FORMAT_FRICTION.md ]; then
    cp paper/generated/FORMAT_FRICTION.md "$RELEASE_DIR/paper/"
fi

if [ -f paper/PREREGISTRATION.md ]; then
    cp paper/PREREGISTRATION.md "$RELEASE_DIR/paper/"
fi

# =============================================================================
# Verification
# =============================================================================
echo "Copying verification files..."

cp verification/checkpoint_*.json "$RELEASE_DIR/verification/" 2>/dev/null || true
cp verification/preregistration_lock.json "$RELEASE_DIR/verification/" 2>/dev/null || true
cp verification/osf_registration.json "$RELEASE_DIR/verification/" 2>/dev/null || true
cp verification/FINAL_REPORT.md "$RELEASE_DIR/verification/" 2>/dev/null || true

# =============================================================================
# Checksums
# =============================================================================
echo "Computing checksums..."

cd "$RELEASE_DIR"

# Use appropriate checksum tool
if command -v sha256sum &> /dev/null; then
    find . -type f -not -name "CHECKSUMS.sha256" -exec sha256sum {} \; > CHECKSUMS.sha256
else
    find . -type f -not -name "CHECKSUMS.sha256" -exec shasum -a 256 {} \; > CHECKSUMS.sha256
fi

cd ..

# =============================================================================
# Create archive
# =============================================================================
echo "Creating archive..."

zip -r "$ARCHIVE_NAME" "$RELEASE_DIR"

# Report
echo ""
echo "=== Release Package Created ==="
echo "Archive: $ARCHIVE_NAME"
ls -la "$ARCHIVE_NAME"
echo ""
echo "Contents:"
unzip -l "$ARCHIVE_NAME" | tail -20

# Cleanup
# rm -rf "$RELEASE_DIR"  # Keep for inspection

echo ""
echo "Ready for upload to Zenodo: https://zenodo.org/deposit/new"
