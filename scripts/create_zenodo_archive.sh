#!/bin/bash
# create_zenodo_archive.sh
# Creates a clean archive for Zenodo publication

set -e

VERSION="1.0"
ARCHIVE_NAME="format-friction-v${VERSION}"

echo "Creating Zenodo archive: ${ARCHIVE_NAME}"

# Create clean directory
rm -rf "${ARCHIVE_NAME}"
mkdir -p "${ARCHIVE_NAME}"

# Copy paper
echo "Copying paper..."
mkdir -p "${ARCHIVE_NAME}/paper/figures"
cp paper/FORMAT_FRICTION.md "${ARCHIVE_NAME}/paper/"
cp paper/LIMITATIONS.md "${ARCHIVE_NAME}/paper/"
cp paper/REVIEW.md "${ARCHIVE_NAME}/paper/"
cp paper/figures/*.png "${ARCHIVE_NAME}/paper/figures/" 2>/dev/null || echo "No figures found"

# Copy experiments (code only, not venv or cache)
echo "Copying experiments..."
mkdir -p "${ARCHIVE_NAME}/experiments/scenarios"
mkdir -p "${ARCHIVE_NAME}/experiments/results/primary"
cp experiments/signal_detection_experiment.py "${ARCHIVE_NAME}/experiments/"
cp experiments/judge_scoring.py "${ARCHIVE_NAME}/experiments/"
cp experiments/analyze_judged_results.py "${ARCHIVE_NAME}/experiments/"
cp experiments/two_pass_extraction.py "${ARCHIVE_NAME}/experiments/"
cp experiments/remediation_analysis.py "${ARCHIVE_NAME}/experiments/"
cp experiments/scenarios/signal_detection.py "${ARCHIVE_NAME}/experiments/scenarios/"
cp experiments/results/DATA_MANIFEST.md "${ARCHIVE_NAME}/experiments/results/"
cp experiments/results/primary/*.json "${ARCHIVE_NAME}/experiments/results/primary/"

# Copy root files
echo "Copying root files..."
cp requirements.txt "${ARCHIVE_NAME}/"
cp LICENSE "${ARCHIVE_NAME}/"
cp CITATION.cff "${ARCHIVE_NAME}/"
cp .zenodo.json "${ARCHIVE_NAME}/"
cp README_ZENODO.md "${ARCHIVE_NAME}/README.md"

# Create archive
echo "Creating ZIP archive..."
zip -r "${ARCHIVE_NAME}.zip" "${ARCHIVE_NAME}"

echo ""
echo "Archive created: ${ARCHIVE_NAME}.zip"
echo "Contents:"
unzip -l "${ARCHIVE_NAME}.zip" | head -30
echo "..."

# Cleanup
rm -rf "${ARCHIVE_NAME}"

echo ""
echo "Done! Upload ${ARCHIVE_NAME}.zip to Zenodo."
