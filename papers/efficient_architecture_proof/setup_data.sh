#!/usr/bin/env bash
# Setup data for the COCONUT study.
#
# This script copies ProsQA data from Meta's COCONUT submodule into
# code/data/ where training configs and experiment scripts expect it,
# then generates the OOD test sets.
#
# Prerequisites:
#   - Git submodules must be initialized:
#       git submodule update --init --recursive
#   - Python environment with dependencies installed:
#       pip install -r requirements.txt
#
# Usage:
#   cd papers/efficient_architecture_proof
#   bash setup_data.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="${SCRIPT_DIR}/code"
DATA_DIR="${CODE_DIR}/data"
REF_DATA="${SCRIPT_DIR}/reference_repos/coconut/data"

echo "=== COCONUT Study: Data Setup ==="
echo ""

# Step 1: Check that the submodule is initialized
if [ ! -f "${REF_DATA}/prosqa_train.json" ]; then
    echo "ERROR: Meta's COCONUT submodule data not found at:"
    echo "  ${REF_DATA}"
    echo ""
    echo "Initialize the submodule first:"
    echo "  git submodule update --init --recursive"
    exit 1
fi

# Step 2: Copy ProsQA data to code/data/
echo "Step 1: Copying ProsQA data to code/data/..."
mkdir -p "${DATA_DIR}"

for f in prosqa_train.json prosqa_valid.json prosqa_test.json; do
    if [ -f "${DATA_DIR}/${f}" ]; then
        echo "  ${f} already exists, skipping"
    else
        cp "${REF_DATA}/${f}" "${DATA_DIR}/${f}"
        echo "  Copied ${f}"
    fi
done

# Step 3: Verify data
echo ""
echo "Step 2: Verifying data files..."
for f in prosqa_train.json prosqa_valid.json prosqa_test.json; do
    size=$(du -h "${DATA_DIR}/${f}" | cut -f1)
    echo "  ${f}: ${size}"
done

# Step 4: Generate OOD test sets
echo ""
echo "Step 3: Generating OOD test sets..."
cd "${CODE_DIR}"

if [ -f data/ood_7hop.json ] && [ -f data/ood_8hop.json ] && \
   [ -f data/ood_dag.json ] && [ -f data/ood_dense.json ]; then
    echo "  OOD test sets already exist, skipping"
else
    python generate_ood_data.py
    echo "  Generated 4 OOD test sets"
fi

echo ""
echo "=== Data setup complete ==="
echo ""
echo "Files in code/data/:"
ls -lh "${DATA_DIR}"/*.json 2>/dev/null || echo "  (none)"
