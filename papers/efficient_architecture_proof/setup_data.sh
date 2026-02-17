#!/usr/bin/env bash
# Setup data for the COCONUT study.
#
# ProsQA data files are tracked in code/data/ and available after cloning.
# This script is only needed if code/data/ is missing (e.g., fresh clone
# without LFS, or data files were deleted) — it copies from the COCONUT
# git submodule. It also regenerates OOD test sets if missing.
#
# Prerequisites:
#   - Git submodules initialized (for ProsQA source):
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

# Step 1: Copy ProsQA data to code/data/ if missing
mkdir -p "${DATA_DIR}"
NEED_COPY=false

for f in prosqa_train.json prosqa_valid.json prosqa_test.json; do
    if [ ! -f "${DATA_DIR}/${f}" ]; then
        NEED_COPY=true
        break
    fi
done

if [ "${NEED_COPY}" = true ]; then
    echo "Step 1: Copying ProsQA data to code/data/..."
    if [ ! -f "${REF_DATA}/prosqa_train.json" ]; then
        echo "ERROR: ProsQA data not found in ${REF_DATA}"
        echo ""
        echo "Initialize the submodule first:"
        echo "  git submodule update --init --recursive"
        exit 1
    fi
    for f in prosqa_train.json prosqa_valid.json prosqa_test.json; do
        if [ -f "${DATA_DIR}/${f}" ]; then
            echo "  ${f} already exists, skipping"
        else
            cp "${REF_DATA}/${f}" "${DATA_DIR}/${f}"
            echo "  Copied ${f} from reference_repos/coconut/data/"
        fi
    done
else
    echo "Step 1: ProsQA data already present in code/data/, skipping"
fi

# Step 2: Verify data
echo ""
echo "Step 2: Verifying data files..."
for f in prosqa_train.json prosqa_valid.json prosqa_test.json; do
    size=$(du -h "${DATA_DIR}/${f}" | cut -f1)
    echo "  ${f}: ${size}"
done

# Step 3: Generate OOD test sets if missing
echo ""
echo "Step 3: Checking OOD test sets..."
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
