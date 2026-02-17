#!/usr/bin/env bash
# Option B: Atomic M-number renumbering for the manuscript.
#
# Renames M3 -> M2, M5 -> M3, M6 -> M4 across all manuscript files.
# Uses intermediate placeholders (__M2__, __M3__, __M4__) to avoid
# collision: M3->M2 must not then be caught by M5->M3's pass.
#
# Run from the paper root: papers/efficient_architecture_proof/
#
# DRY RUN (preview only):
#   bash renumber_manuscript.sh --dry-run
#
# EXECUTE:
#   bash renumber_manuscript.sh
#
# After running, manually verify:
#   grep -rn 'M[356]' manuscript/ --include='*.md'
#   grep -rn '__M[234]__' manuscript/ --include='*.md'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MANUSCRIPT_DIR="$SCRIPT_DIR/manuscript"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# Collect only files that exist
TARGETS=()
for f in "$MANUSCRIPT_DIR/manuscript.md" \
         "$MANUSCRIPT_DIR/metadata.yaml"; do
    [[ -f "$f" ]] && TARGETS+=("$f")
done
# Add any section files
if [[ -d "$MANUSCRIPT_DIR/sections" ]]; then
    for f in "$MANUSCRIPT_DIR/sections/"*.md; do
        [[ -f "$f" ]] && TARGETS+=("$f")
    done
fi

if [[ ${#TARGETS[@]} -eq 0 ]]; then
    echo "ERROR: No manuscript files found in $MANUSCRIPT_DIR"
    exit 1
fi

echo "Option B: Renumber M3->M2, M5->M3, M6->M4"
echo "Files to process: ${#TARGETS[@]}"
for f in "${TARGETS[@]}"; do
    echo "  $(basename "$f")"
done
echo ""

# Count occurrences before (grep -w for word boundary, works on macOS)
echo "Before:"
for pat in 'M3' 'M5' 'M6'; do
    count=0
    for f in "${TARGETS[@]}"; do
        c=$(grep -ow "${pat}" "$f" 2>/dev/null | wc -l | tr -d ' ')
        count=$((count + c))
    done
    echo "  $pat: $count occurrences"
done
echo ""

if $DRY_RUN; then
    echo "[DRY RUN] Would apply the following perl replacements:"
    echo "  Pass 1: M3 -> __M2__, M5 -> __M3__, M6 -> __M4__"
    echo "  Pass 2: __M2__ -> M2, __M3__ -> M3, __M4__ -> M4"
    echo ""
    echo "Preview of lines that would change (M3/M5/M6 references):"
    for f in "${TARGETS[@]}"; do
        grep -nw 'M[356]' "$f" 2>/dev/null | head -20 | while IFS= read -r line; do
            echo "  $(basename "$f"):$line"
        done
    done
    exit 0
fi

# === Pass 1: Replace old names with unique placeholders ===
# Using perl -pi -e because macOS BSD sed does not support \b word boundaries
for f in "${TARGETS[@]}"; do
    # M3 (COCONUT) -> __M2__ (new M2)
    perl -pi -e 's/\bM3\b/__M2__/g' "$f"
    # M5 (Pause) -> __M3__ (new M3)
    perl -pi -e 's/\bM5\b/__M3__/g' "$f"
    # M6 (Pause-Multipass) -> __M4__ (new M4)
    perl -pi -e 's/\bM6\b/__M4__/g' "$f"
done

# === Pass 2: Replace placeholders with final names ===
# Placeholders contain no word-boundary issues, but use perl for consistency
for f in "${TARGETS[@]}"; do
    perl -pi -e 's/__M2__/M2/g' "$f"
    perl -pi -e 's/__M3__/M3/g' "$f"
    perl -pi -e 's/__M4__/M4/g' "$f"
done

# === Verification ===
echo "After:"
errors=0
for pat in 'M3' 'M5' 'M6'; do
    count=0
    for f in "${TARGETS[@]}"; do
        c=$(grep -ow "${pat}" "$f" 2>/dev/null | wc -l | tr -d ' ')
        count=$((count + c))
    done
    # M3 should now have NEW M3 (formerly M5) occurrences
    if [[ "$pat" == "M5" || "$pat" == "M6" ]]; then
        if [[ $count -gt 0 ]]; then
            echo "  ERROR: $pat still has $count occurrences (should be 0)"
            errors=$((errors + 1))
        else
            echo "  $pat: 0 (clean)"
        fi
    else
        echo "  $pat: $count (these are new M3 = formerly M5)"
    fi
done

# Check no placeholders remain
for f in "${TARGETS[@]}"; do
    if grep -q '__M[234]__' "$f"; then
        echo "  ERROR: Placeholders remain in $(basename "$f")"
        errors=$((errors + 1))
    fi
done

if [[ $errors -gt 0 ]]; then
    echo ""
    echo "ERRORS FOUND. Check output above."
    exit 1
fi

echo ""
echo "Renumbering complete. Verify with:"
echo "  git diff manuscript/"
echo "  grep -rn 'M[1234]' manuscript/ --include='*.md' | head -30"
