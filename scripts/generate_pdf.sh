#!/bin/bash
# Generate PDF from paper markdown using pandoc
#
# Usage: bash scripts/generate_pdf.sh
#
# Environment variables for deterministic builds:
#   SOURCE_DATE_EPOCH=0   - Use epoch for timestamps
#   TZ=UTC                - Use UTC timezone

set -e

INPUT="paper/generated/FORMAT_FRICTION.md"
OUTPUT="paper/output/FORMAT_FRICTION.pdf"

# Fallback to existing paper if generated doesn't exist
if [ ! -f "$INPUT" ]; then
    if [ -f "paper/FORMAT_FRICTION.md" ]; then
        INPUT="paper/FORMAT_FRICTION.md"
        echo "Using existing paper: $INPUT"
    else
        echo "ERROR: No paper markdown found"
        echo "Expected: $INPUT or paper/FORMAT_FRICTION.md"
        exit 1
    fi
fi

# Create output directory
mkdir -p paper/output

# Check for pandoc
if ! command -v pandoc &> /dev/null; then
    echo "ERROR: pandoc not found"
    echo "Install via: brew install pandoc (macOS) or apt install pandoc (Linux)"
    exit 1
fi

# Check for LaTeX (required for PDF output)
if ! command -v pdflatex &> /dev/null && ! command -v xelatex &> /dev/null; then
    echo "WARNING: LaTeX not found. Attempting HTML output instead."

    pandoc "$INPUT" \
        -o "paper/output/FORMAT_FRICTION.html" \
        --standalone \
        --toc \
        --metadata title="Format Friction in LLM Tool Calling"

    echo "Generated: paper/output/FORMAT_FRICTION.html"
    exit 0
fi

# Generate PDF with pandoc
echo "Generating PDF from $INPUT..."

pandoc "$INPUT" \
    -o "$OUTPUT" \
    --pdf-engine=pdflatex \
    --toc \
    --number-sections \
    --variable geometry:margin=1in \
    --variable fontsize=11pt \
    --metadata title="Format Friction in LLM Tool Calling" \
    --metadata date="$(date +%Y-%m-%d)" \
    2>&1 || {
        echo "WARNING: pdflatex failed, trying xelatex..."
        pandoc "$INPUT" \
            -o "$OUTPUT" \
            --pdf-engine=xelatex \
            --toc \
            --number-sections \
            --variable geometry:margin=1in \
            --variable fontsize=11pt \
            --metadata title="Format Friction in LLM Tool Calling" \
            --metadata date="$(date +%Y-%m-%d)"
    }

# Verify output
if [ -f "$OUTPUT" ]; then
    SIZE=$(ls -la "$OUTPUT" | awk '{print $5}')
    echo "Generated: $OUTPUT ($SIZE bytes)"

    if [ "$SIZE" -lt 50000 ]; then
        echo "WARNING: PDF is smaller than expected (< 50KB)"
    fi
else
    echo "ERROR: PDF generation failed"
    exit 1
fi
