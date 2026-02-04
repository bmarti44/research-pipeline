#!/bin/bash
# Setup environment for format friction experiment
#
# This script:
# 1. Installs Python dependencies from requirements.txt
# 2. Validates system tools (pandoc, sha256sum, git)
# 3. Validates CLI tools (claude, codex, gemini)
#
# Usage: bash scripts/setup_env.sh

set -e

echo "=== Format Friction Environment Setup ==="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track errors
ERRORS=0

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]); then
    echo -e "${RED}ERROR: Python 3.11+ required, found $PYTHON_VERSION${NC}"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}OK: Python $PYTHON_VERSION${NC}"
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
if [ -f requirements.txt ]; then
    pip install -r requirements.txt --quiet
    echo -e "${GREEN}OK: Python dependencies installed${NC}"
else
    echo -e "${YELLOW}WARNING: requirements.txt not found${NC}"
fi

# Check system tools
echo ""
echo "Checking system tools..."

for tool in pandoc git; do
    if command -v "$tool" &> /dev/null; then
        VERSION=$($tool --version 2>&1 | head -1)
        echo -e "${GREEN}OK: $tool - $VERSION${NC}"
    else
        echo -e "${RED}ERROR: $tool not found${NC}"
        ERRORS=$((ERRORS + 1))
    fi
done

# Check sha256sum (different on macOS vs Linux)
if command -v sha256sum &> /dev/null; then
    echo -e "${GREEN}OK: sha256sum available${NC}"
elif command -v shasum &> /dev/null; then
    echo -e "${GREEN}OK: shasum available (macOS)${NC}"
else
    echo -e "${RED}ERROR: sha256sum/shasum not found${NC}"
    ERRORS=$((ERRORS + 1))
fi

# Check CLI tools (these may not be available but we report status)
echo ""
echo "Checking model CLIs..."

CLAUDE_AVAILABLE=false
CODEX_AVAILABLE=false
GEMINI_AVAILABLE=false

if command -v claude &> /dev/null; then
    echo -e "${GREEN}OK: claude CLI available${NC}"
    CLAUDE_AVAILABLE=true
else
    echo -e "${YELLOW}INFO: claude CLI not found (required for primary experiment)${NC}"
fi

if command -v codex &> /dev/null; then
    echo -e "${GREEN}OK: codex CLI available${NC}"
    CODEX_AVAILABLE=true
else
    echo -e "${YELLOW}INFO: codex CLI not found (optional for multi-model validation)${NC}"
fi

if command -v gemini &> /dev/null; then
    echo -e "${GREEN}OK: gemini CLI available${NC}"
    GEMINI_AVAILABLE=true
else
    echo -e "${YELLOW}INFO: gemini CLI not found (optional for multi-model validation)${NC}"
fi

# CRITICAL: Unset API keys to prevent SDK usage
echo ""
echo "Checking API key environment variables..."
KEYS_SET=false

if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo -e "${YELLOW}WARNING: ANTHROPIC_API_KEY is set - consider unsetting for subscription-based CLI usage${NC}"
    KEYS_SET=true
fi

if [ -n "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}WARNING: OPENAI_API_KEY is set - consider unsetting for subscription-based CLI usage${NC}"
    KEYS_SET=true
fi

if [ -n "$GOOGLE_API_KEY" ]; then
    echo -e "${YELLOW}WARNING: GOOGLE_API_KEY is set - consider unsetting for subscription-based CLI usage${NC}"
    KEYS_SET=true
fi

if [ "$KEYS_SET" = false ]; then
    echo -e "${GREEN}OK: No API keys set (using subscription-based CLIs)${NC}"
fi

# Summary
echo ""
echo "=== Setup Summary ==="
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}Environment setup complete. No critical errors.${NC}"
else
    echo -e "${RED}Environment setup found $ERRORS critical error(s).${NC}"
    exit 1
fi

echo ""
echo "CLI availability:"
echo "  claude: $CLAUDE_AVAILABLE"
echo "  codex:  $CODEX_AVAILABLE"
echo "  gemini: $GEMINI_AVAILABLE"

# Write CLI availability to a temp file for pipeline
cat > /tmp/cli_availability.json << EOF
{
    "claude": $CLAUDE_AVAILABLE,
    "codex": $CODEX_AVAILABLE,
    "gemini": $GEMINI_AVAILABLE
}
EOF

echo ""
echo "Ready to run: ./scripts/run_pipeline.sh"
