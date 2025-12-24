#!/usr/bin/env bash
# =============================================================================
# MATS-BACKTRACKING Snapshot Run Script
# Creates a timestamped snapshot of the current run state
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Determine project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Source environment
source "$PROJECT_ROOT/scripts/activate_env.sh" 2>/dev/null || true

# Create timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="$PROJECT_ROOT/runs/$TIMESTAMP"

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              Creating Run Snapshot: $TIMESTAMP              ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Create run directory
mkdir -p "$RUN_DIR"
echo -e "${GREEN}✓${NC} Created run directory: $RUN_DIR"

# -----------------------------------------------------------------------------
# Write metadata
# -----------------------------------------------------------------------------
METADATA_FILE="$RUN_DIR/metadata.txt"

cat > "$METADATA_FILE" << EOF
=============================================================================
MATS-BACKTRACKING RUN SNAPSHOT
=============================================================================

Timestamp:      $TIMESTAMP
Date:           $(date)
Hostname:       $(hostname)
User:           $(whoami)

=============================================================================
GIT STATUS
=============================================================================
EOF

if [ -d "$PROJECT_ROOT/.git" ]; then
    echo "Branch:         $(git -C "$PROJECT_ROOT" branch --show-current 2>/dev/null || echo 'detached')" >> "$METADATA_FILE"
    echo "Commit SHA:     $(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo 'unknown')" >> "$METADATA_FILE"
    echo "Commit Date:    $(git -C "$PROJECT_ROOT" log -1 --format=%ci 2>/dev/null || echo 'unknown')" >> "$METADATA_FILE"
    echo "Commit Message: $(git -C "$PROJECT_ROOT" log -1 --format=%s 2>/dev/null || echo 'unknown')" >> "$METADATA_FILE"
    echo "" >> "$METADATA_FILE"
    echo "Uncommitted changes:" >> "$METADATA_FILE"
    git -C "$PROJECT_ROOT" status --short >> "$METADATA_FILE" 2>/dev/null || echo "  (none)" >> "$METADATA_FILE"
else
    echo "Git repository not initialized" >> "$METADATA_FILE"
fi

cat >> "$METADATA_FILE" << EOF

=============================================================================
SYSTEM INFO
=============================================================================
EOF
uname -a >> "$METADATA_FILE"
echo "" >> "$METADATA_FILE"

cat >> "$METADATA_FILE" << EOF

=============================================================================
GPU INFO
=============================================================================
EOF
nvidia-smi >> "$METADATA_FILE" 2>/dev/null || echo "nvidia-smi not available" >> "$METADATA_FILE"

cat >> "$METADATA_FILE" << EOF

=============================================================================
PYTHON ENVIRONMENT
=============================================================================
EOF
echo "Python: $(python --version 2>&1)" >> "$METADATA_FILE"
echo "" >> "$METADATA_FILE"
echo "Installed packages:" >> "$METADATA_FILE"
if command -v uv &> /dev/null; then
    uv pip list >> "$METADATA_FILE" 2>/dev/null || pip list >> "$METADATA_FILE" 2>/dev/null
else
    pip list >> "$METADATA_FILE" 2>/dev/null || echo "  pip not available" >> "$METADATA_FILE"
fi

echo -e "${GREEN}✓${NC} Wrote metadata to metadata.txt"

# -----------------------------------------------------------------------------
# Copy uv.lock if it exists
# -----------------------------------------------------------------------------
if [ -f "$PROJECT_ROOT/uv.lock" ]; then
    cp "$PROJECT_ROOT/uv.lock" "$RUN_DIR/uv.lock"
    echo -e "${GREEN}✓${NC} Copied uv.lock"
fi

# -----------------------------------------------------------------------------
# Copy figures, results, reports (small artifacts)
# -----------------------------------------------------------------------------
copy_if_nonempty() {
    local src="$1"
    local dest="$2"
    local name="$3"
    
    if [ -d "$src" ] && [ "$(ls -A "$src" 2>/dev/null)" ]; then
        mkdir -p "$dest"
        cp -r "$src"/* "$dest"/ 2>/dev/null || true
        echo -e "${GREEN}✓${NC} Copied $name"
    else
        echo -e "${YELLOW}○${NC} Skipped $name (empty or missing)"
    fi
}

copy_if_nonempty "$PROJECT_ROOT/figures" "$RUN_DIR/figures" "figures/"
copy_if_nonempty "$PROJECT_ROOT/results" "$RUN_DIR/results" "results/"
copy_if_nonempty "$PROJECT_ROOT/reports" "$RUN_DIR/reports" "reports/"

# -----------------------------------------------------------------------------
# Copy configs
# -----------------------------------------------------------------------------
if [ -d "$PROJECT_ROOT/configs" ] && [ "$(ls -A "$PROJECT_ROOT/configs" 2>/dev/null)" ]; then
    mkdir -p "$RUN_DIR/configs"
    cp -r "$PROJECT_ROOT/configs"/* "$RUN_DIR/configs"/ 2>/dev/null || true
    echo -e "${GREEN}✓${NC} Copied configs/"
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo -e "${BLUE}Snapshot created at:${NC}"
echo "  $RUN_DIR"
echo ""

# Show size
SIZE=$(du -sh "$RUN_DIR" 2>/dev/null | cut -f1)
echo -e "${BLUE}Snapshot size:${NC} $SIZE"
echo ""

# List contents
echo -e "${BLUE}Contents:${NC}"
ls -la "$RUN_DIR"
echo ""

echo -e "${GREEN}✓${NC} Snapshot complete!"
echo ""
echo "Next steps:"
echo "  - Review snapshot at: $RUN_DIR"
echo "  - Run 'make push' to commit and push to remote"
echo ""


