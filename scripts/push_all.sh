#!/usr/bin/env bash
# =============================================================================
# MATS-BACKTRACKING Push All Script
# Commits changes and pushes to remote origin
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Determine project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Load .env for GH_TOKEN if available
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    MATS-BACKTRACKING Push                        ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# -----------------------------------------------------------------------------
# Check git is initialized
# -----------------------------------------------------------------------------
if [ ! -d ".git" ]; then
    echo -e "${RED}✗${NC} Git repository not initialized!"
    echo "  Run 'git init' first."
    exit 1
fi

# -----------------------------------------------------------------------------
# Check for remote
# -----------------------------------------------------------------------------
REMOTE_URL=$(git remote get-url origin 2>/dev/null || echo "")
if [ -z "$REMOTE_URL" ]; then
    echo -e "${YELLOW}⚠${NC} No remote 'origin' configured."
    echo ""
    echo "To set up a remote, run:"
    echo "  git remote add origin <your-repo-url>"
    echo ""
    echo "Example:"
    echo "  git remote add origin git@github.com:username/mats-backtracking.git"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓${NC} Remote: $REMOTE_URL"

# -----------------------------------------------------------------------------
# Optional: Run linting
# -----------------------------------------------------------------------------
if [ "${SKIP_LINT:-}" != "1" ]; then
    echo ""
    echo -e "${BLUE}Running lint check...${NC}"
    if command -v uv &> /dev/null && [ -f ".venv/bin/ruff" ]; then
        source scripts/activate_env.sh 2>/dev/null || true
        if uv run ruff check src/ scripts/ 2>/dev/null; then
            echo -e "${GREEN}✓${NC} Lint passed"
        else
            echo -e "${YELLOW}⚠${NC} Lint warnings (continuing anyway)"
        fi
    else
        echo -e "${YELLOW}○${NC} Skipping lint (ruff not installed)"
    fi
fi

# -----------------------------------------------------------------------------
# Show status
# -----------------------------------------------------------------------------
echo ""
echo -e "${BLUE}Current status:${NC}"
git status --short

CHANGES=$(git status --porcelain | wc -l)
if [ "$CHANGES" -eq 0 ]; then
    echo ""
    echo -e "${YELLOW}○${NC} No changes to commit."
    echo ""
    
    # Still offer to push in case there are unpushed commits
    UNPUSHED=$(git log origin/$(git branch --show-current)..HEAD 2>/dev/null | wc -l || echo "0")
    if [ "$UNPUSHED" -gt 0 ]; then
        echo "But there are unpushed commits. Pushing..."
        git push origin "$(git branch --show-current)"
        echo -e "${GREEN}✓${NC} Pushed to origin!"
    fi
    exit 0
fi

# -----------------------------------------------------------------------------
# Stage all changes
# -----------------------------------------------------------------------------
echo ""
echo -e "${BLUE}Staging changes...${NC}"
git add -A
echo -e "${GREEN}✓${NC} All changes staged"

# -----------------------------------------------------------------------------
# Create commit message
# -----------------------------------------------------------------------------
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
HOSTNAME=$(hostname)
BRANCH=$(git branch --show-current)

# Default commit message
if [ -n "$1" ]; then
    COMMIT_MSG="$1"
else
    COMMIT_MSG="Auto-commit: $TIMESTAMP [$HOSTNAME]"
fi

echo ""
echo -e "${BLUE}Creating commit...${NC}"
echo "  Message: $COMMIT_MSG"
git commit -m "$COMMIT_MSG"
echo -e "${GREEN}✓${NC} Committed"

# -----------------------------------------------------------------------------
# Push to remote
# -----------------------------------------------------------------------------
echo ""
echo -e "${BLUE}Pushing to origin/$BRANCH...${NC}"

# If GH_TOKEN is set, use it for authentication (useful for ephemeral instances)
if [ -n "$GH_TOKEN" ]; then
    # Temporarily set remote with token for push
    REPO_URL=$(git remote get-url origin | sed 's|https://[^@]*@|https://|' | sed 's|https://|https://'"$GH_TOKEN"'@|')
    git push "$REPO_URL" "$BRANCH" 2>/dev/null
else
    git push origin "$BRANCH"
fi
echo -e "${GREEN}✓${NC} Pushed to origin!"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                         Push Complete!                           ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Commit:  $(git rev-parse --short HEAD)"
echo "  Branch:  $BRANCH"
echo "  Remote:  $REMOTE_URL"
echo "  Time:    $TIMESTAMP"
echo ""

