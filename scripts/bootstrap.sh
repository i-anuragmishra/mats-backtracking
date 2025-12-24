#!/usr/bin/env bash
# =============================================================================
# MATS-BACKTRACKING Bootstrap Script
# Run this after git clone to set up everything in one command
# 
# Usage:
#   git clone https://github.com/i-anuragmishra/mats-backtracking.git
#   cd mats-backtracking
#   bash scripts/bootstrap.sh
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           MATS-BACKTRACKING Bootstrap Setup                      ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Determine project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}[1/6]${NC} Checking system requirements..."
echo "  Python: $(python3 --version 2>&1 || echo 'NOT FOUND')"
echo "  Git: $(git --version 2>&1 || echo 'NOT FOUND')"
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo -e "  GPU: ${GREEN}$GPU_NAME${NC}"
else
    echo -e "  GPU: ${YELLOW}No GPU detected (CPU-only mode)${NC}"
fi
echo ""

# -----------------------------------------------------------------------------
# Step 1: Install uv if not present
# -----------------------------------------------------------------------------
echo -e "${BLUE}[2/6]${NC} Checking for uv package manager..."
if command -v uv &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} uv already installed: $(uv --version)"
else
    echo "  Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Source uv
    if [ -f "$HOME/.local/bin/env" ]; then
        source "$HOME/.local/bin/env"
    elif [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi
    
    # Add to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
    
    echo -e "  ${GREEN}✓${NC} uv installed: $(uv --version)"
fi
echo ""

# -----------------------------------------------------------------------------
# Step 2: Create .env from template if it doesn't exist
# -----------------------------------------------------------------------------
echo -e "${BLUE}[3/6]${NC} Setting up environment file..."
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo -e "  ${GREEN}✓${NC} .env already exists"
else
    if [ -f "$PROJECT_ROOT/.env.example" ]; then
        cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
        echo -e "  ${GREEN}✓${NC} Created .env from template"
        echo -e "  ${YELLOW}⚠${NC} Remember to add your tokens to .env:"
        echo "      - GH_TOKEN (GitHub Personal Access Token)"
        echo "      - HF_TOKEN (HuggingFace)"
        echo "      - WANDB_API_KEY (Weights & Biases)"
    else
        echo -e "  ${YELLOW}⚠${NC} No .env.example found, creating minimal .env"
        cat > "$PROJECT_ROOT/.env" << 'EOF'
# Add your tokens here
GH_TOKEN=
HF_TOKEN=
WANDB_API_KEY=
WANDB_PROJECT=mats-backtracking
WANDB_ENTITY=i-anuragmishra
EOF
    fi
fi
echo ""

# -----------------------------------------------------------------------------
# Step 3: Create cache directories
# -----------------------------------------------------------------------------
echo -e "${BLUE}[4/6]${NC} Creating cache directories..."
mkdir -p "$PROJECT_ROOT/.cache/huggingface"
mkdir -p "$PROJECT_ROOT/.cache/torch"
mkdir -p "$PROJECT_ROOT/.cache/wandb"
mkdir -p "$PROJECT_ROOT/.cache/pip"
mkdir -p "$PROJECT_ROOT/.cache/jupyter"
mkdir -p "$PROJECT_ROOT/wandb"
echo -e "  ${GREEN}✓${NC} Cache directories created in .cache/"
echo ""

# -----------------------------------------------------------------------------
# Step 4: Set environment variables and install dependencies
# -----------------------------------------------------------------------------
echo -e "${BLUE}[5/6]${NC} Installing Python dependencies..."

# Set cache environment variables
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export WANDB_DIR="$PROJECT_ROOT/wandb"
export WANDB_CACHE_DIR="$PROJECT_ROOT/.cache/wandb"
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"

# Install dependencies
uv sync
echo -e "  ${GREEN}✓${NC} Dependencies installed"
echo ""

# -----------------------------------------------------------------------------
# Step 5: Verify installation
# -----------------------------------------------------------------------------
echo -e "${BLUE}[6/6]${NC} Verifying installation..."

# Quick Python check
source "$PROJECT_ROOT/.venv/bin/activate"

python3 << 'PYEOF'
import sys
try:
    import torch
    cuda_status = "✓ CUDA available" if torch.cuda.is_available() else "○ CPU only"
    print(f"  PyTorch {torch.__version__}: {cuda_status}")
except ImportError:
    print("  ✗ PyTorch not installed")
    sys.exit(1)

try:
    import transformers
    print(f"  Transformers {transformers.__version__}: ✓")
except ImportError:
    print("  ✗ Transformers not installed")
PYEOF

echo ""

# -----------------------------------------------------------------------------
# Done!
# -----------------------------------------------------------------------------
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Bootstrap Complete! 🎉                        ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Next steps:"
echo ""
echo "  1. Add your tokens to .env:"
echo "     ${BLUE}nano .env${NC}"
echo ""
echo "  2. Activate the environment:"
echo "     ${BLUE}source scripts/activate_env.sh${NC}"
echo ""
echo "  3. Verify everything works:"
echo "     ${BLUE}make doctor${NC}"
echo ""
echo "  4. Before shutting down, save your work:"
echo "     ${BLUE}make backup${NC}"
echo ""


