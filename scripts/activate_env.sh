#!/usr/bin/env bash
# =============================================================================
# MATS-BACKTRACKING Environment Activation Script
# Source this file to activate the environment with all caches configured
# Usage: source scripts/activate_env.sh
# =============================================================================

# Determine the project root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🔧 Activating mats-backtracking environment..."
echo "   Project root: $PROJECT_ROOT"

# -----------------------------------------------------------------------------
# Cache Configuration - All caches inside project directory
# -----------------------------------------------------------------------------

# HuggingFace cache
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_ROOT/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="$PROJECT_ROOT/.cache/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="$PROJECT_ROOT/.cache/huggingface/hub"

# PyTorch cache
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"

# Weights & Biases
export WANDB_DIR="$PROJECT_ROOT/wandb"
export WANDB_CACHE_DIR="$PROJECT_ROOT/.cache/wandb"

# Jupyter
export JUPYTER_DATA_DIR="$PROJECT_ROOT/.cache/jupyter"

# Pip cache (optional, useful for offline work)
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"

# -----------------------------------------------------------------------------
# Load secrets from .env if it exists
# -----------------------------------------------------------------------------
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "   Loading secrets from .env"
    set -a  # automatically export all variables
    source "$PROJECT_ROOT/.env"
    set +a
else
    echo "   ⚠️  No .env file found. Copy .env.example to .env and add your secrets."
fi

# -----------------------------------------------------------------------------
# Activate Python virtual environment
# -----------------------------------------------------------------------------
if [ -d "$PROJECT_ROOT/.venv" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
    echo "   Python venv activated: $(which python)"
else
    echo "   ⚠️  No .venv found. Run 'uv sync' to create it."
fi

# -----------------------------------------------------------------------------
# Add src to PYTHONPATH for imports
# -----------------------------------------------------------------------------
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# -----------------------------------------------------------------------------
# Add uv to PATH if installed in user directory
# -----------------------------------------------------------------------------
if [ -d "$HOME/.local/bin" ]; then
    export PATH="$HOME/.local/bin:$PATH"
fi

# -----------------------------------------------------------------------------
# Convenience aliases
# -----------------------------------------------------------------------------
alias cdp="cd $PROJECT_ROOT"
alias doctor="bash $PROJECT_ROOT/scripts/doctor.sh"
alias snapshot="bash $PROJECT_ROOT/scripts/snapshot_run.sh"
alias pushall="bash $PROJECT_ROOT/scripts/push_all.sh"

echo "✅ Environment activated! Cache dirs set to .cache/"
echo ""
echo "   Useful commands:"
echo "     make doctor    - Check GPU and environment"
echo "     make smoke     - Run smoke test"
echo "     make snapshot  - Create run snapshot"
echo "     make push      - Commit and push"
echo ""


