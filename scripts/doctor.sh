#!/usr/bin/env bash
# =============================================================================
# MATS-BACKTRACKING Doctor Script
# Diagnoses the environment and verifies GPU + PyTorch setup
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║          MATS-BACKTRACKING Environment Diagnostics               ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Determine project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Source environment if not already done
if [ -z "$HF_HOME" ]; then
    source "$PROJECT_ROOT/scripts/activate_env.sh" 2>/dev/null || true
fi

# -----------------------------------------------------------------------------
# System Info
# -----------------------------------------------------------------------------
echo -e "${BLUE}[System]${NC}"
echo "  Hostname:     $(hostname)"
echo "  Date:         $(date)"
echo "  Kernel:       $(uname -r)"
echo "  OS:           $(lsb_release -d 2>/dev/null | cut -f2 || cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
echo ""

# -----------------------------------------------------------------------------
# Python
# -----------------------------------------------------------------------------
echo -e "${BLUE}[Python]${NC}"
if command -v python &> /dev/null; then
    echo -e "  Python:       ${GREEN}$(python --version 2>&1)${NC}"
    echo "  Path:         $(which python)"
else
    echo -e "  Python:       ${RED}NOT FOUND${NC}"
fi
echo ""

# -----------------------------------------------------------------------------
# uv
# -----------------------------------------------------------------------------
echo -e "${BLUE}[Package Manager]${NC}"
if command -v uv &> /dev/null; then
    echo -e "  uv:           ${GREEN}$(uv --version)${NC}"
else
    echo -e "  uv:           ${YELLOW}NOT FOUND (install with: curl -LsSf https://astral.sh/uv/install.sh | sh)${NC}"
fi
echo ""

# -----------------------------------------------------------------------------
# Virtual Environment
# -----------------------------------------------------------------------------
echo -e "${BLUE}[Virtual Environment]${NC}"
if [ -d "$PROJECT_ROOT/.venv" ]; then
    echo -e "  .venv:        ${GREEN}EXISTS${NC}"
    if [ -n "$VIRTUAL_ENV" ]; then
        echo -e "  Activated:    ${GREEN}YES${NC}"
    else
        echo -e "  Activated:    ${YELLOW}NO (run: source scripts/activate_env.sh)${NC}"
    fi
else
    echo -e "  .venv:        ${RED}NOT FOUND (run: uv sync)${NC}"
fi
echo ""

# -----------------------------------------------------------------------------
# GPU / CUDA
# -----------------------------------------------------------------------------
echo -e "${BLUE}[GPU / CUDA]${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' 2>/dev/null)
    
    echo -e "  nvidia-smi:   ${GREEN}AVAILABLE${NC}"
    echo "  GPU:          $GPU_NAME"
    echo "  VRAM:         $GPU_MEMORY"
    echo "  Driver:       $DRIVER_VERSION"
    echo "  CUDA:         $CUDA_VERSION"
else
    echo -e "  nvidia-smi:   ${RED}NOT FOUND${NC}"
fi
echo ""

# -----------------------------------------------------------------------------
# PyTorch CUDA Test
# -----------------------------------------------------------------------------
echo -e "${BLUE}[PyTorch CUDA]${NC}"
python << 'PYEOF'
import sys
try:
    import torch
    print(f"  torch:        \033[0;32m{torch.__version__}\033[0m")
    
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"  CUDA:         \033[0;32mAVAILABLE\033[0m")
        print(f"  Device count: {torch.cuda.device_count()}")
        print(f"  Device name:  {torch.cuda.get_device_name(0)}")
        
        # Quick tensor test
        x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        y = x * 2
        print(f"  Tensor test:  \033[0;32mPASS\033[0m (computed {x.tolist()} * 2 = {y.tolist()} on GPU)")
    else:
        print(f"  CUDA:         \033[0;31mNOT AVAILABLE\033[0m")
        print(f"  torch.cuda.is_available() = False")
        if hasattr(torch.version, 'cuda') and torch.version.cuda:
            print(f"  torch built with CUDA: {torch.version.cuda}")
        else:
            print(f"  torch built with CUDA: \033[0;31mNO (CPU-only build)\033[0m")
except ImportError as e:
    print(f"  torch:        \033[0;31mNOT INSTALLED\033[0m ({e})")
    sys.exit(1)
PYEOF
echo ""

# -----------------------------------------------------------------------------
# Key Libraries
# -----------------------------------------------------------------------------
echo -e "${BLUE}[Key Libraries]${NC}"
python << 'PYEOF'
import importlib
libs = [
    ("transformers", "transformers"),
    ("datasets", "datasets"),
    ("einops", "einops"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("wandb", "wandb"),
    ("tqdm", "tqdm"),
    ("rich", "rich"),
]

for display_name, import_name in libs:
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "installed")
        print(f"  {display_name:15} \033[0;32m{version}\033[0m")
    except ImportError:
        print(f"  {display_name:15} \033[0;33mNOT INSTALLED\033[0m")
PYEOF
echo ""

# -----------------------------------------------------------------------------
# Cache Directories
# -----------------------------------------------------------------------------
echo -e "${BLUE}[Cache Configuration]${NC}"
echo "  HF_HOME:              ${HF_HOME:-not set}"
echo "  TORCH_HOME:           ${TORCH_HOME:-not set}"
echo "  WANDB_DIR:            ${WANDB_DIR:-not set}"
echo ""

# -----------------------------------------------------------------------------
# Git Status
# -----------------------------------------------------------------------------
echo -e "${BLUE}[Git]${NC}"
if [ -d "$PROJECT_ROOT/.git" ]; then
    echo -e "  Repository:   ${GREEN}INITIALIZED${NC}"
    echo "  Branch:       $(git -C "$PROJECT_ROOT" branch --show-current 2>/dev/null || echo 'unknown')"
    REMOTE=$(git -C "$PROJECT_ROOT" remote get-url origin 2>/dev/null || echo "not set")
    echo "  Remote:       $REMOTE"
    DIRTY=$(git -C "$PROJECT_ROOT" status --porcelain 2>/dev/null | wc -l)
    if [ "$DIRTY" -gt 0 ]; then
        echo -e "  Status:       ${YELLOW}$DIRTY uncommitted changes${NC}"
    else
        echo -e "  Status:       ${GREEN}clean${NC}"
    fi
else
    echo -e "  Repository:   ${RED}NOT INITIALIZED${NC}"
fi
echo ""

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                        Diagnostics Complete                       ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

