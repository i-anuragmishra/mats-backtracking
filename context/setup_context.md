# Setup Context for AI Assistants

> **Purpose:** This document provides context for AI assistants working on this project after a fresh `git clone`. Read this first to understand the project structure and conventions.

---

## Project Overview

**Name:** mats-backtracking  
**Purpose:** Mechanistic interpretability research investigating backtracking and state transitions in language models  
**Owner:** i-anuragmishra (Anurag Mishra)  
**Repository:** https://github.com/i-anuragmishra/mats-backtracking (private)

---

## Quick Start (After Clone)

```bash
# 1. Clone and enter project
git clone https://github.com/i-anuragmishra/mats-backtracking.git
cd mats-backtracking

# 2. Run bootstrap (installs uv, creates venv, installs deps)
bash scripts/bootstrap.sh

# 3. Add tokens to .env (GH_TOKEN, HF_TOKEN, WANDB_API_KEY)
nano .env

# 4. Activate environment
source scripts/activate_env.sh

# 5. Verify setup
make doctor
```

---

## Project Structure

```
mats-backtracking/
├── src/backtracking/      # Main Python package - put research code here
├── scripts/               # Utility shell/python scripts
│   ├── bootstrap.sh       # One-command setup after git clone
│   ├── activate_env.sh    # Activates venv + sets cache env vars
│   ├── doctor.sh          # Diagnoses environment (GPU, packages, etc.)
│   ├── run_smoke_test.py  # Full verification (loads model on GPU)
│   ├── snapshot_run.sh    # Creates timestamped run snapshot
│   └── push_all.sh        # Commits and pushes to GitHub
├── notebooks/             # Jupyter notebooks for exploration
├── configs/               # YAML/JSON experiment configurations
├── context/               # AI context files (like this one)
├── data/
│   ├── raw/               # Original data (gitignored)
│   ├── interim/           # Intermediate processed data
│   └── processed/         # Final data ready for experiments
├── results/               # Experiment outputs (metrics, tables)
├── figures/               # Generated visualizations
├── reports/               # Writeups, analysis documents
├── logs/                  # Training/run logs (gitignored)
├── runs/                  # Timestamped run snapshots
├── scratch/               # Temporary work (gitignored)
├── .cache/                # All caches: HuggingFace, Torch, pip (gitignored)
├── wandb/                 # W&B local files (gitignored)
├── .venv/                 # Python virtual environment (gitignored)
├── .env                   # Secrets: tokens, API keys (gitignored, NEVER COMMIT)
├── .env.example           # Template showing required env vars
├── pyproject.toml         # Project config + dependencies
├── uv.lock                # Locked dependency versions
├── Makefile               # Common commands (run `make help`)
└── README.md              # User-facing documentation
```

---

## Key Conventions

### 1. Environment Activation

**Always activate before working:**
```bash
source scripts/activate_env.sh
```

This script:
- Sets `HF_HOME`, `TORCH_HOME`, `WANDB_DIR` to `.cache/` subdirectories
- Loads secrets from `.env`
- Activates the Python virtual environment
- Adds `src/` to `PYTHONPATH`

### 2. Caches Are Local

All caches live inside the project directory (not in `~/`):
- HuggingFace models: `.cache/huggingface/`
- PyTorch hub: `.cache/torch/`
- W&B: `.cache/wandb/` and `wandb/`

This ensures caches don't persist across ephemeral instances (which is intentional).

### 3. Secrets Management

- **`.env`** contains real tokens (GH_TOKEN, HF_TOKEN, WANDB_API_KEY)
- **`.env`** is gitignored and NEVER committed
- **`.env.example`** is a template showing what variables are needed
- The `push_all.sh` script reads `GH_TOKEN` from `.env` for authentication

### 4. Git Workflow

**Before shutting down an ephemeral instance:**
```bash
make backup  # Creates snapshot + commits + pushes
```

Or separately:
```bash
make snapshot  # Creates timestamped folder in runs/ with metadata
make push      # Commits all changes and pushes to GitHub
```

### 5. What's Tracked vs Ignored

**Tracked (committed to git):**
- All code (`src/`, `scripts/`)
- Notebooks, configs
- Small results, figures, reports
- Run metadata (`runs/*/metadata.txt`)
- `pyproject.toml`, `uv.lock`

**Ignored (NOT committed):**
- Model weights (`.pt`, `.bin`, `.safetensors`)
- Large numpy files (`.npy`, `.npz`)
- Caches (`.cache/`, `wandb/`)
- Raw data (`data/raw/`)
- Logs (`logs/`)
- Secrets (`.env`)
- Virtual environment (`.venv/`)

---

## Package Manager: uv

This project uses **uv** (not pip) for fast, reproducible dependency management.

```bash
# Install a new package
uv add package-name

# Install with extras
uv sync --all-extras

# Recreate environment from lockfile
uv sync
```

The `uv.lock` file ensures reproducible installs across machines.

---

## Hardware Context

This project was initially set up on:
- **Instance type:** Ephemeral cloud GPU (Prime Intellect style)
- **GPU:** NVIDIA A100-SXM4-40GB
- **CUDA:** 13.0
- **OS:** Ubuntu 22.04.5 LTS
- **Python:** 3.10.12
- **PyTorch:** 2.9.1+cu128

The setup should work on other CUDA-capable machines, but PyTorch version may need adjustment for different CUDA versions.

---

## Useful Commands

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make doctor` | Run environment diagnostics |
| `make smoke` | Full smoke test (loads model on GPU) |
| `make install` | Install/sync dependencies |
| `make notebook` | Start Jupyter server |
| `make lint` | Run ruff linter |
| `make format` | Auto-format code |
| `make snapshot` | Create timestamped run snapshot |
| `make push` | Commit and push to GitHub |
| `make backup` | Snapshot + push |

---

## For AI Assistants: Important Notes

1. **Always check GPU availability** before suggesting CUDA operations:
   ```python
   import torch
   device = "cuda" if torch.cuda.is_available() else "cpu"
   ```

2. **Use project paths correctly:**
   ```python
   from backtracking import PROJECT_ROOT
   config_path = PROJECT_ROOT / "configs" / "experiment.yaml"
   ```

3. **Large files go in appropriate directories:**
   - Model outputs → `results/`
   - Plots → `figures/`
   - Checkpoints → Consider W&B artifacts or manual backup

4. **Before any destructive operation**, remind user to run `make backup`.

5. **Environment variables** are set by `activate_env.sh`. If a script needs them, either:
   - Source the activation script first
   - Use `python-dotenv` to load `.env`

6. **New context files** should be added to `context/` as `context_N.md` where N is incremental.

---

## File Inventory (After Fresh Clone)

These directories/files already exist after clone:
- ✅ All folder structure (`src/`, `scripts/`, `data/`, etc.)
- ✅ `pyproject.toml`, `uv.lock`
- ✅ All scripts in `scripts/`
- ✅ `.gitignore`, `.env.example`
- ✅ `README.md`, `Makefile`
- ✅ `context/setup_context.md` (this file)

These need to be created by `bootstrap.sh`:
- 🔧 `.venv/` (virtual environment)
- 🔧 `.cache/` subdirectories
- 🔧 `.env` (copied from `.env.example`, user adds tokens)

---

## Change Log

| Date | Change |
|------|--------|
| 2025-12-23 | Initial project setup on ephemeral A100 instance |
| 2025-12-23 | Created bootstrap.sh for one-command setup |
| 2025-12-23 | Created context/setup_context.md |

---

*Last updated: 2025-12-23*

