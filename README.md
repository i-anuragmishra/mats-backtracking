# MATS-BACKTRACKING

> Mechanistic Interpretability Research: Investigating backtracking and state transitions in language models.

## 🚀 Quickstart

### 1. Activate Environment

```bash
# Source the activation script (sets cache dirs + activates venv)
source scripts/activate_env.sh
```

### 2. Install Dependencies

```bash
# First time setup (creates venv and installs packages)
uv sync

# Or with optional mechanistic interp packages:
uv sync --all-extras
```

### 3. Verify Setup

```bash
# Run diagnostics
make doctor

# Run full smoke test (loads a tiny model on GPU)
make smoke
```

### 4. Add Your Secrets

```bash
# Copy example env file
cp .env.example .env

# Edit with your tokens (NEVER commit this file)
nano .env
```

---

## 📁 Project Structure

```
mats-backtracking/
├── README.md              # This file
├── .gitignore             # Git ignore rules (excludes weights, caches, secrets)
├── .env.example           # Template for environment variables
├── .env                   # Your secrets (NEVER committed)
├── Makefile               # Common commands (make help for list)
├── pyproject.toml         # Project config + dependencies
├── uv.lock                # Locked dependency versions
│
├── src/backtracking/      # Main Python package
│   ├── __init__.py
│   └── ...
│
├── scripts/               # Shell/Python utility scripts
│   ├── activate_env.sh    # Environment activation
│   ├── doctor.sh          # Diagnostics
│   ├── run_smoke_test.py  # GPU/import verification
│   ├── snapshot_run.sh    # Create timestamped run snapshot
│   └── push_all.sh        # Commit and push to git
│
├── notebooks/             # Jupyter notebooks for exploration
├── configs/               # YAML/JSON configuration files
│
├── data/
│   ├── raw/               # Original, immutable data (gitignored)
│   ├── interim/           # Intermediate transformed data
│   └── processed/         # Final data for modeling
│
├── results/               # Experiment outputs (metrics, tables)
├── figures/               # Generated plots and visualizations
├── reports/               # Writeups, LaTeX, etc.
├── logs/                  # Training/experiment logs (gitignored)
│
├── runs/                  # Timestamped run snapshots
│   └── YYYYMMDD_HHMMSS/   # Auto-created by snapshot_run.sh
│
├── scratch/               # Temporary work (gitignored)
├── .cache/                # HuggingFace/Torch/pip caches (gitignored)
└── wandb/                 # W&B local files (gitignored)
```

---

## 🔧 Common Commands

Run `make help` for a full list. Key commands:

| Command | Description |
|---------|-------------|
| `make install` | Install dependencies |
| `make doctor` | Run environment diagnostics |
| `make smoke` | Run full smoke test |
| `make notebook` | Start Jupyter server |
| `make lint` | Run linter (ruff) |
| `make format` | Auto-format code |
| `make snapshot` | Create run snapshot |
| `make push` | Commit and push to git |
| `make backup` | Snapshot + push |

---

## 💾 Persistence Workflow

**Before shutting down your instance**, always run:

```bash
# Create a snapshot of current state
make snapshot

# Commit and push everything to GitHub
make push

# Or do both at once:
make backup
```

This ensures your code, configs, and lightweight results are safely stored.

### What Gets Saved

✅ **Tracked by git:**
- All code (`src/`, `scripts/`)
- Notebooks (`notebooks/`)
- Configs (`configs/`)
- Results, figures, reports (small files)
- Run snapshots (`runs/*/metadata.txt`, etc.)

❌ **NOT tracked (too large):**
- Model weights (`.pt`, `.bin`, `.safetensors`)
- Caches (`.cache/`)
- Raw data (`data/raw/`)
- W&B local files (`wandb/`)
- Secrets (`.env`)

### Large Artifact Options

For large artifacts (model checkpoints, activation caches):

1. **W&B Artifacts**: Upload via `wandb.log_artifact()`
2. **Manual archive**: `tar -czvf run_artifacts.tar.gz results/`
3. **DVC** (if configured): `dvc push`

---

## 🔐 Secrets Management

**Never commit secrets to git!**

1. Copy the template: `cp .env.example .env`
2. Edit `.env` with your actual tokens
3. The activation script auto-loads `.env`

Required tokens:
- `HF_TOKEN`: HuggingFace Hub access ([get one here](https://huggingface.co/settings/tokens))
- `WANDB_API_KEY`: Weights & Biases ([get one here](https://wandb.ai/authorize))

---

## 🧪 Running Experiments

### Example: Training/Analysis Script

```python
#!/usr/bin/env python3
"""Example experiment script."""

import os
from pathlib import Path

# Import project package
from backtracking import PROJECT_ROOT

# Configs
config_path = PROJECT_ROOT / "configs" / "experiment.yaml"

# Save results
results_dir = PROJECT_ROOT / "results"
figures_dir = PROJECT_ROOT / "figures"

# Your experiment code here...
```

### Using W&B

```python
import wandb

# Initialize (uses WANDB_* env vars from .env)
wandb.init(
    project=os.environ.get("WANDB_PROJECT", "mats-backtracking"),
    dir=os.environ.get("WANDB_DIR", "./wandb"),
)

# Log metrics
wandb.log({"loss": 0.5, "accuracy": 0.9})

# Save artifacts
wandb.save("results/*.json")
```

---

## 📓 Experiment Log

### Template

```markdown
## YYYY-MM-DD: Experiment Name

**Goal:** What are you trying to learn/test?

**Setup:**
- Model: ...
- Dataset: ...
- Key hyperparameters: ...

**Results:**
- Finding 1
- Finding 2

**Next steps:**
- ...

**Commit:** `abc1234`
```

---

### Log Entries

*(Add your entries below)*

---

## 🔗 Resources

- [TransformerLens Docs](https://transformerlensorg.github.io/TransformerLens/)
- [ARENA 3.0 Curriculum](https://www.arena.education/)
- [Neel Nanda's Concrete MI Tutorial](https://www.neelnanda.io/mechanistic-interpretability/getting-started)

---

## License

MIT

