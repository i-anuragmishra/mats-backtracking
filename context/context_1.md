# Context 1: Backtracking State Transition Pipeline Implementation

> **Date:** 2025-12-23  
> **Purpose:** Documents the full experiment pipeline implementation for investigating backtracking in reasoning models.

---

## What Was Built

A complete mechanistic interpretability pipeline with **18 Python modules** and **9 CLI commands** to investigate how language models perform backtracking ("Wait", "Actually", "Hold on") during reasoning.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         EXPERIMENT PIPELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. init-run         → Creates run folder + persists run_id          │
│  2. prepare-data     → Loads GSM8K, extracts answers                 │
│  3. generate         → Generates completions (baseline/ablation)     │
│  4. detect-events    → Finds backtracking triggers, maps to tokens   │
│  5. logit-lens       → Measures "Wait" logit by layer                │
│  6. ablation-scan    → Tests which layers are causally important     │
│  7. compare-conditions → Compares baseline vs ablation               │
│  8. formatting-sweep → Tests formatting variants                     │
│  9. make-report      → Generates markdown report                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Files Created

### Config
- `configs/backtracking_state_transition.yaml` - All experiment parameters

### Core Modules (`src/backtracking/`)
| File | Purpose |
|------|---------|
| `config.py` | Load YAML config, manage run_id persistence |
| `paths.py` | Centralized path management for runs/figures/data |
| `seed.py` | Deterministic seeding (torch, numpy, random) |
| `io.py` | Read/write JSONL, CSV, JSON utilities |
| `tokenization.py` | **Single source** for all tokenization (critical!) |
| `modeling.py` | Load HuggingFace model/tokenizer |
| `prompts.py` | Format prompts with formatting variants |
| `detect.py` | Detect backtracking triggers, store `onset_token_id` |
| `generate.py` | Batch generation with optional ablation hooks |
| `hooks.py` | `AblationSpec`, context managers for layer ablation |

### Analysis Modules (`src/backtracking/analysis/`)
| File | Purpose |
|------|---------|
| `__init__.py` | Package init |
| `events.py` | Process generations → events CSV + summary |
| `logit_lens.py` | Compute target token logit by layer |
| `ablation_scan.py` | Per-layer ablation sensitivity scan |
| `plots.py` | Generate all proposal-ready figures |

### Output
| File | Purpose |
|------|---------|
| `report.py` | Generate markdown report with figures/metrics |
| `cli.py` | Typer CLI with 9 commands |

### Orchestrator
- `scripts/run_backtracking_pipeline.sh` - Runs full pipeline in order

---

## Critical Design Decisions (Fixes)

### 1. Run ID Persistence
**Problem:** Each CLI call would create a new run folder.  
**Solution:** `init-run` command writes `runs/.current_run_id`. All other commands read this file.

```bash
python -m backtracking.cli init-run --config configs/backtracking_state_transition.yaml
# Creates: runs/20251223_123456/
# Writes: runs/.current_run_id
```

### 2. Position-Matched Controls
**Problem:** Controls have no natural position, confounds analysis.  
**Solution:** For each backtracking event at offset `k`, find control with `completion_len >= k` and set `pred_pos = prompt_tokens + k - 1`.

### 3. Actual Onset Token ID
**Problem:** BPE may tokenize "Wait" as " Wait" or multiple tokens.  
**Solution:** During detection, store the **actual** `onset_token_id` at that position. Logit lens uses this ID, not `tokenizer.encode("Wait")`.

### 4. Shape-Matched Random Ablation
**Problem:** Random ablation could be unfairly weak/strong.  
**Solution:** If targeted = 4 attn + 2 mlp layers, random also ablates 4 attn + 2 mlp from different layers.

### 5. Consistent Tokenization
**Problem:** BOS tokens cause off-by-one errors.  
**Solution:** Single `tokenization.py` module used for:
- Generation input
- Event offset mapping
- Teacher-forced forward passes

---

## Output Artifacts

### Figures (saved to `figures/` and `runs/<run_id>/figures/`)
1. `backtracking_rate_by_variant.png`
2. `backtracking_vs_accuracy.png`
3. `wait_logit_lens_bt_vs_control.png`
4. `ablation_importance_by_layer.png`
5. `backtracking_rate_by_condition.png`
6. `formatting_effect_on_backtracking.png`

### Analysis Files (in `runs/<run_id>/analysis/`)
- `backtracking_events.csv` - All events with token positions
- `summary_metrics.json` - Backtracking rates, accuracy
- `logit_lens.csv` - Per-layer logit values
- `ablation_importance.csv` - Per-layer ablation impact
- `selected_layers.json` - Top-k layers for ablation
- `condition_comparison.csv` - Baseline vs ablation comparison
- `formatting_summary.csv` - Per-variant statistics

### Run Metadata
- `runs/<run_id>/config_resolved.yaml` - Resolved config
- `runs/<run_id>/meta.json` - Git SHA, timestamp, model ID

---

## How to Run

### Full Pipeline (Automated)
```bash
source scripts/activate_env.sh
bash scripts/run_backtracking_pipeline.sh
```

### Step-by-Step (Manual)
```bash
# 1. Initialize run
python -m backtracking.cli init-run --config configs/backtracking_state_transition.yaml

# 2. Prepare dataset
python -m backtracking.cli prepare-data --config configs/backtracking_state_transition.yaml

# 3. Generate baseline (all variants)
python -m backtracking.cli generate --config configs/backtracking_state_transition.yaml --condition baseline

# 4. Detect backtracking events
python -m backtracking.cli detect-events --config configs/backtracking_state_transition.yaml

# 5. Run logit lens analysis
python -m backtracking.cli logit-lens --config configs/backtracking_state_transition.yaml --variant baseline_think_newline

# 6. Run ablation scan
python -m backtracking.cli ablation-scan --config configs/backtracking_state_transition.yaml --variant baseline_think_newline

# 7. Generate with targeted ablation
python -m backtracking.cli generate --config configs/backtracking_state_transition.yaml --condition targeted_ablation --variant baseline_think_newline

# 8. Generate with random ablation
python -m backtracking.cli generate --config configs/backtracking_state_transition.yaml --condition random_ablation --variant baseline_think_newline

# 9. Re-detect events (includes ablation conditions)
python -m backtracking.cli detect-events --config configs/backtracking_state_transition.yaml

# 10. Compare conditions
python -m backtracking.cli compare-conditions --config configs/backtracking_state_transition.yaml --variant baseline_think_newline

# 11. Formatting sweep
python -m backtracking.cli formatting-sweep --config configs/backtracking_state_transition.yaml

# 12. Generate report
python -m backtracking.cli make-report --config configs/backtracking_state_transition.yaml
```

---

## CLI Commands Reference

| Command | Flags | Output |
|---------|-------|--------|
| `init-run` | `--config`, `--run-id` | Creates run folder, `.current_run_id` |
| `prepare-data` | `--config` | `data/processed/gsm8k_200.jsonl` |
| `generate` | `--config`, `--condition`, `--variant` | `runs/<id>/generations/<variant>/<condition>.jsonl` |
| `detect-events` | `--config` | `backtracking_events.csv`, `summary_metrics.json`, figures |
| `logit-lens` | `--config`, `--variant` | `logit_lens.csv`, figure |
| `ablation-scan` | `--config`, `--variant` | `ablation_importance.csv`, `selected_layers.json`, figure |
| `compare-conditions` | `--config`, `--variant` | `condition_comparison.csv`, figure |
| `formatting-sweep` | `--config` | `formatting_summary.csv`, figure |
| `make-report` | `--config` | `reports/backtracking_state_transition_report.md` |

---

## Git Tracking

### Tracked
- All Python code
- Configs, scripts
- `runs/**/analysis/` (small CSV/JSON)
- `runs/**/figures/`
- `runs/**/config_resolved.yaml`
- `runs/**/meta.json`
- `figures/` (proposal-ready)

### Ignored
- `runs/**/generations/` (large JSONL files)
- `.current_run_id`
- Model weights, caches
- `reports/Backtracking_State_Transition_Spec.md` (reference doc)

---

## Hypotheses Being Tested

### H1: Localized Trigger
Backtracking onset is mediated by a small subset of transformer blocks that are causally important for producing "Wait/Actually".

### H0: Diffuse/Reroutable
Backtracking is distributed; ablating any small set doesn't reliably reduce it.

### H2: Format Dependence
Backtracking is heavily conditioned on output formatting (e.g., `<think>` tags) rather than robust error detection.

---

## For AI Assistants

1. **Run ID is persistent** - After `init-run`, all commands use `.current_run_id` automatically
2. **Check `selected_layers.json`** before running ablation generation
3. **Large generations are NOT committed** - Only analysis outputs are tracked
4. **Always source `activate_env.sh`** before running CLI commands
5. **The spec file is gitignored** - It's a reference document, not part of the codebase

---

*Created: 2025-12-23*


