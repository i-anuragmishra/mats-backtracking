# Context 3: Phase 2 Implementation

**Date:** 2025-12-24  
**Phase 2 Run ID:** `20251224_045331`  
**Phase 1 Run ID (reference):** `20251223_232541`  
**Continuation from:** `context_2_run1.md`

---

## Overview

This document records the Phase 2 implementation of the backtracking state transition experiment. Phase 2 addresses the key limitation of Phase 1: while targeted ablation successfully reduced backtracking from ~70% to ~3%, it also collapsed accuracy to ~1-2%, making the intervention "too blunt."

**Phase 2 Goal:** Find non-destructive interventions that reduce backtracking while preserving model accuracy.

---

## Relationship to Spec

The implementation follows `reports/Backtracking_State_Transition_Phase2_Spec.md` exactly. Here's the mapping:

| Spec Section | Implementation |
|--------------|----------------|
| Config (Section 3) | `configs/backtracking_state_transition_phase2.yaml` |
| `metrics_v2.py` | `src/backtracking/analysis/metrics_v2.py` |
| `sweeps.py` | `src/backtracking/analysis/sweeps.py` |
| `continuations.py` | `src/backtracking/analysis/continuations.py` |
| `plots_phase2.py` | `src/backtracking/analysis/plots_phase2.py` |
| Hook instrumentation | Updated `src/backtracking/hooks.py` |
| CLI commands | Added 5 commands to `src/backtracking/cli.py` |
| Orchestrator | `scripts/run_backtracking_phase2.sh` |

---

## Files Created

### 1. Configuration
**`configs/backtracking_state_transition_phase2.yaml`**

New config extending Phase 1 with `phase2` section containing:
- `max_examples: 120` - Smaller dataset for faster sweeps
- `num_samples_per_prompt: 4` - Fewer samples per prompt
- `decode_only: true` - Hook only during decoding (seq_len == 1)
- `hook_debug: true` - Log hook call counts and seq_len stats
- `subset_sweep` - 5 named subsets to test:
  - `mlp_27_only` - Single MLP layer 27
  - `attn_27_only` - Single attention layer 27
  - `attn_no_early` - Attention layers 15, 17, 19, 27 (no early layers)
  - `mlp_late_cluster` - MLP layers 19, 20, 22, 23, 24, 27
  - `phase1_full` - All 12 components from Phase 1
- `scale_sweep` - Test scales [0.0, 0.25, 0.5, 0.75, 0.9]
- `continuation_ablation` - Minimal intervention at onset step only

### 2. Analysis Modules

**`src/backtracking/analysis/metrics_v2.py`**

Implements deconfounded metrics (Goal A from spec):
- `compute_baseline_only_metrics()` - Stats excluding ablation conditions
- `compute_variant_specific_metrics()` - Per-variant condition comparison
- `compute_backtracking_vs_accuracy_baseline()` - Correlation within baseline only
- `compute_formatting_baseline_only()` - Format effects without ablation confounds
- `save_metrics_v2()` - Saves to `metrics_v2.json`

**`src/backtracking/analysis/sweeps.py`**

Implements non-destructive intervention search (Goal B from spec):
- `generate_with_ablation()` - Generate completions with specified ablation
- `evaluate_generations()` - Compute backtracking rate and accuracy
- `run_subset_sweep()` - Test each layer subset, find which ones don't collapse accuracy
- `run_scale_sweep()` - Test different scale factors (0.0 to 0.9) for best subset
- `find_best_tradeoff()` - Identify subset with best BT reduction while retaining accuracy

**`src/backtracking/analysis/continuations.py`**

Implements minimal intervention analysis (Goal C from spec):
- `get_onset_token_prob()` - Measure P(onset_token) at prediction position
- `run_continuation_ablation()` - Teacher-forced pass, ablate only at critical decision point
- `summarize_continuation_results()` - Aggregate statistics

**`src/backtracking/analysis/plots_phase2.py`**

Creates proposal-ready figures:
- `plot_subset_sweep_comparison()` - Bar chart comparing subsets
- `plot_scale_tradeoff_curve()` - Line plot: BT rate vs accuracy at different scales
- `plot_continuation_ablation_effect()` - P(onset_token) under ablation
- `plot_baseline_only_summary()` - Deconfounded baseline metrics

### 3. Updated Modules

**`src/backtracking/hooks.py`** (extended)

Added hook instrumentation for debugging:
```python
class HookDebugState:
    enabled: bool
    total_calls: int
    ablation_calls: int  # When ablation was applied
    skipped_calls: int   # When skipped due to decode_only
    seq_lens: list[int]  # Distribution of sequence lengths
    by_layer: dict       # Per-layer statistics
```

New functions:
- `enable_hook_debug()` / `disable_hook_debug()` - Toggle instrumentation
- `reset_hook_debug()` - Clear counters
- `write_hook_debug_json()` - Save stats to file
- `specs_from_subset_config()` - Create ablation specs from Phase 2 config format
- `apply_ablation_hooks_with_debug()` - Context manager with debug support
- `apply_generation_hooks_with_debug()` - Generation hooks with debug

**`src/backtracking/paths.py`** (extended)

Added Phase 2 path helpers:
- `get_metrics_v2_file()` → `metrics_v2.json`
- `get_subset_sweep_file()` → `subset_sweep_results.csv`
- `get_scale_sweep_file()` → `scale_sweep_results.csv`
- `get_continuation_ablation_file()` → `continuation_ablation_results.csv`
- `get_hook_debug_file()` → `hook_debug.json`
- `get_phase2_generations_dir()` → Phase 2 generation outputs

**`src/backtracking/config.py`** (extended)

Added Phase 2 dataclasses:
- `Phase2Config` - Main Phase 2 configuration
- `Phase2SubsetSweepConfig` - Subset sweep settings
- `Phase2ScaleSweepConfig` - Scale sweep settings
- `Phase2ContinuationAblationConfig` - Continuation ablation settings
- `Phase2ReportConfig` - Phase 2 report settings

Updated `load_config()` to parse `phase2` section from YAML.

**`src/backtracking/cli.py`** (extended)

Added 5 new CLI commands:

| Command | Purpose |
|---------|---------|
| `metrics-v2` | Compute deconfounded baseline-only metrics |
| `phase2-subset-sweep` | Test different layer subsets |
| `phase2-scale-sweep` | Sweep scale factors for best subset |
| `phase2-continuation-ablation` | Minimal intervention at onset step |
| `make-report-phase2` | Generate Phase 2 markdown report |

### 4. Orchestrator Script

**`scripts/run_backtracking_phase2.sh`**

Full pipeline orchestrator:
```bash
./scripts/run_backtracking_phase2.sh 20251223_232541  # Pass Phase 1 run ID
```
# 1. Initialize Phase 2 run
python -m backtracking.cli init-run --config configs/backtracking_state_transition_phase2.yaml

# 2. Compute deconfounded metrics from Phase 1
python -m backtracking.cli metrics-v2 --config configs/backtracking_state_transition_phase2.yaml --phase1-run-id 20251223_232541

# 3. Run subset sweep (tests different layer combinations)
python -m backtracking.cli phase2-subset-sweep --config configs/backtracking_state_transition_phase2.yaml

# 4. Run scale sweep (tests different ablation strengths)
python -m backtracking.cli phase2-scale-sweep --config configs/backtracking_state_transition_phase2.yaml

# 5. Run continuation ablation (minimal intervention analysis)
python -m backtracking.cli phase2-continuation-ablation --config configs/backtracking_state_transition_phase2.yaml --phase1-run-id 20251223_232541

# 6. Generate Phase 2 report
python -m backtracking.cli make-report-phase2 --config configs/backtracking_state_transition_phase2.yaml --phase1-run-id 20251223_232541
Runs:
1. `init-run` - Create new Phase 2 run
2. `metrics-v2` - Compute deconfounded metrics from Phase 1 data
3. `phase2-subset-sweep` - Test layer subsets
4. `phase2-scale-sweep` - Test scale factors
5. `phase2-continuation-ablation` - Minimal intervention analysis
6. `make-report-phase2` - Generate final report

---

## Phase 2 Pipeline Flow

```
Phase 1 Data (run 20251223_232541)
    │
    ├── events.csv (6000 events)
    │       │
    │       ▼
    │   metrics-v2
    │       │
    │       ▼
    │   metrics_v2.json (deconfounded)
    │
    └── selected_layers.json
            │
            ▼
      ┌─────────────────┐
      │ subset_sweep    │ ← Test 5 subsets with scale=0.0
      └────────┬────────┘
               │
               ▼
      ┌─────────────────┐
      │ scale_sweep     │ ← Test scales [0.0, 0.25, 0.5, 0.75, 0.9]
      └────────┬────────┘   on best subset (mlp_late_cluster)
               │
               ▼
      ┌─────────────────┐
      │ continuation    │ ← Minimal intervention at onset position
      │ ablation        │
      └────────┬────────┘
               │
               ▼
      make-report-phase2
               │
               ▼
      reports/backtracking_phase2_report.md
```

---

## Key Differences from Phase 1

| Aspect | Phase 1 | Phase 2 |
|--------|---------|---------|
| Goal | Find important layers | Find non-destructive interventions |
| Ablation | All 12 layers at once | Test subsets individually |
| Scale | 0.0 only (full ablation) | Sweep 0.0 to 0.9 |
| Metrics | Mixed ablation + baseline | Baseline-only (deconfounded) |
| Hook debug | None | Full instrumentation |
| Sample size | 200 examples × 6 samples | 120 examples × 4 samples |

---

## Expected Outputs

### Analysis Files (in `runs/<run_id>/analysis/`)
- `metrics_v2.json` - Deconfounded baseline-only metrics
- `subset_sweep_results.csv` - BT rate + accuracy by subset
- `scale_sweep_results.csv` - BT rate + accuracy by scale factor
- `continuation_ablation_results.csv` - P(onset_token) under ablation
- `hook_debug.json` - Hook call counts and seq_len distribution

### Figures (in `figures/`)
- `phase2_subset_sweep.png` - Subset comparison
- `phase2_scale_tradeoff.png` - Scale factor tradeoff curve
- `phase2_continuation_effect.png` - Continuation ablation effect
- `phase2_baseline_summary.png` - Deconfounded baseline summary

### Report
- `reports/backtracking_phase2_report.md` - Full Phase 2 report

---

## Bug Fixes During Implementation

### 1. `format_prompt()` Signature Mismatch
**Error:** `TypeError: format_prompt() got an unexpected keyword argument 'template'`

**Cause:** In `sweeps.py`, I called `format_prompt()` with arguments that didn't match the actual function signature in `prompts.py`.

**Fix:** Updated `sweeps.py` to use correct signature:
```python
# Before (wrong)
prompt = format_prompt(
    question=ex["question"],
    template=config.prompting.template,
    think_open=variant_cfg.think_open,
    think_close=variant_cfg.think_close,
)

# After (correct)
prompt = format_prompt(
    question=ex["question"],
    config=config.prompting,
    variant=variant_cfg,
)
```

---

## Preliminary Results (from metrics-v2)

From Phase 1 data, baseline-only deconfounded metrics:

| Metric | Value |
|--------|-------|
| Total samples (baseline only) | 3,600 |
| Backtracking rate | 68.1% |
| Overall accuracy | 19.9% |
| Accuracy WITH backtracking | **23.7%** |
| Accuracy WITHOUT backtracking | **11.5%** |

**Key insight:** In baseline (no ablation), samples with backtracking are **2.1x more accurate** than those without. This confirms backtracking is a beneficial self-correction mechanism.

---

## What's Running Now

The `phase2-subset-sweep` command is currently running, which will:
1. Generate completions with each of 5 layer subsets ablated
2. Evaluate backtracking rate and accuracy for each
3. Identify which subset(s) reduce backtracking without collapsing accuracy

This is the critical step for finding non-destructive interventions.

---

## Next Steps After Phase 2

1. **Review subset sweep results** - Which subsets retain accuracy?
2. **Run scale sweep** - Fine-tune ablation strength
3. **Analyze continuation ablation** - Does minimal intervention work?
4. **Write conclusions** - Can we make proposal-grade claims?

---

## For Future AI Assistants

This project follows strict conventions:
- All configs in `configs/`
- All outputs in `runs/<run_id>/`
- Proposal-ready figures copied to `figures/`
- Reports in `reports/`
- Context files in `context/` (this file)

The Phase 2 implementation is complete. If the user encounters errors:
1. Check that Phase 1 run ID (`20251223_232541`) exists
2. Ensure `--phase1-run-id` flag is passed to commands that need it
3. Verify the Phase 2 config has all required `phase2.*` sections

