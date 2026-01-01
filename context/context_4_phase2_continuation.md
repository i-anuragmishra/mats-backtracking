# Context 4: Phase 2 Continuation on New Instance

**Date:** 2025-12-31  
**Previous Instance Shutdown:** 2025-12-24  
**Continuation from:** `context_3_phase2.md`

---

## Overview

This document logs the continuation of Phase 2 experiments after migrating to a new GPU instance. The previous instance was shut down mid-Phase 2, with all code committed to GitHub.

---

## Instance Migration

**Previous state (from context_3):**
- Phase 2 Run ID: `20251224_045331`
- Phase 1 Run ID: `20251223_232541`
- Completed: `init-run`, `metrics-v2`
- Pending: `phase2-subset-sweep`, `phase2-scale-sweep`, `phase2-continuation-ablation`, `make-report-phase2`

**Current state:**
- Fresh clone on new instance
- Need to bootstrap environment
- Need to continue from `phase2-subset-sweep`

---

## Environment Setup Log

### Step 1: Bootstrap Environment

```bash
bash scripts/bootstrap.sh
```

**Status:** ✅ COMPLETE (minor fish config error, worked around)

- uv 0.9.21 installed
- 188 Python packages installed via `uv sync`

### Step 2: Configure .env

```bash
cat > .env << 'EOF'
GH_TOKEN=
HF_TOKEN=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
WANDB_API_KEY=
WANDB_PROJECT=mats-backtracking
WANDB_ENTITY=i-anuragmishra
EOF
```

**Status:** ✅ COMPLETE

### Step 3: Verify Setup

```bash
source scripts/activate_env.sh
make doctor
```

**Status:** ✅ COMPLETE

**New Instance Specs:**
- **Hostname:** 0358-kci2-ty6k-prxmx100101
- **GPU:** NVIDIA RTX 6000 Ada Generation (49GB VRAM)
- **Driver:** 575.64.03
- **CUDA:** 12.9
- **PyTorch:** 2.9.1+cu128
- **Transformers:** 4.57.3
- **HF Token:** Verified working

### Step 4: Fix Matplotlib Config

Added `MPLCONFIGDIR` to `activate_env.sh` to avoid permission issues.

**Status:** ✅ COMPLETE

---

## Phase 2 Continuation Plan

After environment setup, continue with:

1. `phase2-subset-sweep` - Test 5 layer subsets
2. `phase2-scale-sweep` - Sweep scale factors on best subset
3. `phase2-continuation-ablation` - Minimal intervention analysis
4. `make-report-phase2` - Generate final report

---

## Run State Verification

**Phase 1 Run:** `20251223_232541`
- All analysis files present ✅
- Generations gitignored (not present on new instance)

**Phase 2 Run:** `20251224_045331`
- `init-run` ✅ (meta.json exists)
- `metrics-v2` ✅ (metrics_v2.json exists)
- `phase2-subset-sweep` ❌ NOT YET RUN
- `phase2-scale-sweep` ❌ NOT YET RUN
- `phase2-continuation-ablation` ❌ NOT YET RUN
- `make-report-phase2` ❌ NOT YET RUN

**Recreated:** `runs/.current_run_id` → `20251224_045331`

---

## Execution Log

### 2025-12-31: Subset Sweep Attempt 1

**Command:** `python -m backtracking.cli phase2-subset-sweep --config configs/backtracking_state_transition_phase2.yaml`

**Duration:** ~6 hours (ran 4/5 subsets before crash)

**Partial Results:**
| Subset | BT Rate | Accuracy | Components |
|--------|---------|----------|------------|
| baseline | 65.0% | 0.4% | 0 |
| mlp_27_only | **27.5%** | 0.6% | 1 |
| attn_27_only | 86.9% | 1.5% | 1 |
| attn_no_early | 99.2% | 1.7% | 4 |
| mlp_late_cluster | ❌ CRASHED | - | 6 |
| phase1_full | ❌ NOT RUN | - | 12 |

**Key Finding:** `mlp_27_only` reduced backtracking from 65% → 27.5% (58% reduction) with just 1 layer!

**Error:** `OverflowError: cannot convert float infinity to integer`
- Model generated garbage like "1000000000..." (hundreds of digits)
- `normalize_answer()` tried `int(float('inf'))` which fails

**Bug Fix:** Updated `src/backtracking/detect.py` to handle infinity and OverflowError

---

## Notes

- New instance has RTX 6000 Ada (49GB VRAM) vs previous A100 (40GB)
- Phase 1 generation files (~6000 completions) are not present (gitignored)
- Phase 2 can proceed since it generates fresh completions for sweeps

---

*Last updated: 2025-12-31*
