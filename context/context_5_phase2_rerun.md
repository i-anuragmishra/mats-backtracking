# Context 5: Phase 2 Continuation - New Instance (Jan 2026)

**Date:** 2026-01-01  
**Previous Context:** `context_4_phase2_continuation.md`  
**Phase 1 Run ID:** `20251223_232541`  
**Phase 2 Run ID:** `20251224_045331`  
**Compute Node:** `clx-a-02` (RIT Research Computing cluster)

---

## Overview

This document logs the continuation of Phase 2 experiments on a new free GPU instance. The previous paid instance was shut down before completing Phase 2.

**Key difference:** This is a cluster setup where we SSH into `clx-a-02` to access the GPU.

---

## Instance Setup Log

### Step 1: Bootstrap Environment

**Status:** ✅ COMPLETE

```bash
bash scripts/bootstrap.sh
```
- uv 0.9.21 installed
- 185 Python packages installed
- .venv created on shared filesystem

### Step 2: Configure Compute Node Access

**Status:** ✅ COMPLETE

- Login node: `am2552@<hostname>` (no GPU)
- Compute node: `clx-a-02` (has NVIDIA A100-PCIE-40GB)
- All commands run via: `ssh clx-a-02 "cd /home/am2552/mats-backtracking && source scripts/activate_env.sh && <command>"`

### Step 3: GPU Verification

**Status:** ✅ COMPLETE

```
GPU: NVIDIA A100-PCIE-40GB (40GB VRAM)
Driver: 550.90.07
CUDA: 12.4
PyTorch: 2.9.1+cu128
```

### Step 4: HuggingFace Token

**Status:** ✅ COMPLETE

New token configured in `.env` (user: anurag-ai)

---

## Execution Log

### 2026-01-02: Phase 2 Subset Sweep

**Status:** ✅ COMPLETE (~9 hours total)

**Results:**

| Subset | BT Rate | Accuracy | BT Reduction |
|--------|---------|----------|--------------|
| baseline | 63.3% | 1.2% | - |
| mlp_27_only | 28.3% | 1.5% | 55% |
| attn_27_only | 87.5% | 1.0% | -38% |
| attn_no_early | 99.4% | 1.0% | -57% |
| mlp_late_cluster | 9.8% | 1.2% | 85% |
| phase1_full | 1.7% | 1.5% | 97% |

**Key Findings:**
1. **MLP layer 27 alone** reduces backtracking by 55% - single most important layer!
2. **MLP late cluster** (layers 19, 20, 22, 23, 24, 27) gives 85% reduction
3. **Attention ablation increases backtracking** - attention layers are NOT the mechanism
4. **Backtracking is primarily mediated by late MLP layers**

---

## Pre-Existing State

### Phase 1 Run (20251223_232541)
- ✅ All analysis files present
- ✅ Figures present
- ❌ Generation files (gitignored, not present)

### Phase 2 Run (20251224_045331)
- ✅ `init-run` complete (meta.json exists)
- ✅ `metrics-v2` complete (metrics_v2.json exists)
- ✅ `phase2_baseline_summary.png` exists
- ❌ `phase2-subset-sweep` - **NEEDS TO BE RE-RUN**
- ❌ `phase2-scale-sweep` - **NOT YET RUN**
- ❌ `phase2-continuation-ablation` - **NOT YET RUN**
- ❌ `make-report-phase2` - **NOT YET RUN**

### Previous Subset Sweep Partial Results (from context_4)

Before crash, these results were obtained:

| Subset | BT Rate | Accuracy | Components |
|--------|---------|----------|------------|
| baseline | 65.0% | 0.4% | 0 |
| mlp_27_only | **27.5%** | 0.6% | 1 |
| attn_27_only | 86.9% | 1.5% | 1 |
| attn_no_early | 99.2% | 1.7% | 4 |
| mlp_late_cluster | ❌ CRASHED | - | 6 |
| phase1_full | ❌ NOT RUN | - | 12 |

**Key Finding:** `mlp_27_only` (single MLP layer 27) reduced backtracking by 58% with minimal disruption!

---

## Execution Plan

1. ✅ Bootstrap environment (`scripts/bootstrap.sh`)
2. ✅ Create .env with HF_TOKEN
3. ✅ Activate environment (`source scripts/activate_env.sh`)
4. ✅ Verify setup (`make doctor`)
5. ✅ Restore `.current_run_id` to `20251224_045331`
6. ✅ Run `phase2-subset-sweep` - **COMPLETE** (9 hours)
7. 🔄 Run `phase2-scale-sweep` - **IN PROGRESS**
8. ⏳ Run `phase2-continuation-ablation`
9. ⏳ Run `make-report-phase2`

---

## Execution Log

### 2026-01-01: Environment Setup

*Updates will be added as commands complete.*

---

*Last updated: 2026-01-01*
