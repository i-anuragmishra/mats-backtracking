This Phase 2 extends the existing `mats-backtracking` pipeline you already implemented (same repo structure, same `runs/`, `figures/`, `reports/`, `configs/` conventions). The goal is to turn the current “strong but destructive” ablation result into a **proposal-grade mechanistic claim** by (a) fixing metric confounds and (b) designing interventions that reduce backtracking **without collapsing accuracy**.

---

## 0) Context: what Phase 1 found (so Phase 2 has a purpose)

From the Phase 1 run:

- **Targeted ablation** reduced backtracking from ~70% → ~3%, much more than random ablation.  
- But **accuracy collapsed** under both targeted and random generation-time ablation (≈1–2% accuracy), meaning the intervention is currently too blunt and likely breaking general competence rather than selectively removing “backtracking mode”.

Key artifacts:
- Selected layers:
  - attn: `[0, 27, 17, 1, 19, 15]`
  - mlp: `[27, 19, 23, 24, 20, 22]` :contentReference[oaicite:0]{index=0}
- Summary metrics show severe accuracy collapse under ablation conditions. :contentReference[oaicite:1]{index=1}

**Phase 2 objective:** keep the causal story, remove the “it just breaks the model” interpretation.

---

## 1) Phase 2 goals (what success looks like)

### Goal A — Deconfounded metrics (must-have)
Fix aggregation so we can report:
- backtracking rate & accuracy **baseline-only**
- backtracking vs accuracy **within baseline-only** (not mixed with ablation conditions)
- formatting effects **baseline-only**
- condition comparisons **variant-specific** (since ablations were only run on one variant)

### Goal B — Non-destructive interventions (must-have)
Find an intervention that:
- reduces backtracking measurably (vs baseline)
- does **not** collapse accuracy nearly as badly
- ideally produces a **tradeoff curve** (backtracking ↓ vs accuracy ↓) rather than “everything dies”

### Goal C — Mechanistic specificity (nice-to-have)
Show the intervention changes:
- probability of the actual backtracking onset token (“ Wait” vs “Wait” etc.)
- without simply forcing a different synonym (“Actually”) or causing degenerate outputs

---

## 2) Implementation: what the coding agent should change/add

### 2.1 Files to add (Phase 2 additions)

Create new config:
- `configs/backtracking_state_transition_phase2.yaml`

Add new analysis modules (under `src/backtracking/analysis/`):
1) `metrics_v2.py`
2) `sweeps.py`
3) `continuations.py`
4) `plots_phase2.py`

Add new orchestration script:
- `scripts/run_backtracking_phase2.sh`

Update (modify) existing modules:
- `src/backtracking/hooks.py` (decode-only gating + instrumentation)
- `src/backtracking/analysis/events.py` (improve grouping tables)
- `src/backtracking/analysis/plots.py` (or route Phase 2 plots to `plots_phase2.py`)
- `src/backtracking/cli.py` (add Phase 2 commands)

---

## 3) Config spec for Phase 2 (`configs/backtracking_state_transition_phase2.yaml`)

Start by copying the Phase 1 config, then add these keys.

```yaml
run:
  name: "backtracking_state_transition_phase2"
  seed: 42
  run_id: null
  output_dir: "runs"

phase2:
  # Safety valve: sweeps create many outputs; keep dataset smaller.
  max_examples: 120
  num_samples_per_prompt: 4

  # Always compute baseline-only tables for proposal quoting
  compute_baseline_only_metrics: true

  # Ablation behavior changes for Phase 2
  decode_only: true          # hook only applies when seq_len == 1 during generate
  hook_debug: true           # log hook call counts and seq_len stats

  # Subset sweeps (non-destructive search)
  subset_sweep:
    enabled: true
    variant: "baseline_think_newline"
    # define subsets by name -> list of (component, layer)
    subsets:
      mlp_27_only:
        - ["mlp", 27]
      attn_27_only:
        - ["attn", 27]
      attn_no_early:
        - ["attn", 15]
        - ["attn", 17]
        - ["attn", 19]
        - ["attn", 27]
      mlp_late_cluster:
        - ["mlp", 19]
        - ["mlp", 20]
        - ["mlp", 22]
        - ["mlp", 23]
        - ["mlp", 24]
        - ["mlp", 27]
      phase1_full:
        # derived from selected_layers.json (attn + mlp)
        - ["attn", 0]
        - ["attn", 1]
        - ["attn", 15]
        - ["attn", 17]
        - ["attn", 19]
        - ["attn", 27]
        - ["mlp", 19]
        - ["mlp", 20]
        - ["mlp", 22]
        - ["mlp", 23]
        - ["mlp", 24]
        - ["mlp", 27]

  # Scale sweep for best subset(s)
  scale_sweep:
    enabled: true
    variant: "baseline_think_newline"
    subset_name: "mlp_late_cluster"  # can change after subset sweep
    scales: [0.0, 0.25, 0.5, 0.75, 0.9]

  # Continuation-only ablation: ablate at onset step only (minimal collateral damage)
  continuation_ablation:
    enabled: true
    variant: "baseline_think_newline"
    max_events: 150
    subset_name: "mlp_27_only"
    scales: [0.0, 0.5, 0.9]
    # We use prefixes ending at pred_pos (token before onset).
    # We compute P(onset_token) and optionally sample 1-step continuation.