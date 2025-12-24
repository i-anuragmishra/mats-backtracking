#!/bin/bash
# =============================================================================
# Phase 2: Backtracking State Transition Pipeline
# =============================================================================
# This script runs the complete Phase 2 analysis pipeline.
#
# Phase 2 goals:
# - Deconfounded metrics (baseline-only statistics)
# - Non-destructive ablation sweeps (subset + scale)
# - Continuation-only ablation analysis
#
# Usage:
#   ./scripts/run_backtracking_phase2.sh [PHASE1_RUN_ID]
#
# If PHASE1_RUN_ID is not provided, it will try to use the current run ID.
# =============================================================================

set -e  # Exit on error

# Configuration
CONFIG="configs/backtracking_state_transition_phase2.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Phase 1 run ID (optional argument)
PHASE1_RUN_ID="${1:-}"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        Backtracking State Transition - Phase 2 Pipeline         ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if Phase 1 run ID was provided
if [ -n "$PHASE1_RUN_ID" ]; then
    echo -e "${YELLOW}Using Phase 1 run ID: ${PHASE1_RUN_ID}${NC}"
    PHASE1_FLAG="--phase1-run-id $PHASE1_RUN_ID"
else
    echo -e "${YELLOW}No Phase 1 run ID provided. Will use current run's events.${NC}"
    PHASE1_FLAG=""
fi
echo ""

# =============================================================================
# Step 1: Initialize new run for Phase 2
# =============================================================================
echo -e "${BLUE}[Step 1/6] Initializing Phase 2 run...${NC}"
python -m backtracking.cli init-run --config "$CONFIG"
echo -e "${GREEN}✓ Phase 2 run initialized${NC}"
echo ""

# =============================================================================
# Step 2: Compute deconfounded metrics
# =============================================================================
echo -e "${BLUE}[Step 2/6] Computing deconfounded metrics...${NC}"
python -m backtracking.cli metrics-v2 --config "$CONFIG" $PHASE1_FLAG
echo -e "${GREEN}✓ Deconfounded metrics complete${NC}"
echo ""

# =============================================================================
# Step 3: Run subset sweep
# =============================================================================
echo -e "${BLUE}[Step 3/6] Running subset sweep...${NC}"
echo -e "${YELLOW}  This may take a while (generates completions for each subset)${NC}"
python -m backtracking.cli phase2-subset-sweep --config "$CONFIG"
echo -e "${GREEN}✓ Subset sweep complete${NC}"
echo ""

# =============================================================================
# Step 4: Run scale sweep
# =============================================================================
echo -e "${BLUE}[Step 4/6] Running scale sweep...${NC}"
echo -e "${YELLOW}  Testing scale factors: 0.0, 0.25, 0.5, 0.75, 0.9${NC}"
python -m backtracking.cli phase2-scale-sweep --config "$CONFIG"
echo -e "${GREEN}✓ Scale sweep complete${NC}"
echo ""

# =============================================================================
# Step 5: Run continuation ablation
# =============================================================================
echo -e "${BLUE}[Step 5/6] Running continuation ablation analysis...${NC}"
python -m backtracking.cli phase2-continuation-ablation --config "$CONFIG" $PHASE1_FLAG
echo -e "${GREEN}✓ Continuation ablation complete${NC}"
echo ""

# =============================================================================
# Step 6: Generate Phase 2 report
# =============================================================================
echo -e "${BLUE}[Step 6/6] Generating Phase 2 report...${NC}"
python -m backtracking.cli make-report-phase2 --config "$CONFIG" $PHASE1_FLAG
echo -e "${GREEN}✓ Phase 2 report generated${NC}"
echo ""

# =============================================================================
# Summary
# =============================================================================
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Phase 2 Pipeline Complete!                    ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Outputs:"
echo "  - Metrics: runs/<run_id>/analysis/metrics_v2.json"
echo "  - Subset sweep: runs/<run_id>/analysis/subset_sweep_results.csv"
echo "  - Scale sweep: runs/<run_id>/analysis/scale_sweep_results.csv"
echo "  - Continuation: runs/<run_id>/analysis/continuation_ablation_results.csv"
echo "  - Hook debug: runs/<run_id>/analysis/hook_debug.json"
echo "  - Report: reports/backtracking_phase2_report.md"
echo ""
echo "Figures:"
echo "  - figures/phase2_subset_sweep.png"
echo "  - figures/phase2_scale_tradeoff.png"
echo "  - figures/phase2_continuation_effect.png"
echo "  - figures/phase2_baseline_summary.png"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Review the Phase 2 report"
echo "  2. Check the tradeoff curves for optimal ablation parameters"
echo "  3. Run './scripts/push_all.sh' to save results to GitHub"

