#!/usr/bin/env bash
# =============================================================================
# MATS-BACKTRACKING Full Pipeline Orchestrator
# 
# Runs all stages of the backtracking state transition experiment.
# 
# Usage:
#   source scripts/activate_env.sh
#   bash scripts/run_backtracking_pipeline.sh [--config <path>] [--skip-generation]
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# Defaults
CONFIG="${CONFIG:-configs/backtracking_state_transition.yaml}"
SKIP_GENERATION=false
BASELINE_VARIANT="baseline_think_newline"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --skip-generation)
            SKIP_GENERATION=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Determine project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Ensure environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}Warning: Virtual environment not activated. Sourcing activate_env.sh...${NC}"
    source scripts/activate_env.sh
fi

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       BACKTRACKING STATE TRANSITION EXPERIMENT PIPELINE          ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Config: ${GREEN}$CONFIG${NC}"
echo ""

# Track timing
START_TIME=$(date +%s)

# -----------------------------------------------------------------------------
# Stage 0: Initialize Run
# -----------------------------------------------------------------------------
echo -e "${BOLD}[Stage 0/10] Initializing run...${NC}"
python -m backtracking.cli init-run --config "$CONFIG"
echo ""

# -----------------------------------------------------------------------------
# Stage 1: Prepare Data
# -----------------------------------------------------------------------------
echo -e "${BOLD}[Stage 1/10] Preparing dataset...${NC}"
python -m backtracking.cli prepare-data --config "$CONFIG"
echo ""

# -----------------------------------------------------------------------------
# Stage 2: Baseline Generation (all variants)
# -----------------------------------------------------------------------------
if [ "$SKIP_GENERATION" = false ]; then
    echo -e "${BOLD}[Stage 2/10] Generating baseline completions (all variants)...${NC}"
    python -m backtracking.cli generate --config "$CONFIG" --condition baseline
    echo ""
else
    echo -e "${YELLOW}[Stage 2/10] Skipping baseline generation (--skip-generation)${NC}"
    echo ""
fi

# -----------------------------------------------------------------------------
# Stage 3: Detect Events
# -----------------------------------------------------------------------------
echo -e "${BOLD}[Stage 3/10] Detecting backtracking events...${NC}"
python -m backtracking.cli detect-events --config "$CONFIG"
echo ""

# -----------------------------------------------------------------------------
# Stage 4: Logit Lens Analysis
# -----------------------------------------------------------------------------
echo -e "${BOLD}[Stage 4/10] Running logit lens analysis...${NC}"
python -m backtracking.cli logit-lens --config "$CONFIG" --variant "$BASELINE_VARIANT"
echo ""

# -----------------------------------------------------------------------------
# Stage 5: Ablation Scan
# -----------------------------------------------------------------------------
echo -e "${BOLD}[Stage 5/10] Running ablation scan...${NC}"
python -m backtracking.cli ablation-scan --config "$CONFIG" --variant "$BASELINE_VARIANT"
echo ""

# -----------------------------------------------------------------------------
# Stage 6: Targeted Ablation Generation
# -----------------------------------------------------------------------------
if [ "$SKIP_GENERATION" = false ]; then
    echo -e "${BOLD}[Stage 6/10] Generating with targeted ablation...${NC}"
    python -m backtracking.cli generate --config "$CONFIG" --condition targeted_ablation --variant "$BASELINE_VARIANT"
    echo ""
else
    echo -e "${YELLOW}[Stage 6/10] Skipping targeted ablation generation${NC}"
    echo ""
fi

# -----------------------------------------------------------------------------
# Stage 7: Random Ablation Generation
# -----------------------------------------------------------------------------
if [ "$SKIP_GENERATION" = false ]; then
    echo -e "${BOLD}[Stage 7/10] Generating with random ablation...${NC}"
    python -m backtracking.cli generate --config "$CONFIG" --condition random_ablation --variant "$BASELINE_VARIANT"
    echo ""
else
    echo -e "${YELLOW}[Stage 7/10] Skipping random ablation generation${NC}"
    echo ""
fi

# -----------------------------------------------------------------------------
# Stage 8: Re-detect Events (now includes ablation conditions)
# -----------------------------------------------------------------------------
echo -e "${BOLD}[Stage 8/10] Re-detecting events (including ablation conditions)...${NC}"
python -m backtracking.cli detect-events --config "$CONFIG"
echo ""

# -----------------------------------------------------------------------------
# Stage 9: Compare Conditions
# -----------------------------------------------------------------------------
echo -e "${BOLD}[Stage 9/10] Comparing conditions...${NC}"
python -m backtracking.cli compare-conditions --config "$CONFIG" --variant "$BASELINE_VARIANT"
echo ""

# -----------------------------------------------------------------------------
# Stage 10: Formatting Sweep
# -----------------------------------------------------------------------------
echo -e "${BOLD}[Stage 10/10] Formatting sweep...${NC}"
python -m backtracking.cli formatting-sweep --config "$CONFIG"
echo ""

# -----------------------------------------------------------------------------
# Generate Report
# -----------------------------------------------------------------------------
echo -e "${BOLD}Generating final report...${NC}"
python -m backtracking.cli make-report --config "$CONFIG"
echo ""

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

# Get run_id from .current_run_id
RUN_ID=$(cat runs/.current_run_id 2>/dev/null || echo "unknown")

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    PIPELINE COMPLETE! 🎉                         ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${BOLD}Run ID:${NC} $RUN_ID"
echo -e "  ${BOLD}Duration:${NC} ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo ""
echo -e "  ${BOLD}Key Artifacts:${NC}"
echo "    - runs/$RUN_ID/analysis/summary_metrics.json"
echo "    - runs/$RUN_ID/analysis/selected_layers.json"
echo "    - figures/backtracking_vs_accuracy.png"
echo "    - figures/wait_logit_lens_bt_vs_control.png"
echo "    - figures/ablation_importance_by_layer.png"
echo "    - figures/backtracking_rate_by_condition.png"
echo "    - reports/backtracking_state_transition_report.md"
echo ""
echo -e "  ${BOLD}Next steps:${NC}"
echo "    1. Review the report: cat reports/backtracking_state_transition_report.md"
echo "    2. Backup your work: make backup"
echo ""


