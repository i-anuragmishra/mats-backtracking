"""
Analysis subpackage for backtracking experiments.

Contains modules for:
- events: Processing generations into backtracking events
- logit_lens: Computing "Wait" logit by layer
- ablation_scan: Per-layer ablation sensitivity analysis
- plots: Generating figures for the report
"""

from backtracking.analysis.events import process_generations_to_events
from backtracking.analysis.logit_lens import run_logit_lens_analysis
from backtracking.analysis.ablation_scan import run_ablation_scan
from backtracking.analysis.plots import (
    plot_backtracking_rate_by_variant,
    plot_backtracking_vs_accuracy,
    plot_logit_lens,
    plot_ablation_importance,
    plot_backtracking_by_condition,
    plot_formatting_effect,
)

__all__ = [
    "process_generations_to_events",
    "run_logit_lens_analysis",
    "run_ablation_scan",
    "plot_backtracking_rate_by_variant",
    "plot_backtracking_vs_accuracy",
    "plot_logit_lens",
    "plot_ablation_importance",
    "plot_backtracking_by_condition",
    "plot_formatting_effect",
]


