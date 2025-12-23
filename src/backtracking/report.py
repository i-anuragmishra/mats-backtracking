"""
Report generation for backtracking experiments.

Generates a markdown report summarizing the experiment results.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from backtracking import PROJECT_ROOT
from backtracking.io import read_csv, read_json
from backtracking.paths import (
    get_ablation_importance_file,
    get_analysis_dir,
    get_events_file,
    get_figures_dir,
    get_logit_lens_file,
    get_reports_dir,
    get_run_dir,
    get_selected_layers_file,
    get_summary_metrics_file,
)


def generate_report(
    run_id: str,
    config_path: str | None = None,
    output_path: str | None = None,
) -> Path:
    """
    Generate markdown report for a run.
    
    Args:
        run_id: Run identifier
        config_path: Path to config file used
        output_path: Optional output path (default: reports/backtracking_state_transition_report.md)
        
    Returns:
        Path to generated report
    """
    run_dir = get_run_dir(run_id)
    analysis_dir = get_analysis_dir(run_id)
    figures_dir = get_figures_dir()
    
    # Load data
    summary = {}
    if get_summary_metrics_file(run_id).exists():
        summary = read_json(get_summary_metrics_file(run_id))
    
    selected_layers = {}
    if get_selected_layers_file(run_id).exists():
        selected_layers = read_json(get_selected_layers_file(run_id))
    
    meta = {}
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        meta = read_json(meta_path)
    
    # Build report
    lines = []
    
    # Header
    lines.append("# Backtracking State Transition Experiment Report")
    lines.append("")
    lines.append(f"**Run ID:** `{run_id}`")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if meta:
        lines.append(f"**Model:** `{meta.get('model_id', 'unknown')}`")
        lines.append(f"**Git SHA:** `{meta.get('git_sha', 'unknown')}`")
    lines.append("")
    
    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    
    if summary:
        bt_rate = summary.get("backtracking_strict_rate", 0) * 100
        acc = summary.get("accuracy", 0)
        acc_bt = summary.get("accuracy_with_backtracking", 0)
        acc_no_bt = summary.get("accuracy_without_backtracking", 0)
        
        lines.append(f"- **Backtracking Rate:** {bt_rate:.1f}% of samples exhibit backtracking")
        if acc is not None:
            lines.append(f"- **Overall Accuracy:** {acc*100:.1f}%")
        if acc_bt is not None and acc_no_bt is not None:
            diff = (acc_bt - acc_no_bt) * 100
            direction = "higher" if diff > 0 else "lower"
            lines.append(f"- **Accuracy with Backtracking:** {acc_bt*100:.1f}% ({abs(diff):.1f}% {direction} than without)")
    lines.append("")
    
    # Key Findings
    lines.append("## Key Findings")
    lines.append("")
    
    if selected_layers:
        attn_layers = selected_layers.get("attn", [])
        mlp_layers = selected_layers.get("mlp", [])
        if attn_layers:
            lines.append(f"- **Important Attention Layers:** {attn_layers}")
        if mlp_layers:
            lines.append(f"- **Important MLP Layers:** {mlp_layers}")
    lines.append("")
    
    # Figures
    lines.append("## Figures")
    lines.append("")
    
    figure_files = [
        ("backtracking_rate_by_variant.png", "Backtracking Rate by Formatting Variant"),
        ("backtracking_vs_accuracy.png", "Backtracking vs Accuracy Correlation"),
        ("wait_logit_lens_bt_vs_control.png", "Logit Lens: Target Token Evidence by Layer"),
        ("ablation_importance_by_layer.png", "Ablation Importance by Layer"),
        ("backtracking_rate_by_condition.png", "Backtracking Rate: Baseline vs Ablation"),
        ("formatting_effect_on_backtracking.png", "Formatting Effect on Backtracking"),
    ]
    
    for filename, title in figure_files:
        fig_path = figures_dir / filename
        if fig_path.exists():
            # Use relative path from reports directory
            rel_path = f"../figures/{filename}"
            lines.append(f"### {title}")
            lines.append(f"![{title}]({rel_path})")
            lines.append("")
    
    # Detailed Metrics
    lines.append("## Detailed Metrics")
    lines.append("")
    
    if summary:
        lines.append("### Overall Statistics")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total Samples | {summary.get('total_samples', 'N/A')} |")
        lines.append(f"| Backtracking (Strict) | {summary.get('backtracking_strict_count', 'N/A')} ({summary.get('backtracking_strict_rate', 0)*100:.1f}%) |")
        lines.append(f"| Backtracking (Relaxed) | {summary.get('backtracking_relaxed_count', 'N/A')} ({summary.get('backtracking_relaxed_rate', 0)*100:.1f}%) |")
        lines.append(f"| Correct | {summary.get('correct_count', 'N/A')} |")
        lines.append(f"| Incorrect | {summary.get('incorrect_count', 'N/A')} |")
        lines.append("")
        
        # By variant
        by_variant = summary.get("by_variant", {})
        if by_variant:
            lines.append("### By Formatting Variant")
            lines.append("")
            lines.append("| Variant | Samples | BT Rate | Accuracy |")
            lines.append("|---------|---------|---------|----------|")
            for v, stats in by_variant.items():
                lines.append(f"| {v} | {stats['total']} | {stats['backtracking_rate']*100:.1f}% | {stats['accuracy']*100:.1f}% |")
            lines.append("")
        
        # By condition
        by_condition = summary.get("by_condition", {})
        if by_condition:
            lines.append("### By Condition")
            lines.append("")
            lines.append("| Condition | Samples | BT Rate | Accuracy |")
            lines.append("|-----------|---------|---------|----------|")
            for c, stats in by_condition.items():
                lines.append(f"| {c} | {stats['total']} | {stats['backtracking_rate']*100:.1f}% | {stats['accuracy']*100:.1f}% |")
            lines.append("")
    
    # Selected Layers
    if selected_layers:
        lines.append("## Selected Layers for Ablation")
        lines.append("")
        lines.append("These layers showed the strongest causal effect on backtracking trigger production:")
        lines.append("")
        lines.append("```json")
        import json
        lines.append(json.dumps(selected_layers, indent=2))
        lines.append("```")
        lines.append("")
    
    # Methodology
    lines.append("## Methodology")
    lines.append("")
    lines.append("1. **Baseline Generation:** Generated multiple samples per problem with standard sampling")
    lines.append("2. **Backtracking Detection:** Identified samples containing trigger phrases (Wait, Actually, Hold on)")
    lines.append("3. **Logit Lens Analysis:** Computed target token logit at each layer for backtracking vs position-matched controls")
    lines.append("4. **Ablation Scan:** Tested causal importance of each layer by ablating attention/MLP outputs")
    lines.append("5. **Generation Ablation:** Compared backtracking rates with targeted vs random layer ablation")
    lines.append("")
    
    # Caveats
    lines.append("## Caveats and Limitations")
    lines.append("")
    lines.append("- Results are specific to the model and dataset used")
    lines.append("- Backtracking detection relies on keyword matching, which may miss some instances")
    lines.append("- Ablation effects may interact non-linearly across layers")
    lines.append("- Position-matched controls help but don't perfectly isolate backtracking-specific effects")
    lines.append("")
    
    # Artifacts
    lines.append("## Artifacts")
    lines.append("")
    lines.append("### Analysis Files")
    lines.append(f"- `runs/{run_id}/analysis/backtracking_events.csv`")
    lines.append(f"- `runs/{run_id}/analysis/summary_metrics.json`")
    lines.append(f"- `runs/{run_id}/analysis/logit_lens.csv`")
    lines.append(f"- `runs/{run_id}/analysis/ablation_importance.csv`")
    lines.append(f"- `runs/{run_id}/analysis/selected_layers.json`")
    lines.append("")
    lines.append("### Figures")
    lines.append("- `figures/backtracking_rate_by_variant.png`")
    lines.append("- `figures/backtracking_vs_accuracy.png`")
    lines.append("- `figures/wait_logit_lens_bt_vs_control.png`")
    lines.append("- `figures/ablation_importance_by_layer.png`")
    lines.append("- `figures/backtracking_rate_by_condition.png`")
    lines.append("- `figures/formatting_effect_on_backtracking.png`")
    lines.append("")
    
    # Write report
    if output_path is None:
        output_path = get_reports_dir() / "backtracking_state_transition_report.md"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    
    print(f"Report written to: {output_path}")
    return output_path

