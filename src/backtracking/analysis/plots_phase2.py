"""
Phase 2: Plotting functions for Phase 2 analysis.

Generates figures for subset sweeps, scale tradeoffs, continuation ablation,
and deconfounded baseline metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from backtracking.paths import get_figures_dir, get_run_figures_dir


def save_figure(fig: plt.Figure, name: str, run_id: str) -> None:
    """Save figure to both run-specific and proposal-ready directories."""
    # Save to run figures
    run_fig_dir = get_run_figures_dir(run_id)
    fig.savefig(run_fig_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    
    # Save to proposal-ready figures
    fig_dir = get_figures_dir()
    fig.savefig(fig_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    
    plt.close(fig)


def plot_subset_sweep_comparison(
    results: list[dict[str, Any]],
    run_id: str,
) -> Path:
    """
    Plot subset sweep comparison: backtracking rate and accuracy by subset.
    
    Args:
        results: List of subset sweep results
        run_id: Run identifier
        
    Returns:
        Path to saved figure
    """
    if not results:
        return None
    
    # Extract data
    subsets = [r["subset_name"] for r in results]
    bt_rates = [r["backtracking_rate"] * 100 for r in results]
    accuracies = [r["accuracy"] * 100 for r in results]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(subsets))
    width = 0.6
    
    # Backtracking rate
    colors = ["#4CAF50" if s == "baseline" else "#2196F3" for s in subsets]
    bars1 = ax1.bar(x, bt_rates, width, color=colors, alpha=0.8)
    ax1.set_ylabel("Backtracking Rate (%)", fontsize=12)
    ax1.set_xlabel("Ablation Subset", fontsize=12)
    ax1.set_title("Backtracking Rate by Ablation Subset", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(subsets, rotation=45, ha="right")
    ax1.axhline(y=bt_rates[0], color="red", linestyle="--", alpha=0.5, label="Baseline")
    ax1.legend()
    
    # Add value labels
    for bar, val in zip(bars1, bt_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    
    # Accuracy
    bars2 = ax2.bar(x, accuracies, width, color=colors, alpha=0.8)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_xlabel("Ablation Subset", fontsize=12)
    ax2.set_title("Accuracy by Ablation Subset", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(subsets, rotation=45, ha="right")
    ax2.axhline(y=accuracies[0], color="red", linestyle="--", alpha=0.5, label="Baseline")
    ax2.legend()
    
    # Add value labels
    for bar, val in zip(bars2, accuracies):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    
    plt.tight_layout()
    save_figure(fig, "phase2_subset_sweep", run_id)
    
    return get_figures_dir() / "phase2_subset_sweep.png"


def plot_scale_tradeoff_curve(
    results: list[dict[str, Any]],
    run_id: str,
) -> Path:
    """
    Plot scale sweep tradeoff curve: backtracking rate vs accuracy.
    
    Args:
        results: List of scale sweep results
        run_id: Run identifier
        
    Returns:
        Path to saved figure
    """
    if not results:
        return None
    
    # Extract data
    scales = [r["scale"] for r in results]
    bt_rates = [r["backtracking_rate"] * 100 for r in results]
    accuracies = [r["accuracy"] * 100 for r in results]
    
    subset_name = results[0].get("subset_name", "unknown")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: BT rate and accuracy vs scale
    ax1.plot(scales, bt_rates, "o-", color="#E53935", linewidth=2, markersize=8, label="Backtracking Rate")
    ax1.set_xlabel("Scale Factor", fontsize=12)
    ax1.set_ylabel("Backtracking Rate (%)", fontsize=12, color="#E53935")
    ax1.tick_params(axis="y", labelcolor="#E53935")
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(scales, accuracies, "s-", color="#1E88E5", linewidth=2, markersize=8, label="Accuracy")
    ax1_twin.set_ylabel("Accuracy (%)", fontsize=12, color="#1E88E5")
    ax1_twin.tick_params(axis="y", labelcolor="#1E88E5")
    
    ax1.set_title(f"Scale Sweep: {subset_name}", fontsize=14)
    ax1.legend(loc="upper left")
    ax1_twin.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Tradeoff curve (BT rate vs accuracy)
    ax2.scatter(bt_rates, accuracies, c=scales, cmap="viridis", s=100, edgecolors="black", linewidth=1)
    
    # Add annotations for each point
    for scale, bt, acc in zip(scales, bt_rates, accuracies):
        ax2.annotate(f"s={scale}", (bt, acc), textcoords="offset points",
                    xytext=(5, 5), fontsize=9)
    
    ax2.set_xlabel("Backtracking Rate (%)", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Tradeoff: Backtracking vs Accuracy", fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar.set_label("Scale Factor", fontsize=10)
    
    plt.tight_layout()
    save_figure(fig, "phase2_scale_tradeoff", run_id)
    
    return get_figures_dir() / "phase2_scale_tradeoff.png"


def plot_continuation_ablation_effect(
    results: list[dict[str, Any]],
    run_id: str,
) -> Path:
    """
    Plot continuation ablation effect on onset token probability.
    
    Args:
        results: List of continuation ablation results
        run_id: Run identifier
        
    Returns:
        Path to saved figure
    """
    if not results:
        return None
    
    # Group by condition and scale
    grouped = {}
    for r in results:
        key = (r.get("condition", "unknown"), r.get("scale", 1.0))
        if key not in grouped:
            grouped[key] = {"probs": [], "logits": []}
        
        if "onset_prob" in r and r["onset_prob"] is not None:
            grouped[key]["probs"].append(r["onset_prob"])
        if "onset_logit" in r and r["onset_logit"] is not None:
            grouped[key]["logits"].append(r["onset_logit"])
    
    if not grouped:
        return None
    
    # Compute means
    labels = []
    mean_probs = []
    mean_logits = []
    
    for key in sorted(grouped.keys()):
        cond, scale = key
        data = grouped[key]
        if data["probs"]:
            labels.append(f"{cond}\ns={scale}")
            mean_probs.append(np.mean(data["probs"]) * 100)
            mean_logits.append(np.mean(data["logits"]))
    
    if not labels:
        return None
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.arange(len(labels))
    width = 0.6
    
    # Probability plot
    colors = ["#4CAF50" if "baseline" in l else "#FF9800" for l in labels]
    bars1 = ax1.bar(x, mean_probs, width, color=colors, alpha=0.8)
    ax1.set_ylabel("P(onset_token) (%)", fontsize=12)
    ax1.set_xlabel("Condition", fontsize=12)
    ax1.set_title("Onset Token Probability Under Ablation", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    
    for bar, val in zip(bars1, mean_probs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=9)
    
    # Logit plot
    bars2 = ax2.bar(x, mean_logits, width, color=colors, alpha=0.8)
    ax2.set_ylabel("Mean Logit", fontsize=12)
    ax2.set_xlabel("Condition", fontsize=12)
    ax2.set_title("Onset Token Logit Under Ablation", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    
    for bar, val in zip(bars2, mean_logits):
        y_offset = 0.2 if val >= 0 else -0.5
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_offset,
                f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=9)
    
    plt.tight_layout()
    save_figure(fig, "phase2_continuation_effect", run_id)
    
    return get_figures_dir() / "phase2_continuation_effect.png"


def plot_baseline_only_summary(
    metrics: dict[str, Any],
    run_id: str,
) -> Path:
    """
    Plot deconfounded baseline-only summary metrics.
    
    Args:
        metrics: Metrics dictionary from metrics_v2
        run_id: Run identifier
        
    Returns:
        Path to saved figure
    """
    if not metrics:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Baseline summary
    ax1 = axes[0, 0]
    baseline = metrics.get("baseline_only_summary", {})
    
    if baseline and "error" not in baseline:
        labels = ["BT Rate", "Accuracy", "Acc w/ BT", "Acc w/o BT"]
        values = [
            baseline.get("backtracking_strict_rate", 0) * 100,
            (baseline.get("accuracy") or 0) * 100,
            (baseline.get("accuracy_with_backtracking") or 0) * 100,
            (baseline.get("accuracy_without_backtracking") or 0) * 100,
        ]
        colors = ["#2196F3", "#4CAF50", "#8BC34A", "#CDDC39"]
        bars = ax1.bar(labels, values, color=colors, alpha=0.8)
        ax1.set_ylabel("Percentage (%)", fontsize=11)
        ax1.set_title("Baseline-Only Summary (Deconfounded)", fontsize=12)
        
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
    else:
        ax1.text(0.5, 0.5, "No baseline data", ha="center", va="center", fontsize=12)
        ax1.set_title("Baseline-Only Summary", fontsize=12)
    
    # 2. Backtracking vs accuracy comparison
    ax2 = axes[0, 1]
    bt_acc = metrics.get("backtracking_vs_accuracy", {})
    
    if bt_acc and "error" not in bt_acc:
        with_bt = bt_acc.get("with_backtracking", {})
        without_bt = bt_acc.get("without_backtracking", {})
        
        labels = ["With Backtracking", "Without Backtracking"]
        counts = [with_bt.get("count", 0), without_bt.get("count", 0)]
        accs = [(with_bt.get("accuracy") or 0) * 100, (without_bt.get("accuracy") or 0) * 100]
        
        x = np.arange(len(labels))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, counts, width, label="Count", color="#42A5F5", alpha=0.8)
        ax2_twin = ax2.twinx()
        bars2 = ax2_twin.bar(x + width/2, accs, width, label="Accuracy", color="#66BB6A", alpha=0.8)
        
        ax2.set_ylabel("Count", fontsize=11, color="#42A5F5")
        ax2_twin.set_ylabel("Accuracy (%)", fontsize=11, color="#66BB6A")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.set_title("Backtracking vs Accuracy (Baseline Only)", fontsize=12)
        ax2.legend(loc="upper left")
        ax2_twin.legend(loc="upper right")
        
        lift = bt_acc.get("accuracy_lift_ratio")
        if lift:
            ax2.text(0.5, -0.15, f"Accuracy Lift: {lift:.2f}x", 
                    transform=ax2.transAxes, ha="center", fontsize=11, fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
        ax2.set_title("Backtracking vs Accuracy", fontsize=12)
    
    # 3. Formatting effect (baseline only)
    ax3 = axes[1, 0]
    fmt = metrics.get("formatting_effect", {})
    
    if fmt and "error" not in fmt:
        by_variant = fmt.get("by_variant", {})
        if by_variant:
            variants = list(by_variant.keys())
            bt_rates = [by_variant[v].get("backtracking_rate", 0) * 100 for v in variants]
            
            colors = plt.cm.Set2(np.linspace(0, 1, len(variants)))
            bars = ax3.bar(variants, bt_rates, color=colors, alpha=0.8)
            ax3.set_ylabel("Backtracking Rate (%)", fontsize=11)
            ax3.set_xlabel("Formatting Variant", fontsize=11)
            ax3.set_title("Format Effect on Backtracking (Baseline Only)", fontsize=12)
            ax3.set_xticklabels(variants, rotation=30, ha="right")
            
            for bar, val in zip(bars, bt_rates):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    else:
        ax3.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
        ax3.set_title("Formatting Effect", fontsize=12)
    
    # 4. Variant-specific condition comparison
    ax4 = axes[1, 1]
    var_spec = metrics.get("variant_specific", {})
    
    if var_spec and "error" not in var_spec:
        by_cond = var_spec.get("by_condition", {})
        if by_cond:
            conds = list(by_cond.keys())
            bt_rates = [by_cond[c].get("backtracking_rate", 0) * 100 for c in conds]
            accs = [by_cond[c].get("accuracy", 0) * 100 for c in conds]
            
            x = np.arange(len(conds))
            width = 0.35
            
            bars1 = ax4.bar(x - width/2, bt_rates, width, label="BT Rate", color="#EF5350", alpha=0.8)
            bars2 = ax4.bar(x + width/2, accs, width, label="Accuracy", color="#42A5F5", alpha=0.8)
            
            ax4.set_ylabel("Percentage (%)", fontsize=11)
            ax4.set_xlabel("Condition", fontsize=11)
            ax4.set_xticks(x)
            ax4.set_xticklabels(conds, rotation=30, ha="right")
            ax4.set_title(f"Condition Comparison: {var_spec.get('variant', '')}", fontsize=12)
            ax4.legend()
    else:
        ax4.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
        ax4.set_title("Condition Comparison", fontsize=12)
    
    plt.tight_layout()
    save_figure(fig, "phase2_baseline_summary", run_id)
    
    return get_figures_dir() / "phase2_baseline_summary.png"


def generate_all_phase2_plots(
    subset_results: list[dict[str, Any]] | None,
    scale_results: list[dict[str, Any]] | None,
    continuation_results: list[dict[str, Any]] | None,
    metrics_v2: dict[str, Any] | None,
    run_id: str,
) -> dict[str, Path]:
    """
    Generate all Phase 2 plots.
    
    Args:
        subset_results: Results from subset sweep
        scale_results: Results from scale sweep
        continuation_results: Results from continuation ablation
        metrics_v2: Deconfounded metrics
        run_id: Run identifier
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    paths = {}
    
    if subset_results:
        paths["subset_sweep"] = plot_subset_sweep_comparison(subset_results, run_id)
    
    if scale_results:
        paths["scale_tradeoff"] = plot_scale_tradeoff_curve(scale_results, run_id)
    
    if continuation_results:
        paths["continuation_effect"] = plot_continuation_ablation_effect(continuation_results, run_id)
    
    if metrics_v2:
        paths["baseline_summary"] = plot_baseline_only_summary(metrics_v2, run_id)
    
    return paths

