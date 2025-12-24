"""
Plotting utilities for backtracking analysis.

Generates proposal-ready figures and saves them to both
figures/ and runs/<run_id>/figures/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from backtracking.paths import get_figures_dir, get_run_figures_dir


def setup_plot_style():
    """Set up consistent plot styling."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def save_figure(fig: plt.Figure, name: str, run_id: str | None = None) -> list[Path]:
    """
    Save figure to figures/ and optionally to run folder.
    
    Args:
        fig: Matplotlib figure
        name: Figure filename (without extension)
        run_id: Optional run ID for run-specific copy
        
    Returns:
        List of paths where figure was saved
    """
    paths = []
    
    # Save to main figures directory
    main_path = get_figures_dir() / f"{name}.png"
    fig.savefig(main_path, dpi=150, bbox_inches='tight')
    paths.append(main_path)
    
    # Save to run folder if specified
    if run_id:
        run_path = get_run_figures_dir(run_id) / f"{name}.png"
        fig.savefig(run_path, dpi=150, bbox_inches='tight')
        paths.append(run_path)
    
    plt.close(fig)
    return paths


def plot_backtracking_rate_by_variant(
    summary: dict[str, Any],
    run_id: str | None = None,
) -> Path:
    """
    Plot backtracking rate by formatting variant.
    
    Args:
        summary: Summary metrics from detect-events
        run_id: Optional run ID
        
    Returns:
        Path to saved figure
    """
    setup_plot_style()
    
    by_variant = summary.get("by_variant", {})
    if not by_variant:
        return None
    
    variants = list(by_variant.keys())
    rates = [by_variant[v]["backtracking_rate"] * 100 for v in variants]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(variants, rates, color='steelblue', edgecolor='black')
    
    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=11)
    
    ax.set_xlabel('Formatting Variant')
    ax.set_ylabel('Backtracking Rate (%)')
    ax.set_title('Backtracking Rate by Formatting Variant')
    ax.set_ylim(0, max(rates) * 1.2 if rates else 100)
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    
    paths = save_figure(fig, "backtracking_rate_by_variant", run_id)
    return paths[0]


def plot_backtracking_vs_accuracy(
    summary: dict[str, Any],
    run_id: str | None = None,
) -> Path:
    """
    Plot backtracking rate vs accuracy correlation.
    
    Args:
        summary: Summary metrics
        run_id: Optional run ID
        
    Returns:
        Path to saved figure
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get accuracy with and without backtracking
    acc_with_bt = summary.get("accuracy_with_backtracking")
    acc_without_bt = summary.get("accuracy_without_backtracking")
    
    if acc_with_bt is not None and acc_without_bt is not None:
        categories = ['With Backtracking', 'Without Backtracking']
        accuracies = [acc_with_bt * 100, acc_without_bt * 100]
        colors = ['coral', 'mediumseagreen']
        
        bars = ax.bar(categories, accuracies, color=colors, edgecolor='black')
        
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=12)
        
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy: Backtracking vs Non-Backtracking Samples')
        ax.set_ylim(0, 100)
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title('Accuracy Comparison (No Data)')
    
    plt.tight_layout()
    
    paths = save_figure(fig, "backtracking_vs_accuracy", run_id)
    return paths[0]


def plot_logit_lens(
    logit_lens_data: list[dict[str, Any]],
    run_id: str | None = None,
) -> Path:
    """
    Plot logit lens results: target token logit by layer.
    
    Args:
        logit_lens_data: Per-layer logit lens results
        run_id: Optional run ID
        
    Returns:
        Path to saved figure
    """
    setup_plot_style()
    
    if not logit_lens_data:
        return None
    
    layers = [d["layer"] for d in logit_lens_data]
    bt_logits = [d["mean_target_logit_bt"] for d in logit_lens_data]
    ctrl_logits = [d["mean_target_logit_control"] for d in logit_lens_data]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(layers, bt_logits, 'o-', color='coral', linewidth=2, 
            markersize=6, label='Backtracking samples')
    ax.plot(layers, ctrl_logits, 's-', color='steelblue', linewidth=2,
            markersize=6, label='Control samples (position-matched)')
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Target Token Logit')
    ax.set_title('Logit Lens: Target Token Evidence by Layer\n(Backtracking vs Position-Matched Controls)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Highlight difference region
    ax.fill_between(layers, bt_logits, ctrl_logits, alpha=0.2, color='gray')
    
    plt.tight_layout()
    
    paths = save_figure(fig, "wait_logit_lens_bt_vs_control", run_id)
    return paths[0]


def plot_ablation_importance(
    importance_data: list[dict[str, Any]],
    run_id: str | None = None,
) -> Path:
    """
    Plot ablation importance by layer.
    
    Args:
        importance_data: Per-layer ablation importance results
        run_id: Optional run ID
        
    Returns:
        Path to saved figure
    """
    setup_plot_style()
    
    if not importance_data:
        return None
    
    # Separate by component
    attn_data = [d for d in importance_data if d["component"] == "attn"]
    mlp_data = [d for d in importance_data if d["component"] == "mlp"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot attention
    if attn_data:
        attn_data_sorted = sorted(attn_data, key=lambda x: x["layer"])
        layers = [d["layer"] for d in attn_data_sorted]
        deltas = [d["delta_wait_logit_mean"] for d in attn_data_sorted]
        
        colors = ['coral' if d < 0 else 'steelblue' for d in deltas]
        axes[0].bar(layers, deltas, color=colors, edgecolor='black', alpha=0.8)
        axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[0].set_xlabel('Layer')
        axes[0].set_ylabel('Δ Target Logit (ablated - baseline)')
        axes[0].set_title('Attention Ablation Impact')
    
    # Plot MLP
    if mlp_data:
        mlp_data_sorted = sorted(mlp_data, key=lambda x: x["layer"])
        layers = [d["layer"] for d in mlp_data_sorted]
        deltas = [d["delta_wait_logit_mean"] for d in mlp_data_sorted]
        
        colors = ['coral' if d < 0 else 'steelblue' for d in deltas]
        axes[1].bar(layers, deltas, color=colors, edgecolor='black', alpha=0.8)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].set_xlabel('Layer')
        axes[1].set_ylabel('Δ Target Logit (ablated - baseline)')
        axes[1].set_title('MLP Ablation Impact')
    
    fig.suptitle('Ablation Sensitivity: Which Layers Reduce Backtracking Trigger Logit?',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    paths = save_figure(fig, "ablation_importance_by_layer", run_id)
    return paths[0]


def plot_backtracking_by_condition(
    summary: dict[str, Any],
    run_id: str | None = None,
) -> Path:
    """
    Plot backtracking rate by condition (baseline vs ablation).
    
    Args:
        summary: Summary metrics
        run_id: Optional run ID
        
    Returns:
        Path to saved figure
    """
    setup_plot_style()
    
    by_condition = summary.get("by_condition", {})
    if not by_condition:
        return None
    
    # Order: baseline, targeted_ablation, random_ablation
    order = ["baseline", "targeted_ablation", "random_ablation"]
    conditions = [c for c in order if c in by_condition]
    rates = [by_condition[c]["backtracking_rate"] * 100 for c in conditions]
    
    # Prettier labels
    labels = {
        "baseline": "Baseline",
        "targeted_ablation": "Targeted Ablation",
        "random_ablation": "Random Ablation",
    }
    display_names = [labels.get(c, c) for c in conditions]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['steelblue', 'coral', 'mediumseagreen'][:len(conditions)]
    bars = ax.bar(display_names, rates, color=colors, edgecolor='black')
    
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=12)
    
    ax.set_ylabel('Backtracking Rate (%)')
    ax.set_title('Backtracking Rate: Baseline vs Ablation Conditions')
    ax.set_ylim(0, max(rates) * 1.3 if rates else 100)
    
    plt.tight_layout()
    
    paths = save_figure(fig, "backtracking_rate_by_condition", run_id)
    return paths[0]


def plot_formatting_effect(
    formatting_summary: list[dict[str, Any]],
    run_id: str | None = None,
) -> Path:
    """
    Plot effect of formatting variants on backtracking.
    
    Args:
        formatting_summary: Per-variant statistics
        run_id: Optional run ID
        
    Returns:
        Path to saved figure
    """
    setup_plot_style()
    
    if not formatting_summary:
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    variants = [d["variant"] for d in formatting_summary]
    bt_rates = [d["backtracking_rate"] * 100 for d in formatting_summary]
    accuracies = [d["accuracy"] * 100 if d["accuracy"] else 0 for d in formatting_summary]
    
    # Backtracking rate
    bars1 = axes[0].bar(variants, bt_rates, color='coral', edgecolor='black')
    for bar, rate in zip(bars1, bt_rates):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
    axes[0].set_ylabel('Backtracking Rate (%)')
    axes[0].set_title('Backtracking Rate by Format')
    axes[0].tick_params(axis='x', rotation=15)
    
    # Accuracy
    bars2 = axes[1].bar(variants, accuracies, color='steelblue', edgecolor='black')
    for bar, acc in zip(bars2, accuracies):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy by Format')
    axes[1].tick_params(axis='x', rotation=15)
    
    fig.suptitle('Effect of Formatting Variants on Backtracking and Accuracy',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    paths = save_figure(fig, "formatting_effect_on_backtracking", run_id)
    return paths[0]


