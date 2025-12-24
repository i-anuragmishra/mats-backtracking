"""
Phase 2: Deconfounded metrics computation.

Provides baseline-only metrics to avoid confounding ablation conditions
with baseline behavior in aggregate statistics.
"""

from __future__ import annotations

from typing import Any

from backtracking.io import write_json
from backtracking.paths import get_metrics_v2_file


def compute_baseline_only_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute metrics using only baseline condition events.
    
    This removes the confound of mixing ablation conditions (which have
    collapsed accuracy) with baseline behavior in aggregate stats.
    
    Args:
        events: List of all event records
        
    Returns:
        Dictionary with baseline-only metrics
    """
    # Filter to baseline only
    baseline_events = [e for e in events if e.get("condition") == "baseline"]
    
    if not baseline_events:
        return {"error": "no baseline events found"}
    
    total = len(baseline_events)
    
    # Backtracking rates
    bt_strict = sum(1 for e in baseline_events if e.get("has_backtracking_strict"))
    bt_relaxed = sum(1 for e in baseline_events if e.get("has_backtracking_relaxed"))
    
    # Accuracy
    correct = [e for e in baseline_events if e.get("is_correct") is True]
    incorrect = [e for e in baseline_events if e.get("is_correct") is False]
    unknown = [e for e in baseline_events if e.get("is_correct") is None]
    
    # Accuracy by backtracking status (within baseline)
    bt_events = [e for e in baseline_events if e.get("has_backtracking_strict")]
    non_bt_events = [e for e in baseline_events if not e.get("has_backtracking_strict")]
    
    bt_correct = sum(1 for e in bt_events if e.get("is_correct") is True)
    non_bt_correct = sum(1 for e in non_bt_events if e.get("is_correct") is True)
    
    return {
        "condition": "baseline_only",
        "total_samples": total,
        "backtracking_strict_count": bt_strict,
        "backtracking_strict_rate": bt_strict / total if total > 0 else 0,
        "backtracking_relaxed_count": bt_relaxed,
        "backtracking_relaxed_rate": bt_relaxed / total if total > 0 else 0,
        "correct_count": len(correct),
        "incorrect_count": len(incorrect),
        "unknown_count": len(unknown),
        "accuracy": len(correct) / (len(correct) + len(incorrect)) if (correct or incorrect) else None,
        "accuracy_with_backtracking": bt_correct / len(bt_events) if bt_events else None,
        "accuracy_without_backtracking": non_bt_correct / len(non_bt_events) if non_bt_events else None,
    }


def compute_variant_specific_metrics(
    events: list[dict[str, Any]],
    variant: str,
) -> dict[str, Any]:
    """
    Compute metrics for a specific variant, separated by condition.
    
    Args:
        events: List of all event records
        variant: Variant name to filter on
        
    Returns:
        Dictionary with variant-specific metrics by condition
    """
    variant_events = [e for e in events if e.get("variant") == variant]
    
    if not variant_events:
        return {"error": f"no events for variant {variant}"}
    
    # Get all conditions present
    conditions = list(set(e.get("condition") for e in variant_events))
    
    by_condition = {}
    for cond in conditions:
        cond_events = [e for e in variant_events if e.get("condition") == cond]
        total = len(cond_events)
        bt_count = sum(1 for e in cond_events if e.get("has_backtracking_strict"))
        correct = sum(1 for e in cond_events if e.get("is_correct") is True)
        
        by_condition[cond] = {
            "total": total,
            "backtracking_count": bt_count,
            "backtracking_rate": bt_count / total if total > 0 else 0,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0,
        }
    
    return {
        "variant": variant,
        "total": len(variant_events),
        "by_condition": by_condition,
    }


def compute_backtracking_vs_accuracy_baseline(
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute backtracking vs accuracy correlation within baseline only.
    
    This is the key deconfounded metric: shows whether backtracking
    improves accuracy without mixing in ablation-induced failures.
    
    Args:
        events: List of all event records
        
    Returns:
        Dictionary with correlation metrics
    """
    baseline_events = [e for e in events if e.get("condition") == "baseline"]
    
    if not baseline_events:
        return {"error": "no baseline events"}
    
    # Split by backtracking
    bt_events = [e for e in baseline_events if e.get("has_backtracking_strict")]
    non_bt_events = [e for e in baseline_events if not e.get("has_backtracking_strict")]
    
    # Compute accuracies
    bt_correct = sum(1 for e in bt_events if e.get("is_correct") is True)
    non_bt_correct = sum(1 for e in non_bt_events if e.get("is_correct") is True)
    
    bt_accuracy = bt_correct / len(bt_events) if bt_events else None
    non_bt_accuracy = non_bt_correct / len(non_bt_events) if non_bt_events else None
    
    # Compute lift
    if bt_accuracy is not None and non_bt_accuracy is not None and non_bt_accuracy > 0:
        accuracy_lift = bt_accuracy / non_bt_accuracy
    else:
        accuracy_lift = None
    
    return {
        "baseline_only": True,
        "with_backtracking": {
            "count": len(bt_events),
            "correct": bt_correct,
            "accuracy": bt_accuracy,
        },
        "without_backtracking": {
            "count": len(non_bt_events),
            "correct": non_bt_correct,
            "accuracy": non_bt_accuracy,
        },
        "accuracy_lift_ratio": accuracy_lift,
        "accuracy_difference": (bt_accuracy - non_bt_accuracy) if (bt_accuracy is not None and non_bt_accuracy is not None) else None,
    }


def compute_formatting_baseline_only(
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute formatting effect using baseline-only events.
    
    Args:
        events: List of all event records
        
    Returns:
        Dictionary with formatting metrics by variant
    """
    baseline_events = [e for e in events if e.get("condition") == "baseline"]
    
    if not baseline_events:
        return {"error": "no baseline events"}
    
    # Get all variants
    variants = list(set(e.get("variant") for e in baseline_events))
    
    by_variant = {}
    for var in variants:
        var_events = [e for e in baseline_events if e.get("variant") == var]
        total = len(var_events)
        bt_count = sum(1 for e in var_events if e.get("has_backtracking_strict"))
        correct = sum(1 for e in var_events if e.get("is_correct") is True)
        
        by_variant[var] = {
            "total": total,
            "backtracking_count": bt_count,
            "backtracking_rate": bt_count / total if total > 0 else 0,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0,
        }
    
    return {
        "baseline_only": True,
        "by_variant": by_variant,
    }


def compute_all_v2_metrics(
    events: list[dict[str, Any]],
    variant: str = "baseline_think_newline",
) -> dict[str, Any]:
    """
    Compute all Phase 2 deconfounded metrics.
    
    Args:
        events: List of all event records
        variant: Primary variant for condition comparison
        
    Returns:
        Complete metrics dictionary
    """
    return {
        "baseline_only_summary": compute_baseline_only_metrics(events),
        "backtracking_vs_accuracy": compute_backtracking_vs_accuracy_baseline(events),
        "formatting_effect": compute_formatting_baseline_only(events),
        "variant_specific": compute_variant_specific_metrics(events, variant),
    }


def save_metrics_v2(
    events: list[dict[str, Any]],
    run_id: str,
    variant: str = "baseline_think_newline",
) -> dict[str, Any]:
    """
    Compute and save Phase 2 metrics to file.
    
    Args:
        events: List of all event records
        run_id: Run identifier
        variant: Primary variant for condition comparison
        
    Returns:
        Complete metrics dictionary
    """
    metrics = compute_all_v2_metrics(events, variant)
    output_path = get_metrics_v2_file(run_id)
    write_json(metrics, output_path)
    return metrics

