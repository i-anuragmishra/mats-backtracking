"""
Event processing for backtracking detection.

Processes generation files into backtracking events CSV and summary metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

from backtracking.config import ExperimentConfig
from backtracking.detect import (
    BacktrackingDetection,
    check_answer_correct,
    detect_backtracking,
    extract_final_answer,
)
from backtracking.io import read_jsonl, write_csv, write_json
from backtracking.paths import (
    get_events_file,
    get_generation_file,
    get_generations_dir,
    get_summary_metrics_file,
)
from backtracking.prompts import list_variants

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


def process_single_generation(
    gen: dict[str, Any],
    gold_answer: str | None,
    tokenizer: "PreTrainedTokenizer",
    config: ExperimentConfig,
) -> dict[str, Any]:
    """
    Process a single generation into an event record.
    
    Args:
        gen: Generation record from JSONL
        gold_answer: Gold answer for correctness check
        tokenizer: HuggingFace tokenizer
        config: Experiment configuration
        
    Returns:
        Event record dictionary
    """
    # Detect backtracking
    detection = detect_backtracking(
        completion=gen["completion"],
        prompt=gen["prompt"],
        full_text=gen["full_text"],
        tokenizer=tokenizer,
        config=config.detection,
    )
    
    # Extract predicted answer
    predicted = extract_final_answer(
        gen["completion"],
        config.detection.answer_regex,
    )
    
    # Check correctness
    is_correct = check_answer_correct(predicted, gold_answer)
    
    return {
        "example_id": gen["example_id"],
        "sample_id": gen["sample_id"],
        "variant": gen["variant"],
        "condition": gen["condition"],
        "has_backtracking_strict": detection.has_backtracking_strict,
        "has_backtracking_relaxed": detection.has_backtracking_relaxed,
        "onset_phrase": detection.onset_phrase or "",
        "onset_char": detection.onset_char,
        "onset_token_in_completion": detection.onset_token_in_completion,
        "onset_token_in_full": detection.onset_token_in_full,
        "onset_token_id": detection.onset_token_id,
        "onset_token_text": detection.onset_token_text,
        "pred_pos_in_full": detection.pred_pos_in_full,
        "predicted_answer": predicted or "",
        "is_correct": is_correct,
        "prompt_tokens": gen["prompt_tokens"],
        "completion_tokens": gen["completion_tokens"],
    }


def process_generations_to_events(
    run_id: str,
    config: ExperimentConfig,
    tokenizer: "PreTrainedTokenizer",
    dataset: list[dict[str, Any]],
    variants: list[str] | None = None,
    conditions: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Process all generations for a run into events.
    
    Args:
        run_id: Run identifier
        config: Experiment configuration
        tokenizer: HuggingFace tokenizer
        dataset: Dataset with gold answers
        variants: Variants to process (None = all)
        conditions: Conditions to process (None = all found)
        
    Returns:
        Tuple of (events list, summary metrics dict)
    """
    # Build answer lookup
    answers = {ex["id"]: ex.get("answer_value") for ex in dataset}
    
    # Get variants to process
    if variants is None:
        variants = list_variants(config.prompting)
    
    # Default conditions
    if conditions is None:
        conditions = ["baseline", "targeted_ablation", "random_ablation"]
    
    all_events = []
    
    for variant in variants:
        for condition in conditions:
            gen_file = get_generation_file(run_id, variant, condition)
            
            if not gen_file.exists():
                continue
            
            generations = read_jsonl(gen_file)
            
            for gen in tqdm(generations, desc=f"Processing {variant}/{condition}"):
                gold = answers.get(gen["example_id"])
                event = process_single_generation(gen, gold, tokenizer, config)
                all_events.append(event)
    
    # Compute summary metrics
    summary = compute_summary_metrics(all_events)
    
    # Save outputs
    events_file = get_events_file(run_id)
    write_csv(all_events, events_file)
    
    summary_file = get_summary_metrics_file(run_id)
    write_json(summary, summary_file)
    
    return all_events, summary


def compute_summary_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute summary metrics from events.
    
    Args:
        events: List of event records
        
    Returns:
        Summary metrics dictionary
    """
    if not events:
        return {"error": "no events"}
    
    total = len(events)
    
    # Backtracking rates
    bt_strict = sum(1 for e in events if e["has_backtracking_strict"])
    bt_relaxed = sum(1 for e in events if e["has_backtracking_relaxed"])
    
    # Accuracy
    correct = [e for e in events if e["is_correct"] is True]
    incorrect = [e for e in events if e["is_correct"] is False]
    unknown = [e for e in events if e["is_correct"] is None]
    
    # Accuracy by backtracking status
    bt_events = [e for e in events if e["has_backtracking_strict"]]
    non_bt_events = [e for e in events if not e["has_backtracking_strict"]]
    
    bt_correct = sum(1 for e in bt_events if e["is_correct"] is True)
    non_bt_correct = sum(1 for e in non_bt_events if e["is_correct"] is True)
    
    # By variant
    variants = list(set(e["variant"] for e in events))
    variant_stats = {}
    for v in variants:
        v_events = [e for e in events if e["variant"] == v]
        v_bt = sum(1 for e in v_events if e["has_backtracking_strict"])
        v_correct = sum(1 for e in v_events if e["is_correct"] is True)
        variant_stats[v] = {
            "total": len(v_events),
            "backtracking_strict": v_bt,
            "backtracking_rate": v_bt / len(v_events) if v_events else 0,
            "correct": v_correct,
            "accuracy": v_correct / len(v_events) if v_events else 0,
        }
    
    # By condition
    conditions = list(set(e["condition"] for e in events))
    condition_stats = {}
    for c in conditions:
        c_events = [e for e in events if e["condition"] == c]
        c_bt = sum(1 for e in c_events if e["has_backtracking_strict"])
        c_correct = sum(1 for e in c_events if e["is_correct"] is True)
        condition_stats[c] = {
            "total": len(c_events),
            "backtracking_strict": c_bt,
            "backtracking_rate": c_bt / len(c_events) if c_events else 0,
            "correct": c_correct,
            "accuracy": c_correct / len(c_events) if c_events else 0,
        }
    
    return {
        "total_samples": total,
        "backtracking_strict_count": bt_strict,
        "backtracking_strict_rate": bt_strict / total,
        "backtracking_relaxed_count": bt_relaxed,
        "backtracking_relaxed_rate": bt_relaxed / total,
        "correct_count": len(correct),
        "incorrect_count": len(incorrect),
        "unknown_count": len(unknown),
        "accuracy": len(correct) / (len(correct) + len(incorrect)) if (correct or incorrect) else None,
        "accuracy_with_backtracking": bt_correct / len(bt_events) if bt_events else None,
        "accuracy_without_backtracking": non_bt_correct / len(non_bt_events) if non_bt_events else None,
        "by_variant": variant_stats,
        "by_condition": condition_stats,
    }


def get_backtracking_events(
    events: list[dict[str, Any]],
    strict: bool = True,
    max_events: int | None = None,
) -> list[dict[str, Any]]:
    """
    Filter to only backtracking events.
    
    Args:
        events: All events
        strict: Use strict detection (vs relaxed)
        max_events: Maximum number to return
        
    Returns:
        Filtered list of backtracking events
    """
    key = "has_backtracking_strict" if strict else "has_backtracking_relaxed"
    bt_events = [e for e in events if e[key]]
    
    if max_events and len(bt_events) > max_events:
        bt_events = bt_events[:max_events]
    
    return bt_events


def get_non_backtracking_events(
    events: list[dict[str, Any]],
    strict: bool = True,
) -> list[dict[str, Any]]:
    """
    Filter to only non-backtracking events.
    
    Args:
        events: All events
        strict: Use strict detection (vs relaxed)
        
    Returns:
        Filtered list of non-backtracking events
    """
    key = "has_backtracking_strict" if strict else "has_backtracking_relaxed"
    return [e for e in events if not e[key]]


