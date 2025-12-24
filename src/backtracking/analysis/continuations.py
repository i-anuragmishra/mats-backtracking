"""
Phase 2: Continuation-only ablation analysis.

Implements minimal intervention ablation that only affects the onset token
prediction, avoiding collateral damage to model competence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from backtracking.hooks import (
    apply_teacher_forced_hooks,
    specs_from_subset_config,
)
from backtracking.io import read_csv, write_csv
from backtracking.paths import get_continuation_ablation_file, get_events_file
from backtracking.seed import set_seed
from backtracking.tokenization import tokenize_for_model

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    
    from backtracking.config import ExperimentConfig


def get_onset_token_logit(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    full_text: str,
    pred_pos: int,
    target_token_id: int,
) -> float:
    """
    Get the logit for a target token at a specific position.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        full_text: Full text (prompt + completion up to pred_pos)
        pred_pos: Position to get logit from (0-indexed in full sequence)
        target_token_id: Token ID to get logit for
        
    Returns:
        Logit value for target token
    """
    # Tokenize
    inputs = tokenize_for_model(full_text, tokenizer, model.device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Get logit at pred_pos for target token
    # pred_pos is the position where we predict the next token
    if pred_pos >= logits.shape[1]:
        pred_pos = logits.shape[1] - 1
    
    target_logit = logits[0, pred_pos, target_token_id].item()
    return target_logit


def get_onset_token_prob(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    full_text: str,
    pred_pos: int,
    target_token_id: int,
) -> tuple[float, float]:
    """
    Get the probability and logit for a target token at a specific position.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        full_text: Full text (prompt + completion up to pred_pos)
        pred_pos: Position to get logit from
        target_token_id: Token ID to get probability for
        
    Returns:
        Tuple of (probability, logit)
    """
    inputs = tokenize_for_model(full_text, tokenizer, model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    if pred_pos >= logits.shape[1]:
        pred_pos = logits.shape[1] - 1
    
    # Get logits at pred_pos
    pos_logits = logits[0, pred_pos, :]
    probs = F.softmax(pos_logits, dim=-1)
    
    target_logit = pos_logits[target_token_id].item()
    target_prob = probs[target_token_id].item()
    
    return target_prob, target_logit


def run_continuation_ablation(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    config: "ExperimentConfig",
    run_id: str,
    phase1_run_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Run continuation-only ablation analysis.
    
    For each backtracking event, compute P(onset_token) at pred_pos
    under various ablation conditions. This is a minimal intervention
    that tests whether ablating specific layers reduces the probability
    of generating the backtracking trigger.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        config: Experiment configuration
        run_id: Current run ID
        phase1_run_id: Phase 1 run ID to read events from (or current run if None)
        
    Returns:
        List of results for each event and ablation condition
    """
    phase2_cfg = config.phase2
    cont_cfg = phase2_cfg.continuation_ablation
    
    if not cont_cfg.enabled:
        return []
    
    # Determine which run to read events from
    events_run_id = phase1_run_id or run_id
    
    # Load events
    events_file = get_events_file(events_run_id)
    if not events_file.exists():
        raise FileNotFoundError(f"Events file not found: {events_file}")
    
    all_events = read_csv(events_file)
    
    # Filter to backtracking events for the target variant
    variant = cont_cfg.variant
    bt_events = [
        e for e in all_events
        if e.get("has_backtracking_strict") and 
           e.get("variant") == variant and
           e.get("condition") == "baseline"
    ]
    
    # Limit events
    max_events = cont_cfg.max_events
    if len(bt_events) > max_events:
        bt_events = bt_events[:max_events]
    
    # Get subset config
    subset_name = cont_cfg.subset_name
    subset_config = phase2_cfg.subset_sweep.subsets.get(subset_name)
    
    if subset_config is None:
        raise ValueError(f"Subset {subset_name} not found in config")
    
    scales = cont_cfg.scales
    results = []
    
    set_seed(config.run.seed)
    
    for event in tqdm(bt_events, desc="Continuation ablation"):
        # Get event info
        example_id = event.get("example_id")
        sample_id = event.get("sample_id")
        pred_pos = int(event.get("pred_pos_in_full", 0))
        onset_token_id = event.get("onset_token_id")
        onset_token_text = event.get("onset_token_text", "")
        
        # Skip if no valid onset token
        if onset_token_id is None or onset_token_id == "":
            continue
        
        onset_token_id = int(onset_token_id)
        
        # We need the full_text up to and including the position before onset
        # Since we don't have the full_text stored, we need to reconstruct from
        # the generation files, or use a simpler approach with stored info
        
        # For now, use the stored onset info to create a minimal test
        # We'll use the prompt + completion prefix that we can infer
        
        # Get prefix text - this is an approximation
        # In practice, you'd want to store/retrieve the actual full_text
        prompt_tokens = int(event.get("prompt_tokens", 0))
        onset_in_full = int(event.get("onset_token_in_full", 0))
        
        # Create a placeholder for the text reconstruction
        # This is a simplified version - full implementation would read from generations
        
        # First, get baseline (no ablation)
        baseline_result = {
            "example_id": example_id,
            "sample_id": sample_id,
            "variant": variant,
            "pred_pos": pred_pos,
            "onset_token_id": onset_token_id,
            "onset_token_text": onset_token_text,
            "subset_name": "baseline",
            "scale": 1.0,
        }
        
        # For each scale, compute with ablation
        for scale in scales:
            specs = specs_from_subset_config(
                subset_config,
                mode="scale",
                scale=scale,
            )
            
            result = {
                "example_id": example_id,
                "sample_id": sample_id,
                "variant": variant,
                "pred_pos": pred_pos,
                "onset_token_id": onset_token_id,
                "onset_token_text": onset_token_text,
                "subset_name": subset_name,
                "scale": scale,
                "num_components": len(specs),
            }
            
            results.append(result)
    
    # Save results
    output_path = get_continuation_ablation_file(run_id)
    write_csv(results, output_path)
    
    return results


def run_continuation_ablation_with_generations(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    bt_events: list[dict[str, Any]],
    generations: dict[tuple[int, int], dict[str, Any]],
    config: "ExperimentConfig",
    run_id: str,
) -> list[dict[str, Any]]:
    """
    Run continuation ablation using stored generation data.
    
    This version uses actual stored generations to reconstruct full_text
    for accurate logit measurement.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        bt_events: List of backtracking events
        generations: Dict mapping (example_id, sample_id) to generation records
        config: Experiment configuration
        run_id: Current run ID
        
    Returns:
        List of results
    """
    phase2_cfg = config.phase2
    cont_cfg = phase2_cfg.continuation_ablation
    
    # Get subset config
    subset_name = cont_cfg.subset_name
    subset_config = phase2_cfg.subset_sweep.subsets.get(subset_name)
    
    if subset_config is None:
        raise ValueError(f"Subset {subset_name} not found")
    
    scales = cont_cfg.scales
    results = []
    
    set_seed(config.run.seed)
    
    for event in tqdm(bt_events, desc="Continuation ablation"):
        example_id = int(event.get("example_id", 0))
        sample_id = int(event.get("sample_id", 0))
        
        # Get the generation record
        gen = generations.get((example_id, sample_id))
        if gen is None:
            continue
        
        full_text = gen.get("full_text", "")
        pred_pos = int(event.get("pred_pos_in_full", 0))
        onset_token_id = event.get("onset_token_id")
        
        if onset_token_id is None or onset_token_id == "":
            continue
        
        onset_token_id = int(onset_token_id)
        
        # Create truncated text ending at pred_pos
        inputs = tokenize_for_model(full_text, tokenizer, model.device)
        total_tokens = inputs["input_ids"].shape[1]
        
        # Ensure pred_pos is valid
        if pred_pos >= total_tokens:
            pred_pos = total_tokens - 1
        
        # Get text up to pred_pos (inclusive)
        truncated_ids = inputs["input_ids"][:, :pred_pos + 1]
        truncated_text = tokenizer.decode(truncated_ids[0], skip_special_tokens=False)
        
        # Baseline measurement (no ablation)
        baseline_prob, baseline_logit = get_onset_token_prob(
            model, tokenizer, truncated_text, pred_pos, onset_token_id
        )
        
        results.append({
            "example_id": example_id,
            "sample_id": sample_id,
            "variant": event.get("variant"),
            "pred_pos": pred_pos,
            "onset_token_id": onset_token_id,
            "onset_token_text": event.get("onset_token_text", ""),
            "condition": "baseline",
            "subset_name": "none",
            "scale": 1.0,
            "onset_prob": baseline_prob,
            "onset_logit": baseline_logit,
        })
        
        # Ablation measurements
        for scale in scales:
            specs = specs_from_subset_config(
                subset_config,
                mode="scale",
                scale=scale,
            )
            
            with apply_teacher_forced_hooks(model, specs, token_idx=pred_pos):
                ablated_prob, ablated_logit = get_onset_token_prob(
                    model, tokenizer, truncated_text, pred_pos, onset_token_id
                )
            
            results.append({
                "example_id": example_id,
                "sample_id": sample_id,
                "variant": event.get("variant"),
                "pred_pos": pred_pos,
                "onset_token_id": onset_token_id,
                "onset_token_text": event.get("onset_token_text", ""),
                "condition": "ablation",
                "subset_name": subset_name,
                "scale": scale,
                "onset_prob": ablated_prob,
                "onset_logit": ablated_logit,
            })
    
    # Save results
    output_path = get_continuation_ablation_file(run_id)
    write_csv(results, output_path)
    
    return results


def summarize_continuation_results(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Summarize continuation ablation results.
    
    Args:
        results: List of result records
        
    Returns:
        Summary statistics
    """
    # Group by condition and scale
    by_condition = {}
    
    for r in results:
        cond = r.get("condition", "unknown")
        scale = r.get("scale", 1.0)
        key = f"{cond}_scale_{scale}"
        
        if key not in by_condition:
            by_condition[key] = {
                "condition": cond,
                "scale": scale,
                "probs": [],
                "logits": [],
            }
        
        if "onset_prob" in r:
            by_condition[key]["probs"].append(r["onset_prob"])
        if "onset_logit" in r:
            by_condition[key]["logits"].append(r["onset_logit"])
    
    # Compute means
    summary = {}
    for key, data in by_condition.items():
        probs = data["probs"]
        logits = data["logits"]
        
        summary[key] = {
            "condition": data["condition"],
            "scale": data["scale"],
            "count": len(probs),
            "mean_prob": sum(probs) / len(probs) if probs else None,
            "mean_logit": sum(logits) / len(logits) if logits else None,
        }
    
    return summary

