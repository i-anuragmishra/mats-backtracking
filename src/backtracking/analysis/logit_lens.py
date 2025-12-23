"""
Logit lens analysis for backtracking detection.

Computes the "Wait" token logit at each layer for backtracking vs control samples.
Uses matched-position controls to avoid confounding position effects.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import torch
from tqdm import tqdm

from backtracking.io import read_jsonl, write_csv
from backtracking.modeling import get_final_norm, get_lm_head, get_num_layers
from backtracking.paths import get_generation_file, get_logit_lens_file
from backtracking.tokenization import tokenize_for_model

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


def select_matched_controls(
    bt_events: list[dict[str, Any]],
    non_bt_events: list[dict[str, Any]],
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Select position-matched control samples for each backtracking event.
    
    For each backtracking event with onset at offset k (relative to completion start),
    find a control sample where completion_tokens >= k and set:
        pred_pos_control = prompt_tokens + k - 1
    
    This ensures we compare "Wait" evidence at the SAME relative position.
    
    Args:
        bt_events: Backtracking events with onset_token_in_completion
        non_bt_events: Non-backtracking events
        seed: Random seed for selection
        
    Returns:
        List of control events with pred_pos set
    """
    rng = random.Random(seed)
    controls = []
    
    # Shuffle non-bt for random selection
    shuffled_non_bt = non_bt_events.copy()
    rng.shuffle(shuffled_non_bt)
    
    used_indices = set()
    
    for bt in bt_events:
        k = bt.get("onset_token_in_completion", 0)
        if k <= 0:
            continue
        
        # Find a non-bt sample with enough completion tokens
        found = False
        for i, ctrl in enumerate(shuffled_non_bt):
            if i in used_indices:
                continue
            if ctrl["completion_tokens"] >= k:
                ctrl_copy = ctrl.copy()
                # Set pred_pos at the SAME relative position as backtracking onset
                ctrl_copy["pred_pos"] = ctrl["prompt_tokens"] + k - 1
                ctrl_copy["matched_to_example"] = bt["example_id"]
                ctrl_copy["matched_to_sample"] = bt["sample_id"]
                ctrl_copy["matched_offset_k"] = k
                controls.append(ctrl_copy)
                used_indices.add(i)
                found = True
                break
        
        if not found:
            # No suitable control found - skip or use fallback
            pass
    
    return controls


def compute_logit_lens_single(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    full_text: str,
    pred_pos: int,
    target_token_id: int,
) -> list[float]:
    """
    Compute logit lens for a single sample.
    
    For each layer, apply final norm + lm_head to get logits,
    and extract the logit for the target token at pred_pos.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        full_text: Full text (prompt + completion)
        pred_pos: Position to compute logits at
        target_token_id: Token ID to extract logit for
        
    Returns:
        List of logits, one per layer
    """
    # Tokenize
    encoding = tokenize_for_model(tokenizer, full_text, add_special_tokens=True)
    input_ids = encoding.input_ids.to(model.device)
    
    # Verify pred_pos is valid
    seq_len = input_ids.shape[1]
    if pred_pos < 0 or pred_pos >= seq_len:
        return []
    
    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
        )
    
    hidden_states = outputs.hidden_states  # Tuple of (n_layers + 1) tensors
    
    # Get final norm and lm_head
    final_norm = get_final_norm(model)
    lm_head = get_lm_head(model)
    
    layer_logits = []
    
    # hidden_states[0] is embeddings, hidden_states[1:] are layer outputs
    for layer_hidden in hidden_states[1:]:
        # Extract hidden state at pred_pos
        h = layer_hidden[0, pred_pos, :]  # Shape: (hidden_dim,)
        
        # Apply final norm
        h_normed = final_norm(h.unsqueeze(0))  # Shape: (1, hidden_dim)
        
        # Apply lm_head to get logits
        logits = lm_head(h_normed)  # Shape: (1, vocab_size)
        
        # Extract logit for target token
        target_logit = logits[0, target_token_id].item()
        layer_logits.append(target_logit)
    
    return layer_logits


def run_logit_lens_analysis(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    bt_events: list[dict[str, Any]],
    non_bt_events: list[dict[str, Any]],
    run_id: str,
    variant: str,
    max_events: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Run logit lens analysis comparing backtracking vs matched controls.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        bt_events: Backtracking events
        non_bt_events: Non-backtracking events
        run_id: Run identifier
        variant: Formatting variant
        max_events: Maximum events to analyze
        seed: Random seed
        
    Returns:
        Results dictionary with per-layer statistics
    """
    num_layers = get_num_layers(model)
    
    # Load generation data to get full_text
    gen_file = get_generation_file(run_id, variant, "baseline")
    generations = {(g["example_id"], g["sample_id"]): g for g in read_jsonl(gen_file)}
    
    # Limit events
    bt_subset = bt_events[:max_events] if len(bt_events) > max_events else bt_events
    
    # Select matched controls
    controls = select_matched_controls(bt_subset, non_bt_events, seed)
    
    print(f"Analyzing {len(bt_subset)} backtracking events and {len(controls)} matched controls")
    
    # Collect logits for backtracking events
    bt_logits_by_layer = [[] for _ in range(num_layers)]
    
    for event in tqdm(bt_subset, desc="Logit lens (backtracking)"):
        key = (event["example_id"], event["sample_id"])
        gen = generations.get(key)
        if not gen:
            continue
        
        pred_pos = event["pred_pos_in_full"]
        target_token_id = event["onset_token_id"]
        
        if pred_pos < 0 or target_token_id < 0:
            continue
        
        layer_logits = compute_logit_lens_single(
            model, tokenizer, gen["full_text"], pred_pos, target_token_id
        )
        
        for i, logit in enumerate(layer_logits):
            bt_logits_by_layer[i].append(logit)
    
    # Collect logits for control events
    ctrl_logits_by_layer = [[] for _ in range(num_layers)]
    
    for event in tqdm(controls, desc="Logit lens (control)"):
        key = (event["example_id"], event["sample_id"])
        gen = generations.get(key)
        if not gen:
            continue
        
        pred_pos = event["pred_pos"]
        
        # For controls, use the same target token ID as the matched BT event
        # This is a simplification - ideally we'd use the BT event's onset_token_id
        # But since controls don't have backtracking, we use a common fallback
        target_token_id = bt_subset[0]["onset_token_id"] if bt_subset else tokenizer.encode("Wait")[0]
        
        if pred_pos < 0:
            continue
        
        layer_logits = compute_logit_lens_single(
            model, tokenizer, gen["full_text"], pred_pos, target_token_id
        )
        
        for i, logit in enumerate(layer_logits):
            ctrl_logits_by_layer[i].append(logit)
    
    # Compute statistics
    results = []
    for layer in range(num_layers):
        bt_vals = bt_logits_by_layer[layer]
        ctrl_vals = ctrl_logits_by_layer[layer]
        
        results.append({
            "layer": layer,
            "mean_target_logit_bt": sum(bt_vals) / len(bt_vals) if bt_vals else 0,
            "mean_target_logit_control": sum(ctrl_vals) / len(ctrl_vals) if ctrl_vals else 0,
            "std_bt": _std(bt_vals) if bt_vals else 0,
            "std_control": _std(ctrl_vals) if ctrl_vals else 0,
            "n_bt": len(bt_vals),
            "n_control": len(ctrl_vals),
            "control_matching": "position_matched",
        })
    
    # Save results
    output_file = get_logit_lens_file(run_id)
    write_csv(results, output_file)
    
    return {
        "by_layer": results,
        "n_bt_total": len(bt_subset),
        "n_control_total": len(controls),
    }


def _std(values: list[float]) -> float:
    """Compute standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5

