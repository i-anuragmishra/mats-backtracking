"""
Ablation scan analysis for identifying important layers.

Performs per-layer ablation to find which layers are causally important
for producing backtracking triggers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tqdm import tqdm

from backtracking.hooks import AblationSpec, apply_teacher_forced_hooks
from backtracking.io import read_jsonl, write_csv, write_json
from backtracking.modeling import get_final_norm, get_lm_head, get_num_layers
from backtracking.paths import (
    get_ablation_importance_file,
    get_generation_file,
    get_selected_layers_file,
)
from backtracking.tokenization import tokenize_for_model

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


def compute_baseline_logit(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    full_text: str,
    pred_pos: int,
    target_token_id: int,
) -> float | None:
    """
    Compute baseline logit without any ablation.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        full_text: Full text (prompt + completion)
        pred_pos: Position to compute logit at
        target_token_id: Token ID to extract logit for
        
    Returns:
        Logit value or None if invalid
    """
    encoding = tokenize_for_model(tokenizer, full_text, add_special_tokens=True)
    input_ids = encoding.input_ids.to(model.device)
    
    seq_len = input_ids.shape[1]
    if pred_pos < 0 or pred_pos >= seq_len:
        return None
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, return_dict=True)
    
    logits = outputs.logits[0, pred_pos, :]
    return logits[target_token_id].item()


def compute_ablated_logit(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    full_text: str,
    pred_pos: int,
    target_token_id: int,
    spec: AblationSpec,
) -> float | None:
    """
    Compute logit with a single layer/component ablated.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        full_text: Full text
        pred_pos: Position to compute logit at
        target_token_id: Token ID
        spec: Ablation specification
        
    Returns:
        Ablated logit value or None
    """
    encoding = tokenize_for_model(tokenizer, full_text, add_special_tokens=True)
    input_ids = encoding.input_ids.to(model.device)
    
    seq_len = input_ids.shape[1]
    if pred_pos < 0 or pred_pos >= seq_len:
        return None
    
    with apply_teacher_forced_hooks(model, [spec], token_idx=pred_pos):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, return_dict=True)
    
    logits = outputs.logits[0, pred_pos, :]
    return logits[target_token_id].item()


def run_ablation_scan(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    bt_events: list[dict[str, Any]],
    run_id: str,
    variant: str,
    components: list[str] = ["attn", "mlp"],
    layers: str | list[int] = "all",
    mode: str = "scale",
    scale: float = 0.0,
    max_events: int = 100,
    topk: int = 6,
) -> dict[str, Any]:
    """
    Run ablation scan to find important layers for backtracking.
    
    For each layer and component, ablate at the prediction position
    and measure the change in target token logit.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        bt_events: Backtracking events
        run_id: Run identifier
        variant: Formatting variant
        components: Components to test ("attn", "mlp")
        layers: "all" or list of layer indices
        mode: Ablation mode ("zero" or "scale")
        scale: Scale factor for ablation
        max_events: Maximum events to analyze
        topk: Number of top layers to select
        
    Returns:
        Results dictionary with importance scores
    """
    num_layers = get_num_layers(model)
    
    # Determine which layers to scan
    if layers == "all":
        layer_indices = list(range(num_layers))
    else:
        layer_indices = layers
    
    # Load generations
    gen_file = get_generation_file(run_id, variant, "baseline")
    generations = {(g["example_id"], g["sample_id"]): g for g in read_jsonl(gen_file)}
    
    # Limit events
    bt_subset = bt_events[:max_events] if len(bt_events) > max_events else bt_events
    
    print(f"Running ablation scan on {len(bt_subset)} events, "
          f"{len(layer_indices)} layers, {len(components)} components")
    
    # Store results: (layer, component) -> list of delta logits
    deltas = {}
    
    for event in tqdm(bt_subset, desc="Ablation scan"):
        key = (event["example_id"], event["sample_id"])
        gen = generations.get(key)
        if not gen:
            continue
        
        pred_pos = event["pred_pos_in_full"]
        target_token_id = event["onset_token_id"]
        
        if pred_pos < 0 or target_token_id < 0:
            continue
        
        full_text = gen["full_text"]
        
        # Get baseline logit
        baseline = compute_baseline_logit(
            model, tokenizer, full_text, pred_pos, target_token_id
        )
        if baseline is None:
            continue
        
        # Test each layer/component
        for layer in layer_indices:
            for component in components:
                spec = AblationSpec(
                    layer_idx=layer,
                    component=component,
                    mode=mode,
                    scale=scale,
                )
                
                ablated = compute_ablated_logit(
                    model, tokenizer, full_text, pred_pos, target_token_id, spec
                )
                
                if ablated is not None:
                    delta = ablated - baseline  # Negative = reduced logit
                    key = (layer, component)
                    if key not in deltas:
                        deltas[key] = []
                    deltas[key].append(delta)
    
    # Compute mean delta for each layer/component
    results = []
    for (layer, component), delta_list in deltas.items():
        mean_delta = sum(delta_list) / len(delta_list) if delta_list else 0
        results.append({
            "layer": layer,
            "component": component,
            "delta_wait_logit_mean": mean_delta,
            "delta_wait_logit_std": _std(delta_list) if len(delta_list) > 1 else 0,
            "n_events": len(delta_list),
        })
    
    # Sort by delta (most negative first = most important)
    results.sort(key=lambda x: x["delta_wait_logit_mean"])
    
    # Save ablation importance
    importance_file = get_ablation_importance_file(run_id)
    write_csv(results, importance_file)
    
    # Select top-k layers
    selected = {"attn": [], "mlp": []}
    
    # Get top layers for each component
    for component in components:
        component_results = [r for r in results if r["component"] == component]
        # Most negative delta = most important
        top_layers = [r["layer"] for r in component_results[:topk]]
        selected[component] = top_layers
    
    # Save selected layers
    selected_file = get_selected_layers_file(run_id)
    write_json(selected, selected_file)
    
    return {
        "importance": results,
        "selected_layers": selected,
        "n_events_analyzed": len(bt_subset),
    }


def _std(values: list[float]) -> float:
    """Compute standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


