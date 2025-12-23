"""
Hook utilities for ablation experiments.

Provides context managers and specifications for ablating attention
and MLP outputs at specific layers and positions.
"""

from __future__ import annotations

import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

import torch

from backtracking.modeling import get_attention_module, get_mlp_module, get_num_layers

if TYPE_CHECKING:
    from transformers import PreTrainedModel


@dataclass
class AblationSpec:
    """Specification for a single ablation operation."""
    layer_idx: int
    component: Literal["attn", "mlp"]
    mode: Literal["zero", "scale"] = "scale"
    scale: float = 0.0
    
    def __repr__(self) -> str:
        return f"AblationSpec(layer={self.layer_idx}, {self.component}, {self.mode}={self.scale})"


def create_ablation_hook(
    spec: AblationSpec,
    token_idx: int | None = None,
    decode_only: bool = False,
) -> Callable:
    """
    Create a forward hook for ablation.
    
    Args:
        spec: Ablation specification
        token_idx: Specific token index to ablate (None = all, -1 = last)
        decode_only: Only ablate during decoding (seq_len == 1)
        
    Returns:
        Hook function for register_forward_hook
    """
    def hook(module, input, output):
        # Handle tuple outputs (attention returns (hidden, attn_weights, ...))
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        
        seq_len = hidden.shape[1]
        
        # If decode_only, skip when processing full prompt
        if decode_only and seq_len > 1:
            return output
        
        # Determine which positions to ablate
        if token_idx is None:
            # Ablate all positions
            positions = slice(None)
        elif token_idx == -1:
            # Ablate last position
            positions = slice(-1, None)
        else:
            # Ablate specific position
            positions = slice(token_idx, token_idx + 1)
        
        # Apply ablation
        if spec.mode == "zero":
            hidden = hidden.clone()
            hidden[:, positions, :] = 0
        elif spec.mode == "scale":
            hidden = hidden.clone()
            hidden[:, positions, :] = hidden[:, positions, :] * spec.scale
        
        # Reconstruct output
        if rest is not None:
            return (hidden,) + rest
        return hidden
    
    return hook


@contextmanager
def apply_ablation_hooks(
    model: "PreTrainedModel",
    specs: list[AblationSpec],
    token_idx: int | None = None,
    decode_only: bool = False,
):
    """
    Context manager to apply ablation hooks to model.
    
    Args:
        model: HuggingFace model
        specs: List of ablation specifications
        token_idx: Token index to ablate (None = all, -1 = last)
        decode_only: Only ablate during decoding steps
        
    Yields:
        Model with hooks applied
    """
    handles = []
    
    try:
        for spec in specs:
            # Get the appropriate module
            if spec.component == "attn":
                module = get_attention_module(model, spec.layer_idx)
            elif spec.component == "mlp":
                module = get_mlp_module(model, spec.layer_idx)
            else:
                raise ValueError(f"Unknown component: {spec.component}")
            
            # Create and register hook
            hook = create_ablation_hook(spec, token_idx, decode_only)
            handle = module.register_forward_hook(hook)
            handles.append(handle)
        
        yield model
        
    finally:
        # Remove all hooks
        for handle in handles:
            handle.remove()


@contextmanager
def apply_generation_hooks(
    model: "PreTrainedModel",
    specs: list[AblationSpec],
):
    """
    Context manager for ablation during generation.
    
    Ablates at seq_idx=-1 (last token) and only during decoding steps
    (when seq_len == 1 under KV cache).
    
    Args:
        model: HuggingFace model
        specs: List of ablation specifications
        
    Yields:
        Model with generation hooks applied
    """
    with apply_ablation_hooks(model, specs, token_idx=-1, decode_only=True):
        yield model


@contextmanager
def apply_teacher_forced_hooks(
    model: "PreTrainedModel",
    specs: list[AblationSpec],
    token_idx: int,
):
    """
    Context manager for ablation during teacher-forced forward pass.
    
    Ablates at a specific token index in the sequence.
    
    Args:
        model: HuggingFace model
        specs: List of ablation specifications
        token_idx: Token index to ablate
        
    Yields:
        Model with hooks applied
    """
    with apply_ablation_hooks(model, specs, token_idx=token_idx, decode_only=False):
        yield model


def create_random_ablation_specs(
    targeted_specs: list[AblationSpec],
    num_layers: int,
    seed: int,
    exclude_layers: set[int] | None = None,
) -> list[AblationSpec]:
    """
    Create random ablation specs matching the shape of targeted specs.
    
    Ensures random control has:
    - Same number of attention ablations
    - Same number of MLP ablations
    - Different layers than targeted
    
    Args:
        targeted_specs: Targeted ablation specifications to match
        num_layers: Total number of layers in model
        seed: Random seed for reproducibility
        exclude_layers: Additional layers to exclude (besides targeted)
        
    Returns:
        List of random ablation specs matching targeted shape
    """
    rng = random.Random(seed)
    
    # Count components in targeted
    n_attn = sum(1 for s in targeted_specs if s.component == "attn")
    n_mlp = sum(1 for s in targeted_specs if s.component == "mlp")
    
    # Get layers used by targeted
    targeted_layers = {s.layer_idx for s in targeted_specs}
    
    # Build available layers
    all_layers = set(range(num_layers))
    excluded = targeted_layers | (exclude_layers or set())
    available = list(all_layers - excluded)
    
    if len(available) < (n_attn + n_mlp):
        # Not enough layers - use all available
        available = list(all_layers - targeted_layers)
    
    rng.shuffle(available)
    
    random_specs = []
    
    # Sample layers for attention
    attn_layers = available[:n_attn]
    for layer in attn_layers:
        random_specs.append(AblationSpec(
            layer_idx=layer,
            component="attn",
            mode=targeted_specs[0].mode if targeted_specs else "scale",
            scale=targeted_specs[0].scale if targeted_specs else 0.0,
        ))
    
    # Sample layers for MLP (from remaining)
    remaining = available[n_attn:]
    mlp_layers = remaining[:n_mlp]
    for layer in mlp_layers:
        random_specs.append(AblationSpec(
            layer_idx=layer,
            component="mlp",
            mode=targeted_specs[0].mode if targeted_specs else "scale",
            scale=targeted_specs[0].scale if targeted_specs else 0.0,
        ))
    
    return random_specs


def specs_from_selected_layers(
    selected_layers: dict,
    mode: str = "scale",
    scale: float = 0.0,
) -> list[AblationSpec]:
    """
    Create ablation specs from selected_layers.json format.
    
    Args:
        selected_layers: Dict with "attn" and "mlp" keys containing layer lists
        mode: Ablation mode
        scale: Scale factor
        
    Returns:
        List of AblationSpec
    """
    specs = []
    
    for layer in selected_layers.get("attn", []):
        specs.append(AblationSpec(
            layer_idx=layer,
            component="attn",
            mode=mode,
            scale=scale,
        ))
    
    for layer in selected_layers.get("mlp", []):
        specs.append(AblationSpec(
            layer_idx=layer,
            component="mlp",
            mode=mode,
            scale=scale,
        ))
    
    return specs

