"""
Model loading utilities.

Handles loading HuggingFace models and tokenizers with proper configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from backtracking.config import ModelConfig

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert string dtype to torch.dtype.
    
    Args:
        dtype_str: One of "float16", "bfloat16", "float32"
        
    Returns:
        Corresponding torch dtype
    """
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    return dtype_map.get(dtype_str.lower(), torch.bfloat16)


def load_model_and_tokenizer(
    config: ModelConfig,
) -> tuple["PreTrainedModel", "PreTrainedTokenizer"]:
    """
    Load model and tokenizer from HuggingFace.
    
    Args:
        config: Model configuration
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {config.hf_id}")
    print(f"  dtype: {config.torch_dtype}")
    print(f"  device: {config.device}")
    print(f"  attn_implementation: {config.attn_implementation}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.hf_id,
        trust_remote_code=config.trust_remote_code,
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Build model kwargs
    model_kwargs = {
        "trust_remote_code": config.trust_remote_code,
        "torch_dtype": get_torch_dtype(config.torch_dtype),
        "device_map": config.device if config.device != "cpu" else None,
    }
    
    # Add attention implementation if specified
    if config.attn_implementation:
        model_kwargs["attn_implementation"] = config.attn_implementation
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.hf_id,
        **model_kwargs,
    )
    
    # Move to device if not using device_map
    if config.device == "cpu" or model_kwargs["device_map"] is None:
        model = model.to(config.device)
    
    # Set eval mode
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Layers: {get_num_layers(model)}")
    
    return model, tokenizer


def get_num_layers(model: "PreTrainedModel") -> int:
    """
    Get the number of transformer layers in the model.
    
    Args:
        model: HuggingFace model
        
    Returns:
        Number of layers
    """
    # Try common attribute names
    if hasattr(model, "config"):
        config = model.config
        for attr in ["num_hidden_layers", "n_layer", "num_layers"]:
            if hasattr(config, attr):
                return getattr(config, attr)
    
    # Try to count layers directly
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    
    return 0


def get_layer_module(model: "PreTrainedModel", layer_idx: int):
    """
    Get a specific transformer layer module.
    
    Args:
        model: HuggingFace model
        layer_idx: Layer index (0-based)
        
    Returns:
        Layer module
    """
    # Most common: model.model.layers[i]
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    
    # GPT-2 style: model.transformer.h[i]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    
    raise ValueError(f"Cannot find layer {layer_idx} in model architecture")


def get_attention_module(model: "PreTrainedModel", layer_idx: int):
    """
    Get the attention module for a specific layer.
    
    Args:
        model: HuggingFace model
        layer_idx: Layer index (0-based)
        
    Returns:
        Attention module
    """
    layer = get_layer_module(model, layer_idx)
    
    # Common attribute names
    for attr in ["self_attn", "attn", "attention"]:
        if hasattr(layer, attr):
            return getattr(layer, attr)
    
    raise ValueError(f"Cannot find attention module in layer {layer_idx}")


def get_mlp_module(model: "PreTrainedModel", layer_idx: int):
    """
    Get the MLP module for a specific layer.
    
    Args:
        model: HuggingFace model
        layer_idx: Layer index (0-based)
        
    Returns:
        MLP module
    """
    layer = get_layer_module(model, layer_idx)
    
    # Common attribute names
    for attr in ["mlp", "feed_forward", "ffn"]:
        if hasattr(layer, attr):
            return getattr(layer, attr)
    
    raise ValueError(f"Cannot find MLP module in layer {layer_idx}")


def get_final_norm(model: "PreTrainedModel"):
    """
    Get the final layer normalization module.
    
    Args:
        model: HuggingFace model
        
    Returns:
        LayerNorm module
    """
    # Common paths
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    
    raise ValueError("Cannot find final normalization layer")


def get_lm_head(model: "PreTrainedModel"):
    """
    Get the language model head (output projection).
    
    Args:
        model: HuggingFace model
        
    Returns:
        LM head module
    """
    if hasattr(model, "lm_head"):
        return model.lm_head
    
    raise ValueError("Cannot find lm_head in model")

