"""
Generation utilities for creating model completions.

Handles batch generation with optional ablation hooks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tqdm import tqdm

from backtracking.config import ExperimentConfig, GenerationConfig
from backtracking.hooks import AblationSpec, apply_generation_hooks
from backtracking.io import append_jsonl
from backtracking.prompts import format_prompt_with_system, get_variant_by_name
from backtracking.tokenization import tokenize_for_model

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedModel, PreTrainedTokenizer


def generate_single(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    prompt: str,
    gen_config: GenerationConfig,
    num_samples: int = 1,
    ablation_specs: list[AblationSpec] | None = None,
) -> list[dict[str, Any]]:
    """
    Generate completions for a single prompt.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompt: Input prompt
        gen_config: Generation configuration
        num_samples: Number of samples to generate
        ablation_specs: Optional ablation specifications
        
    Returns:
        List of generation result dicts
    """
    # Tokenize prompt
    encoding = tokenize_for_model(tokenizer, prompt, add_special_tokens=True)
    input_ids = encoding.input_ids.to(model.device)
    attention_mask = encoding.attention_mask.to(model.device)
    prompt_tokens = encoding.num_tokens
    
    # Build generation kwargs
    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": gen_config.max_new_tokens,
        "temperature": gen_config.temperature,
        "top_p": gen_config.top_p,
        "do_sample": gen_config.do_sample,
        "num_return_sequences": num_samples,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    # Generate with or without ablation hooks
    if ablation_specs:
        with apply_generation_hooks(model, ablation_specs):
            with torch.no_grad():
                outputs = model.generate(**gen_kwargs)
    else:
        with torch.no_grad():
            outputs = model.generate(**gen_kwargs)
    
    # Decode outputs
    results = []
    for i in range(num_samples):
        output_ids = outputs[i]
        full_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        completion = tokenizer.decode(output_ids[prompt_tokens:], skip_special_tokens=True)
        
        results.append({
            "prompt": prompt,
            "completion": completion,
            "full_text": full_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(output_ids) - prompt_tokens,
            "gen_params": {
                "temperature": gen_config.temperature,
                "top_p": gen_config.top_p,
                "max_new_tokens": gen_config.max_new_tokens,
                "do_sample": gen_config.do_sample,
            },
        })
    
    return results


def generate_batch(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    prompts: list[str],
    gen_config: GenerationConfig,
    num_samples: int = 1,
    ablation_specs: list[AblationSpec] | None = None,
    show_progress: bool = True,
) -> list[list[dict[str, Any]]]:
    """
    Generate completions for a batch of prompts.
    
    Note: Due to variable-length sampling, we process one prompt at a time
    but could batch if using fixed-length generation.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompts: List of input prompts
        gen_config: Generation configuration
        num_samples: Number of samples per prompt
        ablation_specs: Optional ablation specifications
        show_progress: Whether to show progress bar
        
    Returns:
        List of lists of generation results (outer: prompts, inner: samples)
    """
    all_results = []
    
    iterator = tqdm(prompts, desc="Generating", disable=not show_progress)
    for prompt in iterator:
        results = generate_single(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            gen_config=gen_config,
            num_samples=num_samples,
            ablation_specs=ablation_specs,
        )
        all_results.append(results)
    
    return all_results


def generate_for_dataset(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    dataset: list[dict[str, Any]],
    config: ExperimentConfig,
    variant_name: str,
    condition: str,
    output_path: "Path",
    ablation_specs: list[AblationSpec] | None = None,
) -> int:
    """
    Generate completions for a dataset and save to JSONL.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        dataset: List of dataset examples (must have 'id', 'question')
        config: Experiment configuration
        variant_name: Formatting variant name
        condition: Generation condition (baseline, targeted_ablation, random_ablation)
        output_path: Path to output JSONL file
        ablation_specs: Optional ablation specifications (for ablation conditions)
        
    Returns:
        Total number of generations written
    """
    variant = get_variant_by_name(config.prompting, variant_name)
    gen_config = config.generation
    
    total_written = 0
    
    for example in tqdm(dataset, desc=f"Generating ({condition})"):
        example_id = example["id"]
        question = example["question"]
        
        # Format prompt
        prompt = format_prompt_with_system(
            question=question,
            config=config.prompting,
            variant=variant,
        )
        
        # Generate samples
        results = generate_single(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            gen_config=gen_config,
            num_samples=gen_config.num_samples_per_prompt,
            ablation_specs=ablation_specs,
        )
        
        # Save each sample
        for sample_id, result in enumerate(results):
            record = {
                "example_id": example_id,
                "sample_id": sample_id,
                "variant": variant_name,
                "condition": condition,
                "prompt": result["prompt"],
                "completion": result["completion"],
                "full_text": result["full_text"],
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "finish_reason": "stop",  # HF doesn't provide this easily
                "gen_params": result["gen_params"],
                "model_hf_id": config.model.hf_id,
            }
            append_jsonl(record, output_path)
            total_written += 1
    
    return total_written


def get_generation_stats(generations: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute statistics for a set of generations.
    
    Args:
        generations: List of generation records
        
    Returns:
        Statistics dictionary
    """
    if not generations:
        return {}
    
    completion_lengths = [g["completion_tokens"] for g in generations]
    
    return {
        "count": len(generations),
        "avg_completion_tokens": sum(completion_lengths) / len(completion_lengths),
        "min_completion_tokens": min(completion_lengths),
        "max_completion_tokens": max(completion_lengths),
        "conditions": list(set(g["condition"] for g in generations)),
        "variants": list(set(g["variant"] for g in generations)),
    }

