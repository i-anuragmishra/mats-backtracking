"""
Phase 2: Subset and scale sweep analysis.

Implements non-destructive ablation sweeps to find interventions that
reduce backtracking without collapsing accuracy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tqdm import tqdm

from backtracking.detect import detect_backtracking, extract_final_answer, check_answer_correct
from backtracking.hooks import (
    AblationSpec,
    apply_generation_hooks_with_debug,
    get_hook_debug_state,
    reset_hook_debug,
    specs_from_subset_config,
    write_hook_debug_json,
)
from backtracking.io import read_jsonl, write_csv, write_json
from backtracking.paths import (
    get_phase2_generations_dir,
    get_scale_sweep_file,
    get_subset_sweep_file,
)
from backtracking.prompts import format_prompt, get_variant_by_name
from backtracking.seed import set_seed

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    
    from backtracking.config import ExperimentConfig


def generate_with_ablation(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    prompts: list[str],
    specs: list[AblationSpec],
    config: "ExperimentConfig",
    debug: bool = True,
) -> list[dict[str, Any]]:
    """
    Generate completions with ablation applied.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompts: List of formatted prompts
        specs: Ablation specifications
        config: Experiment configuration
        debug: Whether to enable hook debug
        
    Returns:
        List of generation records
    """
    generations = []
    gen_cfg = config.generation
    
    for prompt in tqdm(prompts, desc="Generating"):
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(model.device)
        
        prompt_len = inputs.input_ids.shape[1]
        
        # Generate with ablation
        with apply_generation_hooks_with_debug(model, specs, debug=debug):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=gen_cfg.max_new_tokens,
                    temperature=gen_cfg.temperature,
                    top_p=gen_cfg.top_p,
                    do_sample=gen_cfg.do_sample,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
        
        # Decode
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        
        generations.append({
            "prompt": prompt,
            "completion": completion,
            "full_text": full_text,
            "prompt_tokens": prompt_len,
            "completion_tokens": outputs.shape[1] - prompt_len,
        })
    
    return generations


def evaluate_generations(
    generations: list[dict[str, Any]],
    dataset: list[dict[str, Any]],
    tokenizer: "PreTrainedTokenizer",
    config: "ExperimentConfig",
) -> dict[str, Any]:
    """
    Evaluate generations for backtracking rate and accuracy.
    
    Args:
        generations: List of generation records
        dataset: Dataset with gold answers
        tokenizer: HuggingFace tokenizer
        config: Experiment configuration
        
    Returns:
        Summary statistics
    """
    # Build answer lookup
    answers = {ex["id"]: ex.get("answer_value") for ex in dataset}
    
    bt_count = 0
    correct_count = 0
    total = len(generations)
    
    for i, gen in enumerate(generations):
        # Detect backtracking
        detection = detect_backtracking(
            completion=gen["completion"],
            prompt=gen["prompt"],
            full_text=gen["full_text"],
            tokenizer=tokenizer,
            config=config.detection,
        )
        
        if detection.has_backtracking_strict:
            bt_count += 1
        
        # Check accuracy
        predicted = extract_final_answer(gen["completion"], config.detection.answer_regex)
        gold = answers.get(i % len(dataset))  # Handle multiple samples per example
        if check_answer_correct(predicted, gold):
            correct_count += 1
    
    return {
        "total": total,
        "backtracking_count": bt_count,
        "backtracking_rate": bt_count / total if total > 0 else 0,
        "correct_count": correct_count,
        "accuracy": correct_count / total if total > 0 else 0,
    }


def run_subset_sweep(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    dataset: list[dict[str, Any]],
    config: "ExperimentConfig",
    run_id: str,
    phase1_run_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Run ablation sweep across different layer subsets.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        dataset: Dataset with examples
        config: Experiment configuration
        run_id: Current run ID
        phase1_run_id: Phase 1 run ID for reference (optional)
        
    Returns:
        List of results for each subset
    """
    phase2_cfg = config.phase2
    sweep_cfg = phase2_cfg.subset_sweep
    
    if not sweep_cfg.enabled:
        return []
    
    # Limit dataset for Phase 2
    max_examples = phase2_cfg.max_examples
    num_samples = phase2_cfg.num_samples_per_prompt
    
    if len(dataset) > max_examples:
        dataset = dataset[:max_examples]
    
    # Get variant config
    variant_name = sweep_cfg.variant
    variant_cfg = get_variant_by_name(config.prompting, variant_name)
    
    # Format prompts
    prompts = []
    for ex in dataset:
        prompt = format_prompt(
            question=ex["question"],
            config=config.prompting,
            variant=variant_cfg,
        )
        # Repeat for num_samples
        for _ in range(num_samples):
            prompts.append(prompt)
    
    results = []
    
    # First, run baseline (no ablation)
    set_seed(config.run.seed)
    baseline_gens = generate_with_ablation(
        model, tokenizer, prompts,
        specs=[],  # No ablation
        config=config,
        debug=False,
    )
    baseline_stats = evaluate_generations(baseline_gens, dataset, tokenizer, config)
    baseline_stats["subset_name"] = "baseline"
    baseline_stats["num_components"] = 0
    results.append(baseline_stats)
    
    # Run each subset
    for subset_name, subset_config in sweep_cfg.subsets.items():
        set_seed(config.run.seed)
        reset_hook_debug()
        
        specs = specs_from_subset_config(
            subset_config,
            mode=config.ablation_generation.mode,
            scale=config.ablation_generation.scale,
        )
        
        generations = generate_with_ablation(
            model, tokenizer, prompts,
            specs=specs,
            config=config,
            debug=phase2_cfg.hook_debug,
        )
        
        stats = evaluate_generations(generations, dataset, tokenizer, config)
        stats["subset_name"] = subset_name
        stats["num_components"] = len(specs)
        results.append(stats)
    
    # Save results
    output_path = get_subset_sweep_file(run_id)
    write_csv(results, output_path)
    
    # Save hook debug info
    if phase2_cfg.hook_debug:
        write_hook_debug_json(run_id)
    
    return results


def run_scale_sweep(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    dataset: list[dict[str, Any]],
    config: "ExperimentConfig",
    run_id: str,
) -> list[dict[str, Any]]:
    """
    Run ablation sweep across different scale factors.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        dataset: Dataset with examples
        config: Experiment configuration
        run_id: Current run ID
        
    Returns:
        List of results for each scale
    """
    phase2_cfg = config.phase2
    sweep_cfg = phase2_cfg.scale_sweep
    
    if not sweep_cfg.enabled:
        return []
    
    # Limit dataset for Phase 2
    max_examples = phase2_cfg.max_examples
    num_samples = phase2_cfg.num_samples_per_prompt
    
    if len(dataset) > max_examples:
        dataset = dataset[:max_examples]
    
    # Get variant config
    variant_name = sweep_cfg.variant
    variant_cfg = get_variant_by_name(config.prompting, variant_name)
    
    # Get subset config
    subset_name = sweep_cfg.subset_name
    subset_config = phase2_cfg.subset_sweep.subsets.get(subset_name)
    
    if subset_config is None:
        raise ValueError(f"Subset {subset_name} not found in config")
    
    # Format prompts
    prompts = []
    for ex in dataset:
        prompt = format_prompt(
            question=ex["question"],
            config=config.prompting,
            variant=variant_cfg,
        )
        for _ in range(num_samples):
            prompts.append(prompt)
    
    results = []
    
    for scale in sweep_cfg.scales:
        set_seed(config.run.seed)
        reset_hook_debug()
        
        specs = specs_from_subset_config(
            subset_config,
            mode="scale",
            scale=scale,
        )
        
        generations = generate_with_ablation(
            model, tokenizer, prompts,
            specs=specs,
            config=config,
            debug=phase2_cfg.hook_debug,
        )
        
        stats = evaluate_generations(generations, dataset, tokenizer, config)
        stats["subset_name"] = subset_name
        stats["scale"] = scale
        stats["num_components"] = len(specs)
        results.append(stats)
    
    # Save results
    output_path = get_scale_sweep_file(run_id)
    write_csv(results, output_path)
    
    return results


def find_best_tradeoff(
    subset_results: list[dict[str, Any]],
    baseline_bt_rate: float,
    min_accuracy_retention: float = 0.5,
) -> dict[str, Any] | None:
    """
    Find the subset with best backtracking reduction while retaining accuracy.
    
    Args:
        subset_results: Results from subset sweep
        baseline_bt_rate: Baseline backtracking rate
        min_accuracy_retention: Minimum accuracy retention ratio (vs baseline)
        
    Returns:
        Best subset result, or None if no good tradeoff found
    """
    # Get baseline accuracy
    baseline = next((r for r in subset_results if r["subset_name"] == "baseline"), None)
    if baseline is None:
        return None
    
    baseline_acc = baseline["accuracy"]
    
    # Filter to subsets with acceptable accuracy
    candidates = []
    for r in subset_results:
        if r["subset_name"] == "baseline":
            continue
        
        if baseline_acc > 0 and r["accuracy"] >= baseline_acc * min_accuracy_retention:
            bt_reduction = (baseline_bt_rate - r["backtracking_rate"]) / baseline_bt_rate
            candidates.append({
                **r,
                "bt_reduction": bt_reduction,
                "accuracy_retention": r["accuracy"] / baseline_acc if baseline_acc > 0 else 0,
            })
    
    if not candidates:
        return None
    
    # Sort by backtracking reduction
    candidates.sort(key=lambda x: x["bt_reduction"], reverse=True)
    return candidates[0]

