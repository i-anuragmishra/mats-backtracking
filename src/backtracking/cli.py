"""
CLI for backtracking state transition experiments.

Provides commands for the full experimental pipeline:
- init-run: Initialize a new run with persistent run_id
- prepare-data: Prepare dataset
- generate: Generate completions
- detect-events: Detect backtracking events
- logit-lens: Run logit lens analysis
- ablation-scan: Run ablation scan
- compare-conditions: Compare generation conditions
- formatting-sweep: Test formatting variants
- make-report: Generate markdown report
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from backtracking import PROJECT_ROOT

app = typer.Typer(
    name="backtracking",
    help="Backtracking State Transition Experiment Pipeline",
    add_completion=False,
)

console = Console()


def load_config_and_resolve_run(
    config_path: str,
    run_id: str | None = None,
    require_run: bool = True,
):
    """
    Load config and resolve run_id.
    
    Args:
        config_path: Path to YAML config
        run_id: Optional explicit run_id
        require_run: Whether to require an existing run
        
    Returns:
        Tuple of (config, resolved_run_id)
    """
    from backtracking.config import load_config, resolve_run_id
    
    config = load_config(config_path)
    
    if require_run:
        resolved = resolve_run_id(config, run_id)
        return config, resolved
    
    return config, run_id


@app.command()
def init_run(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Explicit run ID (default: auto-generate)"),
):
    """
    Initialize a new run with persistent run_id.
    
    Creates run directory, saves resolved config and metadata,
    and writes .current_run_id for subsequent commands.
    """
    from backtracking.config import (
        create_run_metadata,
        generate_run_id,
        load_config,
        save_resolved_config,
        save_run_metadata,
        write_current_run_id,
    )
    from backtracking.paths import ensure_run_structure
    
    console.print("[bold blue]Initializing new run...[/bold blue]")
    
    cfg = load_config(config)
    
    # Generate or use provided run_id
    if run_id:
        rid = run_id
    else:
        rid = generate_run_id()
    
    # Update config with run_id
    cfg.run.run_id = rid
    
    # Create directory structure
    dirs = ensure_run_structure(rid)
    console.print(f"  Created run directory: [green]{dirs['run']}[/green]")
    
    # Save resolved config
    save_resolved_config(cfg, dirs["run"])
    console.print(f"  Saved config_resolved.yaml")
    
    # Save metadata
    meta = create_run_metadata(cfg, rid)
    save_run_metadata(meta, dirs["run"])
    console.print(f"  Saved meta.json (git SHA: {meta['git_sha']})")
    
    # Write current run_id
    write_current_run_id(rid)
    console.print(f"  Wrote .current_run_id")
    
    console.print(f"\n[bold green]✓ Run initialized: {rid}[/bold green]")
    console.print(f"  All subsequent commands will use this run_id automatically.")


@app.command()
def prepare_data(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Override run ID"),
):
    """
    Prepare dataset for experiments.
    
    Loads dataset from HuggingFace, subsamples, extracts answers,
    and saves to data/processed/.
    """
    from datasets import load_dataset
    
    from backtracking.io import write_jsonl
    from backtracking.paths import get_data_dir
    from backtracking.seed import set_seed
    
    cfg, _ = load_config_and_resolve_run(config, run_id, require_run=False)
    
    console.print(f"[bold blue]Preparing dataset: {cfg.dataset.name}[/bold blue]")
    
    # Set seed for reproducibility
    set_seed(cfg.dataset.seed)
    
    # Load dataset
    ds = load_dataset(cfg.dataset.name, "main", split=cfg.dataset.split)
    console.print(f"  Loaded {len(ds)} examples from {cfg.dataset.split} split")
    
    # Shuffle if configured
    if cfg.dataset.shuffle:
        ds = ds.shuffle(seed=cfg.dataset.seed)
    
    # Subsample
    if cfg.dataset.max_examples and len(ds) > cfg.dataset.max_examples:
        ds = ds.select(range(cfg.dataset.max_examples))
    
    console.print(f"  Subsampled to {len(ds)} examples")
    
    # Process examples
    records = []
    for i, ex in enumerate(ds):
        # Extract answer value from GSM8K format
        answer_text = ex.get("answer", "")
        # GSM8K answers end with "#### <number>"
        answer_value = None
        if "####" in answer_text:
            answer_value = answer_text.split("####")[-1].strip()
        
        records.append({
            "id": i,
            "question": ex["question"],
            "answer": answer_text,
            "answer_value": answer_value,
        })
    
    # Save
    output_path = PROJECT_ROOT / cfg.dataset.save_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, output_path)
    
    console.print(f"\n[bold green]✓ Saved {len(records)} examples to {output_path}[/bold green]")


@app.command()
def generate(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
    condition: str = typer.Option("baseline", "--condition", help="Generation condition: baseline, targeted_ablation, random_ablation"),
    variant: Optional[str] = typer.Option(None, "--variant", "-v", help="Specific variant (default: all)"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Override run ID"),
):
    """
    Generate completions for the dataset.
    
    Supports baseline generation and ablation conditions.
    """
    from backtracking.generate import generate_for_dataset
    from backtracking.hooks import create_random_ablation_specs, specs_from_selected_layers
    from backtracking.io import read_json, read_jsonl
    from backtracking.modeling import get_num_layers, load_model_and_tokenizer
    from backtracking.paths import get_generation_file, get_selected_layers_file
    from backtracking.prompts import list_variants
    from backtracking.seed import set_seed
    
    cfg, rid = load_config_and_resolve_run(config, run_id)
    
    console.print(f"[bold blue]Generating completions ({condition})[/bold blue]")
    console.print(f"  Run ID: {rid}")
    
    # Set seed
    set_seed(cfg.run.seed)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(cfg.model)
    num_layers = get_num_layers(model)
    
    # Load dataset
    dataset_path = PROJECT_ROOT / cfg.dataset.save_path
    dataset = read_jsonl(dataset_path)
    console.print(f"  Loaded {len(dataset)} examples")
    
    # Determine variants to process
    variants_to_run = [variant] if variant else list_variants(cfg.prompting)
    
    # For ablation conditions, only process baseline variant by default
    if condition in ["targeted_ablation", "random_ablation"] and not variant:
        variants_to_run = [cfg.prompting.formatting_variants[0].name]
        console.print(f"  Ablation: using variant '{variants_to_run[0]}'")
    
    # Prepare ablation specs if needed
    ablation_specs = None
    if condition == "targeted_ablation":
        selected_file = get_selected_layers_file(rid)
        if not selected_file.exists():
            console.print("[red]Error: No selected_layers.json found. Run ablation-scan first.[/red]")
            raise typer.Exit(1)
        selected = read_json(selected_file)
        ablation_specs = specs_from_selected_layers(
            selected,
            mode=cfg.ablation_generation.mode,
            scale=cfg.ablation_generation.scale,
        )
        console.print(f"  Targeted ablation: {len(ablation_specs)} specs")
    
    elif condition == "random_ablation":
        selected_file = get_selected_layers_file(rid)
        if not selected_file.exists():
            console.print("[red]Error: No selected_layers.json found. Run ablation-scan first.[/red]")
            raise typer.Exit(1)
        selected = read_json(selected_file)
        targeted_specs = specs_from_selected_layers(
            selected,
            mode=cfg.ablation_generation.mode,
            scale=cfg.ablation_generation.scale,
        )
        ablation_specs = create_random_ablation_specs(
            targeted_specs,
            num_layers,
            cfg.ablation_generation.random_control_seed,
        )
        console.print(f"  Random ablation: {len(ablation_specs)} specs (shape-matched)")
    
    # Generate for each variant
    for var_name in variants_to_run:
        console.print(f"\n  Processing variant: {var_name}")
        output_path = get_generation_file(rid, var_name, condition)
        
        n_written = generate_for_dataset(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            config=cfg,
            variant_name=var_name,
            condition=condition,
            output_path=output_path,
            ablation_specs=ablation_specs,
        )
        
        console.print(f"    Written {n_written} generations to {output_path}")
    
    console.print(f"\n[bold green]✓ Generation complete[/bold green]")


@app.command()
def detect_events(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Override run ID"),
):
    """
    Detect backtracking events in generations.
    
    Produces backtracking_events.csv, summary_metrics.json, and figures.
    """
    from backtracking.analysis.events import process_generations_to_events
    from backtracking.analysis.plots import (
        plot_backtracking_rate_by_variant,
        plot_backtracking_vs_accuracy,
    )
    from backtracking.io import read_jsonl
    from backtracking.modeling import load_model_and_tokenizer
    
    cfg, rid = load_config_and_resolve_run(config, run_id)
    
    console.print(f"[bold blue]Detecting backtracking events[/bold blue]")
    console.print(f"  Run ID: {rid}")
    
    # Load tokenizer only (no model needed)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.hf_id,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    
    # Load dataset for gold answers
    dataset_path = PROJECT_ROOT / cfg.dataset.save_path
    dataset = read_jsonl(dataset_path)
    
    # Process events
    events, summary = process_generations_to_events(
        run_id=rid,
        config=cfg,
        tokenizer=tokenizer,
        dataset=dataset,
    )
    
    console.print(f"  Processed {len(events)} generations")
    console.print(f"  Backtracking rate (strict): {summary['backtracking_strict_rate']*100:.1f}%")
    
    # Generate figures
    console.print("  Generating figures...")
    plot_backtracking_rate_by_variant(summary, rid)
    plot_backtracking_vs_accuracy(summary, rid)
    
    console.print(f"\n[bold green]✓ Event detection complete[/bold green]")


@app.command()
def logit_lens(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
    variant: str = typer.Option("baseline_think_newline", "--variant", "-v", help="Formatting variant"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Override run ID"),
):
    """
    Run logit lens analysis on backtracking events.
    
    Computes target token logit by layer for backtracking vs controls.
    """
    from backtracking.analysis.events import (
        get_backtracking_events,
        get_non_backtracking_events,
    )
    from backtracking.analysis.logit_lens import run_logit_lens_analysis
    from backtracking.analysis.plots import plot_logit_lens
    from backtracking.io import read_csv
    from backtracking.modeling import load_model_and_tokenizer
    from backtracking.paths import get_events_file
    from backtracking.seed import set_seed
    
    cfg, rid = load_config_and_resolve_run(config, run_id)
    
    console.print(f"[bold blue]Running logit lens analysis[/bold blue]")
    console.print(f"  Run ID: {rid}")
    console.print(f"  Variant: {variant}")
    
    set_seed(cfg.run.seed)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(cfg.model)
    
    # Load events
    events_file = get_events_file(rid)
    if not events_file.exists():
        console.print("[red]Error: No events file found. Run detect-events first.[/red]")
        raise typer.Exit(1)
    
    all_events = read_csv(events_file)
    
    # Filter by variant and baseline condition
    variant_events = [e for e in all_events if e["variant"] == variant and e["condition"] == "baseline"]
    
    bt_events = get_backtracking_events(variant_events, strict=True, max_events=cfg.analysis.max_events)
    non_bt_events = get_non_backtracking_events(variant_events, strict=True)
    
    console.print(f"  Found {len(bt_events)} backtracking events, {len(non_bt_events)} non-backtracking")
    
    if not bt_events:
        console.print("[yellow]Warning: No backtracking events found[/yellow]")
        return
    
    # Run analysis
    results = run_logit_lens_analysis(
        model=model,
        tokenizer=tokenizer,
        bt_events=bt_events,
        non_bt_events=non_bt_events,
        run_id=rid,
        variant=variant,
        max_events=cfg.analysis.max_events,
        seed=cfg.run.seed,
    )
    
    # Generate figure
    from backtracking.io import read_csv
    from backtracking.paths import get_logit_lens_file
    logit_data = read_csv(get_logit_lens_file(rid))
    plot_logit_lens(logit_data, rid)
    
    console.print(f"\n[bold green]✓ Logit lens analysis complete[/bold green]")


@app.command()
def ablation_scan(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
    variant: str = typer.Option("baseline_think_newline", "--variant", "-v", help="Formatting variant"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Override run ID"),
):
    """
    Run ablation scan to find important layers.
    
    Tests each layer/component for causal importance in backtracking.
    """
    from backtracking.analysis.ablation_scan import run_ablation_scan
    from backtracking.analysis.events import get_backtracking_events
    from backtracking.analysis.plots import plot_ablation_importance
    from backtracking.io import read_csv
    from backtracking.modeling import load_model_and_tokenizer
    from backtracking.paths import get_ablation_importance_file, get_events_file
    from backtracking.seed import set_seed
    
    cfg, rid = load_config_and_resolve_run(config, run_id)
    
    console.print(f"[bold blue]Running ablation scan[/bold blue]")
    console.print(f"  Run ID: {rid}")
    console.print(f"  Variant: {variant}")
    
    set_seed(cfg.run.seed)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(cfg.model)
    
    # Load events
    events_file = get_events_file(rid)
    if not events_file.exists():
        console.print("[red]Error: No events file found. Run detect-events first.[/red]")
        raise typer.Exit(1)
    
    all_events = read_csv(events_file)
    variant_events = [e for e in all_events if e["variant"] == variant and e["condition"] == "baseline"]
    bt_events = get_backtracking_events(variant_events, strict=True, max_events=cfg.analysis.max_events)
    
    console.print(f"  Found {len(bt_events)} backtracking events")
    
    if not bt_events:
        console.print("[yellow]Warning: No backtracking events found[/yellow]")
        return
    
    # Run ablation scan
    results = run_ablation_scan(
        model=model,
        tokenizer=tokenizer,
        bt_events=bt_events,
        run_id=rid,
        variant=variant,
        components=cfg.analysis.ablation_components,
        layers=cfg.analysis.ablation_layers,
        mode=cfg.ablation_generation.mode,
        scale=cfg.ablation_generation.scale,
        max_events=cfg.analysis.max_events,
        topk=cfg.analysis.topk_layers_for_generation,
    )
    
    # Generate figure
    importance_data = read_csv(get_ablation_importance_file(rid))
    plot_ablation_importance(importance_data, rid)
    
    console.print(f"\n  Selected layers: {results['selected_layers']}")
    console.print(f"\n[bold green]✓ Ablation scan complete[/bold green]")


@app.command()
def compare_conditions(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
    variant: str = typer.Option("baseline_think_newline", "--variant", "-v", help="Formatting variant"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Override run ID"),
):
    """
    Compare backtracking rates across conditions.
    
    Generates comparison figure and CSV.
    """
    from backtracking.analysis.events import compute_summary_metrics
    from backtracking.analysis.plots import plot_backtracking_by_condition
    from backtracking.io import read_csv, write_csv
    from backtracking.paths import get_condition_comparison_file, get_events_file
    
    cfg, rid = load_config_and_resolve_run(config, run_id)
    
    console.print(f"[bold blue]Comparing conditions[/bold blue]")
    console.print(f"  Run ID: {rid}")
    console.print(f"  Variant: {variant}")
    
    # Load events
    events_file = get_events_file(rid)
    if not events_file.exists():
        console.print("[red]Error: No events file found. Run detect-events first.[/red]")
        raise typer.Exit(1)
    
    all_events = read_csv(events_file)
    variant_events = [e for e in all_events if e["variant"] == variant]
    
    # Compute summary for this variant
    summary = compute_summary_metrics(variant_events)
    
    # Save comparison
    by_condition = summary.get("by_condition", {})
    comparison = []
    for cond, stats in by_condition.items():
        comparison.append({
            "variant": variant,
            "condition": cond,
            "total": stats["total"],
            "backtracking_count": stats["backtracking_strict"],
            "backtracking_rate": stats["backtracking_rate"],
            "accuracy": stats["accuracy"],
        })
    
    output_file = get_condition_comparison_file(rid)
    write_csv(comparison, output_file)
    
    # Generate figure
    plot_backtracking_by_condition(summary, rid)
    
    console.print(f"  Saved comparison to {output_file}")
    console.print(f"\n[bold green]✓ Condition comparison complete[/bold green]")


@app.command()
def formatting_sweep(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Override run ID"),
):
    """
    Summarize backtracking across formatting variants.
    
    Generates formatting comparison figure and CSV.
    """
    from backtracking.analysis.events import compute_summary_metrics
    from backtracking.analysis.plots import plot_formatting_effect
    from backtracking.io import read_csv, write_csv
    from backtracking.paths import get_events_file, get_formatting_summary_file
    from backtracking.prompts import list_variants
    
    cfg, rid = load_config_and_resolve_run(config, run_id)
    
    console.print(f"[bold blue]Formatting sweep[/bold blue]")
    console.print(f"  Run ID: {rid}")
    
    # Load events
    events_file = get_events_file(rid)
    if not events_file.exists():
        console.print("[red]Error: No events file found. Run detect-events first.[/red]")
        raise typer.Exit(1)
    
    all_events = read_csv(events_file)
    
    # Summarize by variant
    formatting_summary = []
    for variant in list_variants(cfg.prompting):
        variant_events = [e for e in all_events if e["variant"] == variant and e["condition"] == "baseline"]
        if not variant_events:
            continue
        
        summary = compute_summary_metrics(variant_events)
        formatting_summary.append({
            "variant": variant,
            "total": summary["total_samples"],
            "backtracking_count": summary["backtracking_strict_count"],
            "backtracking_rate": summary["backtracking_strict_rate"],
            "accuracy": summary.get("accuracy"),
        })
    
    # Save
    output_file = get_formatting_summary_file(rid)
    write_csv(formatting_summary, output_file)
    
    # Generate figure
    plot_formatting_effect(formatting_summary, rid)
    
    console.print(f"  Saved summary to {output_file}")
    console.print(f"\n[bold green]✓ Formatting sweep complete[/bold green]")


@app.command()
def make_report(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Override run ID"),
):
    """
    Generate markdown report for the run.
    """
    from backtracking.report import generate_report
    
    cfg, rid = load_config_and_resolve_run(config, run_id)
    
    console.print(f"[bold blue]Generating report[/bold blue]")
    console.print(f"  Run ID: {rid}")
    
    output_path = generate_report(
        run_id=rid,
        config_path=config,
        output_path=cfg.report.report_path,
    )
    
    console.print(f"\n[bold green]✓ Report generated: {output_path}[/bold green]")


@app.command()
def doctor():
    """
    Run environment diagnostics.
    """
    import subprocess
    import sys
    
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "doctor.sh")], check=False)


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
