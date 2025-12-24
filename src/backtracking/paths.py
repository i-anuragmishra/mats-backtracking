"""
Centralized path management for the backtracking experiment.

All paths are relative to PROJECT_ROOT and follow a consistent structure.
"""

from __future__ import annotations

from pathlib import Path

from backtracking import PROJECT_ROOT


def get_runs_dir() -> Path:
    """Get the base runs directory."""
    return PROJECT_ROOT / "runs"


def get_run_dir(run_id: str) -> Path:
    """
    Get the directory for a specific run.
    
    Args:
        run_id: The run identifier
        
    Returns:
        Path to runs/<run_id>/
    """
    return get_runs_dir() / run_id


def get_generations_dir(run_id: str, variant: str) -> Path:
    """
    Get the generations directory for a variant.
    
    Args:
        run_id: The run identifier
        variant: Formatting variant name
        
    Returns:
        Path to runs/<run_id>/generations/<variant>/
    """
    path = get_run_dir(run_id) / "generations" / variant
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_generation_file(run_id: str, variant: str, condition: str) -> Path:
    """
    Get the path for a generation output file.
    
    Args:
        run_id: The run identifier
        variant: Formatting variant name
        condition: Generation condition (baseline, targeted_ablation, random_ablation)
        
    Returns:
        Path to runs/<run_id>/generations/<variant>/<condition>.jsonl
    """
    return get_generations_dir(run_id, variant) / f"{condition}.jsonl"


def get_analysis_dir(run_id: str) -> Path:
    """
    Get the analysis directory for a run.
    
    Args:
        run_id: The run identifier
        
    Returns:
        Path to runs/<run_id>/analysis/
    """
    path = get_run_dir(run_id) / "analysis"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_events_file(run_id: str) -> Path:
    """Get path to backtracking_events.csv."""
    return get_analysis_dir(run_id) / "backtracking_events.csv"


def get_summary_metrics_file(run_id: str) -> Path:
    """Get path to summary_metrics.json."""
    return get_analysis_dir(run_id) / "summary_metrics.json"


def get_logit_lens_file(run_id: str) -> Path:
    """Get path to logit_lens.csv."""
    return get_analysis_dir(run_id) / "logit_lens.csv"


def get_ablation_importance_file(run_id: str) -> Path:
    """Get path to ablation_importance.csv."""
    return get_analysis_dir(run_id) / "ablation_importance.csv"


def get_selected_layers_file(run_id: str) -> Path:
    """Get path to selected_layers.json."""
    return get_analysis_dir(run_id) / "selected_layers.json"


def get_condition_comparison_file(run_id: str) -> Path:
    """Get path to condition_comparison.csv."""
    return get_analysis_dir(run_id) / "condition_comparison.csv"


def get_formatting_summary_file(run_id: str) -> Path:
    """Get path to formatting_summary.csv."""
    return get_analysis_dir(run_id) / "formatting_summary.csv"


def get_run_figures_dir(run_id: str) -> Path:
    """
    Get the figures directory within a run.
    
    Args:
        run_id: The run identifier
        
    Returns:
        Path to runs/<run_id>/figures/
    """
    path = get_run_dir(run_id) / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_figures_dir() -> Path:
    """
    Get the proposal-ready figures directory.
    
    Returns:
        Path to figures/
    """
    path = PROJECT_ROOT / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_dir() -> Path:
    """Get the processed data directory."""
    path = PROJECT_ROOT / "data" / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_dataset_file(filename: str) -> Path:
    """
    Get path to a dataset file.
    
    Args:
        filename: Dataset filename
        
    Returns:
        Path to data/processed/<filename>
    """
    return get_data_dir() / filename


def get_reports_dir() -> Path:
    """Get the reports directory."""
    path = PROJECT_ROOT / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_run_structure(run_id: str) -> dict[str, Path]:
    """
    Create the full directory structure for a run.
    
    Args:
        run_id: The run identifier
        
    Returns:
        Dictionary mapping directory names to paths
    """
    run_dir = get_run_dir(run_id)
    
    dirs = {
        "run": run_dir,
        "generations": run_dir / "generations",
        "analysis": run_dir / "analysis",
        "figures": run_dir / "figures",
    }
    
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return dirs


