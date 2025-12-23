"""
Configuration loading and run ID management.

Handles YAML config loading, validation, and run_id persistence across CLI commands.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml

from backtracking import PROJECT_ROOT


@dataclass
class RunConfig:
    """Run-level configuration."""
    name: str = "backtracking_state_transition"
    seed: int = 42
    run_id: str | None = None
    output_dir: str = "runs"


@dataclass
class ModelConfig:
    """Model loading configuration."""
    hf_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    device: str = "cuda"
    attn_implementation: str | None = "flash_attention_2"
    use_cache: bool = True


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str = "gsm8k"
    split: str = "test"
    max_examples: int = 200
    shuffle: bool = True
    seed: int = 123
    save_path: str = "data/processed/gsm8k_200.jsonl"


@dataclass
class FormattingVariant:
    """A single formatting variant for prompts."""
    name: str
    think_open: str
    think_close: str


@dataclass
class PromptingConfig:
    """Prompting configuration."""
    system_prompt: str = "You are a helpful assistant."
    template: str = ""
    formatting_variants: list[FormattingVariant] = field(default_factory=list)


@dataclass
class GenerationConfig:
    """Generation parameters."""
    num_samples_per_prompt: int = 6
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True
    batch_size: int = 4


@dataclass
class DetectionConfig:
    """Backtracking detection configuration."""
    triggers_strict: list[str] = field(default_factory=lambda: ["Wait", "Actually", "Hold on"])
    triggers_relaxed: list[str] = field(default_factory=list)
    onset_priority: list[str] = field(default_factory=lambda: ["Wait", "Actually", "Hold on"])
    answer_regex: str | None = None


@dataclass
class AnalysisConfig:
    """Analysis configuration."""
    max_events: int = 120
    control_samples: int = 120
    logit_lens_token: str = "Wait"
    ablation_components: list[str] = field(default_factory=lambda: ["attn", "mlp"])
    ablation_layers: str | list[int] = "all"
    topk_layers_for_generation: int = 6


@dataclass
class AblationGenerationConfig:
    """Ablation during generation configuration."""
    enabled: bool = True
    mode: Literal["zero", "scale"] = "scale"
    scale: float = 0.0
    random_control_seed: int = 999


@dataclass
class ReportConfig:
    """Report generation configuration."""
    make_report: bool = True
    report_path: str = "reports/backtracking_state_transition_report.md"


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    prompting: PromptingConfig = field(default_factory=PromptingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    ablation_generation: AblationGenerationConfig = field(default_factory=AblationGenerationConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    
    # Resolved paths (set after loading)
    _config_path: Path | None = field(default=None, repr=False)


def load_config(config_path: str | Path) -> ExperimentConfig:
    """
    Load experiment configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        ExperimentConfig with all sections populated
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    
    # Parse each section
    config = ExperimentConfig()
    config._config_path = config_path
    
    if "run" in raw:
        config.run = RunConfig(**raw["run"])
    
    if "model" in raw:
        config.model = ModelConfig(**raw["model"])
    
    if "dataset" in raw:
        config.dataset = DatasetConfig(**raw["dataset"])
    
    if "prompting" in raw:
        prompting_data = raw["prompting"].copy()
        variants_raw = prompting_data.pop("formatting_variants", [])
        config.prompting = PromptingConfig(**prompting_data)
        config.prompting.formatting_variants = [
            FormattingVariant(**v) for v in variants_raw
        ]
    
    if "generation" in raw:
        config.generation = GenerationConfig(**raw["generation"])
    
    if "detection" in raw:
        config.detection = DetectionConfig(**raw["detection"])
    
    if "analysis" in raw:
        config.analysis = AnalysisConfig(**raw["analysis"])
    
    if "ablation_generation" in raw:
        config.ablation_generation = AblationGenerationConfig(**raw["ablation_generation"])
    
    if "report" in raw:
        config.report = ReportConfig(**raw["report"])
    
    return config


# =============================================================================
# Run ID Persistence
# =============================================================================

CURRENT_RUN_ID_FILE = ".current_run_id"


def get_runs_dir() -> Path:
    """Get the runs directory path."""
    return PROJECT_ROOT / "runs"


def get_current_run_id_path() -> Path:
    """Get path to the .current_run_id file."""
    return get_runs_dir() / CURRENT_RUN_ID_FILE


def generate_run_id() -> str:
    """Generate a timestamp-based run ID."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_current_run_id() -> str | None:
    """
    Read the current run ID from .current_run_id file.
    
    Returns:
        run_id string or None if file doesn't exist
    """
    path = get_current_run_id_path()
    if path.exists():
        return path.read_text().strip()
    return None


def write_current_run_id(run_id: str) -> None:
    """
    Write run ID to .current_run_id file.
    
    Args:
        run_id: The run ID to persist
    """
    path = get_current_run_id_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run_id)


def resolve_run_id(config: ExperimentConfig, run_id_override: str | None = None) -> str:
    """
    Resolve the run ID to use.
    
    Priority:
    1. Explicit override from --run-id flag
    2. Persisted .current_run_id
    3. Error (must call init-run first)
    
    Args:
        config: Experiment configuration
        run_id_override: Optional explicit run ID
        
    Returns:
        Resolved run ID
        
    Raises:
        RuntimeError: If no run ID available
    """
    if run_id_override:
        return run_id_override
    
    if config.run.run_id:
        return config.run.run_id
    
    current = read_current_run_id()
    if current:
        return current
    
    raise RuntimeError(
        "No run ID available. Either:\n"
        "  1. Run 'python -m backtracking.cli init-run --config <config.yaml>' first\n"
        "  2. Pass --run-id <id> explicitly\n"
        "  3. Set run.run_id in the config file"
    )


def get_git_sha() -> str:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def create_run_metadata(config: ExperimentConfig, run_id: str) -> dict[str, Any]:
    """
    Create metadata dict for a run.
    
    Args:
        config: Experiment configuration
        run_id: The run ID
        
    Returns:
        Metadata dictionary
    """
    return {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "git_sha": get_git_sha(),
        "hostname": socket.gethostname(),
        "model_id": config.model.hf_id,
        "config_path": str(config._config_path) if config._config_path else None,
    }


def save_resolved_config(config: ExperimentConfig, run_dir: Path) -> None:
    """
    Save the resolved configuration to the run directory.
    
    Args:
        config: Experiment configuration
        run_dir: Run directory path
    """
    # Convert config to dict for YAML serialization
    config_dict = {
        "run": {
            "name": config.run.name,
            "seed": config.run.seed,
            "run_id": config.run.run_id,
            "output_dir": config.run.output_dir,
        },
        "model": {
            "hf_id": config.model.hf_id,
            "trust_remote_code": config.model.trust_remote_code,
            "torch_dtype": config.model.torch_dtype,
            "device": config.model.device,
            "attn_implementation": config.model.attn_implementation,
            "use_cache": config.model.use_cache,
        },
        "dataset": {
            "name": config.dataset.name,
            "split": config.dataset.split,
            "max_examples": config.dataset.max_examples,
            "shuffle": config.dataset.shuffle,
            "seed": config.dataset.seed,
            "save_path": config.dataset.save_path,
        },
        "prompting": {
            "system_prompt": config.prompting.system_prompt,
            "template": config.prompting.template,
            "formatting_variants": [
                {"name": v.name, "think_open": v.think_open, "think_close": v.think_close}
                for v in config.prompting.formatting_variants
            ],
        },
        "generation": {
            "num_samples_per_prompt": config.generation.num_samples_per_prompt,
            "max_new_tokens": config.generation.max_new_tokens,
            "temperature": config.generation.temperature,
            "top_p": config.generation.top_p,
            "do_sample": config.generation.do_sample,
            "batch_size": config.generation.batch_size,
        },
        "detection": {
            "triggers_strict": config.detection.triggers_strict,
            "triggers_relaxed": config.detection.triggers_relaxed,
            "onset_priority": config.detection.onset_priority,
            "answer_regex": config.detection.answer_regex,
        },
        "analysis": {
            "max_events": config.analysis.max_events,
            "control_samples": config.analysis.control_samples,
            "logit_lens_token": config.analysis.logit_lens_token,
            "ablation_components": config.analysis.ablation_components,
            "ablation_layers": config.analysis.ablation_layers,
            "topk_layers_for_generation": config.analysis.topk_layers_for_generation,
        },
        "ablation_generation": {
            "enabled": config.ablation_generation.enabled,
            "mode": config.ablation_generation.mode,
            "scale": config.ablation_generation.scale,
            "random_control_seed": config.ablation_generation.random_control_seed,
        },
        "report": {
            "make_report": config.report.make_report,
            "report_path": config.report.report_path,
        },
    }
    
    config_path = run_dir / "config_resolved.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def save_run_metadata(metadata: dict[str, Any], run_dir: Path) -> None:
    """
    Save run metadata to meta.json.
    
    Args:
        metadata: Metadata dictionary
        run_dir: Run directory path
    """
    meta_path = run_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

