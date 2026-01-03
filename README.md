# Backtracking as a State Transition
### Locating and Intervening on Revision-Mode Onset in DeepSeek-R1-Distill-Qwen-1.5B

Mechanistic interpretability research investigating how reasoning models perform self-correction through backtracking behavior.

## Overview

This project investigates the neural mechanisms behind **backtracking** in language models — the behavior where models use phrases like "Wait", "Actually", or "Hold on" to reconsider and revise their reasoning during chain-of-thought problem solving.

### Key Findings

- **Backtracking is localized**: MLP layer 27 alone accounts for 55% of backtracking behavior
- **Late MLP layers are critical**: Ablating MLP layers 19-27 reduces backtracking by 85%
- **Attention is not the mechanism**: Ablating attention layers actually *increases* backtracking
- **Backtracking improves accuracy**: Samples with backtracking show 2.1× higher accuracy (23.7% vs 11.5%)

## Project Structure

```
mats-backtracking/
├── src/backtracking/          # Core experiment code
│   ├── cli.py                 # Command-line interface
│   ├── config.py              # Configuration management
│   ├── detect.py              # Backtracking detection
│   ├── generate.py            # Model generation with ablations
│   ├── hooks.py               # Ablation hooks and instrumentation
│   └── analysis/              # Analysis modules
│       ├── events.py          # Event processing
│       ├── logit_lens.py      # Logit lens analysis
│       ├── ablation_scan.py   # Layer importance scanning
│       ├── sweeps.py          # Phase 2 sweep experiments
│       └── plots*.py          # Visualization
│
├── configs/                   # Experiment configurations
├── scripts/                   # Utility scripts
├── data/processed/            # Processed datasets
├── figures/                   # Generated visualizations
├── reports/                   # Experiment reports
└── runs/                      # Timestamped experiment runs
    ├── 20251223_232541/       # Phase 1 results
    └── 20251224_045331/       # Phase 2 results
```

## Setup

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (tested on A100 40GB)
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/i-anuragmishra/mats-backtracking.git
cd mats-backtracking

# Run bootstrap script
bash scripts/bootstrap.sh

# Add HuggingFace token to .env
echo "HF_TOKEN=your_token_here" >> .env

# Activate environment
source scripts/activate_env.sh

# Verify setup
make doctor
```

## Running Experiments

### Phase 1: Baseline Analysis & Layer Identification

```bash
# Initialize run
python -m backtracking.cli init-run --config configs/backtracking_state_transition.yaml

# Generate baseline completions
python -m backtracking.cli generate --config configs/backtracking_state_transition.yaml --condition baseline

# Detect backtracking events
python -m backtracking.cli detect-events --config configs/backtracking_state_transition.yaml

# Run ablation scan to identify important layers
python -m backtracking.cli ablation-scan --config configs/backtracking_state_transition.yaml
```

### Phase 2: Non-Destructive Intervention Search

```bash
# Subset sweep - test different layer combinations
python -m backtracking.cli phase2-subset-sweep --config configs/backtracking_state_transition_phase2.yaml

# Scale sweep - test ablation strengths
python -m backtracking.cli phase2-scale-sweep --config configs/backtracking_state_transition_phase2.yaml
```

## Results

### Phase 1: Layer Identification

| Condition | Backtracking Rate | Accuracy |
|-----------|-------------------|----------|
| Baseline | 68.1% | 19.9% |
| Targeted Ablation (12 layers) | 2.8% | 1.2% |
| Random Ablation | 54.9% | 1.6% |

### Phase 2: Subset Analysis

| Ablation Subset | Backtracking Rate | Reduction |
|-----------------|-------------------|-----------|
| Baseline | 63.3% | — |
| MLP Layer 27 Only | 28.3% | 55% |
| MLP Late Cluster | 9.8% | 85% |
| Phase 1 Full | 1.7% | 97% |

See `reports/` for detailed analysis.

## Model

- **Model**: DeepSeek-R1-Distill-Qwen-1.5B
- **Dataset**: GSM8K (200 problems, 6 samples each)
- **Backtracking triggers**: "Wait", "Actually", "Hold on", "Let me reconsider"

## Citation

```bibtex
@misc{mishra2025backtracking,
  title={Localized Neural Circuits for Backtracking in Reasoning Models},
  author={Mishra, Anurag},
  year={2025},
  howpublished={MATS Research}
}
```

## License

MIT
