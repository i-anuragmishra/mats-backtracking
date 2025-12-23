"""
MATS Backtracking: Mechanistic Interpretability Research Package

Investigating backtracking and state transitions in language models.
"""

__version__ = "0.1.0"
__author__ = "MATS Researcher"

from pathlib import Path

# Project root (useful for finding configs, data, etc.)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

def get_project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT

