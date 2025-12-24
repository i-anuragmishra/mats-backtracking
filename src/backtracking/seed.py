"""
Deterministic seeding for reproducibility.

Sets seeds for random, numpy, and torch to ensure reproducible results.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set seeds for all random number generators.
    
    Args:
        seed: The seed value to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_generator(seed: int) -> torch.Generator:
    """
    Create a torch Generator with the given seed.
    
    Useful for reproducible sampling in DataLoaders.
    
    Args:
        seed: The seed value
        
    Returns:
        Seeded torch.Generator
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


