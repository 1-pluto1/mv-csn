"""Reproducibility seeding and hardware device selection helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Fix RNG seeds across Python, NumPy, and PyTorch (CPU & CUDA).

    Args:
        seed: Non-negative integer seed for deterministic experiments.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Select the fastest available accelerator (CUDA > Apple MPS > CPU).

    Returns:
        A :class:`torch.device` instance suitable for model/tensor placement.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
