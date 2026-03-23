"""Shared utility functions: seeding, device selection, directory helpers, sampling."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc


def ensure_directory(path: Path) -> Path:
    """Create *path* (and parents) if it does not exist; return *path*."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_global_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    """Resolve a device string (``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``)."""
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def latin_hypercube_sample(
    n: int,
    bounds: list[tuple[float, float]],
    seed: int = 0,
) -> np.ndarray:
    """Draw *n* points in a *d*-dimensional box via Latin Hypercube Sampling.

    Parameters
    ----------
    n : int
        Number of sample points.
    bounds : list of (lo, hi)
        Per-dimension bounds.
    seed : int
        Random seed for the sampler.

    Returns
    -------
    np.ndarray of shape (n, d)
    """
    d = len(bounds)
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    unit_samples = sampler.random(n=n)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    return qmc.scale(unit_samples, lo, hi)
