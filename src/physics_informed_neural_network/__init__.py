"""Public package namespace for the Burgers-equation PINN project."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .config import ProjectConfig
from .pipeline import ExperimentArtifacts, run_experiment

try:
    __version__ = version("physics-informed-neural-network")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["ExperimentArtifacts", "ProjectConfig", "__version__", "run_experiment"]
