"""Reusable configuration presets for local demos and CI."""

from __future__ import annotations

from pathlib import Path

from physical_informed_neural_network.config import ProjectConfig


def apply_smoke_test_preset(
    config: ProjectConfig,
    output_dir: str | Path | None = None,
) -> ProjectConfig:
    """Return a fast-running configuration for local verification."""
    config = config.model_copy(deep=True)

    config.data.nx = 64
    config.data.nt = 32
    config.data.n_observed = 128

    config.training.device = "cpu"
    config.training.adam_epochs = 5
    config.training.lbfgs_iterations = 0
    config.training.n_collocation = 256
    config.training.n_boundary = 32
    config.training.n_initial = 32
    config.training.log_every = 5
    config.training.warmup_steps = 0

    config.artifacts.prediction_nx = 64
    config.artifacts.prediction_nt = 32
    config.artifacts.output_dir = Path(output_dir) if output_dir is not None else Path("artifacts/burgers_pinn_smoke")

    return config


def build_smoke_test_config(output_dir: str | Path | None = None) -> ProjectConfig:
    """Create a fresh smoke-test configuration."""
    return apply_smoke_test_preset(ProjectConfig(), output_dir=output_dir)
